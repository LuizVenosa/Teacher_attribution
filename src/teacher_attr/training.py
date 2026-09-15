from __future__ import annotations

import importlib.metadata
from functools import partial
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer, set_seed

from teacher_attr.config import file_hash, initialize_run
from teacher_attr.encoders import AttributionEncoder, collate, score_embeddings, to_device
from teacher_attr.io import save_json
from teacher_attr.pairs import load_pairs


def environment() -> dict:
    from teacher_attr.research import provenance

    return {
        **provenance(),
        "packages_attribution": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "numpy", "scikit-learn", "teacher-attribution")
        },
    }


def batch_forward(model, batch, device, cfg, num_teachers):
    student, teacher, labels = batch
    labels = labels.to(device)
    emb, logits = model(**to_device(student, device))
    if cfg["objective"] == "classification":
        return F.cross_entropy(logits, labels), logits, labels
    teacher_emb, _ = model(**to_device(teacher, device))
    teacher_emb = teacher_emb.reshape(len(labels), num_teachers, -1)
    if cfg.get("negatives", "same_prompt") == "random" and model.training:
        # Keep each positive fixed, draw wrong-teacher negatives from other batch prompts.
        rotated = teacher_emb.roll(1, dims=0)
        mask = F.one_hot(labels, num_classes=num_teachers).bool().unsqueeze(-1)
        teacher_emb = torch.where(mask, teacher_emb, rotated)
    scores = score_embeddings(emb, teacher_emb)
    loss = F.cross_entropy(scores / cfg["temperature"], labels)
    if cfg["objective"] == "joint":
        loss = loss + cfg["classification_weight"] * F.cross_entropy(logits, labels)
    return loss, scores, labels


def train(
    cfg: dict,
    objective: str | None = None,
    input_mode: str | None = None,
    seed: int | None = None,
    representation: str | None = None,
    negatives: str | None = None,
    classification_weight: float | None = None,
    baseline: str | None = None,
) -> dict:
    root = initialize_run(cfg)
    enc = {
        **cfg["encoder"],
        "objective": objective or cfg["encoder"]["objective"],
        "input_mode": input_mode or cfg["encoder"]["input_mode"],
    }
    enc["representation"] = representation or enc.get("representation", "semantic")
    enc["negatives"] = negatives or enc.get("negatives", "same_prompt")
    if classification_weight is not None:
        if classification_weight < 0:
            raise ValueError("Classification weight cannot be negative")
        enc["classification_weight"] = classification_weight
    if enc["representation"] != "semantic":
        import json

        if not baseline:
            raise ValueError(
                "A verified single-response baseline is required before latent structure"
            )
        record = json.loads(Path(baseline).read_text())
        if not {"tfidf_matching", "generic_cosine", "pos", "trained_cosine"} <= set(
            record["methods"]
        ):
            raise ValueError("Complete the simpler contrastive baselines first")
        if record["pair_sha256"] != {
            s: file_hash(root / "pairs" / f"{s}.jsonl") for s in ("train", "val", "test")
        }:
            raise ValueError("Baseline belongs to different data")
    seed = cfg["seed"] if seed is None else seed
    directory = root / "models" / f"{enc['objective']}_{enc['input_mode']}_seed{seed}"
    suffix = ""
    if enc["representation"] != "semantic":
        suffix += f"_{enc['representation']}"
    if enc["negatives"] != "same_prompt":
        suffix += f"_{enc['negatives']}"
    if classification_weight is not None:
        suffix += f"_lambda{classification_weight:g}"
    directory = directory.with_name(directory.name + suffix)
    if directory.exists():
        raise ValueError(f"Training directory already exists: {directory}. Use a new seed/run.")
    splits = load_pairs(cfg)
    set_seed(seed)
    teachers = list(cfg["teachers"])
    tokenizer = AutoTokenizer.from_pretrained(
        enc["hf_name"], revision=enc["revision"], trust_remote_code=False
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AttributionEncoder.pretrained(enc, len(teachers)).to(device)
    if enc.get("gradient_checkpointing"):
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    collator = partial(collate, tokenizer=tokenizer, cfg=enc, teachers=teachers)
    loaders = {
        s: DataLoader(
            splits[s], batch_size=enc["batch_size"], shuffle=s == "train", collate_fn=collator
        )
        for s in ("train", "val")
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=enc["learning_rate"], weight_decay=enc["weight_decay"]
    )
    directory.mkdir(parents=True)
    tokenizer.save_pretrained(directory / "tokenizer")
    model.backbone.config.save_pretrained(directory / "backbone")
    data_hashes = {s: file_hash(root / "pairs" / f"{s}.jsonl") for s in splits}
    history, best, stale, best_epoch = [], -1.0, 0, 0
    for epoch in range(1, enc["epochs"] + 1):
        model.train()
        total, n = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        accumulation = enc.get("gradient_accumulation", 1)
        window_rows = 0
        for batch_index, batch in enumerate(loaders["train"]):
            with torch.autocast(
                device_type=device,
                dtype=torch.bfloat16,
                enabled=enc.get("precision") == "bfloat16" and device == "cuda",
            ):
                loss, _, labels = batch_forward(model, batch, device, enc, len(teachers))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            (loss * len(labels)).backward()
            window_rows += len(labels)
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loaders["train"]):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(window_rows)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                window_rows = 0
            total += loss.item() * len(labels)
            n += len(labels)
        model.eval()
        correct, val_n = 0, 0
        with torch.inference_mode():
            for batch in loaders["val"]:
                _, scores, labels = batch_forward(model, batch, device, enc, len(teachers))
                correct += (scores.argmax(-1) == labels).sum().item()
                val_n += len(labels)
        accuracy = correct / val_n
        history.append({"epoch": epoch, "train_loss": total / n, "val_accuracy": accuracy})
        print(f"epoch={epoch} loss={total / n:.4f} val_accuracy={accuracy:.4f}", flush=True)
        if accuracy > best:
            best, stale, best_epoch = accuracy, 0, epoch
            payload = {
                "schema_version": 2,
                "model": model.state_dict(),
                "encoder": enc,
                "teacher_ids": teachers,
                "pair_sha256": data_hashes,
                "seed": seed,
                "epoch": epoch,
                "resolved_revision": getattr(model.backbone.config, "_commit_hash", None),
            }
            temporary = directory / "best.tmp"
            torch.save(payload, temporary)
            temporary.replace(directory / "best.pt")
        else:
            stale += 1
        summary = {
            "encoder": enc,
            "seed": seed,
            "history": history,
            "best_epoch": best_epoch,
            "best_val_accuracy": best,
            "early_stopped": stale >= enc["patience"],
            "pair_sha256": data_hashes,
            "environment": environment(),
            "baseline_sha256": file_hash(baseline) if baseline else None,
            "checkpoint": str(directory / "best.pt"),
        }
        save_json(directory / "training.json", summary)
        if stale >= enc["patience"]:
            break
    return summary


def load_checkpoint(path: str | Path, cfg: dict, device: str):
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 2 or payload["teacher_ids"] != list(cfg["teachers"]):
        raise ValueError("Legacy/incompatible checkpoint; retrain with the corrected pipeline")
    root = Path(cfg["run_dir"])
    if any(
        file_hash(root / "pairs" / f"{s}.jsonl") != digest
        for s, digest in payload["pair_sha256"].items()
    ):
        raise ValueError("Checkpoint belongs to different data")
    backbone = AutoModel.from_config(AutoConfig.from_pretrained(path.parent / "backbone"))
    model = AttributionEncoder(
        backbone,
        payload["encoder"]["projection_dim"],
        len(payload["teacher_ids"]),
        payload["encoder"].get("representation", "semantic"),
        payload["encoder"].get("structure_layer", 1),
    )
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path.parent / "tokenizer")
    return model, tokenizer, payload
