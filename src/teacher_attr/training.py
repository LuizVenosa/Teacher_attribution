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
    return {
        p: importlib.metadata.version(p)
        for p in ("torch", "transformers", "numpy", "scikit-learn", "teacher-attribution")
    }


def batch_forward(model, batch, device, cfg, num_teachers):
    student, teacher, labels = batch
    labels = labels.to(device)
    emb, logits = model(**to_device(student, device))
    if cfg["objective"] == "classification":
        return F.cross_entropy(logits, labels), logits, labels
    teacher_emb, _ = model(**to_device(teacher, device))
    teacher_emb = teacher_emb.reshape(len(labels), num_teachers, -1)
    scores = score_embeddings(emb, teacher_emb)
    loss = F.cross_entropy(scores / cfg["temperature"], labels)
    if cfg["objective"] == "joint":
        loss = loss + cfg["classification_weight"] * F.cross_entropy(logits, labels)
    return loss, scores, labels


def train(
    cfg: dict, objective: str | None = None, input_mode: str | None = None, seed: int | None = None
) -> dict:
    root = initialize_run(cfg)
    enc = {
        **cfg["encoder"],
        "objective": objective or cfg["encoder"]["objective"],
        "input_mode": input_mode or cfg["encoder"]["input_mode"],
    }
    seed = cfg["seed"] if seed is None else seed
    directory = root / "models" / f"{enc['objective']}_{enc['input_mode']}_seed{seed}"
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
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss, _, labels = batch_forward(model, batch, device, enc, len(teachers))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
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
        backbone, payload["encoder"]["projection_dim"], len(payload["teacher_ids"])
    )
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path.parent / "tokenizer")
    return model, tokenizer, payload
