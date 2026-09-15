"""Mean-embedding fingerprints and permutation-invariant learned Deep Sets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from teacher_attr.config import file_hash, initialize_run, load_config
from teacher_attr.evaluation import encode_rows
from teacher_attr.io import save_json, write_jsonl
from teacher_attr.metrics import (
    classification_metrics,
    cluster_interval,
    grouped_indexes,
    support_sets,
)
from teacher_attr.pairs import load_pairs
from teacher_attr.training import load_checkpoint


class DeepSets(nn.Module):
    def __init__(self, dimension: int, hidden: int):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.rho = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, dimension))

    def forward(self, values):
        return F.normalize(self.rho(self.phi(values).mean(dim=-2)), dim=-1)


def set_scores(student, teachers, model=None):
    # student [batch,k,d], teachers [batch,teacher,k,d].
    aggregate = model if model is not None else lambda x: F.normalize(x.mean(dim=-2), dim=-1)
    return torch.einsum("bd,btd->bt", aggregate(student), aggregate(teachers))


def encode_splits(cfg, checkpoint, splits):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, payload = load_checkpoint(checkpoint, cfg, device)
    result = {
        s: encode_rows(
            model, tokenizer, rows, payload["encoder"], list(cfg["teachers"]), device, True
        )
        for s, rows in splits.items()
    }
    del model
    return result


def sample_training_sets(rows, sizes, repeats, seed):
    rng = np.random.default_rng(seed)
    sets = []
    for student, indexes in grouped_indexes([r["student_id"] for r in rows]).items():
        valid = [n for n in sizes if n <= len(indexes)]
        for repeat in range(repeats):
            size = int(rng.choice(valid))
            sets.append(
                {
                    "student_id": student,
                    "size": size,
                    "repeat": repeat,
                    "row_indexes": rng.choice(indexes, size=size, replace=False).tolist(),
                }
            )
    rng.shuffle(sets)
    return sets


def tensors(encoded, sets, labels):
    indexes = np.array([s["row_indexes"] for s in sets])
    return (
        torch.tensor(encoded["embeddings"][indexes]),
        torch.tensor(encoded["teacher_embeddings"][indexes]).transpose(1, 2),
        torch.tensor(labels[indexes[:, 0]]),
    )


def train_sets(cfg: dict, checkpoint: str, baseline: str) -> dict:
    root = initialize_run(cfg)
    reference = json.loads(Path(baseline).read_text())
    if not {"tfidf_matching", "generic_cosine", "pos", "trained_cosine"} <= set(
        reference["methods"]
    ):
        raise ValueError("Complete single-response baselines before Deep Sets")
    if reference.get("pair_sha256") != {
        s: file_hash(root / "pairs" / f"{s}.jsonl") for s in ("train", "val", "test")
    }:
        raise ValueError("Baseline uses different data")
    recorded = reference.get("encoders", {}).get("trained", {}).get("checkpoint")
    if not recorded or file_hash(recorded) != file_hash(checkpoint):
        raise ValueError("Baseline must evaluate this exact contrastive checkpoint")
    out = root / "models" / "deep_sets"
    if out.exists():
        raise ValueError("Deep Sets output already exists")
    splits = load_pairs(cfg)
    encoded = encode_splits(cfg, checkpoint, {s: splits[s] for s in ("train", "val")})
    teachers = list(cfg["teachers"])
    labels = {s: np.array([teachers.index(r["true_teacher"]) for r in splits[s]]) for s in encoded}
    settings = cfg["research"]["sets"]
    torch.manual_seed(cfg["seed"])
    dim = encoded["train"]["embeddings"].shape[1]
    model = DeepSets(dim, settings["hidden_dim"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"])
    val = sample_training_sets(
        splits["val"],
        cfg["evaluation"]["set_sizes"],
        settings["train_sets_per_student"],
        cfg["seed"],
    )
    history, best = [], -1.0
    for epoch in range(settings["epochs"]):
        train = sample_training_sets(
            splits["train"],
            cfg["evaluation"]["set_sizes"],
            settings["train_sets_per_student"],
            cfg["seed"] + epoch,
        )
        total, count = 0.0, 0
        model.train()
        for _size, indexes in grouped_indexes([s["size"] for s in train]).items():
            for start in range(0, len(indexes), settings["batch_size"]):
                chosen = [train[i] for i in indexes[start : start + settings["batch_size"]]]
                student, teacher, y = tensors(encoded["train"], chosen, labels["train"])
                loss = F.cross_entropy(
                    set_scores(student, teacher, model) / settings["temperature"], y
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                total += loss.item() * len(chosen)
                count += len(chosen)
        model.eval()
        correct = 0
        with torch.inference_mode():
            for item in val:
                student, teacher, y = tensors(encoded["val"], [item], labels["val"])
                correct += int((set_scores(student, teacher, model).argmax(1) == y).sum())
        accuracy = correct / len(val)
        history.append({"epoch": epoch + 1, "train_loss": total / count, "val_accuracy": accuracy})
        if accuracy > best:
            best = accuracy
            out.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state": model.state_dict(),
                    "dimension": dim,
                    "hidden": settings["hidden_dim"],
                    "encoder_sha256": file_hash(checkpoint),
                    "configuration": settings,
                    "pair_sha256": reference["pair_sha256"],
                },
                out / "best.pt",
            )
    save_json(
        out / "training.json",
        {"history": history, "baseline_sha256": file_hash(baseline), "seed": cfg["seed"]},
    )
    write_jsonl(out / "validation_sets.jsonl", val)
    return {"checkpoint": str(out / "best.pt"), "best_val_accuracy": best}


def evaluate_sets(
    cfg: dict,
    checkpoint: str,
    set_checkpoint: str | None = None,
    name: str = "sets",
    reference_config: str | None = None,
) -> dict:
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
        raise ValueError("Invalid evaluation name")
    root = initialize_run(cfg)
    out = root / "evaluations" / name
    if out.exists():
        raise ValueError("Set evaluation already exists")
    splits = load_pairs(cfg)
    rows = splits["test"]
    teachers = list(cfg["teachers"])
    labels = np.array([teachers.index(r["true_teacher"]) for r in rows])
    checkpoint_cfg = load_config(reference_config) if reference_config else cfg
    if list(checkpoint_cfg["teachers"]) != teachers:
        raise ValueError("Teacher order differs in reference config")
    encoded = encode_splits(checkpoint_cfg, checkpoint, {"test": rows})["test"]
    variants = {"contrastive_mean": (encoded, None)}
    if cfg["evaluation"].get("generic_encoder"):
        from transformers import AutoTokenizer

        from teacher_attr.encoders import AttributionEncoder

        enc = {**cfg["encoder"], **cfg["evaluation"]["generic_encoder"], "input_mode": "response"}
        model = AttributionEncoder.pretrained(enc, len(teachers))
        tokenizer = AutoTokenizer.from_pretrained(enc["hf_name"], revision=enc["revision"])
        generic = encode_rows(model, tokenizer, rows, enc, teachers, "cpu", False)
        variants["generic_mean"] = (generic, None)
    if set_checkpoint:
        payload = torch.load(set_checkpoint, map_location="cpu", weights_only=True)
        if payload["encoder_sha256"] != file_hash(checkpoint):
            raise ValueError("Set checkpoint belongs to another encoder")
        if payload["pair_sha256"] != {
            s: file_hash(Path(checkpoint_cfg["run_dir"]) / "pairs" / f"{s}.jsonl") for s in splits
        }:
            raise ValueError("Set checkpoint belongs to different data")
        model = DeepSets(payload["dimension"], payload["hidden"])
        model.load_state_dict(payload["state"])
        model.eval()
        variants["deep_sets"] = (encoded, model)
    ev = cfg["evaluation"]
    supports = support_sets(rows, ev["set_sizes"], ev["set_repeats"], cfg["seed"])
    supports += [
        {"scope": "all", "size": 1, "repeat": i, "student_id": r["student_id"], "row_indexes": [i]}
        for i, r in enumerate(rows)
    ]
    results, predictions = {}, []
    with torch.inference_mode():
        for method, (values, model) in variants.items():
            results[method] = {}
            scores = []
            for item in supports:
                student, teacher, y = tensors(values, [item], labels)
                score = set_scores(student, teacher, model)[0].numpy()
                scores.append(score)
                predictions.append(
                    {"method": method, **item, "scores": score.tolist(), "label": int(y[0])}
                )
            scores = np.array(scores)
            y = np.array([labels[s["row_indexes"][0]] for s in supports])
            for key, ix in grouped_indexes([f"{s['scope']}/{s['size']}" for s in supports]).items():
                result = classification_metrics(scores[ix], y[ix], teachers)
                groups = [
                    rows[supports[i]["row_indexes"][0]]["prompt_id"]
                    if supports[i]["size"] == 1
                    else str(supports[i]["repeat"])
                    for i in ix
                ]
                result["accuracy_ci95"] = cluster_interval(
                    scores[ix].argmax(1) == y[ix], groups, ev["bootstrap_repeats"], cfg["seed"]
                )
                results[method][key] = result
    save_json(
        out / "metrics.json",
        {
            "teacher_ids": teachers,
            "set_methods": results,
            "aggregation": "cosine of pooled embeddings (not mean per-prompt scores)",
            "reference_config": reference_config,
            "encoder_sha256": file_hash(checkpoint),
            "interval_scope": (
                "prompt bootstrap for k=1; support repetition bootstrap for k>1; fixed students"
            ),
        },
    )
    write_jsonl(out / "predictions.jsonl", predictions)
    write_jsonl(out / "support_sets.jsonl", supports)
    return {"output": str(out), "methods": list(results)}
