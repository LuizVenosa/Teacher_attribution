from __future__ import annotations

import hashlib
import json
from pathlib import Path

from teacher_attr.io import load_yaml, save_json

SPLITS = ("train", "val", "test")
RELATIONS = {"logit_distillation", "response_distillation", "data_selection"}


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    cfg = load_yaml(path)
    validate_config(cfg)
    cfg["run_dir"] = str((path.parent / cfg["run_dir"]).resolve())
    if cfg["data"].get("local_dir"):
        cfg["data"]["local_dir"] = str((path.parent / cfg["data"]["local_dir"]).resolve())
    return cfg


def validate_config(cfg: dict) -> None:
    for key in (
        "run_dir",
        "seed",
        "protocol",
        "teachers",
        "students",
        "data",
        "encoder",
        "generation",
        "evaluation",
    ):
        if key not in cfg:
            raise ValueError(f"Missing config key: {key}")
    if cfg["protocol"] not in {"public_seen_students", "held_out_students"}:
        raise ValueError("protocol must be public_seen_students or held_out_students")
    if len(cfg["teachers"]) < 2:
        raise ValueError("At least two candidate teachers are required")
    for role in ("teachers", "students"):
        for name, model in cfg[role].items():
            if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
                raise ValueError(f"Model ID must be a safe lowercase identifier: {name}")
            if not model.get("hf_name") or not model.get("revision"):
                raise ValueError(f"{name}: hf_name and revision are required")
            if model.get("prompt_format") not in {"completion", "chat", "text2text"}:
                raise ValueError(f"{name}: unsupported prompt_format")
    by_split = {s: set() for s in SPLITS}
    identities = {s: set() for s in SPLITS}
    for name, model in cfg["students"].items():
        if model.get("teacher") not in cfg["teachers"]:
            raise ValueError(f"{name}: unknown teacher")
        if model.get("relation") not in RELATIONS or not model.get("evidence_url"):
            raise ValueError(f"{name}: documented distillation/data-selection relation required")
        splits = model.get("splits", [])
        if not splits or len(set(splits)) != len(splits) or set(splits) - set(SPLITS):
            raise ValueError(f"{name}: invalid splits")
        for split in splits:
            by_split[split].add(model["teacher"])
            identities[split].add((model["hf_name"], model["revision"]))
    if any(v != set(cfg["teachers"]) for v in by_split.values()):
        raise ValueError("Every split must contain students for every candidate teacher")
    if cfg["protocol"] == "held_out_students":
        if any(
            identities[a] & identities[b]
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        ):
            raise ValueError(
                "held_out_students requires disjoint checkpoint identities across splits"
            )
    data = cfg["data"]
    if not data["datasets"] or len(set(data["datasets"])) != len(data["datasets"]):
        raise ValueError("datasets must be nonempty and unique")
    for key in ("prompts_per_dataset", "split_fractions"):
        if set(data[key]) != set(SPLITS) or any(v <= 0 for v in data[key].values()):
            raise ValueError(f"{key} must define positive train/val/test values")
    if abs(sum(data["split_fractions"].values()) - 1) > 1e-8:
        raise ValueError("split_fractions must sum to one")
    enc = cfg["encoder"]
    if enc["input_mode"] not in {"response", "prompt_response"}:
        raise ValueError("input_mode must be response or prompt_response")
    if enc["objective"] not in {"joint", "classification", "contrastive"}:
        raise ValueError("objective must be joint, classification, or contrastive")
    if not 0 < enc["max_prompt_tokens"] < enc["max_length"] - 4:
        raise ValueError("Reserve at least one response token plus special tokens")
    for key in ("batch_size", "epochs", "patience", "temperature", "projection_dim"):
        if enc[key] <= 0:
            raise ValueError(f"encoder.{key} must be positive")
    if cfg["generation"]["max_input_tokens"] <= 0 or cfg["generation"]["max_new_tokens"] <= 0:
        raise ValueError("Generation token budgets must be positive")
    ev = cfg["evaluation"]
    if any(v < 1 for v in ev["set_sizes"]) or not ev["set_sizes"]:
        raise ValueError("set_sizes must contain positive integers")
    if ev["set_repeats"] < 1 or ev["bootstrap_repeats"] < 1:
        raise ValueError("Evaluation repeat counts must be positive")


def initialize_run(cfg: dict) -> Path:
    root = Path(cfg["run_dir"])
    manifest = root / "experiment.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != cfg:
            raise ValueError(
                "Run configuration changed. Choose a new run_dir; artifacts are immutable."
            )
    else:
        save_json(manifest, cfg)
    return root
