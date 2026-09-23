"""Explicit, output-bound exclusions shared by every teacher's SFT pool."""

import json
from pathlib import Path

from teacher_attr.config import file_hash, initialize_run
from teacher_attr.generation import output_path, read_outputs
from teacher_attr.io import load_jsonl, save_json


def training_exclusions(cfg):
    root = Path(cfg["run_dir"])
    path = root / "training_exclusions.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or not manifest["reason"].strip():
        raise ValueError("Invalid exclusion manifest")
    expected = {r["prompt_id"] for r in load_jsonl(root / "prompts/distill_train.jsonl")}
    ids = manifest["prompt_ids"]
    if not ids or len(ids) != len(set(ids)) or not set(ids) < expected:
        raise ValueError("Exclusions must be a nonempty proper subset of training prompts")
    hashes = {
        t: file_hash(output_path(root, "teachers", t, "distill_train")) for t in cfg["teachers"]
    }
    if hashes != manifest["teacher_outputs_sha256"]:
        raise ValueError("Teacher outputs changed after exclusion review")
    return manifest


def exclude_training_prompts(cfg, prompt_ids, reason):
    root = initialize_run(cfg)
    path = root / "training_exclusions.json"
    if path.exists() or any((root / "students").glob("*")):
        raise ValueError("Exclusions must be frozen once, before any student training")
    expected = {r["prompt_id"] for r in load_jsonl(root / "prompts/distill_train.jsonl")}
    ids = sorted(set(prompt_ids))
    if not reason.strip() or not ids or not set(ids) < expected:
        raise ValueError("Supply a reason and a nonempty proper subset of training prompts")
    hashes = {}
    for teacher in cfg["teachers"]:
        rows = read_outputs(cfg, "teachers", teacher, "distill_train")
        if {r["prompt_id"] for r in rows} != expected:
            raise ValueError("All teacher training outputs must be complete before exclusion")
        hashes[teacher] = file_hash(output_path(root, "teachers", teacher, "distill_train"))
    manifest = {
        "schema_version": 1,
        "prompt_ids": ids,
        "reason": reason,
        "teacher_outputs_sha256": hashes,
    }
    save_json(path, manifest)
    return manifest
