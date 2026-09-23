"""Create a new run after increasing rejection-only input/SFT length budgets."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

from teacher_attr.config import file_hash, fingerprint, initialize_run, load_config
from teacher_attr.generation import generation_spec, output_path, read_outputs
from teacher_attr.io import load_jsonl, load_yaml, save_json, save_yaml, write_jsonl
from teacher_attr.prompts import audit_splits, verify_prompts
from teacher_attr.research import POOLS, provenance


def migrate_budgets(config_path: str, output: str, input_tokens: int, training_tokens: int) -> dict:
    source_config, destination = Path(config_path).resolve(), Path(output).resolve()
    cfg = load_config(source_config)
    source = Path(cfg["run_dir"])
    raw = load_yaml(source_config)
    if destination.parent != source_config.parent:
        raise ValueError(
            "Keep the new config beside the original to preserve relative source paths"
        )
    if raw["research"].get("variant") or raw["research"].get("student_root"):
        raise ValueError("Budget migration supports primary controlled runs only")
    if (
        input_tokens < cfg["generation"]["max_input_tokens"]
        or training_tokens < cfg["research"]["training"]["max_length"]
    ):
        raise ValueError("Budget migration only permits increasing length limits")
    if json.loads((source / "experiment.json").read_text(encoding="utf-8")) != cfg:
        raise ValueError("Source configuration differs from its immutable run record")
    target = source.parent / destination.stem
    if destination.exists() or target.exists():
        raise ValueError("Destination config/run already exists; refusing to overwrite")
    manifest = verify_prompts(source)
    pools = {split: load_jsonl(source / "prompts" / f"{split}.jsonl") for split in POOLS}
    audit_splits(pools)
    for split in POOLS:
        if file_hash(source / "prompts" / f"{split}.jsonl") != manifest["sha256"][split]:
            raise ValueError(f"Source prompt hash mismatch: {split}")
    if file_hash(source / "prompts/nested_subsets.json") != manifest["nested_subsets_sha256"]:
        raise ValueError("Source nested subsets changed")
    # Validate everything to be reused before creating the destination.
    completed, skipped = [], []
    for teacher in cfg["teachers"]:
        for split in POOLS:
            path = output_path(source, "teachers", teacher, split)
            if not path.with_suffix(".meta.json").exists():
                continue
            rows = read_outputs(cfg, "teachers", teacher, split)
            expected = {r["prompt_id"]: r["prompt"] for r in pools[split]}
            if {r["prompt_id"]: r["prompt"] for r in rows} != expected:
                skipped.append({"teacher": teacher, "split": split, "reason": "incomplete"})
                continue
            meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
            if meta.get("response_processing_version") != 2:
                raise ValueError("Cannot reuse responses from a different processing version")
            completed.append((teacher, split, path, rows, meta))
    raw["run_dir"] = str(target)
    raw["generation"]["max_input_tokens"] = input_tokens
    raw["research"]["training"]["max_length"] = training_tokens
    save_yaml(destination, raw)
    new_cfg = load_config(destination)
    initialize_run(new_cfg)
    shutil.copytree(source / "prompts", target / "prompts")
    report = {
        "source_config": str(source_config),
        "source_run": str(source),
        "source_config_sha256": file_hash(source_config),
        "destination_config": str(destination),
        "destination_run": str(target),
        "changes": {"max_input_tokens": input_tokens, "training_max_length": training_tokens},
        "rationale": (
            "Only rejection limits increased; prompt text, sampling and output caps unchanged"
        ),
        "reused": [],
        "skipped": skipped,
        "environment": provenance(),
    }
    for teacher, split, path, rows, original in completed:
        metadata = copy.deepcopy(original)
        metadata["spec"] = generation_spec(new_cfg, "teachers", teacher, split)
        metadata["reuse"] = {
            "source_path": str(path),
            "source_sha256": file_hash(path),
            "original_manifest": original,
            "original_manifest_sha256": file_hash(path.with_suffix(".meta.json")),
            "reason": report["rationale"],
        }
        dest = output_path(target, "teachers", teacher, split)
        save_json(dest.with_suffix(".meta.json"), metadata)
        new_rows = [{**r, "generation_fingerprint": fingerprint(metadata)} for r in rows]
        # Keep the actual original generation_config in each row, including its old
        # input guard. The new manifest explicitly records reuse, not regeneration.
        write_jsonl(dest, new_rows)
        read_outputs(new_cfg, "teachers", teacher, split)
        report["reused"].append({"teacher": teacher, "split": split, "rows": len(rows)})
    save_json(target / "budget_migration.json", report)
    return report
