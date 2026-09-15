from __future__ import annotations

import json

from teacher_attr.config import SPLITS, file_hash, initialize_run
from teacher_attr.generation import output_path, read_outputs
from teacher_attr.io import load_jsonl, save_json, write_jsonl
from teacher_attr.prompts import audit_splits, verify_prompts


def align(
    prompts: list[dict],
    teachers: dict[str, list[dict]],
    students: dict[str, list[dict]],
    student_config: dict,
) -> list[dict]:
    expected = {p["prompt_id"]: p for p in prompts}
    if len(expected) != len(prompts):
        raise ValueError("Duplicate prompt IDs")
    maps = {}
    for role, models in (("teachers", teachers), ("students", students)):
        for name, rows in models.items():
            mapping = {r["prompt_id"]: r for r in rows}
            if len(mapping) != len(rows) or set(mapping) != set(expected):
                raise ValueError(f"{role}/{name}: missing, extra, or duplicate responses")
            for pid, row in mapping.items():
                if (
                    row["prompt"] != expected[pid]["prompt"]
                    or row["split"] != expected[pid]["split"]
                ):
                    raise ValueError(f"{name}/{pid}: prompt or split mismatch")
                if not row["response"].strip():
                    raise ValueError(f"{name}/{pid}: empty response; investigate generation")
                if row.get("quality_flags"):
                    raise ValueError(f"{name}/{pid}: flagged response; inspect generation QC")
                if row.get("model_id") != name:
                    raise ValueError(f"{name}/{pid}: wrong model ID")
            maps[role, name] = mapping
    return [
        {
            **prompt,
            "student_id": name,
            "true_teacher": student_config[name]["teacher"],
            "student_response": maps["students", name][prompt["prompt_id"]]["response"],
            "teacher_responses": {
                t: maps["teachers", t][prompt["prompt_id"]]["response"] for t in teachers
            },
        }
        for prompt in prompts
        for name in sorted(students)
    ]


def build(cfg: dict) -> dict:
    root = initialize_run(cfg)
    verify_prompts(root)
    if (root / "pairs" / "manifest.json").exists():
        load_pairs(cfg)
        return json.loads((root / "pairs" / "manifest.json").read_text(encoding="utf-8"))
    splits = {}
    outputs = {}
    for split in SPLITS:
        teacher_rows = {t: read_outputs(cfg, "teachers", t, split) for t in cfg["teachers"]}
        student_rows = {
            s: read_outputs(cfg, "students", s, split)
            for s, model in cfg["students"].items()
            if split in model["splits"]
        }
        splits[split] = align(
            load_jsonl(root / "prompts" / f"{split}.jsonl"),
            teacher_rows,
            student_rows,
            cfg["students"],
        )
        for role, rows in (("teachers", teacher_rows), ("students", student_rows)):
            for name in rows:
                path = output_path(root, role, name, split)
                outputs[str(path.relative_to(root))] = file_hash(path)
    audit = audit_splits(splits)
    hashes = {}
    for split, rows in splits.items():
        path = root / "pairs" / f"{split}.jsonl"
        write_jsonl(path, rows)
        hashes[split] = file_hash(path)
    result = {
        "schema_version": 2,
        "audit": audit,
        "sha256": hashes,
        "outputs_sha256": outputs,
        "teacher_ids": list(cfg["teachers"]),
        "protocol": cfg["protocol"],
    }
    save_json(root / "pairs" / "manifest.json", result)
    return result


def load_pairs(cfg: dict) -> dict[str, list[dict]]:
    root = initialize_run(cfg)
    verify_prompts(root)
    manifest = json.loads((root / "pairs" / "manifest.json").read_text(encoding="utf-8"))
    if manifest["teacher_ids"] != list(cfg["teachers"]):
        raise ValueError("Teacher order changed")
    for relative, digest in manifest["outputs_sha256"].items():
        if file_hash(root / relative) != digest:
            raise ValueError(f"Source generation changed after pair construction: {relative}")
    splits = {}
    for split in SPLITS:
        path = root / "pairs" / f"{split}.jsonl"
        if file_hash(path) != manifest["sha256"][split]:
            raise ValueError(f"Pair artifact changed: {path}")
        rows = load_jsonl(path)
        if not rows:
            raise ValueError(f"Empty split: {split}")
        keys = [(r["prompt_id"], r["student_id"]) for r in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate student/prompt rows: {split}")
        for row in rows:
            student = cfg["students"][row["student_id"]]
            if row["true_teacher"] != student["teacher"] or split not in student["splits"]:
                raise ValueError("Student lineage/split does not match configuration")
            if set(row["teacher_responses"]) != set(cfg["teachers"]):
                raise ValueError("Candidate teachers do not match configuration")
        splits[split] = rows
    audit_splits(splits)
    return splits
