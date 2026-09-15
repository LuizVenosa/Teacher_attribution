"""Controlled distillation configuration and immutable, answer-free prompt pools."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import subprocess
from pathlib import Path

from teacher_attr.config import file_hash, fingerprint
from teacher_attr.io import save_json, write_jsonl
from teacher_attr.prompts import audit_splits, clean, format_source, text_hash

POOLS = ("distill_train", "distill_val", "train", "val", "test")


def provenance() -> dict:
    versions = {}
    for package in ("torch", "transformers", "datasets", "numpy", "tokenizers", "spacy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    result = {"git_commit": commit.stdout.strip() or None, "packages": versions}
    try:
        import torch

        result.update(
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        )
    except ImportError:
        pass
    return result


def expand_config(cfg: dict, parent: Path) -> dict:
    cfg = copy.deepcopy(cfg)
    root = (parent / cfg["run_dir"]).resolve()
    cfg["run_dir"] = str(root)
    settings = cfg["research"]
    student = settings["student"]
    sizes, seeds = settings["amounts"], settings["student_seeds"]
    if not sizes or sorted(set(sizes)) != sizes or any(n <= 0 for n in sizes):
        raise ValueError("research.amounts must be increasing positive unique sizes")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Student seeds must be nonempty and unique")
    total = sum(s["counts"]["distill_train"] for s in settings["sources"].values())
    if max(sizes) > total or settings["primary_amount"] not in sizes:
        raise ValueError("Distillation amount exceeds available prompts or primary amount missing")
    if settings["primary_seed"] not in seeds:
        raise ValueError("Primary seed is not configured")
    for source in settings["sources"].values():
        if set(source["counts"]) != set(POOLS) or any(n < 0 for n in source["counts"].values()):
            raise ValueError("Source counts must define all five nonnegative pool sizes")
        if source.get("path"):
            source["path"] = str((parent / source["path"]).resolve())
    train = settings["training"]
    for key in ("epochs", "microbatch", "gradient_accumulation", "max_length", "learning_rate"):
        if train[key] <= 0:
            raise ValueError(f"Invalid student training parameter: {key}")
    if train.get("packing"):
        raise ValueError("Packing is disabled until block-diagonal attention is supported")
    if student["prompt_format"] != "completion":
        raise ValueError("Controlled base students use the common completion format")
    cfg["students"] = {}
    student_root = Path(settings.get("student_root", root))
    assignments = settings.get(
        "split_seeds", dict.fromkeys(("train", "val", "test"), settings["primary_seed"])
    )
    for teacher in cfg["teachers"]:
        for split, seed in assignments.items():
            if seed not in seeds:
                raise ValueError("Split seed is not in configured student seeds")
            name = student_id(teacher, seed, settings["primary_amount"]) + settings.get(
                "student_suffix", ""
            )
            if name not in cfg["students"]:
                cfg["students"][name] = {
                    "hf_name": str(student_root / "students" / name / "checkpoint"),
                    "revision": "local",
                    "prompt_format": "completion",
                    "teacher": teacher,
                    "relation": "response_distillation",
                    "evidence_url": "local:training.json",
                    "splits": [],
                }
            cfg["students"][name]["splits"].append(split)
    cfg["data"] = {
        "datasets": list(settings["sources"]),
        "prompts_per_dataset": dict(train=1, val=1, test=1),
        "split_fractions": dict(train=0.8, val=0.1, test=0.1),
        "local_dir": None,
        "max_input_chars": settings["max_input_chars"],
        "include_qa_answer": False,
    }
    return cfg


def student_id(teacher: str, seed: int, amount: int) -> str:
    return f"{teacher}_seed{seed}_n{amount}"


def source_records(source: dict):
    if source.get("path"):
        path = Path(source["path"])
        paths = (
            sorted(path.glob("*.parquet")) + sorted(path.glob("*.jsonl"))
            if path.is_dir()
            else [path]
        )
        for path in paths:
            if path.suffix == ".jsonl":
                from teacher_attr.io import read_jsonl

                records = read_jsonl(path)
            else:
                import pyarrow.parquet as pq

                records = (r for b in pq.ParquetFile(path).iter_batches() for r in b.to_pylist())
            description = {"path": str(path), "sha256": file_hash(path)}
            for row in records:
                yield row, description
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi

        revision = HfApi().dataset_info(source["hf_name"], revision=source["revision"]).sha
        records = load_dataset(
            source["hf_name"],
            source.get("subset"),
            split=source["split"],
            revision=revision,
            streaming=True,
            **({"data_dir": source["data_dir"]} if source.get("data_dir") else {}),
        )
        for row in records:
            yield (
                row,
                {
                    "dataset": source["hf_name"],
                    "revision": revision,
                    "subset": source.get("subset"),
                    "split": source["split"],
                    "data_dir": source.get("data_dir"),
                },
            )


def user_prompt(name: str, row: dict, source: dict, limit: int) -> str | None:
    if source["format"] == "human_assistant":
        conversation = row.get("chosen", "")
        if not conversation.startswith("\n\nHuman:"):
            return None
        return (
            clean(conversation.split("\n\nHuman:", 1)[1].split("\n\nAssistant:", 1)[0], limit)
            or None
        )
    if source["format"] == "messages":
        # Retain only a standalone first user turn; no previous assistant answer leakage.
        messages = row.get("messages", [])
        if not messages or messages[0].get("role") != "user":
            return None
        return clean(messages[0].get("content"), limit) or None
    if source["format"] == "prompt":
        return clean(row.get(source.get("prompt_field", "prompt")), limit) or None
    prompt = format_source(name, row, {"max_input_chars": limit, "include_qa_answer": False})
    if prompt and source["task"] == "qa":
        prompt = prompt.replace(
            "Answer the question and explain briefly.",
            "Give a brief explanation followed by the final answer letter.",
        )
    return prompt


def prepare_research(cfg: dict) -> dict:
    from teacher_attr.config import initialize_run
    from teacher_attr.prompts import verify_prompts

    root = initialize_run(cfg)
    directory = root / "prompts"
    if (directory / "manifest.json").exists():
        return verify_prompts(root)
    if directory.exists() and list(directory.glob("*.jsonl")):
        raise ValueError("Incomplete prompt preparation; use a new run directory")
    settings = cfg["research"]
    pools = {p: [] for p in POOLS}
    seen, groups, sources, scanned = set(), set(), {}, {}
    for name, source in settings["sources"].items():
        candidates = []
        for index, (row, description) in enumerate(source_records(source)):
            if index >= settings["scan_limit"]:
                break
            sources[fingerprint(description)] = description
            prompt = user_prompt(name, row, source, settings["max_input_chars"])
            if not prompt:
                continue
            digest = text_hash(prompt)
            group = str(row.get("movieId") or row.get("id") or digest)
            if digest in seen or (name, group) in groups:
                continue
            seen.add(digest)
            groups.add((name, group))
            candidates.append(
                {
                    "prompt_id": digest,
                    "prompt": prompt,
                    "source_dataset": name,
                    "source": name,
                    "task": source["task"],
                    "source_group": group,
                    "metadata": {
                        "answer": row.get("answerKey"),
                        "reference": row.get("highlights") or row.get("abstract"),
                    },
                }
            )
        scanned[name] = len(candidates)
        candidates.sort(key=lambda r: fingerprint([cfg["seed"], r["prompt_id"]]))
        required = sum(source["counts"].values())
        if len(candidates) < required:
            raise ValueError(f"{name}: need {required} unique prompts, found {len(candidates)}")
        start = 0
        for pool, count in source["counts"].items():
            pools[pool].extend({**r, "split": pool} for r in candidates[start : start + count])
            start += count
    if any(not rows for rows in pools.values()):
        raise ValueError("Each prompt pool must be nonempty")
    audit = audit_splits(pools)
    hashes = {}
    for pool, rows in pools.items():
        rows.sort(key=lambda r: fingerprint([cfg["seed"], "nested", r["prompt_id"]]))
        path = directory / f"{pool}.jsonl"
        write_jsonl(path, rows)
        hashes[pool] = file_hash(path)
    subsets = {
        str(n): [r["prompt_id"] for r in pools["distill_train"][:n]] for n in settings["amounts"]
    }
    save_json(directory / "nested_subsets.json", subsets)
    result = {
        "schema_version": 3,
        "audit": audit,
        "sha256": hashes,
        "sources": list(sources.values()),
        "unique_candidates": scanned,
        "sampling": "seeded hash order within a bounded deterministic source prefix",
        "nested_subsets_sha256": file_hash(directory / "nested_subsets.json"),
        "environment": provenance(),
    }
    save_json(directory / "manifest.json", result)
    return result


def nested_ids(cfg: dict, amount: int) -> list[str]:
    root = Path(cfg["run_dir"]) / "prompts"
    manifest = json.loads((root / "manifest.json").read_text())
    if file_hash(root / "nested_subsets.json") != manifest["nested_subsets_sha256"]:
        raise ValueError("Nested subset manifest changed")
    return json.loads((root / "nested_subsets.json").read_text())[str(amount)]
