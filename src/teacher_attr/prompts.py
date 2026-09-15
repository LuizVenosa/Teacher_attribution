from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from teacher_attr.config import SPLITS, file_hash, fingerprint, initialize_run
from teacher_attr.io import load_jsonl, save_json, write_jsonl

# Names describe actual sources. The old `sumpubmed` folder contains this PubMed dataset.
SOURCES = {
    "cnn_dailymail": ("cnn_dailymail", "3.0.0", "summarization"),
    "pubmed": ("ccdv/pubmed-summarization", "section", "summarization"),
    "rotten_tomatoes": ("contemmcm/rotten_tomatoes", "original", "summarization"),
    "commonsenseqa": ("tau/commonsense_qa", None, "qa"),
    "openbookqa": ("allenai/openbookqa", "main", "qa"),
    "alpaca": ("tatsu-lab/alpaca", None, "instruction_following"),
}


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def clean(value: object, limit: int) -> str:
    if isinstance(value, list):
        value = " ".join(str(x) for x in value)
    text = " ".join(str(value or "").split())
    return text[:limit].rsplit(" ", 1)[0] if len(text) > limit else text


def format_source(name: str, row: dict, settings: dict) -> str | None:
    limit = settings["max_input_chars"]
    if name in {"cnn_dailymail", "pubmed"}:
        article = clean(row.get("article") or row.get("text"), limit)
        return f"Summarize the following article.\n\n{article}" if article else None
    if name == "rotten_tomatoes":
        review = clean(row.get("reviewText") or row.get("text"), limit)
        return f"Summarize this critic's opinion and sentiment.\n\n{review}" if review else None
    if name == "alpaca":
        instruction = clean(row.get("instruction"), limit)
        context = clean(row.get("input"), limit)
        if not instruction or re.search(
            r"\b(code|python|javascript|programming)\b", instruction, re.I
        ):
            return None
        return instruction + (f"\n\n{context}" if context else "")
    question = clean(row.get("question") or row.get("question_stem"), limit)
    choices = row.get("choices", {})
    if isinstance(choices, str):
        choices = json.loads(choices)
    if not question or not choices.get("text"):
        return None
    options = "\n".join(
        f"{label}. {text}" for label, text in zip(choices["label"], choices["text"], strict=True)
    )
    instruction = "Answer the question and explain briefly."
    if settings["include_qa_answer"]:
        answer = row.get("answerKey")
        if not answer:
            return None
        instruction = f"The correct answer is {answer}. Explain why."
    return f"{instruction}\n\n{question}\n{options}"


def source_rows(name: str, settings: dict):
    """Yield (source split, index, row, source description) without loading a corpus into RAM."""
    hf_name, subset, _ = SOURCES[name]
    if settings.get("local_dir"):
        directory = Path(settings["local_dir"]) / name
        if name == "pubmed" and not directory.exists():
            directory = directory.parent / "sumpubmed"
        paths = sorted(directory.glob("*.parquet")) + sorted(directory.glob("*.jsonl"))
        complete = [p for p in paths if p.stem == "complete"]
        paths = complete or paths
        if not paths:
            raise FileNotFoundError(f"No Parquet/JSONL sources in {directory}")
        for path in paths:
            description = {
                "path": str(path),
                "sha256": file_hash(path),
                "dataset": hf_name,
                "subset": subset,
            }
            if path.suffix == ".jsonl":
                from teacher_attr.io import read_jsonl

                iterator = read_jsonl(path)
            else:
                import pyarrow.parquet as pq

                iterator = (
                    row
                    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024)
                    for row in batch.to_pylist()
                )
            for index, row in enumerate(iterator):
                yield path.stem, index, row, description
    else:
        from datasets import load_dataset

        sources = load_dataset(hf_name, subset, streaming=True)
        for split in sorted(sources):
            for index, row in enumerate(sources[split]):
                yield (
                    split,
                    index,
                    row,
                    {
                        "dataset": hf_name,
                        "subset": subset,
                        "revision": "unresolved; persist prompt files",
                    },
                )


def split_for(group: str, fractions: dict, seed: int) -> str:
    value = int(fingerprint([seed, group])[:16], 16) / 2**64
    cumulative = 0.0
    for split in SPLITS:
        cumulative += fractions[split]
        if value < cumulative:
            return split
    return "test"


def audit_splits(splits: dict[str, list[dict]]) -> dict:
    owners: dict[tuple, str] = {}
    counts = {}
    for split, rows in splits.items():
        seen_ids = {}
        for row in rows:
            if not row.get("prompt", "").strip():
                raise ValueError(f"Empty prompt in {split}")
            pid = row["prompt_id"]
            signature = text_hash(row["prompt"])
            if pid in seen_ids and seen_ids[pid] != signature:
                raise ValueError(f"Prompt ID {pid} identifies different text")
            seen_ids[pid] = signature
            keys = [("id", pid), ("text", signature)]
            if row.get("source_group"):
                keys.append(("source", row["source_dataset"], row["source_group"]))
            for key in keys:
                previous = owners.setdefault(key, split)
                if previous != split:
                    raise ValueError(f"Split overlap between {previous} and {split}: {key}")
        counts[split] = {
            "rows": len(rows),
            "prompts": len(seen_ids),
            "datasets": dict(Counter(r["source_dataset"] for r in rows)),
        }
    return counts


def prepare(cfg: dict) -> dict:
    root = initialize_run(cfg)
    manifest_path = root / "prompts" / "manifest.json"
    if manifest_path.exists():
        return verify_prompts(root)
    if any((root / "prompts" / f"{s}.jsonl").exists() for s in SPLITS):
        raise ValueError("Incomplete prompt build; choose a new run_dir")
    settings = cfg["data"]
    splits = {s: [] for s in SPLITS}
    seen_text = set()
    provenance = {}
    scanned = Counter()
    for name in settings["datasets"]:
        if name not in SOURCES:
            raise ValueError(f"Unsupported dataset: {name}; choose from {list(SOURCES)}")
        counts = Counter()
        for source_split, index, raw, source in source_rows(name, settings):
            scanned[name] += 1
            provenance[fingerprint(source)] = source
            prompt = format_source(name, raw, settings)
            if not prompt:
                continue
            digest = text_hash(prompt)
            if digest in seen_text:
                continue
            seen_text.add(digest)
            source_id = raw.get("id") or raw.get("reviewId") or digest
            group = str(raw.get("movieId") or source_id)
            split = split_for(f"{name}:{group}", settings["split_fractions"], cfg["seed"])
            if counts[split] >= settings["prompts_per_dataset"][split]:
                continue
            splits[split].append(
                {
                    "prompt_id": digest,
                    "prompt": prompt,
                    "split": split,
                    "source_dataset": name,
                    "task": SOURCES[name][2],
                    "source_split": source_split,
                    "source_index": index,
                    "source_id": str(source_id),
                    "source_group": group,
                }
            )
            counts[split] += 1
            if all(counts[s] == settings["prompts_per_dataset"][s] for s in SPLITS):
                break
        if any(counts[s] < settings["prompts_per_dataset"][s] for s in SPLITS):
            raise ValueError(
                f"Insufficient disjoint {name} prompts: {dict(counts)}; reduce targets"
            )
    audit = audit_splits(splits)
    hashes = {}
    for split, rows in splits.items():
        rows.sort(key=lambda r: r["prompt_id"])
        path = root / "prompts" / f"{split}.jsonl"
        write_jsonl(path, rows)
        hashes[split] = file_hash(path)
    result = {
        "schema_version": 2,
        "split_policy": "stable source-group hash with text deduplication",
        "audit": audit,
        "sha256": hashes,
        "sources": list(provenance.values()),
        "source_rows_scanned": dict(scanned),
    }
    save_json(manifest_path, result)
    return result


def verify_prompts(root: Path) -> dict:
    manifest = json.loads((root / "prompts" / "manifest.json").read_text(encoding="utf-8"))
    splits = {}
    for split in SPLITS:
        path = root / "prompts" / f"{split}.jsonl"
        if file_hash(path) != manifest["sha256"][split]:
            raise ValueError(f"Prompt artifact changed: {path}")
        splits[split] = load_jsonl(path)
        if len({r["prompt_id"] for r in splits[split]}) != len(splits[split]):
            raise ValueError(f"Duplicate prompt IDs in {split}")
    audit_splits(splits)
    return manifest
