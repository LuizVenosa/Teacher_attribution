"""Prepare nested teacher and student encoder-training prompts alongside a frozen research run."""

import argparse
import json
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

from teacher_attr.config import file_hash, fingerprint, load_config
from teacher_attr.io import load_jsonl, save_json, save_yaml, write_jsonl
from teacher_attr.prompts import audit_splits, text_hash, verify_prompts
from teacher_attr.research import POOLS, source_records, user_prompt


def prepare(config, destination, per_dataset):
    cfg = load_config(config)
    original = Path(cfg["run_dir"])
    destination = destination.resolve()
    if destination.exists() or destination.is_relative_to(original) or per_dataset < 1:
        raise ValueError("Choose a new sibling directory and a positive per-dataset count")
    if json.loads((original / "experiment.json").read_text(encoding="utf-8")) != cfg:
        raise ValueError("Source config must match the original experiment")
    manifest = verify_prompts(original)
    seen, groups = set(), set()
    for pool in POOLS:
        path = original / "prompts" / f"{pool}.jsonl"
        if file_hash(path) != manifest["sha256"][pool]:
            raise ValueError(f"Changed source pool: {pool}")
        for row in load_jsonl(path):
            seen.add(text_hash(row["prompt"]))
            groups.add((row["source_dataset"], row["source_group"]))

    research = cfg["research"]
    fresh = []
    nested = {
        size: [] for size in sorted({n for n in (500, 1000, per_dataset) if n <= per_dataset})
    }
    for name, source in research["sources"].items():
        candidates = []
        for row, _ in islice(source_records(source), research["scan_limit"]):
            prompt = user_prompt(name, row, source, research["max_input_chars"])
            if not prompt:
                continue
            digest = text_hash(prompt)
            group = str(row.get("movieId") or row.get("id") or digest)
            if digest in seen or (name, group) in groups:
                continue
            seen.add(digest)
            groups.add((name, group))
            candidates.append(
                dict(
                    prompt_id=digest,
                    prompt=prompt,
                    source_dataset=name,
                    source=name,
                    task=source["task"],
                    source_group=group,
                    split="train",
                )
            )
        if len(candidates) < per_dataset:
            raise ValueError(f"{name}: only {len(candidates)} eligible prompts within scan limit")
        candidates.sort(key=lambda r: fingerprint([cfg["seed"], "additional", r["prompt_id"]]))
        selected = candidates[:per_dataset]
        fresh.extend(selected)
        for size, ids in nested.items():
            ids.extend(row["prompt_id"] for row in selected[:size])

    cfg["run_dir"] = str(destination)
    roots = {Path(s["hf_name"]).parents[2] for s in cfg["students"].values()}
    if len(roots) != 1:
        raise ValueError("Expected a common student checkpoint root")
    research["student_root"] = str(roots.pop())
    for source in research["sources"].values():
        source["counts"]["train"] = per_dataset
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temp:
        stage = Path(temp) / "run"
        pools = {"train": fresh}
        for split in ("val", "test"):
            pools[split] = load_jsonl(original / "prompts" / f"{split}.jsonl")
        hashes = {}
        for split, rows in pools.items():
            path = stage / "prompts" / f"{split}.jsonl"
            write_jsonl(path, rows)
            hashes[split] = file_hash(path)
        subset_path = stage / "prompts/encoder_subsets.json"
        save_json(subset_path, nested)
        save_json(
            stage / "prompts/manifest.json",
            dict(
                sha256=hashes,
                audit=audit_splits(pools),
                parent_prompt_hashes=manifest["sha256"],
                parent_config=str(config.resolve()),
                subsets_sha256=file_hash(subset_path),
                purpose="encoder expansion",
            ),
        )
        save_yaml(stage / "generation.yaml", cfg)
        resolved = load_config(stage / "generation.yaml")
        if resolved["students"] != cfg["students"]:
            raise ValueError("Generated config changed checkpoint identities")
        verify_prompts(stage)
        stage.rename(destination)
    return destination / "generation.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-dataset", default=2000, type=int)
    args = parser.parse_args()
    print(prepare(args.config, args.output, args.per_dataset))
