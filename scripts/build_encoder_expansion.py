"""Build one encoder-training view from the frozen parent pairs and shared new outputs."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from teacher_attr.config import SPLITS, file_hash, load_config
from teacher_attr.generation import output_path, read_outputs
from teacher_attr.io import load_jsonl, save_json, save_yaml, write_jsonl
from teacher_attr.pairs import align, load_pairs
from teacher_attr.prompts import audit_splits, verify_prompts


def build_view(config, per_dataset):
    cfg = load_config(config)
    root = Path(cfg["run_dir"])
    manifest = verify_prompts(root)
    parent_cfg = load_config(manifest["parent_config"])
    parent = Path(parent_cfg["run_dir"])
    if verify_prompts(parent)["sha256"] != manifest["parent_prompt_hashes"]:
        raise ValueError("Parent prompts changed")
    subsets = root / "prompts/encoder_subsets.json"
    if file_hash(subsets) != manifest["subsets_sha256"]:
        raise ValueError("Expansion subsets changed")
    selected = set(json.loads(subsets.read_text())[str(per_dataset)])
    destination = root / f"encoder_plus_{len(selected)}"
    if destination.exists():
        load_pairs(load_config(destination / "encoder.yaml"))
        return destination / "encoder.yaml"

    original = load_pairs(parent_cfg)
    new_prompts = load_jsonl(root / "prompts/train.jsonl")
    sources = [subsets, parent / "pairs/manifest.json"]
    sources.extend(parent / "pairs" / f"{s}.jsonl" for s in SPLITS)
    flags = {}
    models = {"teachers": {}, "students": {}}
    for role, outputs in models.items():
        for name, model in cfg[role].items():
            if role == "students" and "train" not in model["splits"]:
                continue
            path = output_path(root, role, name, "train")
            sources.extend([path, path.with_suffix(".meta.json")])
            compact = []
            for row in read_outputs(cfg, role, name, "train"):
                compact.append(
                    {
                        k: row[k]
                        for k in (
                            "prompt_id",
                            "prompt",
                            "split",
                            "response",
                            "model_id",
                            "quality_flags",
                        )
                    }
                )
                issues = row["quality_flags"] or ([] if row["response"].strip() else ["empty"])
                if issues:
                    flags.setdefault(row["prompt_id"], {})[name] = issues
            outputs[name] = compact

    fresh = align(new_prompts, models["teachers"], models["students"], cfg["students"], set(flags))
    rows = {
        **original,
        "train": original["train"] + [r for r in fresh if r["prompt_id"] in selected],
    }
    cfg["run_dir"] = str(destination)
    with TemporaryDirectory(dir=root) as tmp:
        stage = Path(tmp) / "view"
        prompt_hashes, pair_hashes, pools = {}, {}, {}
        for split in SPLITS:
            pools[split] = load_jsonl(parent / "prompts" / f"{split}.jsonl")
            if split == "train":
                pools[split].extend(r for r in new_prompts if r["prompt_id"] in selected)
            for folder, data, hashes in (
                ("prompts", pools[split], prompt_hashes),
                ("pairs", rows[split], pair_hashes),
            ):
                path = stage / folder / f"{split}.jsonl"
                write_jsonl(path, data)
                hashes[split] = file_hash(path)
        save_json(
            stage / "prompts/manifest.json", dict(sha256=prompt_hashes, audit=audit_splits(pools))
        )
        parent_exclusions = json.loads((parent / "pairs/manifest.json").read_text())[
            "attribution_exclusions"
        ]
        excluded = {
            s: sorted(set(r["prompt_id"] for r in pools[s]) - set(r["prompt_id"] for r in rows[s]))
            for s in SPLITS
        }
        exclusions = dict(
            schema_version=1,
            splits=excluded,
            reason="Parent exclusions and matched expansion quality exclusions.",
        )
        save_json(stage / "attribution_exclusions.json", exclusions)
        save_json(
            stage / "pairs/manifest.json",
            dict(
                schema_version=2,
                attribution_exclusions=exclusions,
                parent_exclusions=parent_exclusions,
                expansion_flags={p: issues for p, issues in flags.items() if p in selected},
                audit=audit_splits(rows),
                sha256=pair_hashes,
                protocol=cfg["protocol"],
                outputs_sha256={os.path.relpath(p, destination): file_hash(p) for p in sources},
                teacher_ids=list(cfg["teachers"]),
            ),
        )
        save_yaml(stage / "encoder.yaml", cfg)
        verify_prompts(stage)
        stage.rename(destination)
    return destination / "encoder.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--per-dataset", type=int, required=True)
    args = parser.parse_args()
    print(build_view(args.config, args.per_dataset))
