import json
from pathlib import Path

import pytest

from teacher_attr.budget_migration import migrate_budgets
from teacher_attr.config import file_hash, fingerprint, initialize_run, load_config
from teacher_attr.generation import generation_spec, output_path, read_outputs
from teacher_attr.io import load_yaml, save_json, save_yaml, write_jsonl
from teacher_attr.research import POOLS


def test_budget_reuse_preserves_answers_and_original_provenance(tmp_path):
    raw = load_yaml(Path(__file__).parents[1] / "configs/research_qc_v3.yaml")
    raw["run_dir"] = str(tmp_path / "original")
    source_config = tmp_path / "source.yaml"
    save_yaml(source_config, raw)
    cfg = load_config(source_config)
    root = initialize_run(cfg)
    hashes = {}
    for split in POOLS:
        rows = [
            {
                "prompt_id": split,
                "prompt": split,
                "task": "instruction",
                "source_dataset": "fixture",
            }
        ]
        path = root / "prompts" / f"{split}.jsonl"
        write_jsonl(path, rows)
        hashes[split] = file_hash(path)
    save_json(root / "prompts/nested_subsets.json", {"5000": ["distill_train"]})
    save_json(
        root / "prompts/manifest.json",
        {
            "sha256": hashes,
            "nested_subsets_sha256": file_hash(root / "prompts/nested_subsets.json"),
        },
    )
    meta = {
        "spec": generation_spec(cfg, "teachers", "gemma", "distill_train"),
        "response_processing_version": 2,
        "resolved_revision": "original",
    }
    path = output_path(root, "teachers", "gemma", "distill_train")
    save_json(path.with_suffix(".meta.json"), meta)
    write_jsonl(
        path,
        [
            {
                "prompt_id": "distill_train",
                "prompt": "distill_train",
                "response": "unchanged answer",
                "generation_config": cfg["generation"],
                "generation_fingerprint": fingerprint(meta),
            }
        ],
    )
    before = file_hash(path)
    destination = tmp_path / "larger.yaml"
    result = migrate_budgets(str(source_config), str(destination), 4608, 4096)
    new_cfg = load_config(destination)
    rows = read_outputs(new_cfg, "teachers", "gemma", "distill_train")
    assert rows[0]["response"] == "unchanged answer"
    assert rows[0]["generation_config"]["max_input_tokens"] == 1024
    assert file_hash(path) == before
    new_path = output_path(Path(new_cfg["run_dir"]), "teachers", "gemma", "distill_train")
    new_meta = json.loads(new_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert new_meta["reuse"]["original_manifest"] == meta
    assert new_meta["reuse"]["source_sha256"] == before
    assert new_meta["spec"]["generation"]["max_input_tokens"] == 4608
    assert (
        file_hash(Path(new_cfg["run_dir"]) / "prompts/distill_train.jsonl")
        == hashes["distill_train"]
    )
    assert result["reused"] == [{"teacher": "gemma", "split": "distill_train", "rows": 1}]
    with pytest.raises(ValueError, match="already exists"):
        migrate_budgets(str(source_config), str(destination), 4608, 4096)
    with pytest.raises(ValueError, match="increasing"):
        migrate_budgets(str(source_config), str(tmp_path / "smaller.yaml"), 512, 4096)
