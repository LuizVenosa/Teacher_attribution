import json

import pytest

from teacher_attr import exclusions


def test_matched_exclusion_is_bound_to_outputs(tmp_path, monkeypatch):
    cfg = {"run_dir": str(tmp_path), "teachers": {"a": {}, "b": {}}}
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    rows = [{"prompt_id": "bad"}, {"prompt_id": "good"}]
    (prompts / "distill_train.jsonl").write_text("\n".join(map(json.dumps, rows)))
    monkeypatch.setattr(exclusions, "initialize_run", lambda cfg: tmp_path)
    monkeypatch.setattr(exclusions, "read_outputs", lambda *args: rows)
    monkeypatch.setattr(exclusions, "output_path", lambda root, role, t, split: root / t)
    for t in cfg["teachers"]:
        (tmp_path / t).write_text(t)
    report = exclusions.exclude_training_prompts(cfg, ["bad"], "Reviewed markup")
    assert exclusions.training_exclusions(cfg) == report
    assert report["prompt_ids"] == ["bad"]
    assert set(report["teacher_outputs_sha256"]) == {"a", "b"}
    with pytest.raises(ValueError, match="frozen"):
        exclusions.exclude_training_prompts(cfg, ["good"], "Changed decision")
    (tmp_path / "a").write_text("modified")
    with pytest.raises(ValueError, match="changed"):
        exclusions.training_exclusions(cfg)


def test_no_manifest_preserves_existing_runs(tmp_path):
    assert exclusions.training_exclusions({"run_dir": str(tmp_path)}) is None
