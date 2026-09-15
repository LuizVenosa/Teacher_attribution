import copy
from pathlib import Path

import pytest

from teacher_attr.config import SPLITS, initialize_run, validate_config
from teacher_attr.io import load_jsonl
from teacher_attr.pairs import align
from teacher_attr.prompts import audit_splits, prepare, split_for, verify_prompts


def test_shared_sources_are_disjoint_and_repeatable(config, sources):
    first = prepare(config)
    assert first == prepare(config)
    root = Path(config["run_dir"])
    rows = {s: load_jsonl(root / "prompts" / f"{s}.jsonl") for s in SPLITS}
    assert [len(rows[s]) for s in SPLITS] == [6, 4, 4]
    assert sum(len({r["prompt_id"] for r in v}) for v in rows.values()) == 14
    assert audit_splits(rows) == first["audit"]
    changed = root / "prompts" / "train.jsonl"
    changed.write_text(changed.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        verify_prompts(root)


def test_normalized_text_and_source_overlap_are_rejected():
    a = {"prompt_id": "a", "prompt": "Example TEXT", "source_dataset": "x", "source_group": "1"}
    b = {**a, "prompt_id": "b", "prompt": " example   text ", "source_group": "2"}
    with pytest.raises(ValueError, match="overlap"):
        audit_splits({"train": [a], "test": [b]})
    with pytest.raises(ValueError, match="overlap"):
        audit_splits({"train": [a], "test": [{**a, "prompt_id": "c", "prompt": "different"}]})


def test_group_split_independent_of_requested_count(config):
    fractions = config["data"]["split_fractions"]
    assert split_for("movie:12", fractions, 13) == split_for("movie:12", fractions, 13)


def test_changed_run_config_is_rejected(config):
    initialize_run(config)
    config["seed"] += 1
    with pytest.raises(ValueError, match="configuration changed"):
        initialize_run(config)


def test_held_out_protocol_rejects_seen_student_checkpoints(config):
    validate_config(config)
    config["protocol"] = "held_out_students"
    with pytest.raises(ValueError, match="disjoint"):
        validate_config(config)


def test_held_out_protocol_accepts_distinct_checkpoints(config):
    students = config["students"]
    config["students"] = {}
    config["protocol"] = "held_out_students"
    for name, model in students.items():
        for split in SPLITS:
            config["students"][f"{name}_{split}"] = {
                **model,
                "hf_name": f"{model['hf_name']}-{split}",
                "splits": [split],
            }
    validate_config(config)


def test_base_ancestry_is_not_accepted_as_teacher_evidence(config):
    config["students"]["distilgpt2"]["relation"] = "base_initialization"
    with pytest.raises(ValueError, match="relation"):
        validate_config(config)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "prompt", "empty", "model"])
def test_pair_alignment_fails_on_bad_outputs(failure):
    prompt = {"prompt_id": "p", "prompt": "Question", "split": "test"}
    teacher = {**prompt, "response": "Answer", "model_id": "t"}
    student = {**prompt, "response": "Other", "model_id": "s"}
    rows = [copy.deepcopy(teacher)]
    if failure == "missing":
        rows = []
    elif failure == "duplicate":
        rows *= 2
    elif failure == "prompt":
        rows[0]["prompt"] = "Wrong prompt"
    elif failure == "empty":
        rows[0]["response"] = " "
    else:
        rows[0]["model_id"] = "wrong"
    with pytest.raises(ValueError):
        align([prompt], {"t": rows}, {"s": [student]}, {"s": {"teacher": "t"}})
