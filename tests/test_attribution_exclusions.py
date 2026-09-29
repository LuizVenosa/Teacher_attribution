import pytest

from teacher_attr.pairs import align


def test_exclusions_keep_full_alignment_checks():
    prompts = [{"prompt_id": p, "prompt": p, "split": "train"} for p in ("a", "b")]

    def rows(name):
        return [
            {
                **p,
                "model_id": name,
                "response": "answer",
                "quality_flags": ["identity_marker"] if p["prompt_id"] == "a" else [],
            }
            for p in prompts
        ]

    teachers, students = {"t": rows("t")}, {"s": rows("s")}
    config = {"s": {"teacher": "t"}}
    with pytest.raises(ValueError, match="flagged"):
        align(prompts, teachers, students, config)
    result = align(prompts, teachers, students, config, {"a"})
    assert [r["prompt_id"] for r in result] == ["b"]
    assert len(teachers["t"]) == 2
    with pytest.raises(ValueError, match="missing"):
        align(prompts, {"t": teachers["t"][1:]}, students, config, {"a"})
