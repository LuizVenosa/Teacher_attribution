import numpy as np
import pytest

from teacher_attr.metrics import (
    aggregate_scores,
    classification_metrics,
    cluster_interval,
    support_sets,
)


def test_binary_auc_is_computed():
    result = classification_metrics([[0.9, 0.1], [0.1, 0.9]], [0, 1], ["a", "b"])
    assert result["roc_auc_ovr_macro"] == 1
    assert result["roc_auc_by_teacher"] == {"a": 1, "b": 1}
    assert result["accuracy"] == 1


def test_absent_class_auc_is_null_and_nonfinite_scores_fail():
    assert classification_metrics([[1, 0]], [0], ["a", "b"])["roc_auc_ovr_macro"] is None
    with pytest.raises(ValueError, match="Nonfinite"):
        classification_metrics([[float("nan"), 0]], [0], ["a", "b"])


def test_paired_bootstrap_preserves_shared_prompt_dependence():
    # Every prompt has opposite outcomes; each whole-prompt difference is exactly zero.
    assert cluster_interval([1, -1, 1, -1], ["p", "p", "q", "q"], 200, 13) == [0, 0]


def test_support_sets_are_shared_and_do_not_mix_students():
    rows = [
        {"student_id": s, "source_dataset": ds, "prompt_id": f"{ds}-{i}"}
        for ds in ("qa", "news")
        for i in range(5)
        for s in ("s1", "s2")
    ]
    sets = support_sets(rows, [1, 2, 4], 10, 13)
    assert sets == support_sets(rows, [1, 2, 4], 10, 13)
    for item in sets:
        selected = [rows[i] for i in item["row_indexes"]]
        assert len({r["student_id"] for r in selected}) == 1
        assert len(set(item["row_indexes"])) == item["size"]
        if item["scope"] != "all":
            assert all(r["source_dataset"] == item["scope"] for r in selected)


def test_aggregation_is_mean_of_scores():
    scores = np.array([[0.9, 0.1], [0.2, 0.8]])
    labels = np.array([0, 0])
    sets = [{"scope": "all", "size": 2, "repeat": 0, "row_indexes": [0, 1]}]
    result = aggregate_scores(scores, labels, sets, ["a", "b"], 20, 13)["all/2"]
    assert result["accuracy"] == 1
    assert result["num_sets"] == 1
