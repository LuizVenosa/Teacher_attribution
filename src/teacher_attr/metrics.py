from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize


def topk_accuracy(scores: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    if len(labels) == 0:
        return 0.0
    topk = np.argsort(-scores, axis=1)[:, :k]
    return float(np.mean([label in row for label, row in zip(labels, topk, strict=True)]))


def classification_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    teacher_ids: list[str],
) -> dict[str, Any]:
    preds = scores.argmax(axis=1)
    metrics: dict[str, Any] = {
        "accuracy": float(np.mean(preds == labels)) if len(labels) else 0.0,
        "top2_accuracy": topk_accuracy(scores, labels, k=min(2, len(teacher_ids))),
        "confusion_matrix": confusion_matrix(
            labels,
            preds,
            labels=list(range(len(teacher_ids))),
        ).tolist(),
        "teacher_ids": teacher_ids,
    }
    metrics.update(roc_auc_metrics(scores, labels, teacher_ids))
    return metrics


def roc_auc_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    teacher_ids: list[str],
) -> dict[str, Any]:
    """Return one-vs-rest ROC AUC metrics from class scores.

    Scores do not need to be probabilities; ROC AUC only requires a ranking.
    Per-class AUC is null when the class is absent or the metric is undefined.
    """
    if len(labels) == 0 or scores.size == 0:
        return {
            "roc_auc_ovr_macro": None,
            "roc_auc_ovr_weighted": None,
            "roc_auc_ovr_micro": None,
            "roc_auc_by_teacher": {teacher_id: None for teacher_id in teacher_ids},
        }

    classes = list(range(len(teacher_ids)))
    y_true = label_binarize(labels, classes=classes)
    if y_true.shape[1] == 1:
        return {
            "roc_auc_ovr_macro": None,
            "roc_auc_ovr_weighted": None,
            "roc_auc_ovr_micro": None,
            "roc_auc_by_teacher": {teacher_id: None for teacher_id in teacher_ids},
        }

    out: dict[str, Any] = {}
    for average in ("macro", "weighted", "micro"):
        try:
            out[f"roc_auc_ovr_{average}"] = float(
                roc_auc_score(y_true, scores, average=average, multi_class="ovr")
            )
        except ValueError:
            out[f"roc_auc_ovr_{average}"] = None

    by_teacher: dict[str, float | None] = {}
    for idx, teacher_id in enumerate(teacher_ids):
        positives = int(y_true[:, idx].sum())
        negatives = int(len(labels) - positives)
        if positives == 0 or negatives == 0:
            by_teacher[teacher_id] = None
            continue
        try:
            by_teacher[teacher_id] = float(roc_auc_score(y_true[:, idx], scores[:, idx]))
        except ValueError:
            by_teacher[teacher_id] = None
    out["roc_auc_by_teacher"] = by_teacher
    return out


def grouped_classification_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: list[str | None],
    teacher_ids: list[str],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, group in enumerate(groups):
        grouped[group or "unknown"].append(idx)

    out: dict[str, dict[str, Any]] = {}
    for group, indexes in sorted(grouped.items()):
        idx = np.asarray(indexes, dtype=np.int64)
        metrics = classification_metrics(scores[idx], labels[idx], teacher_ids)
        metrics["num_rows"] = int(len(idx))
        metrics["true_teacher_counts"] = _label_counts(labels[idx], teacher_ids)
        metrics["predicted_teacher_counts"] = _label_counts(scores[idx].argmax(axis=1), teacher_ids)
        out[group] = metrics
    return out


def accuracy_from_grouped(
    grouped_metrics: dict[str, dict[str, Any]],
) -> dict[str, float]:
    return {group: float(metrics["accuracy"]) for group, metrics in grouped_metrics.items()}


def _label_counts(values: np.ndarray, teacher_ids: list[str]) -> dict[str, int]:
    counts = {teacher_id: 0 for teacher_id in teacher_ids}
    for value in values:
        counts[teacher_ids[int(value)]] += 1
    return counts


def accuracy_by_task(
    scores: np.ndarray,
    labels: np.ndarray,
    tasks: list[str | None],
) -> dict[str, float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    preds = scores.argmax(axis=1)
    for idx, task in enumerate(tasks):
        grouped[task or "unknown"].append(idx)
    return {
        task: float(np.mean(preds[indexes] == labels[indexes]))
        for task, indexes in grouped.items()
    }
