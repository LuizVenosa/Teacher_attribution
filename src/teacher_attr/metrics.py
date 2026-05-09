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