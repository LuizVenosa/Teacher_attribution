from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix


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
    return {
        "accuracy": float(np.mean(preds == labels)) if len(labels) else 0.0,
        "top2_accuracy": topk_accuracy(scores, labels, k=min(2, len(teacher_ids))),
        "confusion_matrix": confusion_matrix(
            labels,
            preds,
            labels=list(range(len(teacher_ids))),
        ).tolist(),
        "teacher_ids": teacher_ids,
    }


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
