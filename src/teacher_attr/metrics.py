from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


def classification_metrics(scores, labels, teacher_ids: list[str]) -> dict:
    scores, labels = np.asarray(scores), np.asarray(labels)
    if scores.shape != (len(labels), len(teacher_ids)) or not len(labels):
        raise ValueError("Expected a nonempty rows-by-teachers score matrix")
    if not np.isfinite(scores).all() or not np.isin(labels, np.arange(len(teacher_ids))).all():
        raise ValueError("Nonfinite scores or invalid labels")
    pred = scores.argmax(axis=1)
    auc = {}
    for index, teacher in enumerate(teacher_ids):
        target = labels == index
        auc[teacher] = (
            float(roc_auc_score(target, scores[:, index]))
            if target.any() and not target.all()
            else None
        )
    available = [v for v in auc.values() if v is not None]
    return {
        "num_rows": len(labels),
        "accuracy": float(np.mean(pred == labels)),
        "chance_accuracy": 1 / len(teacher_ids),
        "macro_f1": float(
            f1_score(labels, pred, labels=range(len(teacher_ids)), average="macro", zero_division=0)
        ),
        "per_teacher_accuracy": {
            t: float(np.mean(pred[labels == i] == i)) if np.any(labels == i) else None
            for i, t in enumerate(teacher_ids)
        },
        "top2_accuracy": float(
            np.mean(np.any(np.argsort(-scores, axis=1)[:, :2] == labels[:, None], axis=1))
        ),
        "roc_auc_ovr_macro": float(np.mean(available)) if available else None,
        "roc_auc_by_teacher": auc,
        "confusion_matrix": confusion_matrix(labels, pred, labels=range(len(teacher_ids))).tolist(),
    }


def grouped_indexes(groups):
    result = defaultdict(list)
    for index, group in enumerate(groups):
        result[group].append(index)
    return dict(sorted(result.items()))


def cluster_interval(values, groups, repeats: int, seed: int) -> list[float]:
    """Bootstrap row means, sampling whole prompt clusters with replacement."""
    values = np.asarray(values, dtype=float)
    indexes = list(grouped_indexes(groups).values())
    totals = np.array([values[ix].sum() for ix in indexes])
    counts = np.array([len(ix) for ix in indexes])
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repeats):
        selected = rng.integers(len(indexes), size=len(indexes))
        samples.append(totals[selected].sum() / counts[selected].sum())
    return np.quantile(samples, [0.025, 0.975]).tolist()


def support_sets(rows: list[dict], sizes: list[int], repeats: int, seed: int) -> list[dict]:
    """One support manifest for all methods; keep student checkpoints separate."""
    rng = np.random.default_rng(seed)
    result = []
    for scope in ["all", *sorted({r["source_dataset"] for r in rows})]:
        groups = defaultdict(list)
        for index, row in enumerate(rows):
            if scope == "all" or row["source_dataset"] == scope:
                groups[row["student_id"]].append(index)
        for size in sorted(set(sizes) - {1}):
            for repeat in range(repeats):
                for student, indexes in sorted(groups.items()):
                    if len(indexes) >= size:
                        selected = rng.choice(indexes, size=size, replace=False).tolist()
                        result.append(
                            {
                                "scope": scope,
                                "size": size,
                                "repeat": repeat,
                                "student_id": student,
                                "row_indexes": selected,
                            }
                        )
    return result


def aggregate_scores(
    scores, labels, supports: list[dict], teacher_ids: list[str], bootstrap_repeats: int, seed: int
) -> dict:
    groups = defaultdict(list)
    for support in supports:
        groups[f"{support['scope']}/{support['size']}"].append(support)
    out = {}
    for key, sets in groups.items():
        indexes = [s["row_indexes"] for s in sets]
        if any(len(set(labels[ix])) != 1 for ix in indexes):
            raise ValueError("Support set mixes teacher labels")
        # The same rule applies to lexical, frozen, and trained models.
        aggregated = np.stack([scores[ix].mean(axis=0) for ix in indexes])
        targets = np.array([labels[ix[0]] for ix in indexes])
        result = classification_metrics(aggregated, targets, teacher_ids)
        result["num_sets"] = len(sets)
        result["monte_carlo_accuracy_ci95"] = cluster_interval(
            aggregated.argmax(1) == targets, [s["repeat"] for s in sets], bootstrap_repeats, seed
        )
        result["interval_scope"] = (
            "support sampling conditional on fixed checkpoints and response pool"
        )
        out[key] = result
    return out
