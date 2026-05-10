from __future__ import annotations

import argparse
import logging
from typing import Any, Callable

import numpy as np

from teacher_attr.baselines import (
    load_spacy_model,
    pos_template_scores,
    sentence_embedding_scores,
    student_only_classifier_scores,
    tfidf_scores,
)
from teacher_attr.io import load_jsonl, load_yaml, save_json
from teacher_attr.metrics import (
    accuracy_by_task,
    accuracy_from_grouped,
    classification_metrics,
    grouped_classification_metrics,
)
from teacher_attr.utils import setup_logging, teacher_order_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run teacher-attribution baselines.")
    parser.add_argument("--models_config", required=True)
    parser.add_argument("--train_pairs", default=None)
    parser.add_argument("--test_pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sentence_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model for embedding baseline.",
    )
    parser.add_argument("--skip_sentence", action="store_true")
    parser.add_argument("--teacher_ids", default=None, help="Optional space/comma/colon-separated teacher IDs to evaluate.")
    return parser.parse_args()


def parse_id_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    normalized = value.replace(",", " ").replace(":", " ")
    return [item for item in normalized.split() if item]


def select_teacher_ids(models_cfg: dict[str, Any], requested: str | None) -> list[str]:
    all_teacher_ids = teacher_order_from_config(models_cfg)
    requested_ids = parse_id_list(requested)
    if requested_ids is None:
        return all_teacher_ids
    unknown = sorted(set(requested_ids) - set(all_teacher_ids))
    if unknown:
        raise ValueError(f"Unknown teacher IDs in --teacher_ids: {unknown}")
    requested_set = set(requested_ids)
    return [teacher_id for teacher_id in all_teacher_ids if teacher_id in requested_set]


def labels_for(rows: list[dict], teacher_ids: list[str]) -> np.ndarray:
    return np.asarray([teacher_ids.index(row["true_teacher"]) for row in rows], dtype=np.int64)


def group_value(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    return str(value) if value else "unknown"


def student_teacher_pair(row: dict[str, Any]) -> str:
    student_id = row.get("anchor_student_id") or row.get("student_id") or "unknown_student"
    true_teacher = row.get("true_teacher") or "unknown_teacher"
    return f"{student_id}->{true_teacher}"


def add_breakdowns(
    metrics: dict[str, Any],
    scores: np.ndarray,
    labels: np.ndarray,
    rows: list[dict],
    teacher_ids: list[str],
) -> None:
    metrics["accuracy_by_task"] = accuracy_by_task(
        scores,
        labels,
        [row.get("task") for row in rows],
    )

    by_dataset = grouped_classification_metrics(
        scores,
        labels,
        [group_value(row, "source_dataset") for row in rows],
        teacher_ids,
    )
    by_pair = grouped_classification_metrics(
        scores,
        labels,
        [student_teacher_pair(row) for row in rows],
        teacher_ids,
    )
    by_task = grouped_classification_metrics(
        scores,
        labels,
        [row.get("task") for row in rows],
        teacher_ids,
    )

    metrics["accuracy_by_dataset"] = accuracy_from_grouped(by_dataset)
    metrics["accuracy_by_student_teacher_pair"] = accuracy_from_grouped(by_pair)
    metrics["breakdown_by_task"] = by_task
    metrics["breakdown_by_dataset"] = by_dataset
    metrics["breakdown_by_student_teacher_pair"] = by_pair


def add_method(
    results: dict,
    name: str,
    score_fn: Callable[[], np.ndarray],
    rows: list[dict],
    labels: np.ndarray,
    teacher_ids: list[str],
) -> None:
    try:
        scores = score_fn()
        metrics = classification_metrics(scores, labels, teacher_ids)
        add_breakdowns(metrics, scores, labels, rows, teacher_ids)
        results[name] = metrics
        logging.info("%s accuracy: %.4f", name, metrics["accuracy"])
    except Exception as exc:
        logging.exception("%s failed", name)
        results[name] = {"error": repr(exc)}


def main() -> None:
    setup_logging()
    args = parse_args()

    models_cfg = load_yaml(args.models_config)
    teacher_ids = select_teacher_ids(models_cfg, args.teacher_ids)
    test_rows = load_jsonl(args.test_pairs)
    labels = labels_for(test_rows, teacher_ids)
    results: dict = {
        "num_test_rows": len(test_rows),
        "teacher_ids": teacher_ids,
        "methods": {},
    }

    add_method(
        results["methods"],
        "tfidf_same_prompt",
        lambda: tfidf_scores(test_rows, teacher_ids),
        test_rows,
        labels,
        teacher_ids,
    )

    nlp = load_spacy_model()
    add_method(
        results["methods"],
        "pos_template",
        lambda: pos_template_scores(test_rows, teacher_ids, nlp=nlp),
        test_rows,
        labels,
        teacher_ids,
    )
    results["methods"]["pos_template"]["backend"] = "spacy" if nlp is not None else "shape_fallback"

    if not args.skip_sentence:
        add_method(
            results["methods"],
            "sentence_embedding_same_prompt",
            lambda: sentence_embedding_scores(
                test_rows,
                teacher_ids,
                model_name=args.sentence_model,
            ),
            test_rows,
            labels,
            teacher_ids,
        )

    if args.train_pairs:
        train_rows = load_jsonl(args.train_pairs)
        add_method(
            results["methods"],
            "student_only_tfidf_classifier",
            lambda: student_only_classifier_scores(train_rows, test_rows, teacher_ids),
            test_rows,
            labels,
            teacher_ids,
        )

    save_json(args.output, results)
    logging.info("Wrote baseline metrics to %s", args.output)


if __name__ == "__main__":
    main()
