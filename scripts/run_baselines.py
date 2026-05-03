from __future__ import annotations

import argparse
import logging
from typing import Callable

import numpy as np

from teacher_attr.baselines import (
    load_spacy_model,
    pos_template_scores,
    sentence_embedding_scores,
    student_only_classifier_scores,
    tfidf_scores,
)
from teacher_attr.io import load_jsonl, load_yaml, save_json
from teacher_attr.metrics import accuracy_by_task, classification_metrics
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
    return parser.parse_args()


def labels_for(rows: list[dict], teacher_ids: list[str]) -> np.ndarray:
    return np.asarray([teacher_ids.index(row["true_teacher"]) for row in rows], dtype=np.int64)


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
        metrics["accuracy_by_task"] = accuracy_by_task(
            scores,
            labels,
            [row.get("task") for row in rows],
        )
        results[name] = metrics
        logging.info("%s accuracy: %.4f", name, metrics["accuracy"])
    except Exception as exc:
        logging.exception("%s failed", name)
        results[name] = {"error": repr(exc)}


def main() -> None:
    setup_logging()
    args = parse_args()

    models_cfg = load_yaml(args.models_config)
    teacher_ids = teacher_order_from_config(models_cfg)
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
