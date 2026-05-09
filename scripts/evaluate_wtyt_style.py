from __future__ import annotations

import argparse
import csv as csvlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from teacher_attr.baselines import load_spacy_model
from teacher_attr.io import load_jsonl, load_yaml, save_json
from teacher_attr.metrics import classification_metrics
from teacher_attr.utils import setup_logging, teacher_order_from_config


KNOWN_SOURCE_DATASETS = [
    "cnn_dailymail",
    "sumpubmed",
    "rotten_tomatoes",
    "commonsenseqa",
    "openbookqa",
    "quarel",
    "alpaca",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WTYT-style student-output attribution baselines by dataset/support."
    )
    parser.add_argument("--models_config", required=True)
    parser.add_argument("--train_pairs", required=True)
    parser.add_argument("--test_pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--support_sizes", default="50,200,1000,2000")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max_features_bow", type=int, default=50000)
    parser.add_argument("--max_features_ngram", type=int, default=100000)
    parser.add_argument("--max_features_pos", type=int, default=10000)
    return parser.parse_args()


def labels_for(rows: list[dict[str, Any]], teacher_ids: list[str]) -> np.ndarray:
    label_lookup = {teacher_id: idx for idx, teacher_id in enumerate(teacher_ids)}
    return np.asarray([label_lookup[row["true_teacher"]] for row in rows], dtype=np.int64)


def student_text(row: dict[str, Any]) -> str:
    return row.get("student_response", "") or ""


def source_dataset(row: dict[str, Any]) -> str:
    if row.get("source_dataset"):
        return row["source_dataset"]
    prompt_id = row.get("prompt_id", "")
    for name in sorted(KNOWN_SOURCE_DATASETS, key=len, reverse=True):
        if f"_{name}_" in prompt_id:
            return name
    return row.get("task") or "unknown"


def group_by_dataset(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[source_dataset(row)].append(row)
    return dict(grouped)


def fit_predict_scores(
    train_texts: list[str],
    train_labels: np.ndarray,
    test_texts: list[str],
    *,
    method: str,
    nlp: Any | None,
    max_features: int,
    num_classes: int,
) -> np.ndarray:
    if method == "bow":
        vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=1, max_features=max_features)
        x_train = vectorizer.fit_transform(train_texts)
        x_test = vectorizer.transform(test_texts)
    elif method == "ngram_1_4":
        vectorizer = CountVectorizer(ngram_range=(1, 4), min_df=1, max_features=max_features)
        x_train = vectorizer.fit_transform(train_texts)
        x_test = vectorizer.transform(test_texts)
    elif method == "pos_template_4gram":
        vectorizer = CountVectorizer(
            ngram_range=(4, 4),
            min_df=1,
            max_features=max_features,
            token_pattern=r"(?u)\b\w+\b",
        )
        x_train = vectorizer.fit_transform([pos_sequence(text, nlp) for text in train_texts])
        x_test = vectorizer.transform([pos_sequence(text, nlp) for text in test_texts])
    else:
        raise ValueError(f"Unknown method: {method}")

    clf = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=1)
    clf.fit(x_train, train_labels)
    observed = clf.predict_proba(x_test)
    scores = np.zeros((len(test_texts), num_classes), dtype=np.float32)
    for col, class_idx in enumerate(clf.classes_):
        scores[:, int(class_idx)] = observed[:, col]
    return scores


def pos_sequence(text: str, nlp: Any | None) -> str:
    if nlp is not None:
        return " ".join(token.pos_ for token in nlp(text))
    pieces = []
    for token in text.split():
        if token.isdigit():
            pieces.append("NUM")
        elif token.isupper():
            pieces.append("UPPER")
        elif token.istitle():
            pieces.append("TITLE")
        elif token.endswith("."):
            pieces.append("WORD_PERIOD")
        elif token.endswith(","):
            pieces.append("WORD_COMMA")
        else:
            pieces.append("WORD")
    return " ".join(pieces)


def balanced_support_indexes(
    labels: np.ndarray,
    support_size: int,
    num_classes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    by_class = [np.where(labels == class_idx)[0] for class_idx in range(num_classes)]
    min_count = min(len(indexes) for indexes in by_class)
    if min_count == 0:
        return np.asarray([], dtype=np.int64)
    per_class = max(1, support_size // num_classes)
    per_class = min(per_class, min_count)
    selected = [rng.choice(indexes, size=per_class, replace=False) for indexes in by_class]
    out = np.concatenate(selected)
    rng.shuffle(out)
    return out.astype(np.int64)


def flatten_summary(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset, dataset_payload in payload["datasets"].items():
        for method, method_payload in dataset_payload["methods"].items():
            for support, support_payload in method_payload["support"].items():
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "support_size_requested": int(support),
                        "support_size_actual_mean": support_payload["actual_support_mean"],
                        "accuracy_mean": support_payload["accuracy_mean"],
                        "accuracy_std": support_payload["accuracy_std"],
                        "top2_accuracy_mean": support_payload["top2_accuracy_mean"],
                        "roc_auc_ovr_macro_mean": support_payload.get("roc_auc_ovr_macro_mean"),
                        "repeats": support_payload["repeats"],
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csvlib.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    setup_logging()
    args = parse_args()
    models_cfg = load_yaml(args.models_config)
    teacher_ids = teacher_order_from_config(models_cfg)
    num_classes = len(teacher_ids)
    support_sizes = [int(part) for part in args.support_sizes.split(",") if part.strip()]

    train_rows = load_jsonl(args.train_pairs)
    test_rows = load_jsonl(args.test_pairs)
    train_by_dataset = group_by_dataset(train_rows)
    test_by_dataset = group_by_dataset(test_rows)

    selected_datasets = (
        [part.strip() for part in args.datasets.split(",") if part.strip()]
        if args.datasets
        else sorted(set(train_by_dataset) & set(test_by_dataset))
    )
    nlp = load_spacy_model()
    methods = {
        "bow": args.max_features_bow,
        "ngram_1_4": args.max_features_ngram,
        "pos_template_4gram": args.max_features_pos,
    }

    payload: dict[str, Any] = {
        "teacher_ids": teacher_ids,
        "support_sizes": support_sizes,
        "support_definition": "balanced total test examples per dataset; train uses all rows for that dataset",
        "num_teachers": num_classes,
        "random_accuracy": 1.0 / num_classes,
        "pos_backend": "spacy" if nlp is not None else "shape_fallback",
        "datasets": {},
    }

    for dataset in selected_datasets:
        dataset_train = train_by_dataset.get(dataset, [])
        dataset_test = test_by_dataset.get(dataset, [])
        if not dataset_train or not dataset_test:
            continue
        train_labels = labels_for(dataset_train, teacher_ids)
        test_labels = labels_for(dataset_test, teacher_ids)
        train_texts = [student_text(row) for row in dataset_train]
        test_texts = [student_text(row) for row in dataset_test]

        dataset_payload: dict[str, Any] = {
            "num_train_rows": len(dataset_train),
            "num_test_rows": len(dataset_test),
            "methods": {},
        }
        for method, max_features in methods.items():
            scores = fit_predict_scores(
                train_texts,
                train_labels,
                test_texts,
                method=method,
                nlp=nlp,
                max_features=max_features,
                num_classes=num_classes,
            )
            support_payload = {}
            for support_size in support_sizes:
                metrics_for_support = []
                rng = np.random.default_rng(args.seed + support_size * 997)
                for _ in range(args.repeats):
                    idx = balanced_support_indexes(test_labels, support_size, num_classes, rng)
                    if len(idx) == 0:
                        continue
                    metrics_for_support.append(
                        {
                            "actual_support": int(len(idx)),
                            **classification_metrics(scores[idx], test_labels[idx], teacher_ids),
                        }
                    )
                if metrics_for_support:
                    support_payload[str(support_size)] = {
                        "repeats": len(metrics_for_support),
                        "actual_support_mean": float(
                            np.mean([m["actual_support"] for m in metrics_for_support])
                        ),
                        "accuracy_mean": float(
                            np.mean([m["accuracy"] for m in metrics_for_support])
                        ),
                        "accuracy_std": float(
                            np.std([m["accuracy"] for m in metrics_for_support])
                        ),
                        "top2_accuracy_mean": float(
                            np.mean([m["top2_accuracy"] for m in metrics_for_support])
                        ),
                        "roc_auc_ovr_macro_mean": float(
                            np.mean(
                                [
                                    m["roc_auc_ovr_macro"]
                                    for m in metrics_for_support
                                    if m["roc_auc_ovr_macro"] is not None
                                ]
                            )
                        ),
                        "last_confusion_matrix": metrics_for_support[-1]["confusion_matrix"],
                        "last_roc_auc_by_teacher": metrics_for_support[-1]["roc_auc_by_teacher"],
                    }
                else:
                    support_payload[str(support_size)] = {"repeats": 0}
            dataset_payload["methods"][method] = {"support": support_payload}
        payload["datasets"][dataset] = dataset_payload

    output = Path(args.output)
    save_json(output, payload)
    csv_rows = flatten_summary(payload)
    write_csv(output.with_suffix(".csv"), csv_rows)


if __name__ == "__main__":
    main()