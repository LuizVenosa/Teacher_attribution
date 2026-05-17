from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer

from teacher_attr.datasets import format_prompt_response
from teacher_attr.encoders import AttributionEncoder
from teacher_attr.generation import batch_iter
from teacher_attr.io import load_jsonl, load_yaml, save_json
from teacher_attr.metrics import (
    accuracy_by_task,
    accuracy_from_grouped,
    classification_metrics,
    grouped_classification_metrics,
)
from teacher_attr.utils import get_device, seed_everything, setup_logging, teacher_order_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate contrastive encoder with retrieval and frozen-feature probes.",
    )
    parser.add_argument("--models_config", required=True)
    parser.add_argument("--attribution_config", required=True)
    parser.add_argument("--train_pairs", required=True)
    parser.add_argument("--test_pairs", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--teacher_ids",
        default=None,
        help="Optional space/comma/colon-separated teacher IDs to evaluate.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="Use the slow tokenizer. Useful for DeBERTa/SentencePiece models.",
    )
    parser.add_argument(
        "--skip_similarity_baselines",
        action="store_true",
        help="Only run contrastive-encoder probes, skipping simple cosine baselines.",
    )
    parser.add_argument(
        "--skip_sentence_baseline",
        action="store_true",
        help="Skip the pretrained sentence-transformer cosine baseline.",
    )
    parser.add_argument(
        "--sentence_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer checkpoint for the response-only cosine baseline.",
    )
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


def labels_for(rows: list[dict[str, Any]], teacher_ids: list[str]) -> np.ndarray:
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
    rows: list[dict[str, Any]],
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


def method_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    rows: list[dict[str, Any]],
    teacher_ids: list[str],
) -> dict[str, Any]:
    metrics = classification_metrics(scores, labels, teacher_ids)
    add_breakdowns(metrics, scores, labels, rows, teacher_ids)
    return metrics


def add_method_result(
    results: dict[str, Any],
    name: str,
    score_fn: Any,
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    teacher_ids: list[str],
) -> None:
    try:
        scores = score_fn()
        results[name] = method_metrics(scores, labels, rows, teacher_ids)
        logging.info("%s accuracy: %.4f", name, results[name]["accuracy"])
    except Exception as exc:
        logging.exception("%s failed", name)
        results[name] = {"error": repr(exc)}


def build_student_texts(rows: list[dict[str, Any]]) -> list[str]:
    return [format_prompt_response(row["prompt"], row["student_response"]) for row in rows]


def build_student_responses(rows: list[dict[str, Any]]) -> list[str]:
    return [row["student_response"] for row in rows]


def build_teacher_texts(rows: list[dict[str, Any]], teacher_ids: list[str]) -> list[str]:
    return [
        format_prompt_response(row["prompt"], row["teacher_responses"][teacher_id])
        for row in rows
        for teacher_id in teacher_ids
    ]


def build_teacher_responses(rows: list[dict[str, Any]], teacher_ids: list[str]) -> list[str]:
    return [
        row["teacher_responses"][teacher_id]
        for row in rows
        for teacher_id in teacher_ids
    ]


def load_model(
    checkpoint_path: str | Path,
    model_name: str,
    projection_dim: int,
    num_teachers: int,
    device: torch.device,
) -> AttributionEncoder:
    model = AttributionEncoder(
        model_name=model_name,
        projection_dim=projection_dim,
        num_teachers=num_teachers,
    )
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def encode_students(
    model: AttributionEncoder,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    max_length: int,
    batch_size: int,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    emb_chunks = []
    logit_chunks = []
    total = (len(texts) + batch_size - 1) // batch_size
    for batch in tqdm(batch_iter(texts, batch_size), total=total, desc=desc, dynamic_ncols=True):
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        emb, logits = model(**tokens)
        emb_chunks.append(emb.float().cpu().numpy())
        logit_chunks.append(logits.float().cpu().numpy())
    return np.concatenate(emb_chunks, axis=0), np.concatenate(logit_chunks, axis=0)


@torch.no_grad()
def encode_text_embeddings(
    model: AttributionEncoder,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    max_length: int,
    batch_size: int,
    desc: str,
) -> np.ndarray:
    chunks = []
    total = (len(texts) + batch_size - 1) // batch_size
    for batch in tqdm(batch_iter(texts, batch_size), total=total, desc=desc, dynamic_ncols=True):
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        emb, _ = model(**tokens)
        chunks.append(emb.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def cosine_scores(student_emb: np.ndarray, teacher_emb: np.ndarray) -> np.ndarray:
    student_t = torch.from_numpy(student_emb)
    teacher_t = torch.from_numpy(teacher_emb)
    return torch.einsum(
        "bd,bkd->bk",
        F.normalize(student_t, dim=-1),
        F.normalize(teacher_t, dim=-1),
    ).numpy()


def sparse_rowwise_cosine(
    rows: list[dict[str, Any]],
    teacher_ids: list[str],
    vectorizer_factory: Any,
) -> np.ndarray:
    """Compute same-prompt cosine with a fresh sparse vectorizer per example."""
    all_scores: list[np.ndarray] = []
    for row in rows:
        docs = [
            row["student_response"],
            *[row["teacher_responses"][teacher_id] for teacher_id in teacher_ids],
        ]
        mat = vectorizer_factory().fit_transform(docs)
        all_scores.append(cosine_similarity(mat[0], mat[1:]).ravel())
    return np.vstack(all_scores)


def sentence_embedding_cosine_scores(
    rows: list[dict[str, Any]],
    teacher_ids: list[str],
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    student_emb = model.encode(
        build_student_responses(rows),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    teacher_flat = model.encode(
        build_teacher_responses(rows, teacher_ids),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    teacher_emb = teacher_flat.reshape(len(rows), len(teacher_ids), -1)
    return np.einsum("bd,bkd->bk", student_emb, teacher_emb)


def fit_logreg(x_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(x_train, y_train)
    return clf


def predict_proba_all(
    clf: LogisticRegression,
    x: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    observed = clf.predict_proba(x)
    proba = np.zeros((len(x), num_classes), dtype=np.float32)
    for column, class_idx in enumerate(clf.classes_):
        proba[:, int(class_idx)] = observed[:, column]
    return proba


def set_level_from_rows(
    rows: list[dict[str, Any]],
    labels: np.ndarray,
) -> dict[tuple[str, str], list[int]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[(row["anchor_student_id"], row["true_teacher"])].append(idx)
    return groups


def set_level_scores_from_matrix(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    labels: np.ndarray,
    teacher_ids: list[str],
    set_sizes: list[int],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    groups = set_level_from_rows(rows, labels)
    out: dict[str, Any] = {}
    for set_size in set_sizes:
        score_rows = []
        label_rows = []
        for _ in range(repeats):
            for indexes in groups.values():
                if len(indexes) < set_size:
                    continue
                selected = rng.choice(indexes, size=set_size, replace=False)
                score_rows.append(scores[selected].mean(axis=0))
                label_rows.append(labels[selected[0]])
        if score_rows:
            set_scores = np.vstack(score_rows)
            set_labels = np.asarray(label_rows, dtype=np.int64)
            out[str(set_size)] = classification_metrics(set_scores, set_labels, teacher_ids)
            out[str(set_size)]["num_sets"] = int(len(set_labels))
        else:
            out[str(set_size)] = {"num_sets": 0, "accuracy": None}
    return out


def set_level_cosine_from_embeddings(
    rows: list[dict[str, Any]],
    student_emb: np.ndarray,
    teacher_emb: np.ndarray,
    labels: np.ndarray,
    teacher_ids: list[str],
    set_sizes: list[int],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    groups = set_level_from_rows(rows, labels)
    out: dict[str, Any] = {}
    for set_size in set_sizes:
        score_rows = []
        label_rows = []
        for _ in range(repeats):
            for indexes in groups.values():
                if len(indexes) < set_size:
                    continue
                selected = rng.choice(indexes, size=set_size, replace=False)
                s = student_emb[selected].mean(axis=0)
                s = s / max(np.linalg.norm(s), 1e-12)
                t = teacher_emb[selected].mean(axis=0)
                t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
                score_rows.append(t @ s)
                label_rows.append(labels[selected[0]])
        if score_rows:
            set_scores = np.vstack(score_rows)
            set_labels = np.asarray(label_rows, dtype=np.int64)
            out[str(set_size)] = classification_metrics(set_scores, set_labels, teacher_ids)
            out[str(set_size)]["num_sets"] = int(len(set_labels))
        else:
            out[str(set_size)] = {"num_sets": 0, "accuracy": None}
    return out


def main() -> None:
    setup_logging()
    args = parse_args()

    models_cfg = load_yaml(args.models_config)
    attr_cfg = load_yaml(args.attribution_config)
    seed = int(attr_cfg.get("seed", 13))
    seed_everything(seed)

    teacher_ids = select_teacher_ids(models_cfg, args.teacher_ids)
    model_name = models_cfg["attribution_encoder"]["hf_name"]
    tokenizer_path = Path(args.checkpoint).parent
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=not args.use_slow_tokenizer,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.unk_token

    device = get_device()
    model = load_model(
        args.checkpoint,
        model_name=model_name,
        projection_dim=int(attr_cfg["projection_dim"]),
        num_teachers=len(teacher_ids),
        device=device,
    )

    train_rows = load_jsonl(args.train_pairs)
    test_rows = load_jsonl(args.test_pairs)
    train_labels = labels_for(train_rows, teacher_ids)
    test_labels = labels_for(test_rows, teacher_ids)
    max_length = int(attr_cfg["max_length"])
    batch_size = int(attr_cfg.get("eval_batch_size", 16))

    train_student_emb, train_student_logits = encode_students(
        model,
        tokenizer,
        build_student_texts(train_rows),
        device,
        max_length,
        batch_size,
        desc="train students",
    )
    test_student_emb, test_student_logits = encode_students(
        model,
        tokenizer,
        build_student_texts(test_rows),
        device,
        max_length,
        batch_size,
        desc="test students",
    )

    train_teacher_flat = encode_text_embeddings(
        model,
        tokenizer,
        build_teacher_texts(train_rows, teacher_ids),
        device,
        max_length,
        batch_size,
        desc="train teachers",
    )
    test_teacher_flat = encode_text_embeddings(
        model,
        tokenizer,
        build_teacher_texts(test_rows, teacher_ids),
        device,
        max_length,
        batch_size,
        desc="test teachers",
    )
    train_teacher_emb = train_teacher_flat.reshape(len(train_rows), len(teacher_ids), -1)
    test_teacher_emb = test_teacher_flat.reshape(len(test_rows), len(teacher_ids), -1)

    train_cosine_scores = cosine_scores(train_student_emb, train_teacher_emb)
    test_cosine_scores = cosine_scores(test_student_emb, test_teacher_emb)

    embedding_clf = fit_logreg(train_student_emb, train_labels)
    embedding_logreg_scores = predict_proba_all(
        embedding_clf,
        test_student_emb,
        num_classes=len(teacher_ids),
    )

    score_clf = fit_logreg(train_cosine_scores, train_labels)
    cosine_score_logreg_scores = predict_proba_all(
        score_clf,
        test_cosine_scores,
        num_classes=len(teacher_ids),
    )

    methods = {
        "cosine_retrieval": test_cosine_scores,
        "classifier_head": test_student_logits,
        "embedding_logreg": embedding_logreg_scores,
        "cosine_score_logreg": cosine_score_logreg_scores,
    }
    method_payload = {
        name: method_metrics(scores, test_labels, test_rows, teacher_ids)
        for name, scores in methods.items()
    }
    if not args.skip_similarity_baselines:
        add_method_result(
            method_payload,
            "bow_response_cosine",
            lambda: sparse_rowwise_cosine(
                test_rows,
                teacher_ids,
                lambda: CountVectorizer(ngram_range=(1, 1), min_df=1),
            ),
            test_rows,
            test_labels,
            teacher_ids,
        )
        add_method_result(
            method_payload,
            "tfidf_response_cosine",
            lambda: sparse_rowwise_cosine(
                test_rows,
                teacher_ids,
                lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=1),
            ),
            test_rows,
            test_labels,
            teacher_ids,
        )
        if not args.skip_sentence_baseline:
            add_method_result(
                method_payload,
                "sentence_response_cosine",
                lambda: sentence_embedding_cosine_scores(
                    test_rows,
                    teacher_ids,
                    model_name=args.sentence_model,
                    batch_size=batch_size,
                ),
                test_rows,
                test_labels,
                teacher_ids,
            )

    set_sizes = [int(item) for item in attr_cfg.get("set_sizes", [1, 2, 4, 8, 16, 32, 64])]
    repeats = int(attr_cfg.get("set_repeats", 20))
    set_level = {
        "cosine_retrieval": set_level_cosine_from_embeddings(
            test_rows,
            test_student_emb,
            test_teacher_emb,
            test_labels,
            teacher_ids,
            set_sizes,
            repeats,
            seed,
        ),
        "classifier_head": set_level_scores_from_matrix(
            test_rows,
            test_student_logits,
            test_labels,
            teacher_ids,
            set_sizes,
            repeats,
            seed,
        ),
        "embedding_logreg": set_level_scores_from_matrix(
            test_rows,
            embedding_logreg_scores,
            test_labels,
            teacher_ids,
            set_sizes,
            repeats,
            seed,
        ),
        "cosine_score_logreg": set_level_scores_from_matrix(
            test_rows,
            cosine_score_logreg_scores,
            test_labels,
            teacher_ids,
            set_sizes,
            repeats,
            seed,
        ),
    }

    payload = {
        "run_type": "contrastive_encoder_probe_eval",
        "encoder_model": model_name,
        "models_config_path": args.models_config,
        "attribution_config_path": args.attribution_config,
        "checkpoint": args.checkpoint,
        "train_pairs": args.train_pairs,
        "test_pairs": args.test_pairs,
        "output": args.output,
        "num_train_rows": len(train_rows),
        "num_test_rows": len(test_rows),
        "teacher_ids": teacher_ids,
        "methods": method_payload,
        "set_level": set_level,
        "similarity_baselines": {
            "included": not args.skip_similarity_baselines,
            "sentence_model": None if args.skip_sentence_baseline else args.sentence_model,
            "definition": "response-only same-prompt cosine between the student output and each candidate teacher output",
        },
    }
    save_json(args.output, payload)


if __name__ == "__main__":
    main()
