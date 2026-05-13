from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from teacher_attr.datasets import format_prompt_response


def _candidate_texts(row: dict[str, Any], teacher_ids: list[str]) -> list[str]:
    return [
        format_prompt_response(row["prompt"], row["teacher_responses"][teacher_id])
        for teacher_id in teacher_ids
    ]


def _student_text(row: dict[str, Any]) -> str:
    return format_prompt_response(row["prompt"], row["student_response"])



def _candidate_responses(row: dict[str, Any], teacher_ids: list[str]) -> list[str]:
    return [row["teacher_responses"][teacher_id] for teacher_id in teacher_ids]


def _student_response(row: dict[str, Any]) -> str:
    return row["student_response"]


def bow_similarity_scores(rows: list[dict[str, Any]], teacher_ids: list[str]) -> np.ndarray:
    """WTYT-style output similarity: BoW cosine between student and candidate teachers."""
    all_scores: list[np.ndarray] = []
    for row in rows:
        docs = [_student_response(row), *_candidate_responses(row, teacher_ids)]
        mat = CountVectorizer(ngram_range=(1, 1), min_df=1).fit_transform(docs)
        all_scores.append(cosine_similarity(mat[0], mat[1:]).ravel())
    return np.vstack(all_scores)


def bertscore_scores(
    rows: list[dict[str, Any]],
    teacher_ids: list[str],
    model_type: str = "roberta-large",
    batch_size: int = 16,
) -> np.ndarray:
    """WTYT-style BERTScore F1 between student output and each candidate teacher output."""
    import evaluate

    metric = evaluate.load("bertscore")
    student_outputs = [_student_response(row) for row in rows]
    columns = []
    for teacher_id in teacher_ids:
        teacher_outputs = [row["teacher_responses"][teacher_id] for row in rows]
        result = metric.compute(
            predictions=student_outputs,
            references=teacher_outputs,
            model_type=model_type,
            lang="en",
            batch_size=batch_size,
            rescale_with_baseline=True,
        )
        columns.append(np.asarray(result["f1"], dtype=np.float32))
    return np.stack(columns, axis=1)


def bow_classifier_scores(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    teacher_ids: list[str],
) -> np.ndarray:
    return _student_output_classifier_scores(
        train_rows,
        test_rows,
        teacher_ids,
        vectorizer=CountVectorizer(ngram_range=(1, 1), min_df=1, max_features=50000),
    )


def ngram_classifier_scores(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    teacher_ids: list[str],
) -> np.ndarray:
    return _student_output_classifier_scores(
        train_rows,
        test_rows,
        teacher_ids,
        vectorizer=CountVectorizer(ngram_range=(1, 4), min_df=2, max_features=200000),
    )


def _student_output_classifier_scores(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    teacher_ids: list[str],
    vectorizer: CountVectorizer,
) -> np.ndarray:
    label_lookup = {teacher_id: idx for idx, teacher_id in enumerate(teacher_ids)}
    x_train = vectorizer.fit_transform([_student_response(row) for row in train_rows])
    y_train = np.asarray([label_lookup[row["true_teacher"]] for row in train_rows])
    x_test = vectorizer.transform([_student_response(row) for row in test_rows])

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(x_train, y_train)

    proba = np.zeros((len(test_rows), len(teacher_ids)), dtype=np.float32)
    observed_classes = list(clf.classes_)
    observed_proba = clf.predict_proba(x_test)
    for column, class_idx in enumerate(observed_classes):
        proba[:, int(class_idx)] = observed_proba[:, column]
    return proba


def tfidf_scores(rows: list[dict[str, Any]], teacher_ids: list[str]) -> np.ndarray:
    all_scores: list[np.ndarray] = []
    for row in rows:
        docs = [_student_text(row), *_candidate_texts(row, teacher_ids)]
        mat = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(docs)
        all_scores.append(cosine_similarity(mat[0], mat[1:]).ravel())
    return np.vstack(all_scores)


def _shape_template(text: str) -> str:
    pieces = []
    for token in text.split():
        if token.isdigit():
            pieces.append("NUM")
        elif token.isupper():
            pieces.append("UPPER")
        elif token.istitle():
            pieces.append("TITLE")
        elif token.endswith("."):
            pieces.append("WORD.")
        elif token.endswith(","):
            pieces.append("WORD,")
        else:
            pieces.append("WORD")
    return " ".join(pieces)


def _pos_template(text: str, nlp: Any | None = None) -> str:
    if nlp is None:
        return _shape_template(text)
    return " ".join(token.pos_ for token in nlp(text))


def load_spacy_model(model_name: str = "en_core_web_sm") -> Any | None:
    try:
        import spacy

        return spacy.load(model_name, disable=["ner", "lemmatizer"])
    except Exception:
        return None


def pos_template_scores(
    rows: list[dict[str, Any]],
    teacher_ids: list[str],
    nlp: Any | None = None,
) -> np.ndarray:
    all_scores = []
    for row in rows:
        student_template = _pos_template(row["student_response"], nlp=nlp)
        scores = []
        for teacher_id in teacher_ids:
            teacher_template = _pos_template(row["teacher_responses"][teacher_id], nlp=nlp)
            scores.append(SequenceMatcher(None, student_template, teacher_template).ratio())
        all_scores.append(scores)
    return np.asarray(all_scores, dtype=np.float32)


def sentence_embedding_scores(
    rows: list[dict[str, Any]],
    teacher_ids: list[str],
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 64,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    all_scores = []
    for row in rows:
        texts = [_student_text(row), *_candidate_texts(row, teacher_ids)]
        embs = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_scores.append(embs[1:] @ embs[0])
    return np.vstack(all_scores)


def student_only_classifier_scores(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    teacher_ids: list[str],
) -> np.ndarray:
    label_lookup = {teacher_id: idx for idx, teacher_id in enumerate(teacher_ids)}
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000)
    x_train = vectorizer.fit_transform([_student_text(row) for row in train_rows])
    y_train = np.asarray([label_lookup[row["true_teacher"]] for row in train_rows])
    x_test = vectorizer.transform([_student_text(row) for row in test_rows])

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(x_train, y_train)

    proba = np.zeros((len(test_rows), len(teacher_ids)), dtype=np.float32)
    observed_classes = list(clf.classes_)
    observed_proba = clf.predict_proba(x_test)
    for column, class_idx in enumerate(observed_classes):
        proba[:, int(class_idx)] = observed_proba[:, column]
    return proba
