"""Teacher-only feature learning; student labels are used only for test metrics."""

from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from teacher_attr.evaluation import fit_probe


def teacher_examples(rows, teachers):
    """Pairs repeat teacher references once per student: collapse those repetitions."""
    unique = {}
    for row in rows:
        pid = row["prompt_id"]
        value = (row["source_dataset"], row["teacher_responses"])
        if pid in unique and unique[pid] != value:
            raise ValueError(f"Conflicting teacher references for {pid}")
        if set(value[1]) != set(teachers):
            raise ValueError("Teacher reference set changed")
        unique[pid] = value
    return [
        {"prompt_id": pid, "source_dataset": dataset, "response": responses[t], "label": j}
        for pid, (dataset, responses) in sorted(unique.items())
        for j, t in enumerate(teachers)
    ]


def pos_documents(nlp, texts):
    # Penn-style fine-grained tags, with windows confined to sentence boundaries.
    documents = []
    for doc in nlp.pipe(texts, batch_size=64):
        patterns = []
        for sentence in doc.sents:
            tags = [token.tag_ for token in sentence if not token.is_space]
            if any(not tag for tag in tags):
                raise ValueError("PoS model did not supply fine-grained tags")
            patterns.extend(" ".join(tags[i : i + 4]) for i in range(max(0, len(tags) - 3)))
        documents.append(Counter(patterns))
    return documents


def pos_vocabulary(documents, top_k=50):
    """Most frequent four-tag patterns across all teacher training responses."""
    counts = Counter()
    for doc in documents:
        counts.update(doc)
    vocabulary = [p for p, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:top_k]]
    if not vocabulary:
        raise ValueError("No four-tag patterns in teacher training responses")
    return sorted(vocabulary)


def indicator_matrix(documents, vocabulary):
    return np.asarray([[float(pattern in doc) for pattern in vocabulary] for doc in documents])


def classifier_scores(splits, teachers, cfg, method, scope, nlp=None):
    pools = {s: teacher_examples(splits[s], teachers) for s in ("train", "val")}
    test = splits["test"]
    datasets = sorted({r["source_dataset"] for r in test}) if scope == "per_dataset" else ["all"]
    scores = np.empty((len(test), len(teachers)))
    selected = {}
    for dataset in datasets:
        train_val = {
            s: [r for r in rows if dataset == "all" or r["source_dataset"] == dataset]
            for s, rows in pools.items()
        }
        indexes = [
            i for i, r in enumerate(test) if dataset == "all" or r["source_dataset"] == dataset
        ]
        if any(not rows for rows in train_val.values()):
            raise ValueError(f"No teacher training/validation examples for {dataset}")
        texts = {s: [r["response"] for r in rows] for s, rows in train_val.items()}
        texts["test"] = [test[i]["student_response"] for i in indexes]
        labels = {s: np.array([r["label"] for r in rows]) for s, rows in train_val.items()}
        metadata = {"training_rows": len(texts["train"]), "validation_rows": len(texts["val"])}
        if method == "word_ngram":
            vectorizer = CountVectorizer(
                ngram_range=(1, 4), token_pattern=r"(?u)\b\w+\b|[^\w\s]", max_features=200000
            )
            features = {"train": vectorizer.fit_transform(texts["train"])}
            features.update({s: vectorizer.transform(texts[s]) for s in ("val", "test")})
            metadata["features"] = "word 1-4 gram counts; teacher-training vocabulary only"
        else:
            documents = {s: pos_documents(nlp, values) for s, values in texts.items()}
            vocabulary = pos_vocabulary(documents["train"])
            features = {s: indicator_matrix(values, vocabulary) for s, values in documents.items()}
            metadata.update(vocabulary=vocabulary, features="binary four-tag template indicators")
        # fit_probe consumes labels for train/val only; no student labels supplied.
        prediction, choice = fit_probe(features, labels, cfg["evaluation"]["logreg_c"])
        scores[indexes] = prediction
        selected[dataset] = {**choice, **metadata}
    return scores, {"scope": scope, "selection_data": "teacher validation only", "groups": selected}


def bow_scores(splits, teachers):
    training = teacher_examples(splits["train"], teachers)
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b|[^\w\s]")
    vectorizer.fit([r["response"] for r in training])
    test = splits["test"]
    student = normalize(vectorizer.transform([r["student_response"] for r in test]))
    return np.column_stack(
        [
            np.asarray(
                student.multiply(
                    normalize(vectorizer.transform([r["teacher_responses"][t] for r in test]))
                ).sum(axis=1)
            ).ravel()
            for t in teachers
        ]
    )
