"""Aligned lexical retrieval, a real PoS tagger, and superficial controls."""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


def matching_tfidf(splits: dict, teachers: list[str]) -> np.ndarray:
    train = splits["train"]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b|[^\w\s]")
    vectorizer.fit(
        [r["student_response"] for r in train]
        + [r["teacher_responses"][t] for r in train for t in teachers]
    )
    test = splits["test"]
    student = vectorizer.transform([r["student_response"] for r in test])
    return np.column_stack(
        [
            np.asarray(
                student.multiply(
                    vectorizer.transform([r["teacher_responses"][t] for r in test])
                ).sum(axis=1)
            ).ravel()
            for t in teachers
        ]
    )


def pos_scores(splits: dict, labels: dict, model_name: str, candidates: list[float]):
    import spacy

    from teacher_attr.evaluation import fit_probe

    try:
        nlp = spacy.load(model_name, disable=["ner", "parser"])
    except OSError as exc:
        raise RuntimeError(
            f"Install the configured PoS model: python -m spacy download {model_name}"
        ) from exc
    texts = {}
    for split, rows in splits.items():
        documents = nlp.pipe([r["student_response"] for r in rows], batch_size=64)
        texts[split] = [
            " ".join(token.pos_ for token in doc if not token.is_space) for doc in documents
        ]
        if any(not t.strip() for t in texts[split]):
            raise ValueError("PoS tagger returned empty structural features")
    vectorizer = TfidfVectorizer(ngram_range=(1, 4), lowercase=False, token_pattern=r"\S+")
    features = {"train": vectorizer.fit_transform(texts["train"])}
    features.update({s: vectorizer.transform(texts[s]) for s in ("val", "test")})
    scores, selection = fit_probe(features, labels, candidates)
    return scores, {
        **selection,
        "tagger": nlp.meta,
        "features": "UPOS 1-4 gram TF-IDF; simplified structural probe",
    }


def shortcut_scores(splits: dict, labels: dict, cfg: dict):
    from transformers import AutoTokenizer

    from teacher_attr.evaluation import fit_probe

    base = cfg["research"]["student"]
    tokenizer = AutoTokenizer.from_pretrained(base["hf_name"], revision=base["revision"])
    phrases = cfg["evaluation"]["shortcut_phrases"]
    arrays = {}
    for s, rows in splits.items():
        arrays[s] = np.array(
            [
                [
                    len(tokenizer.encode(r["student_response"], add_special_tokens=False)),
                    sum(c in ".,;:!?" for c in r["student_response"]),
                    *[len(re.findall(re.escape(p), r["student_response"], re.I)) for p in phrases],
                ]
                for r in rows
            ],
            dtype=float,
        )
    methods, selections = {}, {}
    for name, columns in {
        "token_count": [0],
        "punctuation": [1],
        "common_phrases": list(range(2, 2 + len(phrases))),
    }.items():
        scaler = StandardScaler().fit(arrays["train"][:, columns])
        methods[name], selections[name] = fit_probe(
            {s: scaler.transform(x[:, columns]) for s, x in arrays.items()},
            labels,
            cfg["evaluation"]["logreg_c"],
        )
    return methods, selections


def perplexity_scores(splits: dict, labels: dict, cfg: dict):
    import torch
    from torch.nn import functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from teacher_attr.evaluation import fit_probe

    base = cfg["research"]["student"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base["hf_name"], revision=base["revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(base["hf_name"], revision=base["revision"])
        .to(device)
        .eval()
    )
    values = {}
    limit = cfg["research"]["training"]["max_length"]
    with torch.inference_mode():
        for split, rows in splits.items():
            nlls = []
            for row in rows:
                tokens = tokenizer(row["student_response"], return_tensors="pt").to(device)
                if tokens["input_ids"].shape[1] < 2 or tokens["input_ids"].shape[1] > limit:
                    raise ValueError("Perplexity control requires 2..max_length tokens")
                logits = model(**tokens).logits[:, :-1].float()
                target = tokens["input_ids"][:, 1:]
                nlls.append(
                    float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1)))
                )
            values[split] = np.array(nlls)[:, None]
    scaler = StandardScaler().fit(values["train"])
    scores, selection = fit_probe(
        {s: scaler.transform(v) for s, v in values.items()}, labels, cfg["evaluation"]["logreg_c"]
    )
    return scores, {
        **selection,
        "feature": "mean response negative log likelihood (log perplexity)",
        "model": base,
    }
