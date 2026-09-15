from __future__ import annotations

import importlib.metadata
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from teacher_attr.config import file_hash, initialize_run, load_config
from teacher_attr.io import save_json, write_jsonl
from teacher_attr.metrics import (
    aggregate_scores,
    classification_metrics,
    cluster_interval,
    grouped_indexes,
    support_sets,
)
from teacher_attr.pairs import load_pairs


def fit_probe(features: dict, labels: dict, candidates: list[float]):
    best, best_acc, best_c = None, -1.0, None
    for c in candidates:
        model = LogisticRegression(C=c, max_iter=3000, class_weight="balanced")
        # Failed convergence is a failed baseline, not a silently reportable result.
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(features["train"], labels["train"])
        accuracy = np.mean(model.predict(features["val"]) == labels["val"])
        if accuracy > best_acc:
            best, best_acc, best_c = model, accuracy, c
    return best.predict_proba(features["test"]), {"C": best_c, "val_accuracy": float(best_acc)}


def lexical_scores(splits: dict, labels: dict, candidates: list[float]):
    methods, selected = {}, {}
    texts = {s: [r["student_response"] for r in rows] for s, rows in splits.items()}
    for name, vectorizer in {
        "word_ngram": CountVectorizer(
            ngram_range=(1, 4),
            token_pattern=r"(?u)\b\w+\b|[^\w\s]",
            min_df=1,
            max_features=200000,
        ),
        "char_ngram": TfidfVectorizer(analyzer="char", ngram_range=(1, 5), max_features=100000),
    }.items():
        features = {"train": vectorizer.fit_transform(texts["train"])}
        features.update({s: vectorizer.transform(texts[s]) for s in ("val", "test")})
        methods[name], selected[name] = fit_probe(features, labels, candidates)
    features = {
        s: np.array(
            [
                [
                    len(t),
                    len(t.split()),
                    t.count("\n"),
                    t.count("."),
                    t.count("?"),
                    t.count(":"),
                    sum(c.isdigit() for c in t),
                ]
                for t in responses
            ],
            dtype=float,
        )
        for s, responses in texts.items()
    }
    scaler = StandardScaler().fit(features["train"])
    methods["length_format"], selected["length_format"] = fit_probe(
        {s: scaler.transform(x) for s, x in features.items()}, labels, candidates
    )
    return methods, selected


def encode_rows(
    model,
    tokenizer,
    rows: list[dict],
    enc: dict,
    teachers: list[str],
    device: str,
    projected: bool,
    include_teachers: bool = True,
):
    import torch

    from teacher_attr.encoders import score_embeddings, to_device, tokenize_pairs

    embeddings, logits, cosine, audits, teacher_embeddings = [], [], [], [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), enc["batch_size"]):
            batch = rows[start : start + enc["batch_size"]]
            tokens, audit = tokenize_pairs(
                tokenizer, [(r["prompt"], r["student_response"]) for r in batch], enc
            )
            emb, cls = model(projected=projected, **to_device(tokens, device))
            teacher_audit = []
            if include_teachers:
                teacher_tokens, teacher_audit = tokenize_pairs(
                    tokenizer,
                    [(r["prompt"], r["teacher_responses"][t]) for r in batch for t in teachers],
                    enc,
                )
                teacher_emb, _ = model(projected=projected, **to_device(teacher_tokens, device))
                teacher_emb = teacher_emb.reshape(len(batch), len(teachers), -1)
                cosine.append(score_embeddings(emb, teacher_emb).cpu().numpy())
                teacher_embeddings.append(teacher_emb.cpu().numpy())
            embeddings.append(emb.cpu().numpy())
            if cls is not None:
                logits.append(cls.cpu().numpy())
            audits.extend(
                {
                    "row_index": start + i,
                    "student": a,
                    "teachers": {
                        t: teacher_audit[i * len(teachers) + j] for j, t in enumerate(teachers)
                    }
                    if include_teachers
                    else {},
                }
                for i, a in enumerate(audit)
            )
    return {
        "embeddings": np.concatenate(embeddings),
        "teacher_embeddings": np.concatenate(teacher_embeddings) if teacher_embeddings else None,
        "cosine": np.concatenate(cosine) if cosine else None,
        "logits": np.concatenate(logits) if logits else None,
        "audit": audits,
    }


def evaluate(
    cfg: dict,
    checkpoint: str | None = None,
    frozen: bool = False,
    name: str = "evaluation",
    reference_config: str | None = None,
) -> dict:
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
        raise ValueError("Evaluation name must be a safe lowercase identifier")
    root = initialize_run(cfg)
    out = root / "evaluations" / name
    if out.exists():
        raise ValueError(f"Evaluation already exists: {out}; choose a new --name")
    out.mkdir(parents=True, exist_ok=True)
    splits = load_pairs(cfg)
    checkpoint_cfg = load_config(reference_config) if reference_config else cfg
    if reference_config:
        from teacher_attr.prompts import audit_splits

        if list(checkpoint_cfg["teachers"]) != list(cfg["teachers"]):
            raise ValueError("Transfer evaluation requires identical candidate teacher order")
        reference = load_pairs(checkpoint_cfg)
        splits = {"train": reference["train"], "val": reference["val"], "test": splits["test"]}
        audit_splits(splits)
    teachers = list(cfg["teachers"])
    labels = {
        s: np.array([teachers.index(r["true_teacher"]) for r in rows]) for s, rows in splits.items()
    }
    ev = cfg["evaluation"]
    methods, selection = lexical_scores(splits, labels, ev["logreg_c"])
    from teacher_attr.baselines import matching_tfidf

    methods["tfidf_matching"] = matching_tfidf(splits, teachers)
    if ev.get("pos_model"):
        from teacher_attr.baselines import pos_scores

        methods["pos"], selection["pos"] = pos_scores(
            splits, labels, ev["pos_model"], ev["logreg_c"]
        )
    if "research" in cfg:
        from teacher_attr.baselines import shortcut_scores

        shortcuts, chosen = shortcut_scores(splits, labels, cfg)
        methods.update(shortcuts)
        selection.update(chosen)
        if ev.get("perplexity_control"):
            from teacher_attr.baselines import perplexity_scores

            methods["perplexity"], selection["perplexity"] = perplexity_scores(splits, labels, cfg)
    if ev.get("generic_encoder"):
        import torch
        from transformers import AutoTokenizer

        from teacher_attr.encoders import AttributionEncoder

        generic = {**cfg["encoder"], **ev["generic_encoder"], "input_mode": "response"}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AttributionEncoder.pretrained(generic, len(teachers)).to(device)
        tokenizer = AutoTokenizer.from_pretrained(generic["hf_name"], revision=generic["revision"])
        encoded = encode_rows(model, tokenizer, splits["test"], generic, teachers, device, False)
        methods["generic_cosine"] = encoded["cosine"]
        np.savez_compressed(
            out / "generic_embeddings.npz",
            student=encoded["embeddings"],
            teachers=encoded["teacher_embeddings"],
        )
        del model, encoded

    encoder_info = {}
    audits = {}
    if frozen or checkpoint:
        import torch
        from transformers import AutoTokenizer

        from teacher_attr.encoders import AttributionEncoder
        from teacher_attr.training import load_checkpoint

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Run trained first, then match the frozen control to its input construction.
        variants = (["trained"] if checkpoint else []) + (["frozen"] if frozen else [])
        enc = cfg["encoder"]
        for variant in variants:
            payload = None
            if variant == "trained":
                model, tokenizer, payload = load_checkpoint(checkpoint, checkpoint_cfg, device)
                enc = payload["encoder"]
            else:
                model = AttributionEncoder.pretrained(enc, len(teachers)).to(device)
                tokenizer = AutoTokenizer.from_pretrained(enc["hf_name"], revision=enc["revision"])
            encoded = {
                s: encode_rows(
                    model,
                    tokenizer,
                    rows,
                    enc,
                    teachers,
                    device,
                    projected=variant == "trained",
                    include_teachers=s == "test",
                )
                for s, rows in splits.items()
            }
            methods[f"{variant}_cosine"] = encoded["test"]["cosine"]
            methods[f"{variant}_probe"], selection[f"{variant}_probe"] = fit_probe(
                {s: x["embeddings"] for s, x in encoded.items()}, labels, ev["logreg_c"]
            )
            if payload and enc["objective"] != "contrastive":
                methods["trained_classifier"] = encoded["test"]["logits"]
            audits[variant] = {s: encoded[s]["audit"] for s in splits}
            encoder_info[variant] = {
                "config": enc,
                "checkpoint": checkpoint if payload else None,
                "resolved_revision": (
                    payload["resolved_revision"]
                    if payload
                    else getattr(model.backbone.config, "_commit_hash", None)
                ),
            }
            if payload and payload["resolved_revision"]:
                enc = {**enc, "revision": payload["resolved_revision"]}
            np.savez_compressed(
                out / f"{variant}_embeddings.npz",
                student=encoded["test"]["embeddings"],
                teachers=encoded["test"]["teacher_embeddings"],
            )
            del model, encoded
    rows, y = splits["test"], labels["test"]
    prompts = [r["prompt_id"] for r in rows]
    supports = support_sets(rows, ev["set_sizes"], ev["set_repeats"], cfg["seed"])
    results = {}
    for method, scores in methods.items():
        result = classification_metrics(scores, y, teachers)
        correct = scores.argmax(1) == y
        result["accuracy_ci95"] = cluster_interval(
            correct, prompts, ev["bootstrap_repeats"], cfg["seed"]
        )
        result["by_dataset"] = {
            ds: classification_metrics(scores[ix], y[ix], teachers)
            for ds, ix in grouped_indexes([r["source_dataset"] for r in rows]).items()
        }
        result["by_student"] = {
            s: classification_metrics(scores[ix], y[ix], teachers)
            for s, ix in grouped_indexes([r["student_id"] for r in rows]).items()
        }
        reference = methods["word_ngram"].argmax(1) == y
        delta = correct.astype(float) - reference.astype(float)
        result["difference_vs_word_ngram"] = {
            "accuracy": float(delta.mean()),
            "ci95": cluster_interval(delta, prompts, ev["bootstrap_repeats"], cfg["seed"]),
        }
        result["set_level"] = aggregate_scores(
            scores, y, supports, teachers, ev["bootstrap_repeats"], cfg["seed"]
        )
        # Size one always uses all test examples, never 80 random singletons.
        result["set_level"]["all/1"] = {
            **classification_metrics(scores, y, teachers),
            "num_sets": len(rows),
            "accuracy_ci95": result["accuracy_ci95"],
        }
        for ds, value in result["by_dataset"].items():
            result["set_level"][f"{ds}/1"] = {**value, "num_sets": value["num_rows"]}
        results[method] = result
    metadata = {
        "schema_version": 2,
        "teacher_ids": teachers,
        "protocol": cfg["protocol"],
        "reference_config": reference_config,
        "probe_training_run": checkpoint_cfg["run_dir"],
        "num_test_prompts": len(set(prompts)),
        "methods": results,
        "validation_selection": selection,
        "encoders": encoder_info,
        "pair_sha256": {
            s: file_hash(
                (Path(checkpoint_cfg["run_dir"]) if s != "test" else root) / "pairs" / f"{s}.jsonl"
            )
            for s in splits
        },
        "aggregation": "arithmetic mean of per-prompt method scores",
        "interval_scope": "prompt bootstrap conditional on observed student checkpoints",
        "environment": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "scikit-learn", "teacher-attribution")
        },
    }
    if "research" in cfg:
        from teacher_attr.research import provenance

        metadata["environment"] = provenance()
    save_json(out / "metrics.json", metadata)
    write_jsonl(
        out / "predictions.jsonl",
        [
            {
                "row_index": i,
                "prompt_id": r["prompt_id"],
                "original_prompt_id": r.get("original_prompt_id"),
                "student_id": r["student_id"],
                "source_dataset": r["source_dataset"],
                "true_teacher": r["true_teacher"],
                "scores": {m: s[i].tolist() for m, s in methods.items()},
            }
            for i, r in enumerate(rows)
        ],
    )
    write_jsonl(out / "support_sets.jsonl", supports)
    if audits:
        save_json(out / "token_audit.json", audits)
    lines = [
        "# Evaluation",
        "",
        f"Protocol: `{cfg['protocol']}`.",
        "",
        "Intervals condition on observed checkpoints; they do not estimate transfer to new models.",
        "",
        "| Method | Accuracy | 95% prompt CI | Macro AUC |",
        "|---|---:|---|---:|",
    ]
    for method, result in results.items():
        lo, hi = result["accuracy_ci95"]
        lines.append(
            f"| {method} | {result['accuracy']:.2%} | {lo:.2%}–{hi:.2%} | "
            f"{result['roc_auc_ovr_macro']:.4f} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"output": str(out), "methods": {m: r["accuracy"] for m, r in results.items()}}
