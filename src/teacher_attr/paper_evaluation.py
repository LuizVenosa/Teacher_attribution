"""Paper-inspired teacher-to-student transfer, separate from student-supervised evaluation."""

import csv
import json
from pathlib import Path

import numpy as np

from teacher_attr.config import file_hash, initialize_run
from teacher_attr.io import save_json, write_jsonl
from teacher_attr.metrics import (
    aggregate_scores,
    classification_metrics,
    cluster_interval,
    grouped_indexes,
    support_sets,
)
from teacher_attr.pairs import load_pairs
from teacher_attr.paper_scoring import atomic_array
from teacher_attr.research import provenance

METHODS = ("word_ngram", "pos", "bow", "bertscore", "likelihood")


def save_scores(out, name, scores, selection):
    scores = np.asarray(scores)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("Method produced invalid scores")
    path = out / f"{name}.npy"
    atomic_array(path, scores)
    save_json(out / f"{name}.json", {"sha256": file_hash(path), "selection": selection})


def load_scores(out, name, shape):
    path, manifest = out / f"{name}.npy", out / f"{name}.json"
    if not manifest.exists():
        return None
    record = json.loads(manifest.read_text())
    if file_hash(path) != record["sha256"]:
        raise ValueError(f"Stored method scores changed: {name}")
    scores = np.load(path, allow_pickle=False)
    if scores.shape != shape or not np.isfinite(scores).all():
        raise ValueError(f"Invalid stored method scores: {name}")
    audit_hash = record["selection"].get("token_audit_sha256")
    if name == "bertscore" and audit_hash:
        if file_hash(out / "bertscore_token_audit.json") != audit_hash:
            raise ValueError("Stored BERTScore token audit changed")
    return scores, record["selection"]


def summarize(cfg, splits, teachers, methods):
    rows = splits["test"]
    labels = np.array([teachers.index(r["true_teacher"]) for r in rows])
    prompts = [r["prompt_id"] for r in rows]
    ev = cfg["evaluation"]
    supports = support_sets(rows, ev["set_sizes"], ev["set_repeats"], cfg["seed"])
    results = {}
    for name, (scores, selection) in methods.items():
        value = classification_metrics(scores, labels, teachers)
        correct = scores.argmax(1) == labels
        value["accuracy_ci95"] = cluster_interval(
            correct, prompts, ev["bootstrap_repeats"], cfg["seed"]
        )
        value["tied_maximum_rows"] = int(
            np.sum((scores == scores.max(1, keepdims=True)).sum(1) > 1)
        )
        value["by_dataset"] = {
            dataset: classification_metrics(scores[ix], labels[ix], teachers)
            for dataset, ix in grouped_indexes([r["source_dataset"] for r in rows]).items()
        }
        value["by_student"] = {
            student: classification_metrics(scores[ix], labels[ix], teachers)
            for student, ix in grouped_indexes([r["student_id"] for r in rows]).items()
        }
        value["set_level"] = aggregate_scores(
            scores, labels, supports, teachers, ev["bootstrap_repeats"], cfg["seed"]
        )
        value["selection"] = selection
        results[name] = value
    return results, supports


def compare_primary(path, pair_hashes, rows, teachers, methods, cfg):
    directory = Path(path)
    report = json.loads((directory / "metrics.json").read_text())
    if report["pair_sha256"] != pair_hashes or report["teacher_ids"] != teachers:
        raise ValueError("Primary evaluation uses different pairs or teacher ordering")
    from teacher_attr.io import load_jsonl

    predictions = load_jsonl(directory / "predictions.jsonl")
    expected = [(r["prompt_id"], r["student_id"], r["true_teacher"]) for r in rows]
    if [(r["prompt_id"], r["student_id"], r["true_teacher"]) for r in predictions] != expected:
        raise ValueError("Primary prediction rows are not aligned")
    labels = np.array([teachers.index(r["true_teacher"]) for r in rows])
    result = {}
    for name, (scores, _) in methods.items():
        result[name] = {}
        for primary in (
            "word_ngram",
            "pos",
            "trained_probe",
            "trained_classifier",
            "trained_cosine",
        ):
            if primary not in report["methods"]:
                continue
            reference = np.array([r["scores"][primary] for r in predictions])
            if reference.shape != scores.shape or not np.isfinite(reference).all():
                raise ValueError(f"Invalid primary prediction scores: {primary}")
            delta = (scores.argmax(1) == labels).astype(float) - (reference.argmax(1) == labels)
            result[name][primary] = {
                "accuracy_difference": float(delta.mean()),
                "ci95": cluster_interval(
                    delta,
                    [r["prompt_id"] for r in rows],
                    cfg["evaluation"]["bootstrap_repeats"],
                    cfg["seed"],
                ),
            }
    return {
        "primary_metrics_sha256": file_hash(directory / "metrics.json"),
        "primary_predictions_sha256": file_hash(directory / "predictions.jsonl"),
        "interpretation": (
            "Descriptive comparison: primary classifiers had labeled student training data; "
            "teacher-only classifiers did not. These are different information-access protocols."
        ),
        "differences": result,
    }


def evaluate_paper(
    cfg,
    methods=("word_ngram", "pos", "bow"),
    name="paper_v1",
    scope="per_dataset",
    teacher=None,
    collect=False,
    primary=None,
    bertscore_overflow="error",
):
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
        raise ValueError("Evaluation name must be a safe lowercase identifier")
    if scope not in ("pooled", "per_dataset") or not set(methods) <= set(METHODS):
        raise ValueError("Unknown scope or method")
    if bertscore_overflow not in ("error", "truncate"):
        raise ValueError("Unknown BERTScore overflow policy")
    if "likelihood" in methods and not collect and teacher not in cfg["teachers"]:
        raise ValueError("Select one candidate with --teacher for a likelihood job")
    root = initialize_run(cfg)
    splits = load_pairs(cfg)
    teachers = list(cfg["teachers"])
    hashes = {s: file_hash(root / "pairs" / f"{s}.jsonl") for s in splits}
    out = root / "evaluations" / name
    spec = {
        "version": 1,
        "pair_sha256": hashes,
        "teacher_ids": teachers,
        "teacher_models": cfg["teachers"],
        "scope": scope,
        "logreg_c": cfg["evaluation"]["logreg_c"],
        "pos_model": cfg["evaluation"].get("pos_model"),
        "bertscore_model": cfg["evaluation"]["bertscore_model"],
        "bertscore_layers": cfg["evaluation"]["bertscore_layers"],
        "protocol": "teacher_train_teacher_validation_student_test",
    }
    manifest = out / "paper_spec.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != spec:
            raise ValueError("Paper evaluation specification changed; choose a new --name")
    elif out.exists():
        raise ValueError("Directory belongs to another evaluation; choose a new --name")
    else:
        save_json(manifest, spec)
    shape = (len(splits["test"]), len(teachers))
    if not collect:
        from teacher_attr.paper_baselines import bow_scores, classifier_scores
        from teacher_attr.paper_scoring import bertscore_matching, teacher_likelihood

        for method in methods:
            if method == "likelihood":
                teacher_likelihood(cfg, splits["test"], teacher, out)
                continue
            cached = load_scores(out, method, shape)
            if cached is not None:
                if method == "bertscore" and (
                    cached[1].get("overflow_policy", "error") != bertscore_overflow
                ):
                    raise ValueError(
                        "Cached BERTScore uses another overflow policy; choose a new --name"
                    )
                print(f"Using verified cached method: {method}", flush=True)
                continue
            print(f"Running paper method: {method}", flush=True)
            if method in ("word_ngram", "pos"):
                nlp = None
                if method == "pos":
                    import spacy

                    nlp = spacy.load(spec["pos_model"], disable=["parser", "ner"])
                    if "sentencizer" not in nlp.pipe_names:
                        nlp.add_pipe("sentencizer")
                scores, selection = classifier_scores(splits, teachers, cfg, method, scope, nlp)
                if nlp is not None:
                    selection["tagger"] = {k: nlp.meta.get(k) for k in ("name", "version", "lang")}
            elif method == "bow":
                scores = bow_scores(splits, teachers)
                selection = {
                    "features": "unigram counts cosine; vocabulary fit on teacher training only"
                }
            else:
                scores, selection = bertscore_matching(
                    cfg, splits["test"], teachers, out, overflow=bertscore_overflow
                )
            save_scores(out, method, scores, selection)

    all_methods = {}
    for method in METHODS[:-1]:
        value = load_scores(out, method, shape)
        if value is not None:
            all_methods[method] = value
    likelihoods, likelihood_records, pending = [], {}, []
    from teacher_attr.paper_scoring import likelihood_spec

    for candidate in teachers:
        meta = out / f"likelihood_{candidate}.json"
        record = json.loads(meta.read_text()) if meta.exists() else None
        if not record or not record.get("complete"):
            pending.append(candidate)
            continue
        if record["spec"] != likelihood_spec(cfg, splits["test"], candidate):
            raise ValueError("Likelihood specification changed")
        path = out / f"likelihood_{candidate}.npy"
        if file_hash(path) != record["sha256"]:
            raise ValueError("Likelihood scores changed after completion")
        values = np.load(path, allow_pickle=False)
        if values.shape != (shape[0], 2) or not np.isfinite(values).all():
            raise ValueError("Invalid likelihood values")
        likelihoods.append(values)
        likelihood_records[candidate] = record
    if not pending:
        for column, method in enumerate(("teacher_likelihood_token", "teacher_likelihood_byte")):
            all_methods[method] = (
                np.column_stack([v[:, column] for v in likelihoods]),
                {
                    "models": likelihood_records,
                    "warning": (
                        "Raw likelihoods across tokenizers are not calibrated; byte normalization "
                        "is reported as a separate sensitivity analysis."
                    ),
                },
            )
    results, supports = summarize(cfg, splits, teachers, all_methods)
    report = {
        **spec,
        "methods": results,
        "num_test_prompts": len({r["prompt_id"] for r in splits["test"]}),
        "pending_methods": [m for m in METHODS[:-1] if m not in all_methods],
        "pending_likelihood_teachers": pending,
        "environment": provenance(),
        "interval_scope": "Prompt bootstrap conditional on observed student checkpoints",
        "paper": "https://aclanthology.org/2025.findings-acl.173/",
        "reproduction_status": (
            "Paper-inspired matched-data comparison, not an exact reproduction: different "
            "candidate models, four classes rather than five, mixed-task student training, "
            "different instruction/safety data, train-only template selection and "
            "teacher-validation tuning. No claim of beating the published table directly."
        ),
    }
    if primary:
        report["comparison_to_student_supervised"] = compare_primary(
            primary, hashes, splits["test"], teachers, all_methods, cfg
        )
    save_json(out / "metrics.json", report)
    write_jsonl(
        out / "predictions.jsonl",
        [
            {
                "prompt_id": row["prompt_id"],
                "student_id": row["student_id"],
                "true_teacher": row["true_teacher"],
                "source_dataset": row["source_dataset"],
                "scores": {m: scores[i].tolist() for m, (scores, _) in all_methods.items()},
            }
            for i, row in enumerate(splits["test"])
        ],
    )
    write_jsonl(out / "support_sets.jsonl", supports)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "method",
                "dataset",
                "accuracy",
                "macro_f1",
                "num_rows",
                "overflow_policy",
                "max_tokens",
            ]
        )
        for method, result in results.items():
            selection = result["selection"]
            for dataset, value in [("all", result), *result["by_dataset"].items()]:
                writer.writerow(
                    [
                        method,
                        dataset,
                        value["accuracy"],
                        value["macro_f1"],
                        value["num_rows"],
                        selection.get("overflow_policy", ""),
                        selection.get("max_tokens", ""),
                    ]
                )
    lines = ["| Method | Accuracy | 95% prompt CI |", "|---|---:|---:|"]
    for method, result in results.items():
        lo, hi = result["accuracy_ci95"]
        label = method
        selection = result["selection"]
        if method == "bertscore" and selection.get("overflow_policy") == "truncate":
            label += f" (truncate at {selection['max_tokens']} tokens)"
        lines.append(f"| {label} | {result['accuracy']:.2%} | {lo:.2%}–{hi:.2%} |")
    if "bertscore" in results:
        selection = results["bertscore"]["selection"]
        if selection.get("overflow_policy") == "truncate":
            lines.extend(
                [
                    "",
                    f"BERTScore: {selection['overflow_text_occurrences']} text occurrences "
                    f"in {selection['affected_test_rows']} test rows were right-truncated. "
                    "Full responses and the test set are unchanged.",
                ]
            )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "output": str(out),
        "methods": {m: r["accuracy"] for m, r in results.items()},
        "pending_methods": report["pending_methods"],
        "pending_likelihood_teachers": pending,
    }
