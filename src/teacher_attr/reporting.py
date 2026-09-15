"""Export figures and their numerical data directly from saved evaluation artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def figures(evaluation: str) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path = Path(evaluation)
    path = path / "metrics.json" if path.is_dir() else path
    metrics = json.loads(path.read_text())
    out = path.parent / "figures"
    out.mkdir(exist_ok=True)
    records = []
    for method, result in metrics.get("methods", {}).items():
        records.append(
            {
                "method": method,
                "scope": "all",
                "k": 1,
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "ci_low": result["accuracy_ci95"][0],
                "ci_high": result["accuracy_ci95"][1],
            }
        )
    for method, groups in metrics.get("set_methods", {}).items():
        for key, result in groups.items():
            scope, k = key.rsplit("/", 1)
            records.append(
                {
                    "method": method,
                    "scope": scope,
                    "k": int(k),
                    "accuracy": result["accuracy"],
                    "macro_f1": result["macro_f1"],
                    "ci_low": result["accuracy_ci95"][0],
                    "ci_high": result["accuracy_ci95"][1],
                }
            )
    if not records:
        raise ValueError("No supported evaluation metrics")
    with (out / "figure_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if "set_methods" in metrics:
        for method in metrics["set_methods"]:
            series = sorted(
                [r for r in records if r["scope"] == "all" and r["method"] == method],
                key=lambda r: r["k"],
            )
            x, y = [r["k"] for r in series], [r["accuracy"] for r in series]
            ax.plot(x, y, marker="o", label=method)
            ax.fill_between(
                x, [r["ci_low"] for r in series], [r["ci_high"] for r in series], alpha=0.15
            )
        ax.set(
            xlabel="Observed student responses", ylabel="Teacher attribution accuracy", ylim=(0, 1)
        )
        ax.legend()
    else:
        ax.barh([r["method"] for r in records], [r["accuracy"] for r in records])
        ax.set(xlabel="Teacher attribution accuracy", xlim=(0, 1))
    chance = 1 / len(metrics["teacher_ids"])
    (ax.axhline if "set_methods" in metrics else ax.axvline)(
        chance, color="black", linestyle="--", label="Chance"
    )
    fig.tight_layout()
    fig.savefig(out / "accuracy.pdf")
    fig.savefig(out / "accuracy.png", dpi=300)
    plt.close(fig)
    for method, result in metrics.get("methods", {}).items():
        fig, ax = plt.subplots(figsize=(5, 4))
        matrix = np.array(result["confusion_matrix"])
        im = ax.imshow(matrix, cmap="Blues")
        ax.set(
            xticks=range(len(matrix)),
            yticks=range(len(matrix)),
            xticklabels=metrics["teacher_ids"],
            yticklabels=metrics["teacher_ids"],
            xlabel="Predicted teacher",
            ylabel="True teacher",
            title=method,
        )
        ax.tick_params(axis="x", rotation=45)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out / f"confusion_{method}.pdf")
        plt.close(fig)
    return {"output": str(out), "data": str(out / "figure_data.csv")}


def collect(evaluations: list[str], output: str) -> dict:
    """Tidy amount/decoding/seed/cross-task data without pooling incompatible experiments."""
    out = Path(output)
    if out.exists():
        raise ValueError("Collection output already exists")
    records = []
    for evaluation in evaluations:
        path = Path(evaluation).resolve()
        path = path / "metrics.json" if path.is_dir() else path
        metrics = json.loads(path.read_text())
        experiment = json.loads((path.parents[2] / "experiment.json").read_text())
        research = experiment.get("research", {})
        variant = research.get("variant", {})
        methods = {m: {"all/1": r} for m, r in metrics.get("methods", {}).items()}
        methods.update(metrics.get("set_methods", {}))
        for method, groups in methods.items():
            for key, result in groups.items():
                scope, k = key.rsplit("/", 1)
                if scope != "all":
                    continue
                records.append(
                    {
                        "evaluation": str(path),
                        "method": method,
                        "k": int(k),
                        "amount": research.get("primary_amount"),
                        "student_seed": research.get("primary_seed"),
                        "variant": variant.get("kind", "primary"),
                        "value": variant.get("value", ""),
                        "accuracy": result["accuracy"],
                        "ci_low": result["accuracy_ci95"][0],
                        "ci_high": result["accuracy_ci95"][1],
                    }
                )
    if not records:
        raise ValueError("No evaluations to collect")
    out.mkdir(parents=True)
    with (out / "experiments.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for method in sorted({r["method"] for r in records}):
        rows = [r for r in records if r["method"] == method]
        fig, ax = plt.subplots(figsize=(8, 4))
        for run in sorted({r["evaluation"] for r in rows}):
            series = sorted([r for r in rows if r["evaluation"] == run], key=lambda r: r["k"])
            label = f"{series[0]['variant']}:{series[0]['value']} n={series[0]['amount']}"
            ax.plot(
                [r["k"] for r in series], [r["accuracy"] for r in series], marker="o", label=label
            )
        ax.set(
            xlabel="Observed responses", ylabel="Attribution accuracy", ylim=(0, 1), title=method
        )
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"{method}.pdf")
        plt.close(fig)
    return {"data": str(out / "experiments.csv"), "rows": len(records)}


def compare(reference: str, variant: str, output: str, repeats: int, seed: int) -> dict:
    import numpy as np

    from teacher_attr.io import load_jsonl, save_json
    from teacher_attr.metrics import cluster_interval

    def read(directory):
        rows = load_jsonl(Path(directory) / "predictions.jsonl")
        result = {
            (r.get("original_prompt_id") or r["prompt_id"], r["true_teacher"]): r for r in rows
        }
        if len(result) != len(rows):
            raise ValueError("Paired robustness comparison needs one student per teacher/prompt")
        return result

    original, changed = read(reference), read(variant)
    if original.keys() != changed.keys():
        raise ValueError("Robustness predictions must align on original prompts and teachers")
    old_meta = json.loads((Path(reference) / "metrics.json").read_text())
    new_meta = json.loads((Path(variant) / "metrics.json").read_text())
    if old_meta["teacher_ids"] != new_meta["teacher_ids"]:
        raise ValueError("Candidate teacher order differs")
    methods = set(old_meta["methods"]) & set(new_meta["methods"])
    result = {}
    keys = sorted(original)
    labels = np.array([old_meta["teacher_ids"].index(k[1]) for k in keys])
    for method in sorted(methods):
        before = np.array([np.argmax(original[k]["scores"][method]) for k in keys])
        after = np.array([np.argmax(changed[k]["scores"][method]) for k in keys])
        delta = (after == labels).astype(float) - (before == labels).astype(float)
        result[method] = {
            "prediction_consistency": float(np.mean(before == after)),
            "accuracy_difference": float(delta.mean()),
            "difference_ci95": cluster_interval(delta, [k[0] for k in keys], repeats, seed),
        }
    save_json(output, {"reference": reference, "variant": variant, "methods": result})
    return result
