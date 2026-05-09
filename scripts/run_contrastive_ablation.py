from __future__ import annotations

import argparse
import csv as csvlib
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


COMPACT_EXPERIMENTS: list[dict[str, Any]] = [
    {"name": "baseline_temp005_cls02_lr2e5", "temperature": 0.05, "classification_weight": 0.2, "learning_rate": 2e-5},
    {"name": "temp003", "temperature": 0.03, "classification_weight": 0.2, "learning_rate": 2e-5},
    {"name": "temp007", "temperature": 0.07, "classification_weight": 0.2, "learning_rate": 2e-5},
    {"name": "cls010", "temperature": 0.05, "classification_weight": 0.1, "learning_rate": 2e-5},
    {"name": "cls050", "temperature": 0.05, "classification_weight": 0.5, "learning_rate": 2e-5},
    {"name": "lr5e5", "temperature": 0.05, "classification_weight": 0.2, "learning_rate": 5e-5},
    {
        "name": "longer_context_proj256",
        "temperature": 0.05,
        "classification_weight": 0.2,
        "learning_rate": 2e-5,
        "max_length": 768,
        "projection_dim": 256,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sequential contrastive hyperparameter ablations.")
    parser.add_argument("--models_config", default="configs/public_lineage_models.yaml")
    parser.add_argument("--base_config", default="configs/public_lineage_attribution.yaml")
    parser.add_argument("--output_root", default="models/attribution_encoder/public_lineage_ablations")
    parser.add_argument("--summary_dir", default="results/contrastive_ablation")
    parser.add_argument("--mode", choices=["compact", "grid"], default="compact")
    parser.add_argument("--temperatures", default="0.03,0.05,0.07")
    parser.add_argument("--classification_weights", default="0.1,0.2,0.5")
    parser.add_argument("--learning_rates", default="0.00002")
    parser.add_argument("--max_lengths", default="512")
    parser.add_argument("--projection_dims", default="128")
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def csv_values(value: str, cast):
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def safe_value(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def experiment_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.mode == "compact":
        return [dict(item) for item in COMPACT_EXPERIMENTS]

    experiments = []
    for temperature, cls_weight, lr, max_length, projection_dim in itertools.product(
        csv_values(args.temperatures, float),
        csv_values(args.classification_weights, float),
        csv_values(args.learning_rates, float),
        csv_values(args.max_lengths, int),
        csv_values(args.projection_dims, int),
    ):
        experiments.append(
            {
                "name": (
                    f"temp{safe_value(temperature)}_cls{safe_value(cls_weight)}_"
                    f"lr{safe_value(lr)}_len{max_length}_proj{projection_dim}"
                ),
                "temperature": temperature,
                "classification_weight": cls_weight,
                "learning_rate": lr,
                "max_length": max_length,
                "projection_dim": projection_dim,
            }
        )
    return experiments


def write_summary(summary_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / "ablation_summary.json"
    csv_path = summary_dir / "ablation_summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")

    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csvlib.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(args.base_config)
    output_root = Path(args.output_root)
    summary_dir = Path(args.summary_dir)
    config_dir = summary_dir / "generated_configs"

    experiments = experiment_grid(args)
    if args.max_runs is not None:
        experiments = experiments[: args.max_runs]

    rows: list[dict[str, Any]] = []
    for idx, exp in enumerate(experiments, start=1):
        name = exp["name"]
        output_dir = output_root / name
        cfg = dict(base_cfg)
        cfg.update({key: value for key, value in exp.items() if key != "name"})
        cfg["output_dir"] = str(output_dir)
        cfg_path = config_dir / f"{idx:02d}_{name}.yaml"
        write_yaml(cfg_path, cfg)

        metrics_path = output_dir / "training_metrics.json"
        if args.skip_existing and metrics_path.exists():
            print(f"[{idx}/{len(experiments)}] skipping existing {name}", flush=True)
        else:
            cmd = [
                sys.executable,
                "scripts/train_contrastive_encoder.py",
                "--models_config",
                args.models_config,
                "--attribution_config",
                str(cfg_path),
            ]
            print(f"\n[{idx}/{len(experiments)}] {name}", flush=True)
            print("$ " + " ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, check=True)

        row: dict[str, Any] = {"name": name, "output_dir": str(output_dir), **exp}
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            row.update(
                {
                    "best_accuracy": metrics.get("best_accuracy"),
                    "best_epoch": metrics.get("best_epoch"),
                    "best_score": metrics.get("best_score"),
                    "early_stopped": metrics.get("early_stopped"),
                    "stopped_epoch": metrics.get("stopped_epoch"),
                }
            )
        rows.append(row)
        write_summary(summary_dir, rows)

    rows.sort(key=lambda row: row.get("best_accuracy") or -1, reverse=True)
    write_summary(summary_dir, rows)
    print(f"\nWrote {summary_dir / 'ablation_summary.csv'}")
    if rows:
        print("Best run:", rows[0])


if __name__ == "__main__":
    main()