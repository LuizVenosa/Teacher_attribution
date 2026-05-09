from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print confusion matrices and ROC AUC from metrics JSON.")
    parser.add_argument("path")
    return parser.parse_args()


def print_matrix(matrix: list[list[int]], labels: list[str]) -> None:
    width = max(8, max(len(label) for label in labels) + 2)
    print(" " * width + "".join(label[: width - 1].rjust(width) for label in labels))
    for label, row in zip(labels, matrix, strict=True):
        print(label[: width - 1].ljust(width) + "".join(str(value).rjust(width) for value in row))


def print_block(name: str, metrics: dict[str, Any], labels: list[str]) -> None:
    if "accuracy" in metrics:
        print(f"\n## {name}")
        print(f"accuracy: {metrics.get('accuracy')}")
        print(f"top2_accuracy: {metrics.get('top2_accuracy')}")
        for key in ("roc_auc_ovr_macro", "roc_auc_ovr_weighted", "roc_auc_ovr_micro"):
            if key in metrics:
                print(f"{key}: {metrics.get(key)}")
        if "roc_auc_by_teacher" in metrics:
            print("roc_auc_by_teacher:")
            for teacher, value in metrics["roc_auc_by_teacher"].items():
                print(f"  {teacher}: {value}")
        if "confusion_matrix" in metrics:
            print("confusion_matrix rows=true labels, columns=predictions:")
            print_matrix(metrics["confusion_matrix"], labels)


def walk(name: str, obj: Any, labels: list[str]) -> None:
    if isinstance(obj, dict):
        print_block(name, obj, obj.get("teacher_ids", labels))
        for key, value in obj.items():
            if isinstance(value, dict):
                walk(f"{name}.{key}" if name else key, value, obj.get("teacher_ids", labels))


def main() -> None:
    args = parse_args()
    with Path(args.path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    labels = payload.get("teacher_ids", [])
    walk("", payload, labels)


if __name__ == "__main__":
    main()