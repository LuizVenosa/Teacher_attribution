from __future__ import annotations

import argparse
import logging
from pathlib import Path

from teacher_attr.io import load_jsonl, load_yaml, write_jsonl
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
    parser = argparse.ArgumentParser(description="Align student and teacher outputs into InfoNCE rows.")
    parser.add_argument("--models_config", required=True)
    parser.add_argument("--student_outputs_dir", required=True)
    parser.add_argument("--teacher_outputs_dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow_missing", action="store_true")
    parser.add_argument(
        "--teacher_ids",
        default=None,
        help="Optional space/comma/colon-separated teacher IDs to include.",
    )
    parser.add_argument(
        "--student_ids",
        default=None,
        help="Optional space/comma/colon-separated student IDs to include.",
    )
    return parser.parse_args()


def parse_id_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    normalized = value.replace(",", " ").replace(":", " ")
    return [item for item in normalized.split() if item]


def load_teacher_outputs(
    teacher_outputs_dir: Path,
    split: str,
    teacher_ids: list[str],
) -> dict[str, dict[str, dict]]:
    by_teacher: dict[str, dict[str, dict]] = {}
    for teacher_id in teacher_ids:
        path = teacher_outputs_dir / f"{teacher_id}_{split}.jsonl"
        rows = load_jsonl(path)
        by_teacher[teacher_id] = {row["prompt_id"]: row for row in rows}
        logging.info("Loaded %d teacher rows from %s", len(rows), path)
    return by_teacher


def source_dataset_from_prompt_id(prompt_id: str) -> str | None:
    for name in sorted(KNOWN_SOURCE_DATASETS, key=len, reverse=True):
        if f"_{name}_" in prompt_id:
            return name
    return None


def main() -> None:
    setup_logging()
    args = parse_args()

    models_cfg = load_yaml(args.models_config)
    all_teacher_ids = teacher_order_from_config(models_cfg)
    requested_teacher_ids = parse_id_list(args.teacher_ids)
    if requested_teacher_ids is None:
        teacher_ids = all_teacher_ids
    else:
        unknown = sorted(set(requested_teacher_ids) - set(all_teacher_ids))
        if unknown:
            raise ValueError(f"Unknown teacher IDs in --teacher_ids: {unknown}")
        requested_set = set(requested_teacher_ids)
        teacher_ids = [teacher_id for teacher_id in all_teacher_ids if teacher_id in requested_set]
    label_lookup = {teacher_id: idx for idx, teacher_id in enumerate(teacher_ids)}

    teacher_outputs = load_teacher_outputs(
        Path(args.teacher_outputs_dir),
        args.split,
        teacher_ids,
    )

    requested_student_ids = parse_id_list(args.student_ids)
    student_outputs_dir = Path(args.student_outputs_dir)
    if requested_student_ids is None:
        student_paths = sorted(student_outputs_dir.glob(f"student_from_*_{args.split}.jsonl"))
    else:
        student_paths = [
            student_outputs_dir / f"student_from_{student_id}_{args.split}.jsonl"
            for student_id in requested_student_ids
        ]
    missing_student_paths = [str(path) for path in student_paths if not path.exists()]
    if missing_student_paths:
        raise FileNotFoundError(f"Missing student output files: {missing_student_paths}")
    if not student_paths:
        raise FileNotFoundError(f"No student outputs found for split={args.split}")

    pairs = []
    missing = 0
    for student_path in student_paths:
        for row in load_jsonl(student_path):
            prompt_id = row["prompt_id"]
            teacher_responses = {}
            for teacher_id in teacher_ids:
                teacher_row = teacher_outputs[teacher_id].get(prompt_id)
                if teacher_row is None:
                    missing += 1
                    break
                teacher_responses[teacher_id] = teacher_row["response"]
            else:
                true_teacher = row["true_teacher"]
                if true_teacher not in label_lookup:
                    raise ValueError(
                        f"Student {row.get('student_id')} has true_teacher={true_teacher}, "
                        f"which is not in the selected teachers: {teacher_ids}"
                    )
                pairs.append(
                    {
                        "prompt_id": prompt_id,
                        "task": row.get("task"),
                        "split": row.get("split", args.split),
                        "source_dataset": row.get("source_dataset")
                        or source_dataset_from_prompt_id(prompt_id),
                        "source_hf_dataset": row.get("source_hf_dataset"),
                        "source_split": row.get("source_split"),
                        "source_index": row.get("source_index"),
                        "source_id": row.get("source_id"),
                        "anchor_student_id": row["student_id"],
                        "true_teacher": true_teacher,
                        "prompt": row["prompt"],
                        "student_response": row["response"],
                        "teacher_responses": teacher_responses,
                        "label": label_lookup[true_teacher],
                    }
                )

    if missing and not args.allow_missing:
        raise ValueError(
            f"Missing {missing} teacher responses for split={args.split}. "
            "Regenerate teacher outputs or pass --allow_missing to skip incomplete rows."
        )

    pairs.sort(key=lambda item: (item["anchor_student_id"], item["prompt_id"]))
    write_jsonl(args.output, pairs)
    logging.info("Wrote %d attribution pairs to %s", len(pairs), args.output)


if __name__ == "__main__":
    main()