from __future__ import annotations

import argparse
from pathlib import Path

from teacher_attr.io import load_yaml, write_jsonl
from teacher_attr.prompts import (
    WHO_TAUGHT_YOU_THAT_DATASETS,
    PromptBankConfig,
    WTYTPromptConfig,
    make_prompt_rows,
    make_who_taught_you_that_prompt_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create JSONL prompt splits.")
    parser.add_argument("--output_dir", default="data/prompts")
    parser.add_argument("--source", choices=["synthetic", "who_taught_you_that"], default="synthetic")
    parser.add_argument("--dataset_config", default="configs/datasets.yaml")
    parser.add_argument(
        "--datasets",
        default=",".join(WHO_TAUGHT_YOU_THAT_DATASETS),
        help="Comma-separated WTYT datasets to include when --source=who_taught_you_that.",
    )
    parser.add_argument("--distill-size", type=int, default=1000)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--max-input-chars", type=int, default=None)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--include-qa-answer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--allow-missing-datasets", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    dataset_cfg = _load_dataset_config(args.dataset_config)
    split_sizes = {
        "distill": args.distill_size,
        "train": args.train_size,
        "val": args.val_size,
        "test": args.test_size,
    }

    if args.source == "who_taught_you_that":
        selected_datasets = _csv(args.datasets) or dataset_cfg.get("datasets") or list(
            WHO_TAUGHT_YOU_THAT_DATASETS
        )
        max_input_chars = args.max_input_chars or dataset_cfg.get("max_input_chars", 6000)
        include_qa_answer = (
            args.include_qa_answer
            if args.include_qa_answer is not None
            else dataset_cfg.get("include_qa_answer", True)
        )
        allow_missing = args.allow_missing_datasets or dataset_cfg.get("allow_missing_datasets", False)
        cache_dir = args.hf_cache_dir or dataset_cfg.get("hf_cache_dir")
    else:
        selected_datasets = []
        max_input_chars = 6000
        include_qa_answer = True
        allow_missing = False
        cache_dir = None

    for split_idx, (split, size) in enumerate(split_sizes.items()):
        if args.source == "who_taught_you_that":
            rows = make_who_taught_you_that_prompt_rows(
                WTYTPromptConfig(
                    split=split,
                    size=size,
                    seed=args.seed + split_idx * 997,
                    datasets=tuple(selected_datasets),
                    cache_dir=cache_dir,
                    max_input_chars=max_input_chars,
                    include_qa_answer=include_qa_answer,
                    allow_missing_datasets=allow_missing,
                )
            )
        else:
            rows = make_prompt_rows(
                PromptBankConfig(
                    split=split,
                    size=size,
                    seed=args.seed + split_idx * 997,
                )
            )
        output = output_dir / f"prompts_{split}.jsonl"
        write_jsonl(output, rows)
        print(f"wrote {len(rows):,} prompts to {output}")


def _load_dataset_config(path: str) -> dict:
    config_path = Path(path)
    if config_path.exists():
        return load_yaml(config_path)
    return {}


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    main()
