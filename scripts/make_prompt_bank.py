from __future__ import annotations

import argparse
from pathlib import Path

from teacher_attr.io import write_jsonl
from teacher_attr.prompts import PromptBankConfig, make_prompt_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create JSONL prompt splits.")
    parser.add_argument("--output_dir", default="data/prompts")
    parser.add_argument("--distill-size", type=int, default=1000)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    split_sizes = {
        "distill": args.distill_size,
        "train": args.train_size,
        "val": args.val_size,
        "test": args.test_size,
    }

    for split_idx, (split, size) in enumerate(split_sizes.items()):
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


if __name__ == "__main__":
    main()
