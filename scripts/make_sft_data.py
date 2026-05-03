from __future__ import annotations

import argparse
import logging

from teacher_attr.distillation import teacher_output_to_sft_row
from teacher_attr.io import read_jsonl, write_jsonl
from teacher_attr.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert teacher outputs to TRL SFT JSONL.")
    parser.add_argument("--teacher_outputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    rows = []
    for idx, row in enumerate(read_jsonl(args.teacher_outputs)):
        if args.limit is not None and idx >= args.limit:
            break
        rows.append(teacher_output_to_sft_row(row))

    write_jsonl(args.output, rows)
    logging.info("Wrote %d SFT rows to %s", len(rows), args.output)


if __name__ == "__main__":
    main()
