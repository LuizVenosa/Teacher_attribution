from __future__ import annotations

import argparse
import logging
from pathlib import Path

from teacher_attr.generation import batch_iter, clean_generation, take_unseen_prompt_rows
from teacher_attr.io import append_jsonl, existing_ids, load_jsonl, load_yaml, write_jsonl
from teacher_attr.public_generation import generate_responses, load_text_generation_model, render_prompt
from teacher_attr.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate outputs from public teacher models.")
    parser.add_argument("--teacher_id", required=True)
    parser.add_argument("--models_config", required=True)
    parser.add_argument("--generation_config", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    models_cfg = load_yaml(args.models_config)
    gen_cfg = load_yaml(args.generation_config)
    teacher_cfg = models_cfg["teachers"][args.teacher_id]
    model_name = teacher_cfg["hf_name"]
    prompt_format = teacher_cfg.get("prompt_format", "chat")

    output_path = Path(args.output)
    prompt_rows = load_jsonl(args.prompts)
    if args.resume:
        prompt_rows = take_unseen_prompt_rows(prompt_rows, existing_ids(output_path, "prompt_id"))

    if not prompt_rows:
        logging.info("No unseen prompts for %s; nothing to do.", args.teacher_id)
        return

    model, tokenizer, is_encoder_decoder = load_text_generation_model(
        model_name,
        gen_cfg.get("dtype", "auto"),
        gen_cfg.get("trust_remote_code", True),
    )

    rows = []
    for batch_rows in batch_iter(prompt_rows, gen_cfg.get("batch_size", 8)):
        rendered = [render_prompt(tokenizer, row["prompt"], prompt_format) for row in batch_rows]
        responses = generate_responses(
            model,
            tokenizer,
            rendered,
            is_encoder_decoder=is_encoder_decoder,
            max_new_tokens=gen_cfg["max_new_tokens"],
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
            max_input_tokens=gen_cfg.get("max_input_tokens"),
        )
        for row, response in zip(batch_rows, responses, strict=True):
            rows.append(
                {
                    "prompt_id": row["prompt_id"],
                    "task": row.get("task"),
                    "split": row.get("split"),
                    "source_dataset": row.get("source_dataset"),
                    "source_hf_dataset": row.get("source_hf_dataset"),
                    "source_split": row.get("source_split"),
                    "source_index": row.get("source_index"),
                    "source_id": row.get("source_id"),
                    "teacher_id": args.teacher_id,
                    "teacher_model": model_name,
                    "prompt": row["prompt"],
                    "response": clean_generation(response),
                }
            )

    if args.resume and output_path.exists():
        append_jsonl(output_path, rows)
    else:
        write_jsonl(output_path, rows)
    logging.info("Wrote %d rows to %s", len(rows), output_path)


if __name__ == "__main__":
    main()
