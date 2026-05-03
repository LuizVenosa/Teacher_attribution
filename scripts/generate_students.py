from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from teacher_attr.generation import (
    batch_iter,
    clean_generation,
    render_chat_prompt,
    take_unseen_prompt_rows,
)
from teacher_attr.io import append_jsonl, existing_ids, load_jsonl, load_yaml, write_jsonl
from teacher_attr.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate outputs from one LoRA student.")
    parser.add_argument("--teacher_id", required=True)
    parser.add_argument("--models_config", required=True)
    parser.add_argument("--generation_config", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    models_cfg = load_yaml(args.models_config)
    gen_cfg = load_yaml(args.generation_config)
    base_name = models_cfg["student_base"]["hf_name"]

    output_path = Path(args.output)
    prompt_rows = load_jsonl(args.prompts)
    if args.resume:
        prompt_rows = take_unseen_prompt_rows(prompt_rows, existing_ids(output_path, "prompt_id"))

    if not prompt_rows:
        logging.info("No unseen prompts for student_from_%s; nothing to do.", args.teacher_id)
        return

    tokenizer_source = args.adapter_dir if Path(args.adapter_dir).exists() else base_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()
    device = next(model.parameters()).device

    batch_size = gen_cfg.get("batch_size", 16)
    rows = []
    for batch_rows in batch_iter(prompt_rows, batch_size):
        rendered = [render_chat_prompt(tokenizer, row["prompt"]) for row in batch_rows]
        inputs = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True)
        input_len = inputs["input_ids"].shape[1]
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=gen_cfg["max_new_tokens"],
                do_sample=gen_cfg["temperature"] > 0,
                temperature=gen_cfg["temperature"],
                top_p=gen_cfg["top_p"],
                pad_token_id=tokenizer.pad_token_id,
            )

        responses = tokenizer.batch_decode(
            generated[:, input_len:],
            skip_special_tokens=True,
        )
        for row, response in zip(batch_rows, responses, strict=True):
            rows.append(
                {
                    "prompt_id": row["prompt_id"],
                    "task": row.get("task"),
                    "split": row.get("split"),
                    "student_id": f"student_from_{args.teacher_id}",
                    "true_teacher": args.teacher_id,
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
