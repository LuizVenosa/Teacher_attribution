from __future__ import annotations

import argparse
import logging
from pathlib import Path

from teacher_attr.generation import clean_generation, render_chat_prompt, take_unseen_prompt_rows
from teacher_attr.io import append_jsonl, existing_ids, load_jsonl, load_yaml, write_jsonl
from teacher_attr.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate teacher outputs with vLLM.")
    parser.add_argument("--teacher_id", required=True)
    parser.add_argument("--models_config", required=True)
    parser.add_argument("--generation_config", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    models_cfg = load_yaml(args.models_config)
    gen_cfg = load_yaml(args.generation_config)

    model_name = models_cfg["teachers"][args.teacher_id]["hf_name"]
    output_path = Path(args.output)
    prompt_rows = load_jsonl(args.prompts)
    if args.resume:
        prompt_rows = take_unseen_prompt_rows(prompt_rows, existing_ids(output_path, "prompt_id"))

    if not prompt_rows:
        logging.info("No unseen prompts for %s; nothing to do.", args.teacher_id)
        return

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=gen_cfg.get("trust_remote_code", True),
    )
    rendered_prompts = [render_chat_prompt(tokenizer, row["prompt"]) for row in prompt_rows]

    tensor_parallel_size = args.tensor_parallel_size or gen_cfg.get("tensor_parallel_size", 1)
    llm = LLM(
        model=model_name,
        dtype=gen_cfg.get("dtype", "bfloat16"),
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=gen_cfg.get("trust_remote_code", True),
    )

    sampling_params = SamplingParams(
        temperature=gen_cfg["temperature"],
        top_p=gen_cfg["top_p"],
        max_tokens=gen_cfg["max_new_tokens"],
    )
    outputs = llm.generate(rendered_prompts, sampling_params)

    rows = []
    for row, out in zip(prompt_rows, outputs, strict=True):
        rows.append(
            {
                "prompt_id": row["prompt_id"],
                "task": row.get("task"),
                "split": row.get("split"),
                "teacher_id": args.teacher_id,
                "teacher_model": model_name,
                "prompt": row["prompt"],
                "response": clean_generation(out.outputs[0].text),
            }
        )

    if args.resume and output_path.exists():
        append_jsonl(output_path, rows)
    else:
        write_jsonl(output_path, rows)
    logging.info("Wrote %d rows to %s", len(rows), output_path)


if __name__ == "__main__":
    main()
