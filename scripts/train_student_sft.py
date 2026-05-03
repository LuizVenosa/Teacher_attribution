from __future__ import annotations

import argparse
import inspect
import logging

from teacher_attr.io import load_yaml
from teacher_attr.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune one LoRA student from one teacher.")
    parser.add_argument("--teacher_id", required=True)
    parser.add_argument("--models_config", required=True)
    parser.add_argument("--distill_config", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    models_cfg = load_yaml(args.models_config)
    distill_cfg = load_yaml(args.distill_config)
    student_name = models_cfg["student_base"]["hf_name"]

    dataset = load_dataset("json", data_files=args.train_file, split="train")
    tokenizer = AutoTokenizer.from_pretrained(student_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        student_name,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto",
    )

    lora_cfg = distill_cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type="CAUSAL_LM",
        target_modules=lora_cfg["target_modules"],
    )

    sft_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": distill_cfg["num_train_epochs"],
        "per_device_train_batch_size": distill_cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": distill_cfg["gradient_accumulation_steps"],
        "learning_rate": distill_cfg["learning_rate"],
        "warmup_ratio": distill_cfg["warmup_ratio"],
        "logging_steps": distill_cfg["logging_steps"],
        "save_steps": distill_cfg["save_steps"],
        "bf16": distill_cfg["bf16"],
        "packing": distill_cfg["packing"],
        "report_to": distill_cfg["report_to"],
    }
    sft_params = inspect.signature(SFTConfig).parameters
    if "max_seq_length" in sft_params:
        sft_kwargs["max_seq_length"] = distill_cfg["max_seq_length"]
    elif "max_length" in sft_params:
        sft_kwargs["max_length"] = distill_cfg["max_seq_length"]
    training_args = SFTConfig(**sft_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "peft_config": peft_config,
    }
    trainer_params = inspect.signature(SFTTrainer).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)

    logging.info("Training student_from_%s on %d rows", args.teacher_id, len(dataset))
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
