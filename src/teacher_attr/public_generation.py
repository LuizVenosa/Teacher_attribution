from __future__ import annotations

import logging
import os
from typing import Any

import torch


def torch_dtype_from_config(value: str | None):
    if value in (None, "auto"):
        return "auto"
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def render_prompt(tokenizer: Any, prompt: str, prompt_format: str) -> str:
    if prompt_format == "chat":
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"User: {prompt}\nAssistant:"
    if prompt_format == "completion":
        return prompt
    if prompt_format == "text2text":
        return prompt
    raise ValueError(f"Unsupported prompt_format: {prompt_format}")


def offline_mode() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def load_text_generation_model(model_name: str, dtype: str | None, trust_remote_code: bool):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    local_files_only = offline_mode()
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_cls = AutoModelForSeq2SeqLM if getattr(config, "is_encoder_decoder", False) else AutoModelForCausalLM
    model = model_cls.from_pretrained(
        model_name,
        torch_dtype=torch_dtype_from_config(dtype),
        trust_remote_code=trust_remote_code,
        device_map="auto",
        local_files_only=local_files_only,
    )
    model.eval()
    return model, tokenizer, getattr(config, "is_encoder_decoder", False)



def _usable_model_max_length(tokenizer: Any, model: Any) -> int:
    candidates = []
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and tokenizer_limit < 100_000:
        candidates.append(tokenizer_limit)

    config = getattr(model, "config", None)
    for attr in (
        "max_position_embeddings",
        "n_positions",
        "max_sequence_length",
        "seq_length",
        "max_encoder_position_embeddings",
    ):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            candidates.append(value)

    return min(candidates) if candidates else 2048


@torch.no_grad()
def generate_responses(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    is_encoder_decoder: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_input_tokens: int | None = None,
) -> list[str]:
    device = next(model.parameters()).device
    context_window = _usable_model_max_length(tokenizer, model)
    if max_input_tokens is None:
        reserved_tokens = 0 if is_encoder_decoder else max_new_tokens
        max_input_tokens = max(8, context_window - reserved_tokens)
    else:
        max_input_tokens = min(max_input_tokens, context_window)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    input_len = inputs["input_ids"].shape[1]
    if input_len >= context_window and not is_encoder_decoder:
        logging.warning(
            "Decoder-only input length %d is at/above context window %d; reduce max_input_tokens.",
            input_len,
            context_window,
        )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
    )
    if is_encoder_decoder:
        output_tokens = generated
    else:
        output_tokens = generated[:, input_len:]
    return tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
