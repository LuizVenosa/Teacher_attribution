from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any


def batch_iter(items: list[Any], batch_size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def render_chat_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"User: {prompt}\nAssistant:"


def clean_generation(text: str) -> str:
    return text.strip()


def take_unseen_prompt_rows(
    prompt_rows: Iterable[dict[str, Any]], done_prompt_ids: set[str]
) -> list[dict[str, Any]]:
    return [row for row in prompt_rows if str(row["prompt_id"]) not in done_prompt_ids]
