from __future__ import annotations

from typing import Any


def teacher_output_to_sft_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "teacher_id": row["teacher_id"],
        "prompt_id": row["prompt_id"],
        "messages": [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ],
    }
