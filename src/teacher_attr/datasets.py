from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


def format_prompt_response(prompt: str, response: str) -> str:
    return f"[PROMPT]\n{prompt}\n\n[RESPONSE]\n{response}"


class AttributionPairDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], teacher_ids: list[str]):
        self.rows = rows
        self.teacher_ids = teacher_ids

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        true_teacher = row["true_teacher"]
        return {
            "prompt_id": row["prompt_id"],
            "task": row.get("task"),
            "anchor_student_id": row["anchor_student_id"],
            "true_teacher": true_teacher,
            "label": self.teacher_ids.index(true_teacher),
            "student_text": format_prompt_response(row["prompt"], row["student_response"]),
            "teacher_texts": [
                format_prompt_response(row["prompt"], row["teacher_responses"][teacher_id])
                for teacher_id in self.teacher_ids
            ],
        }


class ContrastiveCollator:
    def __init__(self, tokenizer: Any, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        student_texts = [example["student_text"] for example in examples]
        teacher_texts_nested = [example["teacher_texts"] for example in examples]
        teacher_texts = [text for texts in teacher_texts_nested for text in texts]

        labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)
        batch_size = len(examples)
        num_teachers = len(teacher_texts_nested[0]) if teacher_texts_nested else 0

        return {
            "student": self._tokenize(student_texts),
            "teachers": self._tokenize(teacher_texts),
            "labels": labels,
            "batch_size": batch_size,
            "num_teachers": num_teachers,
            "metadata": [
                {
                    "prompt_id": example["prompt_id"],
                    "task": example.get("task"),
                    "anchor_student_id": example["anchor_student_id"],
                    "true_teacher": example["true_teacher"],
                }
                for example in examples
            ],
        }
