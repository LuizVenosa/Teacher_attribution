from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from teacher_attr.datasets import format_prompt_response
from teacher_attr.encoders import AttributionEncoder
from teacher_attr.generation import batch_iter
from teacher_attr.io import load_jsonl, load_yaml, save_json
from teacher_attr.metrics import accuracy_by_task, classification_metrics
from teacher_attr.utils import get_device, seed_everything, setup_logging, teacher_order_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a contrastive attribution encoder.")
    parser.add_argument("--models_config", required=True)
    parser.add_argument("--attribution_config", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_model(
    checkpoint_path: str | Path,
    model_name: str,
    projection_dim: int,
    num_teachers: int,
    device: torch.device,
) -> AttributionEncoder:
    model = AttributionEncoder(
        model_name=model_name,
        projection_dim=projection_dim,
        num_teachers=num_teachers,
    )
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def encode_texts(
    model: AttributionEncoder,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    chunks = []
    for batch in tqdm(list(batch_iter(texts, batch_size)), desc="encoding", dynamic_ncols=True):
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        emb, _ = model(**tokens)
        chunks.append(emb.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def build_texts(rows: list[dict[str, Any]], teacher_ids: list[str]) -> tuple[list[str], list[str]]:
    student_texts = [
        format_prompt_response(row["prompt"], row["student_response"])
        for row in rows
    ]
    teacher_texts = [
        format_prompt_response(row["prompt"], row["teacher_responses"][teacher_id])
        for row in rows
        for teacher_id in teacher_ids
    ]
    return student_texts, teacher_texts


def set_level_metrics(
    rows: list[dict[str, Any]],
    student_emb: np.ndarray,
    teacher_emb: np.ndarray,
    labels: np.ndarray,
    teacher_ids: list[str],
    set_sizes: list[int],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[(row["anchor_student_id"], row["true_teacher"])].append(idx)

    out: dict[str, Any] = {}
    for set_size in set_sizes:
        score_rows = []
        label_rows = []
        for _ in range(repeats):
            for indexes in groups.values():
                if len(indexes) < set_size:
                    continue
                selected = rng.choice(indexes, size=set_size, replace=False)
                s = student_emb[selected].mean(axis=0)
                s = s / max(np.linalg.norm(s), 1e-12)

                t = teacher_emb[selected].mean(axis=0)
                t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)

                score_rows.append(t @ s)
                label_rows.append(labels[selected[0]])

        if score_rows:
            scores = np.vstack(score_rows)
            set_labels = np.asarray(label_rows, dtype=np.int64)
            out[str(set_size)] = classification_metrics(scores, set_labels, teacher_ids)
            out[str(set_size)]["num_sets"] = int(len(set_labels))
        else:
            out[str(set_size)] = {"num_sets": 0, "accuracy": None}
    return out


def main() -> None:
    setup_logging()
    args = parse_args()

    models_cfg = load_yaml(args.models_config)
    attr_cfg = load_yaml(args.attribution_config)
    seed_everything(attr_cfg.get("seed", 13))

    teacher_ids = teacher_order_from_config(models_cfg)
    model_name = models_cfg["attribution_encoder"]["hf_name"]
    device = get_device()
    tokenizer_path = Path(args.checkpoint).parent
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.unk_token

    model = load_model(
        args.checkpoint,
        model_name=model_name,
        projection_dim=attr_cfg["projection_dim"],
        num_teachers=len(teacher_ids),
        device=device,
    )

    rows = load_jsonl(args.pairs)
    labels = np.asarray([teacher_ids.index(row["true_teacher"]) for row in rows], dtype=np.int64)
    student_texts, teacher_texts = build_texts(rows, teacher_ids)

    student_emb = encode_texts(
        model,
        tokenizer,
        student_texts,
        device,
        max_length=attr_cfg["max_length"],
        batch_size=attr_cfg.get("eval_batch_size", 16),
    )
    teacher_flat = encode_texts(
        model,
        tokenizer,
        teacher_texts,
        device,
        max_length=attr_cfg["max_length"],
        batch_size=attr_cfg.get("eval_batch_size", 16),
    )
    teacher_emb = teacher_flat.reshape(len(rows), len(teacher_ids), -1)

    student_t = torch.from_numpy(student_emb)
    teacher_t = torch.from_numpy(teacher_emb)
    scores = torch.einsum(
        "bd,bkd->bk",
        F.normalize(student_t, dim=-1),
        F.normalize(teacher_t, dim=-1),
    ).numpy()

    payload = {
        "num_rows": len(rows),
        "teacher_ids": teacher_ids,
        "single_prompt": classification_metrics(scores, labels, teacher_ids),
        "accuracy_by_task": accuracy_by_task(scores, labels, [row.get("task") for row in rows]),
        "set_level": set_level_metrics(
            rows,
            student_emb,
            teacher_emb,
            labels,
            teacher_ids,
            set_sizes=attr_cfg.get("set_sizes", [1, 2, 4, 8, 16, 32, 64]),
            repeats=attr_cfg.get("set_repeats", 20),
            seed=attr_cfg.get("seed", 13),
        ),
    }
    save_json(args.output, payload)


if __name__ == "__main__":
    main()
