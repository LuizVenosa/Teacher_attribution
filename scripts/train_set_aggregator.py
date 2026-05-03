from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from teacher_attr.aggregators import DeepSetsAggregator
from teacher_attr.datasets import format_prompt_response
from teacher_attr.encoders import AttributionEncoder
from teacher_attr.generation import batch_iter
from teacher_attr.io import load_jsonl, load_yaml, save_json
from teacher_attr.utils import get_device, seed_everything, setup_logging, teacher_order_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an optional Deep Sets classifier on frozen embeddings.")
    parser.add_argument("--models_config", required=True)
    parser.add_argument("--attribution_config", required=True)
    parser.add_argument("--train_pairs", required=True)
    parser.add_argument("--val_pairs", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="models/attribution_encoder/set_aggregator")
    parser.add_argument("--set_size", type=int, default=8)
    parser.add_argument("--samples_per_group", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    return parser.parse_args()


@torch.no_grad()
def encode_student_rows(
    rows: list[dict[str, Any]],
    model: AttributionEncoder,
    tokenizer: Any,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    texts = [format_prompt_response(row["prompt"], row["student_response"]) for row in rows]
    chunks = []
    for batch in batch_iter(texts, batch_size):
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


class SetEmbeddingDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        embeddings: np.ndarray,
        teacher_ids: list[str],
        set_size: int,
        samples_per_group: int,
        seed: int,
    ):
        self.teacher_ids = teacher_ids
        self.set_size = set_size
        self.samples_per_group = samples_per_group
        self.rng = np.random.default_rng(seed)
        grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
        for idx, row in enumerate(rows):
            grouped[(row["anchor_student_id"], row["true_teacher"])].append(idx)
        self.groups = [
            {
                "indexes": indexes,
                "label": teacher_ids.index(true_teacher),
            }
            for (_, true_teacher), indexes in grouped.items()
        ]
        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.groups) * self.samples_per_group

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        group = self.groups[idx % len(self.groups)]
        indexes = group["indexes"]
        replace = len(indexes) < self.set_size
        selected = self.rng.choice(indexes, size=self.set_size, replace=replace)
        x = torch.tensor(self.embeddings[selected], dtype=torch.float32)
        y = torch.tensor(group["label"], dtype=torch.long)
        return x, y


def evaluate_head(
    aggregator: DeepSetsAggregator,
    classifier: nn.Linear,
    loader: DataLoader,
    device: torch.device,
) -> float:
    aggregator.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = classifier(aggregator(x)).argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / max(total, 1)


def main() -> None:
    setup_logging()
    args = parse_args()

    models_cfg = load_yaml(args.models_config)
    attr_cfg = load_yaml(args.attribution_config)
    seed_everything(attr_cfg.get("seed", 13))

    teacher_ids = teacher_order_from_config(models_cfg)
    model_name = models_cfg["attribution_encoder"]["hf_name"]
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(Path(args.checkpoint).parent, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.unk_token

    encoder = AttributionEncoder(
        model_name=model_name,
        projection_dim=attr_cfg["projection_dim"],
        num_teachers=len(teacher_ids),
    )
    payload = torch.load(args.checkpoint, map_location="cpu")
    encoder.load_state_dict(payload["model"])
    encoder.to(device)
    encoder.eval()

    train_rows = load_jsonl(args.train_pairs)
    val_rows = load_jsonl(args.val_pairs)
    train_emb = encode_student_rows(
        train_rows,
        encoder,
        tokenizer,
        device,
        attr_cfg["max_length"],
        attr_cfg.get("eval_batch_size", 16),
    )
    val_emb = encode_student_rows(
        val_rows,
        encoder,
        tokenizer,
        device,
        attr_cfg["max_length"],
        attr_cfg.get("eval_batch_size", 16),
    )

    train_ds = SetEmbeddingDataset(
        train_rows,
        train_emb,
        teacher_ids,
        args.set_size,
        args.samples_per_group,
        seed=attr_cfg.get("seed", 13),
    )
    val_ds = SetEmbeddingDataset(
        val_rows,
        val_emb,
        teacher_ids,
        args.set_size,
        args.samples_per_group,
        seed=attr_cfg.get("seed", 13) + 1,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    aggregator = DeepSetsAggregator(emb_dim=attr_cfg["projection_dim"]).to(device)
    classifier = nn.Linear(attr_cfg["projection_dim"], len(teacher_ids)).to(device)
    optimizer = torch.optim.AdamW(
        list(aggregator.parameters()) + list(classifier.parameters()),
        lr=args.learning_rate,
    )

    best = -1.0
    history = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        aggregator.train()
        classifier.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            loss = nn.functional.cross_entropy(classifier(aggregator(x)), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_acc = evaluate_head(aggregator, classifier, val_loader, device)
        history.append({"epoch": epoch, "val_accuracy": val_acc})
        if val_acc > best:
            best = val_acc
            torch.save(
                {
                    "aggregator": aggregator.state_dict(),
                    "classifier": classifier.state_dict(),
                    "teacher_ids": teacher_ids,
                    "set_size": args.set_size,
                    "projection_dim": attr_cfg["projection_dim"],
                },
                output_dir / "best.pt",
            )

    save_json(output_dir / "training_metrics.json", {"history": history, "best_accuracy": best})


if __name__ == "__main__":
    main()
