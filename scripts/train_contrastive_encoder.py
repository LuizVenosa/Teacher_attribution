from __future__ import annotations

import argparse
import logging
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from teacher_attr.datasets import AttributionPairDataset, ContrastiveCollator
from teacher_attr.encoders import AttributionEncoder
from teacher_attr.io import load_jsonl, load_yaml, save_json
from teacher_attr.losses import same_prompt_infonce, same_prompt_scores
from teacher_attr.metrics import accuracy_by_task, classification_metrics
from teacher_attr.utils import count_parameters, get_device, seed_everything, setup_logging
from teacher_attr.utils import teacher_order_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train prompt-conditioned contrastive encoder.")
    parser.add_argument("--models_config", required=True)
    parser.add_argument("--attribution_config", required=True)
    return parser.parse_args()




def _offline_mode() -> bool:
    import os

    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def _is_improvement(score: float, best: float, *, mode: str, min_delta: float) -> bool:
    if mode == "max":
        return score > best + min_delta
    if mode == "min":
        return score < best - min_delta
    raise ValueError(f"Unsupported early stopping mode: {mode}")

def move_tokens(tokens: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in tokens.items()}


def autocast_context(device: torch.device, mixed_precision: str | None):
    if device.type != "cuda" or mixed_precision in (None, "none", "fp32"):
        return nullcontext()
    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


@torch.no_grad()
def evaluate(
    model: AttributionEncoder,
    loader: DataLoader,
    device: torch.device,
    teacher_ids: list[str],
    temperature: float,
    classification_weight: float,
    mixed_precision: str | None,
) -> dict[str, Any]:
    model.eval()
    score_chunks = []
    label_chunks = []
    tasks: list[str | None] = []
    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_classification_loss = 0.0
    total_examples = 0

    for batch in loader:
        labels = batch["labels"].to(device)
        batch_size = int(batch["batch_size"])
        with autocast_context(device, mixed_precision):
            student_emb, student_logits = model(**move_tokens(batch["student"], device))
            teacher_emb, _ = model(**move_tokens(batch["teachers"], device))
            teacher_emb = teacher_emb.view(batch_size, batch["num_teachers"], -1)
            scores = same_prompt_scores(student_emb, teacher_emb, temperature=temperature)
            contrastive_loss = F.cross_entropy(scores, labels)
            classification_loss = F.cross_entropy(student_logits, labels)
            loss = contrastive_loss + classification_weight * classification_loss

        total_loss += float(loss.detach().cpu()) * batch_size
        total_contrastive_loss += float(contrastive_loss.detach().cpu()) * batch_size
        total_classification_loss += float(classification_loss.detach().cpu()) * batch_size
        total_examples += batch_size
        score_chunks.append(scores.float().cpu().numpy())
        label_chunks.append(labels.cpu().numpy())
        tasks.extend(meta.get("task") for meta in batch["metadata"])

    scores_np = np.concatenate(score_chunks, axis=0)
    labels_np = np.concatenate(label_chunks, axis=0)

    metrics = classification_metrics(scores_np, labels_np, teacher_ids)
    metrics["accuracy_by_task"] = accuracy_by_task(scores_np, labels_np, tasks)
    metrics["loss"] = total_loss / max(total_examples, 1)
    metrics["contrastive_loss"] = total_contrastive_loss / max(total_examples, 1)
    metrics["classification_loss"] = total_classification_loss / max(total_examples, 1)
    return metrics


def save_checkpoint(
    path: Path,
    model: AttributionEncoder,
    tokenizer: Any,
    models_cfg: dict[str, Any],
    attr_cfg: dict[str, Any],
    teacher_ids: list[str],
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "models_config": models_cfg,
            "attribution_config": attr_cfg,
            "teacher_ids": teacher_ids,
            "metrics": metrics,
        },
        path,
    )
    tokenizer.save_pretrained(path.parent)


def main() -> None:
    setup_logging()
    args = parse_args()

    models_cfg = load_yaml(args.models_config)
    attr_cfg = load_yaml(args.attribution_config)
    seed_everything(attr_cfg.get("seed", 13))

    teacher_ids = teacher_order_from_config(models_cfg)
    model_name = models_cfg["attribution_encoder"]["hf_name"]
    output_dir = Path(attr_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=_offline_mode(),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.unk_token

    train_rows = load_jsonl(attr_cfg["train_file"])
    val_rows = load_jsonl(attr_cfg["val_file"])
    train_ds = AttributionPairDataset(train_rows, teacher_ids)
    val_ds = AttributionPairDataset(val_rows, teacher_ids)
    collator = ContrastiveCollator(tokenizer, max_length=attr_cfg["max_length"])

    train_loader = DataLoader(
        train_ds,
        batch_size=attr_cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        collate_fn=collator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=attr_cfg.get("eval_batch_size", attr_cfg["batch_size"]),
        shuffle=False,
        num_workers=2,
        collate_fn=collator,
        pin_memory=True,
    )

    device = get_device()
    model = AttributionEncoder(
        model_name=model_name,
        projection_dim=attr_cfg["projection_dim"],
        num_teachers=len(teacher_ids),
        local_files_only=_offline_mode(),
    ).to(device)
    model.enable_gradient_checkpointing()
    logging.info("Encoder backbone: %s", model_name)
    logging.info("Output directory: %s", output_dir)
    logging.info("Attribution config: %s", args.attribution_config)
    logging.info("Model parameters: %s", count_parameters(model))

    optimizer = AdamW(
        model.parameters(),
        lr=attr_cfg["learning_rate"],
        weight_decay=attr_cfg["weight_decay"],
    )
    max_epochs = int(attr_cfg.get("max_epochs", attr_cfg["num_epochs"]))
    steps_per_epoch = math.ceil(len(train_loader) / attr_cfg["gradient_accumulation_steps"])
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = int(total_steps * attr_cfg["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    monitor_metric = attr_cfg.get("early_stopping_metric", "accuracy")
    early_stopping_mode = attr_cfg.get("early_stopping_mode", "max")
    early_stopping_patience = attr_cfg.get("early_stopping_patience")
    early_stopping_min_delta = float(attr_cfg.get("early_stopping_min_delta", 0.0))
    if early_stopping_mode not in {"max", "min"}:
        raise ValueError(f"Unsupported early_stopping_mode={early_stopping_mode!r}")

    best_score = -math.inf if early_stopping_mode == "max" else math.inf
    best_acc = -1.0
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    epochs_without_improvement = 0
    history = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}", dynamic_ncols=True)
        for step, batch in enumerate(progress, start=1):
            labels = batch["labels"].to(device)
            with autocast_context(device, attr_cfg.get("mixed_precision")):
                student_emb, student_logits = model(**move_tokens(batch["student"], device))
                teacher_emb, _ = model(**move_tokens(batch["teachers"], device))
                teacher_emb = teacher_emb.view(batch["batch_size"], batch["num_teachers"], -1)

                contrastive_loss = same_prompt_infonce(
                    student_emb,
                    teacher_emb,
                    labels,
                    temperature=attr_cfg["temperature"],
                )
                cls_loss = F.cross_entropy(student_logits, labels)
                loss = contrastive_loss + attr_cfg["classification_weight"] * cls_loss
                loss = loss / attr_cfg["gradient_accumulation_steps"]

            loss.backward()
            running_loss += float(loss.detach().cpu()) * attr_cfg["gradient_accumulation_steps"]

            should_step = (
                step % attr_cfg["gradient_accumulation_steps"] == 0 or step == len(train_loader)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), attr_cfg["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(loss=running_loss / step)

        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            teacher_ids=teacher_ids,
            temperature=attr_cfg["temperature"],
            classification_weight=attr_cfg["classification_weight"],
            mixed_precision=attr_cfg.get("mixed_precision"),
        )
        history.append({"epoch": epoch, "train_loss": running_loss / len(train_loader), **val_metrics})
        current_score = float(val_metrics[monitor_metric])
        logging.info(
            "epoch=%d val_accuracy=%.4f val_loss=%.4f %s=%.4f",
            epoch,
            val_metrics["accuracy"],
            val_metrics["loss"],
            monitor_metric,
            current_score,
        )

        save_checkpoint(
            output_dir / "last.pt",
            model,
            tokenizer,
            models_cfg,
            attr_cfg,
            teacher_ids,
            val_metrics,
        )

        if _is_improvement(
            current_score,
            best_score,
            mode=early_stopping_mode,
            min_delta=early_stopping_min_delta,
        ):
            best_score = current_score
            best_acc = val_metrics["accuracy"]
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            epochs_without_improvement = 0
            save_checkpoint(
                output_dir / "best.pt",
                model,
                tokenizer,
                models_cfg,
                attr_cfg,
                teacher_ids,
                val_metrics,
            )
        else:
            epochs_without_improvement += 1

        summary = {
            "run_type": "contrastive_encoder_train",
            "encoder_model": model_name,
            "models_config_path": args.models_config,
            "attribution_config_path": args.attribution_config,
            "output_dir": str(output_dir),
            "train_file": attr_cfg["train_file"],
            "val_file": attr_cfg["val_file"],
            "teacher_ids": teacher_ids,
            "history": history,
            "best_accuracy": best_acc,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_metrics": best_metrics,
            "monitor_metric": monitor_metric,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "early_stopped": False,
            "stopped_epoch": epoch,
        }
        save_json(output_dir / "training_metrics.json", summary)

        if early_stopping_patience is not None and epochs_without_improvement >= int(
            early_stopping_patience
        ):
            logging.info(
                "early stopping at epoch=%d best_epoch=%d best_%s=%.4f",
                epoch,
                best_epoch,
                monitor_metric,
                best_score,
            )
            summary["early_stopped"] = True
            save_json(output_dir / "training_metrics.json", summary)
            break


if __name__ == "__main__":
    main()
