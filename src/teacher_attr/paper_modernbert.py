"""Fit a response-only classifier on teacher text and transfer it to students."""

from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from teacher_attr.encoders import to_device, tokenize_pairs
from teacher_attr.metrics import grouped_indexes
from teacher_attr.paper_baselines import teacher_examples
from teacher_attr.training import environment, fit_encoder, prepare_encoder


def collate_teacher_text(examples, tokenizer, enc):
    tokens, _ = tokenize_pairs(tokenizer, [("", r["response"]) for r in examples], enc)
    return tokens, None, torch.tensor([r["label"] for r in examples])


def modernbert_scores(splits, teachers, cfg, scope):
    enc = {
        **cfg["encoder"],
        "input_mode": "response",
        "objective": "classification",
        "representation": "semantic",
    }
    seed = cfg["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pools = {s: teacher_examples(splits[s], teachers) for s in ("train", "val")}
    test = splits["test"]
    groups = (
        grouped_indexes([r["source_dataset"] for r in test])
        if scope == "per_dataset"
        else {"all": np.arange(len(test))}
    )
    scores = np.empty((len(test), len(teachers)))
    selection = {}
    for dataset, indexes in groups.items():
        examples = {
            split: [r for r in rows if scope == "pooled" or r["source_dataset"] == dataset]
            for split, rows in pools.items()
        }
        if any(not rows for rows in examples.values()):
            raise ValueError(f"No teacher training/validation examples for {dataset}")
        print(f"Teacher-supervised classifier dataset={dataset}", flush=True)
        model, tokenizer = prepare_encoder(enc, len(teachers), seed, device)
        loaders = {
            split: DataLoader(
                rows,
                batch_size=enc["batch_size"],
                shuffle=split == "train",
                collate_fn=partial(collate_teacher_text, tokenizer=tokenizer, enc=enc),
            )
            for split, rows in examples.items()
        }
        best = {}

        def save_best(_epoch, model=model, best=best):
            for name, parameter in model.state_dict().items():
                best[name] = parameter.detach().to("cpu", copy=True)

        progress = fit_encoder(model, loaders, enc, device, len(teachers), save_best)
        model.load_state_dict(best)
        model.eval()
        truncated = 0
        with torch.inference_mode():
            for offset in range(0, len(indexes), enc["batch_size"]):
                batch = indexes[offset : offset + enc["batch_size"]]
                tokens, audit = tokenize_pairs(
                    tokenizer, [("", test[i]["student_response"]) for i in batch], enc
                )
                truncated += sum(
                    r["response_tokens_retained"] < r["response_tokens"] for r in audit
                )
                _, logits = model(**to_device(tokens, device))
                scores[batch] = logits.float().cpu().numpy()
        selection[dataset] = {
            **progress,
            "training_rows": len(examples["train"]),
            "validation_rows": len(examples["val"]),
            "test_rows": len(indexes),
            "truncated_test_rows": truncated,
            "resolved_revision": getattr(model.backbone.config, "_commit_hash", None),
        }
        del model, best, save_best
    return scores, {
        "scope": scope,
        "selection_data": "teacher validation only",
        "encoder": enc,
        "seed": seed,
        "environment": environment(),
        "groups": selection,
    }
