"""Single-GPU full-model SFT from cached teacher answers, with response-only loss."""

from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path

from teacher_attr.config import file_hash, fingerprint, initialize_run
from teacher_attr.generation import output_path, read_outputs
from teacher_attr.io import load_jsonl, save_json
from teacher_attr.research import nested_ids, provenance, student_id


def student_prompt(text: str) -> str:
    return f"User:\n{text}\n\nAssistant:\n"


def controlled_text(cfg: dict, row: dict) -> str:
    gen = cfg["generation"]
    text = row["prompt"]
    if gen.get("system_instruction"):
        text = gen["system_instruction"] + "\n\n" + text
    if gen.get("controlled_length"):
        text += "\n\n" + gen["task_instructions"][row["task"]]
    return text


def encode_example(tokenizer, prompt: str, response: str, max_length: int) -> dict:
    # Separate encoding establishes an exact, auditable loss boundary.
    prefix = tokenizer.encode(student_prompt(prompt), add_special_tokens=False)
    target = tokenizer.encode(response, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        target += [tokenizer.eos_token_id]
    if not target or not response.strip():
        raise ValueError("Empty student target")
    if len(prefix) + len(target) > max_length:
        raise ValueError("SFT example exceeds context; adjust common prompt/output budgets")
    return {"input_ids": prefix + target, "labels": [-100] * len(prefix) + target}


def collate_sft(rows: list[dict], pad_token_id: int):
    import torch

    length = max(len(r["input_ids"]) for r in rows)
    return {
        "input_ids": torch.tensor(
            [r["input_ids"] + [pad_token_id] * (length - len(r["input_ids"])) for r in rows]
        ),
        "labels": torch.tensor([r["labels"] + [-100] * (length - len(r["labels"])) for r in rows]),
        "attention_mask": torch.tensor(
            [[1] * len(r["input_ids"]) + [0] * (length - len(r["input_ids"])) for r in rows]
        ),
    }


def checkpoint_hashes(directory: Path) -> dict:
    return {
        str(p.relative_to(directory)): file_hash(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def verify_student(cfg: dict, name: str) -> dict:
    directory = Path(cfg["students"][name]["hf_name"])
    manifest = json.loads((directory.parent / "training.json").read_text())
    if not manifest.get("complete") or manifest["teacher"] != cfg["students"][name]["teacher"]:
        raise ValueError("Student checkpoint is incomplete or has wrong lineage")
    if manifest["student_id"] != name:
        raise ValueError("Student seed/amount identity does not match its training manifest")
    if checkpoint_hashes(directory) != manifest["checkpoint_sha256"]:
        raise ValueError("Student checkpoint changed after training")
    if manifest["base"] != cfg["research"]["student"]:
        raise ValueError("Student was trained from a different base")
    return manifest


def train_student(
    cfg: dict,
    teacher: str,
    seed: int | None = None,
    amount: int | None = None,
    post_data: str | None = None,
    level: str | None = None,
) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    from teacher_attr.prompts import verify_prompts

    root = initialize_run(cfg)
    verify_prompts(root)
    research = cfg["research"]
    if teacher not in cfg["teachers"]:
        raise ValueError("Unknown teacher")
    seed = research["primary_seed"] if seed is None else seed
    amount = research["primary_amount"] if amount is None else amount
    if seed not in research["student_seeds"] or amount not in research["amounts"]:
        raise ValueError("Student seed/amount must be configured before preparing the run")
    name = student_id(teacher, seed, amount)
    parent_name = name
    if post_data:
        if level not in research["additional_ft_epochs"]:
            raise ValueError("Unknown additional fine-tuning level")
        name += f"_ft_{level}"
    directory = root / "students" / name
    if directory.exists():
        raise ValueError(f"Student directory already exists: {directory}")
    from teacher_attr.exclusions import training_exclusions

    exclusions = training_exclusions(cfg)
    excluded = set(exclusions["prompt_ids"]) if exclusions else set()
    hashes = {}
    for split in ("distill_train", "distill_val"):
        quality = json.loads((root / "quality" / split / "metrics.json").read_text())
        if split == "distill_train" and quality.get("training_exclusions") != exclusions:
            raise ValueError("Training exclusions changed after QC; rerun QC")
        if not quality["passed"]:
            raise ValueError("Teacher QC failed; inspect quality metrics before distillation")
        for t in cfg["teachers"]:
            digest = file_hash(output_path(root, "teachers", t, split))
            if quality["hashes"][t] != digest:
                raise ValueError("Teacher outputs changed after QC")
        hashes[split] = quality["hashes"][teacher]
    rows = {s: read_outputs(cfg, "teachers", teacher, s) for s in hashes}
    subset = [pid for pid in nested_ids(cfg, amount) if pid not in excluded]
    if not subset:
        raise ValueError("No training examples remain after exclusions")
    index = {r["prompt_id"]: r for r in rows["distill_train"]}
    rows["distill_train"] = [index[pid] for pid in subset]
    if any(r.get("quality_flags") or not r["response"].strip() for rs in rows.values() for r in rs):
        raise ValueError("Invalid targets cannot be silently dropped from a controlled dataset")
    train, base = dict(research["training"]), research["student"]
    load_base, post_metadata = base, None
    if post_data:
        from teacher_attr.prompts import text_hash, verify_prompts

        parent_checkpoint = root / "students" / parent_name / "checkpoint"
        parent_manifest = json.loads((parent_checkpoint.parent / "training.json").read_text())
        if checkpoint_hashes(parent_checkpoint) != parent_manifest["checkpoint_sha256"]:
            raise ValueError("Parent student checkpoint changed")
        independent = load_jsonl(post_data)
        forbidden = {
            text_hash(r["prompt"])
            for split in verify_prompts(root)["sha256"]
            for r in load_jsonl(root / "prompts" / f"{split}.jsonl")
        }
        ids = [text_hash(r["prompt"]) for r in independent]
        if len(set(ids)) != len(ids) or set(ids) & forbidden:
            raise ValueError("Additional fine-tuning prompts overlap existing pools or duplicate")
        if {r["split"] for r in independent} != {"train", "val"}:
            raise ValueError("Additional FT data must contain explicit train and val splits")
        rows = {
            s: [
                {**r, "task": r.get("task", "instruction")}
                for r in independent
                if r["split"] == raw
            ]
            for s, raw in (("distill_train", "train"), ("distill_val", "val"))
        }
        train["epochs"] = research["additional_ft_epochs"][level]
        load_base = {"hf_name": str(parent_checkpoint), "revision": "local"}
        post_metadata = {
            "level": level,
            "data_sha256": file_hash(post_data),
            "parent_manifest_sha256": file_hash(parent_checkpoint.parent / "training.json"),
            "train_rows": len(rows["distill_train"]),
            "validation_rows": len(rows["distill_val"]),
        }
    if post_metadata:
        for previous in (root / "students").glob(f"*_ft_{level}/training.json"):
            metadata = json.loads(previous.read_text()).get("additional_finetuning")
            if metadata and metadata["data_sha256"] != post_metadata["data_sha256"]:
                raise ValueError("All students at this FT level must use the same independent data")
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if train["precision"] == "bfloat16" and device != "cuda":
        raise ValueError("Primary BF16 training requires CUDA; use float32 only for CPU pilots")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = train["tf32"]
        torch.cuda.reset_peak_memory_stats()
    kwargs = {"revision": load_base["revision"], "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(load_base["hf_name"], **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Student tokenizer needs PAD or EOS")
    # FP32 master weights/optimizer; BF16 autocast for the forward pass.
    model = AutoModelForCausalLM.from_pretrained(
        load_base["hf_name"],
        torch_dtype=torch.float32,
        attn_implementation=train["attention"],
        **kwargs,
    ).to(device)
    model.config.use_cache = False
    if train["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    encoded = {
        s: [
            encode_example(tokenizer, controlled_text(cfg, r), r["response"], train["max_length"])
            for r in rs
        ]
        for s, rs in rows.items()
    }
    loaders = {
        s: DataLoader(
            rs,
            batch_size=train["microbatch"],
            shuffle=s == "distill_train",
            collate_fn=partial(collate_sft, pad_token_id=tokenizer.pad_token_id),
        )
        for s, rs in encoded.items()
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train["learning_rate"],
        weight_decay=train["weight_decay"],
        fused=train["fused_adamw"] and device == "cuda",
    )
    directory.mkdir(parents=True)
    started, history, tokens_processed, steps = time.perf_counter(), [], 0, 0
    accumulation = train["gradient_accumulation"]
    for epoch in range(train["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, total_tokens, window_tokens = 0.0, 0, 0
        for i, batch in enumerate(loaders["distill_train"]):
            batch = {k: v.to(device) for k, v in batch.items()}
            count = int((batch["labels"][:, 1:] != -100).sum())
            with torch.autocast(
                device_type=device, dtype=torch.bfloat16, enabled=train["precision"] == "bfloat16"
            ):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite distillation loss")
            (loss * count).backward()
            total_loss += loss.item() * count
            total_tokens += count
            window_tokens += count
            if (i + 1) % accumulation == 0 or i + 1 == len(loaders["distill_train"]):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(window_tokens)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train["max_grad_norm"], error_if_nonfinite=True
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                window_tokens = 0
                steps += 1
                if steps % train["log_every"] == 0:
                    print(
                        f"student={name} epoch={epoch + 1} step={steps} "
                        f"loss={total_loss / total_tokens:.4f}",
                        flush=True,
                    )
        model.eval()
        val_loss, val_tokens = 0.0, 0
        with torch.inference_mode():
            for batch in loaders["distill_val"]:
                batch = {k: v.to(device) for k, v in batch.items()}
                count = int((batch["labels"][:, 1:] != -100).sum())
                with torch.autocast(
                    device_type=device,
                    dtype=torch.bfloat16,
                    enabled=train["precision"] == "bfloat16",
                ):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite validation loss")
                val_loss += loss.item() * count
                val_tokens += count
        tokens_processed += total_tokens
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / total_tokens,
                "validation_loss": val_loss / val_tokens,
                "target_tokens": total_tokens,
            }
        )
        save_json(directory / "progress.json", {"history": history, "complete": False})
    # Same fixed epoch count for every teacher: validation is diagnostic, not per-teacher selection.
    model.config.use_cache = True
    model.save_pretrained(directory / "checkpoint", safe_serialization=True)
    tokenizer.save_pretrained(directory / "checkpoint")
    report = {
        "complete": True,
        "additional_finetuning": post_metadata,
        "teacher": teacher,
        "student_id": name,
        "base": base,
        "resolved_base_revision": getattr(model.config, "_commit_hash", None),
        "seed": seed,
        "amount": amount,
        "effective_amount": len(subset),
        "training_exclusions": exclusions,
        "configuration": train,
        "history": history,
        "prompt_ids": subset,
        "prompt_ids_sha256": fingerprint(subset),
        "teacher_outputs_sha256": hashes,
        "target_tokens_processed": tokens_processed,
        "optimizer_steps": steps,
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        "environment": provenance(),
        "checkpoint": str(directory / "checkpoint"),
        "checkpoint_sha256": checkpoint_hashes(directory / "checkpoint"),
    }
    save_json(directory / "training.json", report)
    return report
