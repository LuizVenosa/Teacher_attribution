"""Read-only token budgets for prepared prompts; never loads model weights."""

from __future__ import annotations

from pathlib import Path

from teacher_attr.context import context_limit

from teacher_attr.distillation import controlled_text, student_prompt
from teacher_attr.generation import output_path, read_outputs, render_prompt
from teacher_attr.io import load_jsonl
from teacher_attr.prompts import verify_prompts
from teacher_attr.research import POOLS, provenance


def summarize_lengths(rows: list[dict], lengths: list[int], budget: int) -> dict:
    ordered = sorted(lengths)
    longest = sorted(zip(rows, lengths, strict=True), key=lambda item: item[1], reverse=True)
    return {
        "rows": len(rows),
        "budget": budget,
        "max_tokens": max(lengths, default=0),
        "p99_tokens": ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))] if ordered else 0,
        "over_budget": sum(n > budget for n in lengths),
        "longest": [
            {"prompt_id": row["prompt_id"], "task": row["task"], "tokens": n}
            for row, n in longest[:5]
        ],
    }


def audit_token_budgets(cfg: dict) -> dict:
    import sys

    from transformers import AutoConfig, AutoTokenizer

    root = Path(cfg["run_dir"])
    manifest = verify_prompts(root)
    pools = {split: load_jsonl(root / "prompts" / f"{split}.jsonl") for split in POOLS}
    gen = cfg["generation"]
    reserve = max([gen["max_new_tokens"], *gen.get("task_max_new_tokens", {}).values()])
    report = {
        "weights_loaded": False,
        "prompts_sha256": manifest["sha256"],
        "models": {},
        "student_training": {},
        "environment": provenance(),
    }
    models = {**cfg["teachers"], "student_base": cfg["research"]["student"]}
    for name, model in models.items():
        print(f"Auditing {name}: tokenizer/config only", file=sys.stderr, flush=True)
        kwargs = {"revision": model["revision"], "trust_remote_code": False}
        tokenizer = AutoTokenizer.from_pretrained(model["hf_name"], **kwargs)
        config = AutoConfig.from_pretrained(model["hf_name"], **kwargs)
        context = context_limit(config, tokenizer)
        budget = min(
            gen["max_input_tokens"], context - (0 if config.is_encoder_decoder else reserve)
        )
        results = {}
        for split, rows in pools.items():
            lengths = []
            for row in rows:
                text = controlled_text(cfg, row)
                if name == "student_base":
                    text = student_prompt(text)
                    special = False
                else:
                    text = render_prompt(
                        tokenizer, text, model["prompt_format"], model.get("chat_kwargs")
                    )
                    special = model["prompt_format"] != "chat"
                lengths.append(len(tokenizer.encode(text, add_special_tokens=special)))
            results[split] = summarize_lengths(rows, lengths, budget)
        needed = max(s["max_tokens"] for s in results.values())
        report["models"][name] = {
            "splits": results,
            "context_limit": context,
            "output_reserve": reserve,
            "required_input_tokens": needed,
            "fits_context_with_output": needed + (0 if config.is_encoder_decoder else reserve)
            <= context,
        }
        if name == "student_base":
            for teacher in cfg["teachers"]:
                report["student_training"][teacher] = {}
                for split in ("distill_train", "distill_val"):
                    path = output_path(root, "teachers", teacher, split)
                    if not path.with_suffix(".meta.json").exists():
                        report["student_training"][teacher][split] = {"status": "not_generated"}
                        continue
                    rows = read_outputs(cfg, "teachers", teacher, split)
                    lengths = [
                        len(
                            tokenizer.encode(
                                student_prompt(controlled_text(cfg, row)), add_special_tokens=False
                            )
                        )
                        + len(tokenizer.encode(row["response"], add_special_tokens=False))
                        + int(tokenizer.eos_token_id is not None)
                        for row in rows
                    ]
                    result = summarize_lengths(
                        rows, lengths, cfg["research"]["training"]["max_length"]
                    )
                    result["complete"] = len(rows) == len(pools[split])
                    result["empty_targets"] = sum(not row["response"].strip() for row in rows)
                    result["over_model_context"] = sum(n > context for n in lengths)
                    report["student_training"][teacher][split] = result
    report["required_common_input_tokens"] = max(
        m["required_input_tokens"] for m in report["models"].values()
    )
    report["note"] = (
        "Counts match current generation tokenization, including the Ministral text round-trip. "
        "Missing teacher responses prevent a complete SFT budget guarantee. "
        "This audit does not measure GPU memory or alter prompts, configs, or outputs."
    )
    return report
