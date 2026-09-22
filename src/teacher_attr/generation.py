from __future__ import annotations

import importlib.metadata
import json
from collections import Counter
from pathlib import Path

from teacher_attr.config import file_hash, fingerprint, initialize_run
from teacher_attr.io import append_jsonl, load_jsonl, save_json
from teacher_attr.prompts import verify_prompts


def validate_resume_metadata(stored: dict, current: dict) -> None:
    """A launcher-only Git commit must not invalidate identical generation settings."""

    def comparable(metadata):
        result = dict(metadata)
        if "runtime" in result:
            result["runtime"] = {
                key: value for key, value in result["runtime"].items() if key != "git_commit"
            }
        return result

    before, after = comparable(stored), comparable(current)
    changed = sorted(
        key for key in before.keys() | after.keys() if before.get(key) != after.get(key)
    )
    if changed:
        raise ValueError(
            f"Generation resume metadata changed: {', '.join(changed)}. "
            "Restore the original settings/environment or use a new run."
        )


def output_path(root: Path, role: str, model_id: str, split: str) -> Path:
    return root / "outputs" / role / model_id / f"{split}.jsonl"


def render_prompt(tokenizer, prompt: str, mode: str, chat_kwargs: dict | None = None) -> str:
    if mode == "chat":
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **(chat_kwargs or {}),
        )
    return prompt


def generation_spec(cfg: dict, role: str, model_id: str, split: str) -> dict:
    return {
        "schema_version": 2,
        "role": role,
        "model_id": model_id,
        "split": split,
        "model": cfg[role][model_id],
        "generation": cfg["generation"],
        "seed": cfg["seed"],
        "prompts_sha256": file_hash(Path(cfg["run_dir"]) / "prompts" / f"{split}.jsonl"),
    }


def read_outputs(cfg: dict, role: str, model_id: str, split: str) -> list[dict]:
    path = output_path(Path(cfg["run_dir"]), role, model_id, split)
    manifest = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if manifest["spec"] != generation_spec(cfg, role, model_id, split):
        raise ValueError(f"Generation configuration mismatch: {path}")
    rows = load_jsonl(path) if path.exists() else []
    if len({r["prompt_id"] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate generated prompt IDs: {path}")
    for row in rows:
        if row["generation_fingerprint"] != fingerprint(manifest):
            raise ValueError(f"Mixed generation artifacts: {path}")
    return rows


def generate(cfg: dict, role: str, model_id: str, split: str) -> dict:
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        set_seed,
    )

    root = initialize_run(cfg)
    verify_prompts(root)
    model_cfg = cfg[role][model_id]
    if role == "students" and "research" in cfg:
        from teacher_attr.distillation import verify_student

        verify_student(cfg, model_id)
    if role == "students" and split not in model_cfg["splits"]:
        raise ValueError(f"{model_id} is not assigned to {split}")
    prompts = load_jsonl(root / "prompts" / f"{split}.jsonl")
    path = output_path(root, role, model_id, split)
    meta_path = path.with_suffix(".meta.json")
    previous = read_outputs(cfg, role, model_id, split) if meta_path.exists() else []
    expected = {p["prompt_id"]: p["prompt"] for p in prompts}
    if path.exists() and not meta_path.exists():
        raise ValueError(f"Output has no generation manifest: {path}")
    if any(
        r["prompt_id"] not in expected or r["prompt"] != expected[r["prompt_id"]] for r in previous
    ):
        raise ValueError("Stored generation does not match current prompts")
    done = {r["prompt_id"] for r in previous}
    if len(done) == len(prompts):
        return {"model": model_id, "split": split, "rows": len(done), "resumed": True}
    kwargs = {"revision": model_cfg["revision"], "trust_remote_code": False}
    config = AutoConfig.from_pretrained(model_cfg["hf_name"], **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_name"], **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must have a pad or EOS token")
    gen = cfg["generation"]

    def prepared_prompt(row):
        text = row["prompt"]
        if gen.get("system_instruction"):
            text = gen["system_instruction"] + "\n\n" + text
        if gen.get("controlled_length"):
            text += "\n\n" + gen["task_instructions"][row["task"]]
        if role == "students" and "research" in cfg:
            from teacher_attr.distillation import student_prompt

            return student_prompt(text)
        return render_prompt(
            tokenizer, text, model_cfg["prompt_format"], model_cfg.get("chat_kwargs")
        )

    rendered = [prepared_prompt(p) for p in prompts]
    text_config = getattr(config, "text_config", config)
    limits = [getattr(text_config, k, None) for k in ("max_position_embeddings", "n_positions")]
    limits += [tokenizer.model_max_length]
    context = min((n for n in limits if isinstance(n, int) and 0 < n < 100000), default=2048)
    budget = min(
        gen["max_input_tokens"],
        context
        - (
            0
            if config.is_encoder_decoder
            else max([gen["max_new_tokens"], *gen.get("task_max_new_tokens", {}).values()])
        ),
    )
    # Fail rather than silently give different teachers different article prefixes.
    add_special_tokens = model_cfg["prompt_format"] != "chat" and not (
        role == "students" and "research" in cfg
    )
    lengths = [len(tokenizer.encode(p, add_special_tokens=add_special_tokens)) for p in rendered]
    if max(lengths) > budget:
        raise ValueError(
            f"{model_id}: prompt needs {max(lengths)} tokens, budget is {budget}. "
            "Reduce data.max_input_chars in a new run; generation never truncates."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, gen["dtype"])
    if device == "cpu" and dtype == torch.float16:
        raise ValueError("Use float32 for CPU generation")
    cls = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM
    if model_cfg.get("model_class"):
        import transformers

        cls = getattr(transformers, model_cfg["model_class"])
    load_kwargs = dict(kwargs)
    if gen.get("attention"):
        load_kwargs["attn_implementation"] = gen["attention"]
    model = cls.from_pretrained(model_cfg["hf_name"], torch_dtype=dtype, **load_kwargs).to(device)
    model.eval()
    metadata = {
        "response_processing_version": 2,
        "spec": generation_spec(cfg, role, model_id, split),
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "environment": {p: importlib.metadata.version(p) for p in ("torch", "transformers")},
        "local_model_sha256": {
            p.name: file_hash(p)
            for p in sorted(Path(model_cfg["hf_name"]).glob("*"))
            if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".model", ".txt"}
        }
        if Path(model_cfg["hf_name"]).is_dir()
        else None,
    }
    if "research" in cfg:
        from teacher_attr.research import provenance

        metadata["runtime"] = provenance()
    if meta_path.exists():
        stored_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        validate_resume_metadata(stored_metadata, metadata)
        append_jsonl(
            path.with_suffix(".resume.jsonl"),
            [{"runtime": metadata.get("runtime"), "completed_rows": len(done)}],
        )
        # Existing rows hash the original manifest. Keep that identity on new rows,
        # recording the actual resume runtime separately instead of rewriting history.
        metadata = stored_metadata
    else:
        save_json(meta_path, metadata)
    temperature = (
        gen.get("student_temperature", gen["temperature"])
        if role == "students"
        else gen["temperature"]
    )
    sampling = {"do_sample": temperature > 0}
    if "min_new_tokens" in gen:
        minimum = gen["min_new_tokens"]
        limits = [gen["max_new_tokens"], *gen.get("task_max_new_tokens", {}).values()]
        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, int)
            or not 0 <= minimum <= min(limits)
        ):
            raise ValueError("min_new_tokens must be an integer between zero and every output cap")
        sampling["min_new_tokens"] = minimum
    if "research" in cfg:
        sampling.update(
            top_k=gen["top_k"],
            repetition_penalty=gen["repetition_penalty"],
            min_p=None,
            typical_p=1.0,
            epsilon_cutoff=0.0,
            eta_cutoff=0.0,
        )
    if sampling["do_sample"]:
        sampling.update(temperature=temperature, top_p=gen["top_p"])
    # Stable batches: interrupted batches are regenerated in full with the same seed.
    # Rows already committed are skipped only at write time, preserving random draws.
    tokenizer.padding_side = "left"
    from teacher_attr.quality import clean_response

    batch_size = gen.get("batch_size", 1)
    groups = {}
    for row, text in zip(prompts, rendered, strict=True):
        limit = gen.get("task_max_new_tokens", {}).get(row["task"], gen["max_new_tokens"])
        groups.setdefault(limit, []).append((row, text))
    for limit, items in groups.items():
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            if all(r["prompt_id"] in done for r, _ in batch):
                continue
            seed = int(
                fingerprint([cfg["seed"], role, model_id, [r["prompt_id"] for r, _ in batch]])[:8],
                16,
            )
            set_seed(seed)
            tokens = tokenizer(
                [t for _, t in batch],
                padding=True,
                return_tensors="pt",
                add_special_tokens=add_special_tokens,
            ).to(device)
            with torch.inference_mode():
                outputs = model.generate(
                    **tokens, max_new_tokens=limit, pad_token_id=tokenizer.pad_token_id, **sampling
                )
            new_rows = []
            for i, (row, rendered_prefix) in enumerate(batch):
                if row["prompt_id"] in done:
                    continue
                output = outputs[i]
                if not config.is_encoder_decoder:
                    output = output[tokens["input_ids"].shape[1] :]
                ids = output.tolist()
                eos = model.generation_config.eos_token_id
                eos_ids = eos if isinstance(eos, list) else [eos] if eos is not None else []
                stop = next((j for j, token in enumerate(ids) if token in eos_ids), len(ids))
                used = ids[:stop]
                text = tokenizer.decode(used, skip_special_tokens=False).strip()
                # Record token identities/counts, never raw hidden reasoning text.
                special_ids = set(tokenizer.all_special_ids)
                diagnostics = {
                    "generated_tokens": len(used),
                    "special_tokens": dict(
                        Counter(
                            tokenizer.convert_ids_to_tokens(t) for t in used if t in special_ids
                        )
                    ),
                    "ended_with_eos": stop < len(ids),
                }
                terminal_tokens = tuple(
                    dict.fromkeys(
                        [tokenizer.eos_token or "", tokenizer.pad_token or ""]
                        + [tokenizer.convert_ids_to_tokens(t) for t in eos_ids]
                        + [
                            t
                            for t in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>")
                            if t in tokenizer.all_special_tokens
                        ]
                    )
                )
                parser_flags = []
                if model_cfg["prompt_format"] == "chat" and getattr(
                    tokenizer, "response_template", None
                ):
                    parsed = tokenizer.parse_response(text, prefix=rendered_prefix)
                    content = parsed.get("content", "")
                    if isinstance(content, list):
                        content = "\n".join(
                            c.get("text", "") for c in content if c.get("type") == "text"
                        )
                    text = content or ""
                    diagnostics["parser_content_empty"] = not text.strip()
                    if (
                        parsed.get("reasoning_content")
                        or parsed.get("thinking")
                        or parsed.get("reasoning")
                    ):
                        parser_flags.append("reasoning_removed")
                response, flags = clean_response(
                    text, gen.get("identity_patterns", []), terminal_tokens
                )
                flags += parser_flags
                new_rows.append(
                    {
                        **row,
                        "model_id": model_id,
                        "response": response,
                        "input_tokens": int(tokens["attention_mask"][i].sum()),
                        "output_tokens": len(used),
                        "truncated": stop == len(ids) and len(ids) >= limit,
                        "quality_flags": flags,
                        "response_diagnostics": diagnostics,
                        "generation_seed": seed,
                        "generation_config": {**gen, "actual_max_new_tokens": limit},
                        "generation_fingerprint": fingerprint(metadata),
                    }
                )
            append_jsonl(path, new_rows)
    return {"model": model_id, "split": split, "rows": len(prompts)}
