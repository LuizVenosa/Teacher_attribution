from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

from teacher_attr.config import file_hash, fingerprint, initialize_run
from teacher_attr.io import append_jsonl, load_jsonl, save_json
from teacher_attr.prompts import verify_prompts


def output_path(root: Path, role: str, model_id: str, split: str) -> Path:
    return root / "outputs" / role / model_id / f"{split}.jsonl"


def render_prompt(tokenizer, prompt: str, mode: str) -> str:
    if mode == "chat":
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
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
    rendered = [render_prompt(tokenizer, p["prompt"], model_cfg["prompt_format"]) for p in prompts]
    limits = [getattr(config, k, None) for k in ("max_position_embeddings", "n_positions")]
    limits += [tokenizer.model_max_length]
    context = min((n for n in limits if isinstance(n, int) and 0 < n < 100000), default=2048)
    budget = min(
        gen["max_input_tokens"],
        context - (0 if config.is_encoder_decoder else gen["max_new_tokens"]),
    )
    # Fail rather than silently give different teachers different article prefixes.
    lengths = [len(tokenizer.encode(p, add_special_tokens=True)) for p in rendered]
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
    model = cls.from_pretrained(model_cfg["hf_name"], torch_dtype=dtype, **kwargs).to(device)
    model.eval()
    metadata = {
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
    if meta_path.exists():
        if json.loads(meta_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Resolved model revision changed; use a pinned revision and new run")
    else:
        save_json(meta_path, metadata)
    sampling = {"do_sample": gen["temperature"] > 0}
    if sampling["do_sample"]:
        sampling.update(temperature=gen["temperature"], top_p=gen["top_p"])
    for row, text in zip(prompts, rendered, strict=True):
        if row["prompt_id"] in done:
            continue
        seed = int(fingerprint([cfg["seed"], role, model_id, row["prompt_id"]])[:8], 16)
        set_seed(seed)
        tokens = tokenizer(text, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(
                **tokens,
                max_new_tokens=gen["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id,
                **sampling,
            )[0]
        if not config.is_encoder_decoder:
            output = output[tokens["input_ids"].shape[1] :]
        response = tokenizer.decode(output, skip_special_tokens=True).strip()
        append_jsonl(
            path,
            [
                {
                    **row,
                    "model_id": model_id,
                    "response": response,
                    "input_tokens": tokens["input_ids"].shape[1],
                    "output_tokens": len(output),
                    "generation_seed": seed,
                    "generation_fingerprint": fingerprint(metadata),
                }
            ],
        )
    return {"model": model_id, "split": split, "rows": len(prompts)}
