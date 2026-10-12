"""BERTScore matching and resumable per-candidate teacher likelihood."""

import gc
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from teacher_attr.config import file_hash, fingerprint
from teacher_attr.context import context_limit
from teacher_attr.io import save_json
from teacher_attr.perplexity import response_tokens


def atomic_array(path, values):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


def likelihood_spec(cfg, rows, teacher):
    model = cfg["teachers"][teacher]
    revision = model["revision"]
    if revision != "local" and (
        len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision)
    ):
        raise ValueError("Pin candidate-teacher revisions before likelihood evaluation")
    return {
        "version": 1,
        "teacher": teacher,
        "model": model,
        "rows_sha256": fingerprint(
            [[r["prompt_id"], r["student_id"], r["student_response"]] for r in rows]
        ),
        "conditioning": "response-only, BOS (EOS fallback), no appended EOS",
        "dtype": cfg["generation"]["dtype"],
        "max_length": cfg["research"]["training"]["max_length"],
        "columns": ["negative_mean_token_nll", "negative_nll_per_utf8_byte"],
    }


def teacher_likelihood(cfg, rows, teacher, out):
    """One teacher per invocation; persist progress without storing weights in the run."""
    spec = likelihood_spec(cfg, rows, teacher)
    path = out / f"likelihood_{teacher}.npy"
    manifest = out / f"likelihood_{teacher}.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["spec"] != spec:
            raise ValueError("Likelihood inputs/settings changed; use a new evaluation name")
    else:
        if path.exists():
            raise ValueError("Likelihood progress has no manifest")
        saved = {"spec": spec, "complete": False}
        save_json(manifest, saved)
    values = np.load(path, allow_pickle=False) if path.exists() else np.full((len(rows), 2), np.nan)
    if values.shape != (len(rows), 2) or np.isinf(values).any():
        raise ValueError("Invalid likelihood progress")
    if saved["complete"]:
        if file_hash(path) != saved["sha256"] or not np.isfinite(values).all():
            raise ValueError("Completed likelihood scores changed")
        return values, saved

    import torch
    import transformers
    from torch.nn import functional as F

    if not torch.cuda.is_available():
        raise ValueError("Candidate-teacher likelihood requires a GPU allocation")
    environment = {
        p: importlib.metadata.version(p) for p in ("torch", "transformers", "tokenizers")
    }
    if saved.get("environment") not in (None, environment):
        raise ValueError("Likelihood runtime packages changed; choose a new evaluation name")
    saved["environment"] = environment
    model_cfg = cfg["teachers"][teacher]
    kwargs = {"revision": model_cfg["revision"], "trust_remote_code": False}
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_cfg["hf_name"], **kwargs)
    cls = getattr(transformers, model_cfg.get("model_class", "AutoModelForCausalLM"))
    model = (
        cls.from_pretrained(
            model_cfg["hf_name"], torch_dtype=getattr(torch, cfg["generation"]["dtype"]), **kwargs
        )
        .to("cuda")
        .eval()
    )
    if getattr(model.config, "is_encoder_decoder", False):
        raise ValueError("Teacher likelihood currently supports causal text scoring only")
    limit = min(spec["max_length"], context_limit(model.config, tokenizer))
    resolved = getattr(model.config, "_commit_hash", None)
    if resolved and model_cfg["revision"] != "local" and resolved != model_cfg["revision"]:
        raise ValueError("Candidate teacher resolved to a different revision")
    saved["resolved_revision"] = resolved
    saved["boundary_token_id"] = (
        tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    )
    saved["context_budget"] = limit
    save_json(manifest, saved)
    try:
        with torch.inference_mode():
            for index, row in enumerate(rows):
                if np.isfinite(values[index]).all():
                    continue
                ids = response_tokens(
                    tokenizer,
                    row["student_response"],
                    limit,
                    f"{teacher}/{row['student_id']}/{row['prompt_id']}",
                )
                inputs = torch.tensor([ids], dtype=torch.long, device="cuda")
                # One response per forward; only loss chunks are cast to FP32.
                logits = model(input_ids=inputs, use_cache=False).logits
                total = 0.0
                for start in range(0, len(ids) - 1, 64):
                    end = min(start + 64, len(ids) - 1)
                    loss = F.cross_entropy(
                        logits[0, start:end].float(),
                        inputs[0, start + 1 : end + 1],
                        reduction="sum",
                    )
                    total += float(loss)
                if not np.isfinite(total):
                    raise ValueError(f"Nonfinite teacher likelihood at row {index}")
                values[index] = [
                    -total / (len(ids) - 1),
                    -total / len(row["student_response"].encode("utf-8")),
                ]
                del logits, inputs
                if (index + 1) % 25 == 0:
                    atomic_array(path, values)
                    print(f"teacher_likelihood={teacher} rows={index + 1}/{len(rows)}", flush=True)
    finally:
        atomic_array(path, values)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    saved.update(complete=True, sha256=file_hash(path))
    save_json(manifest, saved)
    return values, saved


def bertscore_matching(cfg, rows, teachers, out, overflow="error"):
    if overflow not in ("error", "truncate"):
        raise ValueError("Unknown BERTScore overflow policy")
    import torch
    from bert_score import BERTScorer
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoConfig, AutoTokenizer, GPT2Tokenizer, RobertaTokenizer

    model_name = cfg["evaluation"]["bertscore_model"]
    layers = cfg["evaluation"]["bertscore_layers"]
    model_manifest = out / "bertscore_model.json"
    if model_manifest.exists():
        record = json.loads(model_manifest.read_text())
        if record["model"] != model_name or record["layers"] != layers:
            raise ValueError("BERTScore configuration changed")
        if record.get("overflow_policy", overflow) != overflow:
            raise ValueError("BERTScore overflow policy is frozen; choose a new evaluation name")
    else:
        revision = HfApi().model_info(model_name).sha
        record = {"model": model_name, "revision": revision, "layers": layers}
        save_json(model_manifest, record)
    local = snapshot_download(
        model_name,
        revision=record["revision"],
        allow_patterns=["*.json", "*.safetensors", "vocab.*", "merges.txt", "*.model"],
    )
    tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True, use_fast=False)
    config = AutoConfig.from_pretrained(local, local_files_only=True)
    # Match BERTScore's tokenization, auditing full lengths before any clipping.
    limit = min(512, context_limit(config, tokenizer))
    prefix = (
        {"add_prefix_space": True}
        if isinstance(tokenizer, (GPT2Tokenizer, RobertaTokenizer))
        else {}
    )
    overlong, longest = [], 0
    for index, row in enumerate(rows):
        texts = [("student", row["student_response"])] + [
            (f"teacher:{teacher}", row["teacher_responses"][teacher]) for teacher in teachers
        ]
        for source, text in texts:
            length = len(
                tokenizer.encode(text.strip(), add_special_tokens=True, truncation=False, **prefix)
            )
            longest = max(longest, length)
            if length > limit:
                overlong.append(
                    {
                        "row_index": index,
                        "prompt_id": row["prompt_id"],
                        "student_id": row["student_id"],
                        "source": source,
                        "original_tokens": length,
                        "tokens_if_truncated": limit,
                    }
                )
    audit = {
        "overflow_policy": overflow,
        "max_tokens": limit,
        "max_observed_tokens": longest,
        "overflow_text_occurrences": len(overlong),
        "affected_test_rows": len({r["row_index"] for r in overlong}),
        "counting": (
            "Per test row, one student text and one reference per candidate; references may repeat"
        ),
        "overlong": overlong,
    }
    save_json(out / "bertscore_token_audit.json", audit)
    if overlong and overflow == "error":
        first = overlong[0]
        raise ValueError(
            f"BERTScore row {first['row_index']}/{first['prompt_id']} needs "
            f"{first['original_tokens']} tokens; limit {limit}. No truncation applied. "
            "Use --bertscore-overflow truncate for an explicitly audited truncated baseline."
        )
    record.update(overflow_policy=overflow, truncation_side="right", max_tokens=limit)
    # Freeze only once the length audit passes. A failed strict audit can be retried explicitly.
    save_json(model_manifest, record)
    scorer = BERTScorer(
        model_type=str(Path(local)),
        num_layers=layers,
        idf=False,
        rescale_with_baseline=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=8,
        use_fast_tokenizer=False,
    )
    # BERTScore's sent_encode reads this tokenizer's limit and truncates natively.
    # Keep it equal to the audited budget, including special tokens.
    scorer._tokenizer.model_max_length = limit
    scorer._tokenizer.truncation_side = "right"
    scores = []
    for teacher in teachers:
        _, _, f1 = scorer.score(
            [r["student_response"] for r in rows],
            [r["teacher_responses"][teacher] for r in rows],
            batch_size=8,
        )
        scores.append(f1.cpu().numpy())
    record.update(
        idf=False,
        rescale_with_baseline=False,
        statistic="F1",
        overflow_text_occurrences=len(overlong),
        affected_test_rows=audit["affected_test_rows"],
        token_audit_sha256=file_hash(out / "bertscore_token_audit.json"),
    )
    return np.column_stack(scores), record
