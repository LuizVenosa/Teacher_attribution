"""Materialize explicit experiment variants; never launch an implicit GPU sweep."""

from __future__ import annotations

import copy
import json
import shlex
from pathlib import Path

from teacher_attr.config import file_hash, initialize_run, load_config
from teacher_attr.io import load_jsonl, load_yaml, save_json, save_yaml, write_jsonl
from teacher_attr.prompts import audit_splits, text_hash, verify_prompts


def preflight(cfg: dict, cache_dir: str | None = None) -> dict:
    """Check model/config/tokenizer access without downloading model weights."""
    import transformers
    from transformers import AutoConfig, AutoTokenizer

    from teacher_attr.research import provenance

    results = {}
    models = {
        **cfg["teachers"],
        "student_base": cfg["research"]["student"],
        "encoder": cfg["encoder"],
        "generic_encoder": cfg["evaluation"]["generic_encoder"],
    }
    for name, model in models.items():
        try:
            kwargs = {
                "revision": model["revision"],
                "trust_remote_code": False,
                "cache_dir": cache_dir,
            }
            config = AutoConfig.from_pretrained(model["hf_name"], **kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model["hf_name"], **kwargs)
            if name in cfg["teachers"]:
                cls = getattr(transformers, model["model_class"])
                if type(config) not in cls._model_mapping:
                    raise ValueError("Configured model loader does not support this architecture")
                if model["prompt_format"] == "chat":
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": "Say hello."}],
                        tokenize=True,
                        add_generation_prompt=True,
                        **model.get("chat_kwargs", {}),
                    )
            results[name] = {
                "accessible": True,
                "model_type": config.model_type,
                "resolved_revision": getattr(config, "_commit_hash", None),
                "tokenizer_class": type(tokenizer).__name__,
            }
        except (OSError, ValueError, ImportError, AttributeError) as exc:
            results[name] = {"accessible": False, "error": str(exc)}
    return {
        "models": results,
        "passed": all(r["accessible"] for r in results.values()),
        "environment": provenance(),
        "weights_downloaded": False,
    }


def pin_config(path: str, output: str) -> dict:
    from huggingface_hub import HfApi

    path, output = Path(path).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Pinned configuration already exists")
    cfg = load_yaml(path)
    cfg["run_dir"] = str((path.parent / cfg["run_dir"]).resolve())
    api = HfApi()
    models = [
        *cfg["teachers"].values(),
        cfg["encoder"],
        cfg["research"]["student"],
        cfg["evaluation"]["generic_encoder"],
    ]
    for model in models:
        if not Path(model["hf_name"]).is_dir():
            model["revision"] = api.model_info(model["hf_name"], revision=model["revision"]).sha
    for source in cfg["research"]["sources"].values():
        if source.get("path"):
            source["path"] = str((path.parent / source["path"]).resolve())
        else:
            source["revision"] = api.dataset_info(
                source["hf_name"], revision=source["revision"]
            ).sha
    save_yaml(output, cfg)
    return {
        "configuration": str(output),
        "status": "Remote revisions resolved; no model weights downloaded",
    }


def create_variant(config_path: str, kind: str, value: str, output: str) -> dict:
    cfg = load_config(config_path)
    parent = Path(cfg["run_dir"])
    manifest = verify_prompts(parent)
    raw = copy.deepcopy(cfg)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Variant configuration already exists")
    raw["run_dir"] = str(output.parent / output.stem)
    root = Path(raw["run_dir"])
    if root.exists():
        raise ValueError("Variant run directory already exists")
    research = raw["research"]
    research["student_root"] = str(parent)
    research["variant"] = {
        "kind": kind,
        "value": value,
        "parent_run": str(parent),
        "parent_prompts_sha256": manifest["sha256"],
    }
    if kind == "amount":
        research["primary_amount"] = int(value)
    elif kind == "seed":
        research["primary_seed"] = int(value)
    elif kind == "held_out_students":
        if len(research["student_seeds"]) < 3:
            raise ValueError("At least three student seeds are required")
        research["split_seeds"] = dict(
            zip(("train", "val", "test"), research["student_seeds"][:3], strict=True)
        )
        raw["protocol"] = "held_out_students"
    elif kind == "decoding":
        if float(value) < 0:
            raise ValueError("Temperature cannot be negative")
        raw["generation"]["student_temperature"] = float(value)
    elif kind == "extra_ft":
        if value not in research["additional_ft_epochs"]:
            raise ValueError("Unknown additional FT level")
        research["student_suffix"] = f"_ft_{value}"
    elif kind == "natural":
        raw["generation"]["controlled_length"] = False
        research.pop("student_root", None)
    elif kind not in {"cross_task", "paraphrase"}:
        raise ValueError("Unsupported variant")
    pools = {s: load_jsonl(parent / "prompts" / f"{s}.jsonl") for s in manifest["sha256"]}
    if kind == "cross_task":
        for split in ("train", "val", "test"):
            pools[split] = [r for r in pools[split] if (r["task"] == value) == (split == "test")]
        if any(not pools[s] for s in ("train", "val", "test")):
            raise ValueError("Cross-task variant has empty attribution pools")
    if kind == "paraphrase":
        entries = load_jsonl(Path(value).resolve())
        mapping = {r["original_prompt_id"]: r["prompt"] for r in entries}
        if len(mapping) != len(entries) or set(mapping) != {r["prompt_id"] for r in pools["test"]}:
            raise ValueError("Paraphrases must cover every test prompt exactly once")
        for row in pools["test"]:
            original = row["prompt_id"]
            row.update(
                prompt=mapping[original],
                prompt_id=text_hash(mapping[original]),
                original_prompt_id=original,
            )
        research["variant"]["paraphrases_sha256"] = file_hash(Path(value).resolve())
    audit = audit_splits(pools)
    # Save config only after all validation succeeds.
    save_yaml(output, raw)
    resolved = load_config(output)
    initialize_run(resolved)
    hashes = {}
    for split, rows in pools.items():
        target = root / "prompts" / f"{split}.jsonl"
        write_jsonl(target, rows)
        hashes[split] = file_hash(target)
    subsets = json.loads((parent / "prompts" / "nested_subsets.json").read_text())
    save_json(root / "prompts" / "nested_subsets.json", subsets)
    save_json(
        root / "prompts" / "manifest.json",
        {
            **manifest,
            "audit": audit,
            "sha256": hashes,
            "nested_subsets_sha256": file_hash(root / "prompts" / "nested_subsets.json"),
            "variant": research["variant"],
        },
    )
    return {
        "configuration": str(output),
        "run_dir": str(root),
        "next": (
            "Generate aligned outputs and build pairs in the new run. "
            "Natural variants require new student distillation."
        ),
    }


def experiment_plan(cfg: dict, path: str) -> dict:
    root = initialize_run(cfg)
    command = f"teacher-attr --config {shlex.quote(str(Path(path).resolve()))}"
    phases = [
        f"{command} prepare",
        f"{command} generate --role teachers --split distill_train",
        f"{command} generate --role teachers --split distill_val",
        f"{command} qc --split distill_train",
        f"{command} qc --split distill_val",
    ]
    phases += [f"{command} distill --teacher {teacher}" for teacher in cfg["teachers"]]
    phases += [
        f"{command} generate",
        f"{command} build",
        f"{command} audit",
        f"{command} diagnose",
        f"{command} evaluate --name baselines",
        f"{command} train",
        f"{command} evaluate --checkpoint CHECKPOINT --name single",
        f"{command} evaluate-sets --checkpoint CHECKPOINT --name mean",
        f"{command} train-sets --checkpoint CHECKPOINT --baseline SINGLE_METRICS_JSON",
        f"{command} evaluate-sets --checkpoint CHECKPOINT --set-checkpoint SET_CHECKPOINT",
    ]
    extensions = [
        f"{command} distill --teacher {teacher} --seed {seed} --amount {amount}"
        for amount in cfg["research"]["amounts"]
        for seed in cfg["research"]["student_seeds"]
        for teacher in cfg["teachers"]
        if (amount, seed) != (cfg["research"]["primary_amount"], cfg["research"]["primary_seed"])
    ]
    result = {
        "primary_sequential_commands": phases,
        "optional_student_grid": extensions,
        "latent_structure_gate": "Verify single-response baselines before extensions",
        "executed": False,
    }
    save_json(root / "execution_plan.json", result)
    return result
