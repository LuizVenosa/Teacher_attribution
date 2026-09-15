import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from teacher_attr.config import load_config
from teacher_attr.io import load_jsonl, write_jsonl
from teacher_attr.research import POOLS, nested_ids, user_prompt


@pytest.fixture
def research_config(config, tiny_models, tmp_path):
    raw = yaml.safe_load((Path(__file__).parents[1] / "configs" / "research.yaml").read_text())
    raw["run_dir"] = str(tmp_path / "research")
    prototype = next(iter(config["teachers"].values()))
    raw["teachers"] = {f"teacher{i}": copy.deepcopy(prototype) for i in range(4)}
    raw["encoder"] = config["encoder"]
    raw["evaluation"] = {
        **config["evaluation"],
        "generic_encoder": {"hf_name": config["encoder"]["hf_name"], "revision": "main"},
        "shortcut_phrases": ["alpha", "beta"],
        "pos_model": "en_core_web_sm",
        "perplexity_control": True,
    }
    raw["generation"] = {
        **config["generation"],
        "temperature": 0.7,
        "batch_size": 2,
        "top_k": 0,
        "repetition_penalty": 1.0,
    }
    settings = raw["research"]
    settings["student"] = {
        "hf_name": prototype["hf_name"],
        "revision": "main",
        "prompt_format": "completion",
    }
    settings.update(amounts=[2, 4, 6], primary_amount=6, max_input_chars=80, scan_limit=100)
    settings["training"].update(
        precision="float32",
        max_length=64,
        microbatch=2,
        gradient_accumulation=2,
        epochs=1,
        gradient_checkpointing=False,
        log_every=1,
    )
    settings["sets"].update(hidden_dim=8, epochs=1, train_sets_per_student=2, batch_size=2)
    settings["quality"].update(max_truncated_rate=1, max_length_ratio=100)
    source = tmp_path / "instructions.jsonl"
    write_jsonl(
        source,
        [
            {
                "id": str(i),
                "messages": [
                    {"role": "user", "content": f"explain item {i}"},
                    {"role": "assistant", "content": "SECRET ANSWER"},
                ],
            }
            for i in range(100)
        ],
    )
    settings["sources"] = {
        "tulu3": {
            "path": str(source),
            "format": "messages",
            "task": "instruction",
            "counts": dict(zip(POOLS, [6, 2, 6, 4, 4], strict=True)),
        }
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path), path


def test_answer_free_inputs():
    row = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "SECRET"},
        ]
    }
    assert user_prompt("tulu3", row, {"format": "messages"}, 100) == "question"
    hh = {"chosen": "\n\nHuman: question\n\nAssistant: SECRET\n\nHuman: follow-up"}
    assert user_prompt("safety", hh, {"format": "human_assistant"}, 100) == "question"
    assert (
        user_prompt("tulu3", {"messages": row["messages"][1:]}, {"format": "messages"}, 100) is None
    )


def test_response_mask_and_overflow(tiny_models, config):
    from transformers import AutoTokenizer

    from teacher_attr.distillation import collate_sft, encode_example

    tokenizer = AutoTokenizer.from_pretrained(next(iter(config["teachers"].values()))["hf_name"])
    encoded = encode_example(tokenizer, "alpha", "beta response", 32)
    target = [t for t in encoded["labels"] if t != -100]
    assert tokenizer.convert_tokens_to_ids("beta") in target
    assert tokenizer.convert_tokens_to_ids("alpha") not in target
    batch = collate_sft(
        [encoded, encode_example(tokenizer, "alpha", "beta", 32)], tokenizer.pad_token_id
    )
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    with pytest.raises(ValueError, match="exceeds"):
        encode_example(tokenizer, "alpha " * 100, "beta", 32)


def test_no_reasoning_persisted():
    from teacher_attr.quality import clean_response

    answer, flags = clean_response("<think>private internal steps</think>Final answer", [])
    assert answer == "Final answer" and flags == ["reasoning_markup"]
    assert clean_response("private steps</think>Final", [])[0] == "Final"
    assert clean_response("<think>unfinished", [])[0] == ""
    assert clean_response("<|channel>thought\nprivate<channel|>Final", [])[0] == "Final"
    assert clean_response("<|channel>thought\n<channel|>Final", []) == ("Final", [])
    assert "identity_marker" in clean_response("I am Qwen", [r"\bqwen\b"])[1]


def test_research_pools_and_variants(research_config, tmp_path):
    from teacher_attr.experiments import create_variant
    from teacher_attr.prompts import prepare, verify_prompts

    cfg, path = research_config
    manifest = prepare(cfg)
    assert set(manifest["sha256"]) == set(POOLS)
    assert nested_ids(cfg, 2) == nested_ids(cfg, 6)[:2]
    root = Path(cfg["run_dir"])
    all_rows = [r for s in POOLS for r in load_jsonl(root / "prompts" / f"{s}.jsonl")]
    assert len({r["prompt_id"] for r in all_rows}) == len(all_rows)
    assert all("SECRET" not in r["prompt"] for r in all_rows)
    result = create_variant(str(path), "held_out_students", "", str(tmp_path / "heldout.yaml"))
    heldout = load_config(result["configuration"])
    assert heldout["protocol"] == "held_out_students"
    assert len(heldout["students"]) == 12
    assert all(len(s["splits"]) == 1 for s in heldout["students"].values())
    assert verify_prompts(Path(heldout["run_dir"]))["sha256"] == manifest["sha256"]


def test_deep_sets_permutation_invariance():
    import torch

    from teacher_attr.sets import DeepSets, set_scores

    torch.manual_seed(1)
    model = DeepSets(8, 4)
    x = torch.randn(2, 5, 8)
    teachers = torch.randn(2, 4, 5, 8)
    scores = set_scores(x, teachers, model)
    assert torch.allclose(scores, set_scores(x.flip(1), teachers.flip(2), model), atol=1e-6)
    scores.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_controlled_distillation_end_to_end(research_config):
    from teacher_attr.diagnostics import diagnose
    from teacher_attr.distillation import train_student, verify_student
    from teacher_attr.evaluation import evaluate
    from teacher_attr.generation import generate
    from teacher_attr.pairs import build
    from teacher_attr.prompts import prepare
    from teacher_attr.quality import quality_control
    from teacher_attr.reporting import figures
    from teacher_attr.sets import evaluate_sets, train_sets
    from teacher_attr.training import train

    cfg, config_path = research_config
    prepare(cfg)
    for teacher in cfg["teachers"]:
        for split in ("distill_train", "distill_val"):
            generate(cfg, "teachers", teacher, split)
    for split in ("distill_train", "distill_val"):
        assert quality_control(cfg, split)["passed"]
    for teacher in cfg["teachers"]:
        summary = train_student(cfg, teacher)
        assert summary["complete"] and summary["target_tokens_processed"] > 0
        assert np.isfinite(summary["history"][0]["validation_loss"])
    for student in cfg["students"]:
        assert verify_student(cfg, student)["complete"]
    for role in ("teachers", "students"):
        for name in cfg[role]:
            for split in ("train", "val", "test"):
                generate(cfg, role, name, split)
    build(cfg)
    assert len(diagnose(cfg)["students"]) == 4
    checkpoint = train(cfg)["checkpoint"]
    evaluation = evaluate(cfg, checkpoint=checkpoint, name="single")
    metrics_path = Path(evaluation["output"]) / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    assert metrics["methods"]["trained_cosine"]["chance_accuracy"] == 0.25
    assert "generic_cosine" in metrics["methods"]
    trained_sets = train_sets(cfg, checkpoint, str(metrics_path))
    sets = evaluate_sets(cfg, checkpoint, trained_sets["checkpoint"])
    assert "generic_mean" in sets["methods"]
    assert Path(figures(sets["output"])["data"]).exists()
    fusion = train(cfg, representation="fusion", baseline=str(metrics_path))["checkpoint"]
    from teacher_attr.training import load_checkpoint

    model, _, payload = load_checkpoint(fusion, cfg, "cpu")
    assert model.representation == "fusion" and payload["encoder"]["representation"] == "fusion"
    independent = Path(cfg["run_dir"]) / "independent.jsonl"
    write_jsonl(
        independent,
        [
            {
                "prompt": f"sentinel independent {i}",
                "response": "alpha beta",
                "split": "train" if i < 4 else "val",
            }
            for i in range(6)
        ],
    )
    teacher = next(iter(cfg["teachers"]))
    post = train_student(cfg, teacher, post_data=str(independent), level="light")
    assert post["additional_finetuning"]["train_rows"] == 4
    mismatched = independent.with_name("different_independent.jsonl")
    changed = load_jsonl(independent)
    changed[0]["response"] = "beta alpha"
    write_jsonl(mismatched, changed)
    with pytest.raises(ValueError, match="same independent data"):
        train_student(cfg, list(cfg["teachers"])[1], post_data=str(mismatched), level="light")
    transferred = evaluate(
        cfg, checkpoint=checkpoint, name="transfer", reference_config=str(config_path)
    )
    assert "trained_cosine" in transferred["methods"]


@pytest.mark.parametrize("representation", ["structure", "fusion"])
def test_latent_branch_gradients_and_gate(config, tiny_models, representation):
    import torch

    from teacher_attr.encoders import AttributionEncoder, collate
    from teacher_attr.training import batch_forward, train

    enc = {**config["encoder"], "representation": representation, "structure_layer": 1}
    model = AttributionEncoder.pretrained(enc, 2)
    rows = [
        {
            "prompt": "item",
            "student_response": "alpha",
            "teacher_responses": {"a": "alpha", "b": "beta"},
            "true_teacher": "a",
        }
    ]
    loss, _, _ = batch_forward(model, collate(rows, tiny_models, enc, ["a", "b"]), "cpu", enc, 2)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.structure[0].weight.grad is not None
    with pytest.raises(ValueError, match="verified"):
        train(config, representation=representation)


def test_paired_robustness_uses_original_ids(tmp_path):
    from teacher_attr.io import save_json
    from teacher_attr.reporting import compare

    for folder, changed in (("before", False), ("after", True)):
        root = tmp_path / folder
        save_json(root / "metrics.json", {"teacher_ids": ["a", "b"], "methods": {"m": {}}})
        write_jsonl(
            root / "predictions.jsonl",
            [
                {
                    "prompt_id": "new" if changed else "old",
                    "original_prompt_id": "old" if changed else None,
                    "true_teacher": "a",
                    "scores": {"m": [0.0, 1.0] if changed else [1.0, 0.0]},
                }
            ],
        )
    result = compare(
        str(tmp_path / "before"), str(tmp_path / "after"), str(tmp_path / "comparison.json"), 20, 1
    )
    assert result["m"]["prediction_consistency"] == 0
    assert result["m"]["accuracy_difference"] == -1
