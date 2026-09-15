import json
from pathlib import Path

import pytest

from teacher_attr.config import SPLITS
from teacher_attr.io import load_jsonl, write_jsonl


def test_long_prompt_cannot_remove_response(config, tiny_models):
    from teacher_attr.encoders import tokenize_pairs

    tokenizer = tiny_models
    tokens, audit = tokenize_pairs(
        tokenizer, [("item " * 1000, "response sentinel")], config["encoder"]
    )
    assert tokens["input_ids"].shape[1] <= 32
    assert audit[0]["prompt_tokens_retained"] == 8
    assert audit[0]["response_tokens_retained"] == 2
    assert tokenizer.convert_tokens_to_ids("sentinel") in tokens["input_ids"][0].tolist()
    with pytest.raises(ValueError, match="nonempty"):
        tokenize_pairs(tokenizer, [("prompt", "")], config["encoder"])


def test_response_only_does_not_include_prompt(config, tiny_models):
    from teacher_attr.encoders import tokenize_pairs

    cfg = {**config["encoder"], "input_mode": "response"}
    tokens, audit = tokenize_pairs(tiny_models, [("alpha " * 100, "beta")], cfg)
    assert audit[0]["prompt_tokens_retained"] == 0
    assert tiny_models.convert_tokens_to_ids("alpha") not in tokens["input_ids"][0].tolist()


def test_full_local_pipeline_and_resume(config, sources, tiny_models):
    from teacher_attr.evaluation import evaluate
    from teacher_attr.generation import generate, output_path, read_outputs
    from teacher_attr.pairs import build, load_pairs
    from teacher_attr.prompts import prepare
    from teacher_attr.training import train

    config["generation"]["temperature"] = 0.7
    prepare(config)
    root = Path(config["run_dir"])
    for role in ("teachers", "students"):
        for name in config[role]:
            for split in SPLITS:
                generate(config, role, name, split)
    name = next(iter(config["teachers"]))
    path = output_path(root, "teachers", name, "test")
    original = load_jsonl(path)
    assert all(r["response"] for r in original)
    write_jsonl(path, original[:1])
    generate(config, "teachers", name, "test")
    assert load_jsonl(path) == original
    manifest = build(config)
    assert manifest["audit"]["test"]["rows"] == 8
    assert build(config) == manifest
    assert len(load_pairs(config)["train"]) == 12
    checkpoint = train(config)["checkpoint"]
    summary = evaluate(config, checkpoint=checkpoint, frozen=True)
    metrics = json.loads((Path(summary["output"]) / "metrics.json").read_text())
    assert {"word_ngram", "char_ngram", "length_format", "frozen_probe", "trained_probe"} <= set(
        metrics["methods"]
    )
    assert "trained_classifier" in metrics["methods"]
    assert metrics["methods"]["trained_probe"]["set_level"]["all/1"]["num_sets"] == 8
    scores = load_jsonl(Path(summary["output"]) / "predictions.jsonl")
    assert len(scores) == 8
    assert (Path(summary["output"]) / "token_audit.json").exists()
    with pytest.raises(ValueError, match="already exists"):
        evaluate(config, name="evaluation")
    # Training compares data hashes when loading a checkpoint.
    import torch

    from teacher_attr.training import load_checkpoint

    payload = torch.load(checkpoint, weights_only=True)
    payload["pair_sha256"]["train"] = "changed"
    bad = Path(checkpoint).with_name("bad.pt")
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="different data"):
        load_checkpoint(bad, config, "cpu")
    changed = dict(config)
    changed["seed"] = 99
    with pytest.raises(ValueError, match="mismatch"):
        read_outputs(changed, "teachers", name, "test")
    write_jsonl(path, original[:1])
    with pytest.raises(ValueError, match="generation changed"):
        load_pairs(config)


def test_generation_refuses_silent_truncation(config, sources, tiny_models):
    from teacher_attr.generation import generate
    from teacher_attr.prompts import prepare

    config["generation"]["max_input_tokens"] = 1
    prepare(config)
    with pytest.raises(ValueError, match="never truncates"):
        generate(config, "teachers", "gpt2", "test")


@pytest.mark.parametrize("objective", ["classification", "contrastive"])
def test_loss_ablation_has_finite_gradients(config, tiny_models, objective):
    import torch

    from teacher_attr.encoders import AttributionEncoder, collate
    from teacher_attr.training import batch_forward

    enc = {**config["encoder"], "objective": objective}
    model = AttributionEncoder.pretrained(enc, 2)
    rows = [
        {
            "prompt": "item",
            "student_response": "alpha beta",
            "teacher_responses": {"a": "alpha", "b": "beta"},
            "true_teacher": t,
        }
        for t in ("a", "b")
    ]
    loss, scores, _ = batch_forward(
        model, collate(rows, tiny_models, enc, ["a", "b"]), "cpu", enc, 2
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert scores.shape == (2, 2)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
