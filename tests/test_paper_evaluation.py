import copy
import json
from collections import Counter

import numpy as np
import pytest

from teacher_attr import paper_evaluation as paper
from teacher_attr.config import file_hash
from teacher_attr.io import save_json
from teacher_attr.paper_baselines import (
    bow_scores,
    classifier_scores,
    indicator_matrix,
    pos_documents,
    pos_vocabulary,
    teacher_examples,
)
from teacher_attr.paper_scoring import atomic_array, likelihood_spec, teacher_likelihood


@pytest.fixture
def paper_data(tmp_path, monkeypatch):
    cfg = {
        "run_dir": str(tmp_path),
        "seed": 13,
        "teachers": {t: {"hf_name": t, "revision": "a" * 40} for t in ("a", "b")},
        "generation": {"dtype": "bfloat16"},
        "research": {"training": {"max_length": 4096}},
        "evaluation": {
            "logreg_c": [0.1, 1],
            "pos_model": "fake_pos",
            "bertscore_model": "roberta-base",
            "bertscore_layers": 10,
            "set_sizes": [1, 2],
            "set_repeats": 2,
            "bootstrap_repeats": 20,
        },
    }
    splits = {}
    for split in ("train", "val", "test"):
        rows = []
        for dataset in ("one", "two"):
            refs = {"a": "red red apple", "b": "blue blue sky"}
            if dataset == "two":
                refs = {"b": refs["a"], "a": refs["b"]}
            for i in range(4):
                for t in ("a", "b"):
                    rows.append(
                        {
                            "prompt_id": f"{split}-{dataset}-{i}",
                            "student_id": f"s_{t}",
                            "true_teacher": t,
                            "source_dataset": dataset,
                            "student_response": refs[t] if split == "test" else "CANARY",
                            "teacher_responses": dict(refs),
                        }
                    )
        splits[split] = rows
    (tmp_path / "pairs").mkdir()
    for split, rows in splits.items():
        save_json(tmp_path / "pairs" / f"{split}.jsonl", rows)
    monkeypatch.setattr(paper, "initialize_run", lambda cfg: tmp_path)
    monkeypatch.setattr(paper, "load_pairs", lambda cfg: splits)
    return cfg, splits, tmp_path


def test_deduplicate_teachers_and_reject_conflicts(paper_data):
    cfg, splits, _ = paper_data
    examples = teacher_examples(splits["train"], list(cfg["teachers"]))
    assert len(examples) == 16  # Not 32: no repetition for each student.
    assert all("CANARY" not in r["response"] for r in examples)
    bad = copy.deepcopy(splits["train"])
    bad[1]["teacher_responses"]["a"] = "changed"
    with pytest.raises(ValueError, match="Conflicting"):
        teacher_examples(bad, ["a", "b"])


@pytest.mark.parametrize("method", ["word_ngram", "pos"])
def test_teacher_only_selection_ignores_student_train_labels(paper_data, method):
    from types import SimpleNamespace

    def documents(texts, batch_size):
        for text in texts:
            tags = ["NN", "VB", "DT", "NN"] if "red" in text else ["JJ", "NN", "VB", "NN"]
            yield SimpleNamespace(
                sents=[[SimpleNamespace(tag_=tag, is_space=False) for tag in tags]]
            )

    nlp = SimpleNamespace(pipe=documents)
    cfg, splits, _ = paper_data
    baseline, choices = classifier_scores(splits, ["a", "b"], cfg, method, "per_dataset", nlp)
    assert np.array_equal(baseline.argmax(1), [0, 1] * 8)
    assert choices["groups"]["one"]["training_rows"] == 8
    poisoned = copy.deepcopy(splits)
    for split in ("train", "val"):
        for row in poisoned[split]:
            row["student_response"] = "POISON STUDENT TEXT"
            row["true_teacher"] = "NOT A CANDIDATE"
    for row in poisoned["test"]:
        row["true_teacher"] = "NOT USED DURING FIT"
    changed, selection = classifier_scores(poisoned, ["a", "b"], cfg, method, "per_dataset", nlp)
    np.testing.assert_array_equal(changed, baseline)
    assert selection == choices


def test_pos_top_patterns_and_binary_features():
    docs = [Counter({"NN VB DT NN": 5, "DT NN VB JJ": 1}), Counter({"JJ NN VB NN": 4})]
    assert pos_vocabulary(docs, top_k=1) == ["NN VB DT NN"]
    vocabulary = pos_vocabulary(docs, top_k=2)
    assert set(vocabulary) == {"NN VB DT NN", "JJ NN VB NN"}
    features = indicator_matrix(docs, vocabulary)
    assert set(features.ravel()) == {0.0, 1.0}  # Indicators, not frequencies.
    assert indicator_matrix([Counter({"UNSEEN TEST PATTERN": 9})], vocabulary).sum() == 0


def test_pos_windows_do_not_cross_sentences():
    from types import SimpleNamespace

    def tokens(tags):
        return [SimpleNamespace(tag_=t, is_space=False) for t in tags]

    doc = SimpleNamespace(sents=[tokens(["NN", "VB"]), tokens(["DT", "NN", "VB", "JJ"])])
    nlp = SimpleNamespace(pipe=lambda texts, batch_size: [doc])
    assert pos_documents(nlp, ["ignored"]) == [Counter({"DT NN VB JJ": 1})]


def test_bow_matches_candidate_responses_without_student_training(paper_data):
    _, splits, _ = paper_data
    scores = bow_scores(splits, ["a", "b"])
    np.testing.assert_array_equal(scores.argmax(1), [0, 1] * 8)


def test_evaluation_resume_and_specification_guard(paper_data):
    cfg, _, root = paper_data
    result = paper.evaluate_paper(cfg, methods=["word_ngram", "bow"])
    assert result["methods"] == {"word_ngram": 1.0, "bow": 1.0}
    report = json.loads((root / "evaluations/paper_v1/metrics.json").read_text())
    assert report["protocol"] == "teacher_train_teacher_validation_student_test"
    assert report["num_test_prompts"] == 8
    assert set(report["methods"]["word_ngram"]["by_dataset"]) == {"one", "two"}
    assert paper.evaluate_paper(cfg, collect=True)["methods"] == result["methods"]
    with pytest.raises(ValueError, match="specification changed"):
        paper.evaluate_paper(cfg, scope="pooled", collect=True)


def test_score_tampering_is_rejected(tmp_path):
    scores = np.array([[1.0, 0.0]])
    paper.save_scores(tmp_path, "bow", scores, {})
    atomic_array(tmp_path / "bow.npy", scores[:, ::-1])
    with pytest.raises(ValueError, match="changed"):
        paper.load_scores(tmp_path, "bow", scores.shape)


def test_completed_likelihood_avoids_model_download_and_checks_inputs(paper_data):
    cfg, splits, out = paper_data
    values = np.ones((len(splits["test"]), 2))
    path = out / "likelihood_a.npy"
    atomic_array(path, values)
    save_json(
        out / "likelihood_a.json",
        {
            "complete": True,
            "spec": likelihood_spec(cfg, splits["test"], "a"),
            "sha256": file_hash(path),
        },
    )
    scores, record = teacher_likelihood(cfg, splits["test"], "a", out)
    np.testing.assert_array_equal(scores, values)
    assert record["complete"]
    changed = copy.deepcopy(splits["test"])
    changed[0]["student_response"] = "different"
    with pytest.raises(ValueError, match="inputs/settings changed"):
        teacher_likelihood(cfg, changed, "a", out)


def test_likelihood_requires_pinned_candidate(paper_data):
    cfg, splits, _ = paper_data
    cfg["teachers"]["a"]["revision"] = "main"
    with pytest.raises(ValueError, match="Pin candidate"):
        likelihood_spec(cfg, splits["test"], "a")


def test_primary_comparison_refuses_different_pairs(paper_data):
    cfg, splits, root = paper_data
    save_json(root / "metrics.json", {"pair_sha256": {}, "teacher_ids": ["a", "b"]})
    with pytest.raises(ValueError, match="different pairs"):
        paper.compare_primary(root, {"train": "wrong"}, splits["test"], ["a", "b"], {}, cfg)


def test_primary_comparison_checks_alignment_and_scores(paper_data):
    from teacher_attr.io import write_jsonl

    cfg, splits, root = paper_data
    hashes = {split: file_hash(root / "pairs" / f"{split}.jsonl") for split in splits}
    save_json(
        root / "metrics.json",
        {"pair_sha256": hashes, "teacher_ids": ["a", "b"], "methods": {"trained_probe": {}}},
    )
    rows = splits["test"]
    scores = bow_scores(splits, ["a", "b"])
    predictions = [
        {**row, "scores": {"trained_probe": scores[i].tolist()}} for i, row in enumerate(rows)
    ]
    path = root / "predictions.jsonl"
    write_jsonl(path, predictions)
    result = paper.compare_primary(root, hashes, rows, ["a", "b"], {"bow": (scores, {})}, cfg)
    assert result["differences"]["bow"]["trained_probe"]["accuracy_difference"] == 0
    write_jsonl(path, predictions[::-1])
    with pytest.raises(ValueError, match="not aligned"):
        paper.compare_primary(root, hashes, rows, ["a", "b"], {"bow": (scores, {})}, cfg)
    predictions[0]["scores"]["trained_probe"] = [1.0]
    write_jsonl(path, predictions)
    with pytest.raises(ValueError):
        paper.compare_primary(root, hashes, rows, ["a", "b"], {"bow": (scores, {})}, cfg)


def test_likelihood_collection_waits_for_every_candidate(paper_data):
    cfg, splits, root = paper_data
    paper.evaluate_paper(cfg, methods=["bow"])
    out = root / "evaluations/paper_v1"
    for candidate in cfg["teachers"]:
        values = np.array(
            [[-1, -2] if row["true_teacher"] == candidate else [-4, -8] for row in splits["test"]],
            dtype=float,
        )
        path = out / f"likelihood_{candidate}.npy"
        atomic_array(path, values)
        save_json(
            out / f"likelihood_{candidate}.json",
            {
                "complete": True,
                "spec": likelihood_spec(cfg, splits["test"], candidate),
                "sha256": file_hash(path),
            },
        )
        result = paper.evaluate_paper(cfg, collect=True)
        if candidate == "a":
            assert result["pending_likelihood_teachers"] == ["b"]
            assert "teacher_likelihood_token" not in result["methods"]
    assert result["methods"]["teacher_likelihood_token"] == 1.0
    assert result["methods"]["teacher_likelihood_byte"] == 1.0


def test_bertscore_pins_model_checks_lengths_and_limits_batch(paper_data, monkeypatch):
    import sys
    from types import SimpleNamespace

    from teacher_attr.paper_scoring import bertscore_matching

    cfg, splits, out = paper_data
    resolved = []

    class Tokenizer:
        model_max_length = 512

        def encode(self, text, add_special_tokens, truncation, **kwargs):
            assert truncation is False
            assert kwargs == {"add_prefix_space": True}
            return [0] * (len(text.split()) + 2)

    class Scorer:
        def __init__(self, **kwargs):
            assert kwargs["num_layers"] == 10
            assert kwargs["idf"] is False

        def score(self, candidates, references, batch_size):
            assert batch_size == 8
            assert len(candidates) == len(references)
            values = SimpleNamespace(
                cpu=lambda: SimpleNamespace(numpy=lambda: np.ones(len(candidates)))
            )
            return None, None, values

    def model_info(name):
        resolved.append(name)
        return SimpleNamespace(sha="f" * 40)

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    monkeypatch.setitem(sys.modules, "bert_score", SimpleNamespace(BERTScorer=Scorer))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=lambda: SimpleNamespace(model_info=model_info),
            snapshot_download=lambda *a, **kw: str(out),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            RobertaTokenizer=Tokenizer,
            GPT2Tokenizer=type("GPT2Tokenizer", (), {}),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer()),
            AutoConfig=SimpleNamespace(
                from_pretrained=lambda *a, **kw: SimpleNamespace(max_position_embeddings=16)
            ),
        ),
    )
    scores, record = bertscore_matching(cfg, splits["test"], ["a", "b"], out)
    assert scores.shape == (16, 2)
    assert record["revision"] == "f" * 40
    bertscore_matching(cfg, splits["test"], ["a", "b"], out)
    assert resolved == ["roberta-base"]  # Reuse frozen revision, no moving-head resolution.
    bad = copy.deepcopy(splits["test"])
    bad[0]["student_response"] = "long " * 20
    with pytest.raises(ValueError, match="No truncation"):
        bertscore_matching(cfg, bad, ["a", "b"], out)
