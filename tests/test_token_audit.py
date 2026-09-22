import sys
from types import SimpleNamespace

from teacher_attr import token_audit


def test_audit_no_weights_and_complete_student_sequences(tmp_path, monkeypatch):
    class Tokenizer:
        model_max_length = 200
        eos_token_id = 1

        def encode(self, text, add_special_tokens=False):
            return list(range(len(text) + int(add_special_tokens)))

        def apply_chat_template(self, messages, **kwargs):
            return "CHAT:" + messages[0]["content"]

    calls = []

    def tokenizer_loader(name, **kwargs):
        calls.append((name, kwargs))
        return Tokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
            AutoConfig=SimpleNamespace(
                from_pretrained=lambda *a, **k: SimpleNamespace(
                    max_position_embeddings=200, is_encoder_decoder=False
                )
            ),
        ),
    )
    rows = [{"prompt_id": "one", "prompt": "hello", "task": "qa"}]
    monkeypatch.setattr(token_audit, "verify_prompts", lambda root: {"sha256": {"train": "abc"}})
    monkeypatch.setattr(token_audit, "load_jsonl", lambda path: rows)
    monkeypatch.setattr(token_audit, "provenance", lambda: {})
    monkeypatch.setattr(
        token_audit, "read_outputs", lambda *args: [{**rows[0], "response": "answer"}]
    )
    meta = tmp_path / "outputs/teachers/teacher/distill_train.meta.json"
    meta.parent.mkdir(parents=True)
    meta.write_text("{}")
    cfg = {
        "run_dir": str(tmp_path),
        "teachers": {"teacher": {"hf_name": "teacher", "revision": "pin", "prompt_format": "chat"}},
        "generation": {"max_input_tokens": 8, "max_new_tokens": 20},
        "research": {
            "student": {"hf_name": "student", "revision": "pin"},
            "training": {"max_length": 25},
        },
    }
    report = token_audit.audit_token_budgets(cfg)
    assert len(calls) == 2
    assert all(kwargs["revision"] == "pin" for _, kwargs in calls)
    assert report["models"]["teacher"]["splits"]["train"]["max_tokens"] == 10
    assert report["models"]["teacher"]["splits"]["train"]["over_budget"] == 1
    # Exact student prefix + response + EOS, using separate encoding as in SFT.
    expected = len("User:\nhello\n\nAssistant:\n") + len("answer") + 1
    sft = report["student_training"]["teacher"]
    assert sft["distill_train"]["max_tokens"] == expected
    assert sft["distill_train"]["over_budget"] == 1
    assert sft["distill_val"] == {"status": "not_generated"}
    assert report["weights_loaded"] is False
