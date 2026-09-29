import pytest

from teacher_attr.perplexity import response_tokens


class Tokenizer:
    bos_token_id = 0
    eos_token_id = 9

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return list(range(1, len(text.split()) + 1))


def test_single_token_is_scored_after_boundary():
    assert response_tokens(Tokenizer(), "Yes", 2, "train/s/p") == [0, 1]


def test_consistent_boundary_and_context_budget():
    tok = Tokenizer()
    tok.bos_token_id = None
    assert response_tokens(tok, "two words", 3, "train/s/p") == [9, 1, 2]
    with pytest.raises(ValueError, match="train/s/p.*needs 3"):
        response_tokens(tok, "two words", 2, "train/s/p")
    with pytest.raises(ValueError, match="empty"):
        response_tokens(tok, " ", 3, "train/s/p")
    tok.eos_token_id = None
    with pytest.raises(ValueError, match="BOS or EOS"):
        response_tokens(tok, "Yes", 3, "train/s/p")
