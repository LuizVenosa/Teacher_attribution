from types import SimpleNamespace

import pytest

from teacher_attr.context import context_limit


def test_long_context_and_sentinel():
    assert (
        context_limit(
            SimpleNamespace(max_position_embeddings=131072),
            SimpleNamespace(model_max_length=10**30),
        )
        == 131072
    )
    assert (
        context_limit(
            SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=262144)),
            SimpleNamespace(model_max_length=131072),
        )
        == 131072
    )
    assert (
        context_limit(
            SimpleNamespace(max_position_embeddings=4096), SimpleNamespace(model_max_length=10**30)
        )
        == 4096
    )


def test_unknown_context_fails_instead_of_inventing_limit():
    with pytest.raises(ValueError, match="No explicit"):
        context_limit(SimpleNamespace(), SimpleNamespace(model_max_length=10**30))
