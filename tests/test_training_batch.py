import pytest

from teacher_attr.distillation import training_batch_settings


def test_microbatch_preserves_effective_batch_and_source():
    settings = {"microbatch": 4, "gradient_accumulation": 8}
    assert training_batch_settings(settings, 1) == {"microbatch": 1, "gradient_accumulation": 32}
    assert settings == {"microbatch": 4, "gradient_accumulation": 8}
    assert training_batch_settings(settings) == settings
    for value in (0, -1, 3, 64):
        with pytest.raises(ValueError):
            training_batch_settings(settings, value)
