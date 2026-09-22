import copy

import pytest

from teacher_attr.generation import validate_resume_metadata


def test_resume_allows_commit_change_without_mutating_manifest():
    stored = {
        "response_processing_version": 2,
        "resolved_revision": "pinned",
        "spec": {"seed": 13},
        "runtime": {"git_commit": "old", "packages": {"torch": "2.6"}, "gpu": "A100"},
    }
    original = copy.deepcopy(stored)
    current = copy.deepcopy(stored)
    current["runtime"]["git_commit"] = "launcher-fix"
    validate_resume_metadata(stored, current)
    assert stored == original
    for key, value in (
        ("response_processing_version", 3),
        ("resolved_revision", "different"),
        ("spec", {"seed": 42}),
        ("runtime", {"git_commit": "new", "packages": {"torch": "2.7"}, "gpu": "A100"}),
    ):
        changed = copy.deepcopy(current)
        changed[key] = value
        with pytest.raises(ValueError, match=key):
            validate_resume_metadata(stored, changed)
