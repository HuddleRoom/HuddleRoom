"""Unit tests for _apply_blocker_floor deterministic safety guard."""

import pytest

from huddleroom.services.orchestration_effectiveness_review import (
    _apply_blocker_floor,
    CheckResult,
)


def test_blocker_floor_overrides_illegal_continue_disposition():
    """When disposition is 'continue' but any check failed, override to 'revise'."""
    checks = [
        CheckResult(name="check_1", passed=True, detail="This one passed"),
        CheckResult(name="check_2", passed=False, detail="This one failed"),
        CheckResult(name="check_3", passed=True, detail="This one passed"),
    ]
    assert _apply_blocker_floor("continue", checks) == "revise"


def test_blocker_floor_leaves_continue_when_all_checks_pass():
    """When disposition is 'continue' and all checks pass, stay 'continue'."""
    checks = [
        CheckResult(name="check_1", passed=True, detail="Pass"),
        CheckResult(name="check_2", passed=True, detail="Pass"),
        CheckResult(name="check_3", passed=True, detail="Pass"),
    ]
    assert _apply_blocker_floor("continue", checks) == "continue"


def test_blocker_floor_never_touches_non_continue():
    """Other dispositions are never overridden, even with failed checks."""
    checks_with_failure = [
        CheckResult(name="check_1", passed=True, detail="Pass"),
        CheckResult(name="check_2", passed=False, detail="Fail"),
    ]

    # Test each non-continue disposition
    for disposition in ["revise", "split", "pause"]:
        result = _apply_blocker_floor(disposition, checks_with_failure)
        assert result == disposition, f"Disposition '{disposition}' should not be changed"


def test_blocker_floor_continue_with_empty_checks():
    """When 'continue' and no checks exist, nothing failed so stay 'continue'."""
    assert _apply_blocker_floor("continue", []) == "continue"
