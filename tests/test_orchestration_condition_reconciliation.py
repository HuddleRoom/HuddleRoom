import pytest

from huddleroom.services.orchestration_condition import (
    CONDITIONS,
    ConditionError,
    is_reconcilable,
    validate_condition,
)


@pytest.mark.parametrize("condition", sorted(CONDITIONS))
def test_valid_condition_is_normalized(condition):
    assert validate_condition({"condition": condition, "subject": " t1 "}) == {
        "condition": condition,
        "subject": "t1",
    }


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "session_running",
        {"condition": "whenever", "subject": "t1"},
        {"condition": "session_running"},
        {"condition": "plan_missing", "subject": "  "},
        {"condition": "gate_unverified", "subject": 3},
    ],
)
def test_invalid_condition_rejected(raw):
    with pytest.raises(ConditionError):
        validate_condition(raw)
    assert not is_reconcilable(raw)


@pytest.mark.parametrize("kind", ["budget", "permission", "provider", "Owner_Decision"])
def test_owner_decisions_never_reconcilable(kind):
    raw = {"condition": "session_running", "subject": "t1", "kind": kind}
    assert not is_reconcilable(raw)


@pytest.mark.parametrize("subject", ["budget", " Permission ", "provider"])
def test_owner_decision_subject_rejected(subject):
    with pytest.raises(ConditionError):
        validate_condition({"condition": "session_running", "subject": subject})


@pytest.mark.parametrize("kw", [{"action_kind": "budget"}, {"category": "permission_required"}])
def test_reconcile_time_action_context_blocks_owner_decisions(kw):
    stored = validate_condition({"condition": "valid_result_missing", "subject": "t1"})
    assert is_reconcilable(stored)
    assert not is_reconcilable(stored, **kw)
    assert is_reconcilable(stored, action_kind="record_warning", category="other")
