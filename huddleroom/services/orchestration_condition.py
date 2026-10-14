"""Typed conditions that make a warning or question auto-resolvable."""

from __future__ import annotations

SESSION_RUNNING = "session_running"
VALID_RESULT_MISSING = "valid_result_missing"
PLAN_MISSING = "plan_missing"
GATE_UNVERIFIED = "gate_unverified"

CONDITIONS = frozenset({SESSION_RUNNING, VALID_RESULT_MISSING, PLAN_MISSING, GATE_UNVERIFIED})

# Owner decisions are never auto-cancelled, whatever condition they declare.
OWNER_DECISION_SUBJECTS = frozenset(
    {"budget", "permission", "provider", "owner_decision", "budget_exhausted", "permission_required"}
)


class ConditionError(ValueError):
    pass


def validate_condition(raw: object) -> dict[str, str]:
    """Return the normalized ``{"condition", "subject"}`` or raise ConditionError."""
    if not isinstance(raw, dict):
        raise ConditionError("condition must be an object")
    condition, subject = raw.get("condition"), raw.get("subject")
    if condition not in CONDITIONS:
        raise ConditionError(f"unknown condition {condition!r}")
    if not isinstance(subject, str) or not subject.strip():
        raise ConditionError("condition requires a non-empty subject")
    for key in ("kind", "category", "owner_subject"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip().lower() in OWNER_DECISION_SUBJECTS:
            raise ConditionError(f"owner decision ({value}) is never reconcilable")
    return {"condition": condition, "subject": subject.strip()}


def is_reconcilable(raw: object) -> bool:
    try:
        validate_condition(raw)
    except ConditionError:
        return False
    return True
