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
    subject = subject.strip()
    values = [subject, *(raw.get(k) for k in ("kind", "category", "owner_subject"))]
    for value in values:
        if isinstance(value, str) and value.strip().lower() in OWNER_DECISION_SUBJECTS:
            raise ConditionError(f"owner decision ({value}) is never reconcilable")
    return {"condition": condition, "subject": subject}


def is_reconcilable(
    raw: object, *, action_kind: str | None = None, category: str | None = None
) -> bool:
    """True when a persisted condition may be auto-resolved.

    The normalized condition drops kind/category, so at reconcile time the caller
    must pass the originating action's own kind and category (the owner-ask
    category, e.g. ``budget_exhausted``); either being an owner decision blocks it.
    """
    try:
        validate_condition(raw)
    except ConditionError:
        return False
    return not any(
        isinstance(v, str) and v.strip().lower() in OWNER_DECISION_SUBJECTS
        for v in (action_kind, category)
    )
