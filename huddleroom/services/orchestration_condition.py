"""Typed conditions that make a warning or question auto-resolvable."""

from __future__ import annotations

import uuid

SESSION_RUNNING = "session_running"
VALID_RESULT_MISSING = "valid_result_missing"
PLAN_MISSING = "plan_missing"
GATE_UNVERIFIED = "gate_unverified"
DECISION_PENDING = "decision_pending"

CONDITIONS = frozenset({SESSION_RUNNING, VALID_RESULT_MISSING, PLAN_MISSING, GATE_UNVERIFIED, DECISION_PENDING})

# Owner decisions (including every owner-escalation category) are never auto-cancelled,
# whatever condition they declare.
OWNER_DECISION_SUBJECTS = frozenset({
    "budget", "permission", "provider", "owner_decision", "budget_exhausted", "permission_required",
    "capability_unavailable", "result_correction_exhausted", "revision_exhausted", "review_input_missing",
})


# Shown to the models that may declare a condition; the subject formats match validate_condition.
CONDITION_PROMPT = (
    "Where condition is listed as an optional field, it is shaped "
    '{"condition":"<name>","subject":"<id>"} and auto-resolves the warning or question once it holds: '
    "session_running (subject: task UUID; holds when no session of the task is pending or running), "
    "valid_result_missing (subject: task UUID; holds once a valid result for the task is accepted), "
    "plan_missing (subject: run UUID, optionally :<planning_version>; holds once a plan for that run/version is accepted), "
    "gate_unverified (subject: gate UUID; holds once the gate is accepted). "
    "Owner decisions (budget, permission, provider) never take a condition."
)


class ConditionError(ValueError):
    pass


def _uuid(value: str, label: str) -> str:
    try:
        return str(uuid.UUID(value))
    except ValueError:
        raise ConditionError(f"{label} must be a UUID, got {value!r}") from None


def _normalized_subject(condition: str, subject: str) -> str:
    if condition != PLAN_MISSING:
        return _uuid(subject, f"{condition} subject")
    run_part, sep, version = subject.partition(":")
    if sep and not (version.isascii() and version.isdigit()):
        raise ConditionError(f"plan_missing subject version must be digits, got {version!r}")
    return _uuid(run_part, "plan_missing subject run") + (f":{int(version)}" if sep else "")


def validate_condition(raw: object) -> dict[str, str]:
    """Return the normalized ``{"condition", "subject"}`` or raise ConditionError.

    The subject is a UUID (task for session_running/valid_result_missing, gate for gate_unverified) or
    ``<run UUID>[:<planning_version digits>]`` for plan_missing; UUIDs are normalized to canonical form."""
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
    return {"condition": condition, "subject": _normalized_subject(condition, subject)}


def is_reconcilable(
    raw: object, *, action_kind: str | None = None, category: str | None = None
) -> bool:
    """True when a persisted condition may be auto-resolved.

    The normalized condition drops kind/category, so at reconcile time the caller
    must pass the originating action's own kind and category (the owner-ask
    category, e.g. ``budget_exhausted``); either being an owner decision blocks it.
    """
    try:
        normalized = validate_condition(raw)
    except ConditionError:
        return False
    # decision_pending is internal: it may only auto-resolve warnings, never cancel questions.
    if normalized["condition"] == DECISION_PENDING and action_kind != "record_warning":
        return False
    return not any(
        isinstance(v, str) and v.strip().lower() in OWNER_DECISION_SUBJECTS
        for v in (action_kind, category)
    )
