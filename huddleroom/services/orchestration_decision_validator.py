from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping
from uuid import UUID

from pydantic import ValidationError

from huddleroom.models.orchestration_process import WARNING_SEVERITIES, WARNING_SEVERITY_VALUES
from huddleroom.schemas.orchestration import AuthorizationScope
from huddleroom.services.orchestration_dependencies import ADHOC_GATED_WORK_FUNCTIONS, plan_in_flight
from huddleroom.services.orchestration_wake_when import normalize_wake_when


ALLOWED_ACTION_SCHEMAS: dict[str, frozenset[str]] = {
    "noop": frozenset({"wake_when"}),
    "request_plan": frozenset({"agent_id", "scope", "work_function"}),
    "request_roadmap_replan": frozenset({"agent_id", "scope", "reason"}),
    "request_plan_revision": frozenset({"plan_task_id", "revision_request"}),
    "accept_plan": frozenset({"plan_artifact_id"}),
    "create_delegation_task": frozenset({"agent_id", "deliverable", "scope", "work_function"}),
    "request_verification": frozenset({"gate_id", "work_function"}),
    "resolve_meeting_action_item": frozenset({"item_id", "reason"}),
    "retry_task": frozenset({"task_id"}),
    "reassign_task": frozenset({"agent_id", "task_id"}),
    "schedule_meeting": frozenset({"participant_agent_ids", "topic"}),
    "start_graph": frozenset({"graph_id", "subject_id", "subject_type"}),
    "ask_human": frozenset({"question"}),
    "pause_run": frozenset({"reason"}),
    "suggest_agent": frozenset({"missing_work_function", "reason"}),
    "record_warning": frozenset({"warning_type", "severity", "message"}),
    # Non-mutating: records the observed blocker; the server builds any remedy (SPR 180/184).
    "recover_blocker": frozenset({"reason", "evidence_refs"}),
}

RECOVER_BLOCKER_EVIDENCE_KINDS = frozenset({"action", "task", "gate", "process", "decision", "warning"})

OPTIONAL_ACTION_FIELDS: dict[str, frozenset[str]] = {
    "schedule_meeting": frozenset(
        {
            "task_id",
            "gate_id",
            "organizer_agent_id",
        }
    ),
    "create_delegation_task": frozenset(
        {
            "title",
            "inputs",
            "forbidden_work",
            "success_evidence",
            "budget",
            "report_schema",
            "parent_task_id",
            "source_session_id",
            "depends_on",
            "decision_use",
            "why_runnable",
            "priority",
            "artifact_ids",
            "authorization_scope",
        }
    ),
    "ask_human": frozenset(
        {
            "work_function",
            "required_capabilities",
            "candidate_agent_ids",
            "gate_id",
        }
    ),
    "suggest_agent": frozenset(
        {
            "suggested_role",
            "suggested_capabilities",
            "suggested_adapter_type",
            "suggested_model",
            "suggested_system_prompt_outline",
        }
    ),
    "record_warning": frozenset(
        {"run_id", "source_process_run_id", "related_gate_id", "related_action_id", "related_agent_id"}
    ),
    "retry_task": frozenset({"timeout", "max_tokens"}),
}

FORBIDDEN_ARTIFACT_KEYS = {
    "plan_text",
    "test_code",
    "review_text",
    "validation_report",
    "meeting_decision_text",
    "final_summary",
    "artifact_content",
    "file_content",
}

OPTIONAL_TOP_LEVEL_DECISION_KEYS = frozenset({"action_type", "reason", "applies_decision_id"})

MAX_VALIDATION_DEPTH = 64
PLAN_WORK_FUNCTION = "planning"
FINAL_SUMMARY_WORK_FUNCTION = "summarization"


@dataclass(frozen=True)
class DecisionValidationResult:
    accepted: bool
    rejection_reason: str | None = None


def validate_orchestration_decision(decision: Mapping[str, Any]) -> DecisionValidationResult:
    if not isinstance(decision, Mapping):
        return DecisionValidationResult(False, "Decision must be an object")

    action_type = decision.get("action_type")
    if not isinstance(action_type, str) or not action_type:
        return DecisionValidationResult(False, "Decision action_type is required")

    required_fields = ALLOWED_ACTION_SCHEMAS.get(action_type)
    if required_fields is None:
        return DecisionValidationResult(False, f"Unknown orchestration action type '{action_type}'")

    try:
        forbidden_path = _find_forbidden_key(decision)
    except ValueError as exc:
        return DecisionValidationResult(False, str(exc))
    if forbidden_path is not None:
        return DecisionValidationResult(False, f"Decision includes forbidden artifact content at '{forbidden_path}'")

    allowed_top_level_keys = required_fields | OPTIONAL_TOP_LEVEL_DECISION_KEYS | OPTIONAL_ACTION_FIELDS.get(action_type, frozenset())
    unexpected_keys = sorted(str(key) for key in decision if str(key) not in allowed_top_level_keys)
    if unexpected_keys:
        label = "key" if len(unexpected_keys) == 1 else "keys"
        return DecisionValidationResult(
            False,
            f"Decision '{action_type}' includes unknown top-level {label}: {', '.join(unexpected_keys)}",
        )

    missing_fields = sorted(
        field
        for field in required_fields
        if field not in decision or _is_missing_required_value(decision[field])
    )
    if missing_fields:
        return DecisionValidationResult(
            False,
            f"Decision '{action_type}' missing required fields: {', '.join(missing_fields)}",
        )

    if action_type == "noop":
        try:
            normalize_wake_when(decision["wake_when"])
        except ValueError as exc:
            return DecisionValidationResult(False, f"Decision 'noop' wake_when invalid: {exc}")

    if action_type == "request_plan" and str(decision.get("work_function")).strip() != PLAN_WORK_FUNCTION:
        return DecisionValidationResult(
            False,
            f"Decision 'request_plan' work_function must be '{PLAN_WORK_FUNCTION}'",
        )

    if "applies_decision_id" in decision and not _is_uuid_string(decision["applies_decision_id"]):
        return DecisionValidationResult(False, "applies_decision_id must be a UUID")

    if action_type == "record_warning" and (
        not isinstance(decision["severity"], str) or decision["severity"] not in WARNING_SEVERITIES
    ):
        return DecisionValidationResult(
            False,
            f"Decision 'record_warning' severity must be one of: {', '.join(WARNING_SEVERITY_VALUES)}",
        )

    if action_type == "create_delegation_task":
        error = delegation_optional_field_error(decision)
        if error is not None:
            return DecisionValidationResult(False, f"Decision 'create_delegation_task' {error}")

    if action_type == "recover_blocker" and not _valid_evidence_refs(decision["evidence_refs"]):
        return DecisionValidationResult(
            False,
            "Decision 'recover_blocker' evidence_refs must be a list of {kind, id} objects with kind in "
            f"{', '.join(sorted(RECOVER_BLOCKER_EVIDENCE_KINDS))} and a UUID id",
        )

    if action_type == "resolve_meeting_action_item":
        if not _is_uuid_string(decision["item_id"]):
            return DecisionValidationResult(False, "item_id must be a UUID")
        if not isinstance(decision["reason"], str) or not 1 <= len(decision["reason"].strip()) <= 500:
            return DecisionValidationResult(False, "reason must be a nonblank string of 1–500 characters")

    if action_type == "retry_task":
        for key in ("timeout", "max_tokens"):
            if key in decision and (isinstance(decision[key], bool) or not isinstance(decision[key], int) or decision[key] <= 0):
                return DecisionValidationResult(False, f"Decision 'retry_task' {key} must be a positive integer")

    return DecisionValidationResult(True)


def delegation_optional_field_error(request: Mapping[str, Any]) -> str | None:
    """Type-check the optional create_delegation_task fields; None when all present ones are valid."""
    request = {k: v for k, v in request.items() if v is not None}  # null means absent
    for key in ("title", "decision_use", "why_runnable"):
        if key in request and not isinstance(request[key], str):
            return f"{key} must be a string"
    if "priority" in request:
        priority = request["priority"]
        if isinstance(priority, bool) or not isinstance(priority, int) or not 0 <= priority <= 100:
            return "priority must be an integer from 0 to 100"
    for key, label in (("depends_on", "task"), ("artifact_ids", "artifact")):
        if key in request:
            ids = request[key]
            if (
                not isinstance(ids, list)
                or not all(_is_uuid_string(i) for i in ids)
                or len({str(UUID(i)) for i in ids}) != len(ids)
            ):
                return f"{key} must be a list of distinct {label} UUIDs"
    if "authorization_scope" in request:
        try:
            scope = AuthorizationScope.model_validate(request["authorization_scope"])
        except ValidationError:
            return "authorization_scope is not a valid authorization scope"
        if scope.external_writes.allowed or scope.live_sends.allowed:
            return SELF_GRANT_ERROR
    return None


LABEL_WORDS = frozenset({
    "criterion", "criteria", "success", "for", "the", "a", "an", "to", "on", "of", "advance", "advances",
    "progress", "work", "complete", "satisfy", "meet", "cover",
})
SELF_GRANT_ERROR = "authorization_scope cannot self-grant external writes or live sends; route them to ask_human"
FILLER_PHRASES = (
    "keep busy", "stay busy", "keep the team busy", "busy work", "busywork", "fill time", "fill capacity",
    "idle capacity", "show activity", "show progress",
)


def _normalized(text: Any) -> str:
    return re.sub(r"[^a-z0-9_:]", " ", str(text).lower())


def independent_work_rejection(
    request: Mapping[str, Any], criterion_keys: Any, plan_state: Any
) -> str | None:
    """Reject investigation/review proposed while planning is in flight unless it names real use and evidence."""
    if request.get("work_function") not in ADHOC_GATED_WORK_FUNCTIONS or not plan_in_flight(plan_state):
        return None
    keys = {t for k in criterion_keys or [] if isinstance(k, str) for t in _normalized(k).split()}
    ignored = LABEL_WORDS | keys | {f"criterion:{k}" for k in keys}

    def residual(text: Any) -> list[str]:
        return [t for t in _normalized(text).split() if t not in ignored]

    def invalid(reason: str) -> str:
        return f"independent_work_invalid: {reason}"

    decision_use = request.get("decision_use")
    if not isinstance(decision_use, str) or not decision_use.strip():
        return invalid("name the decision or execution step this work improves")
    if not residual(decision_use):
        return invalid("decision_use only names a success criterion")
    evidence = [e for e in request.get("success_evidence") or [] if isinstance(e, str)]
    # Whole-token phrase match on the rationale fields only: scope/deliverable may legitimately mention such words.
    rationale = [f" {' '.join(_normalized(t).split())} " for t in (decision_use, request.get("why_runnable")) if t]
    if any(f" {phrase} " in text for text in rationale for phrase in FILLER_PHRASES):
        return invalid("work whose only rationale is keeping agents busy is not allowed")
    if not any(residual(e) for e in evidence):
        return invalid("name concrete expected evidence")
    unfinished = {
        str(plan_state.get(k)) for k in ("planning_task_id", "plan_artifact_id") if plan_state.get(k)
    }
    if unfinished & {str(i) for i in [*(request.get("depends_on") or []), *(request.get("artifact_ids") or [])]}:
        return invalid("depends on unfinished planning output")
    return None


def _valid_evidence_refs(refs: Any) -> bool:
    return isinstance(refs, list) and all(
        isinstance(ref, Mapping) and set(ref) == {"kind", "id"}
        and ref["kind"] in RECOVER_BLOCKER_EVIDENCE_KINDS and _is_uuid_string(ref["id"])
        for ref in refs
    )


def _is_uuid_string(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        UUID(value)
    except ValueError:
        return False
    return True


def _is_missing_required_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return False


def _find_forbidden_key(value: Any, path: str = "", depth: int = 0) -> str | None:
    if depth > MAX_VALIDATION_DEPTH:
        raise ValueError(f"Decision nesting exceeds maximum depth at '{path or '<root>'}'")

    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            key_str = str(key)
            key_path = f"{path}.{key_str}" if path else key_str
            if key_str in FORBIDDEN_ARTIFACT_KEYS:
                return key_path
            forbidden_path = _find_forbidden_key(nested_value, key_path, depth + 1)
            if forbidden_path is not None:
                return forbidden_path
    elif isinstance(value, list):
        for index, nested_value in enumerate(value):
            nested_path = f"{path}[{index}]" if path else f"[{index}]"
            forbidden_path = _find_forbidden_key(nested_value, nested_path, depth + 1)
            if forbidden_path is not None:
                return forbidden_path

    return None
