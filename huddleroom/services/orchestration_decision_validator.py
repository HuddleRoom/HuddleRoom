from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


ALLOWED_ACTION_SCHEMAS: dict[str, frozenset[str]] = {
    "noop": frozenset(),
    "request_plan": frozenset({"agent_id", "scope", "work_function"}),
    "request_roadmap_replan": frozenset({"agent_id", "scope", "reason"}),
    "request_plan_revision": frozenset({"plan_task_id", "revision_request"}),
    "accept_plan": frozenset({"plan_artifact_id"}),
    "create_delegation_task": frozenset({"agent_id", "deliverable", "scope", "work_function"}),
    "expand_plan_item": frozenset({"plan_item_id", "work_function"}),
    "open_gate": frozenset({"gate_type", "required_evidence", "success_criterion_key"}),
    "request_verification": frozenset({"gate_id", "work_function"}),
    "retry_task": frozenset({"task_id"}),
    "reassign_task": frozenset({"agent_id", "task_id"}),
    "request_split": frozenset({"reason", "task_id"}),
    "schedule_meeting": frozenset({"participant_agent_ids", "topic"}),
    "start_graph": frozenset({"graph_id", "subject_id", "subject_type"}),
    "ask_human": frozenset({"question"}),
    "pause_run": frozenset({"reason"}),
    "complete_run": frozenset({"reason"}),
    "request_final_summary": frozenset({"work_function"}),
    "suggest_agent": frozenset({"missing_work_function", "reason"}),
    "request_human_decision": frozenset({"title", "question"}),
    "request_manager_decision": frozenset({"title", "question"}),
    "record_authority_decision": frozenset({"decision_id", "artifact_id"}),
    "cancel_pending_decision": frozenset({"decision_id", "reason"}),
    "record_warning": frozenset({"warning_type", "severity", "message"}),
    "acknowledge_warning": frozenset({"warning_id", "reason", "decided_by_user_id"}),
    "resolve_warning": frozenset({"warning_id", "reason"}),
}

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
            "inputs",
            "forbidden_work",
            "success_evidence",
            "budget",
            "report_schema",
            "parent_task_id",
            "source_session_id",
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
    "request_human_decision": frozenset({"options", "context", "recommendation", "consequences"}),
    "request_manager_decision": frozenset({"options", "context", "recommendation", "consequences"}),
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

OPTIONAL_TOP_LEVEL_DECISION_KEYS = frozenset({"action_type", "reason"})

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

    if action_type == "request_plan" and str(decision.get("work_function")).strip() != PLAN_WORK_FUNCTION:
        return DecisionValidationResult(
            False,
            f"Decision 'request_plan' work_function must be '{PLAN_WORK_FUNCTION}'",
        )

    if (
        action_type == "request_final_summary"
        and str(decision.get("work_function")).strip() != FINAL_SUMMARY_WORK_FUNCTION
    ):
        return DecisionValidationResult(
            False,
            "Decision 'request_final_summary' work_function must be 'summarization'",
        )

    if action_type == "retry_task":
        for key in ("timeout", "max_tokens"):
            if key in decision and (isinstance(decision[key], bool) or not isinstance(decision[key], int) or decision[key] <= 0):
                return DecisionValidationResult(False, f"Decision 'retry_task' {key} must be a positive integer")

    return DecisionValidationResult(True)


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
