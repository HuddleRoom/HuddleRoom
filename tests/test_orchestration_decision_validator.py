import uuid

import pytest
import sqlalchemy as sa
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError

from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.services.orchestration_decision_validator import validate_orchestration_decision
from huddleroom.services.orchestration_service import OrchestrationService
from tests.orchestration_wake_helpers import NOOP_WAKE_WHEN


def _id() -> str:
    return str(uuid.uuid4())


ALLOWED_DECISIONS = {
    "noop": {
        "action_type": "noop",
        "reason": "Nothing can advance yet.",
        "wake_when": {"recheck_after_seconds": 300, "expected_result": "Dependency moved on."},
    },
    "request_plan": {
        "action_type": "request_plan",
        "work_function": "planning",
        "agent_id": _id(),
        "scope": "Create an agent-owned implementation plan for this goal.",
    },
    "request_plan_revision": {
        "action_type": "request_plan_revision",
        "plan_task_id": _id(),
        "revision_request": "Split the plan into independently verifiable tasks.",
    },
    "accept_plan": {"action_type": "accept_plan", "plan_artifact_id": _id()},
    "create_delegation_task": {
        "action_type": "create_delegation_task",
        "work_function": "implementation",
        "agent_id": _id(),
        "scope": "Implement the accepted plan item.",
        "deliverable": "A task result containing changed file paths and verification output.",
    },
    "request_verification": {
        "action_type": "request_verification",
        "gate_id": _id(),
        "work_function": "validation",
    },
    "retry_task": {"action_type": "retry_task", "task_id": _id()},
    "reassign_task": {"action_type": "reassign_task", "task_id": _id(), "agent_id": _id()},
    "schedule_meeting": {
        "action_type": "schedule_meeting",
        "topic": "Resolve contradictory validation outputs.",
        "participant_agent_ids": [_id(), _id()],
    },
    "start_graph": {
        "action_type": "start_graph",
        "graph_id": _id(),
        "subject_type": "task",
        "subject_id": _id(),
    },
    "ask_human": {"action_type": "ask_human", "question": "Which agent should validate this gate?"},
    "pause_run": {"action_type": "pause_run", "reason": "Repeated failure threshold reached."},
    "suggest_agent": {
        "action_type": "suggest_agent",
        "missing_work_function": "validation",
        "reason": "No active agent can provide independent validation.",
    },
}

# Schema-only types were removed: no executor exists, so the dispatcher silently
# dropped them. Each must now be rejected as unknown.
REMOVED_ACTION_TYPES = (
    "expand_plan_item",
    "open_gate",
    "request_split",
    "complete_run",
    "request_final_summary",
    "request_human_decision",
    "request_manager_decision",
    "record_authority_decision",
    "cancel_pending_decision",
    "acknowledge_warning",
    "resolve_warning",
)


@pytest.mark.parametrize("action_type", sorted(ALLOWED_DECISIONS))
def test_allowed_coordination_decisions_pass(action_type):
    result = validate_orchestration_decision(ALLOWED_DECISIONS[action_type])

    assert result.accepted is True
    assert result.rejection_reason is None


def test_unknown_action_type_is_rejected():
    result = validate_orchestration_decision({"action_type": "write_code", "code": "print('no')"})

    assert result.accepted is False
    assert result.rejection_reason == "Unknown orchestration action type 'write_code'"


@pytest.mark.parametrize("action_type", REMOVED_ACTION_TYPES)
def test_removed_schema_only_action_types_are_rejected_as_unknown(action_type):
    result = validate_orchestration_decision({"action_type": action_type, "reason": "Attempt."})

    assert result.accepted is False
    assert result.rejection_reason == f"Unknown orchestration action type '{action_type}'"


@pytest.mark.parametrize(
    ("extra_keys", "expected_contains"),
    [
        ({"unexpected": "top-level noise"}, "unexpected"),
        ({"z_key": "later", "a_key": "first"}, "a_key, z_key"),
    ],
    ids=["single", "sorted"],
)
def test_unknown_top_level_keys_are_rejected(extra_keys, expected_contains):
    decision_data = {
        "action_type": "request_plan",
        "work_function": "planning",
        "agent_id": _id(),
        "scope": "Create an implementation plan.",
    }
    decision_data.update(extra_keys)
    result = validate_orchestration_decision(decision_data)

    assert result.accepted is False
    assert result.rejection_reason is not None
    assert expected_contains in result.rejection_reason


def test_ask_human_allows_assignment_context_fields():
    result = validate_orchestration_decision(
        {
            "action_type": "ask_human",
            "question": "No strong validation fit exists. Which existing agent should validate this gate?",
            "work_function": "validation",
            "required_capabilities": ["validation", "testing"],
            "candidate_agent_ids": [_id()],
            "gate_id": _id(),
            "reason": "The best available roster fit is weak.",
        }
    )

    assert result.accepted is True
    assert result.rejection_reason is None


def test_suggest_agent_allows_suggestion_context_fields():
    result = validate_orchestration_decision(
        {
            "action_type": "suggest_agent",
            "missing_work_function": "validation",
            "reason": "No active agent can provide independent validation.",
            "suggested_role": "validator",
            "suggested_capabilities": ["validation", "testing"],
            "suggested_adapter_type": "api",
            "suggested_model": "gpt-4o-mini",
            "suggested_system_prompt_outline": "Validate completed work and report evidence without editing artifacts.",
        }
    )

    assert result.accepted is True
    assert result.rejection_reason is None


def test_create_delegation_task_allows_contract_context_fields():
    result = validate_orchestration_decision(
        {
            "action_type": "create_delegation_task",
            "work_function": "implementation",
            "agent_id": _id(),
            "scope": "Implement the accepted plan item without changing unrelated files.",
            "inputs": ["ROADMAP.md M7 Phase 10", "docs/superpowers/specs/2026-06-28-milestone7-orchestration-design.md"],
            "deliverable": "A code change plus focused pytest output.",
            "forbidden_work": ["Do not edit dashboard files.", "Do not create agents."],
            "success_evidence": ["pytest output for tests/test_orchestration_delegation_contracts.py"],
            "budget": {"max_tokens": 20000},
            "report_schema": {"changed_files": "list[str]", "tests": "list[str]", "notes": "str"},
            "parent_task_id": _id(),
            "reason": "Accepted plan item needs implementation work.",
        }
    )

    assert result.accepted is True
    assert result.rejection_reason is None


def test_noop_accepts_missing_reason_when_wake_when_present():
    result = validate_orchestration_decision(
        {"action_type": "noop", "wake_when": {"recheck_after_seconds": 300, "expected_result": "Wait over."}}
    )

    assert result.accepted is True
    assert result.rejection_reason is None


def test_noop_without_wake_when_is_rejected():
    result = validate_orchestration_decision({"action_type": "noop", "reason": "Waiting."})

    assert result.accepted is False
    assert result.rejection_reason == "Decision 'noop' missing required fields: wake_when"


def test_noop_with_invalid_wake_when_is_rejected():
    result = validate_orchestration_decision(
        {"action_type": "noop", "wake_when": {"events": "not-a-list", "expected_result": "x"}}
    )

    assert result.accepted is False
    assert result.rejection_reason == "Decision 'noop' wake_when invalid: events must be a list"


def test_wake_when_is_rejected_on_other_actions():
    result = validate_orchestration_decision(
        {
            "action_type": "ask_human",
            "question": "Choose scope.",
            "wake_when": {"recheck_after_seconds": 300, "expected_result": "x"},
        }
    )

    assert result.accepted is False
    assert result.rejection_reason == "Decision 'ask_human' includes unknown top-level key: wake_when"


@pytest.mark.parametrize("key", ["timeout", "max_tokens"])
@pytest.mark.parametrize("value", [1, 30])
def test_retry_task_accepts_positive_execution_overrides(key, value):
    result = validate_orchestration_decision({"action_type": "retry_task", "task_id": _id(), key: value})

    assert result.accepted is True


@pytest.mark.parametrize("key", ["timeout", "max_tokens"])
@pytest.mark.parametrize("value", [0, -1, True, "30"])
def test_retry_task_rejects_non_positive_or_non_integer_execution_overrides(key, value):
    result = validate_orchestration_decision({"action_type": "retry_task", "task_id": _id(), key: value})

    assert result.accepted is False
    assert result.rejection_reason == f"Decision 'retry_task' {key} must be a positive integer"


def test_retry_task_rejects_unknown_execution_override():
    result = validate_orchestration_decision({"action_type": "retry_task", "task_id": _id(), "retries": 1})

    assert result.accepted is False


def test_missing_required_field_is_rejected():
    result = validate_orchestration_decision({"action_type": "request_plan", "work_function": "planning"})

    assert result.accepted is False
    assert result.rejection_reason == "Decision 'request_plan' missing required fields: agent_id, scope"


def test_request_plan_work_function_must_be_planning():
    decision = {"action_type": "request_plan", "work_function": "implementation", "agent_id": _id(), "scope": "Ask an agent for a plan."}

    result = validate_orchestration_decision(decision)

    assert result.accepted is False
    assert result.rejection_reason == "Decision 'request_plan' work_function must be 'planning'"


@pytest.mark.parametrize(
    ("decision", "missing"),
    [
        (
            {"action_type": "schedule_meeting", "topic": "Resolve contradictory validation outputs.", "participant_agent_ids": []},
            "participant_agent_ids",
        ),
        (
            {"action_type": "schedule_meeting", "topic": "Resolve contradictory validation outputs.", "participant_agent_ids": {}},
            "participant_agent_ids",
        ),
    ],
)
def test_empty_required_container_is_rejected(decision, missing):
    result = validate_orchestration_decision(decision)

    assert result.accepted is False
    assert result.rejection_reason == (
        f"Decision '{decision['action_type']}' missing required fields: {missing}"
    )


@pytest.mark.parametrize(
    ("decision", "missing"),
    [
        (
            {
                "action_type": "request_plan",
                "work_function": "planning",
                "agent_id": _id(),
                "scope": "   ",
            },
            "scope",
        ),
        (
            {
                "action_type": "request_plan_revision",
                "plan_task_id": _id(),
                "revision_request": "\n\t",
            },
            "revision_request",
        ),
    ],
)
def test_whitespace_only_required_string_is_rejected(decision, missing):
    result = validate_orchestration_decision(decision)

    assert result.accepted is False
    assert result.rejection_reason == (
        f"Decision '{decision['action_type']}' missing required fields: {missing}"
    )


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "plan_text",
        "test_code",
        "review_text",
        "validation_report",
        "meeting_decision_text",
        "final_summary",
        "artifact_content",
        "file_content",
    ],
)
def test_direct_artifact_content_is_rejected(forbidden_key):
    decision = {
        "action_type": "request_plan",
        "work_function": "planning",
        "agent_id": _id(),
        "scope": "Ask an agent for a plan.",
        forbidden_key: "orchestrator-authored work artifact",
    }

    result = validate_orchestration_decision(decision)

    assert result.accepted is False
    assert result.rejection_reason == f"Decision includes forbidden artifact content at '{forbidden_key}'"


def test_first_forbidden_sibling_in_document_order_is_reported():
    result = validate_orchestration_decision(
        {
            "action_type": "request_plan",
            "work_function": "planning",
            "agent_id": _id(),
            "scope": "Ask an agent for a plan.",
            "test_code": "assert False",
            "plan_text": "1. do the work",
        }
    )

    assert result.accepted is False
    assert result.rejection_reason == "Decision includes forbidden artifact content at 'test_code'"


@pytest.mark.parametrize("coordination_key", ["tests", "review", "summary", "diff", "patch", "artifact", "code", "plan"])
def test_ambiguous_coordination_keys_are_allowed(coordination_key):
    result = validate_orchestration_decision(
        {
            "action_type": "create_delegation_task",
            "agent_id": _id(),
            "work_function": "implementation",
            "scope": "Implement the accepted plan item.",
            "deliverable": "A task result containing changed file paths and verification output.",
            "inputs": {coordination_key: {"required_source_types": ["task"], "min_count": 1}},
        }
    )

    assert result.accepted is True
    assert result.rejection_reason is None


def test_nested_artifact_content_is_rejected():
    result = validate_orchestration_decision(
        {
            "action_type": "create_delegation_task",
            "work_function": "implementation",
            "agent_id": _id(),
            "scope": "Assign work only.",
            "deliverable": "Changed files and verification output.",
            "contract": {"file_content": "def orchestrator_authored_artifact(): pass"},
        }
    )

    assert result.accepted is False
    assert result.rejection_reason == "Decision includes forbidden artifact content at 'contract.file_content'"


def test_list_nested_artifact_content_uses_bracket_path():
    result = validate_orchestration_decision(
        {
            "action_type": "create_delegation_task",
            "work_function": "implementation",
            "agent_id": _id(),
            "scope": "Assign work only.",
            "deliverable": "Changed files and verification output.",
            "items": [{"file_content": "def orchestrator_authored_artifact(): pass"}],
        }
    )

    assert result.accepted is False
    assert result.rejection_reason == "Decision includes forbidden artifact content at 'items[0].file_content'"


def test_schedule_meeting_allows_orchestration_linkage_fields():
    participant_id = _id()
    result = validate_orchestration_decision(
        {
            "action_type": "schedule_meeting",
            "topic": "Resolve contradictory validation evidence.",
            "participant_agent_ids": [participant_id],
            "task_id": _id(),
            "gate_id": _id(),
            "organizer_agent_id": participant_id,
        }
    )

    assert result.accepted is True
    assert result.rejection_reason is None


def test_deeply_nested_decision_is_rejected_without_recursion_error():
    decision = {
        "action_type": "create_delegation_task",
        "agent_id": _id(),
        "work_function": "implementation",
        "scope": "Implement the accepted plan item.",
        "deliverable": "A task result containing changed file paths and verification output.",
        "inputs": {},
    }
    cursor = decision["inputs"]
    for index in range(70):
        cursor["nest"] = {}
        cursor = cursor["nest"]
        cursor[f"level_{index}"] = {}

    result = validate_orchestration_decision(decision)

    assert result.accepted is False
    assert result.rejection_reason is not None
    assert result.rejection_reason.startswith("Decision nesting exceeds maximum depth at 'inputs.")


async def _make_run(db_session, test_project):
    service = OrchestrationService()
    _, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Validate orchestration decisions",
            success_criteria=[{"key": "validated", "description": "Decisions are validated before execution."}],
        ),
        created_by_user_id=None,
    )
    return service, run


@pytest.mark.asyncio
async def test_record_validated_decision_persists_accepted_decision(db_session, test_project):
    service, run = await _make_run(db_session, test_project)
    parsed = {
        "action_type": "ask_human",
        "question": "Which existing agent should validate this work?",
        "reason": "No clear independent validator exists.",
    }
    llm_output = {"decision": parsed}

    decision = await service.record_validated_decision(
        db_session,
        run_id=run.id,
        input_snapshot={"open_gates": ["validation"]},
        llm_output=llm_output,
        parsed_decision=parsed,
    )

    assert decision.decision_type == "ask_human"
    assert decision.validator_status == "accepted"
    assert decision.rejection_reason is None
    assert decision.reason == "No clear independent validator exists."
    assert decision.input_snapshot == {"open_gates": ["validation"]}
    assert decision.llm_output == {"decision": parsed}
    assert decision.parsed_decision == parsed

    parsed["question"] = "mutated"
    llm_output["decision"]["question"] = "mutated"
    assert decision.parsed_decision["question"] == "Which existing agent should validate this work?"
    assert decision.llm_output["decision"]["question"] == "Which existing agent should validate this work?"


@pytest.mark.asyncio
async def test_record_validated_decision_persists_unknown_action_rejection(db_session, test_project):
    service, run = await _make_run(db_session, test_project)

    decision = await service.record_validated_decision(
        db_session,
        run_id=run.id,
        input_snapshot={},
        llm_output={"action_type": "write_tests"},
        parsed_decision={"action_type": "write_tests", "tests": "assert forbidden"},
    )

    assert decision.decision_type == "write_tests"
    assert decision.validator_status == "rejected"
    assert decision.rejection_reason == "Unknown orchestration action type 'write_tests'"


@pytest.mark.asyncio
async def test_record_validated_decision_persists_forbidden_content_rejection(db_session, test_project):
    service, run = await _make_run(db_session, test_project)

    decision = await service.record_validated_decision(
        db_session,
        run_id=run.id,
        input_snapshot={},
        llm_output={"action_type": "request_plan", "plan_text": "1. Do the work"},
        parsed_decision={
            "action_type": "request_plan",
            "work_function": "planning",
            "agent_id": _id(),
            "scope": "Ask an agent for a plan.",
            "plan_text": "1. Do the work",
        },
    )

    assert decision.validator_status == "rejected"
    assert decision.rejection_reason == "Decision includes forbidden artifact content at 'plan_text'"


@pytest.mark.asyncio
async def test_record_validated_decision_persists_overlong_action_type_as_invalid(db_session, test_project):
    service, run = await _make_run(db_session, test_project)
    action_type = "x" * 101

    decision = await service.record_validated_decision(
        db_session,
        run_id=run.id,
        input_snapshot={},
        llm_output={"action_type": action_type},
        parsed_decision={"action_type": action_type},
    )

    assert decision.decision_type == "invalid"
    assert decision.validator_status == "rejected"
    assert decision.rejection_reason == f"Unknown orchestration action type '{action_type}'"


@pytest.mark.asyncio
async def test_record_validated_decision_persists_non_object_rejection(db_session, test_project):
    service, run = await _make_run(db_session, test_project)

    decision = await service.record_validated_decision(
        db_session,
        run_id=run.id,
        input_snapshot={},
        llm_output=["noop"],
        parsed_decision=["noop"],
    )

    assert decision.decision_type == "invalid"
    assert decision.validator_status == "rejected"
    assert decision.rejection_reason == "Decision must be an object"
    assert decision.reason is None


@pytest.mark.asyncio
async def test_record_validated_decision_persists_null_rejection(db_session, test_project):
    service, run = await _make_run(db_session, test_project)

    decision = await service.record_validated_decision(
        db_session,
        run_id=run.id,
        input_snapshot=None,
        llm_output="null",
        parsed_decision=None,
    )

    assert decision.decision_type == "invalid"
    assert decision.validator_status == "rejected"
    assert decision.rejection_reason == "Decision must be an object"
    assert decision.reason is None
    assert decision.input_snapshot == {}
    assert decision.parsed_decision == {}


@pytest.mark.asyncio
async def test_record_validated_decision_rejects_missing_run(db_session):
    service = OrchestrationService()

    with pytest.raises(HTTPException) as exc_info:
        await service.record_validated_decision(
            db_session,
            run_id=uuid.uuid4(),
            input_snapshot={},
            llm_output={"action_type": "noop", "wake_when": NOOP_WAKE_WHEN},
            parsed_decision={"action_type": "noop", "wake_when": NOOP_WAKE_WHEN},
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_record_validated_decision_preserves_404_when_run_disappears_before_insert(
    db_session,
    test_project,
    monkeypatch,
):
    from huddleroom.models.orchestration import OrchestrationDecision

    service, run = await _make_run(db_session, test_project)

    async def fake_run_exists(_db, _run_id):
        return False

    monkeypatch.setattr(service, "_run_exists", fake_run_exists)

    class ForeignKeyFailureOnFlushSession:
        def __init__(self, session, run_id):
            self.session = session
            self.failed = False
            self.run_id = run_id

        def __getattr__(self, name):
            return getattr(self.session, name)

        async def flush(self, *args, **kwargs):
            if not self.failed:
                self.failed = True
                raise IntegrityError(
                    "INSERT INTO orchestration_decisions ...",
                    {"run_id": "missing"},
                    Exception("FOREIGN KEY constraint failed"),
                )
            return await self.session.flush(*args, **kwargs)

    with pytest.raises(HTTPException) as exc_info:
        await service.record_validated_decision(
            ForeignKeyFailureOnFlushSession(db_session, run.id),
            run_id=run.id,
            input_snapshot={},
            llm_output={"action_type": "noop", "wake_when": NOOP_WAKE_WHEN},
            parsed_decision={"action_type": "noop", "wake_when": NOOP_WAKE_WHEN},
        )

    assert exc_info.value.status_code == 404
    result = await db_session.execute(sa.select(sa.func.count()).select_from(OrchestrationDecision))
    assert result.scalar_one() == 0


def test_applies_decision_id_uuid_string_is_accepted():
    decision = {**ALLOWED_DECISIONS["request_plan"], "applies_decision_id": _id()}

    assert validate_orchestration_decision(decision).accepted is True


@pytest.mark.parametrize(
    "value",
    ["not-a-uuid", "", 123, None, ["%s" % _id()]],
    ids=["text", "empty", "int", "null", "list"],
)
def test_applies_decision_id_must_be_uuid(value):
    decision = {**ALLOWED_DECISIONS["request_plan"], "applies_decision_id": value}
    result = validate_orchestration_decision(decision)

    assert result.accepted is False
    assert result.rejection_reason == "applies_decision_id must be a UUID"
