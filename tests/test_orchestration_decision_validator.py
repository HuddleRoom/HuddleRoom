import uuid

import pytest
import sqlalchemy as sa
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError

from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.services.orchestration_decision_validator import validate_orchestration_decision
from huddleroom.services.orchestration_service import OrchestrationService


def _id() -> str:
    return str(uuid.uuid4())


ALLOWED_DECISIONS = {
    "noop": {"action_type": "noop", "reason": "Nothing can advance yet."},
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
    "expand_plan_item": {
        "action_type": "expand_plan_item",
        "plan_item_id": "item-1",
        "work_function": "implementation",
    },
    "open_gate": {
        "action_type": "open_gate",
        "success_criterion_key": "tests-pass",
        "gate_type": "validation_passed",
        "required_evidence": {"required_source_types": ["task"], "min_count": 1},
    },
    "request_verification": {
        "action_type": "request_verification",
        "gate_id": _id(),
        "work_function": "validation",
    },
    "retry_task": {"action_type": "retry_task", "task_id": _id()},
    "reassign_task": {"action_type": "reassign_task", "task_id": _id(), "agent_id": _id()},
    "request_split": {
        "action_type": "request_split",
        "task_id": _id(),
        "reason": "The assigned task is too broad for one agent pass.",
    },
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
    "complete_run": {"action_type": "complete_run", "reason": "All gates are accepted."},
    "request_final_summary": {
        "action_type": "request_final_summary",
        "work_function": "summarization",
    },
    "suggest_agent": {
        "action_type": "suggest_agent",
        "missing_work_function": "validation",
        "reason": "No active agent can provide independent validation.",
    },
    "request_human_decision": {
        "action_type": "request_human_decision",
        "title": "Continue without independent verifier?",
        "question": "No safe independent validation agent exists. How should we proceed?",
    },
    "request_manager_decision": {
        "action_type": "request_manager_decision",
        "title": "Approve revised scope?",
        "question": "The clarified scope changes required work functions. Approve?",
    },
    "record_authority_decision": {
        "action_type": "record_authority_decision",
        "decision_id": _id(),
        "artifact_id": _id(),
    },
    "cancel_pending_decision": {
        "action_type": "cancel_pending_decision",
        "decision_id": _id(),
        "reason": "Goal scope changed before the manager could answer.",
    },
}


@pytest.mark.parametrize("action_type", sorted(ALLOWED_DECISIONS))
def test_allowed_coordination_decisions_pass(action_type):
    result = validate_orchestration_decision(ALLOWED_DECISIONS[action_type])

    assert result.accepted is True
    assert result.rejection_reason is None


def test_unknown_action_type_is_rejected():
    result = validate_orchestration_decision({"action_type": "write_code", "code": "print('no')"})

    assert result.accepted is False
    assert result.rejection_reason == "Unknown orchestration action type 'write_code'"


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


def test_noop_accepts_missing_reason():
    result = validate_orchestration_decision({"action_type": "noop"})

    assert result.accepted is True
    assert result.rejection_reason is None


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


@pytest.mark.parametrize(
    ("action_type", "work_function", "expected_function"),
    [
        ("request_plan", "implementation", "planning"),
        ("request_final_summary", "implementation", "summarization"),
    ],
    ids=["request_plan", "request_final_summary"],
)
def test_work_function_must_match_action_type(action_type, work_function, expected_function):
    decision = {"action_type": action_type, "work_function": work_function}
    if action_type == "request_plan":
        decision.update({"agent_id": _id(), "scope": "Ask an agent for a plan."})

    result = validate_orchestration_decision(decision)

    assert result.accepted is False
    assert result.rejection_reason == (
        f"Decision '{action_type}' work_function must be '{expected_function}'"
    )


@pytest.mark.parametrize(
    ("decision", "missing"),
    [
        (
            {"action_type": "schedule_meeting", "topic": "Resolve contradictory validation outputs.", "participant_agent_ids": []},
            "participant_agent_ids",
        ),
        (
            {
                "action_type": "open_gate",
                "success_criterion_key": "tests-pass",
                "gate_type": "validation_passed",
                "required_evidence": {},
            },
            "required_evidence",
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
            "action_type": "open_gate",
            "success_criterion_key": "tests-pass",
            "gate_type": "validation_passed",
            "required_evidence": {coordination_key: {"required_source_types": ["task"], "min_count": 1}},
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
        "action_type": "open_gate",
        "success_criterion_key": "tests-pass",
        "gate_type": "validation_passed",
        "required_evidence": {},
    }
    cursor = decision["required_evidence"]
    for index in range(70):
        cursor["nest"] = {}
        cursor = cursor["nest"]
        cursor[f"level_{index}"] = {}

    result = validate_orchestration_decision(decision)

    assert result.accepted is False
    assert result.rejection_reason is not None
    assert result.rejection_reason.startswith("Decision nesting exceeds maximum depth at 'required_evidence.")


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
            llm_output={"action_type": "noop"},
            parsed_decision={"action_type": "noop"},
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
            llm_output={"action_type": "noop"},
            parsed_decision={"action_type": "noop"},
        )

    assert exc_info.value.status_code == 404
    result = await db_session.execute(sa.select(sa.func.count()).select_from(OrchestrationDecision))
    assert result.scalar_one() == 0
