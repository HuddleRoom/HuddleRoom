import uuid

import pytest
from sqlalchemy import select

from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision, OrchestrationWarning
from huddleroom.services.orchestration_decision_validator import ALLOWED_ACTION_SCHEMAS
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_supervision import (
    DISPOSITIONS,
    OrchestrationSupervisionService,
    SupervisionAssessment,
    disposition_request_fields,
)

EXPECTED_FIELDS = {
    "continue": (frozenset({"wake_when"}), frozenset()),
    "pause": (frozenset(), frozenset()),
    "follow_up": (
        frozenset({"agent_id", "deliverable", "scope", "parent_task_id"}),
        frozenset({"inputs", "forbidden_work", "success_evidence", "budget", "report_schema", "source_session_id"}),
    ),
    "verify": (frozenset({"gate_id"}), frozenset()),
    "reassign": (frozenset({"agent_id", "task_id"}), frozenset()),
    "meeting": (
        frozenset({"participant_agent_ids", "topic"}),
        frozenset({"task_id", "gate_id", "organizer_agent_id"}),
    ),
    "graph": (frozenset({"graph_id", "subject_id", "subject_type"}), frozenset()),
    "replan": (frozenset({"agent_id", "scope"}), frozenset()),
    "ask_human": (
        frozenset({"question"}),
        frozenset({"work_function", "required_capabilities", "candidate_agent_ids", "gate_id"}),
    ),
    "attention": (frozenset(), frozenset()),
}


def _assessment(**disposition):
    return SupervisionAssessment(disposition={
        "action_type": "continue", "origin": "test", "reason": "Why", "expected_result": "Expected", "contract_version": "start",
        **disposition,
    })


async def _run(db, project):
    goal = OrchestrationGoal(project_id=project.id, objective="Ask owner", status="active", goal_type="outcome")
    db.add(goal)
    await db.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized")
    db.add(run)
    await db.flush()
    return goal, run


def test_every_disposition_has_pinned_request_fields():
    assert set(EXPECTED_FIELDS) == set(DISPOSITIONS)
    for kind in DISPOSITIONS:
        required, optional = disposition_request_fields(kind)
        assert isinstance(required, frozenset) and isinstance(optional, frozenset), kind
        assert (required, optional) == EXPECTED_FIELDS[kind], kind


def test_every_mapped_action_type_is_an_allowed_action_with_an_executor():
    for kind in DISPOSITIONS:
        executor, action_type, _ = OrchestrationSupervisionService._mapping(kind, {}, "reason")
        assert action_type in ALLOWED_ACTION_SCHEMAS, kind
        assert callable(getattr(OrchestrationService, executor, None)), kind


def test_unknown_disposition_has_no_mapping():
    with pytest.raises(KeyError):
        OrchestrationSupervisionService._mapping("nope", {}, "reason")


@pytest.mark.asyncio
async def test_ask_human_disposition_creates_pending_authority_decision(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    await OrchestrationService().supervision.apply_disposition(
        db_session, goal, run, _assessment(action_type="ask_human", request={"question": "Ship the release now?"}),
    )
    await db_session.flush()
    decisions = list((await db_session.scalars(
        select(OrchestrationAuthorityDecision).where(OrchestrationAuthorityDecision.run_id == run.id)
    )).all())
    assert len(decisions) == 1
    assert decisions[0].status == "pending"
    assert decisions[0].question == "Ship the release now?"


@pytest.mark.asyncio
async def test_attention_disposition_records_warning_without_authority_decision(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    await OrchestrationService().supervision.apply_disposition(db_session, goal, run, _assessment(action_type="attention"))
    await db_session.flush()
    warnings = list((await db_session.scalars(
        select(OrchestrationWarning).where(OrchestrationWarning.run_id == run.id)
    )).all())
    decisions = list((await db_session.scalars(
        select(OrchestrationAuthorityDecision).where(OrchestrationAuthorityDecision.run_id == run.id)
    )).all())
    assert [w.warning_type for w in warnings] == ["supervision_attention"]
    assert decisions == []


def test_verify_mapping_injects_validation_work_function():
    _, action_type, raw = OrchestrationSupervisionService._mapping("verify", {"gate_id": str(uuid.uuid4())}, "r")
    assert action_type == "request_verification"
    assert raw["work_function"] == "validation"


@pytest.mark.asyncio
async def test_verify_after_failed_attempt_creates_new_action(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    service = OrchestrationService()
    gate_id = str(uuid.uuid4())

    async def fake_executor(db, run_id, request, key):
        action = await service._existing_action_for_key(db, run_id, key)
        action.status = "failed"
        return action

    service.execute_request_verification_action = fake_executor
    assessment = _assessment(action_type="verify", request={"gate_id": gate_id})
    first = await service.supervision.apply_disposition(db_session, goal, run, assessment)
    second = await service.supervision.apply_disposition(db_session, goal, run, assessment)
    assert first.id != second.id
    assert first.idempotency_key != second.idempotency_key


@pytest.mark.asyncio
async def test_ask_human_same_question_new_origin_replays_one_decision(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    service = OrchestrationService()
    first = await service.supervision.apply_disposition(
        db_session, goal, run, _assessment(action_type="ask_human", origin="a", request={"question": "Ship it now?"}))
    second = await service.supervision.apply_disposition(
        db_session, goal, run, _assessment(action_type="ask_human", origin="b", request={"question": "  ship   IT now? "}))
    await db_session.flush()
    decisions = list((await db_session.scalars(
        select(OrchestrationAuthorityDecision).where(OrchestrationAuthorityDecision.run_id == run.id)
    )).all())
    assert first.id == second.id
    assert len(decisions) == 1
