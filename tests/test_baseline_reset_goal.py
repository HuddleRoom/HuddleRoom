"""Reset-goal feature: OrchestrationService.reset_goal / POST .../reset.

Advances a goal far enough through the baseline flow (goal_definition,
manager_selection, agent_definition_review, team_hierarchy) that real rows
exist in every table reset_goal is supposed to wipe, then asserts the reset
leaves a genuinely fresh goal + single fresh run behind, and that run-keyed
children (decisions/actions/gates/evidence/agent_suggestions) are gone too
(cascade via the deleted run row).
"""
import uuid
from huddleroom.models.base import _utcnow

import pytest
from sqlalchemy import func, select

from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationAgentSuggestion,
    OrchestrationDecision,
    OrchestrationEvidence,
    OrchestrationGate,
    OrchestrationGoal,
    OrchestrationRun,
)
from huddleroom.models.orchestration_memory import OrchestrationMemorySection
from huddleroom.models.orchestration_process import (
    OrchestrationAgentReview,
    OrchestrationAuthorityDecision,
    OrchestrationProcessRun,
    OrchestrationWarning,
)


async def _count(db_session, model, **filters):
    query = select(func.count(model.id))
    for column, value in filters.items():
        query = query.where(getattr(model, column) == value)
    return await db_session.scalar(query)


@pytest.mark.asyncio
async def test_reset_goal_wipes_history_and_returns_fresh_goal_and_run(
    db_session, test_project, test_user, client
):
    from tests.conftest import complete_baseline_processes

    # Multiple success criteria -> heuristic weight is "substantial".
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Deliver the reporting module",
        success_criteria=[
            {"key": "a", "description": "first"},
            {"key": "b", "description": "second"},
            {"key": "c", "description": "third"},
        ],
        created_by_user_id=test_user.id,
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    db_session.add(run)
    await db_session.flush()
    old_run_id = run.id

    # Force weight lighter than the heuristic -- creates a goal-keyed
    # OrchestrationWarning and sets weight_overridden_by.
    override = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/weight",
        json={"weight": "standard", "reason": "reduce ceremony for this pass"},
    )
    assert override.status_code == 200
    assert goal.weight_overridden_by is not None

    # Advance the full baseline flow: produces process runs, an answered
    # authority decision (manager selection), a memory section per process,
    # and (via manager selection) manager_user_id / authority_model.
    await complete_baseline_processes(db_session, goal, run)
    assert goal.authority_model == "human_manager"
    assert goal.manager_user_id == test_user.id

    # A later manual mutation to orchestrator_context, so the reset's
    # {} assertion is meaningful rather than trivially still-empty.
    goal.orchestrator_context = {
        "assumptions": [{"text": "test assumption", "destination": "orchestrator_context.assumptions"}]
    }
    await db_session.flush()

    # Seed one row per run-keyed child table directly, to prove the reset's
    # run deletion actually cascades rather than these tables merely never
    # having been populated by the baseline flow.
    gate = OrchestrationGate(
        run_id=run.id, success_criterion_key="a", gate_type="manual",
    )
    db_session.add(gate)
    await db_session.flush()
    db_session.add_all([
        OrchestrationDecision(run_id=run.id, decision_type="test_decision"),
        OrchestrationAction(
            run_id=run.id, idempotency_key="seed-action", action_type="test_action",
        ),
        OrchestrationEvidence(run_id=run.id, gate_id=gate.id, source_type="test"),
        OrchestrationAgentSuggestion(
            run_id=run.id, missing_work_function="planning", reason="test seed",
        ),
    ])
    await db_session.flush()

    assert await _count(db_session, OrchestrationProcessRun, goal_id=goal.id) > 0
    assert await _count(db_session, OrchestrationAuthorityDecision, goal_id=goal.id) > 0
    assert await _count(db_session, OrchestrationWarning, goal_id=goal.id) > 0
    assert await _count(db_session, OrchestrationMemorySection, goal_id=goal.id) > 0
    assert await _count(db_session, OrchestrationDecision, run_id=old_run_id) == 1
    assert await _count(db_session, OrchestrationAction, run_id=old_run_id) == 1
    assert await _count(db_session, OrchestrationGate, run_id=old_run_id) == 1
    assert await _count(db_session, OrchestrationEvidence, run_id=old_run_id) == 1
    assert await _count(db_session, OrchestrationAgentSuggestion, run_id=old_run_id) == 1

    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/reset"
    )
    assert resp.status_code == 200
    body = resp.json()

    # Goal-keyed tables fully wiped.
    assert await _count(db_session, OrchestrationProcessRun, goal_id=goal.id) == 0
    assert await _count(db_session, OrchestrationWarning, goal_id=goal.id) == 0
    assert await _count(db_session, OrchestrationAuthorityDecision, goal_id=goal.id) == 0
    assert await _count(db_session, OrchestrationAgentReview, goal_id=goal.id) == 0
    assert await _count(db_session, OrchestrationMemorySection, goal_id=goal.id) == 0

    # Run-keyed children of the OLD run are gone via cascade delete of the run row.
    assert await _count(db_session, OrchestrationDecision, run_id=old_run_id) == 0
    assert await _count(db_session, OrchestrationAction, run_id=old_run_id) == 0
    assert await _count(db_session, OrchestrationGate, run_id=old_run_id) == 0
    assert await _count(db_session, OrchestrationEvidence, run_id=old_run_id) == 0
    assert await _count(db_session, OrchestrationAgentSuggestion, run_id=old_run_id) == 0

    # Exactly one fresh run for the goal, running / not completed.
    runs = (
        await db_session.execute(select(OrchestrationRun).where(OrchestrationRun.goal_id == goal.id))
    ).scalars().all()
    assert len(runs) == 1
    new_run = runs[0]
    assert new_run.id != old_run_id
    assert new_run.status == "running"
    assert new_run.completed_at is None
    assert body["run"]["id"] == str(new_run.id)
    assert body["run"]["status"] == "running"
    assert body["run"]["completed_at"] is None

    # Goal columns reset to fresh state.
    await db_session.refresh(goal)
    assert goal.orchestrator_context == {}
    assert goal.authority_model is None
    assert goal.manager_agent_id is None
    assert goal.manager_user_id is None
    assert goal.weight_overridden_by is None
    assert goal.status == "active"
    assert body["goal"]["orchestrator_context"] == {}
    assert body["goal"]["authority_model"] is None
    assert body["goal"]["manager_agent_id"] is None
    assert body["goal"]["manager_user_id"] is None
    assert body["goal"]["weight_overridden_by"] is None
    assert body["goal"]["status"] == "active"


@pytest.mark.asyncio
async def test_reset_goal_unknown_goal_404(client, test_project):
    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{uuid.uuid4()}/reset"
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("lineage", ["supersedes_run_id", "terminal_reason"])
async def test_reset_goal_refuses_type_correction_lineage(db_session, test_project, lineage):
    from fastapi import HTTPException

    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService

    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session, test_project.id,
        OrchestrationGoalCreate(objective="Corrected.", success_criteria=[{"key": "k", "description": "d"}]),
        created_by_user_id=None,
    )
    run.status = "cancelled"
    if lineage == "terminal_reason":
        run.terminal_reason = "reclassified"
    await db_session.flush()
    if lineage == "supersedes_run_id":
        db_session.add(OrchestrationRun(goal_id=goal.id, supersedes_run_id=run.id))
        await db_session.flush()

    with pytest.raises(HTTPException) as exc:
        await service.reset_goal(db_session, test_project.id, goal.id)

    assert exc.value.status_code == 409
    assert await _count(db_session, OrchestrationRun, goal_id=goal.id) >= 1


@pytest.mark.asyncio
async def test_reset_goal_wipes_steering_rows_blocking_run_delete(db_session, test_project, test_user, client):
    from huddleroom.models.orchestration_conversation import (
        ConversationFeedback, ConversationMessage, ConversationReservation, ConversationResponse,
    )
    from huddleroom.services.orchestration_conversation_dossier import ConversationDossierBuilder
    from huddleroom.services.orchestration_service import OrchestrationService
    from huddleroom.models.orchestration_steering import (
        OrchestrationSteeringProposal,
        OrchestrationSteeringRequest,
        OrchestrationSteeringResultLink,
        OrchestrationSteeringState,
        OrchestrationSteeringTransition,
        steering_result_link_id,
    )

    goal = OrchestrationGoal(
        project_id=test_project.id, objective="Steer me",
        success_criteria=[{"key": "a", "description": "first"}], created_by_user_id=test_user.id,
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    db_session.add(run)
    await db_session.flush()
    decision = OrchestrationDecision(run_id=run.id, decision_type="test_decision")
    action = OrchestrationAction(run_id=run.id, idempotency_key="steer-action", action_type="test_action")
    message = ConversationMessage(
        goal_id=goal.id, actor_id=test_user.id, client_request_id=uuid.uuid4(), sequence=1, content="q",
    )
    db_session.add_all([decision, action, message])
    await db_session.flush()
    response = ConversationResponse(
        message_id=message.id, run_id=run.id, dossier={}, context_manifest={}, context_version="v1",
        provider_request_id="rally-chat:reset-steering",
        status="completed", answer="Previous answer", started_at=_utcnow(), deadline_at=_utcnow(),
        finished_at=_utcnow(),
    )
    db_session.add(response)
    await db_session.flush()
    db_session.add_all([
        ConversationFeedback(id=uuid.uuid4(), response_id=response.id, actor_id=test_user.id, rating="helpful"),
        ConversationReservation(
            response_id=response.id, goal_id=goal.id, actor_id=test_user.id,
            ceiling_snapshot=100, reserved_tokens=10,
        ),
    ])
    steering_request = OrchestrationSteeringRequest(
        id=uuid.uuid4(), goal_id=goal.id, actor_id=test_user.id, client_request_id=uuid.uuid4(), sequence=1,
        directive="x", target_type="goal", target_id=str(goal.id), scope="goal", lifetime="future_runs",
        impact_summary="x", status="applied", reason_code="submitted", contract_version="none", plan_version="none",
    )
    proposal = OrchestrationSteeringProposal(
        id=uuid.uuid4(), response_id=response.id, goal_id=goal.id, actor_id=test_user.id,
        status="proposed", draft={},
    )
    db_session.add_all([steering_request, proposal])
    await db_session.flush()
    db_session.add_all([
        OrchestrationSteeringTransition(
            id=uuid.uuid4(), request_id=steering_request.id, sequence=1, from_status=None,
            to_status="pending", reason_code="submitted", actor="user",
        ),
        OrchestrationSteeringState(id=uuid.uuid4(), goal_id=goal.id),
    ])
    await db_session.flush()
    db_session.add(OrchestrationSteeringResultLink(
        id=steering_result_link_id(steering_request.id, decision.id, action.id),
        request_id=steering_request.id, decision_id=decision.id, action_id=action.id,
    ))
    await db_session.flush()

    before = await ConversationDossierBuilder(db_session).build(goal, run, "Question", [(message, response)])
    assert before.dossier["prior_turns"]

    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/reset"
    )
    assert resp.status_code == 200

    assert await _count(db_session, OrchestrationSteeringRequest, goal_id=goal.id) == 0
    assert await _count(db_session, OrchestrationSteeringProposal, goal_id=goal.id) == 0
    assert await _count(db_session, OrchestrationSteeringResultLink, request_id=steering_request.id) == 0
    assert await _count(db_session, OrchestrationSteeringTransition, request_id=steering_request.id) == 0
    assert await _count(db_session, OrchestrationSteeringState, goal_id=goal.id) == 1
    assert await _count(db_session, OrchestrationDecision, run_id=run.id) == 0
    assert await _count(db_session, ConversationMessage, goal_id=goal.id) == 0
    assert await _count(db_session, ConversationResponse, message_id=message.id) == 0
    assert await _count(db_session, ConversationFeedback, response_id=response.id) == 0
    assert await _count(db_session, ConversationReservation, response_id=response.id) == 0
    fresh_run = await OrchestrationService().get_run_for_goal(db_session, test_project.id, goal.id)
    prior_turns = (await db_session.execute(
        select(ConversationMessage, ConversationResponse).join(ConversationResponse)
        .where(ConversationMessage.goal_id == goal.id)
    )).all()
    dossier = await ConversationDossierBuilder(db_session).build(goal, fresh_run, "Fresh question", prior_turns)
    assert dossier.dossier["prior_turns"] == []
