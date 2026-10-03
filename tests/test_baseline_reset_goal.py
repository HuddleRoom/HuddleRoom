"""Reset-goal feature: OrchestrationService.reset_goal / POST .../reset.

Advances a goal far enough through the baseline flow (goal_definition,
manager_selection, agent_definition_review, team_hierarchy) that real rows
exist in every table reset_goal is supposed to wipe, then asserts the reset
leaves a genuinely fresh goal + single fresh run behind, and that run-keyed
children (decisions/actions/gates/evidence/agent_suggestions) are gone too
(cascade via the deleted run row).
"""
import uuid

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
