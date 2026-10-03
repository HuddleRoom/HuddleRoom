"""Orchestrator override: a `<process_type>_stale_inputs` suggestion that a
human explicitly DISMISSED (resolved_reason == "dismissed by human") must
stop the baseline stale-readiness gate (orchestration_service.py's
`_baseline_readiness_reason`) from 409-ing Start for that specific
staleness. A pending (unresolved) or absent suggestion must not change the
gate's behavior -- it remains a safety net. Approving a rerun instead
supersedes the process run entirely, which re-derives readiness fresh.
"""
import pytest
import pytest_asyncio

from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_team_hierarchy import TeamHierarchyProcess
from huddleroom.services.orchestration_warning_service import (
    STALE_INPUTS_DISMISSED_REASON,
    OrchestrationWarningService,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("safe_goal_analysis")]


@pytest_asyncio.fixture
async def stale_hierarchy_goal(db_session, test_project, test_user):
    """A trivial goal with a completed team_hierarchy row that's gone stale
    (objective changed after completion) and the other three baseline
    processes terminal, so `_baseline_readiness_reason` reaches the
    team_hierarchy staleness check."""
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Ship the widget",
        success_criteria=[{"key": "done", "description": "it is done"}],
        weight="trivial",
        created_by_user_id=test_user.id,
        authority_model="human_manager",
        manager_user_id=test_user.id,
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    db_session.add(run)
    await db_session.flush()

    process_service = OrchestrationProcessService()
    # goal_definition/manager_selection have no fingerprint re-check in the
    # readiness gate, so a bare completed row is enough. agent_definition_review
    # is seeded "skipped" to bypass its own coverage-fingerprint re-check --
    # only team_hierarchy's staleness is under test here.
    gd = await process_service.start_process(
        db_session, goal.id, process_type="goal_definition",
        trigger_reason="test seed", run_id=run.id,
    )
    await process_service.complete_process(db_session, gd)
    ms = await process_service.start_process(
        db_session, goal.id, process_type="manager_selection",
        trigger_reason="test seed", run_id=run.id,
    )
    await process_service.complete_process(db_session, ms)
    await process_service.skip_process(
        db_session, goal.id, process_type="agent_definition_review",
        skipped_by="human:test-seed", reason="test seed", run_id=run.id,
    )

    hierarchy = TeamHierarchyProcess()
    summary = await hierarchy.advance(db_session, goal, run)
    assert summary["status"] == "completed"
    current = await process_service.get_current(db_session, goal.id, "team_hierarchy")

    # Drift the fingerprint: the stored completed row now disagrees with
    # what team_hierarchy would compute from current inputs.
    goal.objective = "Ship a completely different widget"
    await db_session.flush()

    return goal, run, current


async def test_stale_team_hierarchy_blocks_readiness_by_default(db_session, stale_hierarchy_goal):
    goal, _run, _current = stale_hierarchy_goal

    reason = await OrchestrationService()._baseline_readiness_reason(db_session, goal.id)

    assert reason is not None
    assert "team_hierarchy is stale" in reason


async def test_pending_suggestion_does_not_change_readiness_gate(db_session, stale_hierarchy_goal):
    goal, run, current = stale_hierarchy_goal
    warning_service = OrchestrationWarningService()
    await warning_service.suggest_stale_inputs(
        db_session, goal.id, process_type="team_hierarchy",
        process_run_id=current.id, step_label="team hierarchy", run_id=run.id,
    )

    reason = await OrchestrationService()._baseline_readiness_reason(db_session, goal.id)

    assert reason is not None
    assert "team_hierarchy is stale" in reason


async def test_wrong_resolution_reason_does_not_suppress_the_gate(db_session, stale_hierarchy_goal):
    goal, run, current = stale_hierarchy_goal
    warning_service = OrchestrationWarningService()
    warning = await warning_service.suggest_stale_inputs(
        db_session, goal.id, process_type="team_hierarchy",
        process_run_id=current.id, step_label="team hierarchy", run_id=run.id,
    )
    # Approved (not dismissed) -- must NOT suppress the safety-net check.
    await warning_service.resolve_warning(
        db_session, warning, resolved_by="human:1", reason="approved rerun"
    )

    reason = await OrchestrationService()._baseline_readiness_reason(db_session, goal.id)

    assert reason is not None
    assert "team_hierarchy is stale" in reason


async def test_dismissed_suggestion_unblocks_start_for_that_staleness(db_session, stale_hierarchy_goal):
    goal, run, current = stale_hierarchy_goal
    warning_service = OrchestrationWarningService()
    warning = await warning_service.suggest_stale_inputs(
        db_session, goal.id, process_type="team_hierarchy",
        process_run_id=current.id, step_label="team hierarchy", run_id=run.id,
    )
    await warning_service.resolve_warning(
        db_session, warning, resolved_by="human:1", reason=STALE_INPUTS_DISMISSED_REASON
    )

    reason = await OrchestrationService()._baseline_readiness_reason(db_session, goal.id)

    assert reason is None
