from datetime import datetime, timedelta

import pytest
from sqlalchemy import func, select

from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationBudgetReservation,
    OrchestrationContinuousCandidate,
    OrchestrationGoal,
    OrchestrationRun,
)
from huddleroom.models.session import Session
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.workers.scheduler import reconcile_continuous_goals_async
from tests.test_orchestration_continuous_service import dispatched_discovery_source, started_continuous


pytestmark = pytest.mark.asyncio


async def test_discovery_fan_out_replay_creates_one_durable_lifecycle(db_session, test_project, test_agent, monkeypatch):
    goal, cycle, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    task.status = "done"
    session.status = "completed"
    session.output = '{"schema_version":1,"candidates":[{"origin_key":"source:1","objective":"Inspect record","source_refs":["record:1"]}]}'
    session.metadata_ = {**session.metadata_, "token_count_in": 0, "token_count_out": 0,
                         "token_usage_complete": True, "_roadmap_elapsed_seconds": 0}
    await db_session.flush()
    await db_session.refresh(cycle)
    assert (cycle.status, cycle.phase) == ("running", "authorized")
    assert cycle.plan_state["discovery"]["active_session_id"] == str(session.id)
    for _ in range(4):
        await continuous.orchestration.tick(db_session, cycle.id)

    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == cycle.id,
        OrchestrationAction.action_type == "create_delegation_task",
        OrchestrationAction.target_id == task.id,
    )) == 1
    assert await db_session.scalar(select(func.count()).select_from(Session).where(Session.task_id == task.id)) == 1
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.discovery_run_id == cycle.id,
    )) == 1
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == cycle.id,
        OrchestrationAction.action_type == "report_consumed",
    )) == 1
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationContinuousCandidate).where(
        OrchestrationContinuousCandidate.parent_goal_id == goal.id,
    )) == 1
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationGoal).where(
        OrchestrationGoal.parent_goal_id == goal.id,
    )) == 1
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == cycle.id,
        OrchestrationAction.action_type == "complete_cycle",
    )) == 1


async def test_discovery_overflow_recovery_clears_only_overflow_blocker_once(
    db_session, test_project, test_agent, monkeypatch,
):
    goal, cycle, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch, max_active_cases=1, max_backlog=1,
    )
    policy = cycle.plan_state["continuous_policy"]
    holding_child = await continuous._release_direct_child(  # pylint: disable=protected-access
        db_session, goal, cycle, policy, policy["child_template"], datetime.now(),
    )
    task.status = "done"
    session.status = "completed"
    session.output = ('{"schema_version":1,"candidates":['
                      '{"origin_key":"source:1","objective":"First","source_refs":["record:1"]},'
                      '{"origin_key":"source:2","objective":"Second","source_refs":["record:2"]}]}')
    session.metadata_ = {**session.metadata_, "token_count_in": 0, "token_count_out": 0,
                         "token_usage_complete": True, "_roadmap_elapsed_seconds": 0}

    assert (await continuous.advance(db_session, goal, cycle))["step"] == "needs_attention"
    assert any(item["kind"] == "continuous_backlog_overflow" for item in cycle.active_blockers)
    holding_child.status = "completed"
    holding_run = await OrchestrationService().get_run_for_goal(db_session, test_project.id, holding_child.id)
    holding_run.status = holding_run.phase = "completed"
    await continuous.settle_terminal_children(db_session, goal)

    assert (await continuous.advance(db_session, goal, cycle))["outcome"] == "children_created"
    assert not any(item["kind"] == "continuous_backlog_overflow" for item in cycle.active_blockers)
    await continuous.advance(db_session, goal, cycle)
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationContinuousCandidate).where(
        OrchestrationContinuousCandidate.parent_goal_id == goal.id,
    )) == 2
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == cycle.id,
        OrchestrationAction.action_type == "release_continuous_child",
    )) == 2


async def test_scheduler_claims_creates_one_child_and_leaves_successor_without_waiting(
    db_session, test_project,
):
    goal, cycle = await started_continuous(db_session, test_project)
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])

    assert await reconcile_continuous_goals_async(db_session, due) == 1
    assert cycle.phase == "authorized"

    # Tick only this claimed parent: a global sweep would also execute the new Outcome child.
    await OrchestrationService().tick(db_session, cycle.id)
    child = await db_session.scalar(select(OrchestrationGoal).where(
        OrchestrationGoal.parent_goal_id == goal.id,
    ))
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    successor = await OrchestrationService().get_active_run_for_goal(db_session, test_project.id, goal.id)

    assert cycle.status == cycle.phase == "completed"
    assert child.status == "active" and child_run.phase == "authorized"
    assert successor.phase == "waiting_activation" and successor.cycle_key is None
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == cycle.id,
        OrchestrationAction.action_type == "complete_cycle",
    )) == 1


async def test_restart_replay_duplicates_no_cycle_child_reservation_or_successor(db_session, test_project):
    goal, cycle = await started_continuous(db_session, test_project)
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])

    await reconcile_continuous_goals_async(db_session, due)
    await OrchestrationService().tick(db_session, cycle.id)
    for _ in range(3):
        # The periodic Continuous reconciler may safely replay durable state;
        # do not globally tick the child created by the first parent tick.
        await reconcile_continuous_goals_async(db_session, due)

    assert await db_session.scalar(select(func.count()).select_from(OrchestrationGoal).where(
        OrchestrationGoal.parent_goal_id == goal.id,
    )) == 1
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.parent_goal_id == goal.id,
    )) == 1
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationRun).where(
        OrchestrationRun.goal_id == goal.id,
    )) == 2


async def test_missed_child_completion_event_is_settled_on_periodic_reconcile(db_session, test_project):
    goal, cycle = await started_continuous(db_session, test_project)
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    await reconcile_continuous_goals_async(db_session, due)
    await OrchestrationService().tick(db_session, cycle.id)
    child = await db_session.scalar(select(OrchestrationGoal).where(OrchestrationGoal.parent_goal_id == goal.id))
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    child.status = "completed"
    child_run.status = child_run.phase = "completed"
    child_run.completed_at = due + timedelta(minutes=1)
    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.child_goal_id == child.id,
    ))
    assert reservation.status == "active"

    await reconcile_continuous_goals_async(db_session, due + timedelta(minutes=1))

    assert reservation.status == "settled" and reservation.settled_at is not None


async def test_cancelled_continuous_parent_cannot_be_revived_by_late_child_completion(
    db_session, test_project,
):
    goal, cycle = await started_continuous(db_session, test_project)
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    await reconcile_continuous_goals_async(db_session, due)
    await OrchestrationService().tick(db_session, cycle.id)
    child = await db_session.scalar(select(OrchestrationGoal).where(OrchestrationGoal.parent_goal_id == goal.id))
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    await OrchestrationService().cancel_goal(
        db_session, test_project.id, goal.id, cancelled_by="human:test",
    )
    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.child_goal_id == child.id,
    ))
    snapshot = (goal.status, child.status, child_run.status, reservation.status, dict(reservation.settled_spend))

    child_run.plan_state = {**child_run.plan_state, "late_output": {"claimed_complete": True}}
    await reconcile_continuous_goals_async(db_session, due + timedelta(hours=1))

    assert (goal.status, child.status, child_run.status, reservation.status, reservation.settled_spend) == snapshot
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationGoal).where(
        OrchestrationGoal.parent_goal_id == goal.id,
    )) == 1
