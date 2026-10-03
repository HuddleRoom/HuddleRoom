import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select

from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.services.orchestration_supervision_scheduler import OrchestrationSupervisionScheduler


pytestmark = pytest.mark.asyncio


async def _run(db, project):
    goal = OrchestrationGoal(project_id=project.id, objective="Keep moving", status="active")
    db.add(goal)
    await db.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized")
    db.add(run)
    await db.flush()
    return goal, run


async def test_record_event_returns_matching_run_ids_and_preserves_its_first_deadline(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    scheduler = OrchestrationSupervisionScheduler()
    now = _utcnow()

    assert await scheduler.record_event(db_session, "task.status_changed", {"task_id": str(uuid.uuid4())}, now=now) == []
    assert await scheduler.record_event(db_session, "task.status_changed", {"run_id": str(run.id)}, now=now) == [run.id]
    first = dict(run.supervision_state)
    assert first["judgment_dirty"] is True
    assert _utcnow() <= datetime.fromisoformat(first["judgment_due_at"]) <= now + timedelta(seconds=1)

    assert await scheduler.record_event(db_session, "task.status_changed", {"run_id": str(run.id)}, now=now + timedelta(milliseconds=500)) == [run.id]
    assert run.supervision_state["judgment_due_at"] == first["judgment_due_at"]


async def test_record_event_routes_child_run_events_to_its_active_parent(db_session, test_project):
    parent, parent_run = await _run(db_session, test_project)
    child = OrchestrationGoal(
        project_id=test_project.id, parent_goal_id=parent.id, objective="Child work", status="active",
        continuous_origin_key="child-work", parent_contract_snapshot={}, goal_delta={},
    )
    db_session.add(child)
    await db_session.flush()
    child_run = OrchestrationRun(goal_id=child.id, status="running", phase="authorized")
    db_session.add(child_run)
    await db_session.flush()

    affected = await OrchestrationSupervisionScheduler().record_event(
        db_session, "orchestration.run_completed", {"run_id": str(child_run.id)}
    )

    assert set(affected) == {child_run.id, parent_run.id}


async def test_record_event_rechecks_authorized_control_inside_the_goal_lock(db_session, test_project):
    """A run no longer authorized at mutation time must not be dirtied."""
    _goal, run = await _run(db_session, test_project)
    run.phase = "completed"

    assert await OrchestrationSupervisionScheduler().record_event(
        db_session, "task.status_changed", {"run_id": str(run.id)}
    ) == []
    assert run.supervision_state in (None, {})


async def test_steering_change_wakes_the_matching_run(db_session, test_project):
    _goal, run = await _run(db_session, test_project)

    assert await OrchestrationSupervisionScheduler().record_event(
        db_session, "orchestration.steering_changed", {"run_id": str(run.id)}
    ) == [run.id]
