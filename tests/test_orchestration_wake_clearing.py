import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select

from huddleroom.models.base import _utcnow
from huddleroom.models.meeting import Meeting, MeetingActionItem
from huddleroom.models.session import Session
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun, OrchestrationWait
from huddleroom.models.task import Task
from huddleroom.services.orchestration_progress_view import OrchestrationProgressView, ProgressSituation
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_wake_when import ORCHESTRATOR_WAIT_OWNER_TYPE
from huddleroom.services.orchestration_supervision import SYSTEM_WAIT_OWNER_TYPES

pytestmark = pytest.mark.asyncio

TASK_A, TASK_B = str(uuid.uuid4()), str(uuid.uuid4())


async def _run(db, project):
    goal = OrchestrationGoal(project_id=project.id, objective="Keep moving", status="active", goal_type="outcome")
    db.add(goal)
    await db.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized")
    db.add(run)
    await db.flush()
    return goal, run


def _spec():
    return {
        "events": [
            {"event_type": "task.status_changed", "matcher": {"task_id": TASK_A}},
            {"event_type": "task.status_changed", "matcher": {"task_id": TASK_B}},
        ],
        "expected_result": "Task moved on.",
    }


async def _make(db, run, owner_id=None):
    owner_id = owner_id or uuid.uuid4()
    await OrchestrationService().supervision.create_orchestrator_waits(db, run, owner_id=owner_id, wake_when=_spec())
    return owner_id


async def _waits(db, run):
    await db.flush()
    return list((await db.scalars(select(OrchestrationWait).where(OrchestrationWait.run_id == run.id))).all())


async def _statuses(db, run):
    return sorted(w.status for w in await _waits(db, run))


async def _system_wait(db, run):
    wait = OrchestrationWait(
        run_id=run.id, wait_key=f"run:{run.id}:wait:sys", owner={"type": "system", "id": "x"},
        awaited_event={"event_type": "x.y", "matcher": {}}, fallback={"action_type": "continue", "steering_digest": "old", "follow_up_ids": []},
        due_recheck_at=_utcnow() + timedelta(hours=1), status="open",
    )
    db.add(wait)
    await db.flush()
    return wait


async def _follow_up(db, project, run):
    task = Task(
        project_id=project.id, title="t", status="in_progress",
        metadata_={"orchestration": {"run_id": str(run.id), "success_criterion_keys": []}},
    )
    db.add(task)
    await db.flush()
    meeting = Meeting(project_id=project.id, title="m", meeting_type="standup", source_task_id=task.id)
    db.add(meeting)
    await db.flush()
    item = MeetingActionItem(meeting_id=meeting.id, description="Write the report")
    db.add(item)
    await db.flush()
    return item


async def test_event_clears_matching_wait_and_all_group_siblings(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    sup = OrchestrationService().supervision
    await _make(db_session, run)
    count = await sup.clear_matching_waits(db_session, run, event_type="task.status_changed", matcher={"task_id": TASK_A}, event_id=None)
    assert count == 2
    assert await _statuses(db_session, run) == ["cleared", "cleared"]


async def test_unrelated_event_does_not_clear_orchestrator_wait(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    await _make(db_session, run)
    count = await OrchestrationService().supervision.clear_matching_waits(
        db_session, run, event_type="task.status_changed", matcher={"task_id": str(uuid.uuid4())})
    assert count == 0
    assert await _statuses(db_session, run) == ["open", "open"]


async def test_due_orchestrator_wait_clears_group_and_runs_fallback_noop(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    await _make(db_session, run)
    result = await OrchestrationService().supervision.reconcile_local(
        db_session, goal, run, now=_utcnow() + timedelta(days=1))
    assert result["outcome"] == "due_fallback"
    assert result["fallback"]["action_type"] == "continue"
    assert await _statuses(db_session, run) == ["cleared", "cleared"]


async def test_steering_digest_change_clears_all_orchestrator_waits(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    await _make(db_session, run)
    goal.objective = "A different objective"
    await db_session.flush()
    await OrchestrationService().supervision.reconcile_local(db_session, goal, run)
    assert await _statuses(db_session, run) == ["cleared", "cleared"]


async def test_new_untracked_follow_up_clears_orchestrator_waits(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    await _make(db_session, run)
    await _follow_up(db_session, test_project, run)
    await OrchestrationService().supervision.reconcile_local(db_session, goal, run)
    assert await _statuses(db_session, run) == ["cleared", "cleared"]


async def test_known_follow_up_in_stored_ids_does_not_clear(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    item = await _follow_up(db_session, test_project, run)
    await _make(db_session, run)
    waits = await _waits(db_session, run)
    assert str(item.id) in waits[0].fallback["follow_up_ids"]
    await OrchestrationService().supervision.reconcile_local(db_session, goal, run)
    assert await _statuses(db_session, run) == ["open", "open"]


async def _live_session_wait(db, project, agent, run):
    """A system-owned wait whose live session keeps it open through the stale-owner rules."""
    task = Task(
        project_id=project.id, title="Work", status="in_progress", assigned_to=agent.id,
        metadata_={"orchestration": {"run_id": str(run.id)}},
    )
    db.add(task)
    await db.flush()
    session = Session(project_id=project.id, task_id=task.id, agent_id=agent.id, adapter_type="api", status="running")
    db.add(session)
    await db.flush()
    wait = OrchestrationWait(
        run_id=run.id, wait_key=f"run:{run.id}:wait:sys", owner={"type": "session", "id": str(session.id)},
        awaited_event={"event_type": "x.y", "matcher": {}}, fallback={"action_type": "continue"},
        due_recheck_at=_utcnow() + timedelta(hours=1), status="open",
    )
    db.add(wait)
    await db.flush()
    assert wait.owner["type"] in SYSTEM_WAIT_OWNER_TYPES
    return wait


async def test_safety_net_does_not_clear_system_waits(db_session, test_project, test_agent):
    goal, run = await _run(db_session, test_project)
    sys_wait = await _live_session_wait(db_session, test_project, test_agent, run)
    await _make(db_session, run)
    goal.objective = "Changed"
    await db_session.flush()
    await OrchestrationService().supervision.reconcile_local(db_session, goal, run)
    await db_session.refresh(sys_wait)
    assert sys_wait.status == "open"
    orch = [w for w in await _waits(db_session, run) if w.owner["type"] == ORCHESTRATOR_WAIT_OWNER_TYPE]
    assert [w.status for w in orch] == ["cleared", "cleared"]


async def test_safety_net_clear_then_proactive_work_continues(db_session, test_project, test_agent, monkeypatch):
    goal, run = await _run(db_session, test_project)
    run.plan_state = {"status": "accepted"}
    sys_wait = await _live_session_wait(db_session, test_project, test_agent, run)
    await _make(db_session, run)
    for wait in await _waits(db_session, run):
        if wait.owner["type"] == ORCHESTRATOR_WAIT_OWNER_TYPE:
            wait.fallback = {**wait.fallback, "steering_digest": "stale", "follow_up_ids": []}

    async def build_safe(self, db, goal, run):
        return ProgressSituation(untracked_follow_ups=[{"id": "f1"}])

    monkeypatch.setattr(OrchestrationProgressView, "build_safe", build_safe)
    result = await OrchestrationService().supervision.reconcile_local(db_session, goal, run)
    assert result == {"outcome": "continue"}
    await db_session.refresh(sys_wait)
    assert sys_wait.status == "open"
    orch = [w for w in await _waits(db_session, run) if w.owner["type"] == ORCHESTRATOR_WAIT_OWNER_TYPE]
    assert [w.status for w in orch] == ["cleared", "cleared"]


async def test_waits_with_same_owner_type_and_id_but_extra_keys_are_one_group(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    sup = OrchestrationService().supervision
    for n, extra in enumerate(({}, {"extra": "k"})):
        db_session.add(OrchestrationWait(
            run_id=run.id, wait_key=f"run:{run.id}:wait:g{n}", owner={"type": ORCHESTRATOR_WAIT_OWNER_TYPE, "id": "o", **extra},
            awaited_event={"event_type": "x.y", "matcher": {"k": str(n)}}, fallback={"action_type": "continue"},
            due_recheck_at=_utcnow() + timedelta(hours=1), status="open",
        ))
    await db_session.flush()
    assert await sup.clear_matching_waits(db_session, run, event_type="x.y", matcher={"k": "0"}) == 2
    assert await _statuses(db_session, run) == ["cleared", "cleared"]


async def test_orchestrator_owner_is_not_stale_and_yields_waiting(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    await _make(db_session, run)
    result = await OrchestrationService().supervision.reconcile_local(db_session, goal, run)
    assert result["outcome"] == "waiting"
    assert await _statuses(db_session, run) == ["open", "open"]


async def test_open_orchestrator_wait_makes_reconcile_return_waiting(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    await OrchestrationService().supervision.create_orchestrator_waits(db_session, run, owner_id=uuid.uuid4(), wake_when=None)
    result = await OrchestrationService().supervision.reconcile_local(db_session, goal, run)
    assert result["outcome"] == "waiting"


async def test_clear_matching_waits_count_for_non_orchestrator_waits_unchanged(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    sup = OrchestrationService().supervision
    for n in range(2):
        await sup.create_wait(
            db_session, run, origin=f"sys:{n}", owner={"type": "system", "id": "same"},
            awaited_event={"event_type": "x.y", "matcher": {"k": str(n)}}, recheck_seconds=60,
            fallback={"action_type": "continue", "reason": "r"}, expected_result="r",
        )
    assert await sup.clear_matching_waits(db_session, run, event_type="x.y", matcher={"k": "0"}) == 1
    assert await _statuses(db_session, run) == ["cleared", "open"]


async def test_accepted_roadmap_reconcile_clears_lingering_orchestrator_waits_only(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    goal.goal_type = "roadmap"
    sys_wait = await _system_wait(db_session, run)
    await _make(db_session, run)  # created while the plan was not yet accepted
    run.plan_state = {"status": "accepted"}
    result = await OrchestrationService().supervision.reconcile_local(db_session, goal, run)
    assert result == {"outcome": "continue"}
    await db_session.refresh(sys_wait)
    assert sys_wait.status == "open"
    orch = [w for w in await _waits(db_session, run) if w.owner["type"] == ORCHESTRATOR_WAIT_OWNER_TYPE]
    assert orch and all(w.status == "cleared" and w.cleared_at is not None for w in orch)


async def test_safety_net_runs_no_queries_without_orchestrator_waits(db_session, test_project, monkeypatch):
    goal, run = await _run(db_session, test_project)
    await _system_wait(db_session, run)

    async def boom(self, db, goal, run):
        raise AssertionError("progress view must not be built")

    monkeypatch.setattr(OrchestrationProgressView, "build_safe", boom)
    await OrchestrationService().supervision.reconcile_local(db_session, goal, run)
