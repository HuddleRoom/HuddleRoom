import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select

from huddleroom.config import settings
from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun, OrchestrationWait
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.orchestration_progress_view import OrchestrationProgressView, ProgressSituation
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_wake_when import ORCHESTRATOR_WAIT_OWNER_TYPE


pytestmark = pytest.mark.asyncio

NO_WORK = [{"criterion_key": "c", "state": "no_work"}]


async def _setup(db, project, agent, *, tasks=1, accepted=True):
    goal = OrchestrationGoal(project_id=project.id, objective="Keep moving", status="active", goal_type="outcome")
    db.add(goal)
    await db.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized")
    if accepted:
        run.plan_state = {"status": "accepted"}
    db.add(run)
    await db.flush()
    for n in range(tasks):
        task = Task(
            project_id=project.id, title=f"Work {n}", status="in_progress", assigned_to=agent.id,
            metadata_={"orchestration": {"run_id": str(run.id)}},
        )
        db.add(task)
        await db.flush()
        db.add(Session(project_id=project.id, task_id=task.id, agent_id=agent.id, adapter_type="api", status="running"))
    await db.flush()
    return goal, run


async def _orch_wait(db, run):
    db.add(OrchestrationWait(
        run_id=run.id, wait_key=f"run:{run.id}:wait:orch", owner={"type": ORCHESTRATOR_WAIT_OWNER_TYPE, "id": "x"},
        awaited_event={"event_type": "x.y", "matcher": {}}, fallback={"action_type": "continue"},
        due_recheck_at=_utcnow() + timedelta(hours=1), status="open",
    ))
    await db.flush()


def _situation(monkeypatch, **kwargs):
    async def build_safe(self, db, goal, run):
        return ProgressSituation(**kwargs)
    monkeypatch.setattr(OrchestrationProgressView, "build_safe", build_safe)


async def _reconcile(db, goal, run):
    return await OrchestrationService().supervision.reconcile_local(db, goal, run)


async def test_system_wait_with_untracked_follow_up_continues(db_session, test_project, test_agent, monkeypatch):
    goal, run = await _setup(db_session, test_project, test_agent)
    _situation(monkeypatch, untracked_follow_ups=[{"id": "f1"}])
    assert await _reconcile(db_session, goal, run) == {"outcome": "continue"}
    assert run.supervision_state["last_assessment"]["outcome"] == "continue"


async def test_system_wait_without_proactive_work_waits(db_session, test_project, test_agent, monkeypatch):
    goal, run = await _setup(db_session, test_project, test_agent)
    _situation(monkeypatch)
    assert (await _reconcile(db_session, goal, run))["outcome"] == "waiting"


async def test_system_wait_with_no_work_criterion_below_cap_continues(db_session, test_project, test_agent, monkeypatch):
    goal, run = await _setup(db_session, test_project, test_agent, tasks=1)
    _situation(monkeypatch, progress_view=NO_WORK)
    assert await _reconcile(db_session, goal, run) == {"outcome": "continue"}


async def test_no_work_criterion_at_cap_waits(db_session, test_project, test_agent, monkeypatch):
    goal, run = await _setup(db_session, test_project, test_agent, tasks=OrchestrationService.RELEASE_TWO_TASK_CAP)
    _situation(monkeypatch, progress_view=NO_WORK)
    assert (await _reconcile(db_session, goal, run))["outcome"] == "waiting"


async def test_unaccepted_plan_system_wait_still_waits(db_session, test_project, test_agent, monkeypatch):
    goal, run = await _setup(db_session, test_project, test_agent, accepted=False)
    _situation(monkeypatch, untracked_follow_ups=[{"id": "f1"}])
    assert (await _reconcile(db_session, goal, run))["outcome"] == "waiting"


async def test_orchestrator_wait_with_proactive_work_still_waits(db_session, test_project, test_agent, monkeypatch):
    goal, run = await _setup(db_session, test_project, test_agent, tasks=0)
    await _orch_wait(db_session, run)
    _situation(monkeypatch, untracked_follow_ups=[{"id": "f1"}], progress_view=NO_WORK)
    assert (await _reconcile(db_session, goal, run))["outcome"] == "waiting"


async def test_mixed_system_and_orchestrator_waits_wait(db_session, test_project, test_agent, monkeypatch):
    goal, run = await _setup(db_session, test_project, test_agent)
    await _orch_wait(db_session, run)
    _situation(monkeypatch, untracked_follow_ups=[{"id": "f1"}])
    assert (await _reconcile(db_session, goal, run))["outcome"] == "waiting"


async def _past_orch_wait(db, run, **fallback):
    """A consumed orchestrator wait: history that records what was already known."""
    db.add(OrchestrationWait(
        run_id=run.id, wait_key=f"run:{run.id}:wait:orch", owner={"type": ORCHESTRATOR_WAIT_OWNER_TYPE, "id": "x"},
        awaited_event={"event_type": "x.y", "matcher": {}}, fallback={"action_type": "continue", **fallback},
        due_recheck_at=_utcnow() - timedelta(hours=1), status="cleared", cleared_at=_utcnow(),
    ))
    await db.flush()


async def test_same_follow_ups_as_last_wait_does_not_continue(db_session, test_project, test_agent, monkeypatch):
    goal, run = await _setup(db_session, test_project, test_agent)
    await _past_orch_wait(db_session, run, follow_up_ids=["f1"], no_work_ids=["c"])
    _situation(monkeypatch, untracked_follow_ups=[{"id": "f1"}], progress_view=NO_WORK)
    assert (await _reconcile(db_session, goal, run))["outcome"] == "waiting"


async def test_new_follow_up_since_last_wait_continues(db_session, test_project, test_agent, monkeypatch):
    goal, run = await _setup(db_session, test_project, test_agent)
    await _past_orch_wait(db_session, run, follow_up_ids=["f1"], no_work_ids=[])
    _situation(monkeypatch, untracked_follow_ups=[{"id": "f1"}, {"id": "f2"}])
    assert await _reconcile(db_session, goal, run) == {"outcome": "continue"}


async def test_new_no_work_criterion_since_last_wait_continues(db_session, test_project, test_agent, monkeypatch):
    goal, run = await _setup(db_session, test_project, test_agent)
    await _past_orch_wait(db_session, run, follow_up_ids=[], no_work_ids=["other"])
    _situation(monkeypatch, progress_view=NO_WORK)
    assert await _reconcile(db_session, goal, run) == {"outcome": "continue"}


async def test_no_prior_orchestrator_wait_continues(db_session, test_project, test_agent, monkeypatch):
    goal, run = await _setup(db_session, test_project, test_agent)
    _situation(monkeypatch, untracked_follow_ups=[{"id": "f1"}], progress_view=NO_WORK)
    assert await _reconcile(db_session, goal, run) == {"outcome": "continue"}


async def test_progress_builder_error_falls_back_to_waiting(db_session, test_project, test_agent, monkeypatch):
    goal, run = await _setup(db_session, test_project, test_agent)
    _situation(monkeypatch, error=True, untracked_follow_ups=[{"id": "f1"}], progress_view=NO_WORK)
    assert (await _reconcile(db_session, goal, run))["outcome"] == "waiting"


async def _session_wait(db, project, run):
    """An open wait owned by the live session of the run's task: a system wait."""
    session = await db.scalar(select(Session).where(Session.project_id == project.id, Session.status == "running"))
    db.add(OrchestrationWait(
        run_id=run.id, wait_key=f"run:{run.id}:wait:sys", owner={"type": "session", "id": str(session.id)},
        awaited_event={"event_type": "x.y", "matcher": {}}, fallback={"action_type": "continue"},
        due_recheck_at=_utcnow() + timedelta(hours=1), status="open",
    ))
    await db.flush()


async def _post_action_wait(db, run):
    """The wait the act-until-wait loop creates after a non-noop action, snapshotting the state after it."""
    await OrchestrationService().supervision.create_orchestrator_waits(
        db, run, owner_id=uuid.uuid4(),
        wake_when={"recheck_after_seconds": settings.orchestration_reconcile_interval_seconds, "expected_result": "Post-action recheck"},
    )
    await db.flush()


async def _orchestrator_waits(db, run):
    waits = (await db.scalars(select(OrchestrationWait).where(OrchestrationWait.run_id == run.id))).all()
    return [w for w in waits if (w.owner or {}).get("type") == ORCHESTRATOR_WAIT_OWNER_TYPE]


async def _reconcile_no_release(db, goal, run):
    # ponytail: the fixture's accepted plan has no items, so release would block the run on the first pass; the wait decision does not depend on release.
    return await OrchestrationService().supervision.reconcile_local(db, goal, run, allow_release=False)


async def test_post_action_wait_snapshot_suppresses_decide_while_work(db_session, test_project, test_agent, monkeypatch):
    goal, run = await _setup(db_session, test_project, test_agent)
    await _session_wait(db_session, test_project, run)
    _situation(monkeypatch, untracked_follow_ups=[{"id": "f1"}])
    await _post_action_wait(db_session, run)
    assert (await _reconcile_no_release(db_session, goal, run))["outcome"] == "waiting"

    (post,) = await _orchestrator_waits(db_session, run)
    assert post.fallback["follow_up_ids"] == ["f1"]
    post.status, post.cleared_at = "cleared", _utcnow()
    await db_session.flush()

    assert (await _reconcile_no_release(db_session, goal, run))["outcome"] == "waiting"


async def test_new_follow_up_after_post_action_wait_continues(db_session, test_project, test_agent, monkeypatch):
    goal, run = await _setup(db_session, test_project, test_agent)
    await _session_wait(db_session, test_project, run)
    _situation(monkeypatch, untracked_follow_ups=[{"id": "f1"}])
    await _post_action_wait(db_session, run)
    assert (await _reconcile_no_release(db_session, goal, run))["outcome"] == "waiting"

    _situation(monkeypatch, untracked_follow_ups=[{"id": "f1"}, {"id": "f2"}])
    assert await _reconcile_no_release(db_session, goal, run) == {"outcome": "continue"}
    assert [w.status for w in await _orchestrator_waits(db_session, run)] == ["cleared"]
