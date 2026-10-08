import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select

from huddleroom.config import settings
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun, OrchestrationWait
from huddleroom.services.orchestration_progress_view import OrchestrationProgressView, ProgressSituation
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_steering import OrchestrationSteeringService
from huddleroom.services.orchestration_wake_when import (
    BACKSTOP_RECHECK_SECONDS,
    ORCHESTRATOR_WAIT_OWNER_TYPE,
    WAKE_RECHECK_EVENT_TYPE,
    clamp_recheck_seconds,
)


pytestmark = pytest.mark.asyncio


async def _run(db, project, *, goal_type="outcome"):
    goal = OrchestrationGoal(project_id=project.id, objective="Keep moving", status="active", goal_type=goal_type)
    db.add(goal)
    await db.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized")
    db.add(run)
    await db.flush()
    return goal, run


def _event_spec(**extra):
    return {
        "events": [{"event_type": "task.status_changed", "matcher": {"task_id": str(uuid.uuid4())}}],
        "expected_result": "Task moved on.",
        **extra,
    }


async def _waits(db, run):
    return list((await db.scalars(select(OrchestrationWait).where(OrchestrationWait.run_id == run.id))).all())


async def test_events_create_one_wait_per_event_with_group_owner_and_origin(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    supervision = OrchestrationService().supervision
    owner_id = uuid.uuid4()
    spec = {
        "events": [
            {"event_type": "task.status_changed", "matcher": {"task_id": str(uuid.uuid4())}},
            {"event_type": "graph.run_completed", "matcher": {"graph_run_id": str(uuid.uuid4())}},
        ],
        "expected_result": "Both move.",
    }
    waits = await supervision.create_orchestrator_waits(db_session, run, owner_id=owner_id, wake_when=spec)
    assert len(waits) == 2
    by_origin = {wait.wait_key: wait for wait in await _waits(db_session, run)}
    assert set(by_origin) == {f"run:{run.id}:wait:decision:{owner_id}:event:{n}" for n in (0, 1)}
    for n in (0, 1):
        wait = by_origin[f"run:{run.id}:wait:decision:{owner_id}:event:{n}"]
        assert wait.owner == {"type": ORCHESTRATOR_WAIT_OWNER_TYPE, "id": str(owner_id)}
        assert wait.awaited_event == spec["events"][n]
        assert wait.fallback["recheck_seconds"] == clamp_recheck_seconds(settings.orchestration_wake_max_seconds)
        assert wait.fallback["expected_result"] == "Both move."


async def test_recheck_only_creates_time_only_wait_with_wake_recheck_event(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    supervision = OrchestrationService().supervision
    owner_id = uuid.uuid4()
    waits = await supervision.create_orchestrator_waits(
        db_session, run, owner_id=owner_id,
        wake_when={"events": [], "recheck_after_seconds": 120, "expected_result": "Check back."},
    )
    assert len(waits) == 1
    wait = waits[0]
    assert wait.wait_key == f"run:{run.id}:wait:decision:{owner_id}:recheck"
    assert wait.awaited_event == {"event_type": WAKE_RECHECK_EVENT_TYPE, "matcher": {}}
    assert wait.fallback["recheck_seconds"] == clamp_recheck_seconds(120)


async def test_events_and_recheck_create_event_waits_plus_time_wait(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    supervision = OrchestrationService().supervision
    owner_id = uuid.uuid4()
    waits = await supervision.create_orchestrator_waits(
        db_session, run, owner_id=owner_id, wake_when=_event_spec(recheck_after_seconds=90),
    )
    assert len(waits) == 2
    keys = {wait.wait_key for wait in await _waits(db_session, run)}
    assert keys == {
        f"run:{run.id}:wait:decision:{owner_id}:event:0",
        f"run:{run.id}:wait:decision:{owner_id}:recheck",
    }


async def test_event_wait_recheck_defaults_to_wake_max(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    supervision = OrchestrationService().supervision
    waits = await supervision.create_orchestrator_waits(
        db_session, run, owner_id=uuid.uuid4(), wake_when=_event_spec(),
    )
    assert len(waits) == 1
    assert waits[0].fallback["recheck_seconds"] == clamp_recheck_seconds(settings.orchestration_wake_max_seconds)


async def test_missing_wake_when_creates_300s_backstop(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    supervision = OrchestrationService().supervision
    owner_id = uuid.uuid4()
    waits = await supervision.create_orchestrator_waits(db_session, run, owner_id=owner_id, wake_when=None)
    assert len(waits) == 1
    wait = waits[0]
    assert wait.wait_key == f"run:{run.id}:wait:decision:{owner_id}:recheck"
    assert wait.awaited_event == {"event_type": WAKE_RECHECK_EVENT_TYPE, "matcher": {}}
    assert wait.fallback["recheck_seconds"] == clamp_recheck_seconds(BACKSTOP_RECHECK_SECONDS)
    assert wait.fallback["expected_result"] == "Backstop recheck after a wait without a valid wake_when"


async def test_invalid_wake_when_creates_backstop(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    supervision = OrchestrationService().supervision
    owner_id = uuid.uuid4()
    waits = await supervision.create_orchestrator_waits(
        db_session, run, owner_id=owner_id, wake_when={"events": [{"event_type": "nope", "matcher": {}}], "expected_result": "x"},
    )
    assert len(waits) == 1
    assert waits[0].awaited_event == {"event_type": WAKE_RECHECK_EVENT_TYPE, "matcher": {}}
    assert waits[0].fallback["recheck_seconds"] == clamp_recheck_seconds(BACKSTOP_RECHECK_SECONDS)


async def test_fallback_is_continue_with_follow_up_ids_and_steering_digest_and_reason(db_session, test_project, monkeypatch):
    goal, run = await _run(db_session, test_project)
    supervision = OrchestrationService().supervision

    async def situation(self, db, goal, run):
        return ProgressSituation(untracked_follow_ups=[{"id": "b-follow-up"}, {"id": "a-follow-up"}])

    monkeypatch.setattr(OrchestrationProgressView, "build_safe", situation)
    waits = await supervision.create_orchestrator_waits(
        db_session, run, owner_id=uuid.uuid4(), wake_when=_event_spec(),
    )
    fallback = waits[0].fallback
    steering = OrchestrationSteeringService(supervision.orchestration)
    digest = steering.version_digest(await steering.current_versions(db_session, goal, run))
    assert fallback["action_type"] == "continue"
    assert fallback["reason"] == "Task moved on."
    assert fallback["follow_up_ids"] == ["a-follow-up", "b-follow-up"]
    assert fallback["steering_digest"] == digest


async def test_fallback_stores_no_work_ids_when_progress_ok(db_session, test_project, monkeypatch):
    goal, run = await _run(db_session, test_project)

    async def situation(self, db, goal, run):
        return ProgressSituation(progress_view=[
            {"criterion_key": "z", "state": "no_work"}, {"criterion_key": "m", "state": "in_progress"},
            {"criterion_key": "a", "state": "no_work"},
        ])

    monkeypatch.setattr(OrchestrationProgressView, "build_safe", situation)
    waits = await OrchestrationService().supervision.create_orchestrator_waits(
        db_session, run, owner_id=uuid.uuid4(), wake_when=_event_spec())
    assert waits[0].fallback["no_work_ids"] == ["a", "z"]


async def test_create_orchestrator_waits_returns_empty_when_goal_missing(db_session, test_project, monkeypatch):
    goal, run = await _run(db_session, test_project)

    async def no_goal(*_args, **_kwargs):
        return None

    monkeypatch.setattr(db_session, "get", no_goal)
    waits = await OrchestrationService().supervision.create_orchestrator_waits(
        db_session, run, owner_id=uuid.uuid4(), wake_when=_event_spec())
    assert waits == []


async def test_replay_creates_no_duplicate_waits(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    supervision = OrchestrationService().supervision
    owner_id = uuid.uuid4()
    spec = _event_spec(recheck_after_seconds=60)
    await supervision.create_orchestrator_waits(db_session, run, owner_id=owner_id, wake_when=spec)
    await supervision.create_orchestrator_waits(db_session, run, owner_id=owner_id, wake_when=spec)
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationWait).where(OrchestrationWait.run_id == run.id)) == 2


async def test_replay_conflict_is_swallowed(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    supervision = OrchestrationService().supervision
    owner_id = uuid.uuid4()
    await supervision.create_wait(
        db_session, run, origin=f"decision:{owner_id}:event:0",
        owner={"type": ORCHESTRATOR_WAIT_OWNER_TYPE, "id": str(owner_id)},
        awaited_event={"event_type": "task.status_changed", "matcher": {"task_id": str(uuid.uuid4())}},
        recheck_seconds=60, fallback={"action_type": "continue", "reason": "old"}, expected_result="old",
    )
    waits = await supervision.create_orchestrator_waits(db_session, run, owner_id=owner_id, wake_when=_event_spec())
    assert waits == []
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationWait).where(OrchestrationWait.run_id == run.id)) == 1


async def test_unauthorized_run_swallowed_returns_empty(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    supervision = OrchestrationService().supervision
    run.status = "paused"
    waits = await supervision.create_orchestrator_waits(db_session, run, owner_id=uuid.uuid4(), wake_when=_event_spec())
    assert waits == []
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationWait).where(OrchestrationWait.run_id == run.id)) == 0


async def test_roadmap_with_accepted_plan_creates_no_waits(db_session, test_project):
    goal, run = await _run(db_session, test_project, goal_type="roadmap")
    run.plan_state = {"status": "accepted"}
    supervision = OrchestrationService().supervision
    waits = await supervision.create_orchestrator_waits(db_session, run, owner_id=uuid.uuid4(), wake_when=_event_spec())
    assert waits == []
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationWait).where(OrchestrationWait.run_id == run.id)) == 0


async def test_recreate_after_group_cleared_opens_new_generation(db_session, test_project, monkeypatch):
    goal, run = await _run(db_session, test_project)
    supervision = OrchestrationService().supervision
    owner_id = uuid.uuid4()
    task_a, task_b = str(uuid.uuid4()), str(uuid.uuid4())
    spec = {
        "events": [
            {"event_type": "task.status_changed", "matcher": {"task_id": task_a}},
            {"event_type": "task.status_changed", "matcher": {"task_id": task_b}},
        ],
        "expected_result": "Both move.",
    }
    await supervision.create_orchestrator_waits(db_session, run, owner_id=owner_id, wake_when=spec)
    assert await supervision.clear_matching_waits(
        db_session, run, event_type="task.status_changed", matcher={"task_id": task_a}) == 2

    async def situation(self, db, goal, run):
        return ProgressSituation(untracked_follow_ups=[{"id": "new-follow-up"}])

    monkeypatch.setattr(OrchestrationProgressView, "build_safe", situation)
    waits = await supervision.create_orchestrator_waits(db_session, run, owner_id=owner_id, wake_when=spec)
    assert len(waits) == 2
    rows = await _waits(db_session, run)
    open_rows = [w for w in rows if w.status == "open"]
    assert len(rows) == 4 and len(open_rows) == 2
    assert sorted(w.wait_key for w in open_rows) == sorted(
        f"run:{run.id}:wait:decision:{owner_id}:event:{n}:g1" for n in (0, 1))
    assert all(w.fallback["follow_up_ids"] == ["new-follow-up"] for w in open_rows)
    assert sorted(w.status for w in rows) == ["cleared", "cleared", "open", "open"]


async def test_create_wait_twice_while_open_returns_same_row(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    supervision = OrchestrationService().supervision
    kwargs = dict(
        origin="decision:same:recheck", owner={"type": ORCHESTRATOR_WAIT_OWNER_TYPE, "id": "same"},
        awaited_event={"event_type": WAKE_RECHECK_EVENT_TYPE, "matcher": {}}, recheck_seconds=60,
        fallback={"action_type": "continue", "reason": "r"}, expected_result="r",
    )
    first = await supervision.create_wait(db_session, run, **kwargs)
    second = await supervision.create_wait(db_session, run, **kwargs)
    assert first.id == second.id
    assert len(await _waits(db_session, run)) == 1


async def test_system_wait_replay_after_clear_unchanged(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    supervision = OrchestrationService().supervision
    kwargs = dict(
        origin="sys:1", owner={"type": "session", "id": "s1"},
        awaited_event={"event_type": "x.y", "matcher": {}}, recheck_seconds=60,
        fallback={"action_type": "continue", "reason": "r"}, expected_result="r",
    )
    first = await supervision.create_wait(db_session, run, **kwargs)
    first.status = "cleared"
    await db_session.flush()
    again = await supervision.create_wait(db_session, run, **kwargs)
    assert again.id == first.id and again.status == "cleared"
    assert len(await _waits(db_session, run)) == 1


async def test_progress_error_at_creation_omits_follow_up_ids_and_group_survives_healthy_tick(db_session, test_project, monkeypatch):
    goal, run = await _run(db_session, test_project)
    supervision = OrchestrationService().supervision

    async def broken(self, db, goal, run):
        return ProgressSituation(error=True)

    monkeypatch.setattr(OrchestrationProgressView, "build_safe", broken)
    waits = await supervision.create_orchestrator_waits(db_session, run, owner_id=uuid.uuid4(), wake_when=_event_spec())
    assert len(waits) == 1 and "follow_up_ids" not in waits[0].fallback
    assert "no_work_ids" not in waits[0].fallback

    async def healthy(self, db, goal, run):
        return ProgressSituation(untracked_follow_ups=[{"id": "existing"}])

    monkeypatch.setattr(OrchestrationProgressView, "build_safe", healthy)
    result = await supervision.reconcile_local(db_session, goal, run)
    assert result["outcome"] == "waiting"
    assert [w.status for w in await _waits(db_session, run)] == ["open"]


async def test_non_409_http_error_from_create_wait_propagates(db_session, test_project, monkeypatch):
    goal, run = await _run(db_session, test_project)
    supervision = OrchestrationService().supervision

    async def boom(*_args, **_kwargs):
        raise HTTPException(status_code=500, detail="boom")

    monkeypatch.setattr(supervision, "create_wait", boom)
    with pytest.raises(HTTPException) as info:
        await supervision.create_orchestrator_waits(db_session, run, owner_id=uuid.uuid4(), wake_when=_event_spec())
    assert info.value.status_code == 500
