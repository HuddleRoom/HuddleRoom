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


async def test_judgment_fence_fingerprints_full_reports_while_payload_is_concise(db_session, test_project):
    from types import SimpleNamespace

    from huddleroom.services.orchestration_supervision_context import OrchestrationSupervisionContextBuilder
    from tests.test_orchestration_supervision_context import _report_run

    goal, run, _tasks, _markers = await _report_run(db_session, test_project.id)

    async def contract_version(db, goal, run):
        return "v1"

    service = SimpleNamespace(supervision=SimpleNamespace(_contract_version=contract_version))
    builder = OrchestrationSupervisionContextBuilder()
    payload, fingerprint = await OrchestrationSupervisionScheduler._judgment_payload(db_session, service, builder, goal, run)

    full = await builder.build(db_session, goal, run, body_limit=None, full_reports=True)
    assert fingerprint == builder.fingerprint_snapshot({**payload, "context": full})
    assert fingerprint != builder.fingerprint_snapshot(payload)  # the provider payload itself is concise
    (unrelated,) = [
        a for a in payload["context"]["actions"]
        if a["action_type"] == "report_consumed" and a["request"]["task_id"] == str(_tasks["unrelated"].id)
    ]
    assert unrelated["request"]["report"]["truncated"] == {"changes": 5, "decisions": 4}


async def test_waiver_event_wakes_only_authorized_run(db_session, test_project):
    _, run = await _run(db_session, test_project)
    scheduler = OrchestrationSupervisionScheduler()
    assert await scheduler.record_event(db_session, "meeting.action_item_waived", {"run_id": str(run.id)}) == [run.id]
    assert run.supervision_state["judgment_dirty"] is True
    run.phase = "completed"
    assert await scheduler.record_event(db_session, "meeting.action_item_waived", {"run_id": str(run.id)}) == []


async def test_goal_resumed_event_runs_one_full_tick_after_coalescing_without_local_sweep(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from huddleroom.services import orchestration_service
    from huddleroom.services.orchestration_service import OrchestrationService
    from huddleroom.workers import orchestration_tasks

    run_id = uuid.uuid4()
    ticks, local_sweeps = [], []

    async def fake_tick(self, db, rid, *, local_only=False):
        ticks.append((rid, local_only))
        return {}

    async def fake_evaluate_run(self, db, rid, *, now=None):
        local_sweeps.append(rid)
        return 0

    class FakeDb:
        async def commit(self):
            pass

        async def rollback(self):
            pass

    class FakeSession:
        async def __aenter__(self):
            return FakeDb()

        async def __aexit__(self, *exc):
            return False

    class FakeBus:
        async def subscribe(self, event_types=None):
            event = {"event_type": orchestration_service.GOAL_RESUMED_EVENT_TYPE, "payload": {"run_id": str(run_id)}}
            for _ in range(3):  # repeated/concurrent resumes coalesce to one tick
                yield SimpleNamespace(id=uuid.uuid4(), emitted_at=_utcnow(), project_id=None, **event)

    monkeypatch.setattr(OrchestrationService, "tick", fake_tick)
    monkeypatch.setattr(OrchestrationSupervisionScheduler, "evaluate_run", fake_evaluate_run)
    monkeypatch.setattr(orchestration_tasks, "AsyncSessionLocal", lambda: FakeSession())
    monkeypatch.setattr(orchestration_tasks, "get_event_bus", lambda: FakeBus())
    monkeypatch.setattr(orchestration_tasks.settings, "orchestration_event_coalesce_seconds", 0)

    await orchestration_tasks.run_orchestration_event_supervisor()
    for _ in range(100):
        if ticks:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)

    assert ticks == [(run_id, False)]
    assert local_sweeps == []


async def test_goal_resumed_event_runs_one_full_tick_and_no_local_evaluation(db_session, test_project, monkeypatch):
    from types import SimpleNamespace

    from huddleroom.models.event_log import EventLog
    from huddleroom.services import orchestration_service
    from huddleroom.workers import orchestration_tasks

    goal, run = await _run(db_session, test_project)
    event = EventLog(
        project_id=test_project.id, event_type=orchestration_service.GOAL_RESUMED_EVENT_TYPE,
        dedup_key=f"test:goal-resumed:{run.id}", payload={"run_id": str(run.id)}, source="test",
    )
    db_session.add(event)
    await db_session.flush()

    class _Session:
        async def __aenter__(self):
            return db_session

        async def __aexit__(self, *exc):
            return False

    ticks, evaluations = [], []

    async def fake_tick(self, db, run_id, local_only=True):
        ticks.append((run_id, local_only))

    async def fake_evaluate_run(self, db, run_id, now=None):
        evaluations.append(run_id)
        return 0

    monkeypatch.setattr(orchestration_tasks, "AsyncSessionLocal", lambda: _Session())
    monkeypatch.setattr(orchestration_tasks, "settings", SimpleNamespace(orchestration_event_coalesce_seconds=0))
    monkeypatch.setattr(orchestration_service.OrchestrationService, "tick", fake_tick)
    monkeypatch.setattr(orchestration_tasks.OrchestrationSupervisionScheduler, "evaluate_run", fake_evaluate_run)

    assert await orchestration_tasks.evaluate_supervision_event_async(event.id) == 1
    assert ticks == [(run.id, False)]
    assert evaluations == []
