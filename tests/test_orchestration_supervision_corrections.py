"""Production-path regressions for Task 8 supervision corrections."""
# pylint: disable=cyclic-import
import asyncio
import importlib
import sys
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.models.project import Project
from huddleroom.models.task import Task
from huddleroom.services.event_bus import EventBusService, emit_event
from huddleroom.services.orchestration_supervision_scheduler import (
    OrchestrationSupervisionScheduler,
    SUPPORTED_EVENTS,
)


async def _run_with_task(db, project):
    goal = OrchestrationGoal(project_id=project.id, objective="Supervise durable work", status="active")
    db.add(goal)
    await db.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized")
    db.add(run)
    await db.flush()
    task = Task(
        project_id=project.id,
        title="Durable work",
        status="in_progress",
        metadata_={"orchestration": {"run_id": str(run.id)}},
    )
    db.add(task)
    await db.flush()
    return run, task


@pytest.mark.asyncio
async def test_sqlite_commit_uses_bus_without_the_celery_wakeup(db_session, test_project, monkeypatch):
    """The in-process subscriber is SQLite's only event-to-supervision path."""
    from huddleroom.workers import orchestration_tasks

    wakeups = []
    bus = EventBusService()
    received = []

    async def consume():
        async for event in bus.subscribe(project_id=test_project.id):
            received.append(event)
            break

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)
    monkeypatch.setattr(orchestration_tasks, "request_supervision_wakeup", wakeups.append)
    await emit_event(
        db_session, test_project.id, "task.status_changed", {"task_id": str(uuid.uuid4())}, _bus=bus,
    )
    await db_session.commit()
    await asyncio.wait_for(consumer, timeout=1)

    assert len(received) == 1
    assert not wakeups


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_postgres_wakeup_enqueues_celery_without_an_in_process_task(monkeypatch):
    """Postgres must cross the process boundary through the Celery task."""
    from huddleroom.workers import orchestration_tasks

    delayed = []
    task = SimpleNamespace(delay=delayed.append)
    monkeypatch.setattr(orchestration_tasks, "settings", SimpleNamespace(is_sqlite=False))
    monkeypatch.setattr(orchestration_tasks, "CELERY_APP", object())
    monkeypatch.setattr(orchestration_tasks, "evaluate_supervision_event", task, raising=False)

    event_id = uuid.uuid4()
    orchestration_tasks.request_supervision_wakeup(event_id)

    assert delayed == [str(event_id)]


def test_stale_meeting_completed_event_is_not_a_supervision_trigger():
    """Only the canonical concluded meeting event should wake supervision."""
    assert "meeting.completed" not in SUPPORTED_EVENTS
    assert "meeting.concluded" in SUPPORTED_EVENTS


@pytest.mark.asyncio
async def test_event_worker_rejects_a_cross_project_task_resolver(  # pylint: disable=too-many-locals
    db_session, test_project, tmp_path, monkeypatch,
):
    """A project A event must not dirty a run reached through a project B task id."""
    from huddleroom.workers import orchestration_tasks

    other = Project(name=f"Other {uuid.uuid4()}", workspace_path=str(tmp_path / "other"), config={})
    db_session.add(other)
    await db_session.flush()
    run, task = await _run_with_task(db_session, other)
    event = EventLog(
        project_id=test_project.id,
        event_type="task.status_changed",
        payload={"task_id": str(task.id)},
    )
    db_session.add(event)
    await db_session.commit()
    sessions = async_sessionmaker(db_session.bind, expire_on_commit=False)
    monkeypatch.setattr(orchestration_tasks, "AsyncSessionLocal", sessions)

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(orchestration_tasks.asyncio, "sleep", no_sleep)
    assert await orchestration_tasks.evaluate_supervision_event_async(event.id) == 0
    async with sessions() as observer:
        persisted = await observer.get(OrchestrationRun, run.id)
        assert persisted.supervision_state in (None, {})


@pytest.mark.asyncio
async def test_event_worker_commits_dirty_state_before_fresh_run_evaluation(
    db_session, test_project, monkeypatch,
):
    """A targeted evaluator in another session must see the event's committed due state."""
    from huddleroom.workers import orchestration_tasks

    run, _task = await _run_with_task(db_session, test_project)
    event = EventLog(
        project_id=test_project.id, event_type="orchestration.run_completed", payload={"run_id": str(run.id)},
    )
    db_session.add(event)
    await db_session.commit()
    sessions = async_sessionmaker(db_session.bind, expire_on_commit=False)
    observed = []

    class FakeScheduler:
        async def record_event(self, worker, *_args, **_kwargs):
            row = await worker.get(OrchestrationRun, run.id)
            row.supervision_state = {"judgment_dirty": True}
            await worker.flush()
            return [run.id]

        async def evaluate_run(self, _worker, _run_id, **_kwargs):
            async with sessions() as observer:
                observed.append(dict((await observer.get(OrchestrationRun, run.id)).supervision_state or {}))
            return 0

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(orchestration_tasks, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(orchestration_tasks, "OrchestrationSupervisionScheduler", FakeScheduler)
    monkeypatch.setattr(orchestration_tasks.asyncio, "sleep", no_sleep)

    assert await orchestration_tasks.evaluate_supervision_event_async(event.id) == 0
    assert observed == [{"judgment_dirty": True}]


@pytest.mark.asyncio
async def test_failed_provider_judgment_restores_a_dirty_due_run(  # pylint: disable=too-many-locals
    db_session, test_project, monkeypatch,
):
    """A transient provider failure must leave the run eligible for the next pass."""
    from huddleroom import database as database_module
    import huddleroom.services.orchestration_service as service_module

    goal = OrchestrationGoal(project_id=test_project.id, objective="Supervise durable work", status="active")
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized")
    db_session.add(run)
    await db_session.flush()
    now = datetime(2026, 9, 10, 15, 0, tzinfo=timezone.utc)
    run.supervision_state = {
        "needs_judgment": True,
        "judgment_dirty": True,
        "judgment_due_at": now.isoformat(),
    }
    await db_session.commit()
    sessions = async_sessionmaker(db_session.bind, expire_on_commit=False)
    monkeypatch.setattr(database_module, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(service_module, "AsyncSessionLocal", sessions)

    async def fail_provider(_payload):
        raise RuntimeError("provider unavailable")

    scheduler = OrchestrationSupervisionScheduler(judge=fail_provider)
    async with sessions() as worker:
        assert await scheduler.evaluate_run(worker, run.id, now=now) == 0
    async with sessions() as observer:
        persisted = await observer.get(OrchestrationRun, run.id)
        state = persisted.supervision_state
    assert state["judgment_in_flight"] is False
    assert state["judgment_dirty"] is True
    assert state["judgment_due_at"]


async def test_evaluate_run_makes_two_fresh_provider_attempts_then_persists_failure(
    db_session, test_project, monkeypatch,
):
    """A transient supervision provider error gets one retry, never an in-memory limbo state."""
    from huddleroom import database as database_module
    import huddleroom.services.orchestration_service as service_module

    goal = OrchestrationGoal(project_id=test_project.id, objective="Supervise durable work", status="active")
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized")
    db_session.add(run)
    await db_session.flush()
    now = datetime(2026, 9, 10, 15, 1, tzinfo=timezone.utc)
    run.supervision_state = {
        "needs_judgment": True,
        "judgment_dirty": True,
        "judgment_due_at": now.isoformat(),
    }
    await db_session.commit()
    sessions = async_sessionmaker(db_session.bind, expire_on_commit=False)
    monkeypatch.setattr(database_module, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(service_module, "AsyncSessionLocal", sessions)
    calls = 0

    async def fail_provider(_payload):
        nonlocal calls
        calls += 1
        raise RuntimeError("provider unavailable")

    async with sessions() as worker:
        assert await OrchestrationSupervisionScheduler(judge=fail_provider).evaluate_run(
            worker, run.id, now=now
        ) == 0
    async with sessions() as observer:
        state = (await observer.get(OrchestrationRun, run.id)).supervision_state
    assert calls == 2
    assert state["judgment_failures"] == 2
    assert state["judgment_dirty"] is True
    assert state["judgment_in_flight"] is False


async def test_fair_sweep_returns_a_durable_due_run_for_post_commit_evaluation(
    db_session, test_project, monkeypatch,
):
    """The provider-free sweep must hand durable due work to a later evaluator."""
    from huddleroom import database as database_module
    import huddleroom.services.orchestration_service as service_module

    goal = OrchestrationGoal(project_id=test_project.id, objective="Sweep due work", status="active")
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized")
    db_session.add(run)
    await db_session.commit()
    sessions = async_sessionmaker(db_session.bind, expire_on_commit=False)
    monkeypatch.setattr(database_module, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(service_module, "AsyncSessionLocal", sessions)
    now = datetime(2026, 9, 10, 15, 2, tzinfo=timezone.utc)

    async def local_tick(worker, run_id, *, local_only=False):
        assert local_only is True
        row = await worker.get(OrchestrationRun, run_id)
        row.supervision_state = {
            "needs_judgment": True, "judgment_dirty": True, "judgment_due_at": now.isoformat(),
        }

    async with sessions() as worker:
        due = await OrchestrationSupervisionScheduler(tick=local_tick).sweep(
            worker, timeout_seconds=2, goal_limit=1, now=now, collect_due=True,
        )
        await worker.commit()
    assert due == [run.id]


@pytest.mark.asyncio
async def test_sweep_pass_slot_uses_the_configured_reconcile_interval(test_engine, monkeypatch):
    """Changing the reconcile cadence must change the durable pass-CAS slot."""
    from huddleroom import database as database_module
    from huddleroom.services import orchestration_supervision_scheduler as scheduler_module

    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    monkeypatch.setattr(database_module, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(
        scheduler_module,
        "settings",
        SimpleNamespace(orchestration_reconcile_interval_seconds=17),
    )
    scheduler = OrchestrationSupervisionScheduler()
    first = datetime(2026, 9, 10, 15, 0, tzinfo=timezone.utc)
    second = datetime(2026, 9, 10, 15, 0, 20, tzinfo=timezone.utc)

    assert (await scheduler._claim_sweep_pass(first))[0] is True  # pylint: disable=protected-access
    assert (await scheduler._claim_sweep_pass(second))[0] is True  # pylint: disable=protected-access


async def test_unclaimed_due_sweep_returns_no_targeted_work(test_engine, monkeypatch):
    """A duplicate beat slot must be a harmless empty targeted-evaluation batch."""
    from huddleroom import database as database_module

    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    monkeypatch.setattr(database_module, "AsyncSessionLocal", sessions)
    scheduler = OrchestrationSupervisionScheduler()
    now = datetime(2026, 9, 10, 15, 3, tzinfo=timezone.utc)

    async with sessions() as worker:
        assert await scheduler.sweep(worker, now=now, collect_due=True) == []
        assert await scheduler.sweep(worker, now=now, collect_due=True) == []


@pytest.mark.unsupported_mode
def test_celery_beat_uses_the_configured_reconcile_interval(monkeypatch):
    """The PostgreSQL backstop must use the same cadence as the durable sweep."""
    from huddleroom import config
    from huddleroom.workers import celery_app

    class FakeConf(dict):
        def update(self, **kwargs):
            super().update(kwargs)

    class FakeCelery:
        def __init__(self, *_args):
            self.conf = FakeConf()

        def autodiscover_tasks(self, _tasks):
            return None

        def task(self, **_kwargs):
            return lambda function: function

    original = config.settings
    fake_celery = SimpleNamespace(Celery=FakeCelery)
    monkeypatch.setitem(sys.modules, "celery", fake_celery)
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(drop_params=False))
    monkeypatch.setattr(
        config,
        "settings",
        SimpleNamespace(redis_url=None, orchestration_reconcile_interval_seconds=17),
    )
    monkeypatch.setattr(config, "validate_supported_settings", lambda _settings: None)
    try:
        reloaded = importlib.reload(celery_app)
        assert reloaded.app.conf.beat_schedule["supervise-orchestration"]["schedule"] == 17
    finally:
        monkeypatch.setattr(config, "settings", original)
        sys.modules.pop("celery", None)
        sys.modules.pop("litellm", None)
        importlib.reload(celery_app)
