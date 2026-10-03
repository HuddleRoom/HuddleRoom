"""Task 9 acceptance tests for goal-first recovery scheduling and observation."""
from datetime import datetime, timedelta, timezone
import asyncio
import time
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun, OrchestrationWait
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.orchestration_recovery_service import GoalRecoverySnapshot, OrchestrationRecoveryService, RunnerObservation
from huddleroom.workers import orchestration_recovery_tasks as recovery_tasks
from huddleroom.workers import scheduler as scheduler_module


async def _goal_run(db, project, *, status="active", phase="baseline"):
    goal = OrchestrationGoal(project_id=project.id, objective="scheduled", original_request="scheduled", success_criteria=[], constraints={}, budget={}, status=status)
    db.add(goal); await db.flush()
    run = OrchestrationRun(goal_id=goal.id, status="paused" if status == "paused" else "running", phase=phase)
    db.add(run); await db.flush()
    return goal, run


def _snapshot(goal, run, ready):
    return GoalRecoverySnapshot(goal.id, run.id, goal.project_id, goal.status, run.status, run.phase, (), (), (), (), ready)


@pytest.mark.asyncio
async def test_recovery_scans_eligible_authorized_goals_not_sessions(db_session, test_project):
    eligible, _ = await _goal_run(db_session, test_project)
    waiting, _ = await _goal_run(db_session, test_project, phase="waiting_activation")
    service = OrchestrationRecoveryService(); now = datetime.now(timezone.utc)
    assert await service.build_goal_snapshot(db_session, eligible.id, now)
    assert await service.build_goal_snapshot(db_session, waiting.id, now) is None


@pytest.mark.asyncio
async def test_goal_with_many_sessions_persists_at_most_one_new_disposition_per_pass(
    db_session, test_project, test_agent,
):
    goal, run = await _goal_run(db_session, test_project)
    run.supervision_state = {"task8": {"cursor": "preserve"}, "recovery": {"sessions": {"sibling": {"task8_key": "keep"}}}}
    for index in range(3):
        action = OrchestrationAction(run_id=run.id, idempotency_key=f"seed:{index}", action_type="retry_task",
            request={}, status="completed", dispatch_contract={"owner": "orchestration_recovery"})
        task = Task(project_id=test_project.id, title=str(index), description=str(index), status="in_progress",
            assigned_to=test_agent.id, metadata_={"orchestration": {"run_id": str(run.id)}})
        db_session.add_all((action, task)); await db_session.flush()
        task.metadata_ = {"orchestration": {"run_id": str(run.id), "action_id": str(action.id)}}
        session = Session(project_id=test_project.id, agent_id=test_agent.id, task_id=task.id, adapter_type="cli",
            status="running", runner_task_id=f"runner-{index}", metadata_={"orchestration": {"action_id": str(action.id)}, "attempt": {}})
        db_session.add(session); await db_session.flush()
        action.request, action.target_type, action.target_id = {"task_id": str(task.id)}, "session", session.id
    snapshot = await OrchestrationRecoveryService().build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))
    result = await OrchestrationRecoveryService().apply_goal_recovery(db_session, snapshot, {})
    waits = list(await db_session.scalars(select(OrchestrationWait).where(OrchestrationWait.run_id == run.id)))
    recovery_sessions = run.supervision_state["recovery"]["sessions"]
    assert len(snapshot.sessions) == 3 and result.wait_id and len(waits) == 1
    assert run.supervision_state["task8"] == {"cursor": "preserve"}
    assert recovery_sessions["sibling"] == {"task8_key": "keep"} and len(recovery_sessions) == 4
    assert sum(entry["wait_id"] is not None for key, entry in recovery_sessions.items() if key != "sibling") == 1


@pytest.mark.asyncio
async def test_complete_snapshot_precedes_backend_observation(monkeypatch):
    goal_id, events = uuid.uuid4(), []
    class Result:
        def all(self): return [goal_id]
    class Db:
        async def scalars(self, _query): return Result()
        async def commit(self): events.append("commit")
    class Context:
        async def __aenter__(self): events.append("db"); return Db()
        async def __aexit__(self, *_args): return None
    class Service:
        async def build_goal_snapshot(self, _db, received_goal_id, ready):
            events.append("snapshot"); return SimpleNamespace(goal_id=received_goal_id, scheduler_ready_at=ready,
                sessions=(SimpleNamespace(session_id=goal_id, runner_task_id="runner"),))
        async def apply_goal_recovery(self, _db, _snapshot, observations, recovery_timing):
            events.append("apply"); assert observations[goal_id].state == "active"; return object()
    monkeypatch.setattr(recovery_tasks, "AsyncSessionLocal", lambda: Context())
    monkeypatch.setattr(recovery_tasks, "OrchestrationRecoveryService", Service)
    monkeypatch.setattr(recovery_tasks, "observe_runner", lambda *_args, **_kwargs: events.append("observe") or "active")
    assert await recovery_tasks.recover_orchestration_async(datetime.now(timezone.utc)) == 1
    assert events.index("snapshot") < events.index("observe") < events.index("apply")


@pytest.mark.asyncio
async def test_observation_scalar_fact_requires_no_database_handle(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    snapshot = _snapshot(goal, run, datetime.now(timezone.utc))
    # Observation is a scalar fact produced before the fresh apply call; it does not need a DB handle.
    observation = recovery_tasks.observe_runner(None, is_sqlite=True)
    assert observation == "unknown" and snapshot.sessions == ()


@pytest.mark.asyncio
async def test_apply_uses_fresh_transaction_and_revalidates_complete_snapshot(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project); service = OrchestrationRecoveryService()
    snapshot = await service.build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))
    run.retry_state = {"changed": True}
    assert await service.apply_goal_recovery(db_session, snapshot, {}) is None


@pytest.mark.asyncio
async def test_one_goal_failure_does_not_starve_later_goal(db_session, test_engine, test_project, monkeypatch):
    first, _ = await _goal_run(db_session, test_project); _, second_run = await _goal_run(db_session, test_project)
    await db_session.commit()
    original = OrchestrationRecoveryService.build_goal_snapshot
    async def fail_first(service, db, goal_id, ready):
        if goal_id == first.id:
            raise RuntimeError("injected first-goal failure")
        return await original(service, db, goal_id, ready)
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(recovery_tasks, "AsyncSessionLocal", factory)
    monkeypatch.setattr(OrchestrationRecoveryService, "build_goal_snapshot", fail_first)
    assert await recovery_tasks.recover_orchestration_async(datetime.now(timezone.utc)) == 1
    async with factory() as check:
        assert await check.scalar(select(OrchestrationAction).where(OrchestrationAction.run_id == second_run.id))


@pytest.mark.asyncio
async def test_each_goal_uses_fresh_snapshot_and_apply_sessions(db_session, test_engine, test_project, monkeypatch):
    await _goal_run(db_session, test_project); await _goal_run(db_session, test_project); await db_session.commit()
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    entered = []
    class TrackedSession:
        def __init__(self): self.session = factory()
        async def __aenter__(self): entered.append(id(self.session)); return await self.session.__aenter__()
        async def __aexit__(self, *args): return await self.session.__aexit__(*args)
    monkeypatch.setattr(recovery_tasks, "AsyncSessionLocal", TrackedSession)
    assert await recovery_tasks.recover_orchestration_async(datetime.now(timezone.utc)) == 2
    assert len(entered) == len(set(entered)) == 5


@pytest.mark.asyncio
async def test_sqlite_runs_immediately_after_scheduler_readiness_and_every_r(monkeypatch):
    called = []
    invoked_at = datetime(2026, 1, 2, tzinfo=timezone.utc)

    async def recover(ready):
        called.append(ready)

    monkeypatch.setattr(recovery_tasks, "recover_orchestration_async", recover)
    await scheduler_module.recover_orchestration_job(invoked_at)
    assert called == [invoked_at]
    assert (scheduler_module.create_scheduler().get_job("recover_orchestration").trigger.interval.total_seconds()
            == scheduler_module.settings.orchestration_reconcile_interval_seconds)


def test_scheduler_start_captures_readiness_once_after_start(monkeypatch):
    events = []

    class SchedulerSpy:
        def start(self):
            events.append("start")

        def add_job(self, func, **kwargs):
            events.append((func, kwargs))

    spy = SchedulerSpy()
    monkeypatch.setattr(scheduler_module, "create_scheduler", lambda: spy)
    monkeypatch.setattr(scheduler_module, "_utcnow", lambda: datetime(2026, 1, 2, tzinfo=timezone.utc))
    scheduler_module.start_scheduler()
    immediate = events[1]
    assert events[0] == "start"
    assert immediate[1]["args"] == [datetime(2026, 1, 2, tzinfo=timezone.utc)]


def test_apscheduler_recovery_runs_at_the_configured_interval():
    scheduler = scheduler_module.create_scheduler()
    assert scheduler.get_job("recover_orchestration").trigger.interval.total_seconds() == scheduler_module.settings.orchestration_reconcile_interval_seconds


@pytest.mark.unsupported_mode
def test_celery_worker_ready_runs_immediately_and_beat_runs_every_r(monkeypatch):
    from huddleroom.workers import celery_app
    app = celery_app.app
    assert app is not None
    beat = app.conf.beat_schedule["recover-orchestration"]
    sent = []
    monkeypatch.setattr(app, "send_task", lambda *args, **kwargs: sent.append((args, kwargs)))
    now = datetime(2026, 1, 2, tzinfo=timezone.utc)
    monkeypatch.setattr(celery_app, "_utcnow", lambda: now)
    celery_app.worker_ready.send(sender=app)
    assert sent == [(("rally.workers.orchestration_recovery_tasks.recover_orchestration",), {"args": [now.isoformat()]})]
    headers = {}
    celery_app.before_task_publish.send(sender=beat["task"], headers=headers)
    assert beat["task"].endswith("recover_orchestration")
    assert beat["schedule"] == celery_app.settings.orchestration_reconcile_interval_seconds
    assert headers["orchestration_recovery_ready_at"] == now.isoformat()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_fact", ["active", "unknown"])
async def test_sqlite_facts_produce_expected_dispositions(
        db_session, test_project, test_agent, backend_fact, monkeypatch):
    goal, run = await _goal_run(db_session, test_project)
    action = OrchestrationAction(run_id=run.id, idempotency_key=f"seed:{uuid.uuid4()}", action_type="retry_task",
                                 request={}, status="completed")
    db_session.add(action); await db_session.flush()
    task = Task(project_id=test_project.id, title="owned", description="owned", status="in_progress",
                assigned_to=test_agent.id, metadata_={"orchestration": {"run_id": str(run.id), "action_id": str(action.id)}})
    db_session.add(task); await db_session.flush()
    session = Session(project_id=test_project.id, agent_id=test_agent.id, task_id=task.id, adapter_type="cli",
                      status="running", runner_task_id="sqlite-runner",
                      metadata_={"orchestration": {"action_id": str(action.id)}, "attempt": {}})
    db_session.add(session); await db_session.flush()
    action.request = {"task_id": str(task.id)}
    action.target_type, action.target_id = "session", session.id
    action.dispatch_contract = {"owner": "orchestration_recovery"}
    snapshot = await OrchestrationRecoveryService().build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))
    from huddleroom.workers import task_runner
    monkeypatch.setattr(task_runner, "is_task_live", lambda _runner: backend_fact == "active")
    observed_at = datetime.now(timezone.utc)
    fact = recovery_tasks.observe_runner("sqlite-runner", is_sqlite=True)
    result = await OrchestrationRecoveryService().apply_goal_recovery(
        db_session, snapshot, {session.id: RunnerObservation("sqlite-runner", fact, observed_at)})
    assert fact == backend_fact
    assert result.classification == ("live" if backend_fact == "active" else "unknown_external_effect")
    assert bool(result.action_id) ^ bool(result.wait_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_fact", ["active", "unknown"])
@pytest.mark.unsupported_mode
async def test_sqlite_and_celery_facts_produce_equivalent_dispositions(
        db_session, test_project, test_agent, backend_fact, monkeypatch):
    async def owned(runner_task_id):
        goal, run = await _goal_run(db_session, test_project)
        action = OrchestrationAction(run_id=run.id, idempotency_key=f"seed:{uuid.uuid4()}", action_type="retry_task",
                                     request={}, status="completed")
        db_session.add(action); await db_session.flush()
        task = Task(project_id=test_project.id, title="owned", description="owned", status="in_progress",
                    assigned_to=test_agent.id, metadata_={"orchestration": {"run_id": str(run.id), "action_id": str(action.id)}})
        db_session.add(task); await db_session.flush()
        action.target_type, action.target_id = "task", task.id
        session = Session(project_id=test_project.id, agent_id=test_agent.id, task_id=task.id, adapter_type="cli",
                          status="running", runner_task_id=runner_task_id,
                          metadata_={"orchestration": {"action_id": str(action.id)}, "attempt": {}})
        db_session.add(session); await db_session.flush()
        action.request = {"task_id": str(task.id)}
        action.target_type, action.target_id = "session", session.id
        action.dispatch_contract = {"owner": "orchestration_recovery"}
        snapshot = await OrchestrationRecoveryService().build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))
        return snapshot, session

    sqlite_snapshot, sqlite_session = await owned("sqlite-runner")
    celery_snapshot, celery_session = await owned("celery-runner")
    from huddleroom.workers import task_runner
    from huddleroom.workers import celery_app

    monkeypatch.setattr(task_runner, "is_task_live", lambda _runner: backend_fact == "active")

    class Inspector:
        def active(self):
            return {"worker": [{"id": "celery-runner"}]} if backend_fact == "active" else {}
        def reserved(self): return {}
        def scheduled(self): return {}
    class CelerySpy:
        control = SimpleNamespace(inspect=lambda: Inspector())
        @staticmethod
        def AsyncResult(_runner): return SimpleNamespace(state="PENDING")
    monkeypatch.setattr(celery_app, "app", CelerySpy())

    service = OrchestrationRecoveryService(); observed_at = datetime.now(timezone.utc)
    sqlite_fact = recovery_tasks.observe_runner("sqlite-runner", is_sqlite=True)
    celery_fact = recovery_tasks.observe_runner("celery-runner", is_sqlite=False)
    assert sqlite_snapshot.sessions[0].session_id == sqlite_session.id
    assert celery_snapshot.sessions[0].session_id == celery_session.id
    assert sqlite_fact == celery_fact == backend_fact
    sqlite = await service.apply_goal_recovery(db_session, sqlite_snapshot, {
        sqlite_session.id: RunnerObservation("sqlite-runner", sqlite_fact, observed_at)})
    celery = await service.apply_goal_recovery(db_session, celery_snapshot, {
        celery_session.id: RunnerObservation("celery-runner", celery_fact, observed_at)})
    assert sqlite.classification == celery.classification
    assert sqlite.disposition == celery.disposition
    assert sqlite.classification == ("live" if backend_fact == "active" else "unknown_external_effect")
    assert bool(sqlite.action_id) ^ bool(sqlite.wait_id)
    assert bool(celery.action_id) ^ bool(celery.wait_id)


@pytest.mark.unsupported_mode
def test_celery_recovery_rejects_missing_timestamp():
    task = recovery_tasks.register_celery_task(type("App", (), {"task": lambda _self, **_kwargs: lambda fn: fn})())
    with pytest.raises(ValueError, match="timestamp"):
        task(type("Request", (), {"request": type("Context", (), {"headers": {}})()})())


@pytest.mark.unsupported_mode
def test_celery_beat_task_consumes_its_publish_timestamp(monkeypatch):
    captured = []
    now = datetime(2026, 1, 2, tzinfo=timezone.utc)

    async def recover(ready):
        captured.append(ready)

    monkeypatch.setattr(recovery_tasks, "recover_orchestration_async", recover)
    from huddleroom.workers.celery_app import app as celery_app
    task = celery_app.tasks["rally.workers.orchestration_recovery_tasks.recover_orchestration"]
    task.request_stack.push(type("Request", (), {"headers": {"orchestration_recovery_ready_at": now.isoformat()}})())
    try:
        task.run()
    finally:
        task.request_stack.pop()
    assert captured == [now]


@pytest.mark.asyncio
async def test_recovery_persists_measured_observation_timing(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    ready = datetime(2026, 1, 2, tzinfo=timezone.utc)
    snapshot = await OrchestrationRecoveryService().build_goal_snapshot(db_session, goal.id, ready)
    timing = {
        "scheduler_ready_at": ready.isoformat(),
        "observation_started_at": ready.isoformat(),
        "observation_ended_at": (ready + timedelta(seconds=3)).isoformat(),
        "observation_duration_seconds": 3.0,
        "assessment_at": (ready + timedelta(seconds=5)).isoformat(),
        "excluded_duration_seconds": 3.0,
        "within_2r": True,
    }
    await OrchestrationRecoveryService().apply_goal_recovery(db_session, snapshot, {}, recovery_timing=timing)
    assert run.supervision_state["recovery"]["last_pass"] == timing
    assert run.supervision_state["recovery"]["observation_started_at"] == timing["observation_started_at"]


@pytest.mark.asyncio
async def test_observation_occurs_without_database_transaction_or_goal_lock(
    db_session, test_engine, test_project, test_agent, monkeypatch,
):
    goal, run = await _goal_run(db_session, test_project)
    action = OrchestrationAction(run_id=run.id, idempotency_key="blocking-probe", action_type="retry_task",
        request={}, status="completed", dispatch_contract={"owner": "orchestration_recovery"})
    task = Task(project_id=test_project.id, title="probe", description="probe", status="in_progress",
        assigned_to=test_agent.id, metadata_={"orchestration": {"run_id": str(run.id)}})
    db_session.add_all((action, task)); await db_session.flush()
    task.metadata_ = {"orchestration": {"run_id": str(run.id), "action_id": str(action.id)}}
    session = Session(project_id=test_project.id, agent_id=test_agent.id, task_id=task.id, adapter_type="cli",
        status="running", runner_task_id="blocking", metadata_={"orchestration": {"action_id": str(action.id)}, "attempt": {}})
    db_session.add(session); await db_session.flush()
    action.request, action.target_type, action.target_id = {"task_id": str(task.id)}, "session", session.id
    await db_session.commit()
    from huddleroom.workers import task_runner
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    active = 0
    class TrackedSession:
        def __init__(self): self.session = factory()
        async def __aenter__(self):
            nonlocal active
            active += 1
            return await self.session.__aenter__()
        async def __aexit__(self, *args):
            nonlocal active
            active -= 1
            return await self.session.__aexit__(*args)
    monkeypatch.setattr(recovery_tasks, "AsyncSessionLocal", TrackedSession)
    from huddleroom.config import settings
    monkeypatch.setattr(settings, "orchestration_reconcile_interval_seconds", 0.25)
    def blocking_probe(_runner):
        time.sleep(1.5)
        return True
    monkeypatch.setattr(task_runner, "is_task_live", blocking_probe)
    original_observe = recovery_tasks.observe_runner
    def observe_without_db(*args, **kwargs):
        assert active == 0
        return original_observe(*args, **kwargs)
    monkeypatch.setattr(recovery_tasks, "observe_runner", observe_without_db)
    assert await recovery_tasks.recover_orchestration_async(datetime.now(timezone.utc)) == 1
    async with async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)() as check:
        stored = await check.get(OrchestrationRun, run.id)
        timing = stored.supervision_state["recovery"]["last_pass"]
        assert timing["observation_duration_seconds"] >= 1.4 and timing["excluded_duration_seconds"] >= 1.4
        assert timing["within_2r"] is True


@pytest.mark.asyncio
async def test_two_r_bound_excludes_only_measured_backend_observation(monkeypatch):
    ready = datetime(2026, 1, 2, tzinfo=timezone.utc)
    first, second = uuid.uuid4(), uuid.uuid4()
    snapshots = [
        SimpleNamespace(goal_id=first, sessions=(SimpleNamespace(session_id=first, runner_task_id="a"),)),
        SimpleNamespace(goal_id=second, sessions=(SimpleNamespace(session_id=second, runner_task_id="b"),)),
    ]
    captured, sessions = [], []

    class Result:
        def all(self): return [first, second]
    class Db:
        async def scalars(self, _query): return Result()
        async def commit(self): pass
    class Context:
        async def __aenter__(self):
            db = Db(); sessions.append(db); return db
        async def __aexit__(self, *_args): pass
    class Service:
        async def build_goal_snapshot(self, _db, _goal_id, _ready): return snapshots.pop(0)
        async def apply_goal_recovery(self, _db, snapshot, observations, recovery_timing):
            captured.append((snapshot, observations, recovery_timing)); return object()
    clock = iter([ready + timedelta(seconds=n) for n in (60, 61, 65, 66, 70, 71, 72, 77, 78, 131)])
    monkeypatch.setattr(recovery_tasks, "AsyncSessionLocal", lambda: Context())
    monkeypatch.setattr(recovery_tasks, "OrchestrationRecoveryService", Service)
    monkeypatch.setattr(recovery_tasks, "_utcnow", lambda: next(clock))
    monkeypatch.setattr(recovery_tasks, "observe_runner", lambda *_args, **_kwargs: "unknown")
    assert await recovery_tasks.recover_orchestration_async(ready) == 2
    assert len({id(session) for session in sessions}) == 5
    assert captured[0][2]["excluded_duration_seconds"] == 4.0
    assert captured[-1][2]["observation_duration_seconds"] == 5.0
    assert captured[-1][2]["excluded_duration_seconds"] == 9.0
    assert captured[-1][2]["within_2r"] is False


@pytest.mark.asyncio
async def test_old_ready_timestamp_fails_two_r_bound(monkeypatch):
    ready = datetime(2026, 1, 2, tzinfo=timezone.utc)
    goal_id = uuid.uuid4()
    captured = []

    class Result:
        def all(self): return [goal_id]
    class Db:
        async def scalars(self, _query): return Result()
        async def commit(self): pass
    class Context:
        async def __aenter__(self): return Db()
        async def __aexit__(self, *_args): pass
    class Service:
        async def build_goal_snapshot(self, _db, _goal_id, _ready):
            return SimpleNamespace(goal_id=goal_id, sessions=())
        async def apply_goal_recovery(self, _db, _snapshot, _observations, recovery_timing):
            captured.append(recovery_timing); return object()

    from huddleroom.config import settings
    interval = settings.orchestration_reconcile_interval_seconds
    monkeypatch.setattr(recovery_tasks, "AsyncSessionLocal", lambda: Context())
    monkeypatch.setattr(recovery_tasks, "OrchestrationRecoveryService", Service)
    monkeypatch.setattr(recovery_tasks, "_utcnow", lambda: ready + timedelta(seconds=2 * interval + 1))
    assert await recovery_tasks.recover_orchestration_async(ready) == 1
    assert captured[0]["scheduler_ready_at"] == ready.isoformat()
    assert captured[0]["within_2r"] is False


@pytest.mark.asyncio
async def test_real_recovery_loop_isolates_first_goal_failure(monkeypatch):
    goal_ids = [uuid.uuid4(), uuid.uuid4()]
    applied = []

    class Result:
        def all(self): return goal_ids
    class Db:
        async def scalars(self, _query): return Result()
        async def commit(self): pass
    class Context:
        async def __aenter__(self): return Db()
        async def __aexit__(self, *_args): pass
    class Service:
        calls = 0
        async def build_goal_snapshot(self, _db, goal_id, _ready):
            self.calls += 1
            if self.calls == 1: raise RuntimeError("first goal")
            return SimpleNamespace(goal_id=goal_id, sessions=())
        async def apply_goal_recovery(self, _db, snapshot, _observations, recovery_timing):
            applied.append((snapshot.goal_id, recovery_timing)); return object()

    monkeypatch.setattr(recovery_tasks, "AsyncSessionLocal", lambda: Context())
    monkeypatch.setattr(recovery_tasks, "OrchestrationRecoveryService", Service)
    monkeypatch.setattr(recovery_tasks, "_utcnow", lambda: datetime(2026, 1, 2, tzinfo=timezone.utc))
    assert await recovery_tasks.recover_orchestration_async(datetime(2026, 1, 2, tzinfo=timezone.utc)) == 1
    assert [goal_id for goal_id, _timing in applied] == [goal_ids[1]]


@pytest.mark.asyncio
async def test_recovery_never_enters_supervision_local_only_judgment_path(db_session, test_project, monkeypatch):
    goal, run = await _goal_run(db_session, test_project)
    calls = []
    from huddleroom.services.orchestration_service import OrchestrationService

    async def tick(*_args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(OrchestrationService, "tick", tick)
    snapshot = await OrchestrationRecoveryService().build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))
    result = await OrchestrationRecoveryService().apply_goal_recovery(db_session, snapshot, {})
    assert result.classification == "memory_only" and result.disposition != "local_only" and calls == []
