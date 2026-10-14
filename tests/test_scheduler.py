from __future__ import annotations

import inspect
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from apscheduler.triggers.interval import IntervalTrigger


def test_spr213_deferred_scheduler_has_no_competing_initial_job(monkeypatch):
    from huddleroom.workers import scheduler as scheduler_module

    names = ("reconcile_orchestration_runs", "supervise_orchestration", "recover_orchestration")
    interval = object()

    class FakeScheduler:
        def __init__(self):
            self.jobs = {name: SimpleNamespace(trigger=interval, paused=False) for name in names}
            self.jobs["process_graph_timeouts"] = SimpleNamespace(trigger=object(), paused=False)
            self.added = []
            self.started = False
            self.fired = []

        def pause_job(self, name):
            assert not self.started
            self.jobs[name].paused = True

        def start(self):
            self.started = True

        def add_job(self, _func, **kwargs):
            self.added.append(kwargs["id"])

        def advance_interval(self):
            self.fired.extend(name for name, job in self.jobs.items() if not job.paused)

    deferred = FakeScheduler()
    monkeypatch.setattr(scheduler_module, "create_scheduler", lambda: deferred)
    monkeypatch.setattr(scheduler_module, "_utcnow", lambda: datetime(2026, 10, 14, tzinfo=timezone.utc))
    assert scheduler_module.start_scheduler(defer_orchestration_startup=True) is deferred
    assert all(deferred.jobs[name].paused for name in names)
    assert not deferred.jobs["process_graph_timeouts"].paused
    assert "recover_orchestration_immediate" not in deferred.added
    deferred.advance_interval()
    assert deferred.fired == ["process_graph_timeouts"]

    ordinary = FakeScheduler()
    monkeypatch.setattr(scheduler_module, "create_scheduler", lambda: ordinary)
    assert scheduler_module.start_scheduler() is ordinary
    assert all(not ordinary.jobs[name].paused for name in names)
    assert ordinary.added == ["recover_orchestration_immediate"]


def test_spr213_reconciliation_interval_keeps_single_coalesced_instance():
    from huddleroom.workers import scheduler as scheduler_module

    sched = scheduler_module.create_scheduler()
    job = sched.get_job("reconcile_orchestration_runs")
    assert isinstance(job.trigger, IntervalTrigger)
    assert job.max_instances == 1
    assert job.coalesce is True


@pytest.mark.asyncio
async def test_scheduler_wires_recurring_graph_timeout_processing_job(monkeypatch):
    from huddleroom import database as database_module
    from huddleroom.services import graph_engine as graph_engine_module
    from huddleroom.workers import scheduler as scheduler_module

    class FakeDb:
        def __init__(self) -> None:
            self.commits = 0

        def begin(self) -> "FakeSessionContext":
            return FakeSessionContext(self)

        async def commit(self) -> None:
            self.commits += 1

    class FakeSessionContext:
        def __init__(self, db: FakeDb) -> None:
            self.db = db

        async def __aenter__(self) -> FakeDb:
            return self.db

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

    class FakeEngine:
        def __init__(self) -> None:
            self.calls: list[FakeDb] = []

        async def process_timeouts(self, db: FakeDb) -> None:
            self.calls.append(db)

    fake_db = FakeDb()
    fake_engine = FakeEngine()
    monkeypatch.setattr(database_module, "AsyncSessionLocal", lambda: FakeSessionContext(fake_db))
    monkeypatch.setattr(graph_engine_module, "GraphEngineService", lambda: fake_engine)

    sched = scheduler_module.create_scheduler()
    sched.start(paused=True)
    try:
        job = sched.get_job("process_graph_timeouts")
        assert job is not None
        assert isinstance(job.trigger, IntervalTrigger)
        assert job.trigger.interval.total_seconds() > 0

        result = job.func(*job.args, **job.kwargs)
        if inspect.isawaitable(result):
            await result
    finally:
        if sched.running:
            sched.shutdown(wait=False)
        scheduler_module.scheduler = None

    assert fake_engine.calls == [fake_db]


@pytest.mark.asyncio
async def test_scheduler_wires_recurring_session_recovery_job(monkeypatch):
    from huddleroom import database as database_module
    from huddleroom.services import session_service as session_service_module
    from huddleroom.workers import scheduler as scheduler_module

    class FakeDb:
        def __init__(self) -> None:
            self.commits = 0

        async def commit(self) -> None:
            self.commits += 1

    class FakeSessionContext:
        def __init__(self, db: FakeDb) -> None:
            self.db = db

        async def __aenter__(self) -> FakeDb:
            return self.db

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

    class FakeSessionService:
        def __init__(self) -> None:
            self.calls: list[tuple[FakeDb, int, bool]] = []

        async def recover_orphaned_sessions(
            self,
            db: FakeDb,
            timeout_seconds: int = 0,
            redispatch_pending: bool = True,
        ) -> int:
            self.calls.append((db, timeout_seconds, redispatch_pending))
            return 2

    fake_db = FakeDb()
    fake_service = FakeSessionService()
    monkeypatch.setattr(database_module, "AsyncSessionLocal", lambda: FakeSessionContext(fake_db))
    monkeypatch.setattr(session_service_module, "SessionService", lambda: fake_service)

    sched = scheduler_module.create_scheduler()
    sched.start(paused=True)
    try:
        job = sched.get_job("recover_orphaned_sessions")
        assert job is not None
        assert isinstance(job.trigger, IntervalTrigger)
        assert job.trigger.interval.total_seconds() > 0

        result = job.func(*job.args, **job.kwargs)
        if inspect.isawaitable(result):
            await result
    finally:
        if sched.running:
            sched.shutdown(wait=False)
        scheduler_module.scheduler = None

    assert fake_service.calls == [(fake_db, 3600, False)]
    assert fake_db.commits == 1
