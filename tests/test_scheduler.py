from __future__ import annotations

import inspect

import pytest
from apscheduler.triggers.interval import IntervalTrigger


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
