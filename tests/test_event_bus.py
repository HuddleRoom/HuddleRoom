import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timezone
import uuid
from sqlalchemy import text, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from huddleroom.models.event_log import EventLog
from huddleroom.schemas.event import EventEmit, EventResponse
from huddleroom.services.event_bus import EventBusService, BusEvent, get_event_bus, emit_event, emit_event_once
from tests.test_orchestration_roster_mutation import sessions  # noqa: F401 -- fixture


@pytest.mark.asyncio
async def test_spr213_subscription_ready_means_registered():
    bus = EventBusService()
    ready = asyncio.Event()
    project_id = uuid.uuid4()
    received = []

    async def consume():
        subscription = bus.subscribe(project_id, ["task.created"], 4, ready=ready)
        try:
            received.append(await anext(subscription))
        finally:
            await subscription.aclose()

    task = asyncio.create_task(consume())
    ready_task = asyncio.create_task(ready.wait())
    try:
        done, _ = await asyncio.wait({ready_task, task}, timeout=1, return_when=asyncio.FIRST_COMPLETED)
        assert done
        if task in done:
            await task
    finally:
        ready_task.cancel()
    assert len(bus._subscribers) == 1
    bus.put(BusEvent(uuid.uuid4(), uuid.uuid4(), "task.created", {}, "test"))
    bus.put(BusEvent(uuid.uuid4(), project_id, "task.updated", {}, "test"))
    assert not received
    event = BusEvent(uuid.uuid4(), project_id, "task.created", {}, "test")
    bus.put(event)
    await asyncio.wait_for(task, 1)
    assert received == [event]
    assert bus._subscribers == []

    cancel_ready = asyncio.Event()

    async def wait_for_event():
        async for _ in bus.subscribe(project_id, ["task.created"], 4, ready=cancel_ready):
            pass

    cancelled = asyncio.create_task(wait_for_event())
    await asyncio.wait_for(cancel_ready.wait(), 1)
    assert len(bus._subscribers) == 1
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert bus._subscribers == []



@pytest.mark.asyncio
async def test_event_bus_subscribe_receives_event():
    bus = EventBusService()
    project_id = uuid.uuid4()

    received = []

    async def consume():
        async for event in bus.subscribe(project_id=project_id):
            received.append(event)
            break  # stop after first

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)  # yield to let consumer register

    event = BusEvent(
        id=uuid.uuid4(),
        project_id=project_id,
        event_type="task.created",
        payload={"title": "Test"},
        source="system",
        emitted_at=datetime.now(timezone.utc),
    )
    bus.put(event)
    await asyncio.wait_for(task, timeout=1.0)
    assert len(received) == 1
    assert received[0].event_type == "task.created"


@pytest.mark.asyncio
async def test_event_bus_project_filter():
    bus = EventBusService()
    project_a = uuid.uuid4()
    project_b = uuid.uuid4()

    received = []

    async def consume():
        async for event in bus.subscribe(project_id=project_a):
            received.append(event)
            break

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)

    # Event for project_b should NOT be received
    bus.put(BusEvent(
        id=uuid.uuid4(), project_id=project_b, event_type="task.created",
        payload={}, source="system", emitted_at=datetime.now(timezone.utc),
    ))
    # Event for project_a SHOULD be received
    bus.put(BusEvent(
        id=uuid.uuid4(), project_id=project_a, event_type="task.created",
        payload={}, source="system", emitted_at=datetime.now(timezone.utc),
    ))
    await asyncio.wait_for(task, timeout=1.0)
    assert len(received) == 1


@pytest.mark.asyncio
async def test_event_bus_global_subscription():
    """None project_id = subscribe to all projects."""
    bus = EventBusService()
    project_a = uuid.uuid4()
    project_b = uuid.uuid4()

    received = []

    async def consume():
        async for event in bus.subscribe(project_id=None):
            received.append(event)
            if len(received) == 2:
                break

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)

    bus.put(BusEvent(id=uuid.uuid4(), project_id=project_a, event_type="x", payload={}, source="system", emitted_at=datetime.now(timezone.utc)))
    bus.put(BusEvent(id=uuid.uuid4(), project_id=project_b, event_type="x", payload={}, source="system", emitted_at=datetime.now(timezone.utc)))
    await asyncio.wait_for(task, timeout=1.0)
    assert len(received) == 2


def test_get_event_bus_returns_singleton():
    bus1 = get_event_bus()
    bus2 = get_event_bus()
    assert bus1 is bus2


@pytest.mark.asyncio
async def test_emit_event_writes_event_log(db_session, test_project):
    await emit_event(
        db=db_session,
        project_id=test_project.id,
        event_type="task.created",
        payload={"task_id": "abc123"},
        source="system",
    )
    result = await db_session.execute(
        select(EventLog).where(EventLog.project_id == test_project.id)
    )
    rows = list(result.scalars().all())
    assert len(rows) == 1
    assert rows[0].event_type == "task.created"
    assert rows[0].payload == {"task_id": "abc123"}


@pytest.mark.asyncio
async def test_emit_event_puts_on_bus(db_session, test_project):
    bus = EventBusService()  # fresh bus, not singleton
    received = []

    async def consume():
        async for event in bus.subscribe(project_id=test_project.id):
            received.append(event)
            break

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)

    await emit_event(
        db=db_session,
        project_id=test_project.id,
        event_type="task.created",
        payload={"x": 1},
        source="system",
        _bus=bus,  # inject bus for testing
    )

    # Exercise the actual root outcome rather than draining private queue storage.
    await db_session.commit()

    await asyncio.wait_for(task, timeout=1.0)
    assert len(received) == 1
    assert received[0].event_type == "task.created"


@pytest.mark.asyncio
async def test_resolution_wakeup_waits_for_outer_transaction_outcome(sessions, tmp_path, monkeypatch):
    """SPR230/SPR231 releasing a SAVEPOINT cannot publish a rolled-back resolution."""
    from huddleroom.models.project import Project
    from tests.test_orchestration_roster_mutation import seed_project_user
    async with sessions() as setup:
        project, _ = await seed_project_user(setup, tmp_path)
        await setup.commit()
    bus, published = EventBusService(), []
    monkeypatch.setattr(bus, "put", lambda event: published.append(event.id))
    async with sessions() as caller:
        await caller.execute(update(Project).where(Project.id == project.id).values(name="outer writer"))
        nested = await caller.begin_nested()
        event, created = await emit_event_once(caller, project.id, "authority.decision_resolved",
            {"decision_id": str(uuid.uuid4())}, _bus=bus)
        assert created and published == []
        await nested.commit()
        assert published == [], "SAVEPOINT commit published before the outer outcome"
        await caller.rollback()
        await caller.commit()  # reuse must not revive rolled-back queued work
    assert published == []
    async with sessions() as fresh:
        assert await fresh.scalar(select(EventLog).where(EventLog.id == event.id)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["nested_rollback", "rollback_reuse", "close_reuse", "commit_once"])
async def test_resolution_publication_tracks_outer_transaction_scope(sessions, tmp_path, monkeypatch, outcome):
    """SPR230/SPR231 only durable scoped resolution events wake consumers, once."""
    import sqlite3
    from huddleroom.models.project import Project
    from tests.test_orchestration_roster_mutation import seed_project_user
    async with sessions() as setup:
        project, _ = await seed_project_user(setup, tmp_path)
        await setup.commit()
    bus, published, visible = EventBusService(), [], []
    def publish(event):
        published.append(event.id)
        with sqlite3.connect(sessions.kw["bind"].url.database) as reader:
            visible.append(reader.execute("SELECT count(*) FROM event_log WHERE id=?", (event.id.hex,)).fetchone()[0])
    monkeypatch.setattr(bus, "put", publish)
    caller = sessions()
    try:
        await caller.execute(update(Project).where(Project.id == project.id).values(name="outer writer"))
        outer, _ = await emit_event_once(caller, project.id, "authority.decision_resolved",
            {"decision_id": str(uuid.uuid4())}, dedup_key=f"outer:{uuid.uuid4()}", _bus=bus)
        if outcome == "nested_rollback":
            nested = await caller.begin_nested()
            dropped, _ = await emit_event_once(caller, project.id, "authority.decision_resolved",
                {"decision_id": str(uuid.uuid4())}, dedup_key=f"nested:{uuid.uuid4()}", _bus=bus)
            assert published == []
            await nested.rollback()
            await caller.commit()
            expected, missing = [outer.id], [dropped.id]
        elif outcome in {"rollback_reuse", "close_reuse"}:
            await (caller.rollback() if outcome == "rollback_reuse" else caller.close())
            await caller.commit()  # unrelated commit must not revive old event
            assert published == []
            await caller.execute(update(Project).where(Project.id == project.id).values(name="reused writer"))
            reused, _ = await emit_event_once(caller, project.id, "authority.decision_resolved",
                {"decision_id": str(uuid.uuid4())}, dedup_key=f"reused:{uuid.uuid4()}", _bus=bus)
            await caller.commit()
            expected, missing = [reused.id], [outer.id]
        else:
            assert published == []
            await caller.commit()
            await caller.commit()
            expected, missing = [outer.id], []
        assert published == expected and visible == [1] * len(expected)
        assert not caller.sync_session.info.get("pending_bus_events")
        async with sessions() as fresh:
            for event_id in expected:
                assert await fresh.scalar(select(EventLog).where(EventLog.id == event_id)) is not None
            for event_id in missing:
                assert await fresh.scalar(select(EventLog).where(EventLog.id == event_id)) is None
    finally:
        await caller.close()
