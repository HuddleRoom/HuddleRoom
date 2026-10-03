import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timezone
import uuid
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession
from huddleroom.models.event_log import EventLog
from huddleroom.schemas.event import EventEmit, EventResponse
from huddleroom.services.event_bus import EventBusService, BusEvent, get_event_bus, emit_event



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

    # Fixture rolls back instead of committing, so after_commit never fires.
    # Simulate post-commit dispatch manually.
    sync = db_session.sync_session
    for ev, b in sync.info.pop("pending_bus_events", []):
        b.put(ev)

    await asyncio.wait_for(task, timeout=1.0)
    assert len(received) == 1
    assert received[0].event_type == "task.created"
