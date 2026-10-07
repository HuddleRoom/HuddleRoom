from __future__ import annotations

import asyncio
import uuid

import pytest

from huddleroom.services.event_bus import BusEvent


class _StubBus:
    def __init__(self, events):
        self._events = events

    async def subscribe(self):
        for event in self._events:
            yield event


@pytest.mark.asyncio
async def test_run_graph_engine_processes_events(monkeypatch, test_project):
    from huddleroom.workers.consumers import graph_engine as consumer

    event = BusEvent(
        id=uuid.uuid4(),
        project_id=test_project.id,
        event_type="code.pr_opened",
        payload={},
        source="system",
    )
    processed = []

    class StubEngine:
        def __init__(self, _bus=None):
            self._bus = _bus

        async def process_event(self, db, bus_event):
            processed.append((db, bus_event))

    monkeypatch.setattr(consumer, "get_event_bus", lambda: _StubBus([event]))
    monkeypatch.setattr(consumer, "GraphEngineService", StubEngine)

    await consumer.run_graph_engine()

    assert len(processed) == 1
    assert processed[0][1] == event


@pytest.mark.asyncio
async def test_run_graph_engine_reraises_cancelled_error(monkeypatch, test_project):
    from huddleroom.workers.consumers import graph_engine as consumer

    event = BusEvent(
        id=uuid.uuid4(),
        project_id=test_project.id,
        event_type="code.pr_opened",
        payload={},
        source="system",
    )

    class StubEngine:
        def __init__(self, _bus=None):
            self._bus = _bus

        async def process_event(self, _db, _event):
            raise asyncio.CancelledError()

    monkeypatch.setattr(consumer, "get_event_bus", lambda: _StubBus([event]))
    monkeypatch.setattr(consumer, "GraphEngineService", StubEngine)

    with pytest.raises(asyncio.CancelledError):
        await consumer.run_graph_engine()


@pytest.mark.asyncio
async def test_run_graph_engine_logs_and_continues_after_error(monkeypatch, caplog, test_project):
    from huddleroom.workers.consumers import graph_engine as consumer

    events = [
        BusEvent(
            id=uuid.uuid4(),
            project_id=test_project.id,
            event_type="code.pr_opened",
            payload={},
            source="system",
        ),
        BusEvent(
            id=uuid.uuid4(),
            project_id=test_project.id,
            event_type="test.passed",
            payload={},
            source="system",
        ),
    ]
    processed = []

    class StubEngine:
        def __init__(self, _bus=None):
            self._bus = _bus

        async def process_event(self, _db, event):
            processed.append(event.event_type)
            if event.event_type == "code.pr_opened":
                raise RuntimeError("boom")

    monkeypatch.setattr(consumer, "get_event_bus", lambda: _StubBus(events))
    monkeypatch.setattr(consumer, "GraphEngineService", StubEngine)

    await consumer.run_graph_engine()

    assert processed == ["code.pr_opened", "test.passed"]
    assert "graph_engine error processing code.pr_opened" in caplog.text
