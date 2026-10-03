import asyncio
import json
import uuid
from contextlib import closing
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from huddleroom.workers.consumers.ws_hub import ConnectionRegistry


@pytest.mark.asyncio
async def test_connection_registry_broadcast():
    """Registry sends event to registered mock WebSocket."""
    registry = ConnectionRegistry()
    project_id = uuid.uuid4()
    conn_id = str(uuid.uuid4())

    sent = []

    class _MockWS:
        async def send_text(self, text):
            sent.append(text)

    registry.add(project_id, conn_id, _MockWS(), event_type_filter=None)

    from huddleroom.services.event_bus import BusEvent
    from datetime import datetime, timezone
    event = BusEvent(
        id=uuid.uuid4(), project_id=project_id, event_type="task.created",
        payload={"x": 1}, source="system", emitted_at=datetime.now(timezone.utc),
    )
    await registry.broadcast(event)
    assert len(sent) == 1
    data = json.loads(sent[0])
    assert data["event_type"] == "task.created"


@pytest.mark.asyncio
async def test_connection_registry_event_type_filter():
    """Registry skips events not matching the connection's type filter."""
    registry = ConnectionRegistry()
    project_id = uuid.uuid4()
    conn_id = str(uuid.uuid4())
    sent = []

    class _MockWS:
        async def send_text(self, text):
            sent.append(text)

    registry.add(project_id, conn_id, _MockWS(), event_type_filter={"task.created"})

    from huddleroom.services.event_bus import BusEvent
    from datetime import datetime, timezone
    event = BusEvent(
        id=uuid.uuid4(), project_id=project_id, event_type="session.started",
        payload={}, source="system", emitted_at=datetime.now(timezone.utc),
    )
    await registry.broadcast(event)
    assert len(sent) == 0


def _test_session_factory(test_engine):
    engine = create_async_engine(test_engine.url, poolclass=NullPool)
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest.mark.asyncio
async def test_meeting_websocket_replays_transcript_turns(test_engine, monkeypatch):
    from starlette.testclient import TestClient

    from huddleroom.main import create_app
    from huddleroom.models.agent import Agent
    from huddleroom.models.event_log import EventLog
    from huddleroom.models.meeting import Meeting, MeetingTurn
    from huddleroom.models.project import Project
    import huddleroom.routers.websocket as websocket_router

    session_factory = _test_session_factory(test_engine)
    async with session_factory() as session:
        project = Project(name="Replay Project", description="test", config={})
        agent = Agent(
            name="replay-agent",
            role="developer",
            provider="openai",
            model="gpt-4o-mini",
            adapter_type="api",
            capabilities=[],
            config={},
        )
        session.add_all([project, agent])
        await session.flush()

        meeting = Meeting(
            project_id=project.id,
            title="Replay Test",
            meeting_type="adhoc",
            participant_agent_ids=[str(agent.id)],
            status="active",
        )
        session.add(meeting)
        await session.flush()

        turn = MeetingTurn(
            meeting_id=meeting.id,
            turn_number=1,
            round_number=1,
            speaker_agent_id=agent.id,
            content="First streamed turn",
            references=[],
            is_human_turn=False,
            prompt_messages=[{"role": "user", "content": "Say the first turn"}],
            raw_response="First streamed turn\n\nRaw detail",
            organizer_selection={"reason": "agent should open the discussion"},
            reasoning_content="recorded reasoning summary",
        )
        session.add(turn)
        await session.flush()

        event = EventLog(
            project_id=project.id,
            event_type="meeting.turn_complete",
            payload={"meeting_id": str(meeting.id), "turn_number": 1},
            source="system",
            emitted_at=datetime.now(timezone.utc),
        )
        session.add(event)
        await session.commit()

    monkeypatch.setattr(websocket_router, "AsyncSessionLocal", session_factory)
    app = create_app()

    with closing(TestClient(app)) as client:
        with client.websocket_connect(
            f"/ws/meetings/{meeting.id}?replay_since=1970-01-01T00:00:00+00:00"
        ) as ws:
            data = ws.receive_json()
            ws.close()

    assert data["event_type"] == "meeting.turn_complete"
    assert data["meeting_id"] == str(meeting.id)
    assert data["project_id"] == str(project.id)
    assert data["turn"]["turn_number"] == 1
    assert data["turn"]["content"] == "First streamed turn"
    assert data["turn"]["prompt_messages"] == [{"role": "user", "content": "Say the first turn"}]
    assert data["turn"]["raw_response"] == "First streamed turn\n\nRaw detail"
    assert data["turn"]["organizer_selection"] == {"reason": "agent should open the discussion"}
    assert data["turn"]["reasoning_content"] == "recorded reasoning summary"


@pytest.mark.asyncio
async def test_meeting_websocket_replays_verbose_trace_events(test_engine, monkeypatch):
    from starlette.testclient import TestClient

    from huddleroom.main import create_app
    from huddleroom.models.event_log import EventLog
    from huddleroom.models.meeting import Meeting
    from huddleroom.models.project import Project
    import huddleroom.routers.websocket as websocket_router

    session_factory = _test_session_factory(test_engine)
    async with session_factory() as session:
        project = Project(name="Verbose Replay Project", description="test", config={})
        session.add(project)
        await session.flush()

        meeting = Meeting(
            project_id=project.id,
            title="Verbose Replay Test",
            meeting_type="adhoc",
            participant_agent_ids=[],
            status="active",
        )
        session.add(meeting)
        await session.flush()

        event = EventLog(
            project_id=project.id,
            event_type="meeting.trace",
            payload={
                "meeting_id": str(meeting.id),
                "trace": {
                    "kind": "participant_turn",
                    "stage": "response",
                    "actor_id": "agent-123",
                    "raw_response": "I think we should proceed.",
                },
            },
            source="system",
            emitted_at=datetime.now(timezone.utc),
        )
        session.add(event)
        await session.commit()

    monkeypatch.setattr(websocket_router, "AsyncSessionLocal", session_factory)
    app = create_app()

    with closing(TestClient(app)) as client:
        with client.websocket_connect(
            f"/ws/meetings/{meeting.id}?replay_since=1970-01-01T00:00:00+00:00"
        ) as ws:
            data = ws.receive_json()
            ws.close()

    assert data["event_type"] == "meeting.trace"
    assert data["meeting_id"] == str(meeting.id)
    assert data["payload"]["trace"]["kind"] == "participant_turn"
    assert data["payload"]["trace"]["raw_response"] == "I think we should proceed."


@pytest.mark.asyncio
async def test_meeting_websocket_streams_only_target_meeting(test_engine, monkeypatch):
    from starlette.testclient import TestClient

    from huddleroom.main import create_app
    from huddleroom.models.agent import Agent
    from huddleroom.models.meeting import Meeting, MeetingTurn
    from huddleroom.models.project import Project
    from huddleroom.services.event_bus import BusEvent, get_event_bus
    import huddleroom.routers.websocket as websocket_router

    session_factory = _test_session_factory(test_engine)
    async with session_factory() as session:
        project = Project(name="Target Project", description="test", config={})
        agent = Agent(
            name="target-agent",
            role="developer",
            provider="openai",
            model="gpt-4o-mini",
            adapter_type="api",
            capabilities=[],
            config={},
        )
        session.add_all([project, agent])
        await session.flush()

        target_meeting = Meeting(
            project_id=project.id,
            title="Target Meeting",
            meeting_type="adhoc",
            participant_agent_ids=[str(agent.id)],
            status="active",
        )
        other_meeting = Meeting(
            project_id=project.id,
            title="Other Meeting",
            meeting_type="adhoc",
            participant_agent_ids=[str(agent.id)],
            status="active",
        )
        session.add_all([target_meeting, other_meeting])
        await session.flush()

        target_turn = MeetingTurn(
            meeting_id=target_meeting.id,
            turn_number=1,
            round_number=1,
            speaker_agent_id=agent.id,
            content="Target turn",
            references=[],
            is_human_turn=False,
        )
        other_turn = MeetingTurn(
            meeting_id=other_meeting.id,
            turn_number=1,
            round_number=1,
            speaker_agent_id=agent.id,
            content="Other turn",
            references=[],
            is_human_turn=False,
        )
        session.add_all([target_turn, other_turn])
        await session.commit()

    monkeypatch.setattr(websocket_router, "AsyncSessionLocal", session_factory)
    app = create_app()
    bus = get_event_bus()

    with closing(TestClient(app)) as client:
        with client.websocket_connect(f"/ws/meetings/{target_meeting.id}") as ws:
            await asyncio.sleep(0.05)
            bus.put(
                BusEvent(
                    id=uuid.uuid4(),
                    project_id=project.id,
                    event_type="meeting.turn_complete",
                    payload={
                        "meeting_id": str(other_meeting.id),
                        "turn_number": 1,
                        "turn": {"content": "Other turn"},
                    },
                    source="system",
                    emitted_at=datetime.now(timezone.utc),
                )
            )
            bus.put(
                BusEvent(
                    id=uuid.uuid4(),
                    project_id=project.id,
                    event_type="meeting.turn_complete",
                    payload={
                        "meeting_id": str(target_meeting.id),
                        "turn_number": 1,
                        "turn": {"content": "Target turn"},
                    },
                    source="system",
                    emitted_at=datetime.now(timezone.utc),
                )
            )
            data = ws.receive_json()
            ws.close()

    assert data["meeting_id"] == str(target_meeting.id)
    assert data["turn"]["content"] == "Target turn"
