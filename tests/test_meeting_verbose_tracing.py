from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

import pytest
from sqlalchemy import select


@pytest.mark.asyncio
async def test_execute_agent_turn_emits_verbose_trace_event(db_session, test_project, test_agent):
    from huddleroom.models.event_log import EventLog
    from huddleroom.services.event_bus import get_event_bus
    from huddleroom.services.meeting_runner import MeetingRunner
    from huddleroom.services.meeting_service import MeetingService

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Verbose Trace Meeting",
        meeting_type="review",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "Trace participant turn", "max_rounds": 1}],
        auto_start=False,
    )
    await svc.transition_to_preparing(db=db_session, meeting=meeting)
    await svc.transition_to_active(db=db_session, meeting=meeting)

    payload = "Detailed response from the participant."

    class _Stream:
        def __init__(self):
            self._chunks = [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=payload, reasoning_content=None)
                        )
                    ]
                )
            ]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._chunks:
                return self._chunks.pop()
            raise StopAsyncIteration

    async def mock_acompletion(**_kwargs):
        return _Stream()

    def mock_builder(chunks, messages):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=payload, reasoning_content=None),
                    finish_reason="stop"
                )
            ],
            usage=SimpleNamespace(prompt_tokens=15, completion_tokens=17)
        )

    runner = MeetingRunner(bus=get_event_bus())
    with patch("huddleroom.services.meeting_runner.litellm.acompletion", new=mock_acompletion), \
         patch("huddleroom.services.agent_response_stream.litellm.stream_chunk_builder", new=mock_builder):
        await runner.execute_agent_turn(db=db_session, meeting=meeting, agent=test_agent)

    result = await db_session.execute(
        select(EventLog)
        .where(EventLog.project_id == test_project.id, EventLog.event_type == "meeting.trace")
        .order_by(EventLog.emitted_at.desc())
    )
    trace_events = list(result.scalars().all())

    assert len(trace_events) >= 2
    request_event = next(ev for ev in trace_events if ev.payload["trace"]["stage"] == "request")
    response_event = next(ev for ev in trace_events if ev.payload["trace"]["stage"] == "response")

    assert request_event.payload["meeting_id"] == str(meeting.id)
    assert request_event.payload["trace"]["kind"] == "participant_turn"
    assert request_event.payload["trace"]["messages"]
    assert response_event.payload["trace"]["model"] == test_agent.model
    assert response_event.payload["trace"]["provider"] == test_agent.provider
    assert response_event.payload["trace"]["raw_response"] == "Detailed response from the participant."


@pytest.mark.asyncio
async def test_organizer_agent_select_emits_verbose_trace_events(db_session, test_project, test_agent):
    from huddleroom.models.event_log import EventLog
    from huddleroom.services.event_bus import get_event_bus
    from huddleroom.services.meeting_runner import MeetingRunner
    from huddleroom.services.meeting_service import MeetingService

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Organizer Trace Meeting",
        meeting_type="review",
        participant_agent_ids=[str(test_agent.id)],
        organizer_agent_id=test_agent.id,
        agenda_items=[{"order": 1, "title": "Pick next speaker", "max_rounds": 1}],
        auto_start=False,
    )

    payload = f'{{"next_speaker_id": "{test_agent.id}", "reason": "Only participant available."}}'

    class _Stream:
        def __init__(self):
            self._chunks = [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=payload, reasoning_content=None)
                        )
                    ]
                )
            ]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._chunks:
                return self._chunks.pop()
            raise StopAsyncIteration

    async def mock_acompletion(**_kwargs):
        return _Stream()

    def mock_builder(chunks, messages):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=payload, reasoning_content=None),
                    finish_reason="stop"
                )
            ],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7)
        )

    runner = MeetingRunner(bus=get_event_bus())
    with patch("huddleroom.services.meeting_runner.litellm.acompletion", new=mock_acompletion), \
         patch("huddleroom.services.agent_response_stream.litellm.stream_chunk_builder", new=mock_builder):
        selected = await runner._organizer_agent_select(db=db_session, meeting=meeting, organizer=test_agent)

    assert selected["next_speaker_id"] == str(test_agent.id)

    result = await db_session.execute(
        select(EventLog)
        .where(EventLog.project_id == test_project.id, EventLog.event_type == "meeting.trace")
        .order_by(EventLog.emitted_at.asc())
    )
    trace_events = [
        ev
        for ev in result.scalars().all()
        if ev.payload["trace"].get("kind") == "organizer_selection"
    ]

    assert len(trace_events) == 2
    request_event, response_event = trace_events

    assert request_event.payload["meeting_id"] == str(meeting.id)
    assert request_event.payload["trace"]["stage"] == "request"
    assert request_event.payload["trace"]["messages"]
    assert request_event.payload["trace"]["model"] == f"{test_agent.provider}/{test_agent.model}"
    assert request_event.payload["trace"]["provider"] == test_agent.provider
    assert request_event.payload["trace"]["max_tokens"] == 2000
    assert request_event.payload["trace"]["organizer_agent_id"] == str(test_agent.id)
    assert response_event.payload["meeting_id"] == str(meeting.id)
    assert response_event.payload["trace"]["stage"] == "response"
    assert response_event.payload["trace"]["raw_response"] == payload
    assert response_event.payload["trace"]["max_tokens"] == 2000
    assert response_event.payload["trace"]["finish_reason"] == "stop"
    assert response_event.payload["trace"]["prompt_tokens"] == 11
    assert response_event.payload["trace"]["completion_tokens"] == 7
