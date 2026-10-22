import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

import pytest
from sqlalchemy import select


@pytest.fixture
def db_session(autobegin_db_session):
    """Override: these tests drive runner code that commits mid-turn."""
    return autobegin_db_session



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


_CONTROL = "openrouter/vendor/control-model"


def _completion_resp(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, reasoning_content=None), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


async def _make_meeting(db_session, project, provider, model, adapter_type):
    from huddleroom.models.agent import Agent
    from huddleroom.services.meeting_service import MeetingService

    organizer = Agent(
        name=f"organizer-{uuid.uuid4()}",
        role="pm",
        provider=provider,
        model=model,
        adapter_type=adapter_type,
        cli_runtime=provider if adapter_type == "cli" else None,
        capabilities=[],
        config={},
    )
    db_session.add(organizer)
    await db_session.flush()
    meeting = await MeetingService().create_meeting(
        db=db_session,
        project_id=project.id,
        title="Control model meeting",
        meeting_type="review",
        participant_agent_ids=[str(organizer.id)],
        organizer_agent_id=organizer.id,
        agenda_items=[{"order": 1, "title": "Pick", "max_rounds": 1}],
        auto_start=False,
    )
    return organizer, meeting


async def _select(db_session, meeting, organizer, mock_completion, bus=None):
    from huddleroom.services.meeting_runner import MeetingRunner

    with patch("huddleroom.services.meeting_runner.litellm.acompletion", new=mock_completion), \
         patch("huddleroom.services.meeting_intelligence.settings.meeting_control_model", _CONTROL):
        return await MeetingRunner(bus=bus)._organizer_agent_select(
            db=db_session, meeting=meeting, organizer=organizer
        )


@pytest.mark.asyncio
async def test_organizer_agent_select_cli_organizer_uses_meeting_control_model(db_session, test_project):
    organizer, meeting = await _make_meeting(db_session, test_project, "codex", "gpt-6.1-sol", "cli")
    resp = _completion_resp(f'{{"next_speaker_id": "{organizer.id}", "reason": "Only one."}}')
    mock_completion = AsyncMock(return_value=resp)

    selected = await _select(db_session, meeting, organizer, mock_completion)

    assert [c.kwargs["model"] for c in mock_completion.call_args_list] == [_CONTROL]
    assert selected["next_speaker_id"] == str(organizer.id)
    assert selected["reason"] == "Only one."


@pytest.mark.asyncio
async def test_organizer_agent_select_api_organizer_falls_back_to_control_model(db_session, test_project):
    organizer, meeting = await _make_meeting(db_session, test_project, "openai", "gpt-4o-mini", "api")
    resp = _completion_resp(f'{{"next_speaker_id": "{organizer.id}", "reason": "ok"}}')
    # Own model raises on every attempt; the control model answers.
    mock_completion = AsyncMock(side_effect=lambda **kw: resp if kw["model"] == _CONTROL else (_ for _ in ()).throw(RuntimeError("boom")))

    selected = await _select(db_session, meeting, organizer, mock_completion)

    models = [c.kwargs["model"] for c in mock_completion.call_args_list]
    assert models[0] == "openai/gpt-4o-mini"
    assert models[-1] == _CONTROL
    assert set(models) == {"openai/gpt-4o-mini", _CONTROL}
    assert selected["model_used"] == _CONTROL


@pytest.mark.asyncio
async def test_organizer_agent_select_dedupes_when_own_model_is_control_model(db_session, test_project):
    organizer, meeting = await _make_meeting(db_session, test_project, "openrouter", "vendor/control-model", "api")
    resp = _completion_resp(f'{{"next_speaker_id": "{organizer.id}", "reason": "ok"}}')
    mock_completion = AsyncMock(return_value=resp)

    await _select(db_session, meeting, organizer, mock_completion)

    assert [c.kwargs["model"] for c in mock_completion.call_args_list] == [_CONTROL]


@pytest.mark.asyncio
async def test_organizer_agent_select_labels_match_control_model_for_cli_organizer(db_session, test_project):
    from huddleroom.models.event_log import EventLog
    from huddleroom.services.agent_response_stream import InvocationContext
    from huddleroom.services.event_bus import get_event_bus

    organizer, meeting = await _make_meeting(db_session, test_project, "codex", "gpt-6.1-sol", "cli")
    resp = _completion_resp(f'{{"next_speaker_id": "{organizer.id}", "reason": "ok"}}')
    made = []

    def _record(*args, **kwargs):
        made.append(kwargs)
        return InvocationContext(*args, **kwargs)

    with patch("huddleroom.services.meeting_runner.InvocationContext", side_effect=_record):
        await _select(db_session, meeting, organizer, AsyncMock(return_value=resp), bus=get_event_bus())

    assert [m["model_or_runtime"] for m in made] == ["vendor/control-model"]
    result = await db_session.execute(
        select(EventLog).where(EventLog.project_id == test_project.id, EventLog.event_type == "meeting.trace")
    )
    traces = [e.payload["trace"] for e in result.scalars().all() if e.payload["trace"].get("kind") == "organizer_selection"]
    assert len(traces) == 2
    assert {t["model"] for t in traces} == {_CONTROL}
    assert {t["provider"] for t in traces} == {"openrouter"}


@pytest.mark.asyncio
async def test_probe_for_signals_cli_agent_uses_meeting_control_model(db_session, test_project):
    from huddleroom.models.agent import Agent
    from huddleroom.models.meeting import MeetingTurn
    from huddleroom.services.agent_response_stream import InvocationContext
    from huddleroom.services.meeting_runner import MeetingRunner

    speaker, meeting = await _make_meeting(db_session, test_project, "openai", "gpt-4o-mini", "api")
    listener = Agent(
        name=f"listener-{uuid.uuid4()}", role="pm", provider="codex", model="gpt-6.1-sol",
        adapter_type="cli", cli_runtime="codex", capabilities=[], config={},
    )
    db_session.add(listener)
    await db_session.flush()
    meeting.participant_agent_ids = [str(speaker.id), str(listener.id)]
    turn = MeetingTurn(
        meeting_id=meeting.id, turn_number=1, round_number=1,
        speaker_agent_id=speaker.id, content="POSITION: X", references=[],
    )
    db_session.add(turn)
    await db_session.flush()

    mock_completion = AsyncMock(return_value=_completion_resp("NO"))
    made = []

    def _record(*args, **kwargs):
        made.append(kwargs)
        return InvocationContext(*args, **kwargs)

    with patch("huddleroom.services.meeting_runner.litellm.acompletion", new=mock_completion), \
         patch("huddleroom.services.meeting_intelligence.settings.meeting_control_model", _CONTROL), \
         patch("huddleroom.services.meeting_runner.InvocationContext", side_effect=_record), \
         patch("huddleroom.services.event_bus.emit_event", new_callable=AsyncMock) as mock_emit:
        await MeetingRunner(bus=object())._probe_for_signals(
            db=db_session, meeting=meeting, speaking_agent_id=speaker.id, turn=turn
        )

    assert [c.kwargs["model"] for c in mock_completion.call_args_list] == [_CONTROL]
    assert [m["model_or_runtime"] for m in made] == ["vendor/control-model"]
    assert mock_emit.await_args_list[0].kwargs["payload"]["probe_model"] == _CONTROL
