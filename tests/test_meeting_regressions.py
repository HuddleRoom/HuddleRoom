import asyncio
import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.exc import OperationalError


@contextmanager
def patch_streaming_acompletion(content):
    """Patch meeting_outcome's litellm.acompletion to stream `content`, and patch the
    agent_response_stream reconstruction seam to rebuild it into a usable response.
    """

    class _Stream:
        def __init__(self):
            self._chunks = [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=content, reasoning_content=None)
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
                    message=SimpleNamespace(
                        content=content, reasoning_content=None, tool_calls=None
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
        )

    with patch("huddleroom.services.meeting_outcome.litellm.acompletion") as api_call, patch(
        "huddleroom.services.agent_response_stream.litellm.stream_chunk_builder", new=mock_builder
    ):
        api_call.side_effect = mock_acompletion
        yield api_call


@pytest.mark.asyncio
async def test_consumer_does_not_dispatch_run_turn_on_turn_complete(db_session):
    from huddleroom.services.event_bus import BusEvent
    from huddleroom.workers.consumers.meeting_engine import MeetingEngineConsumer

    consumer = MeetingEngineConsumer()
    event = BusEvent(
        id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        event_type="meeting.turn_complete",
        payload={"meeting_id": str(uuid.uuid4())},
        source="system",
        emitted_at=datetime.now(timezone.utc),
    )

    with patch("huddleroom.workers.meeting_tasks.dispatch_run_meeting_turn") as mock_dispatch:
        await consumer.process_event(db=db_session, event=event)

    mock_dispatch.assert_not_called()


@pytest.mark.asyncio
async def test_start_meeting_dispatches_first_turn(db_session, test_project, test_agent):
    from huddleroom.services.meeting_service import MeetingService
    from huddleroom.workers.meeting_tasks import start_meeting_async

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Start Task Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "Item 1", "question": "Decide?", "max_rounds": 1}],
    )

    db_session.commit = AsyncMock()
    db_session.rollback = AsyncMock()

    with patch("huddleroom.workers.meeting_tasks.AsyncSessionLocal") as mock_session_factory:
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=db_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("huddleroom.services.meeting_context.MeetingContextService.build_initial_context", new_callable=AsyncMock) as mock_ctx:
            mock_ctx.return_value = "context"
            with patch("huddleroom.workers.meeting_tasks.dispatch_run_meeting_turn") as mock_dispatch:
                await start_meeting_async(str(meeting.id))

    mock_dispatch.assert_called_once_with(str(meeting.id), meeting.project_id)


@pytest.mark.asyncio
async def test_timeout_dispatches_finalize_meeting(db_session, test_project):
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem
    from huddleroom.workers.meeting_tasks import meeting_timeout_async

    meeting = Meeting(
        project_id=test_project.id,
        title="Timeout Test",
        meeting_type="decision",
        participant_agent_ids=[],
        status="active",
    )
    db_session.add(meeting)
    await db_session.flush()

    item = MeetingAgendaItem(
        meeting_id=meeting.id,
        order=1,
        title="Q1",
        question="Decide?",
        status="active",
    )
    db_session.add(item)
    await db_session.flush()

    db_session.commit = AsyncMock()
    db_session.rollback = AsyncMock()

    with patch("huddleroom.workers.meeting_tasks.AsyncSessionLocal") as mock_session_factory:
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=db_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("huddleroom.workers.meeting_tasks.dispatch_finalize_meeting") as mock_dispatch:
            await meeting_timeout_async(str(meeting.id))

    assert meeting.status == "concluding"
    assert meeting.is_partial is True
    assert item.status == "abandoned"
    mock_dispatch.assert_called_once_with(str(meeting.id), meeting.project_id)


@pytest.mark.asyncio
async def test_timeout_preserves_human_intervention_outcome_for_active_item(
    db_session, test_project, test_agent
):
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingEvent, MeetingTurn
    from huddleroom.workers.meeting_tasks import meeting_timeout_async

    meeting = Meeting(
        project_id=test_project.id,
        title="Timeout with partial discussion",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        status="active",
        turn_strategy="organizer_controlled",
        deadlock_strategy="human_intervention",
    )
    db_session.add(meeting)
    await db_session.flush()

    active_item = MeetingAgendaItem(
        meeting_id=meeting.id,
        order=1,
        title="Q1",
        question="Decide?",
        status="active",
    )
    pending_item = MeetingAgendaItem(
        meeting_id=meeting.id,
        order=2,
        title="Q2",
        question="Later?",
        status="pending",
    )
    db_session.add_all([active_item, pending_item])
    await db_session.flush()

    db_session.add(
        MeetingTurn(
            meeting_id=meeting.id,
            agenda_item_id=active_item.id,
            turn_number=1,
            round_number=1,
            speaker_agent_id=test_agent.id,
            content="POSITION: Technical debt reduction\nRationale: stabilize delivery before new scope.",
            references=[],
            is_human_turn=False,
        )
    )
    await db_session.flush()

    db_session.commit = AsyncMock()
    db_session.rollback = AsyncMock()

    with patch("huddleroom.workers.meeting_tasks.AsyncSessionLocal") as mock_session_factory:
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=db_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("huddleroom.workers.meeting_tasks.dispatch_finalize_meeting") as mock_dispatch:
            await meeting_timeout_async(str(meeting.id))

    events = (
        await db_session.execute(
            select(MeetingEvent)
            .where(MeetingEvent.meeting_id == meeting.id)
            .order_by(MeetingEvent.created_at)
        )
    ).scalars().all()
    event_types = [event.event_type for event in events]

    assert meeting.status == "concluding"
    assert meeting.is_partial is True
    assert active_item.status == "unresolved"
    assert active_item.resolution_kind == "human_intervention"
    assert active_item.participants_heard == [str(test_agent.id)]
    assert pending_item.status == "abandoned"
    assert "human_intervention_required" in event_types
    assert "agenda_item_completed" in event_types
    assert "timeout" in event_types
    mock_dispatch.assert_called_once_with(str(meeting.id), meeting.project_id)


@pytest.mark.asyncio
async def test_human_intervention_deadlock_concludes_organizer_controlled_meeting(
    db_session, test_project, test_agent
):
    from huddleroom.models.agent import Agent
    from huddleroom.models.meeting import MeetingAgendaItem, MeetingEvent, MeetingTurn
    from huddleroom.services.meeting_runner import MeetingRunner
    from huddleroom.services.meeting_service import MeetingService

    second_agent = Agent(
        name=f"test-agent-{uuid.uuid4()}",
        role="developer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=[],
        config={},
    )
    db_session.add(second_agent)
    await db_session.flush()

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Organizer deadlock",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id), str(second_agent.id)],
        agenda_items=[
            {
                "order": 1,
                "title": "Sprint focus",
                "question": "Which sprint focus should be chosen?",
                "options": ["Technical debt reduction", "New feature delivery"],
                "max_rounds": 2,
            }
        ],
        turn_strategy="organizer_controlled",
        deadlock_strategy="human_intervention",
        organizer_agent_id=test_agent.id,
    )
    await svc.transition_to_preparing(db=db_session, meeting=meeting)
    await svc.transition_to_active(db=db_session, meeting=meeting)
    item = await svc.get_current_agenda_item(db=db_session, meeting_id=meeting.id)
    assert item is not None

    db_session.add_all(
        [
            MeetingTurn(
                meeting_id=meeting.id,
                agenda_item_id=item.id,
                turn_number=1,
                round_number=1,
                speaker_agent_id=test_agent.id,
                content="POSITION: Technical debt reduction\nRationale: reduce delivery risk.",
                references=[],
                is_human_turn=False,
            ),
            MeetingTurn(
                meeting_id=meeting.id,
                agenda_item_id=item.id,
                turn_number=2,
                round_number=1,
                speaker_agent_id=second_agent.id,
                content="POSITION: New feature delivery\nRationale: hit customer commitments.",
                references=[],
                is_human_turn=False,
            ),
        ]
    )
    item.current_round = 1
    await db_session.flush()

    db_session.add_all(
        [
            MeetingTurn(
                meeting_id=meeting.id,
                agenda_item_id=item.id,
                turn_number=3,
                round_number=2,
                speaker_agent_id=test_agent.id,
                content="POSITION: Technical debt reduction\nRationale: still the safest choice.",
                references=[],
                is_human_turn=False,
            ),
            MeetingTurn(
                meeting_id=meeting.id,
                agenda_item_id=item.id,
                turn_number=4,
                round_number=2,
                speaker_agent_id=second_agent.id,
                content="POSITION: New feature delivery\nRationale: still the highest ROI.",
                references=[],
                is_human_turn=False,
            ),
        ]
    )
    await db_session.flush()

    runner = MeetingRunner()
    with patch(
        "huddleroom.services.meeting_intelligence.MeetingIntelligenceService.check_consensus",
        new_callable=AsyncMock,
        return_value={
            "consensus": False,
            "confidence": 0.35,
            "agreed_position": "",
            "rationale": "Positions remain split across the same two options.",
        },
    ), patch(
        "huddleroom.services.meeting_intelligence.MeetingIntelligenceService.extract_positions",
        new_callable=AsyncMock,
        side_effect=[
            [{"option": "Technical debt reduction"}, {"option": "New feature delivery"}],
            [{"option": "Technical debt reduction"}, {"option": "New feature delivery"}],
        ],
    ):
        resolved = await runner.evaluate_round_if_complete(db=db_session, meeting=meeting, item=item)

    await db_session.flush()

    events = (
        await db_session.execute(
            select(MeetingEvent)
            .where(MeetingEvent.meeting_id == meeting.id)
            .order_by(MeetingEvent.created_at)
        )
    ).scalars().all()
    event_types = [event.event_type for event in events]

    assert resolved is True
    assert item.is_deadlocked is True
    assert item.status == "unresolved"
    assert item.resolution_kind == "human_intervention"
    assert meeting.status == "concluding"
    assert "deadlock_detected" in event_types
    assert "agenda_completed_all" in event_types


@pytest.mark.asyncio
async def test_run_turn_dispatches_finalize_when_meeting_concluding(db_session, test_project, test_agent):
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingTurn
    from huddleroom.workers.meeting_tasks import run_meeting_turn_async

    meeting = Meeting(
        project_id=test_project.id,
        title="Finalize Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        status="active",
    )
    db_session.add(meeting)
    await db_session.flush()

    item = MeetingAgendaItem(
        meeting_id=meeting.id,
        order=1,
        title="Q1",
        question="Done?",
        status="active",
    )
    db_session.add(item)
    await db_session.flush()

    async def fake_run_next_turn(*_, db, meeting_id):
        current = await db.get(Meeting, meeting_id)
        current.status = "concluding"
        turn = MeetingTurn(
            meeting_id=meeting_id,
            agenda_item_id=item.id,
            turn_number=1,
            round_number=1,
            speaker_agent_id=test_agent.id,
            content="done",
            references=[],
            is_human_turn=False,
        )
        db.add(turn)
        await db.flush()
        return turn

    db_session.commit = AsyncMock()
    db_session.rollback = AsyncMock()

    with patch("huddleroom.workers.meeting_tasks.AsyncSessionLocal") as mock_session_factory:
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=db_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("huddleroom.services.meeting_runner.MeetingRunner.run_next_turn", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = fake_run_next_turn
            with patch(
                "huddleroom.services.meeting_runner.MeetingRunner.evaluate_round_if_complete",
                new_callable=AsyncMock,
                return_value=False,
            ):
                with patch("huddleroom.workers.meeting_tasks.dispatch_finalize_meeting") as mock_dispatch:
                    await run_meeting_turn_async(str(meeting.id))

    mock_dispatch.assert_called_once_with(str(meeting.id), meeting.project_id)


@pytest.mark.asyncio
async def test_run_turn_self_enqueues_next_turn_when_meeting_stays_active(db_session, test_project, test_agent):
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingTurn
    from huddleroom.workers.meeting_tasks import run_meeting_turn_async

    meeting = Meeting(
        project_id=test_project.id,
        title="Continuation Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id), str(uuid.uuid4())],
        status="active",
    )
    db_session.add(meeting)
    await db_session.flush()

    item = MeetingAgendaItem(
        meeting_id=meeting.id,
        order=1,
        title="Q1",
        question="Keep going?",
        status="active",
    )
    db_session.add(item)
    await db_session.flush()

    async def fake_run_next_turn(*_, db, meeting_id):
        turn = MeetingTurn(
            meeting_id=meeting_id,
            agenda_item_id=item.id,
            turn_number=1,
            round_number=1,
            speaker_agent_id=test_agent.id,
            content="first turn",
            references=[],
            is_human_turn=False,
        )
        db.add(turn)
        await db.flush()
        return turn

    call_order: list[str] = []
    db_session.commit = AsyncMock(side_effect=lambda: call_order.append("commit"))
    db_session.rollback = AsyncMock()

    with patch("huddleroom.workers.meeting_tasks.AsyncSessionLocal") as mock_session_factory:
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=db_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("huddleroom.services.meeting_runner.MeetingRunner.run_next_turn", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = fake_run_next_turn
            with patch(
                "huddleroom.services.meeting_runner.MeetingRunner.evaluate_round_if_complete",
                new_callable=AsyncMock,
                return_value=False,
            ):
                with patch("huddleroom.workers.meeting_tasks.dispatch_run_meeting_turn") as mock_dispatch:
                    mock_dispatch.side_effect = lambda *_args, **_kwargs: call_order.append("dispatch")
                    await run_meeting_turn_async(str(meeting.id))

    assert call_order == ["commit", "dispatch"]
    mock_dispatch.assert_called_once_with(str(meeting.id), meeting.project_id)


@pytest.mark.parametrize("turn_strategy", ["organizer_controlled", "moderated"])
@pytest.mark.asyncio
async def test_run_turn_retries_when_no_turn_is_produced(
    db_session, test_project, test_agent, turn_strategy
):
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem
    from huddleroom.workers.meeting_tasks import run_meeting_turn_async

    meeting = Meeting(
        project_id=test_project.id,
        title="Retry when no turn produced",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id), str(uuid.uuid4())],
        status="active",
        turn_strategy=turn_strategy,
    )
    db_session.add(meeting)
    await db_session.flush()

    item = MeetingAgendaItem(
        meeting_id=meeting.id,
        order=1,
        title="Q1",
        question="Who speaks next?",
        status="active",
    )
    db_session.add(item)
    await db_session.flush()

    call_order: list[str] = []
    db_session.commit = AsyncMock(side_effect=lambda: call_order.append("commit"))
    db_session.rollback = AsyncMock()

    with patch("huddleroom.workers.meeting_tasks.AsyncSessionLocal") as mock_session_factory:
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=db_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch(
            "huddleroom.services.meeting_runner.MeetingRunner.run_next_turn",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with patch(
                "huddleroom.services.meeting_runner.MeetingRunner.evaluate_round_if_complete",
                new_callable=AsyncMock,
                return_value=False,
            ):
                with patch("huddleroom.workers.meeting_tasks.dispatch_run_meeting_turn") as mock_dispatch:
                    mock_dispatch.side_effect = lambda *_args, **_kwargs: call_order.append("dispatch")
                    await run_meeting_turn_async(str(meeting.id))

    assert call_order == ["commit", "dispatch"]
    mock_dispatch.assert_called_once_with(str(meeting.id), meeting.project_id)


@pytest.mark.asyncio
async def test_run_turn_redispatches_after_sqlite_database_lock(
    db_session, test_project, test_agent
):
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem
    from huddleroom.workers.meeting_tasks import run_meeting_turn_async

    meeting = Meeting(
        project_id=test_project.id,
        title="SQLite lock retry",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        status="active",
        turn_strategy="moderated",
    )
    db_session.add(meeting)
    await db_session.flush()

    item = MeetingAgendaItem(
        meeting_id=meeting.id,
        order=1,
        title="Q1",
        question="Who speaks next?",
        status="active",
    )
    db_session.add(item)
    await db_session.flush()

    db_session.commit = AsyncMock()
    db_session.rollback = AsyncMock()

    lock_error = OperationalError(
        "INSERT INTO meeting_turns ...",
        {},
        Exception("database is locked"),
    )

    with patch("huddleroom.workers.meeting_tasks.AsyncSessionLocal") as mock_session_factory:
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=db_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch(
            "huddleroom.services.meeting_runner.MeetingRunner.run_next_turn",
            new_callable=AsyncMock,
            side_effect=lock_error,
        ):
            with patch("huddleroom.workers.meeting_tasks.dispatch_run_meeting_turn") as mock_dispatch:
                scheduled_tasks = []

                def create_task(coro, **kwargs):
                    task = asyncio.get_running_loop().create_task(coro, **kwargs)
                    scheduled_tasks.append(task)
                    return task

                with patch("huddleroom.workers.meeting_tasks.asyncio.create_task", side_effect=create_task) as mock_create_task:
                    with patch("huddleroom.workers.meeting_tasks.asyncio.sleep", new=AsyncMock()) as mock_sleep:
                        await run_meeting_turn_async(str(meeting.id))

                        db_session.rollback.assert_called_once()
                        db_session.commit.assert_not_called()
                        mock_dispatch.assert_not_called()
                        mock_create_task.assert_called_once()

                        await asyncio.wait_for(scheduled_tasks[0], timeout=1)

                        mock_sleep.assert_awaited_once_with(1)
                        mock_dispatch.assert_called_once_with(str(meeting.id), meeting.project_id)


@pytest.mark.asyncio
async def test_run_turn_constructs_runner_with_event_bus(db_session, test_project, test_agent):
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingTurn
    from huddleroom.workers.meeting_tasks import run_meeting_turn_async

    meeting = Meeting(
        project_id=test_project.id,
        title="Bus Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        status="active",
    )
    db_session.add(meeting)
    await db_session.flush()

    item = MeetingAgendaItem(
        meeting_id=meeting.id,
        order=1,
        title="Q1",
        question="Emit events?",
        status="active",
    )
    db_session.add(item)
    await db_session.flush()

    async def fake_run_next_turn(self, *, db, meeting_id):
        assert self._bus is not None
        current = await db.get(Meeting, meeting_id)
        current.status = "concluding"
        turn = MeetingTurn(
            meeting_id=meeting_id,
            agenda_item_id=item.id,
            turn_number=1,
            round_number=1,
            speaker_agent_id=test_agent.id,
            content="first turn",
            references=[],
            is_human_turn=False,
        )
        db.add(turn)
        await db.flush()
        return turn

    db_session.commit = AsyncMock()
    db_session.rollback = AsyncMock()

    with patch("huddleroom.workers.meeting_tasks.AsyncSessionLocal") as mock_session_factory:
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=db_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("huddleroom.services.meeting_runner.MeetingRunner.run_next_turn", new=fake_run_next_turn):
            with patch(
                "huddleroom.services.meeting_runner.MeetingRunner.evaluate_round_if_complete",
                new_callable=AsyncMock,
                return_value=False,
            ):
                with patch("huddleroom.workers.meeting_tasks.dispatch_finalize_meeting") as mock_finalize:
                    await run_meeting_turn_async(str(meeting.id))

    mock_finalize.assert_called_once_with(str(meeting.id), meeting.project_id)


@pytest.mark.parametrize("status", ["active", "preparing"])
@pytest.mark.asyncio
async def test_end_meeting_transitions_to_concluding(
    client, auth_headers, db_session, test_project, status
):
    from huddleroom.models.meeting import Meeting

    veto_hours = 12 if status == "active" else 6
    meeting = Meeting(
        project_id=test_project.id,
        title="End Meeting Test",
        meeting_type="decision",
        participant_agent_ids=[],
        status=status,
        veto_window_hours=veto_hours,
    )
    db_session.add(meeting)
    await db_session.flush()

    call_order: list[str] = []
    db_session.commit = AsyncMock(side_effect=lambda: call_order.append("commit"))

    with patch("huddleroom.workers.meeting_tasks.dispatch_finalize_meeting") as mock_dispatch:
        mock_dispatch.side_effect = lambda *_args, **_kwargs: call_order.append("dispatch")
        response = await client.post(f"/api/v1/meetings/{meeting.id}/end", headers=auth_headers)

    await db_session.refresh(meeting)
    assert response.status_code == 200
    assert meeting.status == "concluding"
    assert meeting.is_partial is True
    assert meeting.veto_window_hours == veto_hours
    assert response.json()["status"] == "concluding"

    # preparing status transition sets active_started_at
    if status == "preparing":
        assert meeting.active_started_at is not None

    assert call_order == ["commit", "dispatch"]
    mock_dispatch.assert_called_once_with(str(meeting.id), meeting.project_id)


@pytest.mark.asyncio
async def test_end_meeting_rejects_scheduled_status(
    client, auth_headers, db_session, test_project
):
    from huddleroom.models.meeting import Meeting

    meeting = Meeting(
        project_id=test_project.id,
        title="Scheduled End Test",
        meeting_type="decision",
        participant_agent_ids=[],
        status="scheduled",
    )
    db_session.add(meeting)
    await db_session.flush()

    response = await client.post(f"/api/v1/meetings/{meeting.id}/end", headers=auth_headers)

    assert response.status_code == 400
    assert "preparing" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_human_turn_accepts_preparing_status(client, auth_headers, test_project, test_agent, db_session):
    from huddleroom.services.meeting_service import MeetingService

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Human Turn Preparing Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "Item 1", "max_rounds": 2}],
    )
    await svc.transition_to_preparing(db=db_session, meeting=meeting)
    await db_session.flush()

    response = await client.post(
        f"/api/v1/meetings/{meeting.id}/human-turn",
        json={"content": "Proceed while the meeting is still preparing."},
        headers=auth_headers,
    )

    await db_session.refresh(meeting)
    assert response.status_code == 201
    assert meeting.status == "active"
    assert meeting.active_started_at is not None
    assert response.json()["is_human_turn"] is True


@pytest.mark.parametrize("expired", [True, False])
@pytest.mark.asyncio
async def test_veto_window_enforcement(client, auth_headers, db_session, test_project, expired):
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingDecision

    meeting = Meeting(
        project_id=test_project.id,
        title="Veto window test",
        meeting_type="decision",
        participant_agent_ids=[],
        status="concluding",
        veto_window_hours=24,
    )
    db_session.add(meeting)
    await db_session.flush()

    item = MeetingAgendaItem(meeting_id=meeting.id, order=1, title="Question", status="resolved")
    db_session.add(item)
    await db_session.flush()

    # Create decision with timestamp outside or inside veto window
    created_at = datetime.now(timezone.utc) - timedelta(hours=25) if expired else datetime.now(timezone.utc)
    decision = MeetingDecision(
        meeting_id=meeting.id,
        agenda_item_id=item.id,
        title="Question",
        chosen_option="yes",
        rationale="because",
        decided_by="consensus",
        created_at=created_at,
    )
    db_session.add(decision)
    await db_session.flush()

    response = await client.post(
        f"/api/v1/meetings/{meeting.id}/veto-decision",
        json={"decision_id": str(decision.id), "reason": "test veto"},
        headers=auth_headers,
    )

    if expired:
        assert response.status_code == 400
        assert "window" in response.json()["detail"].lower()
    else:
        assert response.status_code == 200
        assert response.json()["is_vetoed"] is True


@pytest.mark.asyncio
async def test_finalize_waits_for_veto_window_before_side_effects(db_session, test_project):
    from huddleroom.models.meeting import Meeting
    from huddleroom.workers.meeting_tasks import finalize_meeting_async

    meeting = Meeting(
        project_id=test_project.id,
        title="Delayed Finalize",
        meeting_type="decision",
        participant_agent_ids=[],
        status="concluding",
        veto_window_hours=24,
        concluding_started_at=datetime.now(timezone.utc),
    )
    db_session.add(meeting)
    await db_session.flush()

    db_session.commit = AsyncMock()
    db_session.rollback = AsyncMock()

    with patch("huddleroom.workers.meeting_tasks.AsyncSessionLocal") as mock_session_factory:
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=db_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("huddleroom.services.meeting_outcome.MeetingOutcomeService.finalize_meeting", new_callable=AsyncMock) as mock_outcome:
            await finalize_meeting_async(str(meeting.id))

    mock_outcome.assert_not_called()
    db_session.commit.assert_not_awaited()
    assert meeting.status == "concluding"


@pytest.mark.asyncio
async def test_zero_veto_window_concluding_meeting_finalizes_immediately(
    db_session, test_project, test_agent
):
    from huddleroom.services.meeting_service import MeetingService
    from huddleroom.workers.meeting_tasks import finalize_meeting_async

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Prompt manual end",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "Ship?", "question": "Ship it?", "max_rounds": 1}],
    )
    await svc.transition_to_preparing(db=db_session, meeting=meeting)
    await svc.transition_to_active(db=db_session, meeting=meeting)
    await svc.transition_to_concluding(db=db_session, meeting=meeting)
    meeting.veto_window_hours = 0
    await db_session.flush()

    db_session.commit = AsyncMock()
    db_session.rollback = AsyncMock()

    with patch("huddleroom.workers.meeting_tasks.AsyncSessionLocal") as mock_session_factory:
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=db_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch(
            "huddleroom.services.meeting_outcome.MeetingOutcomeService.finalize_meeting",
            new_callable=AsyncMock,
            return_value=(False, None),
        ) as mock_outcome:
            delay = await finalize_meeting_async(str(meeting.id))

    assert meeting.veto_window_hours == 0
    assert delay is None
    mock_outcome.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "veto_default",
        "veto_24",
        "user_pending",
        "agent_pending",
        "orchestrator_pending",
        "reviewer_failure",
        "concluded_event",
    ],
)
async def test_final_review_preserves_veto_and_in_process_redispatch(
    client, auth_headers, db_session, test_project, test_agent, test_user, case
):
    from huddleroom.models.meeting import MeetingEvent
    from huddleroom.services.meeting_service import MeetingService
    from huddleroom.workers.meeting_tasks import _run_finalize_in_process, finalize_meeting_async

    if case.startswith("veto_"):
        veto_window_hours = None if case == "veto_default" else 24
        meeting = await MeetingService().create_meeting(
            db=db_session,
            project_id=test_project.id,
            title="Manual final review",
            meeting_type="decision",
            participant_agent_ids=[],
            agenda_items=[],
            veto_window_hours=veto_window_hours,
        )
        await MeetingService().transition_to_preparing(db=db_session, meeting=meeting)
        await MeetingService().transition_to_active(db=db_session, meeting=meeting)
        await db_session.flush()
        expected_veto = 0 if veto_window_hours is None else veto_window_hours

        db_session.commit = AsyncMock()
        db_session.rollback = AsyncMock()
        with patch("huddleroom.workers.meeting_tasks.dispatch_finalize_meeting"):
            response = await client.post(
                f"/api/v1/meetings/{meeting.id}/end", headers=auth_headers
            )

        assert response.status_code == 200
        assert meeting.status == "concluding"
        assert meeting.veto_window_hours == expected_veto

        call_order: list[str] = []
        countdowns = [0] if expected_veto == 0 else [expected_veto * 3600, 0]

        async def finalize_outcome(**_kwargs):
            call_order.append("outcome")
            return True, None

        async def sleep(countdown):
            call_order.append(f"sleep:{countdown}")

        with patch("huddleroom.workers.meeting_tasks.AsyncSessionLocal") as session_factory, patch(
            "huddleroom.workers.meeting_tasks._seconds_until_finalize", side_effect=countdowns
        ), patch(
            "huddleroom.services.meeting_outcome.MeetingOutcomeService.finalize_meeting",
            new=AsyncMock(side_effect=finalize_outcome),
        ) as outcome, patch(
            "huddleroom.services.meeting_outcome.MeetingOutcomeService.ask_final_reviewer",
            new_callable=AsyncMock,
        ) as review, patch(
            "huddleroom.workers.meeting_tasks.asyncio.sleep", new=AsyncMock(side_effect=sleep)
        ) as sleep_mock, patch(
            "huddleroom.services.event_bus.emit_event", new_callable=AsyncMock
        ):
            session_factory.return_value.__aenter__ = AsyncMock(return_value=db_session)
            session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
            await _run_finalize_in_process(str(meeting.id))

        assert call_order == (
            ["outcome"]
            if expected_veto == 0
            else [f"sleep:{expected_veto * 3600}", "outcome"]
        )
        assert outcome.await_count == 1
        review.assert_not_awaited()
        if expected_veto == 0:
            sleep_mock.assert_not_awaited()
        else:
            sleep_mock.assert_awaited_once_with(expected_veto * 3600)
        return

    reviewer_kind = {
        "user_pending": "organizer_user",
        "agent_pending": "organizer_agent",
    }.get(case, "orchestrator")
    reviewer_id = (
        str(test_user.id)
        if reviewer_kind == "organizer_user"
        else str(test_agent.id) if reviewer_kind == "organizer_agent" else None
    )
    meeting = await _add_final_review_meeting(
        db_session,
        test_project,
        organizer_agent_id=test_agent.id if reviewer_kind == "organizer_agent" else None,
        organizer_user_id=test_user.id if reviewer_kind == "organizer_user" else None,
    )
    pending = MeetingEvent(
        meeting_id=meeting.id,
        event_type="meeting_final_pass",
        payload={
            "reviewer_kind": reviewer_kind,
            "reviewer_id": reviewer_id,
            "decisions_made": True,
            "decisions_clear": True,
            "suggested_action_items": ["Implement Redis cache"],
        },
    )
    if case != "concluded_event":
        db_session.add(pending)
        await db_session.flush()

    get_response = await client.get(
        f"/api/v1/meetings/{meeting.id}/final-review", headers=auth_headers
    )
    assert get_response.status_code == 200
    if case == "concluded_event":
        assert get_response.json() is None
    else:
        assert get_response.json() == {
            "reviewer_kind": reviewer_kind,
            "reviewer_id": reviewer_id,
            "decisions_made": True,
            "decisions_clear": True,
            "suggested_action_items": ["Implement Redis cache"],
        }

    call_order = []
    db_session.commit = AsyncMock(side_effect=lambda: call_order.append("commit"))
    db_session.rollback = AsyncMock()
    finalize_result = (True, None) if case == "concluded_event" else (False, pending)
    review_result = (True, True, True, ["Implement Redis cache"])
    review_side_effect = RuntimeError("reviewer unavailable") if case == "reviewer_failure" else None

    with patch("huddleroom.workers.meeting_tasks.AsyncSessionLocal") as session_factory, patch(
        "huddleroom.services.meeting_outcome.MeetingOutcomeService.finalize_meeting",
        new_callable=AsyncMock,
        return_value=finalize_result,
    ), patch(
        "huddleroom.services.meeting_outcome.MeetingOutcomeService.ask_final_reviewer",
        new_callable=AsyncMock,
        return_value=review_result,
        side_effect=review_side_effect,
    ) as review, patch(
        "huddleroom.services.meeting_outcome.MeetingOutcomeService.complete_final_review",
        new_callable=AsyncMock,
        return_value=True,
    ) as complete, patch(
        "huddleroom.workers.meeting_tasks.dispatch_finalize_meeting"
    ) as dispatch, patch(
        "huddleroom.services.event_bus.emit_event", new_callable=AsyncMock
    ) as emit:
        dispatch.side_effect = lambda *_args: call_order.append("dispatch")
        emit.side_effect = lambda **_kwargs: call_order.append("event")
        session_factory.return_value.__aenter__ = AsyncMock(return_value=db_session)
        session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        assert await finalize_meeting_async(str(meeting.id)) is None

    if case == "concluded_event":
        assert call_order == ["event", "commit"]
        emit.assert_awaited_once()
        review.assert_not_awaited()
        complete.assert_not_awaited()
        dispatch.assert_not_called()
    elif case == "user_pending":
        assert call_order == ["commit"]
        emit.assert_not_awaited()
        review.assert_not_awaited()
        complete.assert_not_awaited()
        dispatch.assert_not_called()
    elif case == "reviewer_failure":
        assert call_order == ["commit"]
        db_session.rollback.assert_awaited_once()
        emit.assert_not_awaited()
        complete.assert_not_awaited()
        dispatch.assert_not_called()
    else:
        assert call_order == ["commit", "commit", "commit", "dispatch"]
        emit.assert_not_awaited()
        review.assert_awaited_once()
        complete.assert_awaited_once()
        expected_actor = test_agent.id if case == "agent_pending" else None
        assert complete.await_args.kwargs["actor_agent_id"] == expected_actor
        dispatch.assert_called_once_with(str(meeting.id), meeting.project_id)


@pytest.mark.asyncio
async def test_finalize_meeting_degrades_to_partial_and_concludes_on_postprocessing_failure(
    db_session, test_project
):
    from sqlalchemy import select

    from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingDecision, MeetingEvent
    from huddleroom.services.meeting_outcome import MeetingOutcomeService

    meeting = Meeting(
        project_id=test_project.id,
        title="Partial finalize",
        meeting_type="review",
        participant_agent_ids=[],
        status="concluding",
        concluding_started_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        veto_window_hours=0,
    )
    db_session.add(meeting)
    await db_session.flush()
    item = MeetingAgendaItem(meeting_id=meeting.id, order=1, title="Review item", status="resolved")
    db_session.add(item)
    await db_session.flush()
    db_session.add(
        MeetingDecision(
            meeting_id=meeting.id,
            agenda_item_id=item.id,
            title="Review item",
            chosen_option="Reject current artifact",
            rationale="Security review found blocking issues.",
            decided_by="review_outcome",
        )
    )
    pending = MeetingEvent(
        meeting_id=meeting.id,
        event_type="meeting_final_pass",
        payload={"reviewer_kind": "orchestrator", "reviewer_id": None},
    )
    db_session.add(pending)
    await db_session.flush()
    db_session.add(
        MeetingEvent(
            meeting_id=meeting.id,
            event_type="final_pass_completed",
            payload={"action_items_needed": False},
        )
    )
    await db_session.flush()

    outcome = MeetingOutcomeService()
    with patch.object(outcome, "write_knowledge_items", new_callable=AsyncMock) as mock_write_knowledge:
        mock_write_knowledge.side_effect = RuntimeError("summary extractor failed")
        concluded, pending = await outcome.finalize_meeting(db=db_session, meeting=meeting)

    assert concluded is True
    assert pending is None
    assert meeting.status == "concluded"
    assert meeting.is_partial is True
    partial_event = (
        await db_session.execute(
            select(MeetingEvent)
            .where(
                MeetingEvent.meeting_id == meeting.id,
                MeetingEvent.event_type == "partial_finalization",
            )
        )
    ).scalar_one()
    assert "write_knowledge_items" in partial_event.payload["failed_steps"]


async def _add_final_review_meeting(
    db_session,
    test_project,
    *,
    chosen_option="Implement Redis cache",
    organizer_agent_id=None,
    organizer_user_id=None,
    participant_contexts=None,
):
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingDecision

    meeting = Meeting(
        project_id=test_project.id,
        title="Follow-up decision",
        meeting_type="decision",
        participant_agent_ids=[str(organizer_agent_id)] if organizer_agent_id else [],
        organizer_agent_id=organizer_agent_id,
        organizer_user_id=organizer_user_id,
        participant_contexts=participant_contexts,
        status="concluding",
        concluding_started_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        veto_window_hours=0,
    )
    db_session.add(meeting)
    await db_session.flush()
    agenda_item = MeetingAgendaItem(meeting_id=meeting.id, order=1, title="Cache", status="resolved")
    db_session.add(agenda_item)
    await db_session.flush()
    db_session.add(
        MeetingDecision(
            meeting_id=meeting.id,
            agenda_item_id=agenda_item.id,
            title="Cache",
            chosen_option=chosen_option,
            rationale="Latency requires a cache.",
            decided_by="consensus",
        )
    )
    await db_session.flush()
    return meeting


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    [
        "missing_followup",
        "existing_item",
        "extracted_item",
        "non_action_decision",
        "concurrent_concluded",
        "concurrent_pending",
        "locked_pending_retry",
    ],
)
async def test_finalize_meeting_blocks_only_when_followup_missing(
    db_session, concurrent_sessions, test_project, test_agent, scenario
):
    from sqlalchemy import update

    from huddleroom.models.meeting import Meeting, MeetingActionItem, MeetingEvent
    from huddleroom.services.meeting_outcome import MeetingOutcomeService

    chosen_option = "Keep current approach" if scenario == "non_action_decision" else "Implement Redis cache"
    meeting = await _add_final_review_meeting(
        db_session,
        test_project,
        chosen_option=chosen_option,
        organizer_agent_id=test_agent.id,
    )
    if scenario == "locked_pending_retry":
        from sqlalchemy.sql.dml import Insert

        await db_session.commit()
        session = concurrent_sessions[0]
        session_meeting = await session.get(Meeting, meeting.id)
        outcome = MeetingOutcomeService()
        original_execute = session.execute
        lock_injected = False

        async def execute_with_lock(statement, *args, **kwargs):
            nonlocal lock_injected
            if (
                not lock_injected
                and isinstance(statement, Insert)
                and statement.table.name == "meeting_events"
            ):
                lock_injected = True
                raise OperationalError("INSERT", {}, RuntimeError("database is locked"))
            return await original_execute(statement, *args, **kwargs)

        with patch.object(session, "execute", new=execute_with_lock), patch.object(
            outcome, "extract_action_items", new_callable=AsyncMock, return_value=[]
        ):
            concluded, pending = await outcome.finalize_meeting(
                db=session, meeting=session_meeting
            )
        assert lock_injected is True
        assert concluded is False
        assert pending is not None
        await session.commit()

        review_payload = json.dumps(
            {
                "decisions_made": True,
                "decisions_clear": True,
                "action_items_needed": True,
                "action_items": ["Implement Redis cache"],
            }
        )
        with patch_streaming_acompletion(review_payload) as reviewer_call:
            answer = await outcome.ask_final_reviewer(
                db=session, meeting=session_meeting, pending=pending
            )

        assert answer == (True, True, True, ["Implement Redis cache"])
        reviewer_call.assert_awaited_once()
        return
    if scenario == "concurrent_pending":
        await db_session.commit()
        outcome = MeetingOutcomeService()
        original_final_review_events = outcome._final_review_events
        both_read_events = asyncio.Event()
        readers = 0

        async def synchronized_final_review_events(*args, **kwargs):
            nonlocal readers
            events = await original_final_review_events(*args, **kwargs)
            readers += 1
            if readers == 2:
                both_read_events.set()
            await both_read_events.wait()
            return events

        async def finalize(session):
            session_meeting = await session.get(Meeting, meeting.id)
            result = await outcome.finalize_meeting(db=session, meeting=session_meeting)
            await session.commit()
            return result

        with patch.object(
            outcome, "_final_review_events", side_effect=synchronized_final_review_events
        ), patch.object(
            outcome, "extract_action_items", new_callable=AsyncMock, return_value=[]
        ), patch.object(
            outcome, "create_tasks_from_action_items", new_callable=AsyncMock
        ), patch.object(
            outcome, "write_knowledge_items", new_callable=AsyncMock
        ), patch.object(outcome, "resolve_graph_run", new_callable=AsyncMock):
            results = await asyncio.gather(*(finalize(session) for session in concurrent_sessions))
        assert [pending is not None for _concluded, pending in results].count(True) == 1
        assert all(concluded is False for concluded, _pending in results)
        verification_session = concurrent_sessions[0]
        events = list(
            (
                await verification_session.execute(
                    select(MeetingEvent).where(
                        MeetingEvent.meeting_id == meeting.id,
                        MeetingEvent.event_type == "meeting_final_pass",
                    )
                )
            ).scalars().all()
        )
        assert len(events) == 1
        return
    if scenario in {"existing_item", "concurrent_concluded"}:
        db_session.add(MeetingActionItem(meeting_id=meeting.id, description="Implement Redis cache"))
        await db_session.flush()
    if scenario == "concurrent_concluded":
        await db_session.execute(
            update(Meeting)
            .where(Meeting.id == meeting.id)
            .values(status="concluded")
            .execution_options(synchronize_session=False)
        )
        assert meeting.status == "concluding"

    async def extract_item(**_kwargs):
        if scenario != "extracted_item":
            return []
        item = MeetingActionItem(meeting_id=meeting.id, description="Implement Redis cache")
        db_session.add(item)
        await db_session.flush()
        return [item]

    outcome = MeetingOutcomeService()
    with patch.object(outcome, "extract_action_items", new=AsyncMock(side_effect=extract_item)) as extract, patch.object(
        outcome, "create_tasks_from_action_items", new_callable=AsyncMock
    ) as create_tasks, patch.object(
        outcome, "write_knowledge_items", new_callable=AsyncMock
    ) as write_knowledge, patch.object(outcome, "resolve_graph_run", new_callable=AsyncMock) as resolve_graph_run:
        result = await outcome.finalize_meeting(db=db_session, meeting=meeting)
        if scenario == "missing_followup":
            assert result[0] is False
            assert result[1] is not None
            assert await outcome.finalize_meeting(db=db_session, meeting=meeting) == (False, None)

    if scenario == "missing_followup":
        event = result[1]
        assert meeting.status == "concluding"
        assert event.event_type == "meeting_final_pass"
        assert event.payload == {
            "reviewer_kind": "organizer_agent",
            "reviewer_id": str(test_agent.id),
            "decisions_made": True,
            "decisions_clear": True,
            "suggested_action_items": ["Implement Redis cache"],
        }
        events = (
            await db_session.execute(
                select(MeetingEvent).where(
                    MeetingEvent.meeting_id == meeting.id,
                    MeetingEvent.event_type == "meeting_final_pass",
                )
            )
        ).scalars().all()
        assert len(events) == 1
        create_tasks.assert_not_awaited()
        write_knowledge.assert_not_awaited()
        resolve_graph_run.assert_not_awaited()
    elif scenario == "concurrent_concluded":
        assert result == (False, None)
        assert meeting.status == "concluded"
        extract.assert_not_awaited()
        create_tasks.assert_not_awaited()
        write_knowledge.assert_not_awaited()
        resolve_graph_run.assert_not_awaited()
    else:
        assert result == (True, None)
        assert meeting.status == "concluded"
        if scenario == "existing_item":
            extract.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("completion", ["items", "waiver"])
async def test_complete_final_review_items_or_waiver(
    db_session, test_project, test_agent, test_user, completion
):
    from huddleroom.models.meeting import MeetingActionItem, MeetingEvent
    from huddleroom.models.task import Task
    from huddleroom.services.meeting_outcome import MeetingOutcomeService

    meeting = await _add_final_review_meeting(
        db_session, test_project, organizer_agent_id=test_agent.id
    )
    outcome = MeetingOutcomeService()
    with patch.object(outcome, "extract_action_items", new_callable=AsyncMock, return_value=[]), patch.object(
        outcome, "write_knowledge_items", new_callable=AsyncMock
    ), patch.object(outcome, "resolve_graph_run", new_callable=AsyncMock):
        assert (await outcome.finalize_meeting(db=db_session, meeting=meeting))[0] is False

    submitted = ["  Implement Redis cache  "] if completion == "items" else []
    assert await outcome.complete_final_review(
        db=db_session,
        meeting=meeting,
        decisions_made=True,
        decisions_clear=False,
        action_items_needed=completion == "items",
        action_items=submitted,
        actor_user_id=test_user.id,
    ) is True

    completed = (
        await db_session.execute(
            select(MeetingEvent).where(
                MeetingEvent.meeting_id == meeting.id,
                MeetingEvent.event_type == "final_pass_completed",
            )
        )
    ).scalar_one()
    assert completed.payload == {
        "decisions_made": True,
        "decisions_clear": False,
        "action_items_needed": completion == "items",
    }
    assert completed.actor_user_id == test_user.id
    assert completed.actor_agent_id is None

    with patch.object(outcome, "extract_action_items", new_callable=AsyncMock) as extract, patch.object(
        outcome, "write_knowledge_items", new_callable=AsyncMock
    ), patch.object(outcome, "resolve_graph_run", new_callable=AsyncMock):
        assert await outcome.finalize_meeting(db=db_session, meeting=meeting) == (True, None)

    extract.assert_not_awaited()
    items = (
        await db_session.execute(select(MeetingActionItem).where(MeetingActionItem.meeting_id == meeting.id))
    ).scalars().all()
    tasks = (
        await db_session.execute(select(Task).where(Task.created_by_meeting_id == meeting.id))
    ).scalars().all()
    assert len(items) == len(tasks) == (1 if completion == "items" else 0)
    assert meeting.status == "concluded"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reviewer", "cli_context"),
    [
        ("api_agent", None),
        ("cli_agent", {"cli_session_id": "session-123"}),
        ("cli_agent", "LEGACY_INITIAL_CONTEXT"),
        ("organizer_user", None),
        ("orchestrator", None),
    ],
)
async def test_final_review_reviewer_precedence_and_invocation(
    db_session, test_project, test_agent, test_user, reviewer, cli_context
):
    from huddleroom.models.agent import Agent
    from huddleroom.services.meeting_outcome import MeetingOutcomeService, _OUTCOME_MODEL

    agent = test_agent
    participant_contexts = None
    if reviewer == "cli_agent":
        agent = Agent(
            name=f"cli-reviewer-{uuid.uuid4()}",
            role="reviewer",
            provider="anthropic",
            model="claude-sonnet-5-5",
            adapter_type="cli",
            cli_runtime="claude_code",
            capabilities=[],
            config={},
        )
        db_session.add(agent)
        await db_session.flush()
        participant_contexts = {str(agent.id): cli_context}

    organizer_agent_id = agent.id if reviewer in {"api_agent", "cli_agent"} else None
    organizer_user_id = test_user.id if reviewer != "orchestrator" else None
    meeting = await _add_final_review_meeting(
        db_session,
        test_project,
        organizer_agent_id=organizer_agent_id,
        organizer_user_id=organizer_user_id,
        participant_contexts=participant_contexts,
    )
    outcome = MeetingOutcomeService()
    with patch.object(outcome, "extract_action_items", new_callable=AsyncMock, return_value=[]):
        concluded, pending = await outcome.finalize_meeting(db=db_session, meeting=meeting)
    assert concluded is False
    expected_kind = "organizer_agent" if organizer_agent_id else (
        "organizer_user" if organizer_user_id else "orchestrator"
    )
    assert pending.payload["reviewer_kind"] == expected_kind
    assert pending.payload["reviewer_id"] == (
        str(organizer_agent_id or organizer_user_id) if expected_kind != "orchestrator" else None
    )

    review_content = json.dumps(
        {
            "decisions_made": True,
            "decisions_clear": True,
            "action_items_needed": True,
            "action_items": ["Implement Redis cache"],
        }
    )
    with patch_streaming_acompletion(review_content) as api_call, patch(
        "huddleroom.adapters.cli_adapter.CliAdapter.run_meeting_turn",
        new_callable=AsyncMock,
        return_value=(review_content, "session-456", 10),
    ) as cli_call:
        if reviewer == "organizer_user":
            with pytest.raises(
                ValueError, match="organizer user review requires interactive submission"
            ):
                await outcome.ask_final_reviewer(db=db_session, meeting=meeting, pending=pending)
            answer = None
        else:
            answer = await outcome.ask_final_reviewer(db=db_session, meeting=meeting, pending=pending)

    if reviewer == "api_agent":
        assert answer == (True, True, True, ["Implement Redis cache"])
        assert api_call.await_args.kwargs["model"] == "openai/gpt-4o-mini"
        cli_call.assert_not_awaited()
    elif reviewer == "cli_agent":
        assert answer == (True, True, True, ["Implement Redis cache"])
        api_call.assert_not_awaited()
        assert cli_call.await_args.kwargs["project"] == test_project
        assert cli_call.await_args.kwargs["existing_session_id"] == (
            "session-123" if isinstance(cli_context, dict) else None
        )
        if isinstance(cli_context, str):
            assert meeting.participant_contexts[str(agent.id)]["initial_ctx"] == cli_context
        assert meeting.participant_contexts[str(agent.id)]["cli_session_id"] == "session-456"
    elif reviewer == "orchestrator":
        assert answer == (True, True, True, ["Implement Redis cache"])
        assert api_call.await_args.kwargs["model"] == _OUTCOME_MODEL
        cli_call.assert_not_awaited()
    else:
        api_call.assert_not_awaited()
        cli_call.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "needed_without_items",
        "waiver_with_items",
        "blank",
        "too_long",
        "serialized_retry",
        "concurrent_completion",
        "concurrent_cancel",
        "http_organizer",
        "http_member",
        "http_forbidden",
        "http_errors",
    ],
)
async def test_complete_final_review_validation_and_idempotency(
    client, auth_headers, db_session, concurrent_sessions, test_project, test_agent, test_user, case
):
    from sqlalchemy import update

    from huddleroom.models.meeting import Meeting, MeetingActionItem, MeetingEvent
    from huddleroom.services.meeting_outcome import MeetingOutcomeService
    from huddleroom.services.meeting_service import MeetingTransitionError

    meeting = await _add_final_review_meeting(db_session, test_project)
    if case in {"http_organizer", "http_errors"}:
        meeting.organizer_user_id = test_user.id
    elif case == "http_member":
        meeting.participant_user_ids = [str(test_user.id)]
    pending = MeetingEvent(
        meeting_id=meeting.id,
        event_type="meeting_final_pass",
        payload={
            "reviewer_kind": "orchestrator",
            "reviewer_id": None,
            "decisions_made": True,
            "decisions_clear": True,
            "suggested_action_items": ["Implement Redis cache"],
        },
    )
    db_session.add(pending)
    await db_session.flush()
    outcome = MeetingOutcomeService()
    verification_session = db_session

    invalid_args = {
        "needed_without_items": (True, []),
        "waiver_with_items": (False, ["Implement Redis cache"]),
        "blank": (True, ["  "]),
        "too_long": (True, ["x" * 201]),
    }
    if case in invalid_args:
        needed, items = invalid_args[case]
        with pytest.raises(ValueError):
            await outcome.complete_final_review(
                db=db_session,
                meeting=meeting,
                decisions_made=True,
                decisions_clear=True,
                action_items_needed=needed,
                action_items=items,
                actor_user_id=test_user.id,
            )
    elif case == "serialized_retry":
        first = await outcome.complete_final_review(
            db=db_session,
            meeting=meeting,
            decisions_made=True,
            decisions_clear=False,
            action_items_needed=True,
            action_items=["Implement Redis cache"],
            actor_agent_id=test_agent.id,
        )
        second = await outcome.complete_final_review(
            db=db_session,
            meeting=meeting,
            decisions_made=False,
            decisions_clear=True,
            action_items_needed=False,
            action_items=[],
            actor_user_id=test_user.id,
        )
        assert (first, second) == (True, False)
    elif case == "concurrent_completion":
        from huddleroom.workers import meeting_tasks

        await db_session.commit()
        verification_session = concurrent_sessions[0]

        async def complete(session):
            session_meeting = await session.get(Meeting, meeting.id)
            completed = await outcome.complete_final_review(
                db=session,
                meeting=session_meeting,
                decisions_made=True,
                decisions_clear=False,
                action_items_needed=True,
                action_items=["Implement Redis cache"],
                actor_agent_id=test_agent.id,
            )
            await session.commit()
            if completed:
                meeting_tasks.dispatch_finalize_meeting(str(meeting.id), meeting.project_id)
            return completed

        with patch("huddleroom.workers.meeting_tasks.dispatch_finalize_meeting") as dispatch:
            results = await asyncio.gather(*(complete(session) for session in concurrent_sessions))
        assert sorted(results) == [False, True]
        dispatch.assert_called_once_with(str(meeting.id), meeting.project_id)
    elif case == "concurrent_cancel":
        await db_session.execute(
            update(Meeting)
            .where(Meeting.id == meeting.id)
            .values(status="cancelled")
            .execution_options(synchronize_session=False)
        )
        assert meeting.status == "concluding"
        with pytest.raises(MeetingTransitionError, match="active final review"):
            await outcome.complete_final_review(
                db=db_session,
                meeting=meeting,
                decisions_made=True,
                decisions_clear=True,
                action_items_needed=True,
                action_items=["Implement Redis cache"],
                actor_user_id=test_user.id,
            )
    elif case in {"http_organizer", "http_member"}:
        call_order: list[str] = []
        db_session.commit = AsyncMock(side_effect=lambda: call_order.append("commit"))
        with patch("huddleroom.workers.meeting_tasks.dispatch_finalize_meeting") as dispatch:
            dispatch.side_effect = lambda *_args: call_order.append("dispatch")
            get_pending = await client.get(
                f"/api/v1/meetings/{meeting.id}/final-review", headers=auth_headers
            )
            assert get_pending.status_code == 200
            assert get_pending.json()["reviewer_kind"] == "orchestrator"

            body = {
                "decisions_made": True,
                "decisions_clear": False,
                "action_items_needed": False,
                "action_items": [],
            }
            first = await client.post(
                f"/api/v1/meetings/{meeting.id}/final-review",
                json=body,
                headers=auth_headers,
            )
            retry = await client.post(
                f"/api/v1/meetings/{meeting.id}/final-review",
                json=body,
                headers=auth_headers,
            )
            get_complete = await client.get(
                f"/api/v1/meetings/{meeting.id}/final-review", headers=auth_headers
            )

        assert first.status_code == retry.status_code == 204
        assert first.content == retry.content == b""
        assert get_complete.status_code == 200
        assert get_complete.json() is None
        assert call_order == ["commit", "dispatch", "commit"]
        dispatch.assert_called_once_with(str(meeting.id), meeting.project_id)
    elif case == "http_forbidden":
        db_session.commit = AsyncMock()
        with patch("huddleroom.workers.meeting_tasks.dispatch_finalize_meeting") as dispatch:
            response = await client.post(
                f"/api/v1/meetings/{meeting.id}/final-review",
                json={
                    "decisions_made": True,
                    "decisions_clear": True,
                    "action_items_needed": False,
                },
                headers=auth_headers,
            )
        assert response.status_code == 403
        db_session.commit.assert_not_awaited()
        dispatch.assert_not_called()
    else:
        db_session.commit = AsyncMock()
        meeting.status = "active"
        with patch("huddleroom.workers.meeting_tasks.dispatch_finalize_meeting") as dispatch:
            transition_error = await client.post(
                f"/api/v1/meetings/{meeting.id}/final-review",
                json={
                    "decisions_made": True,
                    "decisions_clear": True,
                    "action_items_needed": False,
                },
                headers=auth_headers,
            )
            meeting.status = "concluding"
            validation_error = await client.post(
                f"/api/v1/meetings/{meeting.id}/final-review",
                json={
                    "decisions_made": True,
                    "decisions_clear": True,
                    "action_items_needed": True,
                    "action_items": [],
                },
                headers=auth_headers,
            )
        assert transition_error.status_code == 400
        assert validation_error.status_code == 422
        db_session.commit.assert_not_awaited()
        dispatch.assert_not_called()

    items = (
        await verification_session.execute(
            select(MeetingActionItem).where(MeetingActionItem.meeting_id == meeting.id)
        )
    ).scalars().all()
    completions = (
        await verification_session.execute(
            select(MeetingEvent).where(
                MeetingEvent.meeting_id == meeting.id,
                MeetingEvent.event_type == "final_pass_completed",
            )
        )
    ).scalars().all()
    if case in {"serialized_retry", "concurrent_completion"}:
        assert len(items) == len(completions) == 1
        assert completions[0].payload == {
            "decisions_made": True,
            "decisions_clear": False,
            "action_items_needed": True,
        }
        assert completions[0].actor_agent_id == test_agent.id
        assert completions[0].actor_user_id is None
    elif case in {"http_organizer", "http_member"}:
        assert items == []
        assert len(completions) == 1
        assert completions[0].actor_user_id == test_user.id
    else:
        assert items == []
        assert completions == []


@pytest.mark.asyncio
async def test_finalize_meeting_persists_standup_tasks_and_summary_without_decisions(
    db_session, test_project, test_agent
):
    from sqlalchemy import select

    from huddleroom.models.knowledge_item import KnowledgeItem
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingActionItem, MeetingTurn
    from huddleroom.models.task import Task
    from huddleroom.services.meeting_outcome import MeetingOutcomeService

    meeting = Meeting(
        project_id=test_project.id,
        title="Daily standup",
        meeting_type="standup",
        participant_agent_ids=[str(test_agent.id)],
        status="concluding",
        concluding_started_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        veto_window_hours=0,
    )
    db_session.add(meeting)
    await db_session.flush()

    db_session.add(
        MeetingAgendaItem(
            meeting_id=meeting.id,
            order=1,
            title="Daily updates",
            status="resolved",
            resolution_kind="updates_shared",
            resolution_summary="Standup completed with 1 participant update(s).",
            required_followup="waiting on staging access; need production logs",
        )
    )
    db_session.add(
        MeetingTurn(
            meeting_id=meeting.id,
            turn_number=1,
            round_number=1,
            speaker_agent_id=test_agent.id,
            content="DONE: shipped auth fix\nNOW: rollout metrics\nBLOCKERS: waiting on staging access",
            references=[],
        )
    )
    await db_session.flush()

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "   "

    outcome = MeetingOutcomeService()
    with patch("huddleroom.services.meeting_outcome.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_resp
        await outcome.finalize_meeting(db=db_session, meeting=meeting)

    assert meeting.status == "concluded"
    assert meeting.summary
    assert "## Outcome" in meeting.summary

    action_items = (
        await db_session.execute(
            select(MeetingActionItem).where(MeetingActionItem.meeting_id == meeting.id)
        )
    ).scalars().all()
    assert [item.description for item in action_items] == [
        "waiting on staging access",
        "need production logs",
    ]

    tasks = (
        await db_session.execute(
            select(Task).where(Task.created_by_meeting_id == meeting.id).order_by(Task.title)
        )
    ).scalars().all()
    assert [task.title for task in tasks] == [
        "need production logs",
        "waiting on staging access",
    ]

    summary_ki = (
        await db_session.execute(
            select(KnowledgeItem).where(
                KnowledgeItem.project_id == test_project.id,
                KnowledgeItem.content_type == "summary",
                KnowledgeItem.title == f"Meeting Summary: {meeting.title}",
            )
        )
    ).scalar_one()
    assert summary_ki.content == meeting.summary
    assert summary_ki.content.strip()


@pytest.mark.asyncio
async def test_finalize_meeting_skips_planner_summary_for_standup(
    db_session, test_project, test_agent
):
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem
    from huddleroom.services.meeting_outcome import MeetingOutcomeService

    meeting = Meeting(
        project_id=test_project.id,
        title="Planner-free standup",
        meeting_type="standup",
        participant_agent_ids=[str(test_agent.id)],
        status="concluding",
        concluding_started_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        veto_window_hours=0,
        planner_agent_id=test_agent.id,
    )
    db_session.add(meeting)
    await db_session.flush()

    db_session.add(
        MeetingAgendaItem(
            meeting_id=meeting.id,
            order=1,
            title="Daily updates",
            status="resolved",
            resolution_kind="updates_shared",
            resolution_summary="Standup completed with 1 participant update(s).",
            required_followup="waiting on staging access",
        )
    )
    await db_session.flush()

    outcome = MeetingOutcomeService()
    with patch.object(outcome, "run_planner_summary", new_callable=AsyncMock) as mock_planner:
        await outcome.finalize_meeting(db=db_session, meeting=meeting)

    mock_planner.assert_not_awaited()
    assert meeting.status == "concluded"


@pytest.mark.asyncio
async def test_fetch_knowledge_items_returns_project_items(db_session, test_project):
    from huddleroom.models.knowledge_item import KnowledgeItem
    from huddleroom.models.meeting import Meeting
    from huddleroom.services.meeting_context import MeetingContextService

    meeting = Meeting(
        project_id=test_project.id,
        title="Knowledge fetch",
        meeting_type="decision",
        participant_agent_ids=[],
    )
    db_session.add(meeting)

    expected = KnowledgeItem(
        project_id=test_project.id,
        title="Design doc",
        content="Use async everywhere.",
        content_type="decision",
        provenance_type="human",
    )
    other_project = KnowledgeItem(
        project_id=None,
        title="Other",
        content="ignore me",
        content_type="summary",
        provenance_type="human",
    )
    superseded = KnowledgeItem(
        project_id=test_project.id,
        title="Old doc",
        content="obsolete",
        content_type="decision",
        provenance_type="human",
        is_superseded=True,
    )
    db_session.add_all([expected, other_project, superseded])
    await db_session.flush()

    items = await MeetingContextService()._fetch_knowledge_items(db=db_session, meeting=meeting)

    assert any(item.id == expected.id for item in items)
    assert all(item.project_id == test_project.id for item in items)
    assert all(item.is_superseded is False for item in items)


@pytest.mark.asyncio
async def test_list_meetings_pagination(client, auth_headers, db_session, test_project):
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem

    created_ids: list[str] = []
    for index in range(5):
        meeting = Meeting(
            project_id=test_project.id,
            title=f"Meeting {index}",
            meeting_type="decision",
            participant_agent_ids=[],
        )
        db_session.add(meeting)
        await db_session.flush()
        created_ids.append(str(meeting.id))
        db_session.add(
            MeetingAgendaItem(
                meeting_id=meeting.id,
                order=1,
                title=f"Agenda {index}",
                question="Decide?",
            )
        )
    await db_session.flush()

    response = await client.get(
        f"/api/v1/projects/{test_project.id}/meetings?limit=2",
        headers=auth_headers,
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["items"]) == 2
    assert all(entry["id"] in created_ids for entry in payload["items"])
    assert all(len(entry["agenda_items"]) == 1 for entry in payload["items"])


@pytest.mark.asyncio
async def test_consensus_decision_has_distinct_rationale(db_session, test_project, test_agent):
    from sqlalchemy import select

    from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingDecision, MeetingTurn
    from huddleroom.services.meeting_intelligence import MeetingIntelligenceService
    from huddleroom.services.meeting_runner import MeetingRunner

    meeting = Meeting(
        project_id=test_project.id,
        title="Consensus rationale test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        status="active",
    )
    db_session.add(meeting)
    await db_session.flush()

    item = MeetingAgendaItem(
        meeting_id=meeting.id,
        order=1,
        title="Cache layer",
        question="Should we use Redis?",
        status="active",
        max_rounds=3,
        current_round=0,
        consensus_check_count=0,
    )
    db_session.add(item)
    await db_session.flush()
    turn = MeetingTurn(
        meeting_id=meeting.id,
        agenda_item_id=item.id,
        turn_number=1,
        round_number=1,
        speaker_agent_id=test_agent.id,
        content="Redis is the best fit for latency.\nPOSITION: Use Redis",
        references=[],
        is_human_turn=False,
    )
    db_session.add(turn)
    await db_session.flush()

    runner = MeetingRunner()
    with patch.object(MeetingIntelligenceService, "check_consensus", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = {
            "consensus": True,
            "agreed_position": "Use Redis",
            "confidence": 0.95,
            "rationale": "All participants agreed Redis meets the latency requirement.",
        }
        with patch.object(runner._svc, "is_round_complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = True
            with patch.object(runner._svc, "advance_agenda", new_callable=AsyncMock):
                resolved = await runner.evaluate_round_if_complete(db=db_session, meeting=meeting, item=item)

    assert resolved is True

    decision = (
        await db_session.execute(
            select(MeetingDecision).where(MeetingDecision.meeting_id == meeting.id)
        )
    ).scalar_one()
    assert decision.chosen_option == "Use Redis"
    assert decision.rationale != decision.chosen_option
    assert "latency" in decision.rationale


@pytest.mark.asyncio
async def test_consensus_without_agreed_position_does_not_create_invalid_decision(
    db_session, test_project, test_agent
):
    from sqlalchemy import select

    from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingDecision, MeetingTurn
    from huddleroom.services.meeting_intelligence import MeetingIntelligenceService
    from huddleroom.services.meeting_runner import MeetingRunner

    meeting = Meeting(
        project_id=test_project.id,
        title="Consensus missing option test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        status="active",
    )
    db_session.add(meeting)
    await db_session.flush()

    item = MeetingAgendaItem(
        meeting_id=meeting.id,
        order=1,
        title="Greeting",
        question="Say hello?",
        status="active",
        max_rounds=1,
        current_round=0,
        consensus_check_count=0,
    )
    db_session.add(item)
    await db_session.flush()
    turn = MeetingTurn(
        meeting_id=meeting.id,
        agenda_item_id=item.id,
        turn_number=1,
        round_number=1,
        speaker_agent_id=test_agent.id,
        content="Hello there.\nPOSITION: Greeting",
        references=[],
        is_human_turn=False,
    )
    db_session.add(turn)
    await db_session.flush()

    runner = MeetingRunner()
    with patch.object(MeetingIntelligenceService, "check_consensus", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = {
            "consensus": True,
            "agreed_position": None,
            "confidence": 1.0,
            "rationale": "Everyone greeted each other.",
        }
        with patch.object(runner._svc, "is_round_complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = True
            with patch.object(runner._svc, "advance_agenda", new_callable=AsyncMock) as mock_advance:
                resolved = await runner.evaluate_round_if_complete(db=db_session, meeting=meeting, item=item)

    assert resolved is False
    mock_advance.assert_awaited_once()
    assert mock_advance.await_args.kwargs["resolution"] == "unresolved"
    assert mock_advance.await_args.kwargs["outcome"]["resolution_kind"] == "no_consensus"
    decisions = (
        await db_session.execute(
            select(MeetingDecision).where(MeetingDecision.meeting_id == meeting.id)
        )
    ).scalars().all()
    assert decisions == []


@pytest.mark.asyncio
async def test_evaluate_round_if_complete_applies_majority_rules_on_final_deadlocked_round(
    db_session, test_project, test_agent
):
    from sqlalchemy import select

    from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingDecision, MeetingTurn
    from huddleroom.services.meeting_intelligence import MeetingIntelligenceService
    from huddleroom.services.meeting_runner import MeetingRunner

    meeting = Meeting(
        project_id=test_project.id,
        title="Final-round deadlock test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        status="active",
        deadlock_strategy="majority_rules",
    )
    db_session.add(meeting)
    await db_session.flush()

    item = MeetingAgendaItem(
        meeting_id=meeting.id,
        order=1,
        title="Cache layer",
        question="Should we use Redis?",
        status="active",
        max_rounds=2,
        current_round=1,
        consensus_check_count=1,
    )
    db_session.add(item)
    await db_session.flush()

    db_session.add_all(
        [
            MeetingTurn(
                meeting_id=meeting.id,
                agenda_item_id=item.id,
                turn_number=1,
                round_number=1,
                speaker_agent_id=test_agent.id,
                content="Round 1: use Redis.\nPOSITION: Use Redis",
                references=[],
                is_human_turn=False,
            ),
            MeetingTurn(
                meeting_id=meeting.id,
                agenda_item_id=item.id,
                turn_number=2,
                round_number=2,
                speaker_agent_id=test_agent.id,
                content="Round 2: still use Redis.\nPOSITION: Use Redis",
                references=[],
                is_human_turn=False,
            ),
        ]
    )
    await db_session.flush()

    runner = MeetingRunner()
    with patch.object(MeetingIntelligenceService, "check_consensus", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = {
            "consensus": False,
            "agreed_position": None,
            "confidence": 0.4,
            "rationale": "Positions did not converge enough for consensus.",
        }
        with patch.object(runner._svc, "is_round_complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = True
            with patch.object(MeetingIntelligenceService, "extract_positions", new_callable=AsyncMock) as mock_positions:
                mock_positions.side_effect = [
                    [{"speaker": test_agent.name, "position": "Use Redis"}],
                    [{"speaker": test_agent.name, "position": "Use Redis"}],
                    [{"speaker": test_agent.name, "position": "Use Redis"}],
                ]
                with patch.object(runner._svc, "advance_agenda", new_callable=AsyncMock) as mock_advance:
                    resolved = await runner.evaluate_round_if_complete(db=db_session, meeting=meeting, item=item)

    assert resolved is False
    decision = (
        await db_session.execute(
            select(MeetingDecision).where(MeetingDecision.meeting_id == meeting.id)
        )
    ).scalar_one()
    assert decision.decided_by == "majority"
    assert decision.chosen_option == "Use Redis"
    mock_advance.assert_awaited_once()
    assert mock_advance.await_args.kwargs["resolution"] == "resolved"
    assert mock_advance.await_args.kwargs["outcome"]["resolution_kind"] == "majority"


@pytest.mark.asyncio
async def test_majority_rules_votes_on_positions_not_content(db_session, test_project, test_agent):
    from sqlalchemy import select

    from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingDecision, MeetingTurn
    from huddleroom.services.meeting_intelligence import MeetingIntelligenceService
    from huddleroom.services.meeting_runner import MeetingRunner

    meeting = Meeting(
        project_id=test_project.id,
        title="Majority rules test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        status="active",
        deadlock_strategy="majority_rules",
    )
    db_session.add(meeting)
    await db_session.flush()

    item = MeetingAgendaItem(
        meeting_id=meeting.id,
        order=1,
        title="Caching",
        question="What cache should we use?",
        status="active",
        current_round=1,
        max_rounds=3,
        is_deadlocked=True,
    )
    db_session.add(item)
    await db_session.flush()

    db_session.add_all(
        [
            MeetingTurn(
                meeting_id=meeting.id,
                agenda_item_id=item.id,
                turn_number=1,
                round_number=1,
                speaker_agent_id=test_agent.id,
                content="I think we should use Redis here.",
                references=[],
                is_human_turn=False,
            ),
            MeetingTurn(
                meeting_id=meeting.id,
                agenda_item_id=item.id,
                turn_number=2,
                round_number=1,
                speaker_agent_id=test_agent.id,
                content="Redis is the right tool for this workload.",
                references=[],
                is_human_turn=False,
            ),
        ]
    )
    await db_session.flush()

    runner = MeetingRunner()
    with patch.object(MeetingIntelligenceService, "extract_positions", new_callable=AsyncMock) as mock_positions:
        mock_positions.return_value = [
            {"speaker": test_agent.name, "position": "Use Redis"},
            {"speaker": test_agent.name, "position": "Use Redis"},
        ]
        with patch.object(runner._svc, "advance_agenda", new_callable=AsyncMock):
            await runner._apply_deadlock_strategy(db=db_session, meeting=meeting, item=item)

    decision = (
        await db_session.execute(
            select(MeetingDecision).where(MeetingDecision.meeting_id == meeting.id)
        )
    ).scalar_one()
    assert decision.chosen_option == "Use Redis"
    assert decision.decided_by == "majority"


@pytest.mark.asyncio
async def test_majority_rules_creates_fallback_decision_when_positions_missing(db_session, test_project, test_agent):
    from sqlalchemy import select

    from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingDecision, MeetingTurn
    from huddleroom.services.meeting_intelligence import MeetingIntelligenceService
    from huddleroom.services.meeting_runner import MeetingRunner

    meeting = Meeting(
        project_id=test_project.id,
        title="Majority fallback test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        status="active",
        deadlock_strategy="majority_rules",
    )
    db_session.add(meeting)
    await db_session.flush()

    item = MeetingAgendaItem(
        meeting_id=meeting.id,
        order=1,
        title="Caching",
        question="What cache should we use?",
        status="active",
        current_round=1,
        max_rounds=3,
        is_deadlocked=True,
    )
    db_session.add(item)
    await db_session.flush()

    db_session.add(
        MeetingTurn(
            meeting_id=meeting.id,
            agenda_item_id=item.id,
            turn_number=1,
            round_number=1,
            speaker_agent_id=test_agent.id,
            content="I'm unconvinced by the current options.",
            references=[],
            is_human_turn=False,
        )
    )
    await db_session.flush()

    runner = MeetingRunner()
    with patch.object(MeetingIntelligenceService, "extract_positions", new_callable=AsyncMock) as mock_positions:
        mock_positions.return_value = []
        with patch.object(runner._svc, "advance_agenda", new_callable=AsyncMock):
            await runner._apply_deadlock_strategy(db=db_session, meeting=meeting, item=item)

    decision = (
        await db_session.execute(
            select(MeetingDecision).where(MeetingDecision.meeting_id == meeting.id)
        )
    ).scalar_one()
    assert decision.decided_by == "majority"
    assert decision.chosen_option
    assert "fallback" in decision.rationale.lower()


@pytest.mark.asyncio
async def test_no_hasattr_guards_consensus_check_count(db_session, test_project, test_agent):
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingTurn
    from huddleroom.services.meeting_intelligence import MeetingIntelligenceService
    from huddleroom.services.meeting_runner import MeetingRunner

    meeting = Meeting(
        project_id=test_project.id,
        title="Consensus counter test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        status="active",
    )
    db_session.add(meeting)
    await db_session.flush()

    item = MeetingAgendaItem(
        meeting_id=meeting.id,
        order=1,
        title="Storage",
        question="Should we use object storage?",
        status="active",
        current_round=0,
        max_rounds=3,
        consensus_check_count=0,
    )
    db_session.add(item)
    await db_session.flush()
    turn = MeetingTurn(
        meeting_id=meeting.id,
        agenda_item_id=item.id,
        turn_number=1,
        round_number=1,
        speaker_agent_id=test_agent.id,
        content="Object storage is fine.\nPOSITION: Use object storage",
        references=[],
        is_human_turn=False,
    )
    db_session.add(turn)
    await db_session.flush()

    runner = MeetingRunner()
    with patch.object(MeetingIntelligenceService, "check_consensus", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = {
            "consensus": False,
            "agreed_position": None,
            "confidence": 0.0,
            "rationale": "",
        }
        with patch.object(runner._svc, "is_round_complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = True
            await runner.evaluate_round_if_complete(db=db_session, meeting=meeting, item=item)

    assert item.consensus_check_count == 1
