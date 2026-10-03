import asyncio
import pytest
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch, MagicMock


@contextmanager
def patch_streaming_acompletion(
    service_module, content, prompt_tokens=0, completion_tokens=0, finish_reason="stop"
):
    """Patch a service's litellm.acompletion to stream `content`, and patch the
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
                    finish_reason=finish_reason,
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            ),
        )

    with patch(f"{service_module}.litellm.acompletion", new=mock_acompletion), patch(
        "huddleroom.services.agent_response_stream.litellm.stream_chunk_builder", new=mock_builder
    ):
        yield


@pytest.mark.asyncio
async def test_context_service_builds_initial_package(db_session, test_project, test_agent):
    from huddleroom.services.meeting_service import MeetingService
    from huddleroom.services.meeting_context import MeetingContextService

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Arch Decision",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "REST vs GraphQL", "question": "Which?", "max_rounds": 2}],
    )
    await svc.transition_to_preparing(db=db_session, meeting=meeting)

    ctx_svc = MeetingContextService()
    with patch.object(ctx_svc, "_fetch_knowledge_for_agenda", return_value=[]):
        pkg = await ctx_svc.build_initial_context(db=db_session, meeting=meeting, agent=test_agent)

    assert test_agent.name in pkg
    assert "REST vs GraphQL" in pkg
    assert "round_robin" in pkg.lower() or "turn" in pkg.lower()


@pytest.mark.asyncio
async def test_context_service_appends_turn(db_session, test_project, test_agent):
    from huddleroom.services.meeting_service import MeetingService
    from huddleroom.services.meeting_context import MeetingContextService
    from huddleroom.models.meeting import MeetingTurn

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Turn Append Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "Item 1", "max_rounds": 2}],
    )

    turn = MeetingTurn(
        meeting_id=meeting.id,
        turn_number=1,
        round_number=1,
        speaker_agent_id=test_agent.id,
        content="I support REST because it is simpler.",
        references=[],
    )
    db_session.add(turn)
    await db_session.flush()

    ctx_svc = MeetingContextService()
    transcript = await ctx_svc.format_transcript(db=db_session, meeting_id=meeting.id)
    assert "I support REST" in transcript
    assert test_agent.name in transcript


@pytest.mark.asyncio
async def test_context_service_builds_turn_prompt(db_session, test_project, test_agent):
    from huddleroom.services.meeting_service import MeetingService
    from huddleroom.services.meeting_context import MeetingContextService

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Turn Prompt Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "REST vs GraphQL", "question": "Which?", "max_rounds": 2}],
    )
    await svc.transition_to_preparing(db=db_session, meeting=meeting)
    item = await svc.get_current_agenda_item(db=db_session, meeting_id=meeting.id)

    ctx_svc = MeetingContextService()
    with patch.object(ctx_svc, "_fetch_knowledge_for_agenda", return_value=[]):
        messages = await ctx_svc.build_turn_prompt(
            db=db_session, meeting=meeting, agent=test_agent,
            current_item=item, turn_number=1,
        )

    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "REST vs GraphQL" in messages[1]["content"]
    assert "Round 1" in messages[1]["content"]


@pytest.mark.asyncio
async def test_consensus_detected(db_session, test_project, test_agent):
    from huddleroom.services.meeting_intelligence import MeetingIntelligenceService

    intel = MeetingIntelligenceService()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = (
        '{"consensus": true, "agreed_position": "Use REST", "confidence": 0.92}'
    )

    with patch("huddleroom.services.meeting_intelligence.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_response
        result = await intel.check_consensus(
            item_title="REST vs GraphQL",
            item_question="Which API style?",
            item_options=["Use REST", "Use GraphQL"],
            expected_speakers=["Alice", "Bob"],
            turns_this_round=[
                {"speaker": "Alice", "content": "I support REST for simplicity."},
                {"speaker": "Bob", "content": "Agreed, REST is cleaner."},
            ],
            prior_rounds_summary=None,
        )

    assert result["consensus"] is True
    assert result["confidence"] >= 0.8
    assert "REST" in result["agreed_position"]


@pytest.mark.asyncio
async def test_no_consensus(db_session):
    from huddleroom.services.meeting_intelligence import MeetingIntelligenceService

    intel = MeetingIntelligenceService()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = (
        '{"consensus": false, "agreed_position": null, "confidence": 0.3}'
    )

    with patch("huddleroom.services.meeting_intelligence.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_response
        result = await intel.check_consensus(
            item_title="REST vs GraphQL",
            item_question="Which?",
            item_options=["Use REST", "Use GraphQL"],
            expected_speakers=["Alice", "Bob"],
            turns_this_round=[
                {"speaker": "Alice", "content": "REST is better."},
                {"speaker": "Bob", "content": "GraphQL is better."},
            ],
            prior_rounds_summary=None,
        )

    assert result["consensus"] is False


@pytest.mark.asyncio
async def test_runner_executes_single_turn(db_session, test_project, test_agent):
    from huddleroom.services.meeting_service import MeetingService
    from huddleroom.services.meeting_runner import MeetingRunner
    from huddleroom.models.meeting import MeetingTurn

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Runner Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "REST vs GraphQL", "max_rounds": 1}],
    )
    await svc.transition_to_preparing(db=db_session, meeting=meeting)
    await svc.transition_to_active(db=db_session, meeting=meeting)

    runner = MeetingRunner(bus=object())
    with patch_streaming_acompletion(
        "huddleroom.services.meeting_runner",
        "POSITION: Use REST\nRATIONALE: It is simpler.",
        prompt_tokens=100,
        completion_tokens=5,
    ):
        with patch("huddleroom.services.meeting_context.MeetingContextService._fetch_knowledge_for_agenda", return_value=[]):
            turn = await runner.execute_agent_turn(
                db=db_session, meeting=meeting, agent=test_agent,
            )

    assert turn is not None
    assert turn.speaker_agent_id == test_agent.id
    assert "REST" in turn.content
    assert turn.turn_number == 1
    assert turn.moderator_note is None


@pytest.mark.asyncio
async def test_execute_agent_turn_passes_provider_extras_to_litellm(db_session, test_project):
    from huddleroom.models.agent import Agent
    from huddleroom.services.meeting_runner import MeetingRunner
    from huddleroom.services.meeting_service import MeetingService

    agent = Agent(
        name=f"openrouter-{uuid.uuid4()}",
        role="architect",
        provider="openrouter",
        model="nvidia/nemotron-3-super-120b-a12b:free",
        adapter_type="api",
        capabilities=[],
        config={"provider_extras": {"api_key": "test-openrouter-key"}},
    )
    db_session.add(agent)
    await db_session.flush()

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="OpenRouter Provider Extras Test",
        meeting_type="decision",
        participant_agent_ids=[str(agent.id)],
        agenda_items=[{"order": 1, "title": "REST vs GraphQL", "max_rounds": 1}],
    )
    await svc.transition_to_preparing(db=db_session, meeting=meeting)
    await svc.transition_to_active(db=db_session, meeting=meeting)

    simple_resp = MagicMock()
    simple_resp.choices = [MagicMock()]
    simple_resp.choices[0].message.content = "POSITION: Use REST\nRATIONALE: It is simpler."
    simple_resp.usage = MagicMock(completion_tokens=5, prompt_tokens=100)

    runner = MeetingRunner()
    with patch("huddleroom.services.meeting_runner.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = simple_resp
        with patch("huddleroom.services.meeting_context.MeetingContextService._fetch_knowledge_for_agenda", return_value=[]):
            await runner.execute_agent_turn(db=db_session, meeting=meeting, agent=agent)

    assert mock_llm.await_args.kwargs["api_key"] == "test-openrouter-key"
    assert mock_llm.await_args.kwargs["model"] == "openrouter/nvidia/nemotron-3-super-120b-a12b:free"


@pytest.mark.asyncio
async def test_runner_parses_references(db_session, test_project, test_agent):
    from huddleroom.services.meeting_runner import MeetingRunner
    from huddleroom.services.meeting_service import MeetingService

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Ref Parse Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "Decision", "max_rounds": 1}],
    )
    await svc.transition_to_preparing(db=db_session, meeting=meeting)
    await svc.transition_to_active(db=db_session, meeting=meeting)

    ki_id = uuid.uuid4()
    raw_content = f"POSITION: Use REST\nSee [REF:knowledge:{ki_id}] for details."

    runner = MeetingRunner(bus=object())
    with patch_streaming_acompletion(
        "huddleroom.services.meeting_runner", raw_content, prompt_tokens=50, completion_tokens=10
    ):
        with patch("huddleroom.services.meeting_context.MeetingContextService._fetch_knowledge_for_agenda", return_value=[]):
            turn = await runner.execute_agent_turn(
                db=db_session, meeting=meeting, agent=test_agent,
            )

    assert "[REF:" not in turn.content
    assert len(turn.references) == 1
    assert turn.references[0]["id"] == str(ki_id)
    assert turn.references[0]["type"] == "knowledge"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_content", "expected_error"),
    [
        ("", "empty_response"),
        ("I support REST.", "missing_position"),
    ],
)
async def test_invalid_decision_turn_validation(
    db_session, test_project, test_agent, raw_content, expected_error
):
    from huddleroom.services.meeting_runner import MeetingRunner
    from huddleroom.services.meeting_service import MeetingService

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Invalid Turn Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "Decision", "max_rounds": 1}],
    )
    await svc.transition_to_preparing(db=db_session, meeting=meeting)
    await svc.transition_to_active(db=db_session, meeting=meeting)
    item = await svc.get_current_agenda_item(db=db_session, meeting_id=meeting.id)

    runner = MeetingRunner(bus=object())
    with patch_streaming_acompletion(
        "huddleroom.services.meeting_runner", raw_content, prompt_tokens=50, completion_tokens=0
    ):
        with patch("huddleroom.services.event_bus.emit_event", new_callable=AsyncMock) as mock_emit:
            with patch("huddleroom.services.meeting_context.MeetingContextService._fetch_knowledge_for_agenda", return_value=[]):
                turn = await runner.execute_agent_turn(
                    db=db_session, meeting=meeting, agent=test_agent,
                )

    # Test persistence and turn rejection
    assert turn.content == raw_content
    assert turn.moderator_note == f"validation_error:{expected_error}"

    with patch(
        "huddleroom.services.meeting_runner.MeetingIntelligenceService.check_consensus",
        new_callable=AsyncMock,
    ) as mock_consensus:
        resolved = await runner.evaluate_round_if_complete(db=db_session, meeting=meeting, item=item)

    assert resolved is False
    assert item.current_round == 0
    mock_consensus.assert_not_awaited()

    # Test event emission with validation metadata
    payload = mock_emit.await_args.kwargs["payload"]
    assert payload["is_valid"] is False
    assert payload["validation_error"] == expected_error


@pytest.mark.asyncio
async def test_is_round_complete_ignores_invalid_decision_turns(db_session, test_project, test_agent):
    from huddleroom.models.meeting import MeetingTurn
    from huddleroom.services.meeting_service import MeetingService

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Round Count Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "Decision", "max_rounds": 1}],
    )
    item = await svc.get_current_agenda_item(db=db_session, meeting_id=meeting.id)

    db_session.add(
        MeetingTurn(
            meeting_id=meeting.id,
            agenda_item_id=item.id,
            turn_number=1,
            round_number=1,
            speaker_agent_id=test_agent.id,
            content="",
            references=[],
            moderator_note="validation_error:empty_response",
        )
    )
    await db_session.flush()

    assert await svc.is_round_complete(db_session, meeting.id, item.id, 1) is False


@pytest.mark.asyncio
async def test_agenda_driven_invalid_turn_does_not_advance_turn_order(db_session, test_project, test_agent):
    from huddleroom.models.agent import Agent
    from huddleroom.models.meeting import MeetingTurn
    from huddleroom.services.meeting_runner import MeetingRunner
    from huddleroom.services.meeting_service import MeetingService

    second_agent = Agent(
        name=f"test-agent-{uuid.uuid4()}",
        role="architect",
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
        title="Agenda Driven Invalid Turn Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id), str(second_agent.id)],
        turn_strategy="agenda_driven",
        agenda_items=[
            {
                "order": 1,
                "title": "Decision",
                "max_rounds": 1,
                "turn_order": [str(test_agent.id), str(second_agent.id)],
            }
        ],
    )
    item = await svc.get_current_agenda_item(db=db_session, meeting_id=meeting.id)

    db_session.add(
        MeetingTurn(
            meeting_id=meeting.id,
            agenda_item_id=item.id,
            turn_number=1,
            round_number=1,
            speaker_agent_id=test_agent.id,
            content="I support REST.",
            references=[],
            moderator_note="validation_error:missing_position",
        )
    )
    await db_session.flush()

    runner = MeetingRunner()
    next_speaker_id, _selection = await runner._determine_next_speaker(db_session, meeting, item)

    assert next_speaker_id == str(test_agent.id)


@pytest.mark.asyncio
async def test_moderated_review_fallback_prefers_security_first_when_moderator_output_is_invalid(
    db_session, test_project, test_agent
):
    from huddleroom.models.agent import Agent
    from huddleroom.services.meeting_runner import MeetingRunner
    from huddleroom.services.meeting_service import MeetingService

    security_agent = Agent(
        name=f"security-{uuid.uuid4()}",
        role="security",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=[],
        config={},
    )
    pm_agent = Agent(
        name=f"pm-{uuid.uuid4()}",
        role="pm",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=[],
        config={},
    )
    db_session.add_all([security_agent, pm_agent])
    await db_session.flush()

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Security Review",
        meeting_type="review",
        turn_strategy="moderated",
        participant_agent_ids=[str(test_agent.id), str(security_agent.id), str(pm_agent.id)],
        agenda_items=[{"order": 1, "title": "Auth mechanism review", "max_rounds": 1}],
    )
    item = await svc.get_current_agenda_item(db_session, meeting.id)

    runner = MeetingRunner()
    with patch(
        "huddleroom.services.meeting_runner.MeetingIntelligenceService.select_next_speaker",
        new_callable=AsyncMock,
    ) as mock_select:
        mock_select.return_value = {"next_speaker_id": "not-a-participant", "close_item": False}
        next_speaker_id, _selection = await runner._determine_next_speaker(db_session, meeting, item)

    assert next_speaker_id == str(security_agent.id)


@pytest.mark.asyncio
async def test_moderated_review_prefers_security_first_even_when_moderator_picks_architect(
    db_session, test_project, test_agent
):
    from huddleroom.models.agent import Agent
    from huddleroom.services.meeting_runner import MeetingRunner
    from huddleroom.services.meeting_service import MeetingService

    security_agent = Agent(
        name=f"security-{uuid.uuid4()}",
        role="security",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=[],
        config={},
    )
    db_session.add(security_agent)
    await db_session.flush()

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Security Review",
        meeting_type="review",
        turn_strategy="moderated",
        participant_agent_ids=[str(test_agent.id), str(security_agent.id)],
        agenda_items=[{"order": 1, "title": "Auth mechanism review", "max_rounds": 1}],
    )
    item = await svc.get_current_agenda_item(db_session, meeting.id)

    runner = MeetingRunner()
    with patch(
        "huddleroom.services.meeting_runner.MeetingIntelligenceService.select_next_speaker",
        new_callable=AsyncMock,
    ) as mock_select:
        mock_select.return_value = {"next_speaker_id": str(test_agent.id), "close_item": False}
        next_speaker_id, _selection = await runner._determine_next_speaker(db_session, meeting, item)

    assert next_speaker_id == str(security_agent.id)


@pytest.mark.asyncio
async def test_moderated_review_does_not_repeat_speaker_while_others_have_not_spoken(
    db_session, test_project, test_agent
):
    from huddleroom.models.agent import Agent
    from huddleroom.models.meeting import MeetingTurn
    from huddleroom.services.meeting_runner import MeetingRunner
    from huddleroom.services.meeting_service import MeetingService

    security_agent = Agent(
        name=f"security-{uuid.uuid4()}",
        role="security",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=[],
        config={},
    )
    pm_agent = Agent(
        name=f"pm-{uuid.uuid4()}",
        role="pm",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=[],
        config={},
    )
    db_session.add_all([security_agent, pm_agent])
    await db_session.flush()

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Security Review",
        meeting_type="review",
        turn_strategy="moderated",
        participant_agent_ids=[str(security_agent.id), str(test_agent.id), str(pm_agent.id)],
        agenda_items=[{"order": 1, "title": "Auth mechanism review", "max_rounds": 1}],
    )
    item = await svc.get_current_agenda_item(db_session, meeting.id)

    db_session.add(
        MeetingTurn(
            meeting_id=meeting.id,
            agenda_item_id=item.id,
            turn_number=1,
            round_number=1,
            speaker_agent_id=security_agent.id,
            content="- [severity: blocker] auth: replace Basic Auth",
            references=[],
        )
    )
    await db_session.flush()

    runner = MeetingRunner()
    with patch(
        "huddleroom.services.meeting_runner.MeetingIntelligenceService.select_next_speaker",
        new_callable=AsyncMock,
    ) as mock_select:
        mock_select.return_value = {"next_speaker_id": str(security_agent.id), "close_item": False}
        next_speaker_id, _selection = await runner._determine_next_speaker(db_session, meeting, item)

    assert next_speaker_id in {str(test_agent.id), str(pm_agent.id)}


@pytest.mark.asyncio
async def test_review_round_complete_requires_non_reviewer_response_before_closure(
    db_session, test_project, test_agent
):
    from huddleroom.models.agent import Agent
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingTurn
    from huddleroom.services.meeting_runner import MeetingRunner

    reviewer = Agent(
        name=f"security-{uuid.uuid4()}",
        role="security",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=[],
        config={},
    )
    db_session.add(reviewer)
    await db_session.flush()

    meeting = Meeting(
        project_id=test_project.id,
        title="Security review",
        meeting_type="review",
        participant_agent_ids=[str(reviewer.id), str(test_agent.id)],
        status="active",
    )
    db_session.add(meeting)
    await db_session.flush()

    item = MeetingAgendaItem(
        meeting_id=meeting.id,
        order=1,
        title="Auth review",
        status="active",
        max_rounds=1,
    )
    db_session.add(item)
    await db_session.flush()

    db_session.add(
        MeetingTurn(
            meeting_id=meeting.id,
            agenda_item_id=item.id,
            turn_number=1,
            round_number=1,
            speaker_agent_id=reviewer.id,
            content="- [severity: blocker] auth: Basic Auth remains enabled -> replace it",
            references=[],
        )
    )
    await db_session.flush()

    runner = MeetingRunner()
    with patch.object(runner._svc, "is_round_complete", new_callable=AsyncMock) as mock_complete:
        mock_complete.return_value = True
        with patch.object(runner._svc, "advance_agenda", new_callable=AsyncMock) as mock_advance:
            resolved = await runner.evaluate_round_if_complete(db=db_session, meeting=meeting, item=item)

    assert resolved is False
    mock_advance.assert_not_awaited()


@pytest.mark.asyncio
async def test_review_round_complete_records_structured_rejection(
    db_session, test_project, test_agent
):
    from huddleroom.models.agent import Agent
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingTurn
    from huddleroom.services.meeting_runner import MeetingRunner

    reviewer = Agent(
        name=f"security-{uuid.uuid4()}",
        role="security",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=[],
        config={},
    )
    db_session.add(reviewer)
    await db_session.flush()

    meeting = Meeting(
        project_id=test_project.id,
        title="Security review",
        meeting_type="review",
        participant_agent_ids=[str(reviewer.id), str(test_agent.id)],
        status="active",
    )
    db_session.add(meeting)
    await db_session.flush()

    item = MeetingAgendaItem(
        meeting_id=meeting.id,
        order=1,
        title="Auth review",
        status="active",
        max_rounds=1,
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
                speaker_agent_id=reviewer.id,
                content="- [severity: blocker] auth: Basic Auth remains enabled -> replace it",
                references=[],
            ),
            MeetingTurn(
                meeting_id=meeting.id,
                agenda_item_id=item.id,
                turn_number=2,
                round_number=1,
                speaker_agent_id=test_agent.id,
                content="- [severity: major] auth: I agree with security and will replace it with token auth",
                references=[],
            ),
        ]
    )
    await db_session.flush()

    runner = MeetingRunner()
    with patch.object(runner._svc, "is_round_complete", new_callable=AsyncMock) as mock_complete:
        mock_complete.return_value = True
        resolved = await runner.evaluate_round_if_complete(db=db_session, meeting=meeting, item=item)

    assert resolved is True
    assert item.status == "unresolved"
    assert item.resolution_kind == "rejected"
    assert "basic auth" in (item.resolution_summary or "").lower()
    assert set(item.participants_heard or []) == {str(reviewer.id), str(test_agent.id)}


@pytest.mark.asyncio
async def test_standup_round_complete_advances_without_consensus_check(
    db_session, test_project, test_agent
):
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingTurn
    from huddleroom.services.meeting_intelligence import MeetingIntelligenceService
    from huddleroom.services.meeting_runner import MeetingRunner

    meeting = Meeting(
        project_id=test_project.id,
        title="Daily standup",
        meeting_type="standup",
        participant_agent_ids=[str(test_agent.id)],
        status="active",
    )
    db_session.add(meeting)
    await db_session.flush()

    item = MeetingAgendaItem(
        meeting_id=meeting.id,
        order=1,
        title="Daily updates",
        status="active",
        max_rounds=1,
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
            content="DONE: shipped auth fix\nNOW: wiring rollout metrics\nBLOCKERS: waiting on staging access",
            references=[],
        )
    )
    await db_session.flush()

    runner = MeetingRunner()
    with patch.object(runner._svc, "is_round_complete", new_callable=AsyncMock) as mock_complete:
        mock_complete.return_value = True
        with patch.object(MeetingIntelligenceService, "check_consensus", new_callable=AsyncMock) as mock_consensus:
            resolved = await runner.evaluate_round_if_complete(db=db_session, meeting=meeting, item=item)

    assert resolved is True
    mock_consensus.assert_not_awaited()
    assert item.status == "resolved"
    assert item.resolution_kind == "updates_shared"
    assert "staging access" in (item.required_followup or "").lower()


@pytest.mark.asyncio
async def test_deadlock_detection_ignores_invalid_previous_round_turns(db_session, test_project, test_agent):
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingTurn
    from huddleroom.services.meeting_intelligence import MeetingIntelligenceService
    from huddleroom.services.meeting_runner import MeetingRunner

    meeting = Meeting(
        project_id=test_project.id,
        title="Deadlock Detection Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        status="active",
    )
    db_session.add(meeting)
    await db_session.flush()

    item = MeetingAgendaItem(
        meeting_id=meeting.id,
        order=1,
        title="Decision",
        status="active",
        current_round=1,
        max_rounds=2,
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
                content="POSITION: Use REST",
                references=[],
                is_human_turn=False,
            ),
            MeetingTurn(
                meeting_id=meeting.id,
                agenda_item_id=item.id,
                turn_number=2,
                round_number=1,
                speaker_agent_id=test_agent.id,
                content="I support REST.",
                references=[],
                is_human_turn=False,
                moderator_note="validation_error:missing_position",
            ),
        ]
    )
    await db_session.flush()

    runner = MeetingRunner()

    async def fake_extract_positions(turns_data, **_kwargs):
        assert turns_data == [{"speaker": test_agent.name, "content": "POSITION: Use REST"}]
        return [{"option": "Use REST"}]

    with patch.object(
        MeetingIntelligenceService,
        "extract_positions",
        new=AsyncMock(side_effect=fake_extract_positions),
    ):
        is_deadlocked = await runner._is_deadlocked(
            db=db_session,
            meeting=meeting,
            item=item,
            current_positions=[{"option": "Use REST"}],
            completed_round_number=2,
        )

    assert is_deadlocked is True


@pytest.mark.asyncio
async def test_majority_rules_uses_current_round_valid_turns(db_session, test_project, test_agent):
    from sqlalchemy import select

    from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingDecision, MeetingTurn
    from huddleroom.services.meeting_intelligence import MeetingIntelligenceService
    from huddleroom.services.meeting_runner import MeetingRunner

    meeting = Meeting(
        project_id=test_project.id,
        title="Majority Rules Test",
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
        title="Decision",
        question="Which option?",
        status="active",
        current_round=2,
        max_rounds=2,
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
                content="POSITION: Use REST",
                references=[],
                is_human_turn=False,
            ),
            MeetingTurn(
                meeting_id=meeting.id,
                agenda_item_id=item.id,
                turn_number=2,
                round_number=2,
                speaker_agent_id=test_agent.id,
                content="POSITION: Use GraphQL",
                references=[],
                is_human_turn=False,
            ),
            MeetingTurn(
                meeting_id=meeting.id,
                agenda_item_id=item.id,
                turn_number=3,
                round_number=2,
                speaker_agent_id=test_agent.id,
                content="GraphQL is better.",
                references=[],
                is_human_turn=False,
                moderator_note="validation_error:missing_position",
            ),
        ]
    )
    await db_session.flush()

    runner = MeetingRunner()

    async def fake_extract_positions(turns_data, **_kwargs):
        assert turns_data == [{"speaker": "Agent", "content": "POSITION: Use GraphQL"}]
        return [{"option": "Use GraphQL"}]

    with patch.object(
        MeetingIntelligenceService,
        "extract_positions",
        new=AsyncMock(side_effect=fake_extract_positions),
    ):
        with patch.object(runner._svc, "advance_agenda", new_callable=AsyncMock) as mock_advance:
            await runner._apply_deadlock_strategy(db=db_session, meeting=meeting, item=item)

    decision = (
        await db_session.execute(
            select(MeetingDecision).where(MeetingDecision.meeting_id == meeting.id)
        )
    ).scalar_one()
    assert decision.chosen_option == "Use GraphQL"
    mock_advance.assert_awaited_once()
    assert mock_advance.await_args.kwargs["resolution"] == "resolved"
    assert mock_advance.await_args.kwargs["outcome"]["resolution_kind"] == "majority"


@pytest.mark.asyncio
async def test_human_intervention_deadlock_advances_item_as_unresolved(db_session, test_project, test_agent):
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem
    from huddleroom.services.meeting_runner import MeetingRunner

    meeting = Meeting(
        project_id=test_project.id,
        title="Human intervention deadlock test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        status="active",
        deadlock_strategy="human_intervention",
    )
    db_session.add(meeting)
    await db_session.flush()

    item = MeetingAgendaItem(
        meeting_id=meeting.id,
        order=1,
        title="Choose stack",
        question="Which stack should we use?",
        status="active",
        current_round=2,
        max_rounds=2,
        is_deadlocked=True,
    )
    db_session.add(item)
    await db_session.flush()

    runner = MeetingRunner(bus=object())
    with patch("huddleroom.services.event_bus.emit_event", new_callable=AsyncMock) as mock_emit:
        with patch.object(runner._svc, "advance_agenda", new_callable=AsyncMock) as mock_advance:
            await runner._apply_deadlock_strategy(db=db_session, meeting=meeting, item=item)

    mock_emit.assert_awaited_once()
    mock_advance.assert_awaited_once()
    assert mock_advance.await_args.kwargs["resolution"] == "unresolved"
    assert (
        mock_advance.await_args.kwargs["outcome"]["resolution_kind"]
        == "human_intervention"
    )


@pytest.mark.asyncio
async def test_organizer_agent_select_falls_back_when_llm_times_out(
    db_session, test_project, test_agent
):
    from huddleroom.models.agent import Agent
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem
    from huddleroom.services.meeting_runner import MeetingRunner

    organizer = Agent(
        name=f"pm-{uuid.uuid4()}",
        role="pm",
        provider="ollama",
        model="gemma4:26b",
        adapter_type="api",
        capabilities=[],
        config={},
    )
    second = Agent(
        name=f"architect-{uuid.uuid4()}",
        role="architect",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=[],
        config={},
    )
    db_session.add_all([organizer, second])
    await db_session.flush()

    meeting = Meeting(
        project_id=test_project.id,
        title="Organizer timeout fallback",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id), str(second.id), str(organizer.id)],
        organizer_agent_id=organizer.id,
        status="active",
        turn_strategy="organizer_controlled",
    )
    db_session.add(meeting)
    await db_session.flush()

    item = MeetingAgendaItem(
        meeting_id=meeting.id,
        order=1,
        title="Pick direction",
        status="active",
        max_rounds=2,
    )
    db_session.add(item)
    await db_session.flush()

    runner = MeetingRunner()
    with patch.object(runner._ctx, "format_transcript", new_callable=AsyncMock, return_value="") as _mock_transcript:
        with patch.object(runner._svc, "get_pending_signals", new_callable=AsyncMock, return_value=[]) as _mock_signals:
            with patch(
                "huddleroom.services.meeting_runner.litellm.acompletion",
                new=AsyncMock(side_effect=TimeoutError("llm timeout")),
            ):
                selected, selection = await runner._determine_next_speaker(db_session, meeting, item)

    assert selected == str(test_agent.id)
    assert selection is None


@pytest.mark.asyncio
async def test_probe_for_signals_emits_probe_and_signal_events(db_session, test_project, test_agent):
    from huddleroom.models.agent import Agent
    from huddleroom.models.meeting import Meeting, MeetingTurn
    from huddleroom.services.meeting_runner import MeetingRunner

    listener = Agent(
        name=f"listener-{uuid.uuid4()}",
        role="pm",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=[],
        config={},
    )
    db_session.add(listener)
    await db_session.flush()

    meeting = Meeting(
        project_id=test_project.id,
        title="Signal probing",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id), str(listener.id)],
        status="active",
        signal_check_enabled=True,
    )
    db_session.add(meeting)
    await db_session.flush()

    turn = MeetingTurn(
        meeting_id=meeting.id,
        turn_number=1,
        round_number=1,
        speaker_agent_id=test_agent.id,
        content="POSITION: Event-driven architecture",
        references=[],
    )
    db_session.add(turn)
    await db_session.flush()

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="YES: I want to challenge the migration cost."))]
    )

    runner = MeetingRunner(bus=object())
    with patch("huddleroom.services.meeting_runner.litellm.acompletion", new=AsyncMock(return_value=response)):
        with patch("huddleroom.services.event_bus.emit_event", new_callable=AsyncMock) as mock_emit:
            await runner._probe_for_signals(
                db=db_session,
                meeting=meeting,
                speaking_agent_id=test_agent.id,
                turn=turn,
            )

    emitted_types = [call.kwargs["event_type"] for call in mock_emit.await_args_list]
    assert emitted_types == ["meeting.signal_probe", "meeting.signal"]
    probe_payload = mock_emit.await_args_list[0].kwargs["payload"]
    signal_payload = mock_emit.await_args_list[1].kwargs["payload"]
    assert probe_payload["probe_response"] == "YES: I want to challenge the migration cost."
    assert probe_payload["signaled"] is True
    assert signal_payload["signal_type"] == "want_to_speak"
    assert signal_payload["message"] == "I want to challenge the migration cost."


@pytest.mark.asyncio
async def test_execute_agent_turn_marks_invalid_option_position(db_session, test_project, test_agent):
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem
    from huddleroom.services.meeting_runner import MeetingRunner

    meeting = Meeting(
        project_id=test_project.id,
        title="Invalid option position",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        status="active",
    )
    db_session.add(meeting)
    await db_session.flush()

    item = MeetingAgendaItem(
        meeting_id=meeting.id,
        order=1,
        title="Pick a focus",
        question="Which focus should we choose?",
        options=["Technical debt reduction", "New feature delivery"],
        status="active",
        max_rounds=2,
    )
    db_session.add(item)
    await db_session.flush()

    runner = MeetingRunner()
    with patch.object(runner._ctx, "build_turn_prompt", new_callable=AsyncMock, return_value=[]):
        with patch(
            "huddleroom.services.meeting_runner.litellm.acompletion",
            new=AsyncMock(
                return_value=SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="POSITION: Balanced split"), finish_reason="stop")],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                )
            ),
        ):
            turn = await runner.execute_agent_turn(db=db_session, meeting=meeting, agent=test_agent)

    assert turn.moderator_note == "validation_error:invalid_position_option"


@pytest.mark.asyncio
async def test_round_complete_triggers_consensus_check(db_session, test_project, test_agent):
    from huddleroom.services.meeting_service import MeetingService
    from huddleroom.services.meeting_runner import MeetingRunner
    from huddleroom.models.meeting import MeetingTurn

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Round Complete Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "Item 1", "max_rounds": 2}],
    )
    await svc.transition_to_preparing(db=db_session, meeting=meeting)
    await svc.transition_to_active(db=db_session, meeting=meeting)
    item = await svc.get_current_agenda_item(db=db_session, meeting_id=meeting.id)

    turn = MeetingTurn(
        meeting_id=meeting.id,
        agenda_item_id=item.id,
        turn_number=1,
        round_number=1,
        speaker_agent_id=test_agent.id,
        content="POSITION: Use REST\nRATIONALE: Simpler implementation.",
        references=[],
    )
    db_session.add(turn)
    await db_session.flush()

    consensus_result = {"consensus": True, "agreed_position": "REST", "confidence": 0.95}

    runner = MeetingRunner()
    with patch(
        "huddleroom.services.meeting_runner.MeetingIntelligenceService.check_consensus",
        new_callable=AsyncMock,
        return_value=consensus_result,
    ) as mock_consensus:
        resolved = await runner.evaluate_round_if_complete(db=db_session, meeting=meeting, item=item)

    assert resolved is True
    mock_consensus.assert_awaited_once()
    await db_session.flush()
    assert item.status == "resolved"


@pytest.mark.asyncio
async def test_start_meeting_task_transitions_to_preparing(db_session, test_project, test_agent, tmp_path):
    from huddleroom.services.meeting_service import MeetingService
    from huddleroom.workers.meeting_tasks import start_meeting_async

    svc = MeetingService()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    test_project.workspace_path = str(workspace.resolve())
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Start Task Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "Item 1", "max_rounds": 1}],
    )

    # Add mock methods for commit and rollback
    db_session.commit = AsyncMock()
    db_session.rollback = AsyncMock()

    with patch("huddleroom.workers.meeting_tasks.AsyncSessionLocal") as mock_session_factory:
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=db_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("huddleroom.workers.meeting_tasks.MeetingContextService") as mock_ctx:
            mock_ctx.return_value.build_initial_context = AsyncMock(return_value="context")
            await start_meeting_async(str(meeting.id))

    await db_session.refresh(meeting)
    assert meeting.status in ("preparing", "active")


@pytest.mark.asyncio
async def test_start_meeting_caches_cli_context_with_initial_context_key(
    db_session, test_project, test_agent, tmp_path,
):
    """Fails if CLI contexts are stored as strings rather than runner-compatible dictionaries."""
    from huddleroom.services.meeting_service import MeetingService
    from huddleroom.workers.meeting_tasks import start_meeting_async

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    test_project.workspace_path = str(workspace.resolve())
    test_agent.adapter_type = "cli"
    await db_session.flush()
    meeting = await MeetingService().create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="CLI Context Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "Item 1", "max_rounds": 1}],
    )
    db_session.commit = AsyncMock()
    db_session.rollback = AsyncMock()

    with patch("huddleroom.workers.meeting_tasks.AsyncSessionLocal") as mock_session_factory:
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=db_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("huddleroom.workers.meeting_tasks.MeetingContextService") as mock_ctx:
            mock_ctx.return_value.build_initial_context = AsyncMock(return_value="cli context")
            await start_meeting_async(str(meeting.id))

    assert meeting.participant_contexts[str(test_agent.id)] == {"initial_ctx": "cli context"}


@pytest.mark.asyncio
async def test_meeting_engine_dispatches_start_on_scheduled_event(db_session, test_project, test_agent):
    from huddleroom.services.meeting_service import MeetingService
    from huddleroom.workers.consumers.meeting_engine import MeetingEngineConsumer
    from huddleroom.services.event_bus import BusEvent
    from datetime import datetime, timezone

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Consumer Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "Item 1", "max_rounds": 1}],
        auto_start=True,
    )

    event = BusEvent(
        id=uuid.uuid4(),
        project_id=test_project.id,
        event_type="meeting.scheduled",
        payload={"meeting_id": str(meeting.id), "auto_start": True},
        source="system",
        emitted_at=datetime.now(timezone.utc),
    )

    with patch("huddleroom.workers.meeting_tasks.dispatch_start_meeting") as mock_dispatch:
        consumer = MeetingEngineConsumer()
        await consumer.process_event(db=db_session, event=event)
        mock_dispatch.assert_called_once_with(str(meeting.id), meeting.project_id)


@pytest.mark.asyncio
async def test_meeting_engine_uses_in_process_dispatch_for_sqlite(db_session, test_project, test_agent):
    from datetime import datetime, timezone

    from huddleroom.services.event_bus import BusEvent
    from huddleroom.services.meeting_service import MeetingService
    from huddleroom.workers.consumers.meeting_engine import MeetingEngineConsumer

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="SQLite Dispatch Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "Item 1", "max_rounds": 1}],
        auto_start=True,
    )

    event = BusEvent(
        id=uuid.uuid4(),
        project_id=test_project.id,
        event_type="meeting.scheduled",
        payload={"meeting_id": str(meeting.id), "auto_start": True},
        source="system",
        emitted_at=datetime.now(timezone.utc),
    )

    inline_start = AsyncMock()
    with patch("huddleroom.workers.meeting_tasks.start_meeting_async", inline_start):
        with patch("huddleroom.workers.meeting_tasks.start_meeting") as mock_task:
            mock_task.delay.side_effect = AttributeError("'NoneType' object has no attribute 'Redis'")
            consumer = MeetingEngineConsumer()
            await consumer.process_event(db=db_session, event=event)
            await asyncio.sleep(0)

    inline_start.assert_awaited_once_with(str(meeting.id))
    mock_task.delay.assert_not_called()
