"""Tests for MeetingOutcomeService prompt construction and outcome mapping.

All litellm calls are mocked. SQLAlchemy models are represented as MagicMock
objects with the relevant attributes set directly.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _meeting(meeting_type: str = "decision") -> MagicMock:
    m = MagicMock()
    m.id = uuid.uuid4()
    m.project_id = uuid.uuid4()
    m.title = "Outcome Test Meeting"
    m.meeting_type = meeting_type
    m.status = "concluding"
    m.turn_strategy = "round_robin"
    m.participant_agent_ids = []
    m.participant_contexts = {}
    m.summary = None
    m.planner_agent_id = None
    m.planner_summary = None
    return m


def _agent(name="TestAgent", role="engineer", system_prompt=None) -> MagicMock:
    a = MagicMock()
    a.id = uuid.uuid4()
    a.name = name
    a.role = role
    a.system_prompt = system_prompt
    a.provider = "openai"
    a.model = "gpt-4o-mini"
    return a


def _decision(meeting: MagicMock, title: str, chosen_option: str = "REST") -> MagicMock:
    d = MagicMock()
    d.id = uuid.uuid4()
    d.meeting_id = meeting.id
    d.agenda_item_id = uuid.uuid4()
    d.title = title
    d.question = None
    d.chosen_option = chosen_option
    d.rationale = "Good reasons."
    d.decided_by = "consensus"
    d.confidence = 0.9
    d.is_partial = False
    d.is_vetoed = False
    d.knowledge_item_id = None
    return d


def _llm_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


# ---------------------------------------------------------------------------
# Test 1: extract_action_items includes decision title in prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_action_items_includes_decisions_in_prompt():
    from huddleroom.services.meeting_outcome import MeetingOutcomeService

    mtg = _meeting()
    agent = _agent()
    mtg.participant_agent_ids = [str(agent.id)]

    decision = _decision(mtg, title="API choice", chosen_option="REST")

    captured_messages = []

    async def fake_acompletion(**kwargs):
        captured_messages.extend(kwargs.get("messages", []))
        return _llm_response("[]")

    # Mock transcript turn
    mock_turn = MagicMock()
    mock_turn.speaker_agent_id = agent.id
    mock_turn.content = "We should use REST."

    db = AsyncMock()

    async def fake_get(model, pk):
        if str(pk) == str(agent.id):
            return agent
        return None

    db.get.side_effect = fake_get
    turns_result = MagicMock()
    turns_result.scalars.return_value.all.return_value = [mock_turn]
    db.execute.return_value = turns_result

    svc = MeetingOutcomeService()
    with patch("huddleroom.services.meeting_outcome.litellm.acompletion", side_effect=fake_acompletion):
        await svc.extract_action_items(db=db, meeting=mtg, decisions=[decision])

    user_msgs = [m["content"] for m in captured_messages if m["role"] == "user"]
    assert any("API choice" in msg for msg in user_msgs), (
        "Decision title 'API choice' must appear in the user message sent to litellm"
    )


# ---------------------------------------------------------------------------
# Test 2: extract_action_items maps depends_on_decision_id
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_action_items_maps_depends_on_decision_id():
    from huddleroom.models.meeting import MeetingActionItem
    from huddleroom.services.meeting_outcome import MeetingOutcomeService

    if not hasattr(MeetingActionItem, "depends_on_decision_id"):
        pytest.skip("MeetingActionItem.depends_on_decision_id field absent")

    mtg = _meeting()
    agent = _agent()
    mtg.participant_agent_ids = [str(agent.id)]

    decision = _decision(mtg, title="API choice", chosen_option="REST")

    llm_payload = (
        '[{"description": "do thing", "assignee_agent_name": null, '
        '"priority": 70, "deadline_days": null, '
        '"depends_on_decision_title": "API choice"}]'
    )

    mock_turn = MagicMock()
    mock_turn.speaker_agent_id = agent.id
    mock_turn.content = "We should do thing."

    db = AsyncMock()

    async def fake_get(model, pk):
        if str(pk) == str(agent.id):
            return agent
        return None

    db.get.side_effect = fake_get
    turns_result = MagicMock()
    turns_result.scalars.return_value.all.return_value = [mock_turn]
    db.execute.return_value = turns_result
    db.add = MagicMock()
    db.flush = AsyncMock()

    svc = MeetingOutcomeService()

    # Use streaming mock pattern to match foundation expectations
    from types import SimpleNamespace

    class StreamingResponse:
        def __init__(self, payload):
            self.chunks = [
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=payload, reasoning_content=None))]
                )
            ]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.chunks:
                return self.chunks.pop()
            raise StopAsyncIteration

    async def mock_acompletion(**_kwargs):
        return StreamingResponse(llm_payload)

    with patch("huddleroom.services.meeting_outcome.litellm.acompletion", new_callable=AsyncMock) as mock_llm_comp, \
         patch("huddleroom.services.agent_response_stream.litellm.stream_chunk_builder") as mock_builder:
        mock_llm_comp.side_effect = mock_acompletion
        # stream_chunk_builder should return a response object with .choices[0].message.content
        mock_builder.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=llm_payload))]
        )
        action_items = await svc.extract_action_items(db=db, meeting=mtg, decisions=[decision])

    assert len(action_items) == 1
    assert action_items[0].depends_on_decision_id == decision.id, (
        f"Expected depends_on_decision_id={decision.id}, "
        f"got {action_items[0].depends_on_decision_id}"
    )


# ---------------------------------------------------------------------------
# Test 3: standup fallback action items use required_followup blockers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fallback_action_items_creates_standup_items_from_required_followup():
    from huddleroom.services.meeting_outcome import MeetingOutcomeService

    mtg = _meeting(meeting_type="standup")

    agenda_item = MagicMock()
    agenda_item.resolution_kind = "updates_shared"
    agenda_item.required_followup = "waiting on staging access; need production logs"
    agenda_item.resolution_summary = "Standup completed with blockers."
    agenda_item.title = "Daily updates"

    result = MagicMock()
    result.scalars.return_value.all.return_value = [agenda_item]

    db = AsyncMock()
    db.execute.return_value = result
    db.add = MagicMock()
    db.flush = AsyncMock()

    svc = MeetingOutcomeService()
    action_items = await svc._fallback_action_items(db=db, meeting=mtg)

    assert [item.description for item in action_items] == [
        "waiting on staging access",
        "need production logs",
    ]
    assert all(item.is_partial for item in action_items)


# ---------------------------------------------------------------------------
# Test 4: standup fallback drops explanatory semicolon tails
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fallback_action_items_ignores_explanatory_standup_tail_fragments():
    from huddleroom.services.meeting_outcome import MeetingOutcomeService

    mtg = _meeting(meeting_type="standup")

    agenda_item = MagicMock()
    agenda_item.resolution_kind = "updates_shared"
    agenda_item.required_followup = (
        "waiting on staging access; risk persists until IAM grants propagate; need production logs"
    )
    agenda_item.resolution_summary = "Standup completed with blockers."
    agenda_item.title = "Daily updates"

    result = MagicMock()
    result.scalars.return_value.all.return_value = [agenda_item]

    db = AsyncMock()
    db.execute.return_value = result
    db.add = MagicMock()
    db.flush = AsyncMock()

    svc = MeetingOutcomeService()
    action_items = await svc._fallback_action_items(db=db, meeting=mtg)

    assert [item.description for item in action_items] == [
        "waiting on staging access",
        "need production logs",
    ]


# ---------------------------------------------------------------------------
# Test 5: standup fallback dedupes near-duplicate blocker phrasings
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fallback_action_items_dedupes_near_duplicate_standup_blockers():
    from huddleroom.services.meeting_outcome import MeetingOutcomeService

    mtg = _meeting(meeting_type="standup")

    agenda_item_one = MagicMock()
    agenda_item_one.resolution_kind = "updates_shared"
    agenda_item_one.required_followup = "waiting on staging access; need production logs"
    agenda_item_one.resolution_summary = "Standup completed with blockers."
    agenda_item_one.title = "Agent one update"

    agenda_item_two = MagicMock()
    agenda_item_two.resolution_kind = "updates_shared"
    agenda_item_two.required_followup = "blocked on getting staging access approval"
    agenda_item_two.resolution_summary = "Standup completed with blockers."
    agenda_item_two.title = "Agent two update"

    result = MagicMock()
    result.scalars.return_value.all.return_value = [agenda_item_one, agenda_item_two]

    db = AsyncMock()
    db.execute.return_value = result
    db.add = MagicMock()
    db.flush = AsyncMock()

    svc = MeetingOutcomeService()
    action_items = await svc._fallback_action_items(db=db, meeting=mtg)

    assert [item.description for item in action_items] == [
        "waiting on staging access",
        "need production logs",
    ]


# ---------------------------------------------------------------------------
# Test 6: standup extract_action_items bypasses LLM extraction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_action_items_bypasses_llm_for_standup():
    from huddleroom.services.meeting_outcome import MeetingOutcomeService

    mtg = _meeting(meeting_type="standup")

    agenda_item = MagicMock()
    agenda_item.resolution_kind = "updates_shared"
    agenda_item.required_followup = "waiting on staging access"
    agenda_item.resolution_summary = "Standup completed with blockers."
    agenda_item.title = "Daily updates"

    result = MagicMock()
    result.scalars.return_value.all.return_value = [agenda_item]

    db = AsyncMock()
    db.execute.return_value = result
    db.add = MagicMock()
    db.flush = AsyncMock()

    svc = MeetingOutcomeService()
    svc._format_transcript = AsyncMock(side_effect=AssertionError("standup should bypass transcript formatting"))
    with patch("huddleroom.services.meeting_outcome.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        action_items = await svc.extract_action_items(db=db, meeting=mtg, decisions=[])

    mock_llm.assert_not_awaited()
    svc._format_transcript.assert_not_awaited()
    assert [item.description for item in action_items] == ["waiting on staging access"]


@pytest.mark.asyncio
async def test_extract_action_items_falls_back_to_explicit_transcript_action_items_for_decision_meetings():
    from huddleroom.services.meeting_outcome import MeetingOutcomeService

    mtg = _meeting(meeting_type="decision")

    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()

    svc = MeetingOutcomeService()
    svc._format_transcript = AsyncMock(
        return_value=(
            "live-architect: We should adopt the event bus.\n"
            "live-architect: Action item: Draft the ADR for the event-driven migration.\n"
            "live-pm: Action item: Run a 2-week spike for the event bus.\n"
        )
    )
    with patch("huddleroom.services.meeting_outcome.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _llm_response("[]")
        action_items = await svc.extract_action_items(db=db, meeting=mtg, decisions=[])

    assert [item.description for item in action_items] == [
        "Draft the ADR for the event-driven migration.",
        "Run a 2-week spike for the event bus.",
    ]
    assert all(item.is_partial for item in action_items)


@pytest.mark.asyncio
async def test_extract_action_items_falls_back_to_imperative_decision_outcomes():
    from huddleroom.services.meeting_outcome import MeetingOutcomeService

    mtg = _meeting(meeting_type="decision")
    decision = _decision(
        mtg,
        title="Migration plan",
        chosen_option="Write an ADR and spike the event bus",
    )

    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()

    svc = MeetingOutcomeService()
    svc._format_transcript = AsyncMock(return_value="live-pm: I support the ADR and spike approach.")
    with patch("huddleroom.services.meeting_outcome.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _llm_response("[]")
        action_items = await svc.extract_action_items(db=db, meeting=mtg, decisions=[decision])

    assert [item.description for item in action_items] == ["Write an ADR and spike the event bus"]
    assert all(item.is_partial for item in action_items)


# ---------------------------------------------------------------------------
# Test 7: write_knowledge_items summary contains "## Outcome"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_knowledge_items_summary_has_sections():
    from huddleroom.services.meeting_outcome import MeetingOutcomeService

    mtg = _meeting()
    decision = _decision(mtg, title="API choice")

    summary_text = "## Outcome\nblah\n## Decisions\nblah"

    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()

    svc = MeetingOutcomeService()
    with patch("huddleroom.services.meeting_outcome.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _llm_response(summary_text)
        await svc.write_knowledge_items(
            db=db,
            meeting=mtg,
            decisions=[decision],
            transcript="Alice: We chose REST.",
        )

    assert mtg.summary is not None
    assert "## Outcome" in mtg.summary, (
        f"Expected '## Outcome' in meeting.summary, got: {mtg.summary!r}"
    )


# ---------------------------------------------------------------------------
# Test 8: standup summaries bypass LLM and still persist
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_knowledge_items_bypasses_llm_for_standup():
    from huddleroom.services.meeting_outcome import MeetingOutcomeService

    mtg = _meeting(meeting_type="standup")

    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()

    svc = MeetingOutcomeService()
    expected_summary = svc._build_fallback_summary(meeting=mtg, decisions=[])
    svc._format_transcript = AsyncMock(side_effect=AssertionError("standup should bypass transcript formatting"))
    with patch("huddleroom.services.meeting_outcome.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        kis = await svc.write_knowledge_items(
            db=db,
            meeting=mtg,
            decisions=[],
            transcript=None,
        )

    mock_llm.assert_not_awaited()
    svc._format_transcript.assert_not_awaited()
    assert len(kis) == 1
    assert mtg.summary == expected_summary
    assert kis[0].content == expected_summary
    assert kis[0].content_type == "summary"


# ---------------------------------------------------------------------------
# Test 9: blank summary output falls back to non-empty persisted summary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_knowledge_items_uses_fallback_summary_for_blank_llm_output():
    from huddleroom.services.meeting_outcome import MeetingOutcomeService

    mtg = _meeting(meeting_type="standup")

    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()

    svc = MeetingOutcomeService()
    with patch("huddleroom.services.meeting_outcome.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _llm_response("   ")
        kis = await svc.write_knowledge_items(
            db=db,
            meeting=mtg,
            decisions=[],
            transcript="Alice: DONE: shipped auth fix\nAlice: NOW: rollout\nAlice: BLOCKERS: staging access",
        )

    assert len(kis) == 1
    assert mtg.summary
    assert "## Outcome" in mtg.summary
    assert "None." in mtg.summary


# ---------------------------------------------------------------------------
# Test 10: run_planner_summary system prompt contains agent.system_prompt and "HuddleRoom Planner Role"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_planner_summary_composes_system_prompt():
    from huddleroom.services.meeting_outcome import MeetingOutcomeService

    mtg = _meeting()
    planner = _agent(name="PlannerAgent", role="planner", system_prompt="MY_AGENT_PROMPT")
    mtg.planner_agent_id = planner.id

    captured_messages = []

    async def fake_acompletion(**kwargs):
        captured_messages.extend(kwargs.get("messages", []))
        return _llm_response("DECISIONS\nACTIONS\nOPEN_QUESTIONS")

    db = AsyncMock()

    async def fake_get(model, pk):
        if str(pk) == str(planner.id):
            return planner
        return None

    db.get.side_effect = fake_get
    db.add = MagicMock()
    db.flush = AsyncMock()

    svc = MeetingOutcomeService()
    with patch("huddleroom.services.meeting_outcome.litellm.acompletion", side_effect=fake_acompletion):
        await svc.run_planner_summary(
            db=db, meeting=mtg, decisions=[], transcript="Alice: hello.",
        )

    system_msgs = [m["content"] for m in captured_messages if m["role"] == "system"]
    assert system_msgs, "Expected at least one system message to litellm"
    combined = "\n".join(system_msgs)
    assert "MY_AGENT_PROMPT" in combined, (
        "Agent's system_prompt must appear in the system message"
    )
    assert "HuddleRoom Planner Role" in combined, (
        "'HuddleRoom Planner Role' must appear in the system message"
    )
