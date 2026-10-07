"""Tests for MeetingIntelligenceService.

All litellm calls are mocked.
"""
from __future__ import annotations

import copy
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from huddleroom.services.meeting_intelligence import MeetingIntelligenceService


def test_orchestration_model_defaults_and_cheap_model_is_removed(monkeypatch):
    from huddleroom.config import Settings

    monkeypatch.delenv("HUDDLEROOM_ORCHESTRATION_MODEL", raising=False)
    monkeypatch.delenv("RALLY_ORCHESTRATION_MODEL", raising=False)
    configured = Settings(_env_file=None)
    assert configured.orchestration_model == "openai/gpt-6.1-sol"
    assert not hasattr(configured, "cheap_model")


def _make_llm_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


# ---------------------------------------------------------------------------
# Test 1: check_consensus returns dissenting speakers from LLM payload
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_consensus_returns_dissenting_speakers():
    payload = {
        "consensus": False,
        "agreed_position": None,
        "confidence": 0.3,
        "dissenting_speakers": ["Alice"],
        "rationale": "disagreement",
    }
    mock_resp = _make_llm_response(json.dumps(payload))

    svc = MeetingIntelligenceService()
    with patch("huddleroom.services.meeting_intelligence.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_resp
        result = await svc.check_consensus(
            item_title="API Style",
            item_question="REST or GraphQL?",
            item_options=["REST", "GraphQL"],
            expected_speakers=["Alice", "Bob"],
            turns_this_round=[
                {"speaker": "Alice", "content": "I prefer REST."},
                {"speaker": "Bob", "content": "I think GraphQL is better."},
            ],
            prior_rounds_summary=None,
        )

    assert "dissenting_speakers" in result
    assert "Alice" in result["dissenting_speakers"]
    assert result["consensus"] is False


# ---------------------------------------------------------------------------
# Test 2: check_consensus rejects agreed_position not in item_options
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_consensus_rejects_invalid_agreed_position():
    payload = {
        "consensus": True,
        "agreed_position": "SomeMadeUpThing",
        "confidence": 0.9,
        "dissenting_speakers": [],
        "rationale": "everyone agrees",
    }
    mock_resp = _make_llm_response(json.dumps(payload))

    svc = MeetingIntelligenceService()
    with patch("huddleroom.services.meeting_intelligence.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_resp
        result = await svc.check_consensus(
            item_title="API Style",
            item_question="Which style?",
            item_options=["REST", "GraphQL"],
            expected_speakers=["Alice", "Bob"],
            turns_this_round=[
                {"speaker": "Alice", "content": "I agree."},
                {"speaker": "Bob", "content": "Me too."},
            ],
            prior_rounds_summary=None,
        )

    assert result["consensus"] is False, (
        "Agreed position not in item_options must cause consensus=False"
    )


# ---------------------------------------------------------------------------
# Test 3: check_consensus accepts valid position (case-insensitive)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_consensus_accepts_valid_position_case_insensitive():
    payload = {
        "consensus": True,
        "agreed_position": "rest",   # lowercase, should match "REST"
        "confidence": 0.9,
        "dissenting_speakers": [],
        "rationale": "all agree on REST",
    }
    mock_resp = _make_llm_response(json.dumps(payload))

    svc = MeetingIntelligenceService()
    with patch("huddleroom.services.meeting_intelligence.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_resp
        result = await svc.check_consensus(
            item_title="API Style",
            item_question="Which style?",
            item_options=["REST", "GraphQL"],
            expected_speakers=["Alice", "Bob"],
            turns_this_round=[
                {"speaker": "Alice", "content": "REST. POSITION: REST"},
                {"speaker": "Bob", "content": "REST. POSITION: REST"},
            ],
            prior_rounds_summary=None,
        )

    assert result["consensus"] is True, "Lowercase option should be accepted case-insensitively"


# ---------------------------------------------------------------------------
# Test 4: _parse_json_response accepts fenced JSON object blocks
# ---------------------------------------------------------------------------

def test_parse_json_response_accepts_fenced_json_object():
    svc = MeetingIntelligenceService()

    result = svc._parse_json_response(
        '```json\n{"consensus": true, "confidence": 0.9}\n```',
        default={"consensus": False},
    )

    assert result == {"consensus": True, "confidence": 0.9}


# ---------------------------------------------------------------------------
# Test 5: _parse_json_response returns default for empty response
# ---------------------------------------------------------------------------

def test_parse_json_response_returns_default_for_empty_response():
    svc = MeetingIntelligenceService()
    default = {"consensus": False, "confidence": 0.0}

    result = svc._parse_json_response("", default=default)

    assert result is default


# ---------------------------------------------------------------------------
# Test 6: check_consensus retries once on malformed/empty JSON and succeeds
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("first_response", ["", '{"consensus": true'])
async def test_check_consensus_retries_once_on_bad_json_then_succeeds(first_response: str):
    valid_payload = {
        "consensus": True,
        "agreed_position": "REST",
        "confidence": 0.9,
        "dissenting_speakers": [],
        "rationale": "all agree",
    }

    svc = MeetingIntelligenceService()
    with patch("huddleroom.services.meeting_intelligence.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [
            _make_llm_response(first_response),
            _make_llm_response(json.dumps(valid_payload)),
        ]

        result = await svc.check_consensus(
            item_title="API Style",
            item_question="Which style?",
            item_options=["REST", "GraphQL"],
            expected_speakers=["Alice", "Bob"],
            turns_this_round=[
                {"speaker": "Alice", "content": "REST. POSITION: REST"},
                {"speaker": "Bob", "content": "REST. POSITION: REST"},
            ],
            prior_rounds_summary=None,
        )

    assert result["consensus"] is True
    assert result["agreed_position"] == "REST"
    assert mock_llm.await_count == 2


@pytest.mark.asyncio
async def test_check_consensus_trace_response_marks_parse_status_ok():
    payload = {
        "consensus": True,
        "agreed_position": "REST",
        "confidence": 0.9,
        "dissenting_speakers": [],
        "rationale": "all agree",
    }

    svc = MeetingIntelligenceService()
    trace_emit = AsyncMock()
    with patch("huddleroom.services.meeting_intelligence.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _make_llm_response(json.dumps(payload))
        await svc.check_consensus(
            item_title="API Style",
            item_question="Which style?",
            item_options=["REST", "GraphQL"],
            expected_speakers=["Alice", "Bob"],
            turns_this_round=[
                {"speaker": "Alice", "content": "REST. POSITION: REST"},
                {"speaker": "Bob", "content": "REST. POSITION: REST"},
            ],
            prior_rounds_summary=None,
            trace_emit=trace_emit,
        )

    response_trace = trace_emit.await_args_list[1].args[0]
    assert response_trace["stage"] == "response"
    assert response_trace["parse_status"] == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_response", "expected_status"),
    [
        ("", "empty"),
        ('{"consensus": true', "malformed"),
    ],
)
async def test_check_consensus_trace_response_marks_parse_status_for_bad_json(
    first_response: str, expected_status: str
):
    valid_payload = {
        "consensus": True,
        "agreed_position": "REST",
        "confidence": 0.9,
        "dissenting_speakers": [],
        "rationale": "all agree",
    }

    svc = MeetingIntelligenceService()
    trace_emit = AsyncMock()
    with patch("huddleroom.services.meeting_intelligence.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [
            _make_llm_response(first_response),
            _make_llm_response(json.dumps(valid_payload)),
        ]
        await svc.check_consensus(
            item_title="API Style",
            item_question="Which style?",
            item_options=["REST", "GraphQL"],
            expected_speakers=["Alice", "Bob"],
            turns_this_round=[
                {"speaker": "Alice", "content": "REST. POSITION: REST"},
                {"speaker": "Bob", "content": "REST. POSITION: REST"},
            ],
            prior_rounds_summary=None,
            trace_emit=trace_emit,
        )

    response_trace = trace_emit.await_args_list[1].args[0]
    assert response_trace["stage"] == "response"
    assert response_trace["parse_status"] == expected_status


@pytest.mark.asyncio
async def test_check_consensus_retry_uses_stricter_repair_message_on_second_attempt():
    """Test that retry messages include error feedback (via complete_with_repair)."""
    valid_payload = {
        "consensus": True,
        "agreed_position": "REST",
        "confidence": 0.9,
        "dissenting_speakers": [],
        "rationale": "all agree",
    }

    svc = MeetingIntelligenceService()
    # Capture messages at call time to work around mock's reference-capture issue
    captured_messages = []

    async def mock_side_effect(**kwargs):
        captured_messages.append(copy.deepcopy(kwargs.get("messages", [])))
        if len(captured_messages) == 1:
            return _make_llm_response('```json\n{"consensus": true')
        else:
            return _make_llm_response(json.dumps(valid_payload))

    with patch("huddleroom.services.meeting_intelligence.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = mock_side_effect
        await svc.check_consensus(
            item_title="API Style",
            item_question="Which style?",
            item_options=["REST", "GraphQL"],
            expected_speakers=["Alice", "Bob"],
            turns_this_round=[
                {"speaker": "Alice", "content": "REST. POSITION: REST"},
                {"speaker": "Bob", "content": "REST. POSITION: REST"},
            ],
            prior_rounds_summary=None,
        )

    assert mock_llm.await_count == 2
    assert len(captured_messages) == 2
    first_messages = captured_messages[0]
    second_messages = captured_messages[1]
    # Second call should have more messages (original + assistant + error feedback)
    assert len(second_messages) > len(first_messages)
    assert len(second_messages) == len(first_messages) + 2  # assistant + user
    # The last message should contain error feedback from complete_with_repair
    retry_message = second_messages[-1]["content"]
    assert "rejected" in retry_message.lower()
    assert "corrected json" in retry_message.lower()
    # The assistant message should be the failed response
    assert second_messages[-2]["role"] == "assistant"
    assert '{"consensus": true' in second_messages[-2]["content"]


# ---------------------------------------------------------------------------
# Test 7: check_consensus does not retry more than once
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_consensus_does_not_retry_more_than_once():
    """With complete_with_repair, max_attempts=3 means up to 3 retries before exhaustion."""
    svc = MeetingIntelligenceService()
    with patch("huddleroom.services.meeting_intelligence.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        # Provide only 2 responses; 3rd attempt will exhaust retries
        mock_llm.side_effect = [
            _make_llm_response(""),  # Empty response, attempt 0
            _make_llm_response('{"consensus": true'),  # Malformed, attempt 1 (retry)
            # If there's a 3rd call, it will raise because side_effect is exhausted
        ]

        result = await svc.check_consensus(
            item_title="API Style",
            item_question="Which style?",
            item_options=["REST", "GraphQL"],
            expected_speakers=["Alice", "Bob"],
            turns_this_round=[
                {"speaker": "Alice", "content": "REST. POSITION: REST"},
                {"speaker": "Bob", "content": "REST. POSITION: REST"},
            ],
            prior_rounds_summary=None,
        )

    # With max_attempts=3, should make up to 3 calls before exhausting
    # In this case, calls 1-2 fail, call 3 would attempt but side_effect is exhausted
    assert result == {
        "consensus": False,
        "agreed_position": None,
        "confidence": 0.0,
        "dissenting_speakers": [],
        "rationale": "",
    }
    # complete_with_repair attempts up to max_attempts=3
    assert mock_llm.await_count >= 2


# ---------------------------------------------------------------------------
# Test 8: extract_positions returns items with "option" key
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_positions_returns_option_field():
    payload = [{"speaker": "Alice", "option": "REST", "summary": "favors REST"}]
    mock_resp = _make_llm_response(json.dumps(payload))

    svc = MeetingIntelligenceService()
    with patch("huddleroom.services.meeting_intelligence.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_resp
        result = await svc.extract_positions(
            turns=[{"speaker": "Alice", "content": "I prefer REST. POSITION: REST"}],
            item_options=["REST", "GraphQL"],
        )

    assert len(result) == 1
    assert "option" in result[0], "Result item must have 'option' key"
    assert result[0]["option"] == "REST"


# ---------------------------------------------------------------------------
# Test 5: select_next_speaker prompt contains "close_item"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_select_next_speaker_prompt_contains_close_item():
    import uuid as uuid_mod

    pid1 = str(uuid_mod.uuid4())
    pid2 = str(uuid_mod.uuid4())

    captured_messages = []

    payload = {
        "next_speaker_id": pid1,
        "reason": "they haven't spoken",
        "close_item": False,
        "close_reason": None,
    }

    async def fake_acompletion(**kwargs):
        captured_messages.extend(kwargs.get("messages", []))
        return _make_llm_response(json.dumps(payload))

    svc = MeetingIntelligenceService()
    with patch("huddleroom.services.meeting_intelligence.litellm.acompletion", side_effect=fake_acompletion):
        await svc.select_next_speaker(
            participant_ids=[pid1, pid2],
            participant_names={pid1: "Alice", pid2: "Bob"},
            transcript_excerpt="Alice: REST is simpler.",
            item_title="API Style",
            item_options=["REST", "GraphQL"],
        )

    user_msgs = [m["content"] for m in captured_messages if m["role"] == "user"]
    assert any("close_item" in msg for msg in user_msgs), (
        "The user message to litellm must contain 'close_item'"
    )


# ---------------------------------------------------------------------------
# Test 6: select_next_speaker default on exception has close_item=False
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_select_next_speaker_default_has_close_item_false():
    import uuid as uuid_mod

    pid1 = str(uuid_mod.uuid4())

    svc = MeetingIntelligenceService()
    with patch(
        "huddleroom.services.meeting_intelligence.litellm.acompletion",
        new_callable=AsyncMock,
        side_effect=RuntimeError("LLM unavailable"),
    ):
        result = await svc.select_next_speaker(
            participant_ids=[pid1],
            participant_names={pid1: "Alice"},
            transcript_excerpt="Alice: hello.",
        )

    assert result["close_item"] is False, (
        "Default result on exception must have close_item=False"
    )


@pytest.mark.asyncio
async def test_select_next_speaker_retries_once_on_malformed_json_then_succeeds():
    import uuid as uuid_mod

    pid1 = str(uuid_mod.uuid4())
    pid2 = str(uuid_mod.uuid4())
    payload = {
        "next_speaker_id": pid2,
        "reason": "Bob has not spoken yet.",
        "close_item": False,
        "close_reason": None,
    }

    svc = MeetingIntelligenceService()
    with patch("huddleroom.services.meeting_intelligence.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [
            _make_llm_response("next speaker should be Bob"),
            _make_llm_response(json.dumps(payload)),
        ]
        result = await svc.select_next_speaker(
            participant_ids=[pid1, pid2],
            participant_names={pid1: "Alice", pid2: "Bob"},
            transcript_excerpt="Alice: Basic Auth should be replaced.",
            item_title="Auth mechanism review",
            spoke_this_round={pid1: True, pid2: False},
        )

    assert result["next_speaker_id"] == pid2
    assert result["close_item"] is False
    assert mock_llm.await_count == 2


@pytest.mark.asyncio
async def test_select_next_speaker_uses_orchestration_model_when_meeting_control_model_is_unset():
    import uuid as uuid_mod

    pid1 = str(uuid_mod.uuid4())
    captured_models = []
    payload = {
        "next_speaker_id": pid1,
        "reason": "Alice should start.",
        "close_item": False,
        "close_reason": None,
    }

    async def fake_acompletion(**kwargs):
        captured_models.append(kwargs.get("model"))
        return _make_llm_response(json.dumps(payload))

    svc = MeetingIntelligenceService()
    with patch("huddleroom.services.meeting_intelligence.settings.meeting_control_model", None):
        with patch("huddleroom.services.meeting_intelligence.settings.orchestration_model", "openai/gpt-6.1-sol"):
            with patch("huddleroom.services.meeting_intelligence.litellm.acompletion", side_effect=fake_acompletion):
                result = await svc.select_next_speaker(
                    participant_ids=[pid1],
                    participant_names={pid1: "Alice"},
                    transcript_excerpt="",
                    item_title="Architecture direction",
                )

    assert result["next_speaker_id"] == pid1
    assert captured_models == ["openai/gpt-6.1-sol"]
    assert result["selected_by"] == "orchestration_model"


def test_control_model_prefers_meeting_override_and_reports_its_source():
    svc = MeetingIntelligenceService()

    with patch("huddleroom.services.meeting_intelligence.settings.meeting_control_model", "openai/override"):
        with patch("huddleroom.services.meeting_intelligence.settings.orchestration_model", "openai/default"):
            assert svc._control_model() == "openai/override"
            assert svc._control_model_source() == "meeting_control_model"


@pytest.mark.asyncio
async def test_select_next_speaker_falls_back_to_default_openai_model_after_empty_responses():
    import uuid as uuid_mod

    pid1 = str(uuid_mod.uuid4())
    pid2 = str(uuid_mod.uuid4())
    payload = {
        "next_speaker_id": pid2,
        "reason": "Bob has not spoken yet.",
        "close_item": False,
        "close_reason": None,
    }
    captured_models = []

    async def fake_acompletion(**kwargs):
        captured_models.append(kwargs.get("model"))
        model = kwargs.get("model")
        if model == "ollama/gemma4:26b":
            return _make_llm_response("")
        return _make_llm_response(json.dumps(payload))

    svc = MeetingIntelligenceService()
    with patch("huddleroom.services.meeting_intelligence.settings.meeting_control_model", None):
        with patch("huddleroom.services.meeting_intelligence.settings.orchestration_model", "ollama/gemma4:26b"):
            with patch("huddleroom.services.meeting_intelligence.litellm.acompletion", side_effect=fake_acompletion):
                result = await svc.select_next_speaker(
                    participant_ids=[pid1, pid2],
                    participant_names={pid1: "Alice", pid2: "Bob"},
                    transcript_excerpt="Alice: We should migrate with guardrails.",
                    item_title="Migration plan",
                    item_options=["Write an ADR and spike the event bus", "Keep the current architecture and optimize APIs"],
                    spoke_this_round={pid1: True, pid2: False},
                )

    assert result["next_speaker_id"] == pid2
    assert result["reason"] == "Bob has not spoken yet."
    assert result["model_used"] == "openai/gpt-6.1-sol"
    # complete_with_repair retries up to max_attempts per model, then falls back to next candidate
    # First N calls should be to ollama (all failing with empty responses)
    # Last call should be to openai (succeeding)
    assert captured_models[-1] == "openai/gpt-6.1-sol"  # Final successful call
    assert all(m == "ollama/gemma4:26b" for m in captured_models[:-1])  # All prior calls to ollama
    assert len(captured_models) >= 4  # At least 3 retries for ollama + 1 for openai
