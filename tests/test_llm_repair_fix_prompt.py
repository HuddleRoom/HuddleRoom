"""Test custom fix_prompt_fn injection in complete_with_repair retry loops."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from huddleroom.services.llm_structured_repair import complete_with_repair, _default_fix_prompt
from huddleroom.services.orchestration_agent_definition_analyzer import _repair_instruction


def _mock_response(content: str):
    """Create a minimal mock litellm response envelope for _extract_content."""
    return {
        "choices": [
            {
                "message": {
                    "content": content
                }
            }
        ]
    }


@pytest.mark.asyncio
async def test_custom_fix_prompt_fn_used_in_retry():
    """Assert that custom fix_prompt_fn is used in the retry message."""
    call_count = 0
    captured_messages = []

    async def stub_completion_fn(**kwargs):
        nonlocal call_count
        call_count += 1
        captured_messages.append(kwargs.get("messages", []))
        if call_count == 1:
            # First call: return malformed JSON to trigger parse failure
            return _mock_response("not valid json")
        else:
            # Second call: return valid JSON
            return _mock_response('{"valid": "json"}')

    def stub_parse(raw: str):
        """Parse that rejects non-JSON and returns dict on valid JSON."""
        payload = json.loads(raw)
        if not isinstance(payload, dict) or "valid" not in payload:
            raise ValueError("payload missing 'valid' key")
        return payload

    def custom_fix_prompt(error_text: str) -> str:
        return f"CUSTOM FIX: {error_text}"

    result = await complete_with_repair(
        stub_completion_fn,
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
        },
        stub_parse,
        max_attempts=3,
        fix_prompt_fn=custom_fix_prompt,
    )

    # Verify we got the valid result on retry
    assert result == {"valid": "json"}
    assert call_count == 2

    # Verify the second call's messages include our custom fix prompt
    second_call_messages = captured_messages[1]
    # Should have: original user message, failed assistant response, and custom fix user message
    assert len(second_call_messages) >= 3

    # The last message should be the custom fix prompt
    last_message = second_call_messages[-1]
    assert last_message["role"] == "user"
    assert last_message["content"].startswith("CUSTOM FIX:")


@pytest.mark.asyncio
async def test_custom_fix_prompt_fn_applied_on_extract_content_failure():
    """Assert custom fix_prompt_fn is applied when _extract_content fails (malformed envelope)."""
    call_count = 0
    captured_messages = []

    async def stub_completion_fn(**kwargs):
        nonlocal call_count
        call_count += 1
        captured_messages.append(kwargs.get("messages", []))
        if call_count == 1:
            # First call: malformed envelope (no content) -> _extract_content fails
            return {"choices": [{"message": {}}]}
        else:
            # Second call: valid response
            return _mock_response('{"valid": "json"}')

    def stub_parse(raw: str):
        """Parse that rejects non-JSON."""
        payload = json.loads(raw)
        if not isinstance(payload, dict) or "valid" not in payload:
            raise ValueError("payload missing 'valid' key")
        return payload

    def custom_fix_prompt(error_text: str) -> str:
        return f"EXTRACT_FAILED: {error_text}"

    result = await complete_with_repair(
        stub_completion_fn,
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
        },
        stub_parse,
        max_attempts=3,
        fix_prompt_fn=custom_fix_prompt,
    )

    # Verify we got the valid result on retry
    assert result == {"valid": "json"}
    assert call_count == 2

    # Verify the second call's messages include our custom fix prompt
    second_call_messages = captured_messages[1]
    # Should have: original user message + custom fix user message (no assistant turn)
    assert len(second_call_messages) == 2
    assert second_call_messages[0] == {"role": "user", "content": "test"}

    # The second message should be the custom fix prompt (no assistant turn appended)
    last_message = second_call_messages[1]
    assert last_message["role"] == "user"
    assert last_message["content"].startswith("EXTRACT_FAILED:")


@pytest.mark.asyncio
async def test_default_fix_prompt_when_none_provided():
    """Assert that _default_fix_prompt is used when fix_prompt_fn is None."""
    call_count = 0
    captured_messages = []

    async def stub_completion_fn(**kwargs):
        nonlocal call_count
        call_count += 1
        captured_messages.append(kwargs.get("messages", []))
        if call_count == 1:
            return _mock_response("bad response")
        else:
            return _mock_response('{"valid": "json"}')

    def stub_parse(raw: str):
        payload = json.loads(raw)
        if not isinstance(payload, dict) or "valid" not in payload:
            raise ValueError("test error message")
        return payload

    result = await complete_with_repair(
        stub_completion_fn,
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
        },
        stub_parse,
        max_attempts=3,
        fix_prompt_fn=None,  # Explicitly None to use default
    )

    assert result == {"valid": "json"}
    assert call_count == 2

    # Verify the retry message uses default prompt content
    second_call_messages = captured_messages[1]
    last_message = second_call_messages[-1]
    assert last_message["role"] == "user"
    # Default prompt should contain these words
    assert "rejected" in last_message["content"]
    assert "corrected JSON" in last_message["content"]
    assert "schema" not in last_message["content"]


def test_repair_instruction_goal_reference_error():
    """Assert _repair_instruction handles goal/project reference errors correctly."""
    error_text = "proposals must not reference a goal or project"
    result = _repair_instruction(error_text)

    # Should contain guidance about generic role
    assert "generic role" in result
    # Should NOT mention schema (that's the old hardcoded message)
    assert "schema" not in result
    # Should mention the specific constraint
    assert "proposed_description" in result
    assert "proposed_persona" in result
    assert "no mention of any goal, project" in result


def test_repair_instruction_wrong_fields_error():
    """Assert _repair_instruction handles field mismatch errors."""
    error_text = "must contain exactly the required fields"
    result = _repair_instruction(error_text)

    # Should list the exact fields required
    assert "status" in result
    assert "problems" in result
    assert "reason" in result
    assert "approved_work_functions" in result
    assert "proposed_description" in result
    assert "proposed_persona" in result


def test_repair_instruction_null_proposals_error():
    """Assert _repair_instruction handles approved-status null constraint."""
    error_text = "approved assessment proposals must be null"
    result = _repair_instruction(error_text)

    # Should explain the null requirement
    assert "null" in result
    assert "approved" in result
    assert "proposed_description" in result
    assert "proposed_persona" in result


def test_repair_instruction_unknown_error():
    """Assert _repair_instruction falls back to default for unknown errors."""
    error_text = "some unknown error condition"
    result = _repair_instruction(error_text)

    # Should fall back to default prompt (which includes the error_text)
    assert "some unknown error condition" in result
    assert "rejected" in result
    # Should not be attempting to provide specialized guidance
    assert "proposed_description" not in result


def test_default_fix_prompt_includes_error_and_instructions():
    """Assert _default_fix_prompt includes error text and generic guidance."""
    error_text = "JSON parse failed at line 5"
    result = _default_fix_prompt(error_text)

    # Should include the error
    assert "JSON parse failed at line 5" in result
    # Should include generic guidance
    assert "rejected" in result
    assert "corrected JSON" in result
    # Should NOT mention schema (that's the hardcoded mistake we're fixing)
    assert "schema" not in result
