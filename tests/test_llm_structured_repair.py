"""Unit tests for llm_structured_repair module.

Tests cover:
- complete_with_repair with successful repair on retry
- Message history accumulation across retries
- Exception re-raising after exhaustion
- cli_complete_with_repair with run_fn and resume_fn
"""

import json
from typing import Any
from unittest.mock import AsyncMock
from types import SimpleNamespace
from uuid import uuid4

import pytest

from huddleroom.services.llm_structured_repair import (
    complete_with_repair,
    cli_complete_with_repair,
)
from huddleroom.services.agent_response_stream import AgentResponseInvocation, InvocationContext


class TestCompleteWithRepair:
    """Test complete_with_repair API repair loop."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        """Successful parse on first attempt returns immediately."""
        valid_json = '{"result": "success"}'

        completion_fn = AsyncMock(return_value={
            "choices": [
                {"message": {"content": valid_json}}
            ]
        })

        def parse(raw: str) -> dict:
            return json.loads(raw)

        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
        }

        result = await complete_with_repair(completion_fn, request, parse)

        assert result == {"result": "success"}
        assert completion_fn.call_count == 1

    @pytest.mark.parametrize(
        "content",
        [
            '{"result": "success"}',
            ' \n```json\n{"result": "success"}\n```\n ',
        ],
        ids=["raw", "json-fence-with-whitespace"],
    )
    @pytest.mark.asyncio
    async def test_parse_receives_outer_fence_normalized_content(self, content):
        completion_fn = AsyncMock(return_value={"choices": [{"message": {"content": content}}]})

        assert await complete_with_repair(completion_fn, {"messages": []}, json.loads) == {"result": "success"}

    @pytest.mark.asyncio
    async def test_schema_repair_keeps_fenced_provider_output_in_history(self):
        fenced_wrong_schema = '```json\n{"result": 3}\n```'
        calls = []

        async def completion_fn(**kwargs):
            calls.append(kwargs["messages"])
            content = fenced_wrong_schema if len(calls) == 1 else '{"result": "success"}'
            return {"choices": [{"message": {"content": content}}]}

        def parse(payload):
            value = json.loads(payload)
            if not isinstance(value.get("result"), str):
                raise ValueError("result must be a string")
            return value

        assert await complete_with_repair(completion_fn, {"messages": []}, parse, max_attempts=2) == {
            "result": "success"
        }
        assert calls[1][-2] == {"role": "assistant", "content": fenced_wrong_schema}

    @pytest.mark.asyncio
    async def test_repair_after_stream_fallback_uses_fallback_as_parent(self, monkeypatch):
        events, responses = [], iter(("bad", "good"))

        async def publish(event):
            events.append(event)

        async def no_reset(_project_id):
            await __import__("asyncio").Event().wait()

        monkeypatch.setattr(
            "huddleroom.services.agent_response_relay.register_project_reset_monitor",
            AsyncMock(return_value=SimpleNamespace(generation=None)),
        )
        monkeypatch.setattr(
            "huddleroom.services.agent_response_relay.unregister_project_reset_monitor", AsyncMock()
        )
        monkeypatch.setattr("huddleroom.services.agent_response_stream.wait_for_project_reset", no_reset)
        invocation = AgentResponseInvocation(
            InvocationContext(uuid4(), "system", "test", None, "api", "test", "model"),
            publish=publish,
        )

        async def completion(**kwargs):
            if kwargs.get("stream"):
                raise TypeError("unexpected keyword argument 'stream'")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))]
            )

        def parse(raw):
            if raw == "bad":
                raise ValueError("invalid")
            return raw

        assert await complete_with_repair(completion, {"messages": []}, parse, invocation=invocation) == "good"
        starts = [event for event in events if event.event_type == "agent_response.started"]
        assert starts[2].parent_call_id == starts[1].call_id

    @pytest.mark.asyncio
    async def test_recovery_after_bad_json_on_attempts_1_and_2(self):
        """Parse failure on attempts 1 and 2, success on attempt 3.

        Verifies:
        - Messages accumulate across retries
        - Both prior bad responses and user fix messages are present on the 3rd call
        - The model gets full context of all failures
        """
        from copy import deepcopy

        bad_json_1 = '{broken json'
        bad_json_2 = '{"incomplete":'
        good_json = '{"result": "success"}'

        # Track messages passed on each call (must deepcopy since list is mutated)
        call_messages = []

        async def completion_fn(**kwargs):
            # Deepcopy to capture the state at this point in time
            call_messages.append(deepcopy(kwargs.get("messages")))
            call_num = len(call_messages)

            if call_num == 1:
                content = bad_json_1
            elif call_num == 2:
                content = bad_json_2
            else:
                content = good_json

            return {
                "choices": [
                    {"message": {"content": content}}
                ]
            }

        def parse(raw: str) -> dict:
            return json.loads(raw)

        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "original request"}],
        }

        result = await complete_with_repair(completion_fn, request, parse, max_attempts=3)

        assert result == {"result": "success"}
        assert len(call_messages) == 3

        # First call: only original message
        assert call_messages[0] == [{"role": "user", "content": "original request"}]

        # Second call: original + assistant(bad_json_1) + user(fix_message)
        assert len(call_messages[1]) == 3
        assert call_messages[1][0] == {"role": "user", "content": "original request"}
        assert call_messages[1][1]["role"] == "assistant"
        assert call_messages[1][1]["content"] == bad_json_1
        assert call_messages[1][2]["role"] == "user"
        assert "rejected" in call_messages[1][2]["content"]
        assert "corrected JSON" in call_messages[1][2]["content"]

        # Third call: original + first bad + first fix + second bad + second fix
        assert len(call_messages[2]) == 5
        assert call_messages[2][0] == {"role": "user", "content": "original request"}
        assert call_messages[2][1]["role"] == "assistant"
        assert call_messages[2][1]["content"] == bad_json_1
        assert call_messages[2][2]["role"] == "user"
        assert "rejected" in call_messages[2][2]["content"]
        assert call_messages[2][3]["role"] == "assistant"
        assert call_messages[2][3]["content"] == bad_json_2
        assert call_messages[2][4]["role"] == "user"
        assert "rejected" in call_messages[2][4]["content"]

    @pytest.mark.asyncio
    async def test_reraise_last_exception_after_exhaustion(self):
        """Three consecutive bad responses cause the original exception type to be re-raised."""
        bad_json = '{broken'

        completion_fn = AsyncMock(return_value={
            "choices": [
                {"message": {"content": bad_json}}
            ]
        })

        def parse(raw: str) -> dict:
            return json.loads(raw)

        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
        }

        with pytest.raises(json.JSONDecodeError):
            await complete_with_repair(completion_fn, request, parse, max_attempts=3)

        assert completion_fn.call_count == 3

    @pytest.mark.asyncio
    async def test_malformed_response_envelope_treatment(self):
        """Malformed response envelope (no content) treated as parse failure."""
        # First attempt: no content in response
        # Second attempt: malformed again
        # Third attempt: valid

        call_count = [0]

        async def completion_fn(**kwargs):
            call_count[0] += 1

            if call_count[0] <= 2:
                # Malformed response: missing content
                return {"choices": [{"message": {}}]}
            else:
                # Valid response
                return {
                    "choices": [
                        {"message": {"content": '{"result": "success"}'}}
                    ]
                }

        def parse(raw: str) -> dict:
            return json.loads(raw)

        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
        }

        result = await complete_with_repair(completion_fn, request, parse, max_attempts=3)

        assert result == {"result": "success"}
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_request_dict_not_mutated(self):
        """Original request dict and messages list are not mutated by repair loop."""
        original_messages = [{"role": "user", "content": "test"}]
        request = {
            "model": "test-model",
            "messages": original_messages,
            "temperature": 0,
        }

        completion_fn = AsyncMock(return_value={
            "choices": [
                {"message": {"content": '{"result": "ok"}'}}
            ]
        })

        def parse(raw: str) -> dict:
            return json.loads(raw)

        await complete_with_repair(completion_fn, request, parse)

        # Original request must not be mutated: same list object, same content
        assert request["messages"] is original_messages
        assert original_messages == [{"role": "user", "content": "test"}]

    @pytest.mark.asyncio
    async def test_non_value_error_from_parse_propagates_immediately(self):
        """A parse() that raises KeyError (not ValueError/JSONDecodeError) is not retried."""
        completion_fn = AsyncMock(return_value={
            "choices": [
                {"message": {"content": "irrelevant"}}
            ]
        })

        def parse(raw: str) -> dict:
            raise KeyError("missing field")

        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
        }

        with pytest.raises(KeyError):
            await complete_with_repair(completion_fn, request, parse, max_attempts=3)

        assert completion_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_type_error_from_parse_propagates_immediately(self):
        """A parse() that raises TypeError is not retried."""
        completion_fn = AsyncMock(return_value={
            "choices": [
                {"message": {"content": "irrelevant"}}
            ]
        })

        def parse(raw: str) -> dict:
            raise TypeError("bad type")

        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
        }

        with pytest.raises(TypeError):
            await complete_with_repair(completion_fn, request, parse, max_attempts=3)

        assert completion_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_completion_fn_exception_propagates_immediately(self):
        """An exception raised by completion_fn itself (e.g. network error) is not retried."""
        completion_fn = AsyncMock(side_effect=ConnectionError("network unreachable"))

        def parse(raw: str) -> dict:
            return json.loads(raw)

        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
        }

        with pytest.raises(ConnectionError, match="network unreachable"):
            await complete_with_repair(completion_fn, request, parse, max_attempts=3)

        assert completion_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_extract_content_fails_all_attempts(self):
        """_extract_content failing on every attempt exhausts the loop and re-raises ValueError.

        No assistant turn should be appended for content-less attempts (only user
        fix turns), and completion_fn must be called exactly max_attempts times.
        """
        call_messages = []

        async def completion_fn(**kwargs):
            call_messages.append(kwargs.get("messages"))
            # Always malformed: no content anywhere in the envelope
            return {"choices": [{"message": {}}]}

        def parse(raw: str) -> dict:
            return json.loads(raw)

        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
        }

        with pytest.raises(ValueError):
            await complete_with_repair(completion_fn, request, parse, max_attempts=3)

        assert len(call_messages) == 3

        # No assistant turns ever appended; only the original user message plus
        # accumulated user fix turns.
        for messages in call_messages:
            assert all(m["role"] != "assistant" for m in messages)

        # Last call should have accumulated 2 user fix turns on top of the original.
        assert len(call_messages[2]) == 3
        assert call_messages[2][0] == {"role": "user", "content": "test"}
        assert call_messages[2][1]["role"] == "user"
        assert call_messages[2][2]["role"] == "user"

    @pytest.mark.asyncio
    async def test_max_attempts_one_reraises_immediately(self):
        """max_attempts=1: a first-attempt parse failure re-raises without retry."""
        bad_json = '{broken'

        completion_fn = AsyncMock(return_value={
            "choices": [
                {"message": {"content": bad_json}}
            ]
        })

        def parse(raw: str) -> dict:
            return json.loads(raw)

        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
        }

        with pytest.raises(json.JSONDecodeError):
            await complete_with_repair(completion_fn, request, parse, max_attempts=1)

        assert completion_fn.call_count == 1


class TestCliCompleteWithRepair:
    """Test cli_complete_with_repair CLI repair loop."""

    @pytest.mark.asyncio
    async def test_success_on_first_run(self):
        """Successful parse on first run returns immediately."""
        good_output = '{"result": "success"}'

        run_fn = AsyncMock(return_value=good_output)
        resume_fn = AsyncMock()  # Should not be called

        def parse(raw: str) -> dict:
            return json.loads(raw)

        result = await cli_complete_with_repair(
            run_fn,
            resume_fn,
            parse,
            session_id="test-session",
        )

        assert result == {"result": "success"}
        assert run_fn.call_count == 1
        assert resume_fn.call_count == 0

    @pytest.mark.asyncio
    async def test_recovery_with_resume_fn(self):
        """Parse failure on first run, success on resume."""
        bad_output = '{broken'
        good_output = '{"result": "success"}'

        run_fn = AsyncMock(return_value=bad_output)
        resume_fn = AsyncMock(return_value=good_output)

        def parse(raw: str) -> dict:
            return json.loads(raw)

        result = await cli_complete_with_repair(
            run_fn,
            resume_fn,
            parse,
            session_id="test-session",
            max_attempts=2,
        )

        assert result == {"result": "success"}
        assert run_fn.call_count == 1
        assert resume_fn.call_count == 1

        # Verify resume_fn was called with correct session_id
        resume_call_args = resume_fn.call_args
        assert resume_call_args[0][0] == "test-session"
        # Check that fix_prompt is passed (second arg)
        assert "could not be parsed" in resume_call_args[0][1]

    @pytest.mark.asyncio
    async def test_cli_reraise_after_exhaustion(self):
        """Multiple failed parses cause exception re-raise."""
        bad_output = '{broken'

        run_fn = AsyncMock(return_value=bad_output)
        resume_fn = AsyncMock(return_value=bad_output)

        def parse(raw: str) -> dict:
            return json.loads(raw)

        with pytest.raises(json.JSONDecodeError):
            await cli_complete_with_repair(
                run_fn,
                resume_fn,
                parse,
                session_id="test-session",
                max_attempts=2,
            )

        assert run_fn.call_count == 1
        assert resume_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_cli_fix_prompt_redacts_secrets(self):
        """Fix prompt has secrets redacted before sending to resume_fn."""
        bad_output = 'error with sk-1234567890abcdefghij'
        good_output = '{"result": "ok"}'

        run_fn = AsyncMock(return_value=bad_output)
        resume_fn = AsyncMock(return_value=good_output)

        def parse(raw: str) -> dict:
            # Intentionally fail on bad output to trigger fix prompt
            # (the secret in raw is in the error message that gets redacted)
            if "sk-" in raw:
                raise ValueError(f"invalid output: {raw}")
            return json.loads(raw)

        result = await cli_complete_with_repair(
            run_fn,
            resume_fn,
            parse,
            session_id="test-session",
            max_attempts=2,
        )

        assert result == {"result": "ok"}

        # Check that fix_prompt does not contain the secret
        resume_call_args = resume_fn.call_args
        fix_prompt = resume_call_args[0][1]
        assert "sk-1234567890abcdefghij" not in fix_prompt
        assert "[REDACTED]" in fix_prompt

    @pytest.mark.asyncio
    async def test_cli_run_fn_exception_not_retried(self):
        """Exception from run_fn itself is not retried."""
        run_fn = AsyncMock(side_effect=RuntimeError("subprocess failed"))
        resume_fn = AsyncMock()

        def parse(raw: str) -> dict:
            return json.loads(raw)

        with pytest.raises(RuntimeError, match="subprocess failed"):
            await cli_complete_with_repair(
                run_fn,
                resume_fn,
                parse,
                session_id="test-session",
            )

        assert run_fn.call_count == 1
        assert resume_fn.call_count == 0

    @pytest.mark.asyncio
    async def test_cli_max_attempts_one_reraises_immediately(self):
        """max_attempts=1: a first-attempt parse failure re-raises without calling resume_fn."""
        bad_output = '{broken'

        run_fn = AsyncMock(return_value=bad_output)
        resume_fn = AsyncMock()

        def parse(raw: str) -> dict:
            return json.loads(raw)

        with pytest.raises(json.JSONDecodeError):
            await cli_complete_with_repair(
                run_fn,
                resume_fn,
                parse,
                session_id="test-session",
                max_attempts=1,
            )

        assert run_fn.call_count == 1
        assert resume_fn.call_count == 0
