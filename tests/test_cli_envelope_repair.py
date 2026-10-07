"""Test CLI adapter JSON envelope parsing and auto-repair logic."""

import asyncio
import json
import pytest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from huddleroom.adapters.cli_adapter import CliAdapter
from huddleroom.models.session import Session
from huddleroom.services.llm_structured_repair import cli_complete_with_repair


def test_cli_terminal_usage_is_cumulative_across_resumes():
    """A CLI retry retains prior turn and elapsed usage before the next resume."""
    session = Session(metadata_={
        "_run_config": {
            "_roadmap_budget_enforced": True,
            "_roadmap_prior_usage": {"max_turns": "2", "max_hours": "1.5"},
        },
    })
    session.started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    session.ended_at = session.started_at + timedelta(minutes=30)

    CliAdapter._store_roadmap_elapsed(session)

    assert session.metadata_["_roadmap_turn_count"] == 3
    assert session.metadata_["_roadmap_elapsed_seconds"] == "7200"


@pytest.mark.asyncio
async def test_envelope_repair_success_on_retry():
    """Test that bad envelope twice then valid resolves via 2 resumes."""
    attempt_count = 0

    async def run_fn():
        """First attempt returns invalid JSON."""
        return "not json"

    async def resume_fn(session_id, fix_prompt):
        """Attempts 2 and 3: second is still bad, third is valid."""
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            # Second attempt: still bad
            return "still not json"
        else:
            # Third attempt: valid
            return json.dumps({"result": "success", "session_id": "new-session-123"})

    def parse_envelope(raw):
        """Parse JSON envelope or raise on invalid."""
        parsed = json.loads(raw)
        result = parsed.get("result", "")
        session_id = parsed.get("session_id")
        return (result, session_id)

    result, session_id = await cli_complete_with_repair(
        run_fn=run_fn,
        resume_fn=resume_fn,
        parse=parse_envelope,
        session_id="initial-session",
        max_attempts=3,
    )

    assert result == "success"
    assert session_id == "new-session-123"
    assert attempt_count == 2  # Two resumes called


@pytest.mark.asyncio
async def test_envelope_repair_exhaustion_fallback():
    """Test that 3 bad envelopes result in fallback to raw text (re-raises)."""
    call_count = 0

    async def run_fn():
        return "not json at all"

    async def resume_fn(session_id, fix_prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "still not json"
        else:
            return "nope, still bad"

    def parse_envelope(raw):
        parsed = json.loads(raw)  # Will raise JSONDecodeError
        return (parsed.get("result", ""), parsed.get("session_id"))

    # After 3 attempts with bad JSON, cli_complete_with_repair re-raises the last JSONDecodeError
    with pytest.raises(json.JSONDecodeError):
        await cli_complete_with_repair(
            run_fn=run_fn,
            resume_fn=resume_fn,
            parse=parse_envelope,
            session_id="session",
            max_attempts=3,
        )

    # Two resumes were attempted (attempt 2 and 3; first attempt was run_fn)
    assert call_count == 2


@pytest.mark.asyncio
async def test_envelope_repair_immediate_success():
    """Test that first attempt with valid JSON does not trigger resumes."""
    resume_called = False

    async def run_fn():
        return json.dumps({"result": "immediate success", "session_id": "sess-1"})

    async def resume_fn(session_id, fix_prompt):
        nonlocal resume_called
        resume_called = True
        return ""  # Should never be called

    def parse_envelope(raw):
        parsed = json.loads(raw)
        return (parsed.get("result", ""), parsed.get("session_id"))

    result, session_id = await cli_complete_with_repair(
        run_fn=run_fn,
        resume_fn=resume_fn,
        parse=parse_envelope,
        session_id="initial",
        max_attempts=3,
    )

    assert result == "immediate success"
    assert session_id == "sess-1"
    assert not resume_called


@pytest.mark.asyncio
async def test_cli_adapter_envelope_fallback_on_exhaustion():
    """Test that when repair exhausts, raw text fallback is used."""
    # This simulates the behavior in run() and run_meeting_turn()
    attempt_count = 0

    async def run_fn_initial():
        return "bad json"

    async def resume_fn_stub(session_id, fix_prompt):
        nonlocal attempt_count
        attempt_count += 1
        return "still bad json"

    def parse_envelope(raw):
        parsed = json.loads(raw)  # Will raise JSONDecodeError
        return (parsed.get("result", ""), parsed.get("session_id"))

    # Verify the exception is re-raised after exhaustion
    with pytest.raises(json.JSONDecodeError):
        await cli_complete_with_repair(
            run_fn=run_fn_initial,
            resume_fn=resume_fn_stub,
            parse=parse_envelope,
            session_id="session",
            max_attempts=3,
        )


@pytest.mark.asyncio
async def test_cli_adapter_resume_raw_basic():
    """Test that _resume_raw launches subprocess correctly."""
    adapter = CliAdapter()

    # Mock subprocess to return valid JSON
    async def mock_create_subprocess_exec(*args, **kwargs):
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(
            json.dumps({"result": "repaired", "session_id": "new-sid"}).encode("utf-8"),
            b"",
        ))
        proc.returncode = 0
        return proc

    # Patch the subprocess creation
    import huddleroom.adapters.cli_adapter
    original_create = huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec
    huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec = mock_create_subprocess_exec

    try:
        result = await adapter._resume_raw(
            "session-id",
            "fix prompt",
            cli_runtime="claude_code",
            agent_config={},
            workspace=None,
            task_path=None,
            model="claude-sonnet-5-5",
            cwd=".",
            env={},
            timeout=30,
        )

        parsed = json.loads(result)
        assert parsed["result"] == "repaired"
        assert parsed["session_id"] == "new-sid"
    finally:
        huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec = original_create


@pytest.mark.asyncio
async def test_cli_adapter_does_not_launch_repair_after_deadline(monkeypatch):
    """A repair attempt must not reset a session's expired absolute deadline."""
    launch = AsyncMock()
    monkeypatch.setattr("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", launch)
    monkeypatch.setattr("huddleroom.adapters.cli_adapter.time.monotonic", lambda: 10.0)

    with pytest.raises(RuntimeError, match="deadline"):
        await CliAdapter()._resume_raw(
            "session-id", "fix", cli_runtime="claude_code", agent_config={}, workspace=None,
            task_path=None, model=None, cwd=".", env={}, timeout=30, deadline=9.0,
        )

    launch.assert_not_awaited()


@pytest.mark.asyncio
async def test_cli_adapter_repairs_share_one_decreasing_deadline(monkeypatch):
    """Later JSON repairs receive only the time left from the original attempt."""
    clock = iter((0.0, 0.0, 1.0, 5.0, 5.0, 7.0))
    monkeypatch.setattr("huddleroom.adapters.cli_adapter.time", SimpleNamespace(monotonic=lambda: next(clock)))
    observed_timeouts = []
    real_wait_for = asyncio.wait_for

    async def observe_wait_for(awaitable, timeout):
        observed_timeouts.append(timeout)
        return await real_wait_for(awaitable, timeout)

    async def launch(*_args, **_kwargs):
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"{}", b""))
        return proc

    monkeypatch.setattr("huddleroom.adapters.cli_adapter.asyncio.wait_for", observe_wait_for)
    monkeypatch.setattr("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", launch)
    adapter = CliAdapter()
    for session_id in ("one", "two"):
        await adapter._resume_raw(
            session_id, "fix", cli_runtime="claude_code", agent_config={}, workspace=None,
            task_path=None, model=None, cwd=".", env={}, timeout=10, deadline=10.0,
        )

    assert observed_timeouts == [9.0, 3.0]
