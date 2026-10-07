"""Collector and decoder tests for CLI streaming."""
import asyncio
import json
import pytest
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from huddleroom.services.agent_response_stream import (
    AgentResponseInvocation,
    InvocationContext,
)
from huddleroom.services.cli_streaming import (
    CliDisplayUpdate,
    DecodedCliOutput,
    CollectedCliOutput,
    decoder_for_runtime,
    collect_cli_process,
    parse_jsonl_final,
    is_resume_not_found,
)


def context():
    return InvocationContext(
        uuid4(), "agent", "agent-1", "Agent", "cli_main", "task", "test-runtime"
    )


def publisher(events):
    async def publish(event):
        events.append(event)
    return publish


# Test 1: Claude decoder with assistant text
def test_claude_decoder_parses_assistant_text():
    """Test that Claude decoder extracts readable text from assistant message."""
    decoder = decoder_for_runtime("claude_code")
    updates = decoder.feed_stdout(
        b'{"type":"assistant","message":{"content":[{"type":"text","text":"Readable"}]}}\n'
    )
    assert updates == [CliDisplayUpdate(kind="output", text="Readable")]
    assert not any('{"type"' in (update.text or "") for update in updates)


# Test 2: Claude decoder with explicit thinking
def test_claude_decoder_parses_thinking():
    """Test that Claude decoder extracts thinking as reasoning update."""
    decoder = decoder_for_runtime("claude_code")
    updates = decoder.feed_stdout(
        b'{"type":"assistant","message":{"content":[{"type":"thinking","text":"I think"}]}}\n'
    )
    assert len(updates) == 1
    assert updates[0].kind == "reasoning"
    assert updates[0].text == "I think"


# Test 3: Claude decoder with tool use
def test_claude_decoder_parses_tool_use():
    """Test that Claude decoder emits tool_started for tool_use blocks."""
    decoder = decoder_for_runtime("claude_code")
    updates = decoder.feed_stdout(
        b'{"type":"tool_use","id":"tool-123","name":"bash","input":{"command":"ls"}}\n'
    )
    assert len(updates) == 1
    assert updates[0].kind == "tool_started"
    assert updates[0].tool_name == "bash"
    assert updates[0].value == {"command": "ls"}


# Test 4: Claude decoder with tool result
def test_claude_decoder_parses_tool_result():
    """Test that Claude decoder emits tool_finished for tool_result."""
    decoder = decoder_for_runtime("claude_code")
    updates = decoder.feed_stdout(
        b'{"type":"tool_result","id":"tool-123","content":"Output"}\n'
    )
    assert len(updates) == 1
    assert updates[0].kind == "tool_finished"
    assert updates[0].tool_name == "tool-123"
    assert updates[0].value == "Output"


# Test 5: Claude decoder ignores malformed lines
def test_claude_decoder_ignores_malformed_lines():
    """Test that malformed JSON lines produce no live updates."""
    decoder = decoder_for_runtime("claude_code")
    updates = decoder.feed_stdout(b'{"incomplete":')
    assert updates == []
    updates = decoder.feed_stdout(b'not json\n')
    assert updates == []


# Test 6: Plain runtime decoder (aider)
def test_plain_runtime_decoder_publishes_text():
    """Test that plain runtimes publish stdout as readable output."""
    decoder = decoder_for_runtime("aider")
    updates = decoder.feed_stdout(b"Hello\nWorld\n")
    assert len(updates) == 2
    assert updates[0] == CliDisplayUpdate(kind="output", text="Hello")
    assert updates[1] == CliDisplayUpdate(kind="output", text="World")


# Test 7: Plain runtime stderr
def test_plain_runtime_decoder_stderr():
    """Test that stderr uses stderr stream."""
    decoder = decoder_for_runtime("aider")
    updates = decoder.feed_stderr(b"Error\n")
    assert len(updates) == 1
    assert updates[0] == CliDisplayUpdate(kind="stderr", text="Error")


# Test 8: UTF-8 incremental decoding with chunks
def test_incremental_utf8_decoding():
    """Test that incomplete UTF-8 sequences are handled correctly."""
    decoder = decoder_for_runtime("aider")
    # UTF-8 for 🙂 is b'\xf0\x9f\x99\x82' - split across chunks
    updates = decoder.feed_stdout(b"Hi \xf0\x9f")
    # No newline yet, so no output
    assert len(updates) == 0
    # Next chunk completes the emoji and line
    updates = decoder.feed_stdout(b"\x99\x82 OK\n")
    assert len(updates) == 1
    assert "🙂" in updates[0].text
    assert updates[0].text == "Hi 🙂 OK"


# Test 9: Decoder finish with exact bytes
def test_decoder_finish_retains_exact_bytes():
    """Test that finish includes accumulated content in decoded output."""
    decoder = decoder_for_runtime("aider")
    stdout_bytes = b"Output line\n"
    stderr_bytes = b"Error line\n"

    decoder.feed_stdout(stdout_bytes)
    decoder.feed_stderr(stderr_bytes)

    result = decoder.finish(stdout_bytes, stderr_bytes)
    assert "Output line" in result.content
    assert "Error line" in result.content


# Test 10: Claude decoder with terminal result record
def test_claude_decoder_terminal_result_record():
    """Test that Claude decoder extracts content and session_id from terminal result record."""
    decoder = decoder_for_runtime("claude_code")
    stdout = b'{"type":"assistant","message":{"content":[{"type":"text","text":"Thinking"}]}}\n{"type":"result","result":"Final answer","session_id":"sess-123"}\n'
    result = decoder.finish(stdout, b"")
    assert result.content == "Final answer"
    assert result.session_id == "sess-123"


# Test 10b: Claude decoder with legacy one-shot envelope
def test_claude_decoder_legacy_envelope():
    """Test that Claude decoder accepts legacy one-shot result envelope."""
    decoder = decoder_for_runtime("claude_code")
    stdout = b'{"result":"Legacy answer","session_id":"sess-456"}'
    result = decoder.finish(stdout, b"")
    assert result.content == "Legacy answer"
    assert result.session_id == "sess-456"


# Test 10c: Claude runtime doesn't include stderr in content
def test_claude_runtime_no_stderr_in_content():
    """Test that claude_code finish excludes stderr from parsed content."""
    decoder = decoder_for_runtime("claude_code")
    stdout = b'{"type":"result","result":"Answer","session_id":"s1"}\n'
    # stderr is present but should not be in the parsed content
    result = decoder.finish(stdout, b"Diagnostic error\n")
    assert result.content == "Answer"
    assert "Diagnostic" not in result.content


# Test 11: Concurrent collection (integration)
@pytest.mark.asyncio
async def test_collect_cli_process(stream_monitors):
    """Test concurrent collection from subprocess."""
    # Create real asyncio StreamReaders
    stdout_reader = asyncio.StreamReader()
    stderr_reader = asyncio.StreamReader()

    # Mock subprocess with async stdout/stderr
    proc = MagicMock()
    proc.stdout = stdout_reader
    proc.stderr = stderr_reader
    proc.returncode = None

    # Mock wait to signal completion
    async def mock_wait():
        return 0

    proc.wait = mock_wait

    # Create invocation and call
    project_id = uuid4()
    ctx = InvocationContext(
        project_id, "agent", "agent-1", "Agent", "cli_main", "task", "codex"
    )
    events = []
    invocation = AgentResponseInvocation(ctx, publish=publisher(events))
    call = invocation.call(messages=[{"role": "user", "content": "test"}])

    # Feed some output to the streams in a concurrent task
    async def feed_streams():
        await asyncio.sleep(0.001)
        stdout_reader.feed_data(b"Hello\n")
        stderr_reader.feed_data(b"Warning\n")
        await asyncio.sleep(0.001)
        stdout_reader.feed_eof()
        stderr_reader.feed_eof()

    async with call:
        feed_task = asyncio.create_task(feed_streams())
        collected = await collect_cli_process(
            proc, runtime="codex", call=call
        )
        await feed_task

    assert collected.returncode == 0
    assert collected.stdout == b"Hello\n"
    assert collected.stderr == b"Warning\n"
    # Verify output was routed through call
    output_events = [e for e in events if e.event_type == "agent_response.output"]
    assert len(output_events) > 0


# Test 12: Tool finished result routed through collection
@pytest.mark.asyncio
async def test_tool_finished_routed_with_result(stream_monitors):
    """Test that tool_finished result payload is routed through call."""
    # Create real asyncio StreamReaders
    stdout_reader = asyncio.StreamReader()
    stderr_reader = asyncio.StreamReader()

    # Mock subprocess
    proc = MagicMock()
    proc.stdout = stdout_reader
    proc.stderr = stderr_reader

    async def mock_wait():
        return 0

    proc.wait = mock_wait

    # Create invocation and call
    project_id = uuid4()
    ctx = InvocationContext(
        project_id, "agent", "agent-1", "Agent", "cli_main", "task", "claude_code"
    )
    events = []
    invocation = AgentResponseInvocation(ctx, publish=publisher(events))
    call = invocation.call(messages=[{"role": "user", "content": "test"}])

    # Feed tool result with payload
    async def feed_streams():
        await asyncio.sleep(0.001)
        # Claude tool_result with actual content/result payload
        stdout_reader.feed_data(
            b'{"type":"tool_result","id":"bash-1","content":{"exit_code":0,"output":"Success"}}\n'
        )
        await asyncio.sleep(0.001)
        stdout_reader.feed_eof()
        stderr_reader.feed_eof()

    async with call:
        feed_task = asyncio.create_task(feed_streams())
        await collect_cli_process(
            proc, runtime="claude_code", call=call
        )
        await feed_task

    # Verify tool_finished was emitted with result payload
    tool_events = [e for e in events if e.event_type == "agent_response.tool_finished"]
    assert len(tool_events) == 1
    assert tool_events[0].payload["result"] == {"exit_code": 0, "output": "Success"}


# Test 13: Claude command uses stream-json and verbose
def test_cli_build_command_claude_uses_stream_json():
    """Test that _build_command uses --output-format stream-json and --verbose for claude."""
    from huddleroom.adapters.cli_adapter import CliAdapter
    from pathlib import Path

    cmd = CliAdapter._build_command(
        "claude_code", "test prompt", {}, Path("/tmp"), Path("/tmp/task.md")
    )
    assert cmd is not None
    assert "--verbose" in cmd
    assert "--output-format" in cmd
    idx = cmd.index("--output-format")
    assert cmd[idx + 1] == "stream-json"


# Test 14: Parse claude final with NDJSON and legacy envelope
def test_parse_claude_final_ndjson():
    """Test that parse_claude_final extracts result from NDJSON."""
    from huddleroom.services.cli_streaming import parse_claude_final

    ndjson = '{"type":"assistant","message":{"content":[]}}\n{"type":"result","result":"Answer","session_id":"s1"}\n'
    result, session_id = parse_claude_final(ndjson)
    assert result == "Answer"
    assert session_id == "s1"


def test_parse_claude_final_legacy_envelope():
    """Test that parse_claude_final still accepts legacy one-shot envelope."""
    from huddleroom.services.cli_streaming import parse_claude_final

    envelope = '{"result":"Legacy","session_id":"s2"}'
    result, session_id = parse_claude_final(envelope)
    assert result == "Legacy"
    assert session_id == "s2"


def test_parse_claude_final_raises_on_empty():
    """Test that parse_claude_final raises ValueError on empty output."""
    from huddleroom.services.cli_streaming import parse_claude_final

    with pytest.raises(ValueError):
        parse_claude_final("")


def test_parse_claude_final_raises_on_no_envelope():
    """Test that parse_claude_final raises ValueError when no valid envelope found."""
    from huddleroom.services.cli_streaming import parse_claude_final

    with pytest.raises(ValueError):
        parse_claude_final("just some text")


@pytest.mark.parametrize(
    ("runtime", "raw", "expected_updates", "expected_final"),
    [
        (
            "codex",
            "\n".join((
                '{"type":"thread.started","thread_id":"01a1152c-9d95-7910-b2e5-7ee397bb85b5"}',
                '{"type":"item.completed","item":{"id":"item_0","type":"error","message":"Codex is ignoring 2 unrecognized configuration settings."}}',
                '{"type":"turn.started"}',
                '{"type":"item.completed","item":{"id":"item_2","type":"agent_message","text":"hi"}}',
                '{"type":"turn.completed","usage":{"input_tokens":18746,"cached_input_tokens":13184,"cache_write_input_tokens":0,"output_tokens":5,"reasoning_output_tokens":0}}',
            )),
            [CliDisplayUpdate(kind="output", text="hi")],
            DecodedCliOutput("hi", "01a1152c-9d95-7910-b2e5-7ee397bb85b5"),
        ),
        (
            "copilot",
            "\n".join((
                '{"type":"session.auto_mode_resolved","data":{"chosenModel":"gpt-6"}}',
                '{"type":"user.message","data":{"content":"secret task prompt"}}',
                '{"type":"assistant.message_delta","data":{"deltaContent":"answer"}}',
                '{"type":"assistant.message","data":{"content":"answer"}}',
                '{"type":"result","sessionId":"copilot-session"}',
            )),
            [CliDisplayUpdate(kind="output", text="answer")],
            DecodedCliOutput("answer", "copilot-session"),
        ),
        (
            "opencode",
            "\n".join((
                '{"type":"step_start","sessionID":"opencode-session","part":{"type":"step-start"}}',
                '{"type":"text","sessionID":"opencode-session","part":{"type":"text","text":"answer"}}',
                '{"type":"step_finish","sessionID":"opencode-session","part":{"type":"step-finish"}}',
            )),
            [CliDisplayUpdate(kind="output", text="answer")],
            DecodedCliOutput("answer", "opencode-session"),
        ),
        (
            "pi",
            "\n".join((
                '{"type":"session","id":"pi-session","cwd":"/tmp/workspace"}',
                '{"type":"message_start","message":{"role":"user","content":[{"type":"text","text":"secret task prompt"}]}}',
                '{"type":"message_update","assistantMessageEvent":{"type":"thinking_delta","delta":"private thought"}}',
                '{"type":"tool_result","assistantMessageEvent":{"type":"text_delta","delta":"secret tool output"}}',
                '{"type":"message_update","assistantMessageEvent":{"type":"text_delta","delta":"answer"}}',
                '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"answer"}]}}',
                '{"type":"agent_end","messages":[{"role":"assistant","content":[{"type":"text","text":"answer"}]}]}',
            )),
            [CliDisplayUpdate(kind="output", text="answer")],
            DecodedCliOutput("answer", "pi-session"),
        ),
    ],
)
def test_step8_json_runtime_streams_only_assistant_text(
    runtime, raw, expected_updates, expected_final,
):
    """Live provider schemas never expose prompts, tool output, or progress as agent output."""
    decoder = decoder_for_runtime(runtime)

    assert decoder.feed_stdout(f"{raw}\n".encode()) == expected_updates
    assert decoder.finish(raw.encode(), b"") == expected_final


@pytest.mark.parametrize(
    ("runtime", "raw", "expected"),
    [
        (
            "codex",
            "\n".join((
                '{"type":"thread.started","thread_id":"01a1152c-9d95-7910-b2e5-7ee397bb85b5"}',
                '{"type":"item.completed","item":{"id":"item_0","type":"error","message":"Codex is ignoring 2 unrecognized configuration settings."}}',
                '{"type":"item.completed","item":{"id":"item_2","type":"agent_message","text":"hi"}}',
            )),
            ("hi", "01a1152c-9d95-7910-b2e5-7ee397bb85b5"),
        ),
        (
            "copilot",
            "\n".join((
                '{"type":"assistant.message","data":{"content":"answer"}}',
                '{"type":"result","sessionId":"copilot-session"}',
            )),
            ("answer", "copilot-session"),
        ),
        (
            "opencode",
            '{"type":"text","sessionID":"opencode-session","part":{"type":"text","text":"answer"}}',
            ("answer", "opencode-session"),
        ),
        (
            "pi",
            "\n".join((
                '{"type":"session","id":"pi-session"}',
                '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"answer"}]}}',
            )),
            ("answer", "pi-session"),
        ),
    ],
)
def test_step8_jsonl_final_reads_live_provider_schemas(runtime, raw, expected):
    assert parse_jsonl_final(raw, runtime) == expected


# Test 15: Meeting request display contains only agenda
def test_build_meeting_cli_request_display():
    """Test that meeting request display contains only agenda + question, no context."""
    from huddleroom.adapters.cli_adapter import CliAdapter

    display = CliAdapter._build_meeting_cli_request_display(
        "Discuss performance", "What are key metrics?"
    )

    assert display is not None
    assert display["agenda"] == "Discuss performance"
    assert display["question"] == "What are key metrics?"
    # Verify it's minimal (no transcript, system prompt, etc.)
    assert len(display) == 2


# Test 16-25: Meeting seam census - drive each real service method with a mocked
# streaming LLM boundary and assert the emitted agent_response.* events carry the
# correct actor identity, operation, and (started-only) request_display.
#
# Mechanics (see module docstring context in the plan for rationale):
#  1. Event capture: patch huddleroom.services.agent_response_relay.publish_agent_response
#     with an async collector. Foundation's publish_agent_response lazily re-imports
#     this symbol on every call, so patching it here is observed.
#  2. Streaming mock: patch the service's litellm.acompletion to return an
#     async-iterable "Stream" whose chunks carry delta.content, and patch
#     huddleroom.services.agent_response_stream.litellm.stream_chunk_builder to return the
#     reconstructed response object the service actually parses.


class _Stream:
    """Minimal async-iterable litellm streaming response with one text chunk."""

    def __init__(self, payload: str):
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


def _capture():
    """Return (captured_events_list, async collector fn) for publish_agent_response."""
    captured = []

    async def collect(event):
        captured.append(event)

    return captured, collect


@contextmanager
def _meeting_llm_mock(acompletion_target: str, payload: str, collector):
    """Patch a service's litellm.acompletion to stream `payload`, wire the
    foundation's stream reconstruction to rebuild it, and route emitted
    agent_response events to `collector`."""

    async def mock_acompletion(**_kwargs):
        return _Stream(payload)

    def mock_builder(chunks, messages):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=payload, reasoning_content=None),
                    finish_reason="stop"
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        )

    with patch(acompletion_target, new=mock_acompletion), \
         patch("huddleroom.services.agent_response_stream.litellm.stream_chunk_builder", new=mock_builder), \
         patch("huddleroom.services.agent_response_relay.publish_agent_response", new=collector):
        yield


def _started_event(captured):
    return next(e for e in captured if e.event_type == "agent_response.started")


# --- Orchestrator control-plane seams (huddleroom/services/meeting_intelligence.py) ---


@pytest.mark.asyncio
async def test_meeting_seam_consensus_orchestrator(stream_monitors):
    """check_consensus: (system, orchestrator) / meeting_consensus."""
    from huddleroom.services.meeting_intelligence import MeetingIntelligenceService

    project_id = uuid4()
    payload = json.dumps({
        "consensus": False, "agreed_position": None, "confidence": 0.0,
        "dissenting_speakers": [], "rationale": "test",
    })
    captured, collector = _capture()

    with _meeting_llm_mock("huddleroom.services.meeting_intelligence.litellm.acompletion", payload, collector):
        svc = MeetingIntelligenceService()
        result = await svc.check_consensus(
            item_title="Test Item",
            item_question="Test?",
            item_options=["A", "B"],
            expected_speakers=["Speaker1"],
            turns_this_round=[{"speaker": "Speaker1", "content": "I like A"}],
            prior_rounds_summary=None,
            project_id=project_id,
        )

    assert result["consensus"] is False
    assert captured, "consensus seam must emit agent_response events"
    started = _started_event(captured)
    assert started.actor_kind == "system"
    assert started.actor_id == "orchestrator"
    assert started.operation == "meeting_consensus"
    assert started.payload["request_display"]["content"] == {
        "operation": "meeting_consensus", "title": "Test Item",
    }
    assert all(e.operation == "meeting_consensus" for e in captured)
    non_started = [e for e in captured if e.event_type != "agent_response.started"]
    assert non_started, "consensus seam must emit output/terminal events too"
    assert all("request_display" not in e.payload for e in non_started)


@pytest.mark.asyncio
async def test_meeting_seam_positions_orchestrator(stream_monitors):
    """extract_positions: (system, orchestrator) / meeting_positions."""
    from huddleroom.services.meeting_intelligence import MeetingIntelligenceService

    project_id = uuid4()
    payload = json.dumps([{"speaker": "Speaker1", "option": "A", "summary": "likes A"}])
    captured, collector = _capture()

    with _meeting_llm_mock("huddleroom.services.meeting_intelligence.litellm.acompletion", payload, collector):
        svc = MeetingIntelligenceService()
        result = await svc.extract_positions(
            turns=[{"speaker": "Speaker1", "content": "I like A"}],
            item_options=["A", "B"],
            project_id=project_id,
        )

    assert result == [{"speaker": "Speaker1", "option": "A", "summary": "likes A"}]
    started = _started_event(captured)
    assert started.actor_kind == "system"
    assert started.actor_id == "orchestrator"
    assert started.operation == "meeting_positions"
    assert started.payload["request_display"]["content"] == {"operation": "meeting_positions"}
    assert all(e.operation == "meeting_positions" for e in captured)


@pytest.mark.asyncio
async def test_meeting_seam_speaker_selection_orchestrator(stream_monitors):
    """select_next_speaker: (system, orchestrator) / meeting_speaker_selection."""
    from huddleroom.services.meeting_intelligence import MeetingIntelligenceService

    project_id = uuid4()
    speaker_id = str(uuid4())
    payload = json.dumps({
        "next_speaker_id": speaker_id, "reason": "next up",
        "close_item": False, "close_reason": None,
    })
    captured, collector = _capture()

    with _meeting_llm_mock("huddleroom.services.meeting_intelligence.litellm.acompletion", payload, collector):
        svc = MeetingIntelligenceService()
        result = await svc.select_next_speaker(
            participant_ids=[speaker_id],
            participant_names={speaker_id: "Agent A"},
            transcript_excerpt="",
            item_title="Test Item",
            item_options=["A", "B"],
            project_id=project_id,
        )

    assert result["next_speaker_id"] == speaker_id
    started = _started_event(captured)
    assert started.actor_kind == "system"
    assert started.actor_id == "orchestrator"
    assert started.operation == "meeting_speaker_selection"
    assert started.payload["request_display"]["content"] == {
        "operation": "meeting_speaker_selection", "title": "Test Item",
    }
    assert all(e.operation == "meeting_speaker_selection" for e in captured)


# --- Outcome seams (huddleroom/services/meeting_outcome.py), DB mocked as in
# tests/test_meeting_outcome_prompts.py ---


@pytest.mark.asyncio
async def test_meeting_seam_action_items_orchestrator(stream_monitors):
    """extract_action_items: (system, orchestrator) / meeting_action_items."""
    from huddleroom.services.meeting_outcome import MeetingOutcomeService

    meeting = MagicMock()
    meeting.id = uuid4()
    meeting.project_id = uuid4()
    meeting.title = "Outcome Test Meeting"
    meeting.meeting_type = "decision"
    meeting.participant_agent_ids = []

    payload = json.dumps([])
    captured, collector = _capture()

    mock_turn = MagicMock()
    mock_turn.speaker_agent_id = None
    mock_turn.content = "We agreed on REST."

    db = AsyncMock()
    turns_result = MagicMock()
    turns_result.scalars.return_value.all.return_value = [mock_turn]
    db.execute.return_value = turns_result
    db.get.return_value = None
    db.add = MagicMock()
    db.flush = AsyncMock()

    with _meeting_llm_mock("huddleroom.services.meeting_outcome.litellm.acompletion", payload, collector):
        svc = MeetingOutcomeService()
        await svc.extract_action_items(db=db, meeting=meeting, decisions=[])

    started = _started_event(captured)
    assert started.actor_kind == "system"
    assert started.actor_id == "orchestrator"
    assert started.operation == "meeting_action_items"
    assert started.payload["request_display"]["content"] == {
        "operation": "meeting_action_items", "title": "Outcome Test Meeting",
    }
    assert all(e.operation == "meeting_action_items" for e in captured)


@pytest.mark.asyncio
async def test_meeting_seam_summary_orchestrator(stream_monitors):
    """write_knowledge_items summary call: (system, orchestrator) / meeting_summary."""
    from huddleroom.services.meeting_outcome import MeetingOutcomeService

    meeting = MagicMock()
    meeting.id = uuid4()
    meeting.project_id = uuid4()
    meeting.title = "Outcome Test Meeting"
    meeting.meeting_type = "decision"

    payload = (
        "## Outcome\nDone.\n## Decisions\nNone.\n"
        "## Dissent and Open Questions\nNone.\n## Action Items\nNone."
    )
    captured, collector = _capture()

    db = AsyncMock()
    db.get.return_value = None
    db.add = MagicMock()
    db.flush = AsyncMock()

    with _meeting_llm_mock("huddleroom.services.meeting_outcome.litellm.acompletion", payload, collector):
        svc = MeetingOutcomeService()
        await svc.write_knowledge_items(
            db=db, meeting=meeting, decisions=[], transcript="Alice: We chose REST."
        )

    assert meeting.summary == payload
    started = _started_event(captured)
    assert started.actor_kind == "system"
    assert started.actor_id == "orchestrator"
    assert started.operation == "meeting_summary"
    assert started.payload["request_display"]["content"] == {
        "operation": "meeting_summary", "title": "Outcome Test Meeting",
    }
    assert all(e.operation == "meeting_summary" for e in captured)


@pytest.mark.asyncio
async def test_meeting_seam_planner_summary_agent(stream_monitors):
    """run_planner_summary: planner agent identity / meeting_planner_summary."""
    from huddleroom.services.meeting_outcome import MeetingOutcomeService

    meeting = MagicMock()
    meeting.id = uuid4()
    meeting.project_id = uuid4()
    meeting.title = "Outcome Test Meeting"
    meeting.meeting_type = "decision"

    planner = MagicMock()
    planner.id = uuid4()
    planner.name = "PlannerAgent"
    planner.role = "planner"
    planner.system_prompt = "Plan things."
    planner.provider = "openai"
    planner.model = "gpt-4o-mini"
    meeting.planner_agent_id = planner.id

    payload = "DECISIONS\nACTIONS\nOPEN_QUESTIONS"
    captured, collector = _capture()

    db = AsyncMock()

    async def fake_get(_model, pk):
        return planner if str(pk) == str(planner.id) else None

    db.get.side_effect = fake_get
    db.add = MagicMock()
    db.flush = AsyncMock()

    with _meeting_llm_mock("huddleroom.services.meeting_outcome.litellm.acompletion", payload, collector):
        svc = MeetingOutcomeService()
        summary = await svc.run_planner_summary(
            db=db, meeting=meeting, decisions=[], transcript="Alice: hello."
        )

    assert summary == payload
    started = _started_event(captured)
    assert started.actor_kind == "agent"
    assert started.actor_id == str(planner.id)
    assert started.operation == "meeting_planner_summary"
    assert started.payload["request_display"]["content"] == {
        "operation": "meeting_planner_summary", "title": "Outcome Test Meeting",
    }
    assert all(e.operation == "meeting_planner_summary" for e in captured)


@pytest.mark.asyncio
async def test_meeting_seam_final_review_orchestrator(db_session, test_project, test_agent, stream_monitors):
    """ask_final_reviewer, reviewer_kind=orchestrator: (system, orchestrator) / meeting_final_review."""
    from huddleroom.services.meeting_outcome import MeetingOutcomeService
    from huddleroom.services.meeting_service import MeetingService

    meeting = await MeetingService().create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Final Review Meeting",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "Cache", "max_rounds": 1}],
        auto_start=False,
    )
    pending = SimpleNamespace(payload={"reviewer_kind": "orchestrator", "suggested_action_items": []})
    payload = json.dumps({
        "decisions_made": True, "decisions_clear": True,
        "action_items_needed": False, "action_items": [],
    })
    captured, collector = _capture()

    with _meeting_llm_mock("huddleroom.services.meeting_outcome.litellm.acompletion", payload, collector):
        svc = MeetingOutcomeService()
        result = await svc.ask_final_reviewer(db=db_session, meeting=meeting, pending=pending)

    assert result == (True, True, False, [])
    started = _started_event(captured)
    assert started.actor_kind == "system"
    assert started.actor_id == "orchestrator"
    assert started.operation == "meeting_final_review"
    assert started.payload["request_display"]["content"] == {
        "operation": "meeting_final_review", "title": "Final Review Meeting",
    }
    assert all(e.operation == "meeting_final_review" for e in captured)


@pytest.mark.asyncio
async def test_meeting_seam_final_review_agent(db_session, test_project, test_agent, stream_monitors):
    """ask_final_reviewer, reviewer_kind=organizer_agent: selected agent / meeting_final_review."""
    from huddleroom.services.meeting_outcome import MeetingOutcomeService
    from huddleroom.services.meeting_service import MeetingService

    meeting = await MeetingService().create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Final Review Meeting Agent",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        organizer_agent_id=test_agent.id,
        agenda_items=[{"order": 1, "title": "Cache", "max_rounds": 1}],
        auto_start=False,
    )
    pending = SimpleNamespace(payload={
        "reviewer_kind": "organizer_agent",
        "reviewer_id": str(test_agent.id),
        "suggested_action_items": [],
    })
    payload = json.dumps({
        "decisions_made": True, "decisions_clear": True,
        "action_items_needed": False, "action_items": [],
    })
    captured, collector = _capture()

    with _meeting_llm_mock("huddleroom.services.meeting_outcome.litellm.acompletion", payload, collector):
        svc = MeetingOutcomeService()
        result = await svc.ask_final_reviewer(db=db_session, meeting=meeting, pending=pending)

    assert result == (True, True, False, [])
    started = _started_event(captured)
    assert started.actor_kind == "agent"
    assert started.actor_id == str(test_agent.id)
    assert started.operation == "meeting_final_review"
    assert started.payload["request_display"]["content"] == {
        "operation": "meeting_final_review", "title": "Final Review Meeting Agent",
    }
    assert all(e.operation == "meeting_final_review" for e in captured)


# --- Runner seams (huddleroom/services/meeting_runner.py), real DB fixtures as in
# tests/test_meeting_verbose_tracing.py ---


@pytest.mark.asyncio
async def test_meeting_seam_participant_turn_agent(db_session, test_project, test_agent, stream_monitors):
    """execute_agent_turn: participant agent identity / meeting_turn."""
    from huddleroom.services.meeting_runner import MeetingRunner
    from huddleroom.services.meeting_service import MeetingService

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Participant Turn Meeting",
        meeting_type="review",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{
            "order": 1, "title": "Discuss performance",
            "question": "What are key metrics?", "max_rounds": 1,
        }],
        auto_start=False,
    )
    await svc.transition_to_preparing(db=db_session, meeting=meeting)
    await svc.transition_to_active(db=db_session, meeting=meeting)

    payload = "Solid response from the participant."
    captured, collector = _capture()

    runner = MeetingRunner()
    with _meeting_llm_mock("huddleroom.services.meeting_runner.litellm.acompletion", payload, collector):
        turn = await runner.execute_agent_turn(db=db_session, meeting=meeting, agent=test_agent)

    assert turn is not None, "execute_agent_turn should return a MeetingTurn"
    assert turn.content == payload
    started = _started_event(captured)
    assert started.actor_kind == "agent"
    assert started.actor_id == str(test_agent.id)
    assert started.operation == "meeting_turn"
    assert started.payload["request_display"]["content"] == {
        "agenda": "Discuss performance", "question": "What are key metrics?",
    }
    assert all(e.operation == "meeting_turn" for e in captured)
    output_events = [e for e in captured if e.event_type == "agent_response.output"]
    assert output_events and output_events[0].payload["text"] == payload
    terminal_events = [e for e in captured if e.event_type == "agent_response.terminal"]
    assert terminal_events and terminal_events[0].payload["status"] == "completed"


@pytest.mark.asyncio
async def test_meeting_seam_signal_probe_agent(db_session, test_project, test_agent, stream_monitors):
    """_probe_for_signals: probed agent identity / meeting_signal_probe."""
    from huddleroom.models.agent import Agent
    from huddleroom.services.meeting_runner import MeetingRunner
    from huddleroom.services.meeting_service import MeetingService

    probed_agent = Agent(
        name=f"probe-agent-{uuid4()}",
        role="developer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=[],
        config={},
    )
    db_session.add(probed_agent)
    await db_session.flush()

    meeting = await MeetingService().create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Signal Probe Meeting",
        meeting_type="review",
        participant_agent_ids=[str(test_agent.id), str(probed_agent.id)],
        agenda_items=[{"order": 1, "title": "Discuss", "max_rounds": 1}],
        auto_start=False,
    )

    payload = "NO"
    captured, collector = _capture()

    runner = MeetingRunner()
    turn = SimpleNamespace(content="I think we should ship it.")
    with _meeting_llm_mock("huddleroom.services.meeting_runner.litellm.acompletion", payload, collector):
        await runner._probe_for_signals(
            db=db_session, meeting=meeting, speaking_agent_id=test_agent.id, turn=turn,
        )

    started = _started_event(captured)
    assert started.actor_kind == "agent"
    assert started.actor_id == str(probed_agent.id)
    assert started.operation == "meeting_signal_probe"
    assert started.payload["request_display"]["content"] == {
        "operation": "Decide whether to contribute",
    }
    assert all(e.operation == "meeting_signal_probe" for e in captured)


# Tests for is_resume_not_found

def test_is_resume_not_found_claude_not_found_in_stderr():
    """claude_code: found not-found marker in stderr → True."""
    assert is_resume_not_found(
        "claude_code",
        1,
        "",
        "No conversation found with session ID: 11111111-2222-3333-4444-555555555555"
    ) is True


def test_is_resume_not_found_claude_result_record_errors():
    """claude_code: terminal stream-json result record listing the marker in errors → True."""
    stdout = ('{"type":"result","subtype":"error_during_execution","is_error":true,'
              '"errors":["No conversation found with session ID: 1111"]}\n')
    assert is_resume_not_found("claude_code", 1, stdout, "") is True


def test_is_resume_not_found_claude_ignores_assistant_text_in_stdout():
    stdout = ('{"type":"assistant","message":{"content":[{"type":"text",'
              '"text":"No conversation found with session ID: x"}]}}\n'
              'No conversation found with session ID: raw\n')
    assert is_resume_not_found("claude_code", 1, stdout, "") is False


def test_is_resume_not_found_codex_not_found_in_stderr():
    """codex: found not-found marker in stderr → True."""
    assert is_resume_not_found(
        "codex",
        1,
        "",
        "Error: thread/resume: thread/resume failed: no rollout found for thread id 00000000-0000-0000-0000-000000000000 (code -32600)"
    ) is True


def test_is_resume_not_found_codex_stdout_ignored():
    """codex: marker only counts on stderr."""
    assert not is_resume_not_found(
        "codex",
        1,
        "Error: thread/resume: thread/resume failed: no rollout found for thread id 00000000-0000-0000-0000-000000000000 (code -32600)",
        ""
    )


def test_is_resume_not_found_success_exit():
    """exit_code=0 → False (resume succeeded)."""
    assert is_resume_not_found("claude_code", 0, "", "") is False
    assert is_resume_not_found("codex", 0, "", "") is False


def test_is_resume_not_found_none_exit():
    """exit_code=None → False."""
    assert is_resume_not_found("claude_code", None, "", "") is False
    assert is_resume_not_found("codex", None, "", "") is False


def test_is_resume_not_found_other_error():
    """exit_code != 0 but no not-found marker → False."""
    assert is_resume_not_found(
        "claude_code",
        1,
        "some other error",
        "more errors"
    ) is False
    assert is_resume_not_found(
        "codex",
        1,
        "some other error",
        "more errors"
    ) is False


def test_is_resume_not_found_unknown_runtime():
    """unknown runtime → False."""
    assert is_resume_not_found(
        "unknown_runtime",
        1,
        "no rollout found for thread id abc",
        ""
    ) is False


def test_parse_jsonl_final_codex_returns_last_agent_message_only():
    raw = (
        '{"type":"thread.started","thread_id":"t1"}\n'
        '{"type":"item.completed","item":{"type":"agent_message","text":"Let me look."}}\n'
        '{"type":"item.completed","item":{"type":"agent_message","text":"{\\"ok\\": true}"}}\n'
    )
    assert parse_jsonl_final(raw, "codex") == ('{"ok": true}', "t1")
