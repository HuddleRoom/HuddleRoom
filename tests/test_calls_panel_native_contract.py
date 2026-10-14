"""Provider-free Calls regressions using Codex exec 0.162.1 item events.

These synthetic fixtures model completed public summary/message and tool events;
they do not claim token streaming or exercise a live provider.
"""
import asyncio
import json
from uuid import uuid4

import pytest

from huddleroom.services.agent_response_stream import AgentResponseInvocation, InvocationContext
from huddleroom.services.cli_streaming import collect_cli_process, decoder_for_runtime, parse_jsonl_final


def wire(*records):
    return ("\n".join(json.dumps(record) for record in records) + "\n").encode()


def item(event, identifier, kind, **fields):
    return {"type": event, "item": {"id": identifier, "type": kind, **fields}}


class ControlledProcess:
    def __init__(self, data, returncode=0):
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stdout.feed_data(data)
        self.returncode = None
        self.exit_code = returncode
        self.released = asyncio.Event()

    async def wait(self):
        await self.released.wait()
        return self.returncode

    def finish(self):
        self.returncode = self.exit_code
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        self.released.set()


def invocation(events):
    async def publish(event):
        events.append(event)
    return AgentResponseInvocation(
        InvocationContext(uuid4(), "agent", "fixture-agent", "Team Lead", "cli_main", "task", "codex"),
        publish=publish,
    )


@pytest.mark.parametrize("kind,fields", [
    ("mcp_tool_call", {"server": "fixtures", "tool": "lookup", "arguments": {"topic": "events"}, "result": {"ok": True}}),
    ("web_search", {"query": "documented event contract"}),
    ("file_change", {"changes": [{"path": "fixture.txt", "kind": "update"}]}),
])
def test_supported_codex_tool_lifecycle_variants_deduplicate(kind, fields):
    decoder = decoder_for_runtime("codex")
    started = item("item.started", "tool", kind, status="in_progress", **fields)
    finished = item("item.completed", "tool", kind, status="failed", **fields)
    updates = decoder.feed_stdout(wire(started, started, finished, finished))
    assert [update.kind for update in updates] == ["tool_started", "tool_finished"]
    assert updates[-1].outcome == "failed"


def test_empty_completed_final_does_not_promote_commentary():
    raw = wire(item("item.completed", "c", "agent_message", text="Checking contracts."),
               item("item.completed", "f", "agent_message", text="")).decode()
    with pytest.raises(ValueError):
        parse_jsonl_final(raw, "codex")


def test_unknown_and_hidden_reasoning_events_are_inert():
    decoder = decoder_for_runtime("codex")
    assert decoder.feed_stdout(wire(
        item("item.started", "r", "reasoning", text="not a completed public summary"),
        {"type": "response.reasoning.delta", "delta": "hidden raw internals"},
        {"type": "unknown", "text": "unrecognized data"},
    )) == []


@pytest.mark.asyncio
async def test_orchestration_observer_routes_public_progress_before_process_exit(monkeypatch, stream_monitors):
    from huddleroom.services import orchestration_completion as completion
    proc = ControlledProcess(wire(
        item("item.completed", "r", "reasoning", text="Checking completed public summaries."),
        item("item.started", "t", "web_search", query="public contract", status="in_progress"),
        item("item.completed", "f", "agent_message", text='{"answer":"ready"}'),
    ))
    class Stdin:
        def close(self):
            pass
    proc.stdin = Stdin()
    async def create(*_args, **_kwargs):
        return proc
    monkeypatch.setattr(completion.asyncio, "create_subprocess_exec", create)
    events = []
    async with invocation(events).call(messages=[]) as call:
        task = asyncio.create_task(completion._run(("codex",), None, backend="codex", observer=call.output))
        try:
            async with asyncio.timeout(2):
                while len(events) < 4:
                    await asyncio.sleep(0.001)
            assert proc.returncode is None and not task.done()
            assert [event.payload.get("message_id") for event in events if event.event_type == "agent_response.output"] == ['r', 'f']
            assert any(event.event_type == "agent_response.tool_started" for event in events)
            proc.finish()
            await task
        finally:
            if not proc.released.is_set():
                proc.finish()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_completed_progress_boundaries_and_final_snapshot_precede_terminal(stream_monitors):
    final = json.dumps({"schema_version": 1, "plan_kind": "implementation", "summary": "Use existing events"})
    reasoning = item("item.completed", "r", "reasoning", text="Checking the supported event contract.")
    tool_start = item("item.started", "t", "command_execution", command="python -V", status="in_progress")
    tool_finish = item("item.completed", "t", "command_execution", command="python -V",
                       aggregated_output="Python 3.12", exit_code=0, status="completed")
    commentary = item("item.completed", "c", "agent_message", text="I will check the existing collector.")
    result = item("item.completed", "f", "agent_message", text=final)
    proc = ControlledProcess(wire(reasoning, tool_start, tool_start, tool_finish, tool_finish,
                                  commentary, result, result))
    events = []
    async with invocation(events).call(messages=[]) as call:
        task = asyncio.create_task(collect_cli_process(proc, runtime="codex", call=call))
        try:
            async with asyncio.timeout(2):
                while len(events) < 6:
                    await asyncio.sleep(0.001)
            assert proc.returncode is None
            assert not task.done()
            assert not any(event.event_type == "agent_response.terminal" for event in events)
            output = [event.payload for event in events if event.event_type == "agent_response.output"]
            assert [(payload["stream"], payload["message_id"], payload["message_complete"])
                    for payload in output] == [("reasoning", "r", True), ("output", "c", True), ("output", "f", True)]
            assert [payload["text"] for payload in output] == [reasoning["item"]["text"], commentary["item"]["text"], final]
            assert [event.event_type for event in events].count("agent_response.tool_started") == 1
            assert [event.event_type for event in events].count("agent_response.tool_finished") == 1
            proc.finish()
            assert (await task).decoded.content == final
            assert call.final_response.content == final
        finally:
            if not proc.released.is_set():
                proc.finish()
            await asyncio.gather(task, return_exceptions=True)
    assert events[-1].event_type == "agent_response.terminal"
    assert events[-1].payload["final_response"]["content"] == final


@pytest.mark.asyncio
async def test_failed_native_process_never_attaches_successful_final_snapshot(stream_monitors):
    proc = ControlledProcess(wire(item("item.completed", "f", "agent_message", text="Partial candidate")), returncode=1)
    events = []
    proc.finish()
    async with invocation(events).call(messages=[]) as call:
        result = await collect_cli_process(proc, runtime="codex", call=call)
        assert result.returncode == 1
        assert call.final_response is None
        await call.terminal("failed", error="Synthetic process failure")
    assert "final_response" not in events[-1].payload


@pytest.mark.asyncio
async def test_completed_message_redaction_is_independent_across_boundaries(stream_monitors):
    proc = ControlledProcess(wire(
        item("item.completed", "c", "agent_message", text='{"password":"first-secret"}'),
        item("item.completed", "f", "agent_message", text='{"password":"second-secret","summary":"Readable"}'),
    ))
    events = []
    proc.finish()
    async with invocation(events).call(messages=[]) as call:
        await collect_cli_process(proc, runtime="codex", call=call)
    visible = json.dumps([event.payload for event in events])
    assert "first-secret" not in visible and "second-secret" not in visible
    assert "[REDACTED]" in visible
    assert "Readable" in visible


@pytest.mark.asyncio
async def test_completed_messages_and_final_snapshot_are_bounded(stream_monitors):
    proc = ControlledProcess(wire(item("item.completed", "f", "agent_message", text="x" * 80_000)))
    proc.finish()
    events = []
    async with invocation(events).call(messages=[]) as call:
        await collect_cli_process(proc, runtime="codex", call=call)
    output = next(event.payload for event in events if event.event_type == "agent_response.output")
    terminal = events[-1].payload
    assert len(output["text"].encode()) <= 16_384
    assert any(event.payload.get("observation_truncated") is True for event in events)
    assert terminal["final_response"]["content_truncated"] is True
    assert len(terminal["final_response"]["content"].encode()) <= 16_384
