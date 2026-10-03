import asyncio

# pylint: disable=unnecessary-dunder-call
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import litellm
import pytest
from pydantic import ValidationError
from unittest.mock import AsyncMock

from huddleroom.services.agent_response_stream import (
    AgentResponseEvent,
    AgentResponseInvocation,
    InvocationContext,
    extract_request_display,
)


async def _never_reset(_project_id):
    await asyncio.Event().wait()


@pytest.fixture
def stream_monitors(monkeypatch):
    """Keep physical-call tests local to their in-memory event publisher."""
    ack = SimpleNamespace(generation=None)
    monkeypatch.setattr(
        "huddleroom.services.agent_response_relay.register_project_reset_monitor",
        AsyncMock(return_value=ack),
    )
    monkeypatch.setattr(
        "huddleroom.services.agent_response_relay.unregister_project_reset_monitor",
        AsyncMock(),
    )
    monkeypatch.setattr(
        "huddleroom.services.agent_response_stream.wait_for_project_reset", _never_reset
    )


def context():
    return InvocationContext(
        uuid4(), "agent", "agent-1", "Agent", "api", "task", "test-model"
    )


def publisher(events):
    async def publish(event):
        events.append(event)

    return publish


def test_request_is_source_minimized_verbatim_and_bounded():
    display = extract_request_display(
        [
            {"role": "system", "content": "hidden"},
            {"role": "user", "content": "old"},
            {"role": "tool", "content": "context"},
            {
                "role": "user",
                "content": {"question": "Deploy?", "token": "secret-value"},
            },
        ]
    )
    assert display.content == {"question": "Deploy?", "token": "secret-value"}
    assert not display.truncated
    assert extract_request_display(
        [{"role": "user", "content": '{"token":"plaintext"}'}]
    ).content == {"token": "plaintext"}
    assert extract_request_display([{"role": "user", "content": "[1, 2]"}]).content == [
        1,
        2,
    ]
    assert extract_request_display([{"role": "user", "content": "42"}]).content == "42"
    bounded = extract_request_display([{"role": "user", "content": "🙂" * 5000}])
    assert bounded.truncated and len(bounded.content.encode()) <= 16 * 1024
    assert bounded.content.startswith('"')


@pytest.mark.asyncio
async def test_verbatim_tools_terminal_and_lossless_output():
    events = []
    call = AgentResponseInvocation(context(), publish=publisher(events)).call(
        messages=[{"role": "user", "content": "go"}]
    )
    source = "prefix-sk-plaintext\ntoken=secret-value\n" + "🙂" * 5000
    async with call:
        for split in (1, 7, 31, 4096):
            await call.output("output", source[:split])
            source = source[split:]
        await call.output("output", source)
        await call.tool_finished("tool", "ok", {"authorization": "Bearer plaintext"})
        await call.terminal(
            "failed", error="token=secret-value", metadata={"token": "plaintext"}
        )
    output = [event for event in events if event.event_type == "agent_response.output"]
    assert (
        "".join(event.payload["text"] for event in output)
        == "prefix-sk-plaintext\ntoken=secret-value\n" + "🙂" * 5000
    )
    assert all(len(event.payload["text"].encode()) <= 16 * 1024 for event in output)
    tool = next(
        event for event in events if event.event_type == "agent_response.tool_finished"
    )
    terminal = next(
        event for event in events if event.event_type == "agent_response.terminal"
    )
    assert tool.payload == {
        "name": "tool",
        "outcome": "ok",
        "result": {"authorization": "Bearer plaintext"},
        "truncated": False,
    }
    assert (
        terminal.payload["error"] == "token=secret-value"
        and not terminal.payload["error_truncated"]
    )
    assert (
        terminal.payload["metadata"] == {"token": "plaintext"}
        and not terminal.payload["metadata_truncated"]
    )


@pytest.mark.asyncio
async def test_utf8_caps_lifecycle_and_strict_payloads():
    events = []
    call = AgentResponseInvocation(context(), publish=publisher(events)).call(
        messages=[]
    )
    async with call:
        await call.tool_started("tool", "🙂" * 5000)
        await call.terminal(
            "failed", error="🙂" * 1000, metadata={"large": "🙂" * 5000}
        )
        await call.terminal("completed")
    assert events[0].sequence == 0 and events[0].emitted_at == events[0].call_started_at
    assert [event.sequence for event in events] == list(range(len(events)))
    assert (
        len(
            [event for event in events if event.event_type == "agent_response.terminal"]
        )
        == 1
    )
    tool, terminal = events[1], events[-1]
    assert (
        tool.payload["truncated"]
        and len(tool.payload["arguments"].encode()) <= 16 * 1024
    )
    assert tool.payload["arguments"].startswith('"')
    assert (
        terminal.payload["error_truncated"]
        and len(terminal.payload["error"].encode()) <= 2048
    )
    assert (
        terminal.payload["metadata_truncated"]
        and len(terminal.payload["metadata"].encode()) <= 16 * 1024
    )
    assert terminal.payload["error"].startswith('"')
    assert terminal.payload["metadata"].startswith('{"large":"')
    with pytest.raises(ValidationError):
        AgentResponseEvent(
            project_id=uuid4(),
            call_id=uuid4(),
            sequence=0,
            event_type="agent_response.output",
            emitted_at=datetime.now(timezone.utc),
            call_started_at=datetime.now(timezone.utc),
            invocation_kind="api",
            operation="task",
            parent_call_id=None,
            actor_kind="agent",
            actor_id="a",
            payload={"stream": "output", "text": "x", "extra": True},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception",
    [
        litellm.UnsupportedParamsError("no stream"),
        litellm.BadRequestError("STREAMING IS NOT SUPPORTED", "model", "provider"),
        litellm.BadRequestError("UNSUPPORTED PARAMETER: STREAM", "model", "provider"),
        litellm.BadRequestError("DOES NOT SUPPORT STREAM", "model", "provider"),
        TypeError("unexpected keyword argument 'stream'"),
    ],
)
async def test_pre_chunk_unsupported_stream_falls_back_once(exception):
    events, attempts = [], []
    invocation = AgentResponseInvocation(context(), publish=publisher(events))
    primary = invocation.call(
        messages=[{"role": "user", "content": "go"}], invocation_kind="cli_resume"
    )

    async def completion(**kwargs):
        attempts.append(kwargs)
        if kwargs.get("stream"):
            raise exception
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="fallback", reasoning_content="think"
                    )
                )
            ]
        )

    async with primary:
        response = await primary.complete(completion, {"messages": []})
    starts = [event for event in events if event.event_type == "agent_response.started"]
    terminals = [
        event for event in events if event.event_type == "agent_response.terminal"
    ]
    assert (
        response.choices[0].message.content == "fallback"
        and len(attempts) == 2
        and "stream" not in attempts[1]
    )
    assert (
        starts[1].parent_call_id == primary.call_id
        and starts[1].invocation_kind == "cli_resume"
    )
    assert starts[1].payload["request_display"] == starts[0].payload["request_display"]
    assert [event.payload["status"] for event in terminals] == ["failed", "completed"]
    assert terminals[0].payload["error"] == str(exception)
    assert terminals[0].payload["metadata"] == {
        "reason": "streaming_unsupported",
        "fallback_call_id": str(starts[1].call_id),
    }


@pytest.mark.asyncio
async def test_post_chunk_rejection_does_not_retry_and_context_exit_statuses(caplog):
    events, calls = [], 0
    call = AgentResponseInvocation(context(), publish=publisher(events)).call(
        messages=[]
    )

    class Stream:
        yielded = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.yielded:
                self.yielded = True
                return SimpleNamespace(choices=[])
            raise TypeError("unexpected keyword argument 'stream'")

    async def completion(**_kwargs):
        nonlocal calls
        calls += 1
        return Stream()

    with pytest.raises(TypeError):
        async with call:
            await call.complete(completion, {"messages": []})
    assert calls == 1 and [
        event.payload["status"]
        for event in events
        if event.event_type == "agent_response.terminal"
    ] == ["failed"]
    omitted = AgentResponseInvocation(context(), publish=publisher([])).call(
        messages=[]
    )

    async def owner():
        await omitted.__aenter__()

    await asyncio.create_task(owner())
    await asyncio.sleep(0)
    assert str(omitted.call_id) in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raised", "status"), [(asyncio.TimeoutError, "timeout"), (RuntimeError, "failed")]
)
async def test_context_exit_terminalizes_timeout_and_failure(raised, status):
    events = []
    call = AgentResponseInvocation(context(), publish=publisher(events)).call(
        messages=[]
    )
    with pytest.raises(raised):
        async with call:
            raise raised("provider failed")
    assert [
        event.payload["status"]
        for event in events
        if event.event_type == "agent_response.terminal"
    ] == [status]


@pytest.mark.asyncio
async def test_coalescing_flushes_on_deadline_and_manual_reset_terminal():
    events = []
    call = AgentResponseInvocation(context(), publish=publisher(events)).call(
        messages=[]
    )
    async with call:
        await call.output("reasoning", "think")
        await asyncio.sleep(0.11)
        await call.terminal("project_reset", metadata={"generation": "g"})
    output = [
        event.payload for event in events if event.event_type == "agent_response.output"
    ]
    terminal = next(
        event for event in events if event.event_type == "agent_response.terminal"
    )
    assert output == [{"stream": "reasoning", "text": "think"}]
    assert terminal.payload["status"] == "project_reset"


@pytest.mark.asyncio
async def test_coalescing_flushes_on_newline_and_four_kib():
    events = []
    call = AgentResponseInvocation(context(), publish=publisher(events)).call(
        messages=[]
    )
    async with call:
        await call.output("output", "line\n")
        await call.output("output", "x" * 4096)
        assert [
            event.payload["text"]
            for event in events
            if event.event_type == "agent_response.output"
        ] == ["line\n", "x" * 4096]


@pytest.mark.asyncio
async def test_complete_reconstructs_stream_and_accepts_direct_response(monkeypatch):
    events = []
    call = AgentResponseInvocation(context(), publish=publisher(events)).call(
        messages=[]
    )

    class Stream:
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="answer", reasoning_content="think"
                        )
                    )
                ]
            )
        ]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.chunks:
                return self.chunks.pop()
            raise StopAsyncIteration

    async def streaming(**_kwargs):
        return Stream()

    monkeypatch.setattr(
        "huddleroom.services.agent_response_stream.litellm.stream_chunk_builder",
        lambda chunks, messages: "rebuilt",
    )
    async with call:
        assert await call.complete(streaming, {"messages": []}) == "rebuilt"
    assert [
        event.payload for event in events if event.event_type == "agent_response.output"
    ] == [
        {"stream": "output", "text": "answer"},
        {"stream": "reasoning", "text": "think"},
    ]
    direct = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="direct", reasoning="reason")
            )
        ]
    )
    events = []
    call = AgentResponseInvocation(context(), publish=publisher(events)).call(
        messages=[]
    )

    async def completion(**_kwargs):
        return direct

    async with call:
        assert await call.complete(completion, {"messages": []}) is direct
    assert [
        event.payload for event in events if event.event_type == "agent_response.output"
    ] == [
        {"stream": "output", "text": "direct"},
        {"stream": "reasoning", "text": "reason"},
    ]


@pytest.mark.asyncio
async def test_complete_forces_stream_when_request_already_contains_stream():
    call = AgentResponseInvocation(context(), publish=publisher([])).call(messages=[])
    received = {}

    async def completion(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(choices=[])

    async with call:
        await call.complete(completion, {"messages": [], "stream": False})

    assert received["stream"] is True


@pytest.mark.asyncio
async def test_terminal_closes_concurrent_emitters():
    events = []
    call = AgentResponseInvocation(context(), publish=publisher(events)).call(
        messages=[]
    )
    await call.__aenter__()
    await call.output("output", "buffered")
    await call.terminal("completed")
    await asyncio.gather(
        call.output("output", "late"), call.tool_started("late", {"x": 1})
    )
    await asyncio.sleep(0.11)
    assert [event.event_type for event in events][-1] == "agent_response.terminal"
    assert "late" not in str([event.payload for event in events])


@pytest.mark.asyncio
async def test_terminal_serializes_queued_timer_drain_before_terminal():
    events = []
    call = AgentResponseInvocation(context(), publish=publisher(events)).call(
        messages=[]
    )
    await call.__aenter__()
    await call.output("output", "buffered")
    await call._output._lock.acquire()  # pylint: disable=protected-access
    timer = asyncio.create_task(call._output.flush())  # pylint: disable=protected-access
    terminal = asyncio.create_task(call.terminal("completed"))
    await asyncio.sleep(0)
    call._output._lock.release()  # pylint: disable=protected-access
    await asyncio.gather(timer, terminal)
    assert [event.event_type for event in events] == [
        "agent_response.started",
        "agent_response.output",
        "agent_response.terminal",
    ]
    assert events[1].payload["text"] == "buffered"


@pytest.mark.asyncio
async def test_cancellation_and_reset_do_not_duplicate_terminal():
    events = []
    call = AgentResponseInvocation(context(), publish=publisher(events)).call(
        messages=[]
    )
    entered = asyncio.Event()

    async def owner():
        async with call:
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(owner())
    await entered.wait()
    await call.terminal("project_reset", metadata={"generation": "g"})
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [
        event.payload["status"]
        for event in events
        if event.event_type == "agent_response.terminal"
    ] == ["project_reset"]


@pytest.mark.asyncio
async def test_repair_attempts_link_and_inherit_the_root_display(stream_monitors):
    from huddleroom.services.llm_structured_repair import complete_with_repair

    events, responses = [], iter(("bad", "good"))
    invocation = AgentResponseInvocation(context(), publish=publisher(events))

    async def completion(**_kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))]
        )

    def parse(raw):
        if raw == "bad":
            raise ValueError("not ready")
        return raw

    assert await complete_with_repair(
        completion,
        [{"messages": []}][0],
        parse,
        invocation=invocation,
    ) == "good"
    starts = [event for event in events if event.event_type == "agent_response.started"]
    assert len(starts) == 2
    assert starts[1].parent_call_id == starts[0].call_id
    assert starts[1].payload["request_display"] == starts[0].payload["request_display"]


@pytest.mark.asyncio
async def test_tool_continuation_emits_verbatim_tool_events(stream_monitors, monkeypatch):
    from huddleroom.services.tool_executor import run_tool_loop

    events = []
    invocation = AgentResponseInvocation(context(), publish=publisher(events))
    tool_call = SimpleNamespace(
        id="tool-1",
        function=SimpleNamespace(name="memory_search", arguments='{"query":"plaintext"}'),
    )
    first = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[tool_call],
                    model_dump=lambda: {"role": "assistant", "content": None},
                )
            )
        ]
    )
    second = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=None))]
    )
    responses = iter((first, second))

    async def completion(**_kwargs):
        return next(responses)

    monkeypatch.setattr(
        "huddleroom.services.tool_executor.execute_memory_tool", AsyncMock(return_value='{"token":"plaintext"}')
    )
    assert await run_tool_loop(
        completion, [{"role": "user", "content": "go"}], [{}], uuid4(), uuid4(), None,
        invocation=invocation,
    ) == "done"
    starts = [event for event in events if event.event_type == "agent_response.started"]
    tool_started = next(event for event in events if event.event_type == "agent_response.tool_started")
    tool_finished = next(event for event in events if event.event_type == "agent_response.tool_finished")
    assert starts[1].parent_call_id == starts[0].call_id
    assert starts[1].payload["request_display"] == {"kind": "continuation", "content": None, "truncated": False}
    assert tool_started.payload["arguments"] == '{"query":"plaintext"}'
    assert tool_finished.payload["result"] == '{"token":"plaintext"}'


@pytest.mark.asyncio
async def test_tool_stream_fallback_keeps_events_and_continuation_on_fallback(
    stream_monitors, monkeypatch
):
    """A fallback must remain the tool interaction's physical parent."""
    from huddleroom.services.tool_executor import run_tool_loop

    events = []
    invocation = AgentResponseInvocation(context(), publish=publisher(events))
    tool_call = SimpleNamespace(
        id="tool-1", function=SimpleNamespace(name="memory_search", arguments="{}")
    )
    tool_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=None, tool_calls=[tool_call],
            model_dump=lambda: {"role": "assistant", "content": None},
        ))]
    )
    final_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=None))]
    )

    async def completion(**kwargs):
        if kwargs.get("stream"):
            if not hasattr(completion, "rejected"):
                completion.rejected = True
                raise TypeError("unexpected keyword argument 'stream'")
            return final_response
        return tool_response

    monkeypatch.setattr(
        "huddleroom.services.tool_executor.execute_memory_tool", AsyncMock(return_value="plaintext-result")
    )
    assert await run_tool_loop(
        completion, [{"role": "user", "content": "go"}], [{}], uuid4(), uuid4(), None,
        invocation=invocation,
    ) == "done"
    starts = [event for event in events if event.event_type == "agent_response.started"]
    fallback = starts[1]
    tool_events = [event for event in events if "tool_" in event.event_type]
    continuation = starts[2]
    assert [event.call_id for event in tool_events] == [fallback.call_id, fallback.call_id]
    assert continuation.parent_call_id == fallback.call_id


@pytest.mark.asyncio
async def test_tool_stream_fallback_failure_closes_its_child(stream_monitors):
    """A fallback provider failure terminalizes its manually-opened child."""
    from huddleroom.services.tool_executor import run_tool_loop

    events = []
    invocation = AgentResponseInvocation(context(), publish=publisher(events))

    async def completion(**kwargs):
        if kwargs.get("stream"):
            raise TypeError("unexpected keyword argument 'stream'")
        raise RuntimeError("fallback failed")

    with pytest.raises(RuntimeError, match="fallback failed"):
        await run_tool_loop(
            completion, [{"role": "user", "content": "go"}], [{}], uuid4(), uuid4(), None,
            invocation=invocation,
        )
    terminals = [event for event in events if event.event_type == "agent_response.terminal"]
    assert [event.payload["status"] for event in terminals] == ["failed", "failed"]


@pytest.mark.asyncio
async def test_api_adapter_uses_reconstructed_stream_usage_and_call_context(
    stream_monitors, monkeypatch
):
    """A streamed provider response must retain rebuilt usage in the API session."""
    from huddleroom.adapters.api_adapter import ApiAdapter

    events, agent_id = [], uuid4()
    agent = SimpleNamespace(
        id=agent_id, name="Ada", provider="openai", model="model", config={},
    )
    session = SimpleNamespace(
        id=uuid4(), project_id=uuid4(), agent_id=agent_id, task_id=None,
        metadata_={}, output=None, status="running", ended_at=None,
    )
    db = SimpleNamespace(refresh=AsyncMock(), flush=AsyncMock())
    rebuilt = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
    )

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if getattr(self, "sent", False):
                raise StopAsyncIteration
            self.sent = True
            return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="done"))])

    async def completion(**kwargs):
        assert kwargs["stream"] is True
        return Stream()

    monkeypatch.setattr("litellm.acompletion", completion)
    monkeypatch.setattr(
        "huddleroom.services.agent_response_stream.litellm.stream_chunk_builder",
        lambda _chunks, messages: rebuilt,
    )
    monkeypatch.setattr(
        "huddleroom.adapters.api_adapter.AgentResponseInvocation",
        lambda context: AgentResponseInvocation(context, publish=publisher(events)),
    )
    monkeypatch.setattr("huddleroom.adapters.api_adapter.emit_event", AsyncMock())
    monkeypatch.setattr("huddleroom.services.session_sync.sync_task_from_session", AsyncMock())
    await ApiAdapter()._run_with_retry(
        agent, session, [{"role": "user", "content": "Task: Ship"}], "Task: Ship", db
    )
    started = next(event for event in events if event.event_type == "agent_response.started")
    assert session.metadata_["token_count_in"] == 11
    assert session.metadata_["token_count_out"] == 7
    assert (started.operation, started.invocation_kind) == ("task", "api")
    assert (started.actor_kind, started.actor_id, started.payload["actor_label"]) == (
        "agent", str(agent_id), "Ada"
    )


@pytest.mark.asyncio
async def test_api_adapter_retries_real_call_with_refreshed_config(stream_monitors, monkeypatch):
    """The retry call must use settings changed while its provider call backs off."""
    from huddleroom.adapters.api_adapter import ApiAdapter

    events, agent_id, calls = [], uuid4(), []
    agent = SimpleNamespace(
        id=agent_id, name="Ada", provider="openai", model="before", config={"temperature": 0.7},
    )
    session = SimpleNamespace(
        id=uuid4(), project_id=uuid4(), agent_id=agent_id, task_id=None,
        metadata_={}, output=None, status="running", ended_at=None,
    )
    db = SimpleNamespace(refresh=AsyncMock(), flush=AsyncMock())

    async def completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise litellm.RateLimitError("retry", "openai", "before")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=None))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5),
        )

    async def change_during_backoff(_delay):
        agent.provider, agent.model = "changed", "after"
        agent.config = {"temperature": 0.1}

    monkeypatch.setattr("litellm.acompletion", completion)
    monkeypatch.setattr("huddleroom.adapters.api_adapter.asyncio.sleep", change_during_backoff)
    monkeypatch.setattr(
        "huddleroom.adapters.api_adapter.AgentResponseInvocation",
        lambda context: AgentResponseInvocation(context, publish=publisher(events)),
    )
    monkeypatch.setattr("huddleroom.adapters.api_adapter.emit_event", AsyncMock())
    monkeypatch.setattr("huddleroom.services.session_sync.sync_task_from_session", AsyncMock())
    await ApiAdapter()._run_with_retry(
        agent, session, [{"role": "user", "content": "Task: Ship"}], "Task: Ship", db
    )
    assert calls[1]["model"] == "changed/after"
    assert calls[1]["temperature"] == 0.1
    starts = [event for event in events if event.event_type == "agent_response.started"]
    assert all(
        (event.operation, event.invocation_kind, event.actor_kind, event.actor_id)
        == ("task", "api", "agent", str(agent_id))
        for event in starts
    )


@pytest.mark.asyncio
async def test_api_adapter_invocation_uses_task_source_before_extra_context(monkeypatch):
    from huddleroom.adapters.api_adapter import ApiAdapter

    project_id, agent_id, task_id = uuid4(), uuid4(), uuid4()
    session = SimpleNamespace(
        id=uuid4(), project_id=project_id, agent_id=agent_id, task_id=task_id,
        status="pending", metadata_=None,
    )
    agent = SimpleNamespace(
        id=agent_id, name="Ada", role="developer", model="model", provider="openai", config={}, system_prompt=None
    )
    task = SimpleNamespace(id=task_id, project_id=project_id, title="Ship", description="Now")
    db = SimpleNamespace(
        execute=AsyncMock(side_effect=[
            SimpleNamespace(scalar_one_or_none=lambda: session),
            SimpleNamespace(scalar_one_or_none=lambda: agent),
            SimpleNamespace(scalar_one_or_none=lambda: task),
        ]),
        flush=AsyncMock(), commit=AsyncMock(),
        refresh=AsyncMock(),
        get=AsyncMock(return_value=None),
    )
    adapter, captured = ApiAdapter(), {}
    monkeypatch.setattr(
        adapter, "_build_messages", AsyncMock(return_value=([
            {"role": "user", "content": "Task: Ship\nDescription: Now\nRelevant knowledge: hidden"}
        ], "Task: Ship\nDescription: Now\n", 1, 1)),
    )

    original = adapter._run_with_retry

    async def run_with_retry(*args):
        async def capture_tool_loop(**kwargs):
            captured["invocation"] = kwargs["invocation"]
            return "done"

        monkeypatch.setattr("huddleroom.adapters.api_adapter.run_tool_loop", capture_tool_loop)
        await original(*args)

    monkeypatch.setattr(adapter, "_run_with_retry", run_with_retry)
    monkeypatch.setattr("huddleroom.adapters.api_adapter.emit_event", AsyncMock())
    monkeypatch.setattr("huddleroom.services.session_sync.sync_task_from_session", AsyncMock())
    await adapter.run(session.id, db)
    invocation = captured["invocation"]
    assert invocation.context.operation == "task"
    assert invocation.context.actor_id == str(agent_id)
    assert invocation.root_request.content == "Task: Ship\nDescription: Now\n"


@pytest.mark.asyncio
async def test_orchestration_analyzers_use_reserved_context_only_with_a_project_id(monkeypatch):
    from huddleroom.services.orchestration_agent_definition_analyzer import AgentDefinitionSemanticAnalyzer
    from huddleroom.services.orchestration_goal_analyzer import GoalClarificationAnalyzer
    from huddleroom.services.orchestration_llm_decision_adapter import OrchestrationDecisionAdapter
    from huddleroom.services.orchestration_manager_analyzer import ManagerSelectionAnalyzer
    from huddleroom.services.orchestration_team_hierarchy_analyzer import TeamHierarchyAnalyzer

    calls = []

    async def complete(*_args, **kwargs):
        calls.append(kwargs)
        return object()

    modules = (
        "huddleroom.services.orchestration_goal_analyzer.complete_with_repair",
        "huddleroom.services.orchestration_manager_analyzer.complete_with_repair",
        "huddleroom.services.orchestration_agent_definition_analyzer.complete_with_repair",
        "huddleroom.services.orchestration_team_hierarchy_analyzer.complete_with_repair",
        "huddleroom.services.llm_structured_repair.complete_with_repair",
    )
    for target in modules:
        monkeypatch.setattr(target, complete)
    project_id = uuid4()
    requests = [
        (GoalClarificationAnalyzer(), "analyze_request", ({"model": "goal", "messages": []},), "goal_analysis",
         "Identify safe assumptions and material clarification questions for this goal."),
        (ManagerSelectionAnalyzer(), "review_request", ({"model": "manager", "messages": [{}, {"content": '{"candidates":[],"deterministic_recommendation":"none"}'}]},), "manager_selection",
         "Select the best manager for this goal."),
        (AgentDefinitionSemanticAnalyzer(), "review_request", ({"model": "agent", "messages": [{}, {"content": '{"agent":{}}'}]}, []), "agent_definition_review",
         "Review this agent definition for the proposed work functions."),
        (TeamHierarchyAnalyzer(), "review_request", ({"model": "team", "messages": [{}, {"content": "{}"}]},), "team_hierarchy",
         "Design the smallest team and reporting structure that covers the required work."),
    ]
    for analyzer, method_name, args, operation, display in requests:
        method = getattr(analyzer, method_name)
        await method(*args)
        assert "invocation" not in calls[-1]
        await method(*args, project_id=project_id)
        context = calls[-1]["invocation"].context
        assert (context.project_id, context.actor_kind, context.actor_id, context.operation, context.request_prompt) == (
            project_id, "system", "orchestrator", operation, display
        )

    await OrchestrationDecisionAdapter(completion_fn=AsyncMock()).decide({}, project={"id": str(project_id)})
    context = calls[-1]["invocation"].context
    assert (context.project_id, context.operation, context.request_prompt) == (
        project_id, "decision", "Choose the next orchestration action for this goal."
    )
    await OrchestrationDecisionAdapter(completion_fn=AsyncMock()).decide({}, project={"id": "not-a-uuid"})
    assert "invocation" not in calls[-1]
