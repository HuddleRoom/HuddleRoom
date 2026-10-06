"""Bounded, ephemeral events around physical LLM response calls."""
# pylint: disable=too-many-instance-attributes

import asyncio
import json
import logging
from weakref import WeakKeyDictionary
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

import litellm
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

ActorKind = Literal["system", "agent"]
InvocationKind = Literal["api", "cli_main", "cli_resume", "cli_meeting"]
OutputStream = Literal["output", "reasoning", "stderr"]
TerminalStatus = Literal["completed", "failed", "cancelled", "timeout", "project_reset"]
EventPublisher = Callable[["AgentResponseEvent"], Awaitable[None]]
CompletionFn = Callable[..., Awaitable[Any]]
logger = logging.getLogger(__name__)
_ACTIVE_RESET_CALLS: WeakKeyDictionary[asyncio.AbstractEventLoop, set["AgentResponseCall"]] = WeakKeyDictionary()
_CAP = 16 * 1024
_ERROR_CAP = 2 * 1024


class StrictWireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RequestDisplay(StrictWireModel):
    kind: Literal["prompt", "continuation", "unavailable"]
    content: JsonValue | None = None
    truncated: bool = False


class StartedPayload(StrictWireModel):
    request_display: RequestDisplay
    actor_label: str | None
    model_or_runtime: str


class OutputPayload(StrictWireModel):
    stream: OutputStream
    text: str


class ToolStartedPayload(StrictWireModel):
    name: str
    arguments: JsonValue | None
    truncated: bool = False


class ToolFinishedPayload(StrictWireModel):
    name: str
    outcome: str
    result: JsonValue | None
    truncated: bool = False


class TerminalPayload(StrictWireModel):
    status: TerminalStatus
    error: str | None = None
    error_truncated: bool = False
    metadata: JsonValue | None = None
    metadata_truncated: bool = False


class AgentResponseEvent(StrictWireModel):
    id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    call_id: UUID
    sequence: int
    event_type: Literal[
        "agent_response.started",
        "agent_response.output",
        "agent_response.tool_started",
        "agent_response.tool_finished",
        "agent_response.terminal",
    ]
    emitted_at: datetime
    call_started_at: datetime
    invocation_kind: InvocationKind
    operation: str
    parent_call_id: UUID | None
    actor_kind: ActorKind
    actor_id: str
    payload: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_payload(self):
        model = {
            "agent_response.started": StartedPayload,
            "agent_response.output": OutputPayload,
            "agent_response.tool_started": ToolStartedPayload,
            "agent_response.tool_finished": ToolFinishedPayload,
            "agent_response.terminal": TerminalPayload,
        }[self.event_type]
        self.payload = model.model_validate(self.payload).model_dump(mode="json")
        return self


@dataclass(frozen=True)
class InvocationContext:
    project_id: UUID
    actor_kind: ActorKind
    actor_id: str
    actor_label: str | None
    invocation_kind: InvocationKind
    operation: str
    model_or_runtime: str
    request_prompt: JsonValue | None = None


@dataclass
class TerminalAudit:
    call_id: UUID
    started: bool = False
    terminal: bool = False


def _audit_owner_done(audit: TerminalAudit) -> None:
    if audit.started and not audit.terminal:
        logger.error("agent response terminal omitted call_id=%s", audit.call_id)


def _prefix(text: str, limit: int) -> tuple[str, bool]:
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text, False
    return data[:limit].decode("utf-8", errors="ignore"), True


def _bounded(value: Any, limit: int) -> tuple[JsonValue | None, bool]:
    normalized = json.loads(json.dumps(value, ensure_ascii=False, default=str))
    serialized = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    if len(serialized.encode("utf-8")) <= limit:
        return normalized, False
    return _prefix(serialized, limit)


def extract_request_display(
    messages: Sequence[Mapping[str, Any]], continuation: bool = False
) -> RequestDisplay:
    if continuation:
        return RequestDisplay(kind="continuation")
    prompt = next(
        (
            message.get("content")
            for message in reversed(messages)
            if message.get("role") in {"user", "task"}
        ),
        None,
    )
    if prompt is None:
        return RequestDisplay(kind="unavailable")
    if isinstance(prompt, str):
        try:
            parsed = json.loads(prompt)
            if isinstance(parsed, (dict, list)):
                prompt = parsed
        except json.JSONDecodeError:
            pass
    content, truncated = _bounded(prompt, _CAP)
    return RequestDisplay(kind="prompt", content=content, truncated=truncated)


async def publish_agent_response(event: AgentResponseEvent) -> None:
    from huddleroom.services.agent_response_relay import publish_agent_response as publish

    await publish(event)


async def get_agent_response_metrics() -> dict[str, int | float]:
    from huddleroom.services.agent_response_relay import (
        get_agent_response_metrics as get_metrics,
    )

    return await get_metrics()


async def close_agent_response_relay() -> None:
    from huddleroom.services.agent_response_relay import close_agent_response_relay as close

    await close()


async def close_agent_response_monitors() -> None:
    """Release per-call reset leases before the process closes its Redis client."""
    calls = _ACTIVE_RESET_CALLS.get(asyncio.get_running_loop(), set())
    failure = None
    for call in tuple(calls):
        try:
            await call.terminal("cancelled")
        except BaseException as exc:
            if failure is None:
                failure = exc
    if failure is not None:
        raise failure


def _track_reset_call(call: "AgentResponseCall") -> None:
    _ACTIVE_RESET_CALLS.setdefault(asyncio.get_running_loop(), set()).add(call)


def _untrack_reset_call(call: "AgentResponseCall") -> None:
    loop = asyncio.get_running_loop()
    calls = _ACTIVE_RESET_CALLS.get(loop)
    if calls:
        calls.discard(call)
        if not calls:
            _ACTIVE_RESET_CALLS.pop(loop, None)


async def publish_project_reset(project_id: UUID) -> str:
    from huddleroom.services.agent_response_relay import publish_project_reset as publish

    return await publish(project_id)


async def wait_for_project_reset(project_id: UUID) -> str:
    from huddleroom.services.agent_response_relay import wait_for_project_reset as wait

    return await wait(project_id)


async def clear_project_reset(project_id: UUID, generation: str) -> None:
    from huddleroom.services.agent_response_relay import clear_project_reset as clear

    await clear(project_id, generation)


class _OutputBuffer:
    def __init__(self, call: "AgentResponseCall"):
        self.call = call
        self.buffers: dict[OutputStream, str] = {
            "output": "",
            "reasoning": "",
            "stderr": "",
        }
        self.timer: asyncio.TimerHandle | None = None
        self.timer_task: asyncio.Task[None] | None = None
        self.closed = False
        self._lock = asyncio.Lock()

    async def feed(self, stream: OutputStream, text: str | None) -> None:
        if not text:
            return
        async with self._lock:
            if self.closed:
                return
            self.buffers[stream] += text
            if "\n" in text or len(self.buffers[stream].encode()) >= 4096:
                await self._flush(stream)
            elif self.timer is None:
                self.timer = asyncio.get_running_loop().call_later(
                    0.1, self._timer_flush
                )

    def _timer_flush(self) -> None:
        self.timer = None
        self.timer_task = asyncio.create_task(self.flush())

    async def close_and_flush(self) -> None:
        async with self._lock:
            self.closed = True
            self.call._closing = True  # pylint: disable=protected-access
            if self.timer:
                self.timer.cancel()
                self.timer = None
            await self._flush(allow_closing=True)

    async def flush(self, stream: OutputStream | None = None) -> None:
        async with self._lock:
            if not self.closed:
                await self._flush(stream)

    async def _flush(
        self, stream: OutputStream | None = None, *, allow_closing: bool = False
    ) -> None:
        streams = (stream,) if stream else tuple(self.buffers)
        for channel in streams:
            text, self.buffers[channel] = self.buffers[channel], ""
            while text:
                part, split = _prefix(text, _CAP)
                await self.call._event(  # pylint: disable=protected-access
                    "agent_response.output",
                    {"stream": channel, "text": part},
                    allow_closing=allow_closing,
                )
                text = text[len(part) :] if split else ""


class AgentResponseInvocation:
    def __init__(
        self,
        context: InvocationContext,
        *,
        publish: EventPublisher = publish_agent_response,
    ):
        self.context = context
        self.publish = publish
        self.root_request = (
            self._display(context.request_prompt)
            if context.request_prompt is not None
            else None
        )

    @staticmethod
    def _display(prompt: JsonValue) -> RequestDisplay:
        content, truncated = _bounded(prompt, _CAP)
        return RequestDisplay(kind="prompt", content=content, truncated=truncated)

    def call(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        continuation: bool = False,
        inherit_request: bool = False,
        invocation_kind: InvocationKind | None = None,
    ) -> "AgentResponseCall":
        if inherit_request and self.root_request:
            display = self.root_request
        elif continuation:
            display = extract_request_display(messages, continuation=True)
        else:
            display = self.root_request or extract_request_display(messages)
        if self.root_request is None:
            self.root_request = display
        return AgentResponseCall(
            self, display, invocation_kind or self.context.invocation_kind
        )


class AgentResponseCall:
    def __init__(
        self,
        invocation: AgentResponseInvocation,
        request_display: RequestDisplay,
        invocation_kind: InvocationKind,
        parent_call_id: UUID | None = None,
    ):
        self.invocation, self.request_display, self.invocation_kind = (
            invocation,
            request_display,
            invocation_kind,
        )
        self.parent_call_id, self.call_id = parent_call_id, uuid4()
        self.call_started_at: datetime | None = None
        self.sequence, self._terminal_sent, self._closing = 0, False, False
        self._publish_lock, self._terminal_lock = asyncio.Lock(), asyncio.Lock()
        self._output, self._audit = _OutputBuffer(self), TerminalAudit(self.call_id)
        self._reset_monitor: asyncio.Task[None] | None = None
        self._reset_heartbeat: asyncio.Task[None] | None = None
        self._reset_ack = None
        self._owner_task: asyncio.Task | None = None
        self.response_call: AgentResponseCall | None = None

    async def __aenter__(self):
        self.call_started_at = datetime.now(timezone.utc)
        from huddleroom.services.agent_response_relay import register_project_reset_monitor

        self._reset_ack = await register_project_reset_monitor(self.invocation.context.project_id)
        if self._reset_ack.generation:
            await self.terminal(
                "project_reset", metadata={"generation": self._reset_ack.generation}
            )
            self._reset_ack = None
            raise asyncio.CancelledError
        if task := asyncio.current_task():
            self._owner_task = task
            audit = self._audit
            task.add_done_callback(lambda _, audit=audit: _audit_owner_done(audit))
        _track_reset_call(self)
        self._reset_heartbeat = asyncio.create_task(self._renew_reset_monitor())
        try:
            await self._event(
                "agent_response.started",
                {
                    "request_display": self.request_display.model_dump(mode="json"),
                    "actor_label": self.invocation.context.actor_label,
                    "model_or_runtime": self.invocation.context.model_or_runtime,
                },
                self.call_started_at,
            )
        except BaseException as original_error:
            from huddleroom.services.agent_response_relay import unregister_project_reset_monitor

            failure = None
            try:
                await unregister_project_reset_monitor(self.invocation.context.project_id, self._reset_ack)
                self._reset_ack = None
            except BaseException as cleanup_error:
                failure = cleanup_error
            finally:
                heartbeat = self._reset_heartbeat
                if heartbeat:
                    heartbeat.cancel()
                    await asyncio.gather(heartbeat, return_exceptions=True)
                self._reset_heartbeat = None
                if self._reset_ack is None:
                    _untrack_reset_call(self)
            if failure is not None:
                logger.exception("Could not unregister failed call entry call_id=%s", self.call_id)
            raise original_error
        self._audit.started = True
        self._reset_monitor = asyncio.create_task(self._monitor_reset())
        return self

    async def _renew_reset_monitor(self) -> None:
        from huddleroom.services.agent_response_relay import _RESET_RENEW_SECONDS, renew_project_reset_monitor

        while self._reset_ack is not None:
            await asyncio.sleep(_RESET_RENEW_SECONDS)
            if self._reset_ack is not None:
                try:
                    generation = await renew_project_reset_monitor(
                        self.invocation.context.project_id, self._reset_ack
                    )
                except asyncio.CancelledError:
                    raise
                except BaseException:
                    try:
                        await self.terminal("failed", error="reset monitor renewal failed")
                    except BaseException:
                        logger.exception("Could not publish reset monitor failure call_id=%s", self.call_id)
                        self._terminal_sent = True
                    finally:
                        if self._owner_task is not None and self._owner_task is not asyncio.current_task():
                            self._owner_task.cancel()
                    return
                if generation:
                    await self._trip_reset(generation)
                    return

    async def _monitor_reset(self) -> None:
        generation = await wait_for_project_reset(self.invocation.context.project_id)
        await self._trip_reset(generation)

    async def _trip_reset(self, generation: str) -> None:
        await self.terminal(
            "project_reset", metadata={"generation": generation}, defer_unregister=True
        )
        if self._owner_task is not None and self._owner_task is not asyncio.current_task():
            self._owner_task.cancel()
            await asyncio.sleep(0)  # let the owner observe cancellation before reset quorum ack
        await self._terminal_cleanup()

    async def __aexit__(self, exc_type, exc, tb):
        if not self._terminal_sent:
            status = (
                "completed"
                if exc_type is None
                else "cancelled"
                if issubclass(exc_type, asyncio.CancelledError)
                else "timeout"
                if issubclass(exc_type, asyncio.TimeoutError)
                else "failed"
            )
            await self.terminal(status, error=None if exc_type is None else str(exc))
        return False

    async def _event(
        self,
        event_type: str,
        payload: dict[str, Any],
        emitted_at: datetime | None = None,
        *,
        allow_closing: bool = False,
    ) -> None:
        if self.call_started_at is None:
            raise RuntimeError("call has not started")
        async with self._publish_lock:
            if self._closing and not allow_closing:
                return
            event = AgentResponseEvent(
                project_id=self.invocation.context.project_id,
                call_id=self.call_id,
                sequence=self.sequence,
                event_type=event_type,
                emitted_at=emitted_at or datetime.now(timezone.utc),
                call_started_at=self.call_started_at,
                invocation_kind=self.invocation_kind,
                operation=self.invocation.context.operation,
                parent_call_id=self.parent_call_id,
                actor_kind=self.invocation.context.actor_kind,
                actor_id=self.invocation.context.actor_id,
                payload=payload,
            )
            await self.invocation.publish(event)
            self.sequence += 1

    async def output(self, stream: OutputStream, text: str) -> None:
        if self._closing:
            return
        await self._output.feed(stream, text)

    async def tool_started(self, name: str, arguments: JsonValue | None) -> None:
        if self._closing:
            return
        value, truncated = _bounded(arguments, _CAP)
        await self._event(
            "agent_response.tool_started",
            {"name": name, "arguments": value, "truncated": truncated},
        )

    async def tool_finished(
        self, name: str, outcome: str, result: JsonValue | None
    ) -> None:
        if self._closing:
            return
        value, truncated = _bounded(result, _CAP)
        await self._event(
            "agent_response.tool_finished",
            {"name": name, "outcome": outcome, "result": value, "truncated": truncated},
        )

    async def terminal(
        self,
        status: TerminalStatus,
        *,
        error: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        defer_unregister: bool = False,
    ) -> None:
        async with self._terminal_lock:
            if self._terminal_sent:
                if self._reset_ack is not None:
                    await self._terminal_cleanup()
                return
            try:
                await self._output.close_and_flush()
                safe_error, error_truncated = _bounded(error, _ERROR_CAP)
                safe_metadata, metadata_truncated = _bounded(metadata, _CAP)
                await self._event(
                    "agent_response.terminal",
                    {
                        "status": status,
                        "error": safe_error,
                        "error_truncated": error_truncated,
                        "metadata": safe_metadata,
                        "metadata_truncated": metadata_truncated,
                    },
                    allow_closing=True,
                )
                self._terminal_sent = self._audit.terminal = True
            except BaseException:
                await self._terminal_cleanup(unregister=False)
                raise
            if not defer_unregister:
                await self._terminal_cleanup()

    async def _terminal_cleanup(self, *, unregister: bool = True) -> None:
        monitor = self._reset_monitor
        heartbeat = self._reset_heartbeat
        failure = None
        try:
            if unregister and self._reset_ack:
                from huddleroom.services.agent_response_relay import unregister_project_reset_monitor

                await unregister_project_reset_monitor(self.invocation.context.project_id, self._reset_ack)
                self._reset_ack = None
        except BaseException as exc:
            failure = exc
        finally:
            for task in (monitor, heartbeat):
                if task and task is not asyncio.current_task():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (monitor, heartbeat) if task and task is not asyncio.current_task()),
                return_exceptions=True,
            )
            if self._reset_monitor is monitor:
                self._reset_monitor = None
            if self._reset_heartbeat is heartbeat:
                self._reset_heartbeat = None
            if not unregister or self._reset_ack is None:
                _untrack_reset_call(self)
        if failure is not None:
            raise failure

    async def complete(
        self,
        completion_fn: CompletionFn,
        request: Mapping[str, Any],
        *,
        keep_fallback_open: bool = False,
    ) -> Any:
        provider_chunk_seen = False
        try:
            stream_request = dict(request)
            stream_request["stream"] = True
            stream = await completion_fn(**stream_request)
            if not hasattr(stream, "__aiter__"):
                await self._emit_response(stream)
                self.response_call = self
                return stream
            chunks = []
            async for chunk in stream:
                provider_chunk_seen = True
                chunks.append(chunk)
                delta = (
                    chunk.choices[0].delta if getattr(chunk, "choices", None) else None
                )
                await self._output.feed("output", getattr(delta, "content", None))
                await self._output.feed(
                    "reasoning",
                    getattr(delta, "reasoning_content", None)
                    or getattr(delta, "reasoning", None),
                )
            response = litellm.stream_chunk_builder(
                chunks, messages=list(request.get("messages", []))
            )
            if response is None:
                raise RuntimeError("LiteLLM could not reconstruct streamed completion")
            self.response_call = self
            return response
        except BaseException as exc:
            if not provider_chunk_seen and self._streaming_unsupported(exc):
                return await self._fallback(
                    completion_fn, request, exc, keep_open=keep_fallback_open
                )
            raise

    @staticmethod
    def _streaming_unsupported(exc: BaseException) -> bool:
        if isinstance(exc, TypeError):
            return "unexpected keyword argument 'stream'" in str(exc).lower()
        if isinstance(exc, getattr(litellm, "UnsupportedParamsError", ())):
            return True
        if isinstance(exc, getattr(litellm, "BadRequestError", ())):
            return any(
                phrase in str(exc).lower()
                for phrase in (
                    "streaming is not supported",
                    "stream is not supported",
                    "unsupported parameter: stream",
                    "does not support stream",
                )
            )
        return False

    async def _fallback(
        self,
        completion_fn: CompletionFn,
        request: Mapping[str, Any],
        exc: BaseException,
        *,
        keep_open: bool = False,
    ) -> Any:
        child = AgentResponseCall(
            self.invocation,
            self.invocation.root_request or self.request_display,
            self.invocation_kind,
            self.call_id,
        )
        await self.terminal(
            "failed",
            error=str(exc),
            metadata={
                "reason": "streaming_unsupported",
                "fallback_call_id": str(child.call_id),
            },
        )
        fallback_request = dict(request)
        fallback_request.pop("stream", None)
        if keep_open:
            await child.__aenter__()
            try:
                response = await completion_fn(**fallback_request)
                await child._emit_response(response)  # pylint: disable=protected-access
                child.response_call = child
                self.response_call = child
                return response
            except BaseException as error:
                await child.__aexit__(type(error), error, error.__traceback__)
                raise
        async with child:
            response = await completion_fn(**fallback_request)
            await child._emit_response(response)  # pylint: disable=protected-access
            child.response_call = child
            self.response_call = child
            return response

    async def _emit_response(self, response: Any) -> None:
        choices = (response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)) or []
        choice = choices[0] if choices else None
        message = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
        if message:
            await self._output.feed(
                "reasoning",
                (message.get("reasoning_content") or message.get("reasoning")) if isinstance(message, dict) else (getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)),
            )
            await self._output.feed("output", message.get("content") if isinstance(message, dict) else getattr(message, "content", None))
