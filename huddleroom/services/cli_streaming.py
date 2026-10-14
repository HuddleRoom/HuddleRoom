"""Concurrent CLI subprocess collection and incremental decoding."""
import asyncio
import codecs
import json
import logging
import os
import signal
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import JsonValue

logger = logging.getLogger(__name__)


async def terminate_process_group(proc: Any, grace_seconds: float = 0.5) -> None:
    """Stop a process group created with ``start_new_session=True``."""
    process_group_id = proc.pid

    async def group_exited() -> bool:
        deadline = asyncio.get_running_loop().time() + grace_seconds
        while True:
            try:
                os.killpg(process_group_id, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                # macOS can report EPERM for a zombie group until its leader is reaped.
                if asyncio.get_running_loop().time() >= deadline:
                    raise
                await asyncio.sleep(min(0.05, grace_seconds))
                continue
            if asyncio.get_running_loop().time() >= deadline:
                return False
            await asyncio.sleep(min(0.05, grace_seconds))

    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        pass
    else:
        if not await group_exited():
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if not await group_exited():
                logger.warning("CLI process group %s did not exit after SIGKILL", process_group_id)
    if proc.returncode is None:
        try:
            await asyncio.wait_for(proc.wait(), timeout=grace_seconds)
        except asyncio.TimeoutError:
            logger.warning("CLI group leader %s did not reap after termination", proc.pid)


@dataclass(frozen=True)
class CliDisplayUpdate:
    """A readable update extracted from CLI output."""
    kind: Literal["output", "reasoning", "stderr", "tool_started", "tool_finished"]
    text: str | None = None
    tool_name: str | None = None
    value: JsonValue | None = None
    message_id: str | None = None
    message_complete: bool = False
    outcome: str = "completed"


@dataclass(frozen=True)
class DecodedCliOutput:
    """Parsed final output from CLI session."""
    content: str
    session_id: str | None = None


@dataclass(frozen=True)
class CollectedCliOutput:
    """Complete collected subprocess output."""
    stdout: bytes
    stderr: bytes
    returncode: int
    decoded: DecodedCliOutput


class CliStreamDecoder:
    """Incremental decoder for CLI output with runtime-specific parsing."""

    def __init__(self, runtime: str):
        from huddleroom.services.agent_response_stream import DisplayTextBuffer
        self.runtime = runtime
        self._stderr_display = DisplayTextBuffer()
        # Separate incremental decoders for stdout and stderr
        self.stdout_decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
        self.stderr_decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
        # Line buffer for parsing
        self.stdout_buffer = ""
        self.stderr_buffer = ""
        self._claude_message_id: str | None = None
        self._claude_partial_blocks: dict[int, tuple[str | None, int]] = {}
        self._claude_seen_uuids: set[str] = set()
        self._codex_items: set[tuple[str, str]] = set()

    def feed_stdout(self, chunk: bytes) -> list[CliDisplayUpdate]:
        """Process stdout chunk and return any readable updates."""
        self.stdout_buffer += self.stdout_decoder.decode(chunk, final=False)
        return self._parse_output_buffer()

    def feed_stderr(self, chunk: bytes) -> list[CliDisplayUpdate]:
        """Process stderr chunk and return any readable updates."""
        self.stderr_buffer += self.stderr_decoder.decode(chunk, final=False)
        return self._parse_error_buffer()

    def _parse_output_buffer(self) -> list[CliDisplayUpdate]:
        """Parse accumulated stdout buffer for updates."""
        if self.runtime == "claude_code":
            return self._parse_claude_output()
        if self.runtime in {"codex", "copilot", "opencode", "pi"}:
            return self._parse_json_output()
        else:
            # Plain runtimes: aider, custom
            return self._parse_plain_output()

    def _parse_error_buffer(self) -> list[CliDisplayUpdate]:
        """Parse accumulated stderr buffer for updates."""
        if self.runtime == "claude_code":
            return self._parse_claude_error()
        else:
            # Plain runtimes: stderr is always plain text
            return self._parse_plain_stderr()

    def _parse_claude_output(self) -> list[CliDisplayUpdate]:
        """Parse Claude stream-JSON lines from stdout."""
        updates = []
        lines = self.stdout_buffer.split("\n")
        # Keep incomplete last line in buffer
        self.stdout_buffer = lines[-1]

        for line in lines[:-1]:
            line = line.strip()
            if not line:
                continue
            updates.extend(self._parse_claude_json_line(line))
        return updates

    def _parse_claude_error(self) -> list[CliDisplayUpdate]:
        """Parse Claude error output (usually plain text)."""
        updates = []
        lines = self.stderr_buffer.split("\n")
        self.stderr_buffer = lines[-1]

        for line in lines[:-1]:
            if line:
                safe = self._stderr_display.feed("stderr", line + "\n")
                if safe:
                    updates.append(CliDisplayUpdate(kind="stderr", text=safe.rstrip("\n")))
        return updates

    def _parse_claude_json_line(self, line: str) -> list[CliDisplayUpdate]:
        """Parse a single Claude stream-JSON line."""
        updates = []
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            # Malformed line: keep for final parsing, emit no live update
            return []

        if not isinstance(data, dict):
            return []
        msg_type = data.get("type")

        if msg_type == "stream_event":
            event = data.get("event")
            if not isinstance(event, dict):
                return []
            if event.get("type") == "message_start":
                message = event.get("message")
                self._claude_message_id = message.get("id") if isinstance(message, dict) else None
                self._claude_partial_blocks.clear()
            elif event.get("type") == "content_block_start":
                block = event.get("content_block", {})
                block_type = block.get("type") if isinstance(block, dict) else None
                kind = "output" if block_type == "text" else "reasoning" if block_type == "thinking" else None
                self._claude_partial_blocks[event.get("index", 0)] = (kind, 0)
                if len(self._claude_partial_blocks) > 128:
                    self._claude_partial_blocks.pop(next(iter(self._claude_partial_blocks)))
            elif event.get("type") == "content_block_delta":
                delta = event.get("delta")
                if not isinstance(delta, dict):
                    return []
                kind, text = (
                    ("output", delta.get("text")) if delta.get("type") == "text_delta"
                    else ("reasoning", delta.get("thinking")) if delta.get("type") == "thinking_delta"
                    else (None, None)
                )
                if kind and isinstance(text, str) and text:
                    index = event.get("index", 0)
                    _, count = self._claude_partial_blocks.get(index, (kind, 0))
                    self._claude_partial_blocks[index] = (kind, count + len(text))
                    if len(self._claude_partial_blocks) > 128:
                        self._claude_partial_blocks.pop(next(iter(self._claude_partial_blocks)))
                    updates.append(CliDisplayUpdate(kind=kind, text=text))
            return updates

        if msg_type == "assistant":
            # Extract text from message content
            message = data.get("message", {})
            if not isinstance(message, dict):
                return []
            message_id = message.get("id")
            record_id = data.get("uuid")
            if isinstance(record_id, str):
                if record_id in self._claude_seen_uuids:
                    return []
                if len(self._claude_seen_uuids) >= 128:
                    self._claude_seen_uuids.clear()
                self._claude_seen_uuids.add(record_id)
            content = message.get("content", [])
            if not isinstance(content, list):
                return []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                kind = "output" if block_type == "text" else "reasoning" if block_type == "thinking" else None
                text = (block.get("thinking") or block.get("text")) if kind == "reasoning" else block.get("text")
                if kind and isinstance(text, str) and text:
                    # SDK full AssistantMessages are per-block envelopes without
                    # an index. Consume their streamed counterparts in block order;
                    # text equality never identifies a distinct block.
                    if message_id == self._claude_message_id:
                        index = next((index for index, (channel, _) in sorted(self._claude_partial_blocks.items())
                                      if channel == kind), None)
                        if index is not None:
                            _, count = self._claude_partial_blocks.pop(index)
                            text = text[count:]
                    if text:
                        updates.append(CliDisplayUpdate(kind=kind, text=text))

        elif msg_type == "tool_use":
            # Tool invocation: tool_started
            tool_name = data.get("name")
            tool_input = data.get("input")
            updates.append(
                CliDisplayUpdate(
                    kind="tool_started",
                    tool_name=tool_name,
                    value=tool_input,
                )
            )

        elif msg_type == "tool_result":
            # Tool completion: tool_finished
            tool_id = data.get("id")
            result_payload = data.get("content")
            updates.append(
                CliDisplayUpdate(
                    kind="tool_finished",
                    tool_name=tool_id,
                    value=result_payload,
                )
            )

        # Unknown message types: keep for final parsing, emit no live update
        return updates

    def _parse_plain_output(self) -> list[CliDisplayUpdate]:
        """Parse plain stdout as text lines."""
        updates = []
        lines = self.stdout_buffer.split("\n")
        # Keep incomplete last line in buffer
        self.stdout_buffer = lines[-1]

        for line in lines[:-1]:
            if line:  # Skip empty lines
                updates.append(CliDisplayUpdate(kind="output", text=line))
        return updates

    def _parse_json_output(self) -> list[CliDisplayUpdate]:
        """Render assistant text from known JSONL runtime events."""
        updates = []
        lines = self.stdout_buffer.split("\n")
        self.stdout_buffer = lines[-1]
        for line in lines[:-1]:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            if self.runtime == "codex":
                updates.extend(self._codex_updates(record))
                continue
            text = _json_live_text(record, self.runtime)
            if text:
                updates.append(CliDisplayUpdate(kind="output", text=text))
        return updates

    def _codex_updates(self, record: dict[str, Any]) -> list[CliDisplayUpdate]:
        """Read only exec JSON item lifecycle events and intended reasoning summaries."""
        event = record.get("type")
        item = record.get("item")
        if event not in {"item.started", "item.completed"} or not isinstance(item, dict):
            return []
        item_id = item.get("id") if isinstance(item.get("id"), str) else None
        if item_id and len(item_id.encode()) > 256:
            return []
        if item_id:
            key = (event, item_id)
            if key in self._codex_items or len(self._codex_items) >= 4096:
                return []
            self._codex_items.add(key)
        kind = item.get("type")
        if event == "item.completed" and kind in {"agent_message", "reasoning"}:
            text = item.get("text")
            if isinstance(text, str):
                return [CliDisplayUpdate(kind="output" if kind == "agent_message" else "reasoning",
                                         text=text, message_id=item_id, message_complete=True)]
        if kind not in {"command_execution", "mcp_tool_call", "web_search", "file_change"}:
            return []
        name = "mcp_tool_call"
        if kind == "mcp_tool_call":
            server, tool = item.get("server"), item.get("tool")
            if isinstance(server, str) and isinstance(tool, str):
                name = f"{server[:128]}.{tool[:128]}"
        else:
            name = kind
        keys = ("command", "arguments", "query", "changes") if event == "item.started" else (
            "aggregated_output", "exit_code", "result", "error", "query", "changes")
        value = {key: item[key] for key in keys if key in item}
        return [CliDisplayUpdate(kind="tool_started" if event == "item.started" else "tool_finished",
                                 tool_name=name, value=value,
                                 outcome="failed" if item.get("status") == "failed" else "completed")]

    def _parse_plain_stderr(self) -> list[CliDisplayUpdate]:
        """Parse plain stderr as text lines."""
        updates = []
        lines = self.stderr_buffer.split("\n")
        self.stderr_buffer = lines[-1]

        for line in lines[:-1]:
            if line:
                safe = self._stderr_display.feed("stderr", line + "\n")
                if safe:
                    updates.append(CliDisplayUpdate(kind="stderr", text=safe.rstrip("\n")))
        return updates

    def finish(self, stdout: bytes, stderr: bytes) -> DecodedCliOutput:
        """Finalize decoding with complete stdout/stderr and extract session ID."""
        if self.runtime == "claude_code":
            content, session_id = self._parse_claude_final(stdout)
        elif self.runtime in {"codex", "copilot", "opencode", "pi"}:
            try:
                content, session_id = parse_jsonl_final(
                    stdout.decode("utf-8", errors="ignore"), self.runtime
                )
            except ValueError:
                content, session_id = "", None
        else:
            # Plain runtimes: use decoded stdout and include stderr for diagnostics
            stdout_text = stdout.decode("utf-8", errors="ignore")
            from huddleroom.services.agent_response_stream import DisplayTextBuffer
            stderr_text = DisplayTextBuffer().feed("stderr", stderr.decode("utf-8", errors="ignore"), final=True)
            content = stdout_text
            if stderr_text:
                if content:
                    content += "\n" + stderr_text
                else:
                    content = stderr_text
            session_id = None

        return DecodedCliOutput(content=content, session_id=session_id)

    def _parse_claude_final(self, stdout: bytes) -> tuple[str, str | None]:
        """Parse Claude final result and session_id from stdout."""
        text = stdout.decode("utf-8", errors="ignore")
        try:
            result, session_id = parse_claude_final(text)
            return (result, session_id)
        except ValueError:
            # No valid envelope; return raw text for fallback
            return (text, None)


def parse_claude_final(raw: str) -> tuple[str, str | None]:
    """Parse Claude final result and session_id from NDJSON or legacy one-shot envelope.

    Raises ValueError if no valid result/envelope found.
    """
    text = raw.strip()
    if not text:
        raise ValueError("empty output")

    lines = text.split("\n")

    # Look for terminal {"type":"result",...} line in NDJSON
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict) and data.get("type") == "result":
                result = data.get("result", "")
                session_id = data.get("session_id")
                return (result, session_id)
        except json.JSONDecodeError:
            pass

    # Fall back: try to parse as legacy one-shot envelope (single JSON object)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            result = parsed.get("result", "")
            session_id = parsed.get("session_id")
            if result or session_id:
                return (result, session_id)
    except json.JSONDecodeError:
        pass

    # No valid envelope found
    raise ValueError("no valid result envelope")


def _pi_message_text(record: dict[str, Any]) -> str | None:
    message = record.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return None
    content = message.get("content")
    if not isinstance(content, list):
        return None
    text = "".join(
        block["text"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )
    return text or None


def _json_live_text(record: dict[str, Any], runtime: str) -> str | None:
    """Return only assistant text from a documented runtime event."""
    event_type = record.get("type")
    if runtime == "codex":
        if event_type != "item.completed":
            return None
        item = record.get("item")
        if not isinstance(item, dict):
            return None
        # Ignore error items (config warnings, etc.)
        if item.get("type") == "error":
            return None
        if item.get("type") != "agent_message":
            return None
        return item.get("text") if isinstance(item.get("text"), str) else None
    if runtime == "copilot":
        data = record.get("data")
        if event_type != "assistant.message_delta" or not isinstance(data, dict):
            return None
        return data.get("deltaContent") if isinstance(data.get("deltaContent"), str) else None
    if runtime == "opencode":
        part = record.get("part")
        if event_type != "text" or not isinstance(part, dict) or part.get("type") != "text":
            return None
        return part.get("text") if isinstance(part.get("text"), str) else None
    if runtime == "pi":
        event = record.get("assistantMessageEvent")
        if event_type != "message_update" or not isinstance(event, dict) or event.get("type") != "text_delta":
            return None
        return event.get("delta") if isinstance(event.get("delta"), str) else None
    return None


def _json_session_id(record: dict[str, Any], runtime: str) -> str | None:
    event_type = record.get("type")
    if runtime == "codex" and event_type == "thread.started":
        return record.get("thread_id") if isinstance(record.get("thread_id"), str) else None
    if runtime == "pi" and event_type == "session":
        return record.get("id") if isinstance(record.get("id"), str) else None
    if runtime == "opencode":
        return record.get("sessionID") if isinstance(record.get("sessionID"), str) else None
    if runtime == "copilot" and event_type == "result":
        for value in (record, record.get("data")):
            if isinstance(value, dict) and isinstance(value.get("sessionId"), str):
                return value["sessionId"]
    return None


def _jsonl_records(raw: str) -> list[dict[str, Any]]:
    records = []
    for line in raw.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    if not records:
        raise ValueError("no JSON output")
    return records


def parse_jsonl_session_id(raw: str, runtime: str) -> str | None:
    """Read a resumable session ID without treating transport output as an answer."""
    session_id = None
    for record in _jsonl_records(raw):
        session_id = _json_session_id(record, runtime) or session_id
    return session_id


def parse_jsonl_final(raw: str, runtime: str) -> tuple[str, str | None]:
    """Extract validated assistant output and a session ID from runtime JSONL."""
    records = _jsonl_records(raw)

    session_id = None
    fragments: list[str] = []
    final_text = None
    for record in records:
        session_id = _json_session_id(record, runtime) or session_id
        text = _json_live_text(record, runtime)
        if text:
            fragments.append(text)
        if runtime == "copilot" and record.get("type") == "assistant.message":
            data = record.get("data")
            final_text = data.get("content") if isinstance(data, dict) and isinstance(data.get("content"), str) else None
        elif runtime == "pi" and record.get("type") in {"message_end", "turn_end"}:
            final_text = _pi_message_text(record) or final_text
        elif runtime == "codex" and text is not None:
            # Earlier agent_messages are narration before tool calls; only the last is the answer.
            final_text = text

    content = final_text if final_text is not None else "".join(fragments)
    if not content:
        raise ValueError("no assistant output")
    return content, session_id


def is_resume_not_found(runtime: str, exit_code: int | None, stdout: str, stderr: str) -> bool:
    """True only when a failed CLI exit says the resumed session id does not exist.

    codex: marker on stderr. claude_code: marker on stderr, or in the `errors` of the
    terminal stream-json `result` record (never in arbitrary assistant/tool text).
    """
    if exit_code == 0 or exit_code is None:
        return False
    if runtime == "codex":
        return "no rollout found for thread id" in stderr
    if runtime == "claude_code":
        marker = "No conversation found with session ID"
        if marker in stderr:
            return True
        for line in stdout.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (isinstance(record, dict) and record.get("type") == "result"
                    and isinstance(record.get("errors"), list)
                    and any(isinstance(e, str) and marker in e for e in record["errors"])):
                return True
    return False


def decoder_for_runtime(runtime: str) -> CliStreamDecoder:
    """Create appropriate CLI decoder for the given runtime."""
    return CliStreamDecoder(runtime)


def parse_partial_cli(runtime: str, stdout: bytes) -> tuple[str, str | None]:
    """Best-effort (output, provider session id) from stdout of a CLI that never finished."""
    text = stdout.decode("utf-8", errors="ignore")
    if runtime == "claude_code":
        session_id, parts = None, []
        for line in text.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            if isinstance(record.get("session_id"), str):
                session_id = session_id or record["session_id"]
            message = record.get("message")
            if record.get("type") == "assistant" and isinstance(message, dict) and isinstance(message.get("content"), list):
                parts.extend(b["text"] for b in message["content"]
                             if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str))
        return "\n".join(parts), session_id
    if runtime in {"codex", "copilot", "opencode", "pi"}:
        try:
            return parse_jsonl_final(text, runtime)
        except ValueError:
            try:
                return "", parse_jsonl_session_id(text, runtime)
            except ValueError:
                return "", None
    return text, None


async def collect_cli_process(
    proc: Any, *, runtime: str, call: Any, partial: dict[str, bytes] | None = None
) -> CollectedCliOutput:
    """Collect CLI subprocess output concurrently, routing updates through call.

    Args:
        proc: asyncio subprocess.Popen-like object with stdout, stderr, wait()
        runtime: CLI runtime identifier (claude_code, codex, aider, custom)
        call: AgentResponseCall to route updates through
        partial: optional dict kept current with the stdout seen so far (key "stdout"),
            so a caller that abandons the collection can still recover early output

    Returns:
        CollectedCliOutput with exact bytes and decoded content

    Fallback: if proc lacks asyncio streams (mocked in tests), uses communicate().
    """
    decoder = decoder_for_runtime(runtime)

    # Check if proc has real asyncio streams
    if not (hasattr(proc, "stdout") and hasattr(proc, "stderr")):
        # Fallback for mocked objects without real streams
        stdout_bytes, stderr_bytes = await proc.communicate()
        returncode = proc.returncode
        decoded = decoder.finish(stdout_bytes, stderr_bytes)
        if returncode == 0 and decoded.content and hasattr(call, "snapshot_cli_response"):
            call.snapshot_cli_response(decoded.content)
        return CollectedCliOutput(
            stdout=stdout_bytes,
            stderr=stderr_bytes,
            returncode=returncode,
            decoded=decoded,
        )

    async def route(updates: list[CliDisplayUpdate]) -> None:
        for update in updates:
            if update.kind in {"output", "reasoning", "stderr"}:
                if update.message_complete and hasattr(call, "completed_message"):
                    await call.completed_message(update.kind, update.text or "", update.message_id)
                else:
                    await call.output(update.kind, update.text or "")
            elif update.kind == "tool_started":
                await call.tool_started(update.tool_name or "unknown", update.value)
            elif update.kind == "tool_finished":
                await call.tool_finished(update.tool_name or "unknown", update.outcome, update.value)

    async def read_stream(stream: Any, feed_fn) -> bytes:
        """Read concurrently, retaining exact transport bytes and completed boundaries."""
        accumulated = b""
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
            accumulated += chunk
            if partial is not None and feed_fn == decoder.feed_stdout:
                partial["stdout"] = accumulated
            await route(feed_fn(chunk))
        await route(feed_fn(b"\n"))
        return accumulated

    # Read both streams concurrently
    stdout_task = asyncio.create_task(
        read_stream(proc.stdout, decoder.feed_stdout)
    )
    stderr_task = asyncio.create_task(
        read_stream(proc.stderr, decoder.feed_stderr)
    )

    # Wait for both readers to reach EOF
    stdout_bytes, stderr_bytes = await asyncio.gather(stdout_task, stderr_task)

    # Now wait for process to exit
    returncode = await proc.wait()

    # Finalize decoder and extract session ID
    decoded = decoder.finish(stdout_bytes, stderr_bytes)
    if returncode == 0 and decoded.content and hasattr(call, "snapshot_cli_response"):
        call.snapshot_cli_response(decoded.content)

    return CollectedCliOutput(
        stdout=stdout_bytes,
        stderr=stderr_bytes,
        returncode=returncode,
        decoded=decoded,
    )
