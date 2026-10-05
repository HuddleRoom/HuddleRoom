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

from huddleroom.services.secret_redaction import redact_secrets

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
        self.runtime = runtime
        # Separate incremental decoders for stdout and stderr
        self.stdout_decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
        self.stderr_decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
        # Line buffer for parsing
        self.stdout_buffer = ""
        self.stderr_buffer = ""

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
        if self.runtime in {"copilot", "opencode", "pi"}:
            return self._parse_json_output()
        else:
            # Plain runtimes: codex, aider, custom
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
                updates.append(CliDisplayUpdate(kind="stderr", text=redact_secrets(line)))
        return updates

    def _parse_claude_json_line(self, line: str) -> list[CliDisplayUpdate]:
        """Parse a single Claude stream-JSON line."""
        updates = []
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            # Malformed line: keep for final parsing, emit no live update
            return []

        msg_type = data.get("type")

        if msg_type == "assistant":
            # Extract text from message content
            message = data.get("message", {})
            content = message.get("content", [])
            for block in content:
                block_type = block.get("type")
                text = block.get("text")
                if block_type == "text" and text:
                    updates.append(CliDisplayUpdate(kind="output", text=text))
                elif block_type == "thinking" and text:
                    updates.append(CliDisplayUpdate(kind="reasoning", text=text))

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
            text = _json_live_text(record, self.runtime)
            if text:
                updates.append(CliDisplayUpdate(kind="output", text=text))
        return updates

    def _parse_plain_stderr(self) -> list[CliDisplayUpdate]:
        """Parse plain stderr as text lines."""
        updates = []
        lines = self.stderr_buffer.split("\n")
        self.stderr_buffer = lines[-1]

        for line in lines[:-1]:
            if line:
                updates.append(CliDisplayUpdate(kind="stderr", text=redact_secrets(line)))
        return updates

    def finish(self, stdout: bytes, stderr: bytes) -> DecodedCliOutput:
        """Finalize decoding with complete stdout/stderr and extract session ID."""
        if self.runtime == "claude_code":
            content, session_id = self._parse_claude_final(stdout)
        elif self.runtime in {"copilot", "opencode", "pi"}:
            try:
                content, session_id = parse_jsonl_final(
                    stdout.decode("utf-8", errors="ignore"), self.runtime
                )
            except ValueError:
                content, session_id = "", None
        else:
            # Plain runtimes: use decoded stdout and include stderr for diagnostics
            stdout_text = stdout.decode("utf-8", errors="ignore")
            stderr_text = redact_secrets(stderr.decode("utf-8", errors="ignore"))
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

    content = final_text or "".join(fragments)
    if not content:
        raise ValueError("no assistant output")
    return content, session_id


def decoder_for_runtime(runtime: str) -> CliStreamDecoder:
    """Create appropriate CLI decoder for the given runtime."""
    return CliStreamDecoder(runtime)


async def collect_cli_process(
    proc: Any, *, runtime: str, call: Any
) -> CollectedCliOutput:
    """Collect CLI subprocess output concurrently, routing updates through call.

    Args:
        proc: asyncio subprocess.Popen-like object with stdout, stderr, wait()
        runtime: CLI runtime identifier (claude_code, codex, aider, custom)
        call: AgentResponseCall to route updates through

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
        return CollectedCliOutput(
            stdout=stdout_bytes,
            stderr=stderr_bytes,
            returncode=returncode,
            decoded=decoded,
        )

    async def read_stream(stream: Any, feed_fn) -> bytes:
        """Read from stream in 4 KiB chunks, route updates through call."""
        accumulated = b""
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
            accumulated += chunk
            updates = feed_fn(chunk)
            for update in updates:
                if update.kind in {"output", "reasoning", "stderr"}:
                    await call.output(update.kind, update.text or "")
                elif update.kind == "tool_started":
                    await call.tool_started(update.tool_name or "unknown", update.value)
                elif update.kind == "tool_finished":
                    await call.tool_finished(
                        update.tool_name or "unknown", "completed", update.value
                    )
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

    return CollectedCliOutput(
        stdout=stdout_bytes,
        stderr=stderr_bytes,
        returncode=returncode,
        decoded=decoded,
    )
