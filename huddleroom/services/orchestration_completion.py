"""Completion boundary for orchestration control-plane calls."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import uuid4

import litellm

from huddleroom.config import Settings, settings, validate_cli_model
from huddleroom.services.cli_streaming import _json_session_id, _jsonl_records, terminate_process_group
from huddleroom.services.llm_debug_logging import log_orchestration_exchange
from huddleroom.services.secret_redaction import redact_secrets

CompletionFn = Callable[..., Awaitable[Any]]
API_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
CLI_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
_TIMEOUT = 120
_MAX_OUTPUT = 1_000_000


class OrchestrationBackendErrorKind(StrEnum):
    MISSING = "missing"
    UNSUPPORTED = "unsupported"
    UNAUTHENTICATED = "unauthenticated"
    TIMEOUT = "timeout"
    EXIT = "exit"
    PROTOCOL = "protocol"
    MALFORMED_OUTPUT = "malformed_output"


class OrchestrationBackendError(RuntimeError):
    def __init__(self, kind: OrchestrationBackendErrorKind, message: str) -> None:
        self.kind = kind
        super().__init__(redact_secrets(message))


def supported_orchestration_efforts(backend: str, model: str | None = None) -> frozenset[str]:
    if backend == "api":
        return _api_supported_efforts(model or settings.orchestration_model)
    if backend == "claude":
        return CLI_EFFORTS
    if backend == "codex":
        return CLI_EFFORTS - {"max"} if model else _codex_efforts()
    raise OrchestrationBackendError(OrchestrationBackendErrorKind.UNSUPPORTED, f"Unknown orchestration backend: {backend}")


def is_orchestration_backend_supported(backend: str) -> bool:
    return backend in {"api", "claude", "codex"}


def validate_orchestration_backend(config: Settings = settings) -> None:
    if config.orchestration_backend == "api":
        _validate_api_effort(config.orchestration_model, config.orchestration_effort)
        return
    if config.orchestration_backend not in {"claude", "codex"}:
        raise OrchestrationBackendError(OrchestrationBackendErrorKind.UNSUPPORTED, "Unknown orchestration backend.")
    executable = _executable(config.orchestration_backend)
    _validate_native_effort(config.orchestration_backend, config.orchestration_effort, config.orchestration_cli_model)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_authenticate(executable, config.orchestration_backend))


def get_orchestration_completion(completion_fn: CompletionFn | None = None) -> CompletionFn:
    return completion_fn or orchestration_completion


def orchestration_runtime_metadata(completion_fn: CompletionFn, api_model: str, config: Settings = settings) -> tuple[str, str]:
    if completion_fn is not orchestration_completion or config.orchestration_backend == "api":
        return "api", api_model
    return "cli_main", f"{config.orchestration_backend} CLI ({config.orchestration_cli_model or 'configured CLI default'})"


async def orchestration_completion(**request: Any) -> Any:
    backend = settings.orchestration_backend
    if backend == "api":
        _validate_api_effort(request.get("model") or settings.orchestration_model, settings.orchestration_effort)
        if settings.orchestration_effort is not None:
            request = {key: value for key, value in request.items() if key != "temperature"}
            request["reasoning_effort"] = settings.orchestration_effort
        return await litellm.acompletion(**request)
    if backend not in {"claude", "codex"}:
        raise OrchestrationBackendError(OrchestrationBackendErrorKind.UNSUPPORTED, "Unknown orchestration backend.")
    executable = _executable(backend)
    await _authenticate(executable, backend)
    _validate_native_effort(backend, settings.orchestration_effort, settings.orchestration_cli_model)
    meta = {"exchange_id": uuid4().hex, "backend": backend, "model": settings.orchestration_cli_model, "effort": settings.orchestration_effort, "session_id": None}
    log_orchestration_exchange("request", request=request, **meta)
    try:
        result = await _complete_cli(executable, backend, request, meta)
    except BaseException as error:
        log_orchestration_exchange("failure", error=error, **meta)
        raise
    log_orchestration_exchange("response", response=result, **meta)
    return result


def _executable(backend: str) -> str:
    executable = shutil.which(backend)
    if executable:
        return executable
    raise OrchestrationBackendError(OrchestrationBackendErrorKind.MISSING, f"The {backend} CLI is not installed or is not on PATH. Install it, rerun setup, or select API via LiteLLM.")


async def _authenticate(executable: str, backend: str) -> None:
    command = (executable, "auth", "status") if backend == "claude" else (executable, "login", "status")
    stdout, _ = await _run(command, None, backend=backend)
    if backend == "claude":
        try:
            authenticated = bool(json.loads(stdout).get("loggedIn"))
        except (json.JSONDecodeError, AttributeError):
            authenticated = False
    else:
        authenticated = bool(_.strip())
    if not authenticated:
        raise OrchestrationBackendError(OrchestrationBackendErrorKind.UNAUTHENTICATED, f"The {backend} CLI is not signed in. Sign in, rerun setup, or select API via LiteLLM.")


async def _complete_cli(executable: str, backend: str, request: dict[str, Any], meta: dict[str, Any] | None = None) -> dict[str, Any]:
    schema = _schema(request.get("tools"))
    prompt = "Return only the required JSON envelope for this request:\n" + json.dumps({key: request[key] for key in ("messages", "tools", "tool_choice", "response_format") if key in request}, default=str)
    if len(prompt.encode()) + len(json.dumps(schema).encode()) > _MAX_OUTPUT:
        raise OrchestrationBackendError(OrchestrationBackendErrorKind.MALFORMED_OUTPUT, "CLI request exceeded the allowed size.")
    with tempfile.TemporaryDirectory(prefix="huddleroom-orchestration-") as directory:
        if backend == "claude":
            command = [executable, "--print", "--output-format", "json", "--json-schema", json.dumps(schema), "--no-session-persistence", "--safe-mode", "--permission-prompts", "none", "--tools", ""]
            if model := _cli_model():
                command.extend(("--model", model))
            if settings.orchestration_effort is not None:
                _validate_cli_effort(settings.orchestration_effort)
                command.extend(("--effort", settings.orchestration_effort))
            stdout, _ = await _run(tuple(command), prompt, cwd=directory, timeout=_timeout(request), backend=backend)
            if meta is not None:
                meta["session_id"] = _cli_session_id(backend, stdout)
            envelope, usage = _claude_result(stdout)
        else:
            result_path = Path(directory) / "result.json"
            schema_path = Path(directory) / "schema.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            command = [executable, "exec", "--ephemeral", "--json", "--skip-git-repo-check", "-s", "read-only", "--output-schema", str(schema_path), "--output-last-message", str(result_path)]
            if model := _cli_model():
                command.extend(("-m", model))
            if settings.orchestration_effort is not None:
                _validate_cli_effort(settings.orchestration_effort)
                command.extend(("-c", f"model_reasoning_effort={settings.orchestration_effort}"))
            command.append("-")
            stdout, _ = await _run(tuple(command), prompt, cwd=directory, timeout=_timeout(request), artifact_path=result_path, backend=backend)
            if meta is not None:
                meta["session_id"] = _cli_session_id(backend, stdout)
            try:
                if result_path.stat().st_size > _MAX_OUTPUT:
                    raise ValueError("oversized result")
                with result_path.open("rb") as result_file:
                    raw_result = result_file.read(_MAX_OUTPUT + 1)
                if len(raw_result) > _MAX_OUTPUT:
                    raise ValueError("oversized result")
                envelope = json.loads(raw_result)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                raise OrchestrationBackendError(OrchestrationBackendErrorKind.MALFORMED_OUTPUT, "Codex returned no valid structured result.") from error
            usage = _codex_usage(stdout)
    content, calls = _envelope(envelope, request.get("tools"))
    result: dict[str, Any] = {"choices": [{"message": {"content": content, "tool_calls": calls}}]}
    if usage is not None:
        result["usage"] = usage
    return result


def strip_native_auth_env(env: dict[str, str]) -> dict[str, str]:
    """Drop OneCLI gateway/proxy/CA/placeholder-key vars and *_BASE_URL from env (native-login runtimes)."""
    from huddleroom.onecli import _PROTECTED_ENV_KEYS  # lazy: onecli is heavy (click, httpx)
    return {k: v for k, v in env.items() if k not in _PROTECTED_ENV_KEYS and not k.endswith("_BASE_URL")}


async def _run(command: tuple[str, ...], stdin: str | None, *, cwd: str | None = None, timeout: float = _TIMEOUT, artifact_path: Path | None = None, backend: str | None = None) -> tuple[str, str]:
    proc = None
    tasks: list[asyncio.Task[Any]] = []
    environment = dict(os.environ)
    # Map backend names to runtime names for settings lookup
    runtime_map = {"claude": "claude_code", "codex": "codex"}
    runtime = runtime_map.get(backend) if backend else None
    # Native-auth runtimes in onecli mode run on the user's local login: strip gateway/proxy/placeholder keys
    if settings.credential_mode == "onecli" and runtime in settings.onecli_native_auth_runtimes:
        environment = strip_native_auth_env(environment)
    # Always remove HuddleRoom/Rally context keys
    environment = {k: v for k, v in environment.items() if not k.startswith(("HUDDLEROOM_", "RALLY_"))}
    try:
        proc = await asyncio.create_subprocess_exec(*command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=cwd, env=environment, start_new_session=True)
        async def read(stream: Any) -> bytes:
            output = bytearray()
            while chunk := await stream.read(65536):
                output.extend(chunk)
                if len(output) > _MAX_OUTPUT:
                    raise OverflowError
            return bytes(output)
        async def write() -> None:
            if stdin is not None:
                proc.stdin.write(stdin.encode())
                await proc.stdin.drain()
            proc.stdin.close()
        async def monitor() -> None:
            while proc.returncode is None:
                if artifact_path is not None and artifact_path.exists() and artifact_path.stat().st_size > _MAX_OUTPUT:
                    raise OverflowError
                await asyncio.sleep(0.01)
        stdout_task = asyncio.create_task(read(proc.stdout))
        stderr_task = asyncio.create_task(read(proc.stderr))
        tasks = [stdout_task, stderr_task, asyncio.create_task(write()), asyncio.create_task(proc.wait()), asyncio.create_task(monitor())]
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
        stdout, stderr = stdout_task.result(), stderr_task.result()
    except FileNotFoundError as error:
        if proc is not None:
            await terminate_process_group(proc)
            raise OrchestrationBackendError(OrchestrationBackendErrorKind.PROTOCOL, "CLI process failed.") from error
        raise OrchestrationBackendError(OrchestrationBackendErrorKind.MISSING, "Configured CLI executable is unavailable.") from error
    except asyncio.CancelledError:
        if proc is not None:
            await terminate_process_group(proc)
        raise
    except asyncio.TimeoutError as error:
        if proc is not None:
            await terminate_process_group(proc)
        raise OrchestrationBackendError(OrchestrationBackendErrorKind.TIMEOUT, "CLI orchestration request timed out.") from error
    except OverflowError as error:
        if proc is not None:
            await terminate_process_group(proc)
        raise OrchestrationBackendError(OrchestrationBackendErrorKind.MALFORMED_OUTPUT, "CLI output exceeded the allowed size.") from error
    except Exception as error:
        if proc is not None:
            await terminate_process_group(proc)
        raise OrchestrationBackendError(OrchestrationBackendErrorKind.PROTOCOL, "CLI process failed.") from error
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    if len(stdout) > _MAX_OUTPUT or len(stderr) > _MAX_OUTPUT:
        await terminate_process_group(proc)
        raise OrchestrationBackendError(OrchestrationBackendErrorKind.MALFORMED_OUTPUT, "CLI output exceeded the allowed size.")
    if proc.returncode:
        detail = " ".join(redact_secrets(stderr.decode(errors="replace")).split())[-300:]
        raise OrchestrationBackendError(OrchestrationBackendErrorKind.EXIT, f"CLI exited unsuccessfully (code {proc.returncode})." + (f" {detail}" if detail else ""))
    return stdout.decode(errors="replace"), stderr.decode(errors="replace")


def _timeout(request: dict[str, Any]) -> float:
    value = request.get("timeout")
    return min(_TIMEOUT, float(value)) if isinstance(value, (int, float)) and value > 0 else _TIMEOUT


def _schema(tools: Any) -> dict[str, Any]:
    names = [tool.get("function", {}).get("name") for tool in tools or [] if isinstance(tool, dict)]
    names = [name for name in names if isinstance(name, str)]
    function = {"type": "object", "properties": {"name": {"type": "string", "enum": names}, "arguments": {"type": "string"}}, "required": ["name", "arguments"], "additionalProperties": False}
    call = {"type": "object", "properties": {"id": {"type": "string"}, "type": {"type": "string", "const": "function"}, "function": function}, "required": ["id", "type", "function"], "additionalProperties": False}
    calls = {"type": "array", "maxItems": 1, "items": call} if names else {"type": "array", "maxItems": 0, "items": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}
    return {"type": "object", "properties": {"content": {"type": "string"}, "tool_calls": calls}, "required": ["content", "tool_calls"], "additionalProperties": False}


def _claude_result(raw: str) -> tuple[Any, dict[str, int] | None]:
    try:
        result = json.loads(raw)
        envelope = result["structured_output"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise OrchestrationBackendError(OrchestrationBackendErrorKind.MALFORMED_OUTPUT, "Claude returned no valid structured result.") from error
    usage = result.get("usage") if isinstance(result, dict) else None
    keys = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    if not isinstance(usage, dict) or not all(isinstance(usage.get(key), int) for key in keys):
        return envelope, None
    prompt = usage["input_tokens"] + usage["cache_read_input_tokens"] + usage["cache_creation_input_tokens"]
    completion = usage["output_tokens"]
    return envelope, {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def _cli_session_id(backend: str, stdout: str) -> str | None:
    try:
        if backend == "claude":
            session_id = json.loads(stdout).get("session_id")
            return session_id if isinstance(session_id, str) else None
        return next((session_id for record in _jsonl_records(stdout) if (session_id := _json_session_id(record, "codex"))), None)
    except Exception:
        return None


def _codex_usage(raw: str) -> dict[str, int] | None:
    for line in reversed(raw.splitlines()):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        usage = event.get("usage") if isinstance(event, dict) and event.get("type") == "turn.completed" else None
        if isinstance(usage, dict) and all(isinstance(usage.get(key), int) for key in ("input_tokens", "output_tokens")):
            prompt = usage["input_tokens"]
            completion = usage["output_tokens"]
            return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
    return None


def _envelope(envelope: Any, tools: Any) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(envelope, dict) or not isinstance(envelope.get("content"), str) or not isinstance(envelope.get("tool_calls"), list):
        raise OrchestrationBackendError(OrchestrationBackendErrorKind.MALFORMED_OUTPUT, "CLI returned an invalid response envelope.")
    calls = envelope["tool_calls"]
    if len(calls) > 1:
        raise OrchestrationBackendError(OrchestrationBackendErrorKind.PROTOCOL, "CLI returned more than one tool call.")
    known = {tool.get("function", {}).get("name") for tool in tools or [] if isinstance(tool, dict)}
    if calls:
        call = calls[0]
        function = call.get("function") if isinstance(call, dict) else None
        arguments = function.get("arguments") if isinstance(function, dict) else None
        valid = isinstance(call, dict) and isinstance(call.get("id"), str) and call.get("type") == "function" and isinstance(function, dict) and function.get("name") in known and isinstance(arguments, str)
        if not valid:
            raise OrchestrationBackendError(OrchestrationBackendErrorKind.PROTOCOL, "CLI returned an unknown or invalid tool call.")
        try:
            if not isinstance(json.loads(arguments), dict):
                raise ValueError
        except (json.JSONDecodeError, ValueError) as error:
            raise OrchestrationBackendError(OrchestrationBackendErrorKind.PROTOCOL, "CLI tool arguments must be a JSON object.") from error
    return envelope["content"], calls


def _cli_model() -> str | None:
    try:
        return validate_cli_model(settings.orchestration_cli_model)
    except ValueError as error:
        raise OrchestrationBackendError(OrchestrationBackendErrorKind.UNSUPPORTED, f"Invalid orchestration CLI model: {error}.") from None


def _validate_cli_effort(effort: str) -> None:
    if effort not in CLI_EFFORTS:
        raise OrchestrationBackendError(OrchestrationBackendErrorKind.UNSUPPORTED, f"Unsupported CLI orchestration effort {effort!r}; select default.")


def _validate_native_effort(backend: str, effort: str | None, model: str | None = None) -> None:
    if effort is None:
        return
    if model or backend == "claude":
        _validate_cli_effort(effort)  # the CLI itself accepts or rejects it for the model
        if backend == "codex" and effort == "max":
            raise OrchestrationBackendError(OrchestrationBackendErrorKind.UNSUPPORTED, "Codex does not support effort 'max'; select another effort.")
        return
    if effort not in _codex_efforts():
        raise OrchestrationBackendError(OrchestrationBackendErrorKind.UNSUPPORTED, "Codex configured model does not advertise this effort; select default.")


def _codex_efforts() -> frozenset[str]:
    home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    try:
        config = tomllib.loads((home / "config.toml").read_text())
        if not isinstance(config, dict):
            return frozenset()
        model = config.get("model")
        profile = config.get("profile")
        if profile:
            profiles = config.get("profiles")
            selected = profiles.get(profile) if isinstance(profiles, dict) else None
            if not isinstance(selected, dict):
                return frozenset()
            model = selected.get("model", model)
        cache = json.loads((home / "models_cache.json").read_text())
        models = cache.get("models", []) if isinstance(cache, dict) else []
        if not isinstance(models, list):
            return frozenset()
        levels = next(item["supported_reasoning_levels"] for item in models if isinstance(item, dict) and item.get("slug") == model)
        return frozenset(item.get("effort") for item in levels if isinstance(item, dict) and item.get("effort") in CLI_EFFORTS)
    except (OSError, ValueError, StopIteration, KeyError, TypeError):
        return frozenset()


def _validate_api_effort(model: str, effort: str | None) -> None:
    if effort is None:
        return
    if effort not in API_EFFORTS or effort not in _api_supported_efforts(model):
        raise OrchestrationBackendError(OrchestrationBackendErrorKind.UNSUPPORTED, f"Model {model!r} does not report support for reasoning effort {effort!r}; select default.")


def _api_supported_efforts(model: str) -> frozenset[str]:
    try:
        params = litellm.get_supported_openai_params(model=model)
        metadata = litellm.model_cost.get(model) or litellm.model_cost.get(model.split("/", 1)[-1])
    except Exception:
        return frozenset()
    if not params or "reasoning_effort" not in params or not isinstance(metadata, dict):
        return frozenset()
    levels = metadata.get("reasoning_effort_levels")
    supported = {level for level in levels if isinstance(level, str) and level in API_EFFORTS} if isinstance(levels, (list, tuple, set, frozenset)) else set()
    for effort in API_EFFORTS:
        if metadata.get(f"supports_{effort}_reasoning_effort") is True:
            supported.add(effort)
        elif metadata.get(f"supports_{effort}_reasoning_effort") is False:
            supported.discard(effort)
    return frozenset(supported)
