import json
import logging
from pathlib import Path
from typing import Any

import litellm

from huddleroom.config import settings
from huddleroom.services.secret_redaction import redact_secrets

logger = logging.getLogger("huddleroom.llm")
_REQUEST_FIELDS = (
    "tools",
    "tool_choice",
    "temperature",
    "max_tokens",
    "response_format",
    "stop",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "seed",
)


def _redacted(value: Any) -> Any:
    def redact_value(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: redact_value(nested) for key, nested in item.items()}
        if isinstance(item, list):
            return [redact_value(nested) for nested in item]
        if isinstance(item, str):
            return redact_secrets(item)
        return item

    return redact_value(json.loads(json.dumps(value, default=str)))


def _emit(event: str, payload: dict[str, Any]) -> None:
    if not settings.debug:
        return
    try:
        logger.debug("%s %s", event, json.dumps(_redacted(payload), default=str))
    except Exception:
        logger.warning("LLM debug logging failed for %s", event)


def _is_completion(kwargs: dict[str, Any]) -> bool:
    return kwargs.get("call_type") == "acompletion"


class LlmDebugLogger(litellm.CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs):
        if not settings.debug or not _is_completion(kwargs):
            return
        payload = {"model": model, "messages": messages}
        payload.update({key: kwargs[key] for key in _REQUEST_FIELDS if key in kwargs})
        _emit("llm.api.request", payload)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        if not settings.debug or not _is_completion(kwargs):
            return
        try:
            response = response_obj.model_dump(mode="json")
        except Exception:
            try:
                response = str(response_obj)
            except Exception:
                response = "<unserializable response>"
        _emit("llm.api.response", response if isinstance(response, dict) else {"response": response})

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        if not settings.debug or not _is_completion(kwargs):
            return
        error = kwargs.get("exception") or response_obj
        try:
            payload = {
                "error_type": type(error).__name__,
                "error": str(error),
            }
        except Exception:
            payload = {"error_type": type(error).__name__, "error": "<unprintable error>"}
        _emit("llm.api.failure", payload)


def register_litellm_debug_logger() -> None:
    configure_debug_file_logging()
    if not any(isinstance(callback, LlmDebugLogger) for callback in litellm.callbacks):
        litellm.callbacks.append(LlmDebugLogger())


def configure_debug_file_logging() -> None:
    if not settings.debug or not settings.log_file:
        return
    try:
        filename = Path(settings.log_file).expanduser().resolve()
        rally_logger = logging.getLogger("huddleroom")
        if any(getattr(handler, "_rally_debug_log_file", None) == str(filename) for handler in rally_logger.handlers):
            return
        filename.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(filename, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        handler._rally_debug_log_file = str(filename)
        rally_logger.setLevel(logging.DEBUG)
        rally_logger.addHandler(handler)
    except Exception:
        logger.warning("Unable to configure debug log file")


def log_cli_exchange(
    *,
    prompt: str,
    session_id: str | None,
    runtime: str,
    response: str | None = None,
    error: str | BaseException | None = None,
    rally_session_id: str | None = None,
    meeting_id: str | None = None,
    agent_id: str | None = None,
) -> None:
    payload = {
        "prompt": prompt,
        "session_id": session_id,
        "runtime": runtime,
        "rally_session_id": rally_session_id,
        "meeting_id": meeting_id,
        "agent_id": agent_id,
    }
    payload["error" if error is not None else "response"] = error if error is not None else response
    _emit("llm.cli.failure" if error is not None else "llm.cli.exchange", payload)
