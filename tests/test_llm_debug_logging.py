import json
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from huddleroom import cli
from huddleroom.services import llm_debug_logging as debug_logging


def payload_for(caplog, event):
    record = next(record for record in caplog.records if record.message.startswith(event + " "))
    return json.loads(record.message.removeprefix(event + " "))


@pytest.mark.asyncio
async def test_completion_callback_logs_allowlisted_redacted_request_and_response(monkeypatch, caplog):
    monkeypatch.setattr(debug_logging.settings, "debug", True)
    caplog.set_level(logging.DEBUG, logger="huddleroom.llm")
    callback = debug_logging.LlmDebugLogger()
    kwargs = {
        "call_type": "acompletion",
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "temperature": 0.2,
        "max_tokens": 50,
        "api_key": "sk-secret-value-1234567890",
        "headers": {"Authorization": "Bearer hidden"},
        "unknown_provider_internal": "exclude-me",
    }
    messages = [{"role": "user", "content": "token=private-value"}]

    callback.log_pre_api_call("openai/gpt-4o-mini", messages, kwargs)
    await callback.async_log_success_event(
        kwargs,
        SimpleNamespace(model_dump=lambda mode="json": {"choices": [{"message": {"content": "done"}}]}),
        None,
        None,
    )

    request = payload_for(caplog, "llm.api.request")
    response = payload_for(caplog, "llm.api.response")
    assert request["model"] == "openai/gpt-4o-mini"
    assert request["messages"][0]["content"] == "token=[REDACTED]"
    assert request["tools"] == kwargs["tools"]
    assert request["temperature"] == 0.2
    assert request["max_tokens"] == 50
    assert "api_key" not in request
    assert "headers" not in request
    assert "unknown_provider_internal" not in request
    assert response["choices"][0]["message"]["content"] == "done"


@pytest.mark.asyncio
async def test_callback_ignores_disabled_and_non_completion_events(monkeypatch, caplog):
    callback = debug_logging.LlmDebugLogger()
    caplog.set_level(logging.DEBUG, logger="huddleroom.llm")
    monkeypatch.setattr(debug_logging.settings, "debug", False)
    callback.log_pre_api_call("model", [{"role": "user", "content": "hidden"}], {"call_type": "acompletion"})
    monkeypatch.setattr(debug_logging.settings, "debug", True)
    callback.log_pre_api_call("model", [{"role": "user", "content": "embedding input"}], {"call_type": "aembedding"})
    assert not [record for record in caplog.records if record.name == "huddleroom.llm"]


@pytest.mark.asyncio
async def test_completion_failure_is_redacted_and_logging_errors_do_not_escape(monkeypatch, caplog):
    monkeypatch.setattr(debug_logging.settings, "debug", True)
    caplog.set_level(logging.DEBUG, logger="huddleroom.llm")
    callback = debug_logging.LlmDebugLogger()
    await callback.async_log_failure_event(
        {"call_type": "acompletion", "exception": RuntimeError("api_key=secret")}, None, None, None
    )
    assert payload_for(caplog, "llm.api.failure") == {
        "error_type": "RuntimeError",
        "error": "api_key=[REDACTED]",
    }
    await callback.async_log_success_event(
        {"call_type": "acompletion"},
        SimpleNamespace(model_dump=lambda mode="json": (_ for _ in ()).throw(ValueError("bad serialization"))),
        None,
        None,
    )


@pytest.mark.asyncio
async def test_failure_callback_does_not_propagate_unprintable_exception(monkeypatch):
    class UnprintableError(Exception):
        def __str__(self):
            raise RuntimeError("cannot stringify error")

    monkeypatch.setattr(debug_logging.settings, "debug", True)
    await debug_logging.LlmDebugLogger().async_log_failure_event(
        {"call_type": "acompletion", "exception": UnprintableError()}, None, None, None
    )


@pytest.mark.asyncio
async def test_callback_does_not_propagate_redaction_errors(monkeypatch):
    def redaction_failure(_value):
        raise RuntimeError("redaction failed")

    monkeypatch.setattr(debug_logging.settings, "debug", True)
    monkeypatch.setattr(debug_logging, "redact_secrets", redaction_failure)
    debug_logging.LlmDebugLogger().log_pre_api_call(
        "model", [{"role": "user", "content": "hidden"}], {"call_type": "acompletion"}
    )


@pytest.mark.asyncio
async def test_litellm_acompletion_logs_request(monkeypatch, caplog):
    monkeypatch.setattr(debug_logging.settings, "debug", True)
    monkeypatch.setattr(debug_logging.litellm, "callbacks", [])
    caplog.set_level(logging.DEBUG, logger="huddleroom.llm")
    debug_logging.register_litellm_debug_logger()

    await debug_logging.litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        mock_response="pong",
    )

    assert payload_for(caplog, "llm.api.request")["messages"] == [{"role": "user", "content": "hello"}]


def test_registration_is_idempotent(monkeypatch):
    callbacks = []
    monkeypatch.setattr(debug_logging.litellm, "callbacks", callbacks)
    debug_logging.register_litellm_debug_logger()
    debug_logging.register_litellm_debug_logger()
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], debug_logging.LlmDebugLogger)


def test_debug_events_are_written_to_configured_file_only_when_enabled(monkeypatch, tmp_path):
    log_file = tmp_path / "logs" / "rally-debug.log"
    rally_logger = logging.getLogger("huddleroom")
    original_handlers = list(rally_logger.handlers)
    try:
        monkeypatch.setattr(debug_logging.settings, "debug", True)
        monkeypatch.setattr(
            type(debug_logging.settings), "log_file", property(lambda _settings: str(log_file)), raising=False
        )
        monkeypatch.setattr(debug_logging.settings, "debug", False)
        debug_logging.configure_debug_file_logging()
        assert not log_file.exists()
        monkeypatch.setattr(debug_logging.settings, "debug", True)
        debug_logging.configure_debug_file_logging()
        debug_logging.LlmDebugLogger().log_pre_api_call(
            "openai/gpt-4o-mini",
            [{"role": "user", "content": "api_key=private-value"}],
            {"call_type": "acompletion"},
        )
        contents = log_file.read_text()
        assert "llm.api.request" in contents
        assert "api_key=[REDACTED]" in contents
        assert "private-value" not in contents
    finally:
        for handler in list(rally_logger.handlers):
            if handler not in original_handlers:
                rally_logger.removeHandler(handler)
                handler.close()


def test_cli_helper_emits_nothing_when_debug_disabled(monkeypatch, caplog):
    monkeypatch.setattr(debug_logging.settings, "debug", False)
    caplog.set_level(logging.DEBUG, logger="huddleroom.llm")
    debug_logging.log_cli_exchange(
        prompt="hidden", session_id="provider-12345678", response="hidden",
        runtime="claude_code", rally_session_id="rally-id",
    )
    assert not [record for record in caplog.records if record.name == "huddleroom.llm"]


@pytest.mark.parametrize(("debug", "expected_level"), [(True, "DEBUG"), (False, "INFO")])
def test_serve_configures_single_rally_logger(monkeypatch, debug, expected_level):
    monkeypatch.setattr("huddleroom.config.settings.debug", debug)
    with patch("uvicorn.run") as run:
        cli.serve.callback("127.0.0.1", 8001, False)
    config = run.call_args.kwargs["log_config"]
    assert config["loggers"]["huddleroom"] == {
        "handlers": ["default"],
        "level": expected_level,
        "propagate": False,
    }
