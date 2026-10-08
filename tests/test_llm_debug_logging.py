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

    callback.log_pre_api_call("openai/gpt-6.1-sol", messages, kwargs)
    await callback.async_log_success_event(
        kwargs,
        SimpleNamespace(model_dump=lambda mode="json": {"choices": [{"message": {"content": "done"}}]}),
        None,
        None,
    )

    request = payload_for(caplog, "llm.api.request")
    response = payload_for(caplog, "llm.api.response")
    assert request["model"] == "openai/gpt-6.1-sol"
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
        model="openai/gpt-6.1-sol",
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
            "openai/gpt-6.1-sol",
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


# --- orchestration CLI exchange logging ---
import asyncio

from huddleroom.services import orchestration_completion as oc

_REAL_AUTH = oc._authenticate
_REQ = {
    "messages": [{"role": "user", "content": "hello there"}],
    "tools": [{"type": "function", "function": {"name": "lookup"}}],
}


def _setup_cli(monkeypatch, caplog, stub, debug=True):
    monkeypatch.setattr(debug_logging.settings, "debug", debug)
    monkeypatch.setattr(oc.settings, "orchestration_backend", "claude")
    monkeypatch.setattr(oc.settings, "orchestration_cli_model", "sonnet")
    monkeypatch.setattr(oc.settings, "orchestration_effort", None)
    monkeypatch.setattr(oc, "_executable", lambda b: "/bin/claude")

    async def no_auth(*_a):
        return None

    monkeypatch.setattr(oc, "_authenticate", no_auth)
    monkeypatch.setattr(oc, "_complete_cli", stub)
    caplog.set_level(logging.DEBUG, logger="huddleroom.llm")


def _orch_records(caplog):
    return [r for r in caplog.records if r.name == "huddleroom.llm" and r.message.startswith("llm.orchestration.")]


@pytest.mark.asyncio
async def test_orchestration_cli_success_logs_request_and_response(monkeypatch, caplog):
    result = {"choices": [{"message": {"content": "ok"}}]}

    async def stub(*_a):
        return result

    _setup_cli(monkeypatch, caplog, stub)
    assert await oc.orchestration_completion(**_REQ) is result
    req = payload_for(caplog, "llm.orchestration.request")
    resp = payload_for(caplog, "llm.orchestration.response")
    assert req["exchange_id"] == resp["exchange_id"]
    assert req["backend"] == "claude" and req["model"] == "sonnet"
    assert req["messages"] == _REQ["messages"] and req["tools"] == _REQ["tools"]
    assert resp["response"] == result


@pytest.mark.asyncio
async def test_orchestration_cli_failure_logged_and_same_error_reraised(monkeypatch, caplog):
    error = oc.OrchestrationBackendError(oc.OrchestrationBackendErrorKind.TIMEOUT, "slow")

    async def stub(*_a):
        raise error

    _setup_cli(monkeypatch, caplog, stub)
    with pytest.raises(oc.OrchestrationBackendError) as info:
        await oc.orchestration_completion(**_REQ)
    assert info.value is error
    failure = payload_for(caplog, "llm.orchestration.failure")
    assert failure["error_type"] == "OrchestrationBackendError"
    assert "timeout" in str(failure["kind"]).lower()
    assert payload_for(caplog, "llm.orchestration.request")["exchange_id"] == failure["exchange_id"]


@pytest.mark.asyncio
async def test_orchestration_cli_cancellation_propagates_and_is_logged(monkeypatch, caplog):
    async def stub(*_a):
        raise asyncio.CancelledError()

    _setup_cli(monkeypatch, caplog, stub)
    with pytest.raises(asyncio.CancelledError):
        await oc.orchestration_completion(**_REQ)
    assert payload_for(caplog, "llm.orchestration.failure")["error_type"] == "CancelledError"


@pytest.mark.asyncio
async def test_orchestration_cli_debug_disabled_logs_nothing(monkeypatch, caplog):
    result = {"ok": 1}

    async def stub(*_a):
        return result

    _setup_cli(monkeypatch, caplog, stub, debug=False)
    assert await oc.orchestration_completion(**_REQ) is result
    assert not [r for r in caplog.records if r.name == "huddleroom.llm"]


@pytest.mark.asyncio
async def test_orchestration_cli_redacts_request_and_response(monkeypatch, caplog):
    async def stub(*_a):
        return {"content": "leak api_key=sk-response-secret-1234567890"}

    _setup_cli(monkeypatch, caplog, stub)
    await oc.orchestration_completion(messages=[{"role": "user", "content": "api_key=sk-request-secret-1234567890"}])
    assert "sk-request-secret" not in caplog.text and "sk-response-secret" not in caplog.text
    assert "[REDACTED]" in caplog.text


@pytest.mark.asyncio
async def test_orchestration_cli_logging_failures_never_break_call(monkeypatch, caplog):
    result = {"ok": 1}
    error = oc.OrchestrationBackendError(oc.OrchestrationBackendErrorKind.TIMEOUT, "slow")

    async def ok(*_a):
        return result

    async def bad(*_a):
        raise error

    _setup_cli(monkeypatch, caplog, ok)

    def boom(_v):
        raise RuntimeError("redaction broke")

    monkeypatch.setattr(debug_logging, "redact_secrets", boom)
    assert await oc.orchestration_completion(**_REQ) is result
    monkeypatch.setattr(oc, "_complete_cli", bad)
    with pytest.raises(oc.OrchestrationBackendError) as info:
        await oc.orchestration_completion(**_REQ)
    assert info.value is error


@pytest.mark.asyncio
async def test_orchestration_cli_unprintable_error_reraises_original(monkeypatch, caplog):
    class Unprintable(Exception):
        def __str__(self):
            raise RuntimeError("no str")

    error = Unprintable()

    async def stub(*_a):
        raise error

    _setup_cli(monkeypatch, caplog, stub)
    with pytest.raises(Unprintable) as info:
        await oc.orchestration_completion(**_REQ)
    assert info.value is error


@pytest.mark.asyncio
async def test_orchestration_cli_request_metadata_is_logged(monkeypatch, caplog):
    async def stub(*_a):
        return {}

    _setup_cli(monkeypatch, caplog, stub)
    await oc.orchestration_completion(**_REQ, metadata={"goal_id": "g-1"})
    assert payload_for(caplog, "llm.orchestration.request")["metadata"] == {"goal_id": "g-1"}


@pytest.mark.asyncio
async def test_orchestration_auth_check_is_not_logged(monkeypatch, caplog):
    async def stub(*_a):
        raise AssertionError("must not run")

    _setup_cli(monkeypatch, caplog, stub)
    monkeypatch.setattr(oc, "_authenticate", _REAL_AUTH)

    async def fake_run(*_a, **_k):
        return '{"loggedIn": false}', ""

    monkeypatch.setattr(oc, "_run", fake_run)
    with pytest.raises(oc.OrchestrationBackendError) as info:
        await oc.orchestration_completion(**_REQ)
    assert info.value.kind == oc.OrchestrationBackendErrorKind.UNAUTHENTICATED
    assert not _orch_records(caplog)


@pytest.mark.asyncio
async def test_orchestration_api_backend_not_logged_by_cli_hook(monkeypatch, caplog):
    monkeypatch.setattr(debug_logging.settings, "debug", True)
    monkeypatch.setattr(oc.settings, "orchestration_backend", "api")
    monkeypatch.setattr(oc.settings, "orchestration_effort", None)
    caplog.set_level(logging.DEBUG, logger="huddleroom.llm")
    sentinel = object()

    async def fake(**_k):
        return sentinel

    monkeypatch.setattr(oc.litellm, "acompletion", fake)
    assert await oc.orchestration_completion(model="m", messages=[]) is sentinel
    assert not _orch_records(caplog)


_REAL_COMPLETE_CLI = oc._complete_cli
_OK_ENVELOPE = {"content": "ok", "tool_calls": []}


def _setup_real_cli(monkeypatch, caplog, backend, run):
    _setup_cli(monkeypatch, caplog, None)
    monkeypatch.setattr(oc, "_complete_cli", _REAL_COMPLETE_CLI)
    monkeypatch.setattr(oc.settings, "orchestration_backend", backend)
    monkeypatch.setattr(oc.settings, "orchestration_cli_model", None)
    monkeypatch.setattr(oc, "_run", run)


@pytest.mark.asyncio
async def test_orchestration_claude_session_id_logged_on_response(monkeypatch, caplog):
    async def run(*_a, **_k):
        return json.dumps({"session_id": "sess-123", "structured_output": _OK_ENVELOPE}), ""

    _setup_real_cli(monkeypatch, caplog, "claude", run)
    result = await oc.orchestration_completion(**_REQ)
    assert result["choices"][0]["message"]["content"] == "ok"
    assert payload_for(caplog, "llm.orchestration.request")["session_id"] is None
    assert payload_for(caplog, "llm.orchestration.response")["session_id"] == "sess-123"


@pytest.mark.asyncio
async def test_orchestration_claude_malformed_failure_carries_session_id(monkeypatch, caplog):
    async def run(*_a, **_k):
        return json.dumps({"session_id": "sess-456"}), ""

    _setup_real_cli(monkeypatch, caplog, "claude", run)
    with pytest.raises(oc.OrchestrationBackendError) as info:
        await oc.orchestration_completion(**_REQ)
    assert info.value.kind == oc.OrchestrationBackendErrorKind.MALFORMED_OUTPUT
    failure = payload_for(caplog, "llm.orchestration.failure")
    assert failure["session_id"] == "sess-456"
    assert payload_for(caplog, "llm.orchestration.request")["session_id"] is None


@pytest.mark.asyncio
async def test_orchestration_codex_thread_id_logged_as_session_id(monkeypatch, caplog):
    async def run(*_a, artifact_path=None, **_k):
        artifact_path.write_text(json.dumps(_OK_ENVELOPE), encoding="utf-8")
        return '{"type":"thread.started","thread_id":"th-1"}\n{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":2}}', ""

    _setup_real_cli(monkeypatch, caplog, "codex", run)
    result = await oc.orchestration_completion(**_REQ)
    assert result["choices"][0]["message"]["content"] == "ok"
    assert payload_for(caplog, "llm.orchestration.request")["session_id"] is None
    assert payload_for(caplog, "llm.orchestration.response")["session_id"] == "th-1"


@pytest.mark.parametrize("backend", ["claude", "codex"])
@pytest.mark.parametrize("stdout", ["not json at all", "", "[1, 2]", "{broken\n{also broken"])
def test_cli_session_id_garbage_is_none(backend, stdout):
    assert oc._cli_session_id(backend, stdout) is None
