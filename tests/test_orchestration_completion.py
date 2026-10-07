import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest


def _default_completion_services(completion_fn=None):
    from huddleroom.services.orchestration_agent_definition_analyzer import AgentDefinitionSemanticAnalyzer
    from huddleroom.services.orchestration_conversation_investigation import ConversationInvestigationService
    from huddleroom.services.orchestration_conversation_service import OrchestrationConversationService
    from huddleroom.services.orchestration_effectiveness_analyzer import EffectivenessAnalyzer
    from huddleroom.services.orchestration_goal_analyzer import GoalClarificationAnalyzer
    from huddleroom.services.orchestration_llm_decision_adapter import OrchestrationDecisionAdapter
    from huddleroom.services.orchestration_manager_analyzer import ManagerSelectionAnalyzer
    from huddleroom.services.orchestration_project_advisor_service import OrchestrationProjectAdvisorService
    from huddleroom.services.orchestration_supervision_analyzer import OrchestrationSupervisionAnalyzer
    from huddleroom.services.orchestration_team_hierarchy_analyzer import TeamHierarchyAnalyzer

    return [
        OrchestrationDecisionAdapter(completion_fn=completion_fn),
        AgentDefinitionSemanticAnalyzer(completion_fn=completion_fn),
        EffectivenessAnalyzer(completion_fn=completion_fn),
        GoalClarificationAnalyzer(completion_fn=completion_fn),
        ManagerSelectionAnalyzer(completion_fn=completion_fn),
        OrchestrationSupervisionAnalyzer(completion_fn=completion_fn),
        TeamHierarchyAnalyzer(completion_fn=completion_fn),
        OrchestrationConversationService(completion_fn=completion_fn),
        ConversationInvestigationService(completion_fn=completion_fn),
        OrchestrationProjectAdvisorService(completion_fn=completion_fn),
    ]


@pytest.mark.asyncio
async def test_all_control_plane_defaults_route_api_or_selected_cli_boundary(monkeypatch):
    """All ten control-plane constructors retain one selected API or CLI boundary."""
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    api_calls = []

    async def api_completion(**request):
        api_calls.append(request)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(orchestration_completion.litellm, "acompletion", api_completion)
    monkeypatch.setattr(settings, "orchestration_backend", "api")
    api_services = _default_completion_services()
    for service in api_services:
        assert service._completion_fn is orchestration_completion.orchestration_completion
        await service._completion_fn(model="provider/model", messages=[])
    assert len(api_calls) == 10

    for backend in ("claude", "codex"):
        monkeypatch.setattr(settings, "orchestration_backend", backend)
        cli_services = _default_completion_services()
        for service in cli_services:
            assert service._completion_fn is orchestration_completion.orchestration_completion
            assert orchestration_completion.orchestration_runtime_metadata(
                service._completion_fn, "provider/model"
            ) == ("cli_main", f"{backend} CLI (configured CLI default)")


def test_all_control_plane_constructors_preserve_explicit_completion_injection():
    from huddleroom.services import orchestration_completion

    async def injected(**_request):
        return None

    for service in _default_completion_services(injected):
        assert service._completion_fn is injected
        assert orchestration_completion.orchestration_runtime_metadata(
            service._completion_fn, "test/model"
        ) == ("api", "test/model")


@pytest.mark.asyncio
async def test_api_backend_delegates_request_unchanged(monkeypatch):
    from huddleroom.config import settings
    from huddleroom.services.orchestration_completion import get_orchestration_completion

    monkeypatch.setattr(settings, "orchestration_backend", "api")
    seen = {}

    async def completion(**request):
        seen.update(request)
        return {"choices": [{"message": {"content": "ok"}}]}

    result = await get_orchestration_completion(completion)(
        model="openai/gpt-6.1-sol", messages=[{"role": "user", "content": "hello"}], temperature=0
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert seen == {
        "model": "openai/gpt-6.1-sol",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
    }


@pytest.mark.asyncio
async def test_missing_cli_backend_never_falls_back_to_api(monkeypatch):
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(settings, "orchestration_backend", "codex")

    async def api_call(**_request):
        raise AssertionError("CLI mode must never fall back to LiteLLM")

    monkeypatch.setattr(orchestration_completion.litellm, "acompletion", api_call)

    monkeypatch.setattr(orchestration_completion, "shutil", SimpleNamespace(which=lambda _name: None), raising=False)

    with pytest.raises(orchestration_completion.OrchestrationBackendError) as error:
        await orchestration_completion.orchestration_completion(messages=[])

    assert error.value.kind == orchestration_completion.OrchestrationBackendErrorKind.MISSING


@pytest.mark.asyncio
async def test_claude_gateway_preflights_auth_and_normalizes_native_envelope(monkeypatch, tmp_path):
    """Claude uses its authenticated print contract without model or effort overrides by default."""
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(settings, "orchestration_backend", "claude")
    monkeypatch.setattr(settings, "orchestration_effort", None)
    calls = _install_fake_cli(
        monkeypatch,
        orchestration_completion,
        "claude",
        tmp_path / "claude",
        lambda argv: (
            b'{"loggedIn": true}' if argv[-2:] == ("auth", "status") else
            b'{"structured_output":{"content":"Claude result","tool_calls":[]},'
            b'"usage":{"input_tokens":3,"output_tokens":4,'
            b'"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}'
        ),
    )

    orchestration_completion.validate_orchestration_backend()
    result = await orchestration_completion.orchestration_completion(
        model="ignored/request-model", messages=[{"role": "user", "content": "hello"}]
    )

    assert calls[0][0][-2:] == ("auth", "status")
    argv, stdin = next((argv, stdin) for argv, stdin in calls if "--print" in argv)
    assert {"--print", "--output-format", "json", "--no-session-persistence"} <= set(argv)
    assert "--model" not in argv
    assert "--effort" not in argv
    assert b"hello" in stdin
    assert result == {
        "choices": [{"message": {"content": "Claude result", "tool_calls": []}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }


@pytest.mark.asyncio
async def test_codex_gateway_preflights_auth_reads_artifact_and_normalizes_one_known_tool(monkeypatch, tmp_path):
    """Codex reads the ephemeral result artifact and permits one declared function call."""
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(settings, "orchestration_backend", "codex")
    monkeypatch.setattr(settings, "orchestration_effort", None)
    envelope = {
        "content": "",
        "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": '{"q":"x"}'}}],
    }

    def codex_response(argv):
        if argv[-2:] == ("login", "status"):
            return {"stderr": b"Logged in with ChatGPT", "returncode": 0}
        output_path = Path(argv[argv.index("--output-last-message") + 1])
        output_path.write_text(json.dumps(envelope))
        return b'{"type":"turn.completed","usage":{"input_tokens":5,"output_tokens":2}}\n'

    calls = _install_fake_cli(
        monkeypatch, orchestration_completion, "codex", tmp_path / "codex", codex_response
    )

    orchestration_completion.validate_orchestration_backend()
    result = await orchestration_completion.orchestration_completion(
        model="ignored/request-model",
        messages=[{"role": "user", "content": "lookup x"}],
        tools=[{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
    )

    assert calls[0][0][-2:] == ("login", "status")
    argv, stdin = next((argv, stdin) for argv, stdin in calls if "exec" in argv)
    assert {"exec", "--ephemeral", "--json", "--output-schema", "--output-last-message"} <= set(argv)
    assert "--model" not in argv
    assert "model_reasoning_effort=" not in " ".join(argv)
    assert b"lookup x" in stdin
    message = result["choices"][0]["message"]
    assert message["content"] == ""
    assert message["tool_calls"][0] == {
        "id": "call-1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"q":"x"}'},
    }
    assert result["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}


@pytest.mark.asyncio
async def test_cli_gateway_omits_usage_when_native_output_cannot_prove_it(monkeypatch, tmp_path):
    """CLI backends must not invent token usage from absent native accounting."""
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(settings, "orchestration_backend", "claude")
    monkeypatch.setattr(settings, "orchestration_effort", None)
    _install_fake_cli(
        monkeypatch,
        orchestration_completion,
        "claude",
        tmp_path / "claude",
        lambda argv: (
            b'{"loggedIn":true}' if argv[-2:] == ("auth", "status")
            else b'{"structured_output":{"content":"No usage","tool_calls":[]}}'
        ),
    )

    result = await orchestration_completion.orchestration_completion(messages=[{"role": "user", "content": "hello"}])

    assert result["choices"][0]["message"]["content"] == "No usage"
    assert "usage" not in result


def test_cli_gateway_rejects_failed_authentication_before_a_completion(monkeypatch, tmp_path):
    """A present but logged-out native CLI is an authenticated-preflight failure."""
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(settings, "orchestration_backend", "claude")
    calls = _install_fake_cli(
        monkeypatch,
        orchestration_completion,
        "claude",
        tmp_path / "claude",
        lambda _argv: b'{"loggedIn":false}',
    )

    with pytest.raises(orchestration_completion.OrchestrationBackendError) as error:
        orchestration_completion.validate_orchestration_backend()

    assert error.value.kind == orchestration_completion.OrchestrationBackendErrorKind.UNAUTHENTICATED
    assert [argv for argv, _stdin in calls] == [(str(tmp_path / "claude"), "auth", "status")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_calls",
    [
        [{"id": "call-1", "type": "function", "function": {"name": "unknown", "arguments": "{}"}}],
        [
            {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
            {"id": "call-2", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
        ],
        [{"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "[]"}}],
    ],
)
async def test_cli_gateway_rejects_unknown_multiple_or_nonobject_tool_arguments(monkeypatch, tmp_path, tool_calls):
    """Only one requested function with an object-valued JSON arguments string is accepted."""
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(settings, "orchestration_backend", "claude")
    envelope = json.dumps({"content": "", "tool_calls": tool_calls})
    _install_fake_cli(
        monkeypatch,
        orchestration_completion,
        "claude",
        tmp_path / "claude",
        lambda argv: (
            b'{"loggedIn":true}' if argv[-2:] == ("auth", "status")
            else json.dumps({"structured_output": json.loads(envelope)}).encode()
        ),
    )

    with pytest.raises(orchestration_completion.OrchestrationBackendError) as error:
        await orchestration_completion.orchestration_completion(
            messages=[{"role": "user", "content": "lookup"}],
            tools=[{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        )

    assert error.value.kind == orchestration_completion.OrchestrationBackendErrorKind.PROTOCOL


@pytest.mark.asyncio
async def test_cli_gateway_redacts_native_failure_stderr(monkeypatch, tmp_path):
    """Native failure diagnostics remain actionable without exposing process secrets."""
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(settings, "orchestration_backend", "claude")
    _install_fake_cli(
        monkeypatch,
        orchestration_completion,
        "claude",
        tmp_path / "claude",
        lambda argv: (
            b'{"loggedIn":true}' if argv[-2:] == ("auth", "status")
            else {"stderr": b"native failure OPENAI_API_KEY=super-secret", "returncode": 23}
        ),
    )

    with pytest.raises(orchestration_completion.OrchestrationBackendError) as error:
        await orchestration_completion.orchestration_completion(messages=[{"role": "user", "content": "hello"}])

    assert error.value.kind == orchestration_completion.OrchestrationBackendErrorKind.EXIT
    assert "super-secret" not in str(error.value)
    assert "native failure" in str(error.value)
    assert "23" in str(error.value)


@pytest.mark.asyncio
async def test_cli_gateway_rejects_oversized_native_output(monkeypatch, tmp_path):
    """A CLI cannot force unbounded stdout retention in the orchestration process."""
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(settings, "orchestration_backend", "claude")
    _install_fake_cli(
        monkeypatch,
        orchestration_completion,
        "claude",
        tmp_path / "claude",
        lambda argv: b'{"loggedIn":true}' if argv[-2:] == ("auth", "status") else b"x" * (1024 * 1024 + 1),
    )

    with pytest.raises(orchestration_completion.OrchestrationBackendError) as error:
        await orchestration_completion.orchestration_completion(messages=[{"role": "user", "content": "hello"}])

    assert error.value.kind == orchestration_completion.OrchestrationBackendErrorKind.MALFORMED_OUTPUT


@pytest.mark.asyncio
async def test_cli_gateway_rejects_oversized_encoded_request_before_spawning_completion(monkeypatch, tmp_path):
    """The encoded orchestration prompt is capped before a CLI completion process starts."""
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(settings, "orchestration_backend", "claude")
    calls = _install_fake_cli(
        monkeypatch,
        orchestration_completion,
        "claude",
        tmp_path / "claude",
        lambda _argv: b'{"loggedIn":true}',
    )

    with pytest.raises(orchestration_completion.OrchestrationBackendError) as error:
        await orchestration_completion.orchestration_completion(
            messages=[{"role": "user", "content": "x" * (1024 * 1024 + 1)}]
        )

    assert error.value.kind == orchestration_completion.OrchestrationBackendErrorKind.MALFORMED_OUTPUT
    assert all("--print" not in argv for argv, _stdin in calls)


@pytest.mark.asyncio
async def test_codex_gateway_rejects_oversized_result_artifact(monkeypatch, tmp_path):
    """A growing Codex result artifact cannot bypass the native-output limit."""
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(settings, "orchestration_backend", "codex")

    def response(argv):
        if argv[-2:] == ("login", "status"):
            return {"stderr": b"Logged in with ChatGPT"}
        path = Path(argv[argv.index("--output-last-message") + 1])
        path.write_text(json.dumps({"content": "x" * (1024 * 1024 + 1), "tool_calls": []}))
        return b'{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n'

    _install_fake_cli(monkeypatch, orchestration_completion, "codex", tmp_path / "codex", response)

    with pytest.raises(orchestration_completion.OrchestrationBackendError) as error:
        await orchestration_completion.orchestration_completion(messages=[{"role": "user", "content": "hello"}])

    assert error.value.kind == orchestration_completion.OrchestrationBackendErrorKind.MALFORMED_OUTPUT


@pytest.mark.asyncio
async def test_cli_gateway_timeout_terminates_the_started_process_group(monkeypatch, tmp_path):
    """A bounded CLI request terminates its process group before reporting timeout."""
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(settings, "orchestration_backend", "claude")
    terminated = []

    async def record_termination(proc, *_args, **_kwargs):
        terminated.append(proc.pid)
        proc.release()

    monkeypatch.setattr(orchestration_completion, "terminate_process_group", record_termination)
    _install_fake_cli(
        monkeypatch,
        orchestration_completion,
        "claude",
        tmp_path / "claude",
        lambda argv: b'{"loggedIn":true}' if argv[-2:] == ("auth", "status") else {"hang": True},
    )

    with pytest.raises(orchestration_completion.OrchestrationBackendError) as error:
        await orchestration_completion.orchestration_completion(
            messages=[{"role": "user", "content": "hello"}], timeout=0.01
        )

    assert error.value.kind == orchestration_completion.OrchestrationBackendErrorKind.TIMEOUT
    assert terminated == [12345]


@pytest.mark.asyncio
async def test_cli_gateway_cancellation_terminates_the_started_process_group(monkeypatch, tmp_path):
    """Caller cancellation terminates the spawned CLI group and remains cancelled."""
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(settings, "orchestration_backend", "claude")
    terminated = []

    async def record_termination(proc, *_args, **_kwargs):
        terminated.append(proc.pid)
        proc.release()

    monkeypatch.setattr(orchestration_completion, "terminate_process_group", record_termination)
    calls = _install_fake_cli(
        monkeypatch,
        orchestration_completion,
        "claude",
        tmp_path / "claude",
        lambda argv: b'{"loggedIn":true}' if argv[-2:] == ("auth", "status") else {"hang": True},
    )

    task = asyncio.create_task(orchestration_completion.orchestration_completion(messages=[{"role": "user", "content": "hello"}]))
    while len(calls) < 2:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert terminated and set(terminated) == {12345}


@pytest.mark.asyncio
async def test_cli_runner_stops_a_growing_stdout_stream_before_the_request_timeout(monkeypatch):
    """The shared runner must terminate on stream growth, instead of buffering until timeout."""
    from huddleroom.services import orchestration_completion

    proc = _GrowingCliProcess(stdout_chunks=[b"x" * (1024 * 1024 + 1)])
    terminated = []

    async def create_subprocess_exec(*_args, **_kwargs):
        return proc

    async def terminate(process, *_args, **_kwargs):
        terminated.append(process.pid)
        process.release()

    monkeypatch.setattr(orchestration_completion.asyncio, "create_subprocess_exec", create_subprocess_exec)
    monkeypatch.setattr(orchestration_completion, "terminate_process_group", terminate)

    with pytest.raises(orchestration_completion.OrchestrationBackendError) as error:
        await orchestration_completion._run(("fake-cli",), "hello", timeout=0.01)

    assert error.value.kind == orchestration_completion.OrchestrationBackendErrorKind.MALFORMED_OUTPUT
    assert terminated and set(terminated) == {12345}


@pytest.mark.asyncio
async def test_cli_runner_times_out_when_output_reaches_eof_but_process_never_exits(monkeypatch):
    """EOF from both pipes does not turn a still-running CLI into a successful response."""
    from huddleroom.services import orchestration_completion

    proc = _EofThenHangingCliProcess()
    terminated = []

    async def create_subprocess_exec(*_args, **_kwargs):
        return proc

    async def terminate(process, *_args, **_kwargs):
        terminated.append(process.pid)
        process.release()

    monkeypatch.setattr(orchestration_completion.asyncio, "create_subprocess_exec", create_subprocess_exec)
    monkeypatch.setattr(orchestration_completion, "terminate_process_group", terminate)

    with pytest.raises(orchestration_completion.OrchestrationBackendError) as error:
        await asyncio.wait_for(orchestration_completion._run(("fake-cli",), "hello", timeout=0.01), timeout=0.1)

    assert error.value.kind == orchestration_completion.OrchestrationBackendErrorKind.TIMEOUT
    assert terminated == [12345]


@pytest.mark.asyncio
async def test_cli_runner_reads_output_while_stdin_backpressures(monkeypatch):
    """A blocked stdin drain cannot prevent the runner from enforcing the output cap."""
    from huddleroom.services import orchestration_completion

    proc = _BackpressuredStdinGrowingOutputProcess()
    terminated = []

    async def create_subprocess_exec(*_args, **_kwargs):
        return proc

    async def terminate(process, *_args, **_kwargs):
        terminated.append(process.pid)
        process.release()

    monkeypatch.setattr(orchestration_completion.asyncio, "create_subprocess_exec", create_subprocess_exec)
    monkeypatch.setattr(orchestration_completion, "terminate_process_group", terminate)

    with pytest.raises(orchestration_completion.OrchestrationBackendError) as error:
        await asyncio.wait_for(orchestration_completion._run(("fake-cli",), "hello", timeout=0.01), timeout=0.1)

    assert error.value.kind == orchestration_completion.OrchestrationBackendErrorKind.MALFORMED_OUTPUT
    assert terminated == [12345]


@pytest.mark.asyncio
async def test_cli_runner_omits_only_onecli_placeholder_provider_keys_from_child_environment(monkeypatch):
    """Native CLI login must not inherit synthetic OneCLI API-key placeholders."""
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(settings, "credential_mode", "onecli")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "onecli-anthropic-placeholder")
    monkeypatch.setenv("OPENAI_API_KEY", "onecli-openai-placeholder")
    monkeypatch.setenv("PRESERVED_NATIVE_SETTING", "keep")
    captured = {}
    proc = SimpleNamespace(
        returncode=0,
        stdin=_RecordingCliStdin([None, None]),
        stdout=_FiniteCliStream(b"{}"),
        stderr=_FiniteCliStream(b""),
    )

    async def wait():
        return 0

    proc.wait = wait

    async def create_subprocess_exec(*_args, **kwargs):
        captured.update(kwargs["env"])
        return proc

    monkeypatch.setattr(orchestration_completion.asyncio, "create_subprocess_exec", create_subprocess_exec)

    await orchestration_completion._run(("fake-cli",), "hello", timeout=0.01, backend="claude")

    assert "ANTHROPIC_API_KEY" not in captured
    assert "OPENAI_API_KEY" not in captured
    assert captured["PRESERVED_NATIVE_SETTING"] == "keep"


_M1_KEYS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "GIT_SSL_CAINFO", "DENO_CERT", "NODE_OPTIONS")
_GATEWAY_KEYS = (
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy", "ONECLI_GATEWAY",
    "ONECLI_GATEWAY_SKILL_PATH", "NODE_USE_ENV_PROXY", "NODE_EXTRA_CA_CERTS", "ANTHROPIC_BASE_URL",
) + _M1_KEYS


async def _captured_child_env(monkeypatch, backend):
    from huddleroom.services import orchestration_completion

    for key in _GATEWAY_KEYS:
        monkeypatch.setenv(key, "x-" + key)
    monkeypatch.setenv("HUDDLEROOM_SECRET_KEY", "secret-value")
    monkeypatch.setenv("RALLY_API_TOKEN", "token-value")
    monkeypatch.setenv("HOME", "/home/user")
    captured = {}
    proc = SimpleNamespace(
        returncode=0, stdin=_RecordingCliStdin([None, None]),
        stdout=_FiniteCliStream(b"{}"), stderr=_FiniteCliStream(b""),
    )

    async def wait():
        return 0

    proc.wait = wait

    async def create_subprocess_exec(*_args, **kwargs):
        captured.update(kwargs["env"])
        return proc

    monkeypatch.setattr(orchestration_completion.asyncio, "create_subprocess_exec", create_subprocess_exec)
    await orchestration_completion._run(("fake-cli",), "hello", timeout=0.01, backend=backend)
    return captured


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["claude", "codex"])
async def test_cli_runner_omits_gateway_proxy_ca_and_context_keys_in_onecli_mode(monkeypatch, backend):
    """Native CLI must not inherit proxy, OneCLI gateway, CA/node, base-URL or HuddleRoom context env vars."""
    from huddleroom.config import settings

    monkeypatch.setattr(settings, "credential_mode", "onecli")
    captured = await _captured_child_env(monkeypatch, backend)

    for key in _GATEWAY_KEYS + ("HUDDLEROOM_SECRET_KEY", "RALLY_API_TOKEN"):
        assert key not in captured, key
    assert captured["HOME"] == "/home/user"


@pytest.mark.asyncio
async def test_cli_runner_direct_mode_keeps_proxy_but_strips_context_keys(monkeypatch):
    from huddleroom.config import settings

    monkeypatch.setattr(settings, "credential_mode", "direct")
    captured = await _captured_child_env(monkeypatch, "claude")

    for key in _GATEWAY_KEYS:
        assert captured[key] == "x-" + key
    assert "HUDDLEROOM_SECRET_KEY" not in captured and "RALLY_API_TOKEN" not in captured


@pytest.mark.asyncio
async def test_cli_runner_keeps_proxy_when_native_runtimes_empty(monkeypatch):
    """When onecli_native_auth_runtimes is empty, native CLIs keep gateway proxy in onecli mode."""
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(settings, "credential_mode", "onecli")
    monkeypatch.setattr(settings, "onecli_native_auth_runtimes", [])  # Empty = route through gateway
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example.com:8080")
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.example.com:8080")
    monkeypatch.setenv("ONECLI_GATEWAY", "gateway.example.com")
    monkeypatch.setenv("HUDDLEROOM_SECRET_KEY", "secret-value")
    monkeypatch.setenv("HOME", "/home/user")

    captured = {}
    proc = SimpleNamespace(
        returncode=0,
        stdin=_RecordingCliStdin([None, None]),
        stdout=_FiniteCliStream(b"{}"),
        stderr=_FiniteCliStream(b""),
    )

    async def wait():
        return 0

    proc.wait = wait

    async def create_subprocess_exec(*_args, **kwargs):
        captured.update(kwargs["env"])
        return proc

    monkeypatch.setattr(orchestration_completion.asyncio, "create_subprocess_exec", create_subprocess_exec)

    await orchestration_completion._run(("fake-cli",), "hello", timeout=0.01, backend="claude")

    # Proxy should be kept
    assert captured["HTTP_PROXY"] == "http://proxy.example.com:8080"
    assert captured["HTTPS_PROXY"] == "https://proxy.example.com:8080"
    # Gateway should be kept
    assert captured["ONECLI_GATEWAY"] == "gateway.example.com"
    # HuddleRoom context should still be dropped
    assert "HUDDLEROOM_SECRET_KEY" not in captured
    # Normal keys should be kept
    assert captured["HOME"] == "/home/user"


@pytest.mark.asyncio
async def test_codex_artifact_growth_stops_the_process_before_the_request_timeout(monkeypatch):
    """Codex output-file growth is monitored while the native process is still running."""
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(settings, "orchestration_effort", None)
    proc = _GrowingCliProcess()
    terminated = []

    async def create_subprocess_exec(*argv, **_kwargs):
        result_path = Path(argv[argv.index("--output-last-message") + 1])
        result_path.write_bytes(b"x" * (1024 * 1024 + 1))
        return proc

    async def terminate(process, *_args, **_kwargs):
        terminated.append(process.pid)
        process.release()

    monkeypatch.setattr(orchestration_completion.asyncio, "create_subprocess_exec", create_subprocess_exec)
    monkeypatch.setattr(orchestration_completion, "terminate_process_group", terminate)
    monkeypatch.setattr(orchestration_completion, "_timeout", lambda _request: 0.01)

    with pytest.raises(orchestration_completion.OrchestrationBackendError) as error:
        await orchestration_completion._complete_cli("fake-codex", "codex", {"messages": []})

    assert error.value.kind == orchestration_completion.OrchestrationBackendErrorKind.MALFORMED_OUTPUT
    assert terminated == [12345]


_CLAUDE_OK = (
    b'{"structured_output":{"content":"ok","tool_calls":[]},'
    b'"usage":{"input_tokens":1,"output_tokens":1}}'
)


def _claude_argv_fake(monkeypatch, tmp_path, completion_module):
    return _install_fake_cli(
        monkeypatch, completion_module, "claude", tmp_path / "claude",
        lambda argv: b'{"loggedIn":true}' if argv[-2:] == ("auth", "status") else _CLAUDE_OK,
    )


def _codex_argv_fake(monkeypatch, tmp_path, completion_module):
    def respond(argv):
        if argv[-2:] == ("login", "status"):
            return {"stderr": b"Logged in with ChatGPT", "returncode": 0}
        Path(argv[argv.index("--output-last-message") + 1]).write_text('{"content":"ok","tool_calls":[]}')
        return b'{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n'

    return _install_fake_cli(monkeypatch, completion_module, "codex", tmp_path / "codex", respond)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [None, "claude-opus-4-1"])
async def test_claude_argv_model_and_effort_without_capability_metadata(monkeypatch, tmp_path, model):
    """Claude accepts an enum effort without a model; --model is passed only when configured."""
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(settings, "orchestration_backend", "claude")
    monkeypatch.setattr(settings, "orchestration_effort", "high")
    monkeypatch.setattr(settings, "orchestration_cli_model", model)
    calls = _claude_argv_fake(monkeypatch, tmp_path, orchestration_completion)

    await orchestration_completion.orchestration_completion(messages=[{"role": "user", "content": "hello"}])

    argv = next(argv for argv, _stdin in calls if "--print" in argv)
    assert argv[argv.index("--effort") + 1] == "high"
    if model:
        assert argv[argv.index("--model") + 1] == model
    else:
        assert "--model" not in argv


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [None, "gpt-5-codex"])
async def test_codex_argv_model(monkeypatch, tmp_path, model):
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(settings, "orchestration_backend", "codex")
    monkeypatch.setattr(settings, "orchestration_effort", None)
    monkeypatch.setattr(settings, "orchestration_cli_model", model)
    calls = _codex_argv_fake(monkeypatch, tmp_path, orchestration_completion)

    await orchestration_completion.orchestration_completion(messages=[{"role": "user", "content": "hello"}])

    argv = next(argv for argv, _stdin in calls if "exec" in argv)
    if model:
        assert argv[argv.index("-m") + 1] == model
        assert argv[-1] == "-"
    else:
        assert "-m" not in argv


@pytest.mark.asyncio
async def test_codex_model_with_xhigh_needs_no_catalog(monkeypatch, tmp_path):
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    empty_home = tmp_path / "empty-codex-home"
    empty_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(empty_home))
    monkeypatch.setattr(settings, "orchestration_backend", "codex")
    monkeypatch.setattr(settings, "orchestration_effort", "xhigh")
    monkeypatch.setattr(settings, "orchestration_cli_model", "gpt-5-codex")
    calls = _codex_argv_fake(monkeypatch, tmp_path, orchestration_completion)

    await orchestration_completion.orchestration_completion(messages=[{"role": "user", "content": "hello"}])

    argv = next(argv for argv, _stdin in calls if "exec" in argv)
    assert "model_reasoning_effort=xhigh" in argv


@pytest.mark.asyncio
async def test_codex_effort_without_model_and_empty_metadata_is_unsupported(monkeypatch, tmp_path):
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    empty_home = tmp_path / "empty-codex-home"
    empty_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(empty_home))
    monkeypatch.setattr(settings, "orchestration_backend", "codex")
    monkeypatch.setattr(settings, "orchestration_effort", "high")
    monkeypatch.setattr(settings, "orchestration_cli_model", None)
    _codex_argv_fake(monkeypatch, tmp_path, orchestration_completion)

    with pytest.raises(orchestration_completion.OrchestrationBackendError) as error:
        await orchestration_completion.orchestration_completion(messages=[{"role": "user", "content": "hello"}])

    assert error.value.kind == orchestration_completion.OrchestrationBackendErrorKind.UNSUPPORTED


@pytest.mark.asyncio
async def test_cli_exit_error_includes_redacted_stderr_tail(monkeypatch, tmp_path):
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(settings, "orchestration_backend", "claude")
    monkeypatch.setattr(settings, "orchestration_effort", None)
    _install_fake_cli(
        monkeypatch, orchestration_completion, "claude", tmp_path / "claude",
        lambda argv: (
            b'{"loggedIn":true}' if argv[-2:] == ("auth", "status")
            else {"stderr": b"error:\n  model not found\n key sk-abcdefghijklmnop1234567890", "returncode": 1}
        ),
    )

    with pytest.raises(orchestration_completion.OrchestrationBackendError) as error:
        await orchestration_completion.orchestration_completion(messages=[{"role": "user", "content": "hello"}])

    assert error.value.kind == orchestration_completion.OrchestrationBackendErrorKind.EXIT
    assert "error: model not found" in str(error.value)
    assert "abcdefghijklmnop1234567890" not in str(error.value)


@pytest.mark.parametrize("backend, model, label", [
    ("claude", "opus[1m]", "claude CLI (opus[1m])"),
    ("codex", None, "codex CLI (configured CLI default)"),
])
def test_runtime_metadata_label_uses_cli_model(backend, model, label):
    from huddleroom.services import orchestration_completion

    config = SimpleNamespace(orchestration_backend=backend, orchestration_cli_model=model)
    assert orchestration_completion.orchestration_runtime_metadata(
        orchestration_completion.orchestration_completion, "api/model", config
    ) == ("cli_main", label)


def test_api_runtime_metadata_ignores_cli_model():
    from huddleroom.services import orchestration_completion

    config = SimpleNamespace(orchestration_backend="api", orchestration_cli_model="opus")
    assert orchestration_completion.orchestration_runtime_metadata(
        orchestration_completion.orchestration_completion, "api/model", config
    ) == ("api", "api/model")


@pytest.mark.asyncio
async def test_codex_explicit_effort_uses_only_the_exact_configured_model_catalog_entry(monkeypatch, tmp_path):
    """Codex accepts an effort only when its configured default model advertises that exact level."""
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text('model = "gpt-6-luna"\n')
    (codex_home / "models_cache.json").write_text(json.dumps({
        "models": [{"slug": "gpt-6-luna", "supported_reasoning_levels": [{"effort": "high"}]}],
    }))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr(settings, "orchestration_backend", "codex")
    monkeypatch.setattr(settings, "orchestration_effort", "high")

    def response(argv):
        if argv[-2:] == ("login", "status"):
            return {"stderr": b"Logged in with ChatGPT"}
        result_path = Path(argv[argv.index("--output-last-message") + 1])
        result_path.write_text(json.dumps({"content": "OK", "tool_calls": []}))
        return b'{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n'

    calls = _install_fake_cli(monkeypatch, orchestration_completion, "codex", tmp_path / "codex", response)
    await orchestration_completion.orchestration_completion(messages=[{"role": "user", "content": "hello"}])

    argv, _stdin = next((argv, stdin) for argv, stdin in calls if "exec" in argv)
    assert ("-c", "model_reasoning_effort=high") == (argv[argv.index("-c")], argv[argv.index("-c") + 1])


@pytest.mark.asyncio
async def test_codex_explicit_effort_rejects_an_unadvertised_level_without_spawning_completion(monkeypatch, tmp_path):
    """An exact Codex catalog mismatch must advise default and leave no partial completion."""
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text('model = "gpt-6-luna"\n')
    (codex_home / "models_cache.json").write_text(json.dumps({
        "models": [{"slug": "gpt-6-luna", "supported_reasoning_levels": [{"effort": "low"}]}],
    }))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr(settings, "orchestration_backend", "codex")
    monkeypatch.setattr(settings, "orchestration_effort", "high")
    calls = _install_fake_cli(
        monkeypatch,
        orchestration_completion,
        "codex",
        tmp_path / "codex",
        lambda _argv: {"stderr": b"Logged in with ChatGPT"},
    )

    with pytest.raises(orchestration_completion.OrchestrationBackendError, match="default") as error:
        await orchestration_completion.orchestration_completion(messages=[{"role": "user", "content": "hello"}])

    assert error.value.kind == orchestration_completion.OrchestrationBackendErrorKind.UNSUPPORTED
    assert all("exec" not in argv for argv, _stdin in calls)


@pytest.mark.asyncio
async def test_api_effort_is_forwarded_only_after_metadata_support(monkeypatch):
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(settings, "orchestration_backend", "api")
    monkeypatch.setattr(settings, "orchestration_effort", "high")
    monkeypatch.setattr(orchestration_completion.litellm, "get_supported_openai_params", lambda **_kwargs: ["reasoning_effort"])
    monkeypatch.setitem(orchestration_completion.litellm.model_cost, "provider/model", {"supports_high_reasoning_effort": True})
    seen = {}

    async def completion(**request):
        seen.update(request)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(orchestration_completion.litellm, "acompletion", completion)
    await orchestration_completion.orchestration_completion(model="provider/model", messages=[], temperature=0)

    assert seen["reasoning_effort"] == "high"
    assert "temperature" not in seen


def test_api_efforts_are_empty_when_metadata_cannot_prove_the_level(monkeypatch):
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(orchestration_completion.litellm, "get_supported_openai_params", lambda **_kwargs: ["reasoning_effort"])
    monkeypatch.setitem(orchestration_completion.litellm.model_cost, "provider/model", {"supports_reasoning": True})

    assert orchestration_completion.supported_orchestration_efforts("api", "provider/model") == frozenset()
    assert orchestration_completion.is_orchestration_backend_supported("codex")


def test_cli_effort_catalog_requires_native_model_capability(monkeypatch, tmp_path):
    """CLI setup choices come from a configured native model, never a generic CLI enum."""
    from huddleroom.services import orchestration_completion

    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text('model = "gpt-6-luna"\n')
    (codex_home / "models_cache.json").write_text(json.dumps({
        "models": [{"slug": "gpt-6-luna", "supported_reasoning_levels": [{"effort": "high"}]}],
    }))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    assert orchestration_completion.supported_orchestration_efforts("claude") == orchestration_completion.CLI_EFFORTS
    assert orchestration_completion.supported_orchestration_efforts("codex") == frozenset({"high"})
    assert orchestration_completion.supported_orchestration_efforts("codex", "gpt-5-codex") == (
        orchestration_completion.CLI_EFFORTS - {"max"}
    )


def test_cli_validation_rejects_explicit_effort_before_authentication(monkeypatch, tmp_path):
    """An unsupported saved CLI effort fails at startup before it can authenticate or open state."""
    from huddleroom.services import orchestration_completion

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    config = SimpleNamespace(
        orchestration_backend="codex", orchestration_effort="high", orchestration_model="ignored", orchestration_cli_model=None
    )
    monkeypatch.setattr(orchestration_completion.shutil, "which", lambda _name: "/fake/codex")

    async def authenticate(*_args):
        pytest.fail("explicit effort must be validated before auth")

    monkeypatch.setattr(orchestration_completion, "_authenticate", authenticate)

    with pytest.raises(orchestration_completion.OrchestrationBackendError) as error:
        orchestration_completion.validate_orchestration_backend(config)

    assert error.value.kind == orchestration_completion.OrchestrationBackendErrorKind.UNSUPPORTED
    assert "default" in str(error.value)


def test_api_efforts_use_exact_model_metadata_levels(monkeypatch):
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(orchestration_completion.litellm, "get_supported_openai_params", lambda **_kwargs: ["reasoning_effort"])
    monkeypatch.setitem(
        orchestration_completion.litellm.model_cost,
        "provider/model",
        {"reasoning_effort_levels": ["low", "high", "unsupported"]},
    )

    assert orchestration_completion.supported_orchestration_efforts("api", "provider/model") == frozenset({"low", "high"})


def test_api_effort_metadata_list_honors_explicit_per_level_overrides(monkeypatch):
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(orchestration_completion.litellm, "get_supported_openai_params", lambda **_kwargs: ["reasoning_effort"])
    monkeypatch.setitem(
        orchestration_completion.litellm.model_cost,
        "provider/model",
        {
            "reasoning_effort_levels": ["low", "high"],
            "supports_low_reasoning_effort": False,
            "supports_medium_reasoning_effort": True,
        },
    )

    assert orchestration_completion.supported_orchestration_efforts("api", "provider/model") == frozenset({"high", "medium"})


def test_runtime_metadata_follows_the_resolved_completion_callable(monkeypatch):
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    async def injected(**_request):
        return None

    monkeypatch.setattr(settings, "orchestration_backend", "api")
    assert orchestration_completion.orchestration_runtime_metadata(
        orchestration_completion.orchestration_completion, "api/model"
    ) == ("api", "api/model")

    monkeypatch.setattr(settings, "orchestration_backend", "codex")
    assert orchestration_completion.orchestration_runtime_metadata(
        orchestration_completion.orchestration_completion, "api/model"
    ) == ("cli_main", "codex CLI (configured CLI default)")
    assert orchestration_completion.orchestration_runtime_metadata(injected, "test/model") == ("api", "test/model")


@pytest.mark.asyncio
async def test_default_cli_route_emits_valid_cli_main_event_metadata(monkeypatch):
    """The selected CLI default must emit an event shape accepted by the response stream."""
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion
    from huddleroom.services import orchestration_llm_decision_adapter as adapter_module
    from huddleroom.services import agent_response_stream

    events = []
    ack = SimpleNamespace(generation=None)

    async def never_reset(_project_id):
        await asyncio.Event().wait()

    async def fake_completion(**_request):
        return {
            "choices": [
                {"message": {"content": json.dumps({"decision": {"action_type": "noop", "reason": "wait"}})}}
            ]
        }

    async def publish(event):
        events.append(event)

    original_invocation = agent_response_stream.AgentResponseInvocation

    def local_invocation(context):
        return original_invocation(context, publish=publish)

    monkeypatch.setattr(settings, "orchestration_backend", "codex")
    monkeypatch.setattr(orchestration_completion, "orchestration_completion", fake_completion)
    monkeypatch.setattr(agent_response_stream, "AgentResponseInvocation", local_invocation)
    monkeypatch.setattr(
        "huddleroom.services.agent_response_relay.register_project_reset_monitor",
        lambda _project_id: _async_value(ack),
    )
    monkeypatch.setattr(
        "huddleroom.services.agent_response_relay.unregister_project_reset_monitor",
        lambda *_args: _async_value(None),
    )
    monkeypatch.setattr(agent_response_stream, "wait_for_project_reset", never_reset)

    result = await adapter_module.OrchestrationDecisionAdapter().decide(
        {"run": {"status": "running"}}, project={"id": str(uuid4())}
    )

    assert result.parsed_decision == {"action_type": "noop", "reason": "wait"}
    started = next(event for event in events if event.event_type == "agent_response.started")
    assert (started.invocation_kind, started.payload["model_or_runtime"]) == (
        "cli_main", "codex CLI (configured CLI default)"
    )
    assert [event.payload for event in events if event.event_type == "agent_response.output"] == [
        {
            "stream": "output",
            "text": json.dumps({"decision": {"action_type": "noop", "reason": "wait"}}),
        }
    ]


async def _async_value(value):
    return value


def _install_fake_cli(monkeypatch, completion_module, executable, path, response_for_argv):
    """Install a fake native CLI process without invoking a host executable."""
    import shutil

    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    calls = []

    class FakeProcess:
        pid = 12345

        def __init__(self, call, stdout, stderr=b"", returncode=0, hang=False):
            self._call = call
            self._stdout = stdout
            self._stderr = stderr
            self._hang = hang
            self.returncode = returncode
            self.stdin = _RecordingCliStdin(call)
            self._released = asyncio.Event()
            self.stdout = _GrowingCliStream((stdout,), self._released) if hang else _FiniteCliStream(stdout)
            self.stderr = _GrowingCliStream((stderr,), self._released) if hang else _FiniteCliStream(stderr)

        async def communicate(self, input=None):
            if input is not None:
                self.stdin.write(input)
            if self._hang:
                await asyncio.Event().wait()
            return self._stdout, self._stderr

        async def wait(self):
            if self._hang:
                await self._released.wait()
            return self.returncode

        def terminate(self):
            self.release()

        def kill(self):
            self.returncode = -9
            self._released.set()

        def release(self):
            self.returncode = -15
            self._released.set()

    async def create_subprocess_exec(*argv, **_kwargs):
        call = [tuple(argv), None]
        calls.append(call)
        response = response_for_argv(tuple(argv))
        if isinstance(response, dict):
            return FakeProcess(
                call,
                response.get("stdout", b""),
                response.get("stderr", b""),
                response.get("returncode", 0),
                response.get("hang", False),
            )
        return FakeProcess(call, response)

    monkeypatch.setattr(completion_module, "asyncio", asyncio, raising=False)
    monkeypatch.setattr(completion_module, "shutil", shutil, raising=False)
    monkeypatch.setattr(completion_module.shutil, "which", lambda name: str(path) if name == executable else None)
    monkeypatch.setattr(completion_module.asyncio, "create_subprocess_exec", create_subprocess_exec)
    return calls


class _RecordingCliStdin:
    def __init__(self, call):
        self._call = call

    def write(self, data):
        self._call[1] = (self._call[1] or b"") + data

    async def drain(self):
        return None

    def close(self):
        return None

    async def wait_closed(self):
        return None


class _FiniteCliStream:
    def __init__(self, content):
        self._content = content

    async def read(self, _size=-1):
        content, self._content = self._content, b""
        return content


class _GrowingCliProcess:
    pid = 12345
    returncode = None

    def __init__(self, stdout_chunks=(), stderr_chunks=()):
        self._released = asyncio.Event()
        self.stdin = _RecordingCliStdin([None, None])
        self.stdout = _GrowingCliStream(stdout_chunks, self._released)
        self.stderr = _GrowingCliStream(stderr_chunks, self._released)

    async def communicate(self, _input=None):
        await self._released.wait()
        return b"", b""

    async def wait(self):
        await self._released.wait()
        return self.returncode

    def release(self):
        self.returncode = -15
        self._released.set()


class _GrowingCliStream:
    def __init__(self, chunks, released):
        self._chunks = list(chunks)
        self._released = released

    async def read(self, _size=-1):
        if self._chunks:
            return self._chunks.pop(0)
        await self._released.wait()
        return b""


class _EofThenHangingCliProcess:
    pid = 12345
    returncode = None

    def __init__(self):
        self._released = asyncio.Event()
        self.stdin = _RecordingCliStdin([None, None])
        self.stdout = _FiniteCliStream(b"{}")
        self.stderr = _FiniteCliStream(b"")

    async def wait(self):
        await self._released.wait()
        return self.returncode

    def release(self):
        self.returncode = -15
        self._released.set()


class _BackpressuredStdinGrowingOutputProcess(_GrowingCliProcess):
    def __init__(self):
        super().__init__(stdout_chunks=[b"x" * (1024 * 1024 + 1)])
        self.stdin = _BlockingCliStdin([None, None], self._released)


class _BlockingCliStdin(_RecordingCliStdin):
    def __init__(self, call, released):
        super().__init__(call)
        self._released = released

    async def drain(self):
        await self._released.wait()


@pytest.mark.asyncio
async def test_cli_exit_error_redacts_secret_straddling_the_300_char_cut(monkeypatch, tmp_path):
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    secret = "sk-" + "Zq9" * 12
    stderr = ("head " + secret + " " + "tail " * 54).encode()  # cut lands inside the secret
    monkeypatch.setattr(settings, "orchestration_backend", "claude")
    monkeypatch.setattr(settings, "orchestration_effort", None)
    _install_fake_cli(
        monkeypatch, orchestration_completion, "claude", tmp_path / "claude",
        lambda argv: (
            b'{"loggedIn":true}' if argv[-2:] == ("auth", "status")
            else {"stderr": stderr, "returncode": 1}
        ),
    )

    with pytest.raises(orchestration_completion.OrchestrationBackendError) as error:
        await orchestration_completion.orchestration_completion(messages=[{"role": "user", "content": "hello"}])

    assert "Zq9" not in str(error.value)
    assert "tail" in str(error.value)


@pytest.mark.asyncio
async def test_codex_argv_puts_model_and_effort_before_the_stdin_dash(monkeypatch, tmp_path):
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(settings, "orchestration_backend", "codex")
    monkeypatch.setattr(settings, "orchestration_effort", "high")
    monkeypatch.setattr(settings, "orchestration_cli_model", "gpt-5-codex")
    calls = _codex_argv_fake(monkeypatch, tmp_path, orchestration_completion)

    await orchestration_completion.orchestration_completion(messages=[{"role": "user", "content": "hello"}])

    argv = next(argv for argv, _stdin in calls if "exec" in argv)
    assert argv[-1] == "-"
    assert argv.index("-m") < len(argv) - 1 and argv.index("-c") < len(argv) - 1


@pytest.mark.asyncio
async def test_codex_model_with_max_effort_is_unsupported(monkeypatch, tmp_path):
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(settings, "orchestration_backend", "codex")
    monkeypatch.setattr(settings, "orchestration_effort", "max")
    monkeypatch.setattr(settings, "orchestration_cli_model", "gpt-5-codex")
    _codex_argv_fake(monkeypatch, tmp_path, orchestration_completion)

    with pytest.raises(orchestration_completion.OrchestrationBackendError) as error:
        await orchestration_completion.orchestration_completion(messages=[{"role": "user", "content": "hello"}])

    assert error.value.kind == orchestration_completion.OrchestrationBackendErrorKind.UNSUPPORTED


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["claude", "codex"])
async def test_invalid_cli_model_is_unsupported_at_argv_build(monkeypatch, tmp_path, backend):
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(settings, "orchestration_backend", backend)
    monkeypatch.setattr(settings, "orchestration_effort", None)
    monkeypatch.setattr(settings, "orchestration_cli_model", "-x")
    fake = _claude_argv_fake if backend == "claude" else _codex_argv_fake
    calls = fake(monkeypatch, tmp_path, orchestration_completion)

    with pytest.raises(orchestration_completion.OrchestrationBackendError) as error:
        await orchestration_completion.orchestration_completion(messages=[{"role": "user", "content": "hello"}])

    assert error.value.kind == orchestration_completion.OrchestrationBackendErrorKind.UNSUPPORTED
    assert all("-x" not in argv for argv, _stdin in calls)
