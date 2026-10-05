import asyncio
import json
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
async def test_all_control_plane_defaults_route_api_or_fail_closed_for_each_cli(monkeypatch):
    """All ten control-plane constructors use one selected boundary, never LiteLLM in CLI mode."""
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

    async def forbidden_api(**_request):
        raise AssertionError("CLI orchestration must not fall back to LiteLLM")

    monkeypatch.setattr(orchestration_completion.litellm, "acompletion", forbidden_api)
    for backend in ("claude", "codex"):
        monkeypatch.setattr(settings, "orchestration_backend", backend)
        cli_services = _default_completion_services()
        for service in cli_services:
            assert service._completion_fn is orchestration_completion.orchestration_completion
            with pytest.raises(orchestration_completion.OrchestrationBackendError) as error:
                await service._completion_fn(model="provider/model", messages=[])
            assert error.value.kind == orchestration_completion.OrchestrationBackendErrorKind.UNSUPPORTED


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
        model="openai/gpt-4o-mini", messages=[{"role": "user", "content": "hello"}], temperature=0
    )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert seen == {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
    }


@pytest.mark.asyncio
async def test_cli_backend_fails_closed_without_calling_api(monkeypatch):
    from huddleroom.config import settings
    from huddleroom.services import orchestration_completion

    monkeypatch.setattr(settings, "orchestration_backend", "codex")

    async def api_call(**_request):
        raise AssertionError("CLI mode must never fall back to LiteLLM")

    monkeypatch.setattr(orchestration_completion.litellm, "acompletion", api_call)

    with pytest.raises(orchestration_completion.OrchestrationBackendError, match="no-tools isolation") as error:
        await orchestration_completion.orchestration_completion(messages=[])

    assert error.value.kind == orchestration_completion.OrchestrationBackendErrorKind.UNSUPPORTED


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
    assert not orchestration_completion.is_orchestration_backend_supported("codex")


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


async def _async_value(value):
    return value
