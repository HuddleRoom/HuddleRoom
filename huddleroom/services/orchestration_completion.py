"""Completion boundary for orchestration control-plane calls.

CLI backends deliberately fail closed until their installed command contracts can
prove tool, filesystem, network, and configuration isolation.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Awaitable, Callable

import litellm

from huddleroom.config import Settings, settings
from huddleroom.services.secret_redaction import redact_secrets


CompletionFn = Callable[..., Awaitable[Any]]
API_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
CLI_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


class OrchestrationBackendErrorKind(StrEnum):
    MISSING = "missing"
    UNSUPPORTED = "unsupported"
    UNAUTHENTICATED = "unauthenticated"
    TIMEOUT = "timeout"
    EXIT = "exit"
    PROTOCOL = "protocol"
    MALFORMED_OUTPUT = "malformed_output"


class OrchestrationBackendError(RuntimeError):
    """A safe, actionable orchestration backend failure."""

    def __init__(self, kind: OrchestrationBackendErrorKind, message: str) -> None:
        self.kind = kind
        super().__init__(redact_secrets(message))


def supported_orchestration_efforts(backend: str, model: str | None = None) -> frozenset[str]:
    """Return only categorical effort values verified by local model metadata."""
    if backend == "api":
        return _api_supported_efforts(model or settings.orchestration_model)
    if backend in {"claude", "codex"}:
        return CLI_EFFORTS if is_orchestration_backend_supported(backend) else frozenset()
    raise OrchestrationBackendError(OrchestrationBackendErrorKind.UNSUPPORTED, f"Unknown orchestration backend: {backend}")


def is_orchestration_backend_supported(backend: str) -> bool:
    """Whether this release has a verified restrictive execution contract."""
    return backend == "api"


def validate_orchestration_backend(config: Settings = settings) -> None:
    """Preflight without making a model request."""
    backend = config.orchestration_backend
    if backend == "api":
        _validate_api_effort(config.orchestration_model, config.orchestration_effort)
        return
    _raise_cli_unsupported(backend)


def get_orchestration_completion(completion_fn: CompletionFn | None = None) -> CompletionFn:
    """Return injected completion for tests, otherwise the selected backend boundary."""
    return completion_fn or orchestration_completion


def orchestration_runtime_metadata(
    completion_fn: CompletionFn,
    api_model: str,
    config: Settings = settings,
) -> tuple[str, str]:
    """Describe the resolved completion path without relabeling injected callables."""
    if completion_fn is not orchestration_completion or config.orchestration_backend == "api":
        return "api", api_model
    return "cli_main", f"{config.orchestration_backend} CLI (configured CLI default)"


async def orchestration_completion(**request: Any) -> Any:
    """Perform one non-streaming orchestration completion without backend fallback."""
    if settings.orchestration_backend == "api":
        _validate_api_effort(request.get("model") or settings.orchestration_model, settings.orchestration_effort)
        if settings.orchestration_effort is not None:
            request = {key: value for key, value in request.items() if key != "temperature"}
            request["reasoning_effort"] = settings.orchestration_effort
        return await litellm.acompletion(**request)
    _raise_cli_unsupported(settings.orchestration_backend)


def _validate_api_effort(model: str, effort: str | None) -> None:
    if effort is None:
        return
    if effort not in API_EFFORTS:
        raise OrchestrationBackendError(
            OrchestrationBackendErrorKind.UNSUPPORTED,
            f"Unsupported API orchestration effort {effort!r}; select default or a supported value.",
        )
    if effort not in _api_supported_efforts(model):
        raise OrchestrationBackendError(
            OrchestrationBackendErrorKind.UNSUPPORTED,
            f"Model {model!r} does not report support for reasoning effort {effort!r}; select default.",
        )


def _api_supported_efforts(model: str) -> frozenset[str]:
    """Read only exact per-level LiteLLM metadata; absent fields are indeterminate."""
    try:
        params = litellm.get_supported_openai_params(model=model)
        metadata = litellm.model_cost.get(model) or litellm.model_cost.get(model.split("/", 1)[-1])
    except Exception:
        return frozenset()
    if not params or "reasoning_effort" not in params or not isinstance(metadata, dict):
        return frozenset()
    levels = metadata.get("reasoning_effort_levels")
    supported = {
        level for level in levels
        if isinstance(level, str) and level in API_EFFORTS
    } if isinstance(levels, (list, tuple, set, frozenset)) else set()
    for effort in API_EFFORTS:
        override = metadata.get(f"supports_{effort}_reasoning_effort")
        if override is True:
            supported.add(effort)
        elif override is False:
            supported.discard(effort)
    return frozenset(supported)


def _raise_cli_unsupported(backend: str) -> None:
    raise OrchestrationBackendError(
        OrchestrationBackendErrorKind.UNSUPPORTED,
        f"The {backend} orchestration backend is unsupported: this installed CLI has no verified no-tools "
        "isolation contract. Use API via LiteLLM, or install a CLI version with a verified restrictive contract "
        "and rerun setup.",
    )
