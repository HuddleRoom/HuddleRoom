"""Auto-repair of failed structured LLM responses via retry loops.

This module provides repair loops for both API (litellm) and CLI (subprocess)
LLM invocations. On parse failure, the module accumulates the failed response
and error feedback in the message history, then retries up to max_attempts.
This allows the model to see the full context of all prior failures and corrections.

The module requires:
- completion_fn: async callable with litellm.acompletion signature
- parse: sync callable that raises ValueError or json.JSONDecodeError on bad output
- request: dict with messages list (treated immutably; copy used for retry)
- redact_secrets: applied to error text before sending back to model
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Awaitable, Callable, TypeVar

from huddleroom.services.orchestration_llm_decision_adapter import _extract_content
from huddleroom.services.secret_redaction import redact_secrets
from huddleroom.services.agent_response_stream import AgentResponseCall, AgentResponseInvocation


T = TypeVar("T")


def _unfence_json(raw: Any) -> Any:
    """Remove one whole-response Markdown fence around JSON, if present."""
    if not isinstance(raw, str):
        return raw
    stripped = raw.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        label, newline, payload = stripped[3:-3].partition("\n")
        if newline and label.strip().lower() in {"", "json"}:
            return payload
    return raw


def _default_fix_prompt(error_text: str) -> str:
    """Default corrective user-message generator for retry loops.

    Called when no custom fix_prompt_fn is provided. Frames the error feedback
    generically without assuming the failure is schema-related.
    """
    return (
        f"Your previous response was rejected: {error_text}. "
        "Fix exactly that problem and return only the corrected JSON object — "
        "no markdown, no commentary. Keep every other field that was already valid."
    )


async def complete_with_repair(
    completion_fn: Callable[..., Awaitable[Any]],
    request: dict[str, Any],
    parse: Callable[[str], T],
    *,
    max_attempts: int = 3,
    invocation: AgentResponseInvocation | None = None,
    response_observer: Callable[[Any], None] | None = None,
    fix_prompt_fn: Callable[[str], str] | None = None,
) -> T:
    """Repair loop for API (litellm) completions with structured response parsing.

    Attempts to complete a request and parse the structured response up to max_attempts.
    On parse failure, appends the failed response and error feedback to the message
    history, then retries. The message history accumulates all prior failures and
    corrections so the model sees full context.

    Args:
        completion_fn: Async callable with signature completion_fn(**request),
                      typically litellm.acompletion.
        request: Dict with keys like 'model', 'messages', 'response_format', etc.
                Must contain 'messages' key (a list). The request is not mutated;
                a copy is used for retry to preserve caller's original.
        parse: Sync callable parse(raw_content: str) -> T that raises ValueError
               or json.JSONDecodeError on invalid output. Outer JSON fences are
               normalized before this callback runs.
        max_attempts: Maximum retry attempts (default 3).
        response_observer: Optional callback invoked with the reconstructed response
                          object after successful extraction, for capturing metadata
                          (usage, finish_reason, etc.) from the full response.
        fix_prompt_fn: Optional callable(error_text)->str returning the corrective
                      user-message content. Defaults to a generic hardened instruction.

    Returns:
        The parsed result of type T on first successful parse.

    Raises:
        ValueError or json.JSONDecodeError: The last exception encountered after
                                           max_attempts exhausted (preserves exception type).
    """
    messages = deepcopy(request["messages"])
    last_exc: Exception | None = None
    previous_call: AgentResponseCall | None = None

    for attempt in range(max_attempts):
        try:
            # Call the completion function with a copy of the request
            # to avoid mutating the caller's dict or list
            call_request = {**request, "messages": messages}
            call = None
            if invocation is not None:
                call = (
                    invocation.call(messages=messages)
                    if previous_call is None
                    else AgentResponseCall(
                        invocation,
                        invocation.root_request or invocation.call(messages=messages).request_display,
                        invocation.context.invocation_kind,
                        previous_call.call_id,
                    )
                )
            if call is None:
                response = await completion_fn(**call_request)
            else:
                async with call:
                    response = await call.complete(completion_fn, call_request)
                previous_call = call.response_call or call

            # Capture reconstructed response for metadata extraction (usage, finish_reason, etc.)
            if response_observer is not None:
                response_observer(response)

            # Extract content from the response envelope
            try:
                raw_content = _extract_content(response)
            except ValueError as exc:
                # _extract_content itself failed (malformed envelope, no content).
                # Treat same as parse failure but can't append assistant turn
                # if there's no content — only append the user fix turn.
                last_exc = exc
                if attempt < max_attempts - 1:
                    error_text = redact_secrets(str(exc))
                    fix_message = (fix_prompt_fn or _default_fix_prompt)(error_text)
                    messages.append({"role": "user", "content": fix_message})
                continue

            # Try to parse the content
            try:
                return parse(_unfence_json(raw_content))
            except (ValueError, json.JSONDecodeError) as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    # Append assistant turn (the bad response) and user fix turn
                    messages.append({"role": "assistant", "content": raw_content})
                    error_text = redact_secrets(str(exc))
                    fix_message = (fix_prompt_fn or _default_fix_prompt)(error_text)
                    messages.append({"role": "user", "content": fix_message})

        except Exception as exc:
            # Completion function itself failed (network, provider error, etc.)
            # Re-raise immediately without retry — these are not parse failures.
            raise

    # All attempts exhausted; re-raise the last parse/extract exception
    if last_exc is not None:
        raise last_exc
    # Should never reach here, but fail gracefully
    raise RuntimeError(f"complete_with_repair exhausted {max_attempts} attempts with no exception")


async def cli_complete_with_repair(
    run_fn: Callable[[], Awaitable[str]],
    resume_fn: Callable[[str, str], Awaitable[str]],
    parse: Callable[[str], T],
    *,
    session_id: str,
    max_attempts: int = 3,
) -> T:
    """Repair loop for CLI (subprocess) completions with structured response parsing.

    Attempts to run a CLI command and parse the structured output up to max_attempts.
    On parse failure, calls resume_fn to re-invoke the CLI with error feedback.
    Retries up to max_attempts.

    Args:
        run_fn: Async callable () -> str producing the first raw output.
        resume_fn: Async callable (session_id, fix_prompt) -> str that re-invokes
                  the CLI. Caller supplies the actual 'claude --resume' / fresh-rerun
                  mechanics. Called only on parse failure (attempts 2..max).
        parse: Sync callable parse(raw_content: str) -> T that raises ValueError
               or json.JSONDecodeError on invalid output.
        session_id: String session identifier passed to resume_fn.
        max_attempts: Maximum retry attempts (default 3).

    Returns:
        The parsed result of type T on first successful parse.

    Raises:
        ValueError or json.JSONDecodeError: The last exception encountered after
                                           max_attempts exhausted (preserves exception type).
    """
    last_exc: Exception | None = None

    for attempt in range(max_attempts):
        try:
            # Attempt 1: call run_fn; attempts 2+: call resume_fn
            if attempt == 0:
                raw = await run_fn()
            else:
                error_text = redact_secrets(str(last_exc)) if last_exc else "unknown error"
                fix_prompt = (
                    f"Your previous output could not be parsed: {error_text}. "
                    "Re-emit a corrected response that exactly matches the required format. "
                    "Output only the required content."
                )
                raw = await resume_fn(session_id, fix_prompt)

            # Try to parse
            try:
                return parse(raw)
            except (ValueError, json.JSONDecodeError) as exc:
                last_exc = exc
                if attempt >= max_attempts - 1:
                    # Last attempt exhausted, will re-raise below
                    break

        except Exception as exc:
            # CLI invocation itself failed (subprocess error, etc.)
            # Re-raise immediately without retry.
            raise

    # All attempts exhausted; re-raise the last parse exception
    if last_exc is not None:
        raise last_exc
    # Should never reach here, but fail gracefully
    raise RuntimeError(f"cli_complete_with_repair exhausted {max_attempts} attempts with no exception")
