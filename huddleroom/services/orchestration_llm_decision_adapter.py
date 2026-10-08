from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import re
from typing import Any, Awaitable, Callable, Mapping, Protocol
from uuid import UUID

from huddleroom.config import settings
from huddleroom.services.orchestration_completion import (
    get_orchestration_completion,
    orchestration_runtime_metadata,
)
from huddleroom.services.orchestration_decision_validator import (
    ALLOWED_ACTION_SCHEMAS,
    OPTIONAL_ACTION_FIELDS,
    validate_orchestration_decision,
)
from huddleroom.services.orchestration_wake_when import normalize_wake_when, wake_when_prompt_table
from huddleroom.services.secret_redaction import redact_secrets


CompletionFn = Callable[..., Awaitable[Any]]
INVALID_LLM_OUTPUT_ACTION_TYPE = "invalid_llm_output"
_MAX_ERROR_MESSAGE_LEN = 120
PROGRESS_CONTRACT = (
    "Progress duty: move the goal to completion. Every decision must advance or unblock a success criterion. "
    "Wait only when every useful action is blocked on a named event or time.\n"
    "Fresh run (no accepted plan and no orchestrated work): normally request_plan to the best planning agent in the roster. "
    "Use request_human_decision only when a missing owner decision truly blocks starting. "
    "If a plan task is already in flight, wait for it.\n"
    "Run with history: review progress_view and untracked_follow_ups. Handle the first untracked follow-up. "
    "Otherwise start work on a no_work criterion. Otherwise verify an evidence_pending criterion. "
    "Never create a second task for something that already has one. "
    "When you create a task for a criterion or a meeting action item, add criterion:<key> or meeting_action_item:<id> "
    "to its inputs.\n"
    'Authority: run.phase == "authorized" means the owner explicitly started the goal. Earlier "prepare only" or '
    '"do not start" wording is satisfied by that start. Specific restrictions still bind: live sends or messages, '
    "spending, purchases, account changes, publishing, and destructive operations. "
    "Route those actions to request_human_decision.\n"
    "Waiting: a wait (noop on the decision path, continue on the supervision path) must carry wake_when.\n"
    "Reason: the reason states how the action advances the goal."
)
# ponytail: backward compatibility alias for code that may import _redact_secrets directly
_redact_secrets = redact_secrets


@dataclass(frozen=True)
class OrchestrationDecisionAdapterResult:
    input_snapshot: dict
    llm_output: dict
    parsed_decision: dict


class LLMDecisionAdapter(Protocol):
    async def decide(self, context: Mapping[str, Any]) -> OrchestrationDecisionAdapterResult:
        ...


def orchestrator_preamble(project=None, *, goal=None, meeting=None) -> str:
    """Shared framing for every control-plane orchestrator LLM call: one platform
    sentence, plus optional project + (goal headline | meeting headline). All args
    optional so callers that pass nothing get only the platform sentence
    (preserves prior behavior). goal and meeting are mutually exclusive by caller
    convention."""
    lines = [
        "You are the orchestration control plane of a multi-agent software-delivery platform: "
        "a team of AI agents self-organizes to build and operate software while you make only "
        "the coordination judgments — you never produce the work artifacts yourself."
    ]

    if project and isinstance(project, Mapping):
        name = project.get("name")
        description = project.get("description")
        if name:
            name_redacted = redact_secrets(str(name))
            if description:
                desc_redacted = redact_secrets(str(description))
                lines.append(f"Project: {name_redacted} — {desc_redacted}")
            else:
                lines.append(f"Project: {name_redacted}")

    # goal and meeting are mutually exclusive by caller convention; meeting takes precedence if both provided
    if meeting and isinstance(meeting, Mapping):
        title = meeting.get("title")
        meeting_type = meeting.get("meeting_type")
        if title and meeting_type:
            title_redacted = redact_secrets(str(title))
            type_redacted = redact_secrets(str(meeting_type))
            lines.append(f"Meeting: {title_redacted} ({type_redacted})")
    elif goal and isinstance(goal, Mapping):
        objective = goal.get("objective")
        if objective:
            parts = [redact_secrets(str(objective))]
            meta = []
            weight = goal.get("weight")
            if weight:
                meta.append(f"weight: {redact_secrets(str(weight))}")
            status = goal.get("status")
            if status:
                meta.append(f"status: {redact_secrets(str(status))}")
            if meta:
                lines.append(f"Goal: {parts[0]} ({', '.join(meta)})")
            else:
                lines.append(f"Goal: {parts[0]}")

    return "\n".join(lines)


def build_orchestration_decision_messages(context: Mapping[str, Any], *, project=None, goal=None) -> list[dict[str, str]]:
    allowed_actions = {
        action_type: {
            "required": sorted(required_fields),
            "optional": sorted(OPTIONAL_ACTION_FIELDS.get(action_type, frozenset())),
        }
        for action_type, required_fields in ALLOWED_ACTION_SCHEMAS.items()
    }
    restrictions = (
        "You are the orchestrator's control-plane decision assistant. "
        "You choose coordination actions only. Agents produce all work artifacts and HuddleRoom code executes side effects "
        "after validation. You must not write plans, must not write code, must not write tests, must not write reviews, "
        "must not write validation reports, must not write meeting decisions, must not write project artifacts, "
        "must not write file content, must not write diffs or patches, and must not write final summaries."
    )
    shape = (
        "Return exactly one JSON object with this shape: "
        '{"decision":{"action_type":"<allowed type>", ...required fields, "reason":"how this advances the goal"}}. '
        "The decision object must use one allowed action schema. "
        "Top-level reason is optional for every action."
    )
    wake_when = (
        'A wait requires wake_when. noop is a wait and must carry wake_when shaped {"events":[{"event_type":"<event>",'
        '"matcher":{"<key>":"<uuid>"}}],"recheck_after_seconds":<int>,"expected_result":"<what you expect on wake>"}. '
        "Give events and/or recheck_after_seconds. At most 5 events. Each matcher must be non-empty, use only the listed "
        "keys, and use UUID values from the context. Use recheck_after_seconds to wait on child goals or steering, "
        "which emit no event. "
        f"Allowed wake events and matcher keys: {wake_when_prompt_table()}"
    )
    system_text = (
        restrictions
        + "\n\n"
        + PROGRESS_CONTRACT
        + "\n\n"
        + shape
        + "\n\n"
        + wake_when
        + "\n\n"
        + f"Allowed action schemas: {json.dumps(allowed_actions, sort_keys=True)}"
    )
    preamble = orchestrator_preamble(project, goal=goal)
    system_text = preamble + "\n\n" + system_text
    return [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": json.dumps(context, sort_keys=True, default=str),
        },
    ]


def parse_decision_content(raw_content: str) -> dict:
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError:
        return _invalid_decision("LLM output was not valid JSON")

    if not isinstance(payload, Mapping):
        return _invalid_decision("LLM output JSON root must be an object")

    if "decision" not in payload:
        return _invalid_decision("LLM output must contain a decision object")
    decision = payload.get("decision")
    if not isinstance(decision, Mapping):
        return _invalid_decision("LLM output decision must be an object")

    return dict(decision)


class OrchestrationDecisionAdapter:
    def __init__(self, model: str | None = None, completion_fn: CompletionFn | None = None) -> None:
        self.model = model or settings.orchestration_model
        self._completion_fn = get_orchestration_completion(completion_fn)

    async def decide(self, context: Mapping[str, Any], *, project=None, goal=None) -> OrchestrationDecisionAdapterResult:
        from huddleroom.services.llm_structured_repair import complete_with_repair
        from huddleroom.services.agent_response_stream import AgentResponseInvocation, InvocationContext

        input_snapshot = deepcopy(dict(context))
        messages = build_orchestration_decision_messages(input_snapshot, project=project, goal=goal)

        request = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }

        _holder = {}

        def _observe_response(response: Any) -> None:
            try:
                _holder["raw"] = _extract_content(response)
            except ValueError:
                pass

        def _parse(raw: str) -> dict:
            decision = parse_decision_content(raw)
            if decision.get("action_type") == INVALID_LLM_OUTPUT_ACTION_TYPE:
                raise ValueError(decision.get("reason") or "invalid decision output")
            # Context-free structural validation; the rejection reason is fed back to the model on repair.
            verdict = validate_orchestration_decision(decision)
            if not verdict.accepted:
                raise ValueError(verdict.rejection_reason)
            if decision.get("action_type") == "noop":
                return {**decision, "wake_when": normalize_wake_when(decision["wake_when"])}
            return decision

        try:
            try:
                project_id = (
                    UUID(str(project.get("id")))
                    if isinstance(project, Mapping) and project.get("id")
                    else None
                )
            except (TypeError, ValueError):
                project_id = None
            invocation_kind, runtime_model = orchestration_runtime_metadata(self._completion_fn, self.model)
            invocation = (
                AgentResponseInvocation(InvocationContext(
                    project_id, "system", "orchestrator", "Orchestrator", invocation_kind,
                    "decision", runtime_model,
                    "Choose the next orchestration action for this goal.",
                ))
                if project_id is not None else None
            )
            decision = await complete_with_repair(
                self._completion_fn, request, _parse, response_observer=_observe_response,
                **({"invocation": invocation} if invocation else {})
            )
        except ValueError as exc:
            return _invalid_adapter_result(
                input_snapshot,
                str(exc),
                response_error=str(exc),
                raw_content=_holder.get("raw"),
            )
        except Exception as exc:
            # ponytail: pure function, no db/goal_id in scope. No warning is currently
            # emitted for provider errors on this test-only path; wiring is explicitly
            # deferred. This returns invalid_llm_output decision.
            safe_error = _safe_completion_error(exc)
            return _invalid_adapter_result(
                input_snapshot,
                f"LLM completion failed: {safe_error}",
                completion_error=safe_error,
                raw_content=_holder.get("raw"),
            )

        return OrchestrationDecisionAdapterResult(
            input_snapshot=input_snapshot,
            llm_output={"raw_content": _holder.get("raw")},
            parsed_decision=decision,
        )


def _invalid_decision(reason: str) -> dict:
    return {"action_type": INVALID_LLM_OUTPUT_ACTION_TYPE, "reason": reason}


def _safe_completion_error(exc: Exception) -> str:
    error = _full_completion_error(exc)
    if len(error) <= _MAX_ERROR_MESSAGE_LEN:
        return error
    return f"{error[:_MAX_ERROR_MESSAGE_LEN - 3]}..."


def _full_completion_error(exc: Exception) -> str:
    name = type(exc).__name__
    message = redact_secrets(" ".join(str(exc).split()))
    if not message:
        return name
    return f"{name}: {message}"


def _invalid_adapter_result(
    input_snapshot: dict,
    reason: str,
    *,
    response_error: str | None = None,
    completion_error: str | None = None,
    raw_content: str | None = None,
) -> OrchestrationDecisionAdapterResult:
    llm_output = {"raw_content": raw_content}
    if response_error is not None:
        llm_output["response_error"] = response_error
    if completion_error is not None:
        llm_output["completion_error"] = completion_error
    return OrchestrationDecisionAdapterResult(
        input_snapshot=input_snapshot,
        llm_output=llm_output,
        parsed_decision=_invalid_decision(reason),
    )


def _extract_content(response: Any) -> str:
    choices = _response_field(response, "choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response did not include choices[0]")

    choice = choices[0]
    message = _response_field(choice, "message")
    content = _response_field(message, "content")
    if not isinstance(content, str):
        raise ValueError("LLM response content must be a string")
    return content


def _response_field(value: Any, field: str) -> Any:
    if isinstance(value, Mapping):
        if field not in value:
            raise ValueError(f"LLM response missing '{field}'")
        return value[field]
    if not hasattr(value, field):
        raise ValueError(f"LLM response missing '{field}'")
    return getattr(value, field)
