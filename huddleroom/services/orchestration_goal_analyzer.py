from dataclasses import dataclass
import json
import os
from typing import Any, Awaitable, Callable, Mapping
from uuid import UUID

import litellm

from huddleroom.config import settings
from huddleroom.services.llm_structured_repair import complete_with_repair
from huddleroom.services.orchestration_llm_decision_adapter import (
    _extract_content,
    _full_completion_error,
    _redact_secrets,
    _safe_completion_error,
    orchestrator_preamble,
)


DESTINATIONS = frozenset({
    "objective",
    "success_criteria",
    "orchestrator_context.assumptions",
    "orchestrator_context.resolved_clarifications",
    "orchestrator_context.constraints",
    "orchestrator_context.budget_deadline",
    "orchestrator_context.execution_details",
})


@dataclass(frozen=True)
class GoalAssumption:
    text: str
    destination: str


@dataclass(frozen=True)
class GoalQuestion:
    question: str
    rationale: str
    destination: str


@dataclass(frozen=True)
class GoalAnalysis:
    assumptions: tuple[GoalAssumption, ...]
    questions: tuple[GoalQuestion, ...]
    unsafe_unresolved: bool


def _text(item: Mapping[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be non-empty")
    return value.strip()


def _destination(item: Mapping[str, Any]) -> str:
    value = _text(item, "destination")
    if item["destination"] != value or value not in DESTINATIONS:
        raise ValueError(f"invalid destination: {value}")
    return value


def parse_goal_analysis(payload: Any) -> GoalAnalysis:
    if not isinstance(payload, Mapping):
        raise ValueError("analysis must be an object")
    raw_assumptions = payload.get("assumptions", [])
    raw_questions = payload.get("questions", [])
    if not isinstance(raw_assumptions, list) or not isinstance(raw_questions, list):
        raise ValueError("assumptions and questions must be lists")
    if not isinstance(payload.get("unsafe_unresolved", False), bool):
        raise ValueError("unsafe_unresolved must be boolean")
    assumptions = tuple(
        GoalAssumption(_text(item, "text"), _destination(item))
        for item in raw_assumptions if isinstance(item, Mapping)
    )
    if len(assumptions) != len(raw_assumptions):
        raise ValueError("assumptions must contain objects")
    questions = tuple(
        GoalQuestion(_text(item, "question"), _text(item, "rationale"), _destination(item))
        for item in raw_questions if isinstance(item, Mapping)
    )
    if len(questions) != len(raw_questions):
        raise ValueError("questions must contain objects")
    return GoalAnalysis(assumptions, questions, payload.get("unsafe_unresolved", False))


def _unfence_json(raw: str) -> str:
    stripped = raw.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        label, newline, payload = stripped[3:-3].partition("\n")
        if newline and label.strip() in {"", "json"}:
            return payload
    return raw


class GoalClarificationAnalyzer:
    def __init__(self, completion_fn: Callable[..., Awaitable[Any]] | None = None) -> None:
        self._completion_fn = completion_fn or litellm.acompletion

    def _existing_messages_for(self, goal_snapshot: Mapping[str, Any], project=None) -> list[dict[str, str]]:
        system_content = (
            "Read this complete goal. Return JSON only as this exact object schema: "
            '{"assumptions":[{"text":"short safe assumption","destination":"one allowed destination"}],'
            '"questions":[{"question":"material question","rationale":"why the answer matters",'
            '"destination":"one allowed destination"}],"unsafe_unresolved":false}. '
            "assumptions and questions are always arrays, including when empty; their items are objects, never strings. "
            "Each assumption item has only text and destination. Each question item has only question, rationale, "
            "and destination: no options, no impact, and no extra fields. unsafe_unresolved is a boolean. "
            "Allowed destinations are exactly: objective, success_criteria, orchestrator_context.assumptions, "
            "orchestrator_context.resolved_clarifications, orchestrator_context.constraints, "
            "orchestrator_context.budget_deadline, orchestrator_context.execution_details. "
            "Never invent, alter, or use another destination. "
            "The supplied objective, original request, success criteria, constraints, budget, and existing "
            "orchestrator context are authoritative. Assumptions must not add, remove, weaken, or relabel "
            "safety, authorization, network, cost, deadline, retry, or stop boundaries. Do not claim runtime "
            "facts from this analysis, including calls, spend, mutations, current authorization, confirmation, "
            "or network activity. Turn a material missing boundary into a question; omit an immaterial unknown. "
            "Generate questions from this goal; there is no checklist. Ask only when different answers "
            "materially change scope, success, constraints, cost, deadline, or execution. Record safe "
            "reversible ambiguity as short assumptions. Never record an assumption on the same axis or destination you ask a question about; if you ask, do not also assume on it. Keep proposed objective/success wording short. "
            "Return zero questions for a sufficiently actionable goal."
        )
        preamble = orchestrator_preamble(project, goal=goal_snapshot)
        return [
            {
                "role": "system",
                "content": preamble + "\n\n" + system_content,
            },
            {
                "role": "user",
                "content": json.dumps(goal_snapshot, sort_keys=True, default=str),
            },
        ]

    def build_request(self, snapshot: dict, project=None) -> dict:
        return {
            "model": settings.orchestration_model,
            "messages": self._existing_messages_for(snapshot, project),
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }

    async def analyze_request(self, request: dict, *, project_id: UUID | None = None) -> GoalAnalysis:
        model = request.get("model")
        messages = request.get("messages")
        if settings.debug:
            print(_redact_secrets(f"LLM request model={model} messages={messages}"), flush=True)
        try:
            def _parse(raw):
                if settings.debug:
                    print(_redact_secrets(f"LLM response model={model} content={raw}"), flush=True)
                return parse_goal_analysis(json.loads(_unfence_json(raw)))
            if project_id is None:
                return await complete_with_repair(self._completion_fn, request, _parse)
            from huddleroom.services.agent_response_stream import AgentResponseInvocation, InvocationContext
            return await complete_with_repair(
                self._completion_fn, request, _parse,
                invocation=AgentResponseInvocation(InvocationContext(
                    project_id, "system", "orchestrator", "Orchestrator", "api", "goal_analysis", model,
                    "Identify safe assumptions and material clarification questions for this goal.",
                )),
            )
        except Exception as exc:
            if settings.debug:
                format_error = (
                    _full_completion_error
                    if os.environ.get("HUDDLEROOM_ORCHESTRATION_BASELINE_E2E") == "true"
                    or os.environ.get("RALLY_ORCHESTRATION_BASELINE_E2E") == "true"
                    else _safe_completion_error
                )
                print(f"LLM failed model={model} error={format_error(exc)}", flush=True)
            raise

    async def analyze(self, snapshot: dict, *, project_id: UUID | None = None) -> GoalAnalysis:
        return await self.analyze_request(self.build_request(snapshot), project_id=project_id)
