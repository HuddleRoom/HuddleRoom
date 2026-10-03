from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Awaitable, Callable, Mapping
from uuid import UUID

import litellm

from huddleroom.config import settings
from huddleroom.services.llm_structured_repair import complete_with_repair
from huddleroom.services.orchestration_agent_definition_analyzer import (
    SemanticAgentAnalysisError,
    redact_semantic_payload,
)
from huddleroom.services.orchestration_llm_decision_adapter import (
    _extract_content,
    _full_completion_error,
    _redact_secrets,
    orchestrator_preamble,
)

MAX_RATIONALE_LENGTH = 2000


@dataclass(frozen=True)
class ManagerAssessment:
    verdict: str
    selected_key: str
    rationale: str


def parse_manager_assessment(payload: Any, offered_keys: set[str], recommendation: str) -> ManagerAssessment:
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version", "verdict", "selected_key", "rationale"
    }:
        raise ValueError("manager assessment must contain exactly the required fields")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    selected_key = payload["selected_key"]
    if selected_key not in offered_keys:
        raise ValueError("selected_key must be an offered candidate key")
    verdict = payload["verdict"]
    expected = "confirm" if selected_key == recommendation else "override"
    if verdict != expected:
        raise ValueError(f"verdict must be {expected} for selected_key")
    rationale = payload["rationale"]
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > MAX_RATIONALE_LENGTH:
        raise ValueError("rationale must be a non-empty bounded string")
    return ManagerAssessment(verdict, selected_key, rationale.strip())


class ManagerSelectionAnalyzer:
    def __init__(self, completion_fn: Callable[..., Awaitable[Any]] | None = None) -> None:
        self._completion_fn = completion_fn or litellm.acompletion

    @staticmethod
    def build_request(payload: Mapping[str, Any], project=None) -> dict[str, Any]:
        safe_payload = redact_semantic_payload(payload)
        prompt = (
            orchestrator_preamble(project, goal=payload.get("goal")) + "\n\n"
            "You are the orchestrator's manager-selection reviewer. Select the principal most able to own coordination, "
            "prioritization, trade-offs, escalation, and final routing for this specific goal. The deterministic "
            "ranking is evidence, not an answer. Independently inspect the goal and every offered candidate. Consider "
            "explicit ownership and decision authority, goal/risk fit, coordination ability, conflicting or merely "
            "advisory instructions, capabilities, and workload. Do not reward titles without behavioral evidence. Do "
            "not infer missing capabilities, permissions, availability, or authority. Choose exactly one key copied "
            "from candidates. Never invent a candidate. Use no_manager only when management is genuinely unnecessary. "
            "Use human_as_manager when human judgment or authority is materially preferable, not merely because agent "
            "evidence is imperfect. verdict is confirm exactly when the choice equals deterministic_recommendation; "
            "otherwise it is override. Before answering, verify that the choice can own the goal and compare it with "
            "the strongest alternative. Return only JSON with exactly schema_version (1), verdict (confirm or override), "
            "selected_key, and a concise input-grounded rationale; no hidden chain-of-thought, markdown, or extra keys."
        )
        return {
            "model": settings.orchestration_model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(safe_payload, sort_keys=True, default=str)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": 32768,
        }

    async def review(self, payload: Mapping[str, Any], project=None, *, project_id: UUID | None = None) -> ManagerAssessment:
        request = self.build_request(payload, project=project)
        return await self.review_request(request, project_id=project_id)

    async def review_request(self, request: Mapping[str, Any], *, project_id: UUID | None = None) -> ManagerAssessment:
        request = redact_semantic_payload(request)

        # Precompute from request (before completion)
        payload = json.loads(request["messages"][1]["content"])
        offered = {candidate["key"] for candidate in payload["candidates"]}
        rec = payload["deterministic_recommendation"]

        try:
            args = (self._completion_fn, request, lambda raw: parse_manager_assessment(json.loads(raw), offered, rec))
            if project_id is None:
                return await complete_with_repair(*args)
            from huddleroom.services.agent_response_stream import AgentResponseInvocation, InvocationContext
            return await complete_with_repair(*args, invocation=AgentResponseInvocation(InvocationContext(
                project_id, "system", "orchestrator", "Orchestrator", "api", "manager_selection",
                request["model"], "Select the best manager for this goal.",
            )))
        except Exception as exc:
            # ponytail: repair doesn't expose raw on exhaustion, provider_error/invalid_response distinction lost
            raise SemanticAgentAnalysisError(
                "invalid_response",
                _full_completion_error(exc),
                request,
                None,
            ) from exc
