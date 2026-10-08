import json
import logging
from typing import Any, Awaitable, Callable, Literal
from uuid import UUID

from huddleroom.config import settings
from huddleroom.models.orchestration import EXPLICIT_PLANNING_MODE_BY_GOAL_TYPE, explicit_planning_rationale
from huddleroom.schemas.orchestration import GoalShapeDecision, OrchestrationGoalCreate
from huddleroom.services.orchestration_goal_definition import _redacted_json
from huddleroom.services.llm_structured_repair import complete_with_repair
from huddleroom.services.orchestration_completion import (
    get_orchestration_completion,
    orchestration_runtime_metadata,
)
from huddleroom.services.orchestration_llm_decision_adapter import (
    _safe_completion_error,
    orchestrator_preamble,
)

logger = logging.getLogger(__name__)

_SHAPE_KEYS = frozenset({"goal_type", "goal_type_rationale", "planning_mode", "planning_rationale"})

SHAPE_PROMPT = (
    "Classify the STRUCTURE of this goal. Return JSON only, exactly "
    '{"goal_type":"outcome|roadmap|continuous","goal_type_rationale":"one short sentence",'
    '"planning_mode":"delivery|implementation|roadmap|continuous_policy",'
    '"planning_rationale":"one short sentence"} and no other fields. '
    "outcome = one bounded result, even when large, costly, or with many success criteria. "
    "roadmap = a finite sequence or graph of distinct outcomes, defined progressively, each deliverable on its own. "
    "continuous = ongoing recurring work (monitoring, triage, periodic upkeep) with no natural end result. "
    "Decide from structure only: size, cost, budget, effort, number of success criteria, or number of work "
    "functions never make a goal a roadmap. planning_mode must fit goal_type: roadmap→roadmap; "
    "continuous→continuous_policy; outcome→delivery when the result needs coordinated work across several "
    "functions, implementation when it is a single leaf technical change one specialist can plan. "
    "Rationales are short and user-facing."
)


class GoalShapeResolutionError(Exception):
    def __init__(self, reason: Literal["provider_error", "invalid_response"], message: str) -> None:
        self.reason = reason
        super().__init__(message)


def _parse_shape(raw: str) -> GoalShapeDecision:
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != _SHAPE_KEYS:
        raise ValueError(f"return exactly {sorted(_SHAPE_KEYS)}")
    return GoalShapeDecision(**payload, goal_type_source="auto")


class OrchestrationGoalShapeService:
    def __init__(self, completion_fn: Callable[..., Awaitable[Any]] | None = None) -> None:
        self._completion_fn = get_orchestration_completion(completion_fn)

    async def resolve_creation_shape(self, project_id: UUID, data: OrchestrationGoalCreate) -> GoalShapeDecision:
        if data.goal_type != "auto":
            return GoalShapeDecision(
                goal_type=data.goal_type,
                goal_type_source="user",
                goal_type_rationale="Selected by user at goal creation.",
                planning_mode=EXPLICIT_PLANNING_MODE_BY_GOAL_TYPE[data.goal_type],
                planning_rationale=explicit_planning_rationale(data.goal_type),
            )
        return await self._classify(project_id, data)

    async def _classify(self, project_id: UUID, data: OrchestrationGoalCreate) -> GoalShapeDecision:
        from huddleroom.services.agent_response_stream import AgentResponseInvocation, InvocationContext

        request = {
            "model": settings.orchestration_model,
            "messages": [
                {
                    "role": "system",
                    "content": orchestrator_preamble(goal={"objective": data.objective}) + "\n\n" + SHAPE_PROMPT,
                },
                {
                    "role": "user",
                    "content": json.dumps(_redacted_json({
                        "objective": data.objective,
                        "original_request": data.original_request,
                        "success_criteria": data.success_criteria,
                        "constraints": data.constraints,
                        "budget": data.budget,
                    }), sort_keys=True, default=str),
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        invocation_kind, runtime_model = orchestration_runtime_metadata(self._completion_fn, request["model"])
        try:
            return await complete_with_repair(
                self._completion_fn, request, _parse_shape,
                invocation=AgentResponseInvocation(InvocationContext(
                    project_id, "system", "orchestrator", "Orchestrator", invocation_kind, "goal_shape",
                    runtime_model, "Classify goal structure.",
                )),
            )
        except ValueError as exc:  # JSONDecodeError is a ValueError
            logger.warning("Goal shape classifier returned an invalid response: %s", _safe_completion_error(exc))
            raise GoalShapeResolutionError("invalid_response", _safe_completion_error(exc)) from exc
        except Exception as exc:
            logger.warning("Goal shape classifier call failed: %s", _safe_completion_error(exc))
            raise GoalShapeResolutionError("provider_error", _safe_completion_error(exc)) from exc
