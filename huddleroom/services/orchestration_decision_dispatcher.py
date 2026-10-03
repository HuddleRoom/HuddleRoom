from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationDecision, OrchestrationGoal, OrchestrationRun
from huddleroom.services.orchestration_steering import (
    OrchestrationSteeringService, SteeringDomainError, active_direction_ids,
    steering_versions_from_snapshot,
)


class OrchestrationDecisionDispatcher:
    """Maps one validated LLM coordination decision to the matching
    execute_*_action on OrchestrationService. Code routes; the LLM only chose
    the action. Exactly one action per decision, recorded idempotently."""

    def __init__(self, service: Any) -> None:
        self._service = service

    @staticmethod
    def action_key(run_id: uuid.UUID, action_type: str, request: dict[str, Any]) -> str:
        """Stable replay identity for a regenerated coordination decision.

        Reasons explain a decision but do not change its requested work.
        """
        fingerprint_request = {key: value for key, value in request.items() if key != "reason"}
        encoded = json.dumps(
            fingerprint_request, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        fingerprint = hashlib.sha256(encoded).hexdigest()[:16]
        return f"run:{run_id}:kind:{action_type}:request:{fingerprint}"

    async def dispatch(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        decision: OrchestrationDecision,
    ) -> OrchestrationAction | None:
        if decision.validator_status != "accepted":
            return None
        parsed = decision.parsed_decision or {}
        action_type = parsed.get("action_type")
        if not action_type:
            return None

        executor = self._EXECUTORS.get(action_type)
        if executor is None:
            # Validated but not dispatchable from the runtime loop (e.g. baseline
            # authority actions handled elsewhere). No-op rather than guess.
            return None
        request = self._service.canonical_decision_request(action_type, parsed, run_id=run.id)
        input_snapshot = getattr(decision, "input_snapshot", {}) or {}
        versions = steering_versions_from_snapshot(input_snapshot)
        steering = None
        if versions is not None:
            goal = await db.get(OrchestrationGoal, run.goal_id)
            if goal is None:
                return None
            steering = OrchestrationSteeringService()
            await steering.assert_current_versions(db, goal, run, versions)
            if active_direction_ids(input_snapshot) and action_type in {"pause_run", "ask_human"}:
                raise SteeringDomainError(
                    "steering_forbidden_effect", 409, "Use the dedicated control",
                )
        method = getattr(self._service, executor)
        if action_type == "request_roadmap_replan":
            action_key = await self._service.roadmap_replan_action_key(db, run)
        else:
            action_key = self.action_key(run.id, action_type, request)
        if versions is not None:
            action_key, *_ = (
                await self._service._steering_action_fence(db, run.id, action_key, decision.id)
            )
        if action_type == "request_roadmap_replan":
            action = await method(
                db, run_id=run.id, request=request,
                idempotency_key=action_key,
                decision_id=decision.id,
            )
        else:
            action = await method(
                db,
                run_id=run.id,
                request=request,
                idempotency_key=action_key,
                decision_id=decision.id,
            )
        if steering is not None and action.status != "failed":
            for request_id in active_direction_ids(input_snapshot):
                await steering.link_result(db, request_id, decision.id, action.id)
        return action

    # action_type -> OrchestrationService method name. Only the actions the
    # authorized-execution loop is allowed to take on an Outcome run.
    _EXECUTORS: dict[str, str] = {
        "noop": "execute_noop_action",
        "request_plan": "execute_request_plan_action",
        "request_roadmap_replan": "execute_request_roadmap_replan_action",
        "request_plan_revision": "execute_request_plan_revision_action",
        "accept_plan": "execute_accept_plan_action",
        "create_delegation_task": "execute_create_delegation_task_action",
        "request_verification": "execute_request_verification_action",
        "retry_task": "execute_retry_task_action",
        "reassign_task": "execute_reassign_task_action",
        "schedule_meeting": "execute_schedule_meeting_action",
        "start_protocol": "execute_start_protocol_action",
        "ask_human": "execute_ask_human_action",
        "pause_run": "execute_pause_run_action",
        "record_warning": "execute_record_warning_action",
        "suggest_agent": "execute_suggest_agent_action",
    }
