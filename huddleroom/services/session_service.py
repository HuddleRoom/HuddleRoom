from __future__ import annotations

import uuid
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation
from sqlalchemy import select, update, and_
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException

from huddleroom.models.session import Session
from huddleroom.models.agent import Agent
from huddleroom.models.graph import GraphRun
from huddleroom.models.project import Project
from huddleroom.models.task import Task
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun
from huddleroom.schemas.session import SessionCreate, SessionOutputResponse
from huddleroom.services.event_bus import emit_event
from huddleroom.services.session_sync import sync_task_from_session
from huddleroom.services.project_service import ProjectService

STALE_RUNNING_SESSION_ERROR = "stale_running_session: exceeded recovery timeout before watchdog pass"
STALE_PENDING_SESSION_ERROR = "stale_pending_session: never started before recovery timeout"


class SessionClaimAttention(HTTPException):
    """A claim was refused and its stable blocker must be committed alone."""
    def __init__(self, run_id: uuid.UUID, blocker: dict, detail: str) -> None:
        super().__init__(status_code=409, detail=detail)
        self.run_id = run_id
        self.blocker = blocker


class SessionService:
    @staticmethod
    def _positive_limit(value: object, label: str) -> int:
        if isinstance(value, bool):
            raise HTTPException(status_code=422, detail=f"{label} must be a positive whole number")
        try:
            amount = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"{label} must be a positive whole number") from exc
        if not amount.is_finite() or amount <= 0 or amount != amount.to_integral_value():
            raise HTTPException(status_code=422, detail=f"{label} must be a positive whole number")
        return int(amount)

    @staticmethod
    def _roadmap_context(task: Task | None) -> dict:
        if task is None:
            return {}
        contract = (task.metadata_ or {}).get("orchestration_contract", {})
        context = contract.get("orchestrator_context", {}) if isinstance(contract, dict) else {}
        return context if isinstance(context, dict) else {}

    def _session_limits_from_remaining(self, agent: Agent, remaining: dict[str, str]) -> dict:
        if any(Decimal(value) <= 0 for value in remaining.values()) or (
            "max_turns" in remaining and Decimal(remaining["max_turns"]) < 1
        ):
            raise ValueError("Execution budget is exhausted")
        limits = {}
        if "max_tokens" in remaining:
            limits["max_tokens"] = min(
                int(Decimal(remaining["max_tokens"])),
                self._positive_limit(agent.config.get("max_tokens", 4096), "Agent max_tokens"),
            )
        if "max_hours" in remaining:
            limits["timeout"] = min(
                int(Decimal(remaining["max_hours"]) * 3600),
                self._positive_limit(agent.config.get("session_timeout_seconds", 3600), "Agent session_timeout_seconds"),
            )
        return {**limits, "_roadmap_budget_enforced": True}

    @staticmethod
    def _clamp_to_action_allocation(action: OrchestrationAction | None, remaining: dict[str, str]) -> dict[str, str]:
        ledger = action.budget_ledger if action is not None else None
        if not (isinstance(ledger, dict) and isinstance(ledger.get("allocation"), dict)
                and {"reserved", "committed", "consumed"}.issubset(ledger)):
            return remaining
        try:
            allocation = {
                key: Decimal(str(ledger["allocation"].get(key, "0")))
                for key in remaining
            }
        except (InvalidOperation, ValueError):
            return remaining
        if any(not value.is_finite() or value < 0 for value in allocation.values()):
            return remaining
        return {
            key: format(min(Decimal(value), allocation[key]), "f")
            for key, value in remaining.items()
        }

    async def _immutable_roadmap_context(self, db: AsyncSession, task: Task | None) -> dict:
        """Rebuild mutable-work policy from immutable lineage, never caller-editable task JSON."""
        if task is None:
            return {}
        from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRoadmapItem, OrchestrationRoadmapVersion
        from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision

        item = await db.scalar(select(OrchestrationRoadmapItem).where(OrchestrationRoadmapItem.task_id == task.id))
        task_action = None
        child = None
        if item is None:
            from huddleroom.models.orchestration import OrchestrationAction, OrchestrationRun
            task_action = await db.scalar(select(OrchestrationAction).where(
                OrchestrationAction.target_type == "task",
                OrchestrationAction.target_id == task.id,
                OrchestrationAction.action_type.in_(("create_delegation_task", "request_plan", "request_plan_revision")),
                OrchestrationAction.status == "completed",
            ).order_by(OrchestrationAction.created_at.asc()).limit(1))
            run = await db.scalar(select(OrchestrationRun).join(
                OrchestrationAction, OrchestrationAction.run_id == OrchestrationRun.id,
            ).where(
                OrchestrationAction.target_type == "task",
                OrchestrationAction.target_id == task.id,
                OrchestrationAction.action_type.in_(("create_delegation_task", "request_plan", "request_plan_revision")),
                OrchestrationAction.status == "completed",
            ))
            child = await db.get(OrchestrationGoal, run.goal_id) if run is not None else None
            if child is None or child.parent_goal_id is None:
                return self._roadmap_context(task)
            item = await db.scalar(select(OrchestrationRoadmapItem).where(
                OrchestrationRoadmapItem.child_goal_id == child.id
            ))
            if item is None:
                return self._roadmap_context(task)
        else:
            from huddleroom.models.orchestration import OrchestrationAction
            task_action = await db.scalar(select(OrchestrationAction).where(
                OrchestrationAction.target_id == task.id,
                OrchestrationAction.target_type == "task",
                OrchestrationAction.action_type.in_(("create_delegation_task", "request_plan", "request_plan_revision")),
                OrchestrationAction.status == "completed",
            ).order_by(OrchestrationAction.created_at.asc()).limit(1))
        parent = await db.get(OrchestrationGoal, item.goal_id)
        version = await db.get(OrchestrationRoadmapVersion, item.first_version_id)
        if parent is None or version is None:
            raise HTTPException(status_code=409, detail="Roadmap task lineage is invalid")
        snapshot = item.item_snapshot or {}
        key = f"roadmap_unstaged_mutation:{version.id}:{item.item_key}"
        approval = await db.scalar(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.goal_id == parent.id,
            OrchestrationAuthorityDecision.run_id == version.run_id,
            OrchestrationAuthorityDecision.decision_key == key,
            OrchestrationAuthorityDecision.status == "answered",
            OrchestrationAuthorityDecision.selected_option == "approve",
            OrchestrationAuthorityDecision.authority == "human",
        ).order_by(OrchestrationAuthorityDecision.decided_at.desc()).limit(1))
        unstaged = bool(snapshot.get("mutates_shared_state")) and (
            snapshot.get("staging_boundary") is None or approval is not None
        )
        contract = child.parent_contract_snapshot if child is not None else {
            "team": version.snapshot.get("team", {}),
            "workspace_policy": version.snapshot.get("workspace_policy", {}),
        }
        raw_team = contract.get("team") if isinstance(contract, dict) else None
        from huddleroom.services.orchestration_service import OrchestrationService
        agent_ids = OrchestrationService._roadmap_team_agent_ids(raw_team)
        team = deepcopy(raw_team) if isinstance(raw_team, dict) else None
        if agent_ids is not None:
            team = dict(team or {})
            team["agent_ids"] = sorted(agent_ids)
        expected_agent_id = (task_action.request or {}).get("agent_id") if task_action else None
        binding = (contract.get("workspace_binding") if isinstance(contract, dict) else None) or (
            (version.snapshot.get("workspace_bindings", {}) or {}).get(item.item_key)
        )
        action_context = (task_action.request or {}).get("orchestrator_context", {}) if task_action else {}
        action_approval = (action_context.get("roadmap", {}).get("budget_approval")
                           if isinstance(action_context, dict) else None)
        return {
            "roadmap": {
                "staging_boundary": None if unstaged else snapshot.get("staging_boundary"),
                "mutates_shared_state": bool(snapshot.get("mutates_shared_state")),
                "no_publish_before_integration": True,
                "roadmap_version_id": str(version.id),
                "roadmap_item_key": item.item_key,
                **({"workspace_path": binding["path"]} if isinstance(binding, dict) and isinstance(binding.get("path"), str) else {}),
                **({"unstaged_authority_decision_id": str(approval.id)} if unstaged and approval is not None else {}),
                **({"budget_approval": deepcopy(action_approval)} if isinstance(action_approval, dict) else {}),
            },
            **({"team": team} if team is not None else {}),
            "workspace_policy": contract.get("workspace_policy", {}),
            "assigned_agent_id": expected_agent_id,
        }

    async def _claim_action(self, db: AsyncSession, task: Task | None) -> OrchestrationAction | None:
        """Resolve only the action explicitly bound to the task being claimed."""
        if task is None:
            return None
        from huddleroom.services.orchestration_service import OrchestrationService

        orchestration = OrchestrationService._json_object_or_empty((task.metadata_ or {}).get("orchestration"))
        action_id = OrchestrationService._optional_uuid(orchestration.get("action_id"), "action_id")
        action = await db.scalar(select(OrchestrationAction).where(
            OrchestrationAction.id == action_id,
            OrchestrationAction.status.in_(("reserved", "completed")),
        )) if action_id is not None else None
        if action is None:
            return None
        if action.action_type in {"retry_task", "reassign_task"}:
            requested_task_id = OrchestrationService._optional_uuid(
                OrchestrationService._json_object_or_empty(action.request).get("task_id"), "task_id",
            )
            if requested_task_id != task.id:
                raise HTTPException(status_code=409, detail="Recovery action task lineage is invalid")
        elif not (action.status == "completed" and action.target_type == "task" and action.target_id == task.id):
            if action.action_type == "request_plan":
                raise HTTPException(status_code=409, detail="Planning action task lineage is invalid")
            return None
        return action

    @staticmethod
    async def persist_claim_attention(db: AsyncSession, attention: SessionClaimAttention) -> None:
        from huddleroom.models.orchestration import OrchestrationRun
        from huddleroom.services.orchestration_service import OrchestrationService

        run = await db.get(OrchestrationRun, attention.run_id)
        if run is not None:
            OrchestrationService._upsert_active_blocker(run, attention.blocker)
            await db.flush()

    @classmethod
    async def commit_claim_attention(cls, db: AsyncSession, attention: SessionClaimAttention) -> None:
        """Discard a rejected claim, then commit only its idempotent blocker."""
        await db.rollback()
        await cls.persist_claim_attention(db, attention)
        await db.commit()

    async def _orchestration_budget_limits(self, db: AsyncSession, task: Task | None, agent: Agent) -> dict:
        from huddleroom.models.orchestration import (
            OrchestrationBudgetReservation, OrchestrationGoal, OrchestrationRun,
        )
        from huddleroom.services.orchestration_budget_service import BudgetMeasurementError, OrchestrationBudgetService
        from huddleroom.services.orchestration_service import OrchestrationService

        budget = OrchestrationBudgetService()
        claim_action = await self._claim_action(db, task)
        contract = OrchestrationService._json_object_or_empty(
            OrchestrationService._json_object_or_empty(task.metadata_ if task else {}).get("orchestration_contract")
        )
        continuous = OrchestrationService._json_object_or_empty(
            OrchestrationService._json_object_or_empty(contract.get("orchestrator_context")).get("continuous")
        )
        discovery_run_id = OrchestrationService._event_uuid(continuous.get("discovery_run_id"))
        if discovery_run_id is not None:
            source_run = await db.get(OrchestrationRun, discovery_run_id)
            reservation = await db.scalar(select(OrchestrationBudgetReservation).where(
                OrchestrationBudgetReservation.discovery_run_id == discovery_run_id,
                OrchestrationBudgetReservation.status == "active",
            ))
            scope = f"discovery_claim:{discovery_run_id}"
            if source_run is None or reservation is None:
                raise SessionClaimAttention(discovery_run_id, {
                    "kind": "budget_integrity", "scope": scope,
                    "reason": "Discovery reservation is not active.",
                }, "Discovery reservation is not active.")
            try:
                source_task_ids = budget._discovery_task_ids(source_run)
            except ValueError as exc:
                raise SessionClaimAttention(discovery_run_id, {
                    "kind": "budget_integrity", "scope": scope,
                    "reason": "Discovery task lineage is invalid.",
                }, "Discovery task lineage is invalid.") from exc
            source_state = OrchestrationService._json_object_or_empty(source_run.plan_state).get("discovery")
            source_task_id = OrchestrationService._event_uuid(
                OrchestrationService._json_object_or_empty(source_state).get("source_task_id")
            )
            if (
                source_task_id is None or task is None or task.id not in source_task_ids
                or reservation.parent_goal_id != source_run.goal_id
            ):
                raise SessionClaimAttention(discovery_run_id, {
                    "kind": "budget_integrity", "scope": scope,
                    "reason": "Discovery task lineage does not match its reservation.",
                }, "Discovery task lineage does not match its reservation.")
            try:
                remaining = await budget.discovery_remaining(db, source_run, reservation.allocation)
                limits = self._session_limits_from_remaining(agent, remaining)
            except BudgetMeasurementError as exc:
                raise SessionClaimAttention(discovery_run_id, {
                    "kind": "budget_measurement", "dimension": exc.dimension,
                    "session_id": str(exc.session_id), "scope": scope,
                }, "Execution budget bounds are required") from exc
            except ValueError as exc:
                raise SessionClaimAttention(discovery_run_id, {
                    "kind": "budget_integrity", "scope": scope, "reason": str(exc),
                }, str(exc)) from exc
            source_run.active_blockers = [
                blocker for blocker in (source_run.active_blockers or [])
                if not (isinstance(blocker, dict) and blocker.get("scope") == scope)
            ]
            return limits
        if claim_action is not None and claim_action.action_type == "request_plan":
            run = await db.get(OrchestrationRun, claim_action.run_id)
            goal = await db.get(OrchestrationGoal, run.goal_id) if run is not None else None
            parent = await budget.roadmap_parent(db, goal) or await budget.continuous_parent(db, goal) if goal else None
            if run is not None and goal is not None and parent is None:
                if run.status in ("completed", "failed", "cancelled") or goal.status in ("completed", "failed", "cancelled"):
                    raise SessionClaimAttention(run.id, {
                        "kind": "budget_integrity", "scope": f"claim:{goal.id}",
                        "reason": "Budgeted execution is no longer active.",
                    }, "Budgeted execution is no longer active.")
                caps = await budget.claim_caps(db, run, goal, goal)
                if caps:
                    try:
                        remaining = (await budget.remaining(db, goal))["remaining"]
                        remaining = self._clamp_to_action_allocation(claim_action, remaining)
                        limits = self._session_limits_from_remaining(agent, remaining)
                    except BudgetMeasurementError as exc:
                        raise SessionClaimAttention(run.id, {
                            "kind": "budget_measurement", "dimension": exc.dimension,
                            "session_id": str(exc.session_id), "scope": f"claim:{goal.id}",
                        }, "Execution budget bounds are required") from exc
                    except ValueError as exc:
                        raise SessionClaimAttention(run.id, {
                            "kind": "budget_integrity", "scope": f"claim:{goal.id}", "reason": str(exc),
                        }, str(exc)) from exc
                    return limits
        run = await budget.owning_roadmap_run(db, task) or await budget.owning_continuous_run(db, task)
        if run is None:
            return {}
        plan_state = OrchestrationService._json_object_or_empty(run.plan_state)
        if "discovery" in plan_state:
            raise SessionClaimAttention(run.id, {
                "kind": "budget_integrity", "scope": f"discovery_claim:{run.id}",
                "reason": "Discovery task is missing its discovery contract.",
            }, "Discovery task is missing its discovery contract.")
        goal = await db.get(OrchestrationGoal, run.goal_id)
        parent = await budget.roadmap_parent(db, goal) or await budget.continuous_parent(db, goal)
        if parent is None:
            return {}
        if run.status in ("completed", "failed", "cancelled") or goal.status in ("completed", "failed", "cancelled"):
            raise SessionClaimAttention(run.id, {
                "kind": "budget_integrity", "scope": f"claim:{goal.id}",
                "reason": "Budgeted execution is no longer active.",
            }, "Budgeted execution is no longer active.")
        if goal.id != parent.id:
            reservation = await db.scalar(select(OrchestrationBudgetReservation).where(
                OrchestrationBudgetReservation.parent_goal_id == parent.id,
                OrchestrationBudgetReservation.child_goal_id == goal.id,
            ))
            if reservation is None or reservation.status != "active":
                raise SessionClaimAttention(run.id, {
                    "kind": "budget_integrity", "scope": f"claim:{goal.id}",
                    "reason": "Child reservation is no longer active.",
                }, "Child reservation is no longer active.")
        caps = await budget.claim_caps(db, run, goal, parent)
        if not caps:
            return {}
        try:
            remaining = await (budget.remaining(db, parent) if goal.id == parent.id else budget.run_remaining(db, run, caps))
        except BudgetMeasurementError as exc:
            raise SessionClaimAttention(run.id, {
                "kind": "budget_measurement", "dimension": exc.dimension,
                "session_id": str(exc.session_id), "scope": f"claim:{goal.id}",
            }, "Execution budget bounds are required") from exc
        remaining = remaining["remaining"] if goal.id == parent.id else remaining
        remaining = self._clamp_to_action_allocation(claim_action, remaining)
        try:
            limits = self._session_limits_from_remaining(agent, remaining)
        except ValueError as exc:
            raise SessionClaimAttention(run.id, {
                "kind": "budget_integrity", "scope": f"claim:{goal.id}",
                "reason": str(exc),
            }, str(exc)) from exc
        scope = f"claim:{goal.id}"
        run.active_blockers = [
            blocker for blocker in (run.active_blockers or [])
            if not (isinstance(blocker, dict) and blocker.get("scope") == scope)
        ]
        return limits

    async def _cli_budget_authority(self, db: AsyncSession, task: Task | None, limits: dict,
                                    roadmap_context: dict, adapter_type: str = "cli") -> dict:
        """Authorize the one CLI gap from immutable Roadmap lineage only."""
        unsupported = sorted(key for key in ("max_tokens",) if key in limits)
        if not unsupported:
            return {}
        roadmap = roadmap_context.get("roadmap", {}) if isinstance(roadmap_context, dict) else {}
        version_id = roadmap.get("roadmap_version_id") if isinstance(roadmap, dict) else None
        item_key = roadmap.get("roadmap_item_key") if isinstance(roadmap, dict) else None
        if not isinstance(version_id, str) or not isinstance(item_key, str):
            raise HTTPException(status_code=409, detail="Roadmap CLI budget lineage is invalid")
        from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRoadmapVersion, OrchestrationRun
        from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
        from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService
        from huddleroom.services.orchestration_service import OrchestrationService

        run = await OrchestrationBudgetService().owning_roadmap_run(db, task)
        goal = await db.get(OrchestrationGoal, run.goal_id) if run is not None else None
        parent = await OrchestrationBudgetService().roadmap_parent(db, goal) if goal is not None else None
        if run is None or goal is None or parent is None:
            raise HTTPException(status_code=409, detail="Roadmap CLI budget lineage is invalid")
        try:
            version = await db.get(OrchestrationRoadmapVersion, uuid.UUID(version_id))
        except ValueError as exc:
            raise HTTPException(status_code=409, detail="Roadmap CLI budget lineage is invalid") from exc
        decision_run = await db.get(OrchestrationRun, version.run_id) if version is not None else None
        if decision_run is None or decision_run.goal_id != parent.id:
            raise HTTPException(status_code=409, detail="Roadmap CLI budget lineage is invalid")
        agent = await db.get(Agent, task.assigned_to) if task and task.assigned_to else None
        if agent is None:
            raise HTTPException(status_code=409, detail="Roadmap CLI budget lineage is invalid")
        dimensions = ",".join(unsupported)
        key = f"roadmap_budget_adapter:{version_id}:{item_key}:{agent.id}:{adapter_type}:{dimensions}"
        reference = roadmap.get("budget_approval") if isinstance(roadmap, dict) else None
        try:
            decision_id = uuid.UUID(str(reference.get("decision_id"))) if isinstance(reference, dict) else None
        except (TypeError, ValueError):
            decision_id = None
        if decision_id is None or reference.get("unsupported_dimensions") != unsupported:
            raise HTTPException(status_code=409, detail="Roadmap CLI budget authority is required")
        decision = await db.scalar(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.id == decision_id,
            OrchestrationAuthorityDecision.goal_id == parent.id,
            OrchestrationAuthorityDecision.run_id == decision_run.id,
            OrchestrationAuthorityDecision.decision_key == key,
            OrchestrationAuthorityDecision.status == "answered",
        ).order_by(OrchestrationAuthorityDecision.decided_at.desc()).limit(1))
        authorized_user = parent.manager_user_id or parent.created_by_user_id
        answers = list((await db.scalars(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.goal_id == parent.id,
            OrchestrationAuthorityDecision.run_id == decision_run.id,
            OrchestrationAuthorityDecision.decision_key == key,
            OrchestrationAuthorityDecision.status == "answered",
        ))).all())
        approved = (decision is not None and decision.authority == "human"
                    and decision.selected_option == "approve" and authorized_user is not None and decision.decided_by_user_id == authorized_user
                    and decision.decided_by_agent_id is None and len(answers) == 1)
        if not approved:
            raise HTTPException(status_code=409, detail="Roadmap CLI budget authority is required")
        return {"decision_id": str(decision.id), "unsupported_dimensions": unsupported}

    async def _validate_budget_adapter(
        self, db: AsyncSession, task: Task | None, limits: dict, roadmap_context: dict, adapter_type: str,
    ) -> dict:
        """Keep Roadmap's explicit CLI authority separate from Continuous work."""
        if not limits:
            return {}
        from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService

        run = await OrchestrationBudgetService().owning_continuous_run(db, task)
        if run is not None:
            unsupported = sorted(key for key in ("max_tokens",) if key in limits)
            if adapter_type == "cli" and unsupported:
                raise SessionClaimAttention(run.id, {
                    "kind": "budget_integrity", "scope": f"adapter-capability:{run.id}",
                    "adapter_type": adapter_type, "unsupported_dimensions": unsupported,
                    "reason": f"Continuous CLI cannot enforce reserved budget dimensions: {', '.join(unsupported)}.",
                }, "Continuous CLI cannot enforce reserved budget dimensions")
            if adapter_type not in {"api", "cli"}:
                raise SessionClaimAttention(run.id, {
                    "kind": "budget_integrity", "scope": f"adapter-capability:{run.id}",
                    "adapter_type": adapter_type, "reason": "Adapter cannot enforce reserved budget dimensions.",
                }, "Execution budget requires an adapter with enforced token limits")
            scope = f"adapter-capability:{run.id}"
            run.active_blockers = [
                blocker for blocker in (run.active_blockers or [])
                if not (isinstance(blocker, dict) and blocker.get("scope") == scope)
            ]
            await db.flush()
            return {}
        if adapter_type == "cli":
            return await self._cli_budget_authority(db, task, limits, roadmap_context, adapter_type)
        if adapter_type != "api":
            raise HTTPException(status_code=409, detail="Execution budget requires an adapter with enforced token limits")
        return {}

    def _schedule_dispatch_after_commit(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
        project_id: uuid.UUID,
        adapter_type: str,
        task_id: str,
    ) -> None:
        """Dispatch only after commit so background workers can read the session row."""
        sync_session = db.sync_session
        pending = sync_session.info.setdefault("pending_session_dispatches", [])
        pending.append((session_id, project_id, adapter_type, task_id))

        if sync_session.info.get("pending_session_dispatch_listener"):
            return
        sync_session.info["pending_session_dispatch_listener"] = True

        from sqlalchemy import event as _sa_event

        @_sa_event.listens_for(sync_session, "after_commit", once=True)
        def _dispatch_pending(committed_session) -> None:
            from huddleroom.config import settings

            committed_session.info.pop("pending_session_dispatch_listener", None)
            dispatches = committed_session.info.pop("pending_session_dispatches", [])
            for dispatch_session_id, dispatch_project_id, dispatch_adapter_type, dispatch_task_id in dispatches:
                if settings.is_sqlite:
                    from huddleroom.workers.task_runner import register_session
                    register_session(str(dispatch_session_id), dispatch_adapter_type, dispatch_project_id, task_id=dispatch_task_id)
                else:
                    from huddleroom.workers.session_tasks import run_api_session, run_cli_session
                    task = run_api_session if dispatch_adapter_type == "api" else run_cli_session
                    task.apply_async(args=[str(dispatch_session_id)], task_id=dispatch_task_id)

    def _resolve_adapter_type(self, agent: Agent, task: Task | None, override: str | None) -> str:
        if override:
            return override
        if task and task.adapter_type_override:
            return task.adapter_type_override
        return agent.adapter_type

    async def preflight_claim(self, db: AsyncSession, task: Task, agent: Agent,
                              adapter_override: str | None = None) -> None:
        """Read-only Roadmap claim validation for callers that mutate task state."""
        adapter_type = self._resolve_adapter_type(agent, task, adapter_override)
        context = await self._immutable_roadmap_context(db, task)
        roadmap = context.get("roadmap", {}) if isinstance(context, dict) else {}
        if roadmap:
            from huddleroom.services.orchestration_service import OrchestrationService
            allowed = OrchestrationService._roadmap_team_agent_ids(context.get("team"))
            if (
                task.assigned_to != agent.id
                or context.get("assigned_agent_id") not in (None, str(agent.id))
                or (allowed is not None and str(agent.id) not in allowed)
            ):
                raise HTTPException(status_code=409, detail="Roadmap claim must use its immutable assigned team agent")
        if roadmap.get("mutates_shared_state") and roadmap.get("staging_boundary"):
            if adapter_type != "cli":
                raise HTTPException(status_code=409, detail="Roadmap staged mutable work requires a CLI adapter")
            path = roadmap.get("workspace_path")
            if not isinstance(path, str):
                raise HTTPException(status_code=409, detail="Roadmap staged mutable work lacks frozen workspace")
            await ProjectService().require_frozen_roadmap_workspace(
                db, task.project_id, roadmap["staging_boundary"], path
            )
        limits = await self._orchestration_budget_limits(db, task, agent)
        await self._validate_budget_adapter(db, task, limits, context, adapter_type)

    async def create(self, db: AsyncSession, data: SessionCreate) -> Session:
        project_service = ProjectService()
        await project_service.lock_workspace_boundary(db, data.project_id)
        await project_service.require_runnable_project(db, data.project_id)

        result = await db.execute(select(Agent).where(Agent.id == data.agent_id))
        agent = result.scalar_one_or_none()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        if data.max_tokens is not None:
            self._positive_limit(data.max_tokens, "max_tokens")
        if data.timeout is not None:
            self._positive_limit(data.timeout, "timeout")
        self._positive_limit(agent.config.get("max_tokens", 4096), "Agent max_tokens")
        self._positive_limit(agent.config.get("session_timeout_seconds", 3600), "Agent session_timeout_seconds")

        task = None
        if data.task_id:
            result2 = await db.execute(select(Task).where(Task.id == data.task_id))
            task = result2.scalar_one_or_none()
            if task is None or task.project_id != data.project_id:
                raise HTTPException(status_code=404, detail="Task not found")

        graph_run = None
        if data.graph_run_id:
            result3 = await db.execute(
                select(GraphRun).where(
                    GraphRun.id == data.graph_run_id,
                    GraphRun.project_id == data.project_id,
                )
            )
            graph_run = result3.scalar_one_or_none()
            if graph_run is None:
                raise HTTPException(status_code=404, detail="Graph run not found")
            if (
                data.task_id is not None
                and graph_run.linked_task_id != data.task_id
                and (task is None or task.graph_run_id != graph_run.id)
            ):
                raise HTTPException(status_code=409, detail="Graph run is linked to a different task")

        adapter_type = self._resolve_adapter_type(agent, task, data.adapter_type_override)
        roadmap_context = await self._immutable_roadmap_context(db, task)
        roadmap_policy = roadmap_context.get("roadmap", {}) if isinstance(roadmap_context, dict) else {}
        execution_workspace = None
        if isinstance(roadmap_policy, dict) and roadmap_policy:
            team = roadmap_context.get("team") if isinstance(roadmap_context, dict) else None
            from huddleroom.services.orchestration_service import OrchestrationService
            allowed = OrchestrationService._roadmap_team_agent_ids(team)
            if (
                task is None or task.assigned_to != agent.id or (allowed is not None and str(agent.id) not in allowed)
                or roadmap_context.get("assigned_agent_id") not in (None, str(agent.id))
            ):
                raise HTTPException(status_code=409, detail="Roadmap claim must use its immutable assigned team agent")
            supplied_context = data.context_override.get("orchestrator_context") if isinstance(data.context_override, dict) else None
            if supplied_context is not None and supplied_context != roadmap_context:
                raise HTTPException(status_code=409, detail="Roadmap execution context cannot be overridden")
        if isinstance(roadmap_policy, dict) and roadmap_policy.get("mutates_shared_state"):
            boundary = roadmap_policy.get("staging_boundary")
            if boundary is not None:
                if adapter_type != "cli":
                    raise HTTPException(status_code=409, detail="Roadmap staged mutable work requires a CLI adapter")
                workspace_path = roadmap_policy.get("workspace_path")
                if not isinstance(workspace_path, str):
                    raise HTTPException(status_code=409, detail="Roadmap staged mutable work lacks frozen workspace")
                execution_workspace = await project_service.require_frozen_roadmap_workspace(
                    db, data.project_id, boundary, workspace_path
                )
            else:
                from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
                try:
                    approval_id = uuid.UUID(str(roadmap_policy.get("unstaged_authority_decision_id")))
                except (TypeError, ValueError) as exc:
                    raise HTTPException(status_code=409, detail="Roadmap unstaged mutation lacks human approval") from exc
                approval = await db.get(OrchestrationAuthorityDecision, approval_id)
                if approval is None or approval.status != "answered" or approval.selected_option != "approve":
                    raise HTTPException(status_code=409, detail="Roadmap unstaged mutation lacks human approval")
        if task is not None:
            active = await db.scalar(select(Session.id).where(
                Session.task_id == task.id, Session.status.in_(("pending", "running"))
            ).limit(1))
            if active is not None:
                raise HTTPException(status_code=409, detail="Task already has an active session")
        budget_limits = await self._orchestration_budget_limits(db, task, agent)
        budget_approval = {}
        budget_approval = await self._validate_budget_adapter(
            db, task, budget_limits, roadmap_context, adapter_type,
        )
        if budget_approval:
            roadmap_context = deepcopy(roadmap_context)
            roadmap_context["roadmap"] = {**roadmap_context["roadmap"], "budget_approval": budget_approval}

        session = Session(
            agent_id=data.agent_id,
            task_id=data.task_id,
            project_id=data.project_id,
            graph_run_id=data.graph_run_id,
            adapter_type=adapter_type,
            status="pending",
            input_context={
                **(data.context_override or {}),
                **({"orchestrator_context": roadmap_context} if roadmap_context else {}),
            },
            metadata_={},
            origin=data.origin,
        )
        db.add(session)

        run_config = dict(budget_limits)
        if budget_limits:
            run_config["_roadmap_budget_enforced"] = True
        if data.model_override:
            run_config["model_override"] = data.model_override
        if data.timeout is not None:
            run_config["timeout"] = min(data.timeout, run_config.get("timeout", data.timeout))
        if data.max_tokens is not None:
            run_config["max_tokens"] = min(data.max_tokens, run_config.get("max_tokens", data.max_tokens))
        if run_config:
            session.metadata_ = {"_run_config": run_config}
        if task is not None:
            from huddleroom.services.orchestration_service import OrchestrationService

            orchestration = OrchestrationService._json_object_or_empty((task.metadata_ or {}).get("orchestration"))
            action_id = OrchestrationService._optional_uuid(orchestration.get("action_id"), "action_id")
            if action_id is not None:
                session.metadata_ = {
                    **(session.metadata_ or {}),
                    "orchestration": {"action_id": str(action_id)},
                }
        if budget_approval:
            session.metadata_ = {
                **(session.metadata_ or {}),
                "_roadmap_cli_budget_approval": deepcopy(budget_approval),
                "_roadmap_cli_token_grants": [run_config["max_tokens"]],
                "token_usage_complete": False,
            }
        if execution_workspace is not None:
            session.metadata_ = {
                **(session.metadata_ or {}),
                "_roadmap_workspace": str(execution_workspace),
            }

        await db.flush()

        task_id = str(uuid.uuid4())
        session.runner_task_id = task_id
        if task is not None:
            from huddleroom.services.orchestration_service import OrchestrationService

            action = await self._claim_action(db, task)
            if action is not None and action.budget_ledger:
                run = await db.get(OrchestrationRun, action.run_id)
                goal = await db.get(OrchestrationGoal, run.goal_id) if run is not None else None
                if run is not None and goal is not None:
                    from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService

                    action.budget_ledger = {
                        **action.budget_ledger,
                        "enforceability": "enforceable" if adapter_type in {"api", "cli"} else "non_enforceable",
                    }
                    await OrchestrationBudgetService().commit_action_budget(db, goal, run, action)
        self._schedule_dispatch_after_commit(db, session.id, session.project_id, adapter_type, task_id)
        await db.flush()
        await emit_event(db, session.project_id, "session.created", {
            "session_id": str(session.id),
            "task_id": str(session.task_id) if session.task_id else None,
            "graph_run_id": str(session.graph_run_id) if session.graph_run_id else None,
            "agent_id": str(session.agent_id),
            "adapter_type": session.adapter_type,
            "origin": session.origin,
            "project_id": str(session.project_id),
        })
        return session

    async def get(self, db: AsyncSession, session_id: uuid.UUID) -> Session | None:
        result = await db.execute(select(Session).where(Session.id == session_id))
        return result.scalar_one_or_none()

    async def get_or_404(self, db: AsyncSession, session_id: uuid.UUID) -> Session:
        session = await self.get(db, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session

    async def list(
        self,
        db: AsyncSession,
        agent_id: uuid.UUID | None = None,
        task_id: uuid.UUID | None = None,
        project_id: uuid.UUID | None = None,
        status: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> tuple[list[Session], str | None]:
        from datetime import datetime
        from sqlalchemy import or_, and_
        query = select(Session).order_by(Session.created_at.desc(), Session.id.desc()).limit(limit + 1)
        if agent_id:
            query = query.where(Session.agent_id == agent_id)
        if task_id:
            query = query.where(Session.task_id == task_id)
        if project_id:
            query = query.where(Session.project_id == project_id)
        if status:
            query = query.where(Session.status == status)
        if cursor:
            try:
                cursor_dt_str, cursor_id_str = cursor.split("__", 1)
                cursor_dt = datetime.fromisoformat(cursor_dt_str)
                cursor_id = uuid.UUID(cursor_id_str)
            except (ValueError, AttributeError) as exc:
                raise HTTPException(status_code=400, detail="Invalid cursor") from exc
            query = query.where(
                or_(
                    Session.created_at < cursor_dt,
                    and_(Session.created_at == cursor_dt, Session.id < cursor_id),
                )
            )
        result = await db.execute(query)
        items = list(result.scalars().all())
        next_cursor = None
        if len(items) > limit:
            items = items[:limit]
            last = items[-1]
            next_cursor = f"{last.created_at.isoformat()}__{last.id}"
        return items, next_cursor

    async def cancel(self, db: AsyncSession, session_id: uuid.UUID) -> Session:
        session = await self.get_or_404(db, session_id)
        if session.status in ("completed", "failed", "cancelled"):
            raise HTTPException(status_code=409, detail=f"Session already {session.status}")

        if session.runner_task_id:
            try:
                from huddleroom.workers.task_runner import cancel_task
                await cancel_task(session.runner_task_id)
            except Exception:
                pass

        session.status = "cancelled"
        await db.flush()
        await emit_event(db, session.project_id, "session.cancelled", {
            "session_id": str(session.id),
            "task_id": str(session.task_id) if session.task_id else None,
            "project_id": str(session.project_id),
        })
        return session

    async def resume(self, db: AsyncSession, session_id: uuid.UUID) -> Session:
        session = await self.get_or_404(db, session_id)
        await ProjectService().lock_workspace_boundary(db, session.project_id)
        session = await self.get_or_404(db, session_id)
        task = await db.get(Task, session.task_id) if session.task_id else None
        agent = await db.get(Agent, session.agent_id)
        budget_limits = await self._orchestration_budget_limits(db, task, agent) if agent else {}
        roadmap_context = await self._immutable_roadmap_context(db, task)
        await self._validate_budget_adapter(
            db, task, budget_limits, roadmap_context, session.adapter_type,
        )
        prior_usage = None
        if budget_limits:
            old_config = (session.metadata_ or {}).get("_run_config", {})
            metadata = session.metadata_ or {}
            prior_usage = {"max_turns": str(metadata.get("_roadmap_turn_count", 1))}
            if "max_tokens" in old_config:
                if isinstance(metadata.get("_roadmap_cli_budget_approval"), dict):
                    from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService
                    prior_usage["max_tokens"] = str(OrchestrationBudgetService._session_spend(
                        session, "max_tokens", allow_incomplete=True,
                    ))
                else:
                    prior_usage["max_tokens"] = str(
                        Decimal(str(metadata.get("token_count_in", 0)))
                        + Decimal(str(metadata.get("token_count_out", 0)))
                    )
            if "timeout" in old_config:
                elapsed = metadata.get("_roadmap_elapsed_seconds")
                if elapsed is None:
                    elapsed = max(0, (session.ended_at - session.started_at).total_seconds())
                prior_usage["max_hours"] = str(Decimal(str(elapsed)) / Decimal("3600"))

        # COLLISION GUARD: check for other active sessions on the same task
        if session.task_id:
            result = await db.execute(
                select(Session).where(
                    Session.task_id == session.task_id,
                    Session.status.in_(["pending", "running"]),
                )
            )
            if result.scalar_one_or_none() is not None:
                raise HTTPException(
                    status_code=409,
                    detail="Task already has an active session"
                )

        # Atomic claim: flip status failed→pending and resumable True→False in one UPDATE
        # If 0 rows affected, the session was already resumed or not resumable.
        result = await db.execute(
            update(Session)
            .where(
                and_(
                    Session.id == session_id,
                    Session.status == "failed",
                    Session.resumable == True,
                )
            )
            .values(
                status="pending",
                resumable=False,
                started_at=None,
                ended_at=None,
                error=None,
                output=None,
            )
        )
        if result.rowcount == 0:
            raise HTTPException(
                status_code=409,
                detail="Session must be failed and resumable to resume"
            )

        # Refresh session to get updated values from the atomic UPDATE
        session = await self.get_or_404(db, session_id)
        if budget_limits:
            config = dict((session.metadata_ or {}).get("_run_config", {}))
            config["_roadmap_budget_enforced"] = True
            config["_roadmap_prior_usage"] = prior_usage
            for key, limit in budget_limits.items():
                if key == "_roadmap_budget_enforced":
                    continue
                config[key] = min(config.get(key, limit), limit)
            metadata = session.metadata_ or {}
            if isinstance(metadata.get("_roadmap_cli_budget_approval"), dict) and "max_tokens" in config:
                metadata = {
                    **metadata,
                    # Collapse all completed attempts into their conservative
                    # baseline, then append only this attempt's capped grant.
                    "_roadmap_cli_token_grants": [prior_usage["max_tokens"], config["max_tokens"]],
                    "token_usage_complete": False,
                }
            session.metadata_ = {**metadata, "_run_config": config}

        # Assign fresh runner_task_id and dispatch
        task_id = str(uuid.uuid4())
        session.runner_task_id = task_id
        self._schedule_dispatch_after_commit(db, session.id, session.project_id, session.adapter_type, task_id)
        await db.flush()

        await emit_event(db, session.project_id, "session.resumed", {
            "session_id": str(session.id),
            "task_id": str(session.task_id) if session.task_id else None,
            "project_id": str(session.project_id),
        })
        return session

    async def get_output(self, db: AsyncSession, session_id: uuid.UUID) -> SessionOutputResponse:
        session = await self.get_or_404(db, session_id)
        return SessionOutputResponse(
            session_id=session.id,
            status=session.status,
            adapter_type=session.adapter_type,
            output=session.output,
            error=session.error,
            metadata=session.metadata_,
        )

    async def update_status(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
        status: str,
        output: str | None = None,
        error: str | None = None,
        metadata: dict | None = None,
    ) -> Session:
        from datetime import datetime, timezone
        session = await self.get_or_404(db, session_id)
        session.status = status
        if output is not None:
            session.output = output
        if error is not None:
            session.error = error
        if metadata is not None:
            session.metadata_ = metadata
        now = datetime.now(timezone.utc)
        if status == "running" and session.started_at is None:
            session.started_at = now
        if status in ("completed", "failed", "cancelled"):
            session.ended_at = now
        await db.flush()
        return session

    async def recover_orphaned_sessions(
        self,
        db: AsyncSession,
        timeout_seconds: int = 3600,
        redispatch_pending: bool = True,
    ) -> int:
        """On process restart: fail stuck running sessions, re-dispatch pending sessions that never started."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=timeout_seconds)
        now = datetime.now(timezone.utc)

        stale_running_result = await db.execute(
            select(Session.id).where(and_(Session.status == "running", Session.started_at < cutoff))
        )
        stale_running_ids = list(stale_running_result.scalars().all())

        stale_pending_result = await db.execute(
            select(Session.id).where(and_(Session.status == "pending", Session.created_at < cutoff))
        )
        stale_pending_ids = list(stale_pending_result.scalars().all())

        # Orchestrated work is reconciled by ownership and backend facts, never age.
        from huddleroom.services.orchestration_recovery_service import OrchestrationRecoveryService
        recovery = OrchestrationRecoveryService()
        stale_running_ids = [sid for sid in stale_running_ids if await recovery.resolve_owned_session(db, sid) is None]
        stale_pending_ids = [sid for sid in stale_pending_ids if await recovery.resolve_owned_session(db, sid) is None]

        # Fail running sessions - server restart killed them mid-run, can't resume
        if stale_running_ids:
            await db.execute(
                update(Session)
                .where(Session.id.in_(stale_running_ids))
                .values(
                    status="failed",
                    error=STALE_RUNNING_SESSION_ERROR,
                    ended_at=now,
                )
            )

        # Fail old pending sessions - genuinely stuck (dispatch succeeded but task never ran and timed out)
        if stale_pending_ids:
            await db.execute(
                update(Session)
                .where(Session.id.in_(stale_pending_ids))
                .values(
                    status="failed",
                    error=STALE_PENDING_SESSION_ERROR,
                    ended_at=now,
                )
            )
        failed_count = len(stale_running_ids) + len(stale_pending_ids)
        await db.flush()

        # Update tasks linked to sessions we just failed
        newly_failed_ids = stale_running_ids + stale_pending_ids
        newly_failed_list = []
        if newly_failed_ids:
            res = await db.execute(
                select(Session).where(
                    Session.id.in_(newly_failed_ids),
                    Session.task_id.isnot(None),
                )
            )
            newly_failed_list = list(res.scalars().all())
        for session in newly_failed_list:
            await sync_task_from_session(db, session)
        if newly_failed_list:
            await db.flush()

        if redispatch_pending:
            # On startup, re-dispatch pending sessions that never started
            # (started_at=None means the in-process task likely died on restart).
            redispatch_result = await db.execute(
                select(Session).where(
                    and_(Session.status == "pending", Session.started_at.is_(None))
                )
            )
            pending_sessions = list(redispatch_result.scalars().all())
            pending_sessions = [session for session in pending_sessions if await recovery.resolve_owned_session(db, session.id) is None]

            from huddleroom.workers.task_runner import dispatch_session
            for session in pending_sessions:
                try:
                    task_id = await dispatch_session(str(session.id), session.adapter_type, session.project_id)
                    session.runner_task_id = task_id
                except Exception as e:
                    session.status = "failed"
                    session.error = f"redispatch_failed: {e}"
            if pending_sessions:
                await db.flush()

        return failed_count
