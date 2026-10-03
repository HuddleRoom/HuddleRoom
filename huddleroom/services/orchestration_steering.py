"""Durable, explicit operator steering for orchestration goals."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.config import settings
from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration import (
    OrchestrationAction, OrchestrationDecision, OrchestrationGoal,
    OrchestrationRoadmapItem, OrchestrationRoadmapVersion, OrchestrationRun,
)
from huddleroom.models.orchestration_steering import (
    OrchestrationSteeringProposal, OrchestrationSteeringRequest, OrchestrationSteeringResultLink,
    OrchestrationSteeringState, OrchestrationSteeringTransition, steering_request_id,
    steering_result_link_id, steering_state_id, steering_transition_id,
)
from huddleroom.models.orchestration_conversation import ConversationMessage, ConversationResponse
from huddleroom.models.task import Task
from huddleroom.services.event_bus import emit_event_once
from huddleroom.services.orchestration_service import OrchestrationService

SteeringScope = Literal["item", "run", "goal"]
SteeringLifetime = Literal["selected_item", "remaining_current_run", "future_runs"]
SteeringTargetType = Literal["goal", "plan_item", "task"]
_PAIR = {("item", "selected_item"), ("run", "remaining_current_run"), ("goal", "future_runs")}


@dataclass(frozen=True)
class SteeringDraft:
    directive: str
    target_type: SteeringTargetType
    target_id: str
    scope: SteeringScope = "run"
    lifetime: SteeringLifetime = "remaining_current_run"
    impact_summary: str = ""
    source_proposal_id: uuid.UUID | None = None
    supersedes_request_id: uuid.UUID | None = None


@dataclass(frozen=True)
class SteeringVersions:
    inbox_version: int
    direction_version: int
    contract_version: str
    plan_version: str


@dataclass(frozen=True)
class SteeringProcessingResult:
    processed_request_ids: Sequence[uuid.UUID]
    applied_request_ids: Sequence[uuid.UUID]
    inbox_version: int
    direction_version: int


@dataclass(frozen=True)
class SteeringLedger:
    enabled: bool
    eligibility: Literal["active", "paused", "unstarted", "terminal", "forbidden"]
    eligibility_reason: str | None
    requests: Sequence[OrchestrationSteeringRequest]
    proposals: Sequence[OrchestrationSteeringProposal]
    inbox_version: int
    direction_version: int


class SteeringDomainError(Exception):
    def __init__(self, code: str, status_code: int, message: str):
        super().__init__(code)
        self.code, self.status_code, self.message = code, status_code, message


class SteeringVersionsChanged(Exception):
    pass


def normalize_draft(draft: SteeringDraft) -> SteeringDraft:
    directive, impact, target_id = draft.directive.strip(), draft.impact_summary.strip(), draft.target_id.strip()
    if not 1 <= len(directive) <= 4_000:
        raise SteeringDomainError("steering_invalid_directive", 422, "Directive must be 1 to 4000 characters")
    if not 1 <= len(impact) <= 1_000 or not 1 <= len(target_id) <= 255:
        raise SteeringDomainError("steering_invalid_scope", 422, "Steering scope is invalid")
    if (draft.scope, draft.lifetime) not in _PAIR:
        raise SteeringDomainError("steering_invalid_scope", 422, "Steering scope is invalid")
    return replace(draft, directive=directive, impact_summary=impact, target_id=target_id)


def steering_versions_from_snapshot(snapshot: Mapping[str, object]) -> SteeringVersions | None:
    steering = snapshot.get("steering")
    versions = steering.get("versions") if isinstance(steering, Mapping) else None
    if not isinstance(versions, Mapping):
        return None
    keys = ("inbox_version", "direction_version", "contract_version", "plan_version")
    if set(versions) != set(keys):
        return None
    if type(versions["inbox_version"]) is not int or type(versions["direction_version"]) is not int:
        return None
    if not isinstance(versions["contract_version"], str) or not isinstance(versions["plan_version"], str):
        return None
    return SteeringVersions(**versions)  # type: ignore[arg-type]


def active_direction_ids(snapshot: Mapping[str, object]) -> Sequence[uuid.UUID]:
    steering = snapshot.get("steering")
    directions = steering.get("active_directions") if isinstance(steering, Mapping) else None
    if not isinstance(directions, list):
        return ()
    ids = []
    for direction in directions:
        if isinstance(direction, Mapping):
            try:
                ids.append(uuid.UUID(str(direction["request_id"])))
            except (KeyError, ValueError, TypeError):
                return ()
        else:
            return ()
    return tuple(ids)


class OrchestrationSteeringService:
    """The one small, append-only steering ledger implementation."""

    def __init__(self, orchestration: OrchestrationService | None = None):
        self._orchestration = orchestration or OrchestrationService()

    @staticmethod
    def _contract_version(goal: OrchestrationGoal) -> str:
        values = {key: getattr(goal, key) for key in (
            "objective", "original_request", "success_criteria", "constraints", "budget",
            "authority_model", "manager_agent_id", "manager_user_id",
        )}
        canonical = json.dumps(values, default=str, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return "contract:" + hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def _plan_version(run: OrchestrationRun) -> str:
        state = run.plan_state if isinstance(run.plan_state, dict) else {}
        fingerprint = state.get("accepted_plan_fingerprint")
        if isinstance(fingerprint, str):
            return f"roadmap:{fingerprint}"
        snapshot = state.get("accepted_plan_snapshot")
        if isinstance(snapshot, dict) and isinstance(snapshot.get("fingerprint"), str):
            return f"plan:{snapshot['fingerprint']}"
        return "none"

    @staticmethod
    def _eligibility(goal: OrchestrationGoal, run: OrchestrationRun, actor_id: uuid.UUID | None = None) -> tuple[str, str | None]:
        owner = goal.manager_user_id or goal.created_by_user_id
        if actor_id is not None and owner != actor_id:
            return "forbidden", "steering_forbidden"
        if (goal.continuous_state or {}).get("stopped_at") or goal.status in {"completed", "cancelled"}:
            return "terminal", "steering_ineligible"
        if run.phase != "authorized" or run.phase == "waiting_activation" or run.status in {"completed", "cancelled"}:
            return "unstarted" if run.phase != "authorized" else "terminal", "steering_ineligible"
        if goal.status == "paused" and run.status == "paused":
            return "paused", None
        if goal.status in {"active", "blocked"} and run.status in {"running", "blocked"}:
            return "active", None
        return "terminal", "steering_ineligible"

    async def _goal_run(self, db: AsyncSession, project_id: uuid.UUID, goal_id: uuid.UUID) -> tuple[OrchestrationGoal, OrchestrationRun]:
        goal = await self._orchestration.get_goal(db, project_id, goal_id)
        if goal is None:
            raise SteeringDomainError("steering_invalid_target", 404, "Orchestration goal not found")
        run = await db.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == goal_id).order_by(OrchestrationRun.created_at.desc()))
        if run is None:
            raise SteeringDomainError("steering_ineligible", 409, "Steering requires an orchestration run")
        return goal, run

    async def _authorized_goal(self, db: AsyncSession, project_id: uuid.UUID, goal_id: uuid.UUID, actor_id: uuid.UUID) -> OrchestrationGoal:
        """Authorize before reading a run, whose state is lifecycle information."""
        goal = await self._orchestration.get_goal(db, project_id, goal_id)
        if goal is None:
            raise SteeringDomainError("steering_invalid_target", 404, "Orchestration goal not found")
        if (goal.manager_user_id or goal.created_by_user_id) != actor_id:
            raise SteeringDomainError("steering_forbidden", 403, "Steering is forbidden")
        return goal

    async def _run_for_goal(self, db: AsyncSession, goal_id: uuid.UUID) -> OrchestrationRun:
        run = await db.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == goal_id).order_by(OrchestrationRun.created_at.desc()))
        if run is None:
            raise SteeringDomainError("steering_ineligible", 409, "Steering requires an orchestration run")
        return run

    async def _state(
        self, db: AsyncSession, goal_id: uuid.UUID, create: bool, *, fresh: bool = False,
    ) -> OrchestrationSteeringState | None:
        state = await db.get(
            OrchestrationSteeringState, steering_state_id(goal_id), populate_existing=fresh,
        )
        if state is None and create:
            state = OrchestrationSteeringState(id=steering_state_id(goal_id), goal_id=goal_id)
            db.add(state)
            await db.flush()
        return state

    async def _transition(self, db: AsyncSession, request: OrchestrationSteeringRequest, status: str, reason: str, actor: str) -> None:
        sequence = 1 + int(await db.scalar(select(func.coalesce(func.max(OrchestrationSteeringTransition.sequence), 0)).where(OrchestrationSteeringTransition.request_id == request.id)) or 0)
        db.add(OrchestrationSteeringTransition(id=steering_transition_id(request.id, sequence), request_id=request.id, sequence=sequence, from_status=None if sequence == 1 else request.status, to_status=status, reason_code=reason, actor=actor))
        request.status, request.reason_code = status, reason
        if status == "being_considered": request.considered_at = _utcnow()
        if status not in {"pending", "being_considered"}: request.finished_at = _utcnow()
        await db.flush()

    @staticmethod
    def _task_available(task: Task | None, goal: OrchestrationGoal, run: OrchestrationRun) -> bool:
        metadata = task.metadata_ if task and isinstance(task.metadata_, dict) else {}
        orchestration = metadata.get("orchestration") if isinstance(metadata.get("orchestration"), dict) else {}
        return bool(
            task is not None and task.status in {"backlog", "ready"}
            and task.project_id == goal.project_id
            and orchestration.get("goal_id") == str(goal.id)
            and orchestration.get("run_id") == str(run.id)
        )

    async def _target_available(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun,
        target_type: SteeringTargetType, target_id: str, *, fresh: bool = False,
    ) -> bool:
        """Whether an item is still this run's unstarted work."""
        if target_type == "task":
            try:
                task = await db.get(Task, uuid.UUID(target_id), populate_existing=fresh)
            except ValueError:
                return False
            return self._task_available(task, goal, run)
        if target_type != "plan_item":
            return target_type == "goal" and target_id == str(goal.id)
        if goal.goal_type == "roadmap":
            version = await db.scalar(select(OrchestrationRoadmapVersion).where(
                OrchestrationRoadmapVersion.goal_id == goal.id,
                OrchestrationRoadmapVersion.run_id == run.id,
            ).order_by(OrchestrationRoadmapVersion.version.desc()).execution_options(populate_existing=fresh))
            items = version.snapshot.get("items", []) if version and isinstance(version.snapshot, dict) else []
            if not any(isinstance(item, dict) and item.get("item_key") == target_id for item in items):
                return False
            lineage = await db.scalar(select(OrchestrationRoadmapItem).where(
                OrchestrationRoadmapItem.goal_id == goal.id,
                OrchestrationRoadmapItem.item_key == target_id,
            ).execution_options(populate_existing=fresh))
            if lineage is None:
                return True
            if lineage.completed_at is not None or lineage.child_goal_id is not None:
                return False
            return self._task_available(
                await db.get(Task, lineage.task_id, populate_existing=fresh), goal, run,
            )
        state = run.plan_state if isinstance(run.plan_state, dict) else {}
        expanded = state.get("expanded_items", [])
        entry = next((item for item in expanded if isinstance(item, dict) and item.get("plan_item_id") == target_id), None)
        if entry is not None:
            try:
                task = await db.get(Task, uuid.UUID(str(entry.get("task_id"))), populate_existing=fresh)
            except (ValueError, TypeError):
                return False
            return self._task_available(task, goal, run)
        snapshot = state.get("accepted_plan_snapshot", {})
        items = snapshot.get("items", []) if isinstance(snapshot, dict) else []
        item = next((item for item in items if isinstance(item, dict) and item.get("id") == target_id), None)
        return item is not None and item.get("status", "unstarted") in {"unstarted", "pending", "ready"}

    async def _validate_target(self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun, draft: SteeringDraft) -> None:
        if draft.scope == "item" and draft.target_type not in {"plan_item", "task"}:
            raise SteeringDomainError("steering_invalid_target", 422, "Steering target is invalid")
        if draft.scope in {"run", "goal"} and draft.target_type != "goal":
            raise SteeringDomainError("steering_invalid_target", 422, "Steering target is invalid")
        if draft.target_type == "goal":
            if draft.target_id != str(goal.id): raise SteeringDomainError("steering_invalid_target", 422, "Steering target is invalid")
            return
        if draft.target_type in {"plan_item", "task"} and await self._target_available(
            db, goal, run, draft.target_type, draft.target_id
        ):
            return
        raise SteeringDomainError("steering_invalid_target", 422, "Steering target is invalid")

    async def _item_available(
        self, db: AsyncSession, request: OrchestrationSteeringRequest, goal: OrchestrationGoal,
        run: OrchestrationRun, *, fresh: bool = False,
    ) -> bool:
        if request.scope != "item":
            return True
        return await self._target_available(
            db, goal, run, request.target_type, request.target_id, fresh=fresh,
        )

    async def submit(self, db: AsyncSession, project_id: uuid.UUID, goal_id: uuid.UUID, actor_id: uuid.UUID, client_request_id: uuid.UUID, draft: SteeringDraft) -> OrchestrationSteeringRequest:
        if not settings.orchestration_conversation_steering_enabled: raise SteeringDomainError("steering_disabled", 404, "Steering is disabled")
        async with self._orchestration._lock_goal_for_baseline_transition(db, goal_id):
            goal = await self._authorized_goal(db, project_id, goal_id, actor_id)
            draft = normalize_draft(draft)
            request_id = steering_request_id(goal_id, actor_id, client_request_id)
            existing = await db.get(OrchestrationSteeringRequest, request_id)
            if existing:
                same = (existing.directive, existing.target_type, existing.target_id, existing.scope, existing.lifetime, existing.impact_summary, existing.source_proposal_id, existing.supersedes_request_id) == (draft.directive, draft.target_type, draft.target_id, draft.scope, draft.lifetime, draft.impact_summary, draft.source_proposal_id, draft.supersedes_request_id)
                if not same: raise SteeringDomainError("steering_idempotency_conflict", 409, "Request ID has different content")
                return existing
            run = await self._run_for_goal(db, goal_id)
            eligibility, reason = self._eligibility(goal, run, actor_id)
            if eligibility not in {"active", "paused"}: raise SteeringDomainError("steering_ineligible", 409, "Steering is not eligible")
            await self._validate_target(db, goal, run, draft)
            proposal = None
            if draft.source_proposal_id:
                proposal = await db.get(OrchestrationSteeringProposal, draft.source_proposal_id)
                if proposal is None or proposal.goal_id != goal_id or proposal.actor_id != actor_id or proposal.status != "proposed":
                    raise SteeringDomainError("proposal_not_promotable", 409, "Proposal is not promotable")
            state = await self._state(db, goal_id, True)
            sequence = 1 + int(await db.scalar(select(func.coalesce(func.max(OrchestrationSteeringRequest.sequence), 0)).where(OrchestrationSteeringRequest.goal_id == goal_id)) or 0)
            request = OrchestrationSteeringRequest(id=request_id, goal_id=goal_id, actor_id=actor_id, client_request_id=client_request_id, sequence=sequence, submitted_run_id=run.id, directive=draft.directive, target_type=draft.target_type, target_id=draft.target_id, scope=draft.scope, lifetime=draft.lifetime, impact_summary=draft.impact_summary, source_proposal_id=draft.source_proposal_id, supersedes_request_id=draft.supersedes_request_id, status="pending", reason_code="submitted", contract_version=self._contract_version(goal), plan_version=self._plan_version(run))
            db.add(request); await db.flush(); await self._transition(db, request, "pending", "submitted", "operator")
            state.inbox_version += 1
            if proposal is not None:
                proposal.status, proposal.promoted_request_id = "promoted", request.id
            await emit_event_once(db, project_id, "orchestration.steering_changed", {"goal_id": str(goal_id), "run_id": str(run.id), "request_id": str(request.id)}, source="operator", dedup_key=f"orchestration.steering_changed:request:{request.id}:inbox:{state.inbox_version}")
            return request

    async def withdraw(self, db: AsyncSession, project_id: uuid.UUID, goal_id: uuid.UUID, actor_id: uuid.UUID, request_id: uuid.UUID) -> OrchestrationSteeringRequest:
        if not settings.orchestration_conversation_steering_enabled: raise SteeringDomainError("steering_disabled", 404, "Steering is disabled")
        async with self._orchestration._lock_goal_for_baseline_transition(db, goal_id):
            goal = await self._authorized_goal(db, project_id, goal_id, actor_id)
            run = await self._run_for_goal(db, goal_id)
            request = await db.get(OrchestrationSteeringRequest, request_id)
            if request is None or request.goal_id != goal_id: raise SteeringDomainError("steering_invalid_target", 404, "Steering request not found")
            if request.status != "pending": raise SteeringDomainError("steering_not_pending", 409, "Steering request is not pending")
            state = await self._state(db, goal_id, True); await self._transition(db, request, "withdrawn", "withdrawn", "operator"); state.inbox_version += 1
            await emit_event_once(db, project_id, "orchestration.steering_changed", {"goal_id": str(goal_id), "run_id": str(run.id), "request_id": str(request.id)}, source="operator", dedup_key=f"orchestration.steering_changed:request:{request.id}:inbox:{state.inbox_version}")
            return request

    async def ledger(self, db: AsyncSession, project_id: uuid.UUID, goal_id: uuid.UUID, actor_id: uuid.UUID) -> SteeringLedger:
        try:
            goal = await self._authorized_goal(db, project_id, goal_id, actor_id)
        except SteeringDomainError as exc:
            if exc.code == "steering_forbidden":
                return SteeringLedger(bool(settings.orchestration_conversation_steering_enabled), "forbidden", exc.code, (), (), 0, 0)
            raise
        run = await db.scalar(select(OrchestrationRun).where(
            OrchestrationRun.goal_id == goal_id,
        ).order_by(OrchestrationRun.created_at.desc()))
        if not settings.orchestration_conversation_steering_enabled:
            eligibility, reason = (
                self._eligibility(goal, run, actor_id) if run is not None
                else ("unstarted", None)
            )
            return SteeringLedger(False, eligibility, reason, (), (), 0, 0)
        eligibility, reason = (
            self._eligibility(goal, run, actor_id) if run is not None
            else ("unstarted", "steering_ineligible")
        )
        requests = tuple((await db.scalars(select(OrchestrationSteeringRequest).where(OrchestrationSteeringRequest.goal_id == goal_id).order_by(OrchestrationSteeringRequest.sequence))).all())
        proposals = tuple((await db.scalars(select(OrchestrationSteeringProposal).where(OrchestrationSteeringProposal.goal_id == goal_id).order_by(OrchestrationSteeringProposal.created_at))).all())
        state = await self._state(db, goal_id, False)
        return SteeringLedger(bool(settings.orchestration_conversation_steering_enabled), eligibility, reason, requests, proposals, state.inbox_version if state else 0, state.direction_version if state else 0)

    async def dismiss_proposal(self, db: AsyncSession, project_id: uuid.UUID, goal_id: uuid.UUID, actor_id: uuid.UUID, proposal_id: uuid.UUID) -> OrchestrationSteeringProposal:
        if not settings.orchestration_conversation_steering_enabled:
            raise SteeringDomainError("steering_disabled", 404, "Steering is disabled")
        async with self._orchestration._lock_goal_for_baseline_transition(db, goal_id):
            goal = await self._authorized_goal(db, project_id, goal_id, actor_id)
            run = await self._run_for_goal(db, goal_id)
            proposal = await db.get(OrchestrationSteeringProposal, proposal_id)
            if proposal is None or proposal.goal_id != goal_id or proposal.actor_id != actor_id or proposal.status != "proposed": raise SteeringDomainError("proposal_not_dismissible", 409, "Proposal is not dismissible")
            proposal.status, proposal.dismissed_at = "dismissed", _utcnow(); return proposal

    async def _active(
        self, db: AsyncSession, request: OrchestrationSteeringRequest, goal: OrchestrationGoal,
        run: OrchestrationRun, *, fresh: bool = False,
    ) -> bool:
        if request.status != "applied" or request.goal_id != goal.id: return False
        if request.scope == "goal": return True
        if request.submitted_run_id != run.id: return False
        return await self._item_available(db, request, goal, run, fresh=fresh)

    @staticmethod
    def _overlaps(one: OrchestrationSteeringRequest, two: OrchestrationSteeringRequest) -> bool:
        if "goal" in {one.scope, two.scope}: return True
        if "run" in {one.scope, two.scope}: return one.submitted_run_id == two.submitted_run_id
        return one.target_type == two.target_type and one.target_id == two.target_id

    async def process_pending(self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun) -> SteeringProcessingResult:
        if not settings.orchestration_conversation_steering_enabled: return SteeringProcessingResult((), (), 0, 0)
        state = await self._state(db, goal.id, False)
        if state is None or self._eligibility(goal, run)[0] == "paused": return SteeringProcessingResult((), (), state.inbox_version if state else 0, state.direction_version if state else 0)
        pending = tuple((await db.scalars(select(OrchestrationSteeringRequest).where(OrchestrationSteeringRequest.goal_id == goal.id, OrchestrationSteeringRequest.status == "pending").order_by(OrchestrationSteeringRequest.sequence))).all())
        processed, applied = [], []
        active = tuple((await db.scalars(select(OrchestrationSteeringRequest).where(OrchestrationSteeringRequest.goal_id == goal.id, OrchestrationSteeringRequest.status == "applied"))).all())
        for request in pending:
            processed.append(request.id)
            if request.scope in {"run", "item"} and request.submitted_run_id != run.id:
                await self._transition(db, request, "rejected", "run_changed", "system")
                continue
            if self._eligibility(goal, run)[0] != "active": await self._transition(db, request, "rejected", "steering_ineligible", "system"); continue
            if not await self._item_available(db, request, goal, run):
                await self._transition(db, request, "deferred", "target_already_started", "system"); continue
            await self._transition(db, request, "being_considered", "considering", "system")
            overlapping = [old for old in active if await self._active(db, old, goal, run) and self._overlaps(old, request)]
            old = next((item for item in overlapping if item.id == request.supersedes_request_id), None)
            if overlapping and old is None: await self._transition(db, request, "needs_clarification", "supersedes_required", "system"); continue
            if request.supersedes_request_id and old is None: await self._transition(db, request, "needs_clarification", "invalid_supersedes_request", "system"); continue
            if old:
                await self._transition(db, old, "superseded", "superseded", "system"); state.inbox_version += 1; state.direction_version += 1
                await emit_event_once(db, goal.project_id, "orchestration.steering_changed", {"goal_id": str(goal.id), "run_id": str(run.id), "request_id": str(request.id)}, source="operator", dedup_key=f"orchestration.steering_changed:request:{request.id}:inbox:{state.inbox_version}")
            await self._transition(db, request, "applied", "applied", "system"); state.direction_version += 1; active = tuple(item for item in active if item.id != (old.id if old else None)) + (request,); applied.append(request.id)
        return SteeringProcessingResult(tuple(processed), tuple(applied), state.inbox_version, state.direction_version)

    async def current_versions(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun, *, fresh: bool = False,
    ) -> SteeringVersions:
        state = await self._state(db, goal.id, False, fresh=fresh)
        return SteeringVersions(state.inbox_version if state else 0, state.direction_version if state else 0, self._contract_version(goal), self._plan_version(run))

    async def assert_current_versions(self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun, expected: SteeringVersions) -> None:
        if await self.current_versions(db, goal, run) != expected: raise SteeringVersionsChanged()

    async def assert_current_snapshot(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun,
        expected: SteeringVersions, expected_active_ids: Sequence[uuid.UUID],
    ) -> None:
        """Fence both the version tuple and the projected item lifetime."""
        await db.refresh(goal)
        await db.refresh(run)
        if await self.current_versions(db, goal, run, fresh=True) != expected:
            raise SteeringVersionsChanged()
        current = await self.context_snapshot(db, goal, run, fresh=True)
        if tuple(active_direction_ids({"steering": current})) != tuple(expected_active_ids):
            raise SteeringVersionsChanged()

    async def context_snapshot(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun, *, fresh: bool = False,
    ) -> dict[str, object]:
        if not settings.orchestration_conversation_steering_enabled:
            return {}
        versions = await self.current_versions(db, goal, run, fresh=fresh)
        requests = tuple((await db.scalars(select(OrchestrationSteeringRequest).where(OrchestrationSteeringRequest.goal_id == goal.id, OrchestrationSteeringRequest.status == "applied").order_by(OrchestrationSteeringRequest.sequence).execution_options(populate_existing=fresh))).all())
        active = [request for request in requests if await self._active(db, request, goal, run, fresh=fresh)]
        return {"versions": versions.__dict__, "active_directions": [{"request_id": str(request.id), "directive": request.directive, "target_type": request.target_type, "target_id": request.target_id, "scope": request.scope, "lifetime": request.lifetime, "impact_summary": request.impact_summary} for request in active], "policy": {"advisory_input_only": True, "forbidden_effects": ["start", "pause", "resume", "stop", "cancel", "resolve_ask_human", "weaken_contract", "weaken_plan"]}}

    async def persist_proposal(self, db: AsyncSession, message: ConversationMessage, response: ConversationResponse, draft: SteeringDraft) -> OrchestrationSteeringProposal | None:
        if not settings.orchestration_conversation_steering_enabled:
            return None
        goal = await db.get(OrchestrationGoal, message.goal_id)
        if goal is None or response.run_id is None or response.message_id != message.id:
            return None
        run = await db.get(OrchestrationRun, response.run_id)
        if run is None or run.goal_id != goal.id or self._eligibility(goal, run, message.actor_id)[0] not in {"active", "paused"}:
            return None
        try:
            draft = normalize_draft(draft)
            await self._validate_target(db, goal, run, draft)
        except SteeringDomainError:
            return None
        from huddleroom.models.orchestration_steering import steering_proposal_id
        proposal_id = steering_proposal_id(response.id)
        proposal = await db.get(OrchestrationSteeringProposal, proposal_id)
        if proposal is None:
            try:
                async with db.begin_nested():
                    proposal = OrchestrationSteeringProposal(id=proposal_id, response_id=response.id, goal_id=goal.id, actor_id=message.actor_id, draft={"directive": draft.directive, "target_type": draft.target_type, "target_id": draft.target_id, "scope": draft.scope, "lifetime": draft.lifetime, "impact_summary": draft.impact_summary})
                    db.add(proposal)
                    await db.flush()
            except IntegrityError:
                proposal = await db.get(OrchestrationSteeringProposal, proposal_id)
                if proposal is None:
                    raise
        return proposal

    async def link_result(self, db: AsyncSession, request_id: uuid.UUID, decision_id: uuid.UUID, action_id: uuid.UUID) -> OrchestrationSteeringResultLink:
        request, decision, action = await db.get(OrchestrationSteeringRequest, request_id), await db.get(OrchestrationDecision, decision_id), await db.get(OrchestrationAction, action_id)
        if not request or not decision or not action or decision.run_id != action.run_id or action.decision_id != decision.id:
            raise SteeringDomainError("steering_invalid_target", 422, "Steering result lineage is invalid")
        run = await db.get(OrchestrationRun, decision.run_id)
        if run is None or run.goal_id != request.goal_id or (request.scope != "goal" and request.submitted_run_id != run.id): raise SteeringDomainError("steering_invalid_target", 422, "Steering result lineage is invalid")
        link_id = steering_result_link_id(request_id, decision_id, action_id)
        link = await db.get(OrchestrationSteeringResultLink, link_id)
        if link is None:
            try:
                async with db.begin_nested():
                    link = OrchestrationSteeringResultLink(id=link_id, request_id=request_id, decision_id=decision_id, action_id=action_id)
                    db.add(link)
                    await db.flush()
            except IntegrityError:
                link = await db.get(OrchestrationSteeringResultLink, link_id)
                if link is None:
                    raise
        return link

    @staticmethod
    def version_digest(versions: SteeringVersions) -> str:
        return hashlib.sha256(json.dumps(versions.__dict__, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
