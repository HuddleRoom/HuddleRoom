from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from decimal import Decimal
from graphlib import CycleError, TopologicalSorter
import hashlib
import hmac
import json
from textwrap import shorten
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import and_, delete, desc, func, or_, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import SessionTransactionOrigin, object_session

from huddleroom.config import settings
from huddleroom.database import AsyncSessionLocal
from huddleroom.models.agent import Agent
from huddleroom.models.artifact import Artifact
from huddleroom.models.meeting import Meeting, MeetingDecision
from huddleroom.models.orchestration import (
    ACTIVE_RUN_STATUSES,
    GOAL_STATUSES,
    GOAL_WEIGHT_VALUES,
    OrchestrationAction,
    OrchestrationAgentSuggestion,
    OrchestrationDecision,
    OrchestrationEvidence,
    OrchestrationGate,
    OrchestrationGoal,
    OrchestrationBudgetReservation,
    OrchestrationRoadmapItem,
    OrchestrationRoadmapVersion,
    OrchestrationRun,
)
from huddleroom.models.protocol import Protocol, ProtocolInstance, ProtocolTransition
from huddleroom.models.project import Project
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.models.user import User
from huddleroom.models.base import _utcnow
from huddleroom.schemas.orchestration import OrchestrationDelegationContract, OrchestrationGoalCreate, OrchestrationPlanItem
from huddleroom.schemas.task import TaskCreate
from huddleroom.models.event_log import EventLog
from huddleroom.services.event_bus import BusEvent, emit_event_once
from huddleroom.services.meeting_service import MeetingService
from huddleroom.services.orchestration_decision_validator import validate_orchestration_decision
from huddleroom.models.orchestration_process import (
    OrchestrationAgentReview,
    OrchestrationAuthorityDecision,
    OrchestrationProcessRun,
    OrchestrationWarning,
)
from huddleroom.models.orchestration_memory import OrchestrationMemorySection
from huddleroom.services.orchestration_authority_interview import build_checkpoint, sync_deferred_questions_memory
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService, runtime_decision_identity
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService
from huddleroom.services.orchestration_goal_definition import (
    GOAL_WEIGHT_ORDER,
    GoalDefinitionProcess,
    classify_goal_weight,
    required_decision_keys_for_weight,
)
from huddleroom.services.orchestration_manager_selection import ManagerSelectionProcess
from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess
from huddleroom.services.orchestration_team_hierarchy import TeamHierarchyProcess
from huddleroom.services.orchestration_effectiveness_review import EffectivenessReviewProcess
from huddleroom.services.orchestration_goal_closeout import GoalCloseoutProcess
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_llm_decision_adapter import LLMDecisionAdapter, OrchestrationDecisionAdapter
from huddleroom.services.orchestration_memory_preface import OrchestrationMemoryPrefaceBuilder
from huddleroom.services.orchestration_roster_mapper import OrchestrationRosterMapper
from huddleroom.services.protocol_engine import ProtocolEngineService
from huddleroom.services.project_service import ProjectService
from huddleroom.services.session_service import SessionService
from huddleroom.services.task_service import TaskService


LLM_DECISION_RUN_STATUSES = frozenset({"running", "blocked"})
ASK_HUMAN_EVENT_TYPE = "orchestration.human_input_required"
NOOP_WAIT_EVENT_TYPE = "orchestration.waiting"
AGENT_SUGGESTED_EVENT_TYPE = "orchestration.agent_suggested"
DELEGATION_TASK_CREATED_EVENT_TYPE = "orchestration.delegation_task_created"
PLAN_REQUESTED_EVENT_TYPE = "orchestration.plan_requested"
PLAN_REVISION_REQUESTED_EVENT_TYPE = "orchestration.plan_revision_requested"
PLAN_ACCEPTED_EVENT_TYPE = "orchestration.plan_accepted"
PLAN_ITEM_EXPANDED_EVENT_TYPE = "orchestration.plan_item_expanded"
PLAN_ITEM_GATE_TYPE = "work_completed"
PLAN_GATE_KEY = "plan"
PLAN_GATE_TYPE = "plan_accepted"
ACCEPTED_PLAN_SNAPSHOT_VERSION = 1
PLAN_WORK_FUNCTION = "planning"
FINAL_SUMMARY_GATE_KEY = "final_summary"
FINAL_SUMMARY_GATE_TYPE = "final_summary_accepted"
FINAL_SUMMARY_WORK_FUNCTION = "summarization"
FINAL_SUMMARY_REQUESTED_EVENT_TYPE = "orchestration.final_summary_requested"
RUN_COMPLETED_EVENT_TYPE = "orchestration.run_completed"
DECISION_CONTEXT_EVENT_LIMIT = 20
TICK_EVENT_BATCH_LIMIT = 500
TASK_EVIDENCE_EVENT_TYPES = frozenset({"task.status_changed"})
SESSION_EVIDENCE_EVENT_TYPES = frozenset({"session.completed", "session.failed"})
REVIEW_EVIDENCE_EVENT_TYPES = frozenset({"review.approved", "review.changes_requested"})
PROTOCOL_EVIDENCE_EVENT_TYPES = frozenset({"protocol.completed", "protocol.failed", "protocol.state_transitioned"})
MEETING_EVIDENCE_EVENT_TYPES = frozenset({"meeting.concluded", "meeting.decision_recorded"})
EVIDENCE_EVENT_TYPES = (
    TASK_EVIDENCE_EVENT_TYPES
    | SESSION_EVIDENCE_EVENT_TYPES
    | REVIEW_EVIDENCE_EVENT_TYPES
    | PROTOCOL_EVIDENCE_EVENT_TYPES
    | MEETING_EVIDENCE_EVENT_TYPES
)
RECOVERY_RETRY_FAILED_SESSION_LIMIT = 1
RECOVERY_REASSIGN_FAILED_SESSION_LIMIT = 2
RECOVERY_REPEATED_FAILURE_SESSION_LIMIT = 3
RECOVERY_TASK_RETRIED_EVENT_TYPE = "orchestration.task_retried"
RECOVERY_TASK_REASSIGNED_EVENT_TYPE = "orchestration.task_reassigned"
RECOVERY_VERIFICATION_REQUESTED_EVENT_TYPE = "orchestration.verification_requested"
GATE_REPAIRED_EVENT_TYPE = "orchestration.gate_repaired"
RECOVERY_RUN_PAUSED_EVENT_TYPE = "orchestration.run_paused"
MEETING_SCHEDULED_EVENT_TYPE = "orchestration.meeting_scheduled"
PROTOCOL_STARTED_EVENT_TYPE = "orchestration.protocol_started"

class _ReentrantAsyncioLock:
    """Reentrant wrapper around asyncio.Lock for SQLite baseline-transition serialization.

    asyncio.Lock is non-reentrant: calling acquire() twice from the same task
    deadlocks. This wrapper tracks the current owner task and allows the same
    task to re-enter without re-acquiring the lock.
    """
    def __init__(self):
        self._lock = asyncio.Lock()
        self._owner_task: asyncio.Task | None = None
        self._depth = 0

    @asynccontextmanager
    async def acquire_if_needed(self):
        """Acquire lock only if current task doesn't already own it."""
        current = asyncio.current_task()
        if self._owner_task is current:
            # Already own the lock, this is a reentrant call
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return

        # Not the owner, acquire the lock
        async with self._lock:
            self._owner_task = current
            self._depth = 1
            try:
                yield
            finally:
                self._depth = 0
                self._owner_task = None


# In-process per-goal locks for SQLite baseline-transition serialization
# (review finding, MEDIUM) -- see _lock_goal_for_baseline_transition.
_SQLITE_GOAL_LOCKS: dict[uuid.UUID, _ReentrantAsyncioLock] = {}
_SQLITE_GOAL_LOCKS_GUARD = asyncio.Lock()


class OrchestrationService:
    @property
    def supervision(self):
        """Lazy to avoid a circular service import during baseline startup."""
        from huddleroom.services.orchestration_supervision import OrchestrationSupervisionService
        return OrchestrationSupervisionService(self)

    @staticmethod
    def _json_object_or_empty(value: Any) -> dict[str, Any]:
        return deepcopy(dict(value)) if isinstance(value, Mapping) else {}

    @classmethod
    def _budget_is_exhausted(cls, run: OrchestrationRun) -> bool:
        budget_state = cls._json_object_or_empty(run.budget_state)
        return budget_state.get("status") == "exceeded" and budget_state.get("overridden") is not True

    async def _sync_budget_exhaustion_attention(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun
    ) -> bool:
        warning_service = OrchestrationWarningService()
        if self._budget_is_exhausted(run):
            await warning_service.create_warning(
                db,
                goal.id,
                warning_type="budget_exhausted",
                severity="warning",
                message="Execution is blocked because the orchestration budget is exceeded.",
                run_id=run.id,
            )
            self._upsert_active_blocker(
                run,
                {
                    "kind": "budget_exhausted",
                    "reason": "Execution is blocked because the orchestration budget is exceeded.",
                },
            )
            return True

        self._remove_active_blocker_by_kind(run, "budget_exhausted")
        warnings = list(await db.scalars(
            select(OrchestrationWarning).where(
                OrchestrationWarning.run_id == run.id,
                OrchestrationWarning.warning_type == "budget_exhausted",
                OrchestrationWarning.active.is_(True),
            )
        ))
        for warning in warnings:
            await warning_service.resolve_warning(
                db, warning, resolved_by="system", reason="Budget override allows execution."
            )
        return False

    async def _sync_measured_budget_exhaustion(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun
    ) -> bool:
        from huddleroom.services.orchestration_budget_service import (
            BudgetMeasurementError, OrchestrationBudgetService,
        )

        budget = OrchestrationBudgetService()
        measurement_scope = f"run:{run.id}:active_claim"
        try:
            snapshot = await budget.snapshot_for_run(db, goal, run)
        except BudgetMeasurementError as exc:
            self._upsert_active_blocker(run, {
                "kind": "budget_measurement", "dimension": exc.dimension,
                "session_id": str(exc.session_id), "scope": measurement_scope,
            })
            return False
        run.active_blockers = [
            blocker for blocker in (run.active_blockers or [])
            if not (isinstance(blocker, Mapping) and blocker.get("scope") == measurement_scope)
        ]
        if not snapshot["caps"]:
            return False
        if any(Decimal(snapshot["consumed"][key]) >= Decimal(snapshot["caps"][key]) for key in snapshot["caps"]):
            run.budget_state = {**self._json_object_or_empty(run.budget_state), "status": "exceeded"}
        return await self._sync_budget_exhaustion_attention(db, goal, run)

    @staticmethod
    def _is_project_not_runnable(exc: HTTPException) -> bool:
        return (
            exc.status_code == 409
            and isinstance(exc.detail, dict)
            and exc.detail.get("code") == "project_not_runnable"
        )

    def run_condition(self, goal: OrchestrationGoal, run: OrchestrationRun | None) -> str:
        """Derived, display-only runtime condition (spec: stable runtime
        conditions). Not stored; the reconciler owns run.phase, this reads it.
        There is no stable 'failed' condition."""
        if goal.status == "completed":
            return "completed"
        if goal.status == "cancelled":
            return "stopped"
        if getattr(goal, "goal_type", None) == "continuous" and (goal.continuous_state or {}).get("stopped_at"):
            return "stopped"
        if goal.status == "paused" or (run is not None and run.status == "paused"):
            return "paused"
        if run is not None and run.active_blockers:
            return "needs_attention"
        if getattr(goal, "goal_type", None) == "continuous":
            continuous_state = goal.continuous_state or {}
            health = continuous_state.get("health")
            if health == "needs_attention":
                return "needs_attention"
            if health == "degraded":
                return "degraded"
            if int(continuous_state.get("active_cases", 0)) > 0:
                return "working"
            return "waiting_activation" if run is not None and run.phase == "waiting_activation" else "working"
        if run is not None and run.phase == "ready":
            return "waiting_authority"
        if run is not None and run.phase == "authorized":
            # "working" once anything is in flight is refined in Task 7; the
            # deterministic default here is safe (never claims work exists).
            return "waiting_work"
        return "working"

    async def create_goal(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        data: OrchestrationGoalCreate,
        created_by_user_id: uuid.UUID | None,
    ) -> tuple[OrchestrationGoal, OrchestrationRun]:
        await ProjectService().lock_workspace_boundary(db, project_id)
        await ProjectService().require_runnable_project(db, project_id)
        success_criteria = deepcopy(data.success_criteria)
        constraints = deepcopy(data.constraints)
        budget = deepcopy(data.budget)
        goal = OrchestrationGoal(
            project_id=project_id,
            objective=data.objective,
            original_request=data.original_request or data.objective,
            success_criteria=success_criteria,
            constraints=constraints,
            budget=budget,
            weight=classify_goal_weight(
                success_criteria,
                constraints,
                budget,
                data.objective,
                explicit_multi_work_function=data.explicit_multi_work_function,
            ),
            explicit_multi_work_function=data.explicit_multi_work_function,
            created_by_user_id=created_by_user_id,
        )
        db.add(goal)
        await db.flush()

        run = OrchestrationRun(
            goal_id=goal.id,
            budget_state=deepcopy(budget),
            baseline_authorized=False,
        )
        db.add(run)
        await db.flush()
        return goal, run

    async def override_goal_weight(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
        *,
        weight: str,
        reason: str,
        user_id: uuid.UUID | None,
    ) -> OrchestrationGoal:
        """Human-forced weight (spec 6.2): heavier is free; lighter than the
        deterministic heuristic weight is an override and creates a warning.

        Service-level validation (reviewer finding, MEDIUM; spec 15.6): the
        REST schema's `Literal` only protects the router path. Any other
        caller of this method directly must be rejected the same way,
        instead of hitting a bare `KeyError` from `GOAL_WEIGHT_ORDER[weight]`
        below or persisting an invalid value to SQLite.

        Compares the requested weight against a fresh `classify_goal_weight`
        call, not `goal.weight` (reviewer finding, MEDIUM, this revision --
        Deviations 1, 12): comparing to `goal.weight` (the previous
        *effective* weight, itself possibly a prior override) drifts across
        repeated overrides and can both under- and over-warn relative to the
        actual spec-6.2 heuristic -- see Deviation 12 for the two concrete
        failure modes. Since none of `success_criteria`/`constraints`/
        `budget`/`objective` change after creation (no goal-edit endpoint
        exists yet, Deviation 2), recomputing the heuristic here is stable
        and correct as the comparison baseline. `explicit_multi_work_function`
        is omitted from the recompute (it is a creation-time-only input, not
        a persisted column) -- a goal whose creation-time weight was elevated
        solely by that flag can be overridden down to the field-only
        heuristic tier without a warning; a narrower, documented gap
        (Deviation 1's narrowed retention note), not the two-directional bug
        this fix resolves. `goal.weight` is still updated to the requested
        value below, and process re-evaluation (Step 6) still compares
        against the *previous effective* `goal.weight` -- that is a
        different, correct question (did this run's own required depth just
        change) from the warning's question (is this below the heuristic).
        """
        if weight not in GOAL_WEIGHT_VALUES:
            raise HTTPException(
                status_code=422,
                detail=f"invalid weight {weight!r}; must be one of {sorted(GOAL_WEIGHT_VALUES)}",
            )
        goal = await self.get_goal(db, project_id, goal_id)
        if goal is None:
            raise HTTPException(status_code=404, detail="Orchestration goal not found")
        previous_weight = goal.weight
        heuristic_weight = classify_goal_weight(
            goal.success_criteria, goal.constraints, goal.budget, goal.objective,
            explicit_multi_work_function=goal.explicit_multi_work_function,
        )
        process_service = OrchestrationProcessService()
        current_process = await process_service.get_current(db, goal.id, "goal_definition")
        restarting_goal_definition = (
            current_process is not None
            and current_process.status == "completed"
            and GOAL_WEIGHT_ORDER[weight] > GOAL_WEIGHT_ORDER[previous_weight]
        )
        if restarting_goal_definition:
            await ProjectService().lock_workspace_boundary(db, project_id)
            await ProjectService().require_runnable_project(db, project_id)
        goal.weight = weight
        goal.weight_overridden_by = f"human:{user_id}" if user_id is not None else "human:anonymous"
        if GOAL_WEIGHT_ORDER[weight] < GOAL_WEIGHT_ORDER[heuristic_weight]:
            # Added directly rather than via OrchestrationWarningService.
            # create_warning: that method is idempotent on (goal, type,
            # linkage) among active warnings, which would collapse two
            # separate human overrides into one warning. Each override is a
            # distinct human action, not a retry of the previous one, so
            # each below-heuristic override records its own warning.
            db.add(
                OrchestrationWarning(
                    goal_id=goal.id,
                    warning_type="goal_weight_forced_lighter",
                    severity="warning",
                    message=(
                        f"Goal weight forced to '{weight}' below its deterministic "
                        f"heuristic weight '{heuristic_weight}' ({reason}). Baseline "
                        "process depth is reduced; skipped checks may hide scope, "
                        "team, or verification risks."
                    ),
                )
            )
        if current_process is not None:
            new_order = GOAL_WEIGHT_ORDER[weight]
            old_order = GOAL_WEIGHT_ORDER[previous_weight]
            if current_process.status == "completed" and new_order > old_order:
                # The prior run's compressed/lighter-weight pass no longer
                # reflects the depth this goal now requires. Force a fresh
                # run; next tick's advance() runs the full pass, and the
                # goal-level settled-key guard (Deviation 8) still skips
                # gaps whose decision key was already answered.
                await ProjectService().lock_workspace_boundary(db, project_id)
                await ProjectService().require_runnable_project(db, project_id)
                await process_service.start_process(
                    db, goal.id, process_type="goal_definition",
                    trigger_reason=(
                        f"weight override: forced heavier from {previous_weight!r} to {weight!r}"
                    ),
                )
            elif current_process.status == "waiting_decision" and new_order < old_order:
                required_keys = required_decision_keys_for_weight(goal)
                decision_service = OrchestrationAuthorityDecisionService()
                pending = await decision_service.list_decisions(db, goal.id, status="pending")
                for decision in pending:
                    if (
                        decision.decision_key.startswith("goal_definition:")
                        and not decision.decision_key.startswith("goal_definition:adaptive:")
                        and decision.decision_key not in required_keys
                    ):
                        await decision_service.cancel_decision(
                            db, decision,
                            reason=f"superseded by weight override to {weight!r}",
                        )
        await db.flush()
        return goal

    async def get_goal(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
    ) -> OrchestrationGoal | None:
        result = await db.execute(
            select(OrchestrationGoal).where(
                OrchestrationGoal.id == goal_id,
                OrchestrationGoal.project_id == project_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_run_for_goal(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
    ) -> OrchestrationRun | None:
        """Latest run for a goal within a project, regardless of run status."""
        result = await db.execute(
            select(OrchestrationRun)
            .join(OrchestrationGoal, OrchestrationGoal.id == OrchestrationRun.goal_id)
            .where(
                OrchestrationGoal.project_id == project_id,
                OrchestrationRun.goal_id == goal_id,
            )
            .order_by(OrchestrationRun.started_at.desc(), OrchestrationRun.id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def _needs_you_counts(
        self,
        db: AsyncSession,
        goal_ids: list[uuid.UUID],
    ) -> dict[uuid.UUID, int]:
        """Batch compute needs-you counts for goals.

        Returns dict mapping goal_id -> needs_you_count (sum of active_blockers,
        failed gates, pending human decisions, and active warnings).
        """
        if not goal_ids:
            return {}

        # Get latest run per goal (same tie-break as get_run_for_goal)
        result = await db.execute(
            select(OrchestrationRun)
            .where(OrchestrationRun.goal_id.in_(goal_ids))
            .order_by(OrchestrationRun.goal_id, OrchestrationRun.started_at.desc(), OrchestrationRun.id.desc())
        )
        all_runs = list(result.scalars().all())
        latest_runs: dict[uuid.UUID, OrchestrationRun] = {}
        for run in all_runs:
            if run.goal_id not in latest_runs:
                latest_runs[run.goal_id] = run

        # Count failed gates per run
        result = await db.execute(
            select(OrchestrationGate.run_id, func.count(OrchestrationGate.id))  # pylint: disable=not-callable
            .where(
                OrchestrationGate.run_id.in_([r.id for r in latest_runs.values()]),
                OrchestrationGate.status == "failed",
            )
            .group_by(OrchestrationGate.run_id)
        )
        failed_gates_per_run = dict(result.all())

        # Count active warnings per goal
        result = await db.execute(
            select(OrchestrationWarning.goal_id, func.count(OrchestrationWarning.id))  # pylint: disable=not-callable
            .where(
                OrchestrationWarning.goal_id.in_(goal_ids),
                OrchestrationWarning.active.is_(True),
            )
            .group_by(OrchestrationWarning.goal_id)
        )
        active_warnings_per_goal = dict(result.all())

        # Count pending human decisions per goal
        result = await db.execute(
            select(OrchestrationAuthorityDecision.goal_id, func.count(OrchestrationAuthorityDecision.id))  # pylint: disable=not-callable
            .where(
                OrchestrationAuthorityDecision.goal_id.in_(goal_ids),
                OrchestrationAuthorityDecision.status == "pending",
                OrchestrationAuthorityDecision.authority == "human",
            )
            .group_by(OrchestrationAuthorityDecision.goal_id)
        )
        pending_decisions_per_goal = dict(result.all())

        # Compute total per goal
        counts: dict[uuid.UUID, int] = {}
        for goal_id in goal_ids:
            run = latest_runs.get(goal_id)
            count = 0

            # Active blockers from latest run
            if run and run.active_blockers:
                count += len(run.active_blockers)

            # Failed gates from latest run
            if run:
                count += failed_gates_per_run.get(run.id, 0)

            # Pending human decisions for this goal
            count += pending_decisions_per_goal.get(goal_id, 0)

            # Active warnings for this goal
            count += active_warnings_per_goal.get(goal_id, 0)

            counts[goal_id] = count

        return counts

    async def list_goals(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        status: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> tuple[list[OrchestrationGoal], str | None]:
        if status is not None and status not in GOAL_STATUSES:
            raise HTTPException(status_code=400, detail=f"Invalid orchestration goal status '{status}'")

        limit = max(1, min(limit, 100))
        query = (
            select(OrchestrationGoal)
            .where(OrchestrationGoal.project_id == project_id)
            .order_by(OrchestrationGoal.created_at.desc(), OrchestrationGoal.id.desc())
            .limit(limit + 1)
        )
        if status is not None:
            query = query.where(OrchestrationGoal.status == status)
        if cursor:
            try:
                cursor_dt_str, cursor_id_str = cursor.split("__", 1)
                cursor_dt = datetime.fromisoformat(cursor_dt_str)
                cursor_id = uuid.UUID(cursor_id_str)
            except (ValueError, AttributeError) as exc:
                raise HTTPException(status_code=400, detail="Invalid cursor") from exc
            query = query.where(
                or_(
                    OrchestrationGoal.created_at < cursor_dt,
                    and_(OrchestrationGoal.created_at == cursor_dt, OrchestrationGoal.id < cursor_id),
                )
            )

        result = await db.execute(query)
        items = list(result.scalars().all())
        next_cursor = None
        if len(items) > limit:
            items = items[:limit]
            last = items[-1]
            next_cursor = f"{last.created_at.isoformat()}__{last.id}"

        # Compute needs_you_count for each goal
        goal_ids = [item.id for item in items]
        counts = await self._needs_you_counts(db, goal_ids)
        for item in items:
            item.needs_you_count = counts.get(item.id, 0)

        return items, next_cursor

    async def get_active_run_for_goal(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
    ) -> OrchestrationRun | None:
        result = await db.execute(
            select(OrchestrationRun)
            .join(OrchestrationGoal, OrchestrationGoal.id == OrchestrationRun.goal_id)
            .where(
                OrchestrationGoal.project_id == project_id,
                OrchestrationRun.goal_id == goal_id,
                OrchestrationRun.status.in_(ACTIVE_RUN_STATUSES),
            )
            .order_by(OrchestrationRun.started_at.desc(), OrchestrationRun.id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def record_validated_decision(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        input_snapshot: Any,
        llm_output: Any,
        parsed_decision: Any,
    ) -> OrchestrationDecision:
        result = await db.execute(select(OrchestrationRun.id).where(OrchestrationRun.id == run_id))
        if result.scalar_one_or_none() is None:
            raise HTTPException(status_code=404, detail="Orchestration run not found")

        validation = validate_orchestration_decision(parsed_decision)
        action_type = parsed_decision.get("action_type") if isinstance(parsed_decision, Mapping) else None
        decision_type = (
            action_type if isinstance(action_type, str) and action_type and len(action_type) <= 100 else "invalid"
        )
        reason = parsed_decision.get("reason") if isinstance(parsed_decision, Mapping) else None

        decision = OrchestrationDecision(
            run_id=run_id,
            decision_type=decision_type,
            input_snapshot=self._json_object_or_empty(input_snapshot),
            llm_output=deepcopy(llm_output),
            parsed_decision=self._json_object_or_empty(parsed_decision),
            validator_status="accepted" if validation.accepted else "rejected",
            rejection_reason=validation.rejection_reason,
            reason=reason if isinstance(reason, str) else None,
        )
        return await self._insert_decision(db, run_id, decision)

    async def _insert_decision(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        decision: OrchestrationDecision,
    ) -> OrchestrationDecision:
        nested = await db.begin_nested()
        try:
            db.add(decision)
            await db.flush()
        except IntegrityError as exc:
            await nested.rollback()
            session = object_session(decision)
            if session is not None:
                session.expunge(decision)
            if not await self._run_exists(db, run_id):
                raise HTTPException(status_code=404, detail="Orchestration run not found") from exc
            raise

        await nested.commit()
        return decision

    async def request_llm_decision(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        adapter: LLMDecisionAdapter | None = None,
    ) -> OrchestrationDecision:
        """Ask the LLM for a coordination decision and persist the audit row.

        The committed path is authoritative: if the run is already committed and in a
        decision-eligible status, context and writes go through isolated sessions and
        commit before returning. That keeps the caller transaction out of the LLM round
        trip and lets the audit row survive caller rollback. We only fall back to the
        caller session when the run is not committed yet, because the isolated session
        cannot see it.
        """
        from huddleroom.models.project import Project

        def _orch_decide_ctx(project, goal):
            return (
                {"id": str(project.id), "name": project.name, "description": project.description} if project else None,
                {"objective": goal.objective, "weight": getattr(goal, "weight", None),
                 "status": getattr(goal, "status", None)},
            )

        decision_adapter = adapter or OrchestrationDecisionAdapter()
        project_ctx = None
        goal_ctx = None
        context = None
        async with AsyncSessionLocal() as read_db:
            caller_bind = db.get_bind()
            factory_bind = read_db.get_bind()
            caller_engine = getattr(caller_bind, "sync_engine", caller_bind)
            factory_engine = getattr(factory_bind, "sync_engine", factory_bind)
            if caller_engine is factory_engine:
                read_run = await read_db.get(OrchestrationRun, run_id)
                if read_run is not None:
                    if read_run.status not in LLM_DECISION_RUN_STATUSES:
                        raise HTTPException(status_code=409, detail=f"Orchestration run is {read_run.status}")
                    goal = await read_db.get(OrchestrationGoal, read_run.goal_id)
                    if goal is None:
                        raise HTTPException(status_code=404, detail="Orchestration goal not found")

                    project_ctx, goal_ctx = _orch_decide_ctx(
                        await read_db.get(Project, goal.project_id), goal
                    )
                    # Pass allow_heal=False: read-only session won't commit heals (fix #15).
                    await self._ensure_baseline_processes_ready(read_db, goal.id, allow_heal=False)
                    context = await self._decision_context(read_db, goal, read_run)
                    latest_decision = await self._latest_decision_for_run(read_db, run_id)
                    if (
                        latest_decision is not None
                        and latest_decision.validator_status == "accepted"
                        and latest_decision.input_snapshot == context
                    ):
                        return latest_decision

        if context is not None:
            adapter_result = await decision_adapter.decide(context, project=project_ctx, goal=goal_ctx)
            async with AsyncSessionLocal() as write_db:
                write_run = await self._lock_run_for_committed_decision(
                    write_db,
                    run_id,
                    require_active_status=True,
                )
                if write_run is None:
                    raise HTTPException(status_code=404, detail="Orchestration run not found")

                await self._ensure_baseline_processes_ready(write_db, write_run.goal_id)
                latest_decision = await self._latest_decision_for_run(write_db, run_id)
                if (
                    latest_decision is not None
                    and latest_decision.validator_status == "accepted"
                    and latest_decision.input_snapshot == adapter_result.input_snapshot
                ):
                    await write_db.commit()
                    # Detached return is intentional here: AsyncSessionLocal uses
                    # expire_on_commit=False, so committed-path callers can inspect it
                    # after this isolated session closes.
                    return latest_decision

                # ponytail: SQLite is still single-writer here, but only for the
                # short insert/commit window instead of the full LLM round trip.
                # Paused stays writable here on purpose: pause should not block a
                # decision already in flight, only stop future decision reads/ticks.
                decision = await self.record_validated_decision(
                    write_db,
                    run_id=run_id,
                    input_snapshot=adapter_result.input_snapshot,
                    llm_output=adapter_result.llm_output,
                    parsed_decision=adapter_result.parsed_decision,
                )
                await write_db.commit()
                # Detached return is intentional here: AsyncSessionLocal uses
                # expire_on_commit=False, so committed-path callers can inspect it
                # after this isolated session closes.
                return decision

        run = await db.get(OrchestrationRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Orchestration run not found")
        if run.status not in LLM_DECISION_RUN_STATUSES:
            raise HTTPException(status_code=409, detail=f"Orchestration run is {run.status}")

        goal = await db.get(OrchestrationGoal, run.goal_id)
        if goal is None:
            raise HTTPException(status_code=404, detail="Orchestration goal not found")
        await self._ensure_baseline_processes_ready(db, goal.id)
        context = await self._decision_context(db, goal, run)
        project_ctx, goal_ctx = _orch_decide_ctx(await db.get(Project, goal.project_id), goal)
        adapter_result = await decision_adapter.decide(context, project=project_ctx, goal=goal_ctx)
        return await self.record_validated_decision(
            db,
            run_id=run.id,
            input_snapshot=adapter_result.input_snapshot,
            llm_output=adapter_result.llm_output,
            parsed_decision=adapter_result.parsed_decision,
        )

    async def _lock_run_for_committed_decision(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        require_active_status: bool = False,
    ) -> OrchestrationRun | None:
        bind = db.get_bind()
        if bind is not None and bind.dialect.name == "sqlite":
            if db.in_transaction():
                raise RuntimeError(
                    "_lock_run_for_committed_decision requires a fresh sqlite session; "
                    "BEGIN IMMEDIATE must be the first statement"
                )
            await db.execute(text("BEGIN IMMEDIATE"))
            run = await db.get(OrchestrationRun, run_id)
        else:
            result = await db.execute(select(OrchestrationRun).where(OrchestrationRun.id == run_id).with_for_update())
            run = result.scalar_one_or_none()

        if run is None:
            return None
        if require_active_status and run.status not in ACTIVE_RUN_STATUSES:
            raise HTTPException(status_code=409, detail=f"Orchestration run is {run.status}")
        return run

    @asynccontextmanager
    async def _lock_goal_for_baseline_transition(self, db: AsyncSession, goal_id: uuid.UUID):
        """Serialize baseline process advancement/completion for one goal.

        Must run on the SAME session that performs the critical section's
        reads/writes. An earlier version of this lock opened a second,
        dedicated AsyncSessionLocal purely to hold the lock while the real
        work continued on the caller's `db` -- on SQLite that is two
        connections to one file, and any write attempted on `db` while the
        lock session holds BEGIN IMMEDIATE blocks for the full busy_timeout
        and then raises "database is locked" (verified locally: a write from
        a second connection against a BEGIN IMMEDIATE-held connection always
        fails this way, it does not just occasionally contend). Every real
        tick() does at least one write in this section, so that design would
        fail in production on every non-trivial tick.

        On Postgres, `FOR UPDATE` works mid-transaction with no restriction,
        so it is taken directly on `db` -- a real row lock, no second
        connection needed. On SQLite there is no separate row-lock primitive,
        so instead an in-process `asyncio.Lock` keyed by goal_id serializes
        tick()/force-start/skip against each other (review finding, MEDIUM):
        this codebase already treats SQLite as effectively single-process
        (see the `ponytail:` note on the LLM-decision write path), so an
        asyncio-level lock closes the same race a row lock would on Postgres
        -- concurrent requests for the same goal now actually queue instead
        of reading stale baseline state or hitting SQLITE_BUSY_SNAPSHOT.

        CRITICAL FIX: Made reentrant to prevent same-task deadlock. SQLite
        calls from tick() to execute_request_final_summary_action() and
        execute_complete_run_action() both hold the lock while calling
        _ensure_baseline_processes_ready(), which also tries to acquire
        the same lock. The wrapper _ReentrantAsyncioLock detects same-task
        re-entry and skips re-acquiring (only decrements depth on exit).
        """
        bind = db.get_bind()
        if bind is not None and bind.dialect.name == "sqlite":
            async with _SQLITE_GOAL_LOCKS_GUARD:
                lock = _SQLITE_GOAL_LOCKS.setdefault(goal_id, _ReentrantAsyncioLock())
            async with lock.acquire_if_needed():
                yield
            return
        await db.execute(
            select(OrchestrationGoal).where(OrchestrationGoal.id == goal_id).with_for_update()
        )
        yield

    async def _latest_decision_for_run(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
    ) -> OrchestrationDecision | None:
        result = await db.execute(
            select(OrchestrationDecision)
            .where(OrchestrationDecision.run_id == run_id)
            .order_by(OrchestrationDecision.created_at.desc(), OrchestrationDecision.id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def _decision_context(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
    ) -> dict:
        from huddleroom.services.orchestration_goal_definition import goal_objective_with_clarifications

        if run.goal_id != goal.id:
            raise ValueError("orchestration run does not belong to goal")
        roster = await OrchestrationRosterMapper().context_snapshot(db, goal.project_id)
        memory_preface = await OrchestrationMemoryPrefaceBuilder().build(db, goal, run)
        from huddleroom.services.orchestration_supervision_context import OrchestrationSupervisionContextBuilder
        durable_context = await OrchestrationSupervisionContextBuilder().build(db, goal, run)
        return {
            **durable_context,
            "goal": {
                "id": str(goal.id),
                "project_id": str(goal.project_id),
                "objective": goal_objective_with_clarifications(goal),
                "original_request": goal.original_request,
                "success_criteria": deepcopy(goal.success_criteria),
                "constraints": deepcopy(goal.constraints),
                "budget": deepcopy(goal.budget),
                "status": goal.status,
            },
            "run": {
                "id": str(run.id),
                "goal_id": str(run.goal_id),
                "status": run.status,
                "phase": run.phase,
                "cycle_key": run.cycle_key,
                "plan_state": deepcopy(run.plan_state),
                "active_blockers": deepcopy(run.active_blockers),
                "budget_state": deepcopy(run.budget_state),
                "retry_state": deepcopy(run.retry_state),
                "supervision_state": deepcopy(run.supervision_state),
            },
            "open_gates": [
                {
                    "id": str(gate.id), "gate_type": gate.gate_type,
                    "success_criterion_key": gate.success_criterion_key,
                    "roadmap_version_id": self._json_object_or_empty(gate.required_evidence).get("roadmap_version_id"),
                    "work_producer_agent_ids": self._json_object_or_empty(gate.required_evidence).get("work_producer_agent_ids", []),
                }
                for gate in (await db.scalars(select(OrchestrationGate).where(
                    OrchestrationGate.run_id == run.id, OrchestrationGate.status == "open",
                ).order_by(OrchestrationGate.created_at, OrchestrationGate.id)))
            ],
            "memory_preface": memory_preface,
            "roster": roster,
        }

    async def _recent_events_for_decision_context(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        limit: int = DECISION_CONTEXT_EVENT_LIMIT,
    ) -> list[EventLog]:
        query = (
            select(EventLog)
            .where(
                EventLog.project_id == project_id,
                EventLog.event_type.notlike("orchestration.%"),
            )
            .order_by(desc(EventLog.seq))
            .limit(limit)
        )
        result = await db.execute(query)
        return list(reversed(result.scalars().all()))

    async def _steering_action_fence(
        self, db: AsyncSession, run_id: uuid.UUID, idempotency_key: str, decision_id: uuid.UUID | None,
    ):
        """Resolve the final steering key and fence its snapshot at one shared seam."""
        if decision_id is None:
            return idempotency_key, None, None, (), None, None
        from huddleroom.services.orchestration_steering import (
            OrchestrationSteeringService, active_direction_ids, steering_versions_from_snapshot,
        )

        decision = await db.get(OrchestrationDecision, decision_id)
        versions = steering_versions_from_snapshot(decision.input_snapshot or {}) if decision else None
        if versions is None:
            return idempotency_key, None, None, (), None, None
        expected_active_ids = tuple(active_direction_ids(decision.input_snapshot or {}))
        run = await db.get(OrchestrationRun, run_id)
        goal = await db.get(OrchestrationGoal, run.goal_id) if run else None
        if run is None or goal is None:
            return idempotency_key, None, None, (), None, None
        steering = OrchestrationSteeringService()
        async with self._lock_goal_for_baseline_transition(db, goal.id):
            await db.refresh(goal)
            await db.refresh(run)
            await steering.assert_current_snapshot(
                db, goal, run, versions, expected_active_ids,
            )
        suffix = f":steering:{steering.version_digest(versions)}"
        if not idempotency_key.endswith(suffix):
            idempotency_key = idempotency_key[:255 - len(suffix)] + suffix
        return idempotency_key, steering, versions, expected_active_ids, goal, run

    async def _fence_steering_action(
        self, db, action, steering, versions, expected_active_ids, goal, run, *, newly_reserved,
    ):
        if versions is None or action.status != "reserved":
            return action
        from huddleroom.services.orchestration_steering import SteeringVersionsChanged

        action.dispatch_contract = {
            **(action.dispatch_contract or {}), "steering_versions": vars(versions),
        }
        try:
            await steering.assert_current_snapshot(db, goal, run, versions, expected_active_ids)
        except SteeringVersionsChanged:
            await self._fail_reserved_action_for_current_flow(db, action, "stale_steering_versions")
            raise
        return action

    async def reserve_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        idempotency_key: str,
        action_type: str,
        request: dict,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        """Reserve once per idempotency key; existing failed actions replay as-is."""
        if not idempotency_key:
            raise HTTPException(status_code=400, detail="Action idempotency key is required")
        if len(idempotency_key) > 255:
            raise HTTPException(status_code=400, detail="Action idempotency key is too long")
        if not action_type:
            raise HTTPException(status_code=400, detail="Action type is required")
        if len(action_type) > 100:
            raise HTTPException(status_code=400, detail="Action type is too long")

        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        if existing is not None:
            if existing.action_type != action_type:
                raise HTTPException(status_code=409, detail="Action idempotency key conflicts with action type")
            if existing.status != "reserved":
                return existing

        from huddleroom.services.orchestration_steering import SteeringVersionsChanged

        try:
            idempotency_key, steering, versions, expected_active_ids, goal, run = await self._steering_action_fence(
                db, run_id, idempotency_key, decision_id,
            )
        except SteeringVersionsChanged:
            if existing is not None and existing.status == "reserved" and existing.decision_id == decision_id:
                await self._fail_reserved_action_for_current_flow(db, existing, "stale_steering_versions")
            raise
        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        if existing is not None:
            if existing.action_type != action_type:
                raise HTTPException(status_code=409, detail="Action idempotency key conflicts with action type")
            return await self._fence_steering_action(
                db, existing, steering, versions, expected_active_ids, goal, run, newly_reserved=False,
            )

        result = await db.execute(select(OrchestrationRun.status).where(OrchestrationRun.id == run_id))
        run_status = result.scalar_one_or_none()
        if run_status is None:
            raise HTTPException(status_code=404, detail="Orchestration run not found")
        if run_status not in ACTIVE_RUN_STATUSES:
            raise HTTPException(status_code=409, detail=f"Orchestration run is {run_status}")

        if decision_id is not None:
            result = await db.execute(
                select(OrchestrationDecision.id).where(
                    OrchestrationDecision.id == decision_id,
                    OrchestrationDecision.run_id == run_id,
                )
            )
            if result.scalar_one_or_none() is None:
                raise HTTPException(status_code=404, detail="Orchestration decision not found")

        nested = await db.begin_nested()
        try:
            action = OrchestrationAction(
                run_id=run_id,
                decision_id=decision_id,
                idempotency_key=idempotency_key,
                action_type=action_type,
                request=deepcopy(request),
            )
            db.add(action)
            await db.flush()
        except IntegrityError as exc:
            await nested.rollback()
            existing = await self._existing_action_for_key(db, run_id, idempotency_key)
            if existing is None:
                raise
            if existing.action_type != action_type:
                raise HTTPException(
                    status_code=409,
                    detail="Action idempotency key conflicts with action type",
                ) from exc
            return await self._fence_steering_action(
                db, existing, steering, versions, expected_active_ids, goal, run, newly_reserved=False,
            )

        await nested.commit()
        return await self._fence_steering_action(
            db, action, steering, versions, expected_active_ids, goal, run, newly_reserved=True,
        )

    async def _existing_action_for_key(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        idempotency_key: str,
    ) -> OrchestrationAction | None:
        result = await db.execute(
            select(OrchestrationAction).where(
                OrchestrationAction.run_id == run_id,
                OrchestrationAction.idempotency_key == idempotency_key,
            )
        )
        return result.scalar_one_or_none()

    async def mark_action_failed(
        self,
        db: AsyncSession,
        action: OrchestrationAction,
        error: str,
    ) -> OrchestrationAction:
        """Fail a reserved action once; completed actions conflict and prior failures keep their original error."""
        result = await db.execute(
            update(OrchestrationAction)
            .where(
                OrchestrationAction.id == action.id,
                OrchestrationAction.status == "reserved",
            )
            .values(status="failed", error=error)
            .execution_options(synchronize_session="fetch")
        )
        if result.rowcount == 1:
            return action

        await db.refresh(action)
        if action.status == "completed":
            raise HTTPException(status_code=409, detail="Action already completed")
        return action

    async def _persist_reserved_action_failure(
        self,
        db: AsyncSession,
        action_id: uuid.UUID,
        error: str,
    ) -> None:
        """Persist a terminal replay failure so it survives what the caller does next.

        SQLite is a single-writer database: once `db`'s connection has written
        anything in the current transaction (e.g. ProjectService.lock_workspace_boundary,
        or an earlier reserve_action insert), a second connection trying to write the
        same file blocks for the full busy_timeout and then raises "database is locked" --
        there is no way for a second connection to get in until `db`'s transaction ends.
        So on SQLite we always write the failure on `db` itself.

        Whether we then commit depends on who owns `db`'s transaction:
        - If nobody else owns it (a plain autobegin transaction -- production's
          get_db(), or a caller's bare `async with session_factory() as db:` with no
          explicit `.begin()`), the caller may roll back or discard `db` right after
          this call returns, so we commit here to make the failure durable regardless.
          This also commits whatever else is pending on `db` a little earlier than the
          caller's own commit point -- every other write on these action-execution
          paths is idempotency-keyed or dedup-guarded (reserve_action, emit_event_once,
          "ensure_*_gate"/"ensure_*_ready" create-if-missing helpers), so that's safe.
        - If the caller explicitly opened this transaction (`async with session.begin():`,
          e.g. the db_session test fixture) and is going to keep using `db` afterwards,
          committing here would end their transaction out from under them
          (InvalidRequestError: "Can't operate on closed transaction inside context
          manager"). These callers always read the failure back on this SAME session
          before their own transaction concludes, so a flush is enough; leave the
          commit/rollback decision to whoever owns the transaction.

        Postgres uses row-level locking (Project vs orchestration_actions are
        different rows/tables, no conflict), so it keeps using a fully independent
        connection here regardless of transaction ownership.
        """
        if db.bind is not None and db.bind.dialect.name == "sqlite":
            action = await db.get(OrchestrationAction, action_id)
            if action is None:
                return
            try:
                await self.mark_action_failed(db, action, error)
            except HTTPException as exc:
                if exc.status_code != 409:
                    raise

            # get_transaction() returns the ROOT transaction, and a commit here would
            # release any ancestor SAVEPOINT still in effect. So only commit when we
            # truly own a bare autobegin transaction with no open savepoint; if the
            # caller opened this transaction (explicit begin) or we are inside a nested
            # transaction, flush and leave the commit/rollback to whoever owns it.
            transaction = db.sync_session.get_transaction()
            caller_owns_transaction = (
                transaction is not None
                and transaction.origin != SessionTransactionOrigin.AUTOBEGIN
            )
            if caller_owns_transaction or db.in_nested_transaction():
                await db.flush()
            else:
                await db.commit()
            return

        session_factory = AsyncSessionLocal
        if db.bind is not None:
            session_factory = async_sessionmaker(
                db.bind,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        async with session_factory() as write_db:
            action = await write_db.get(OrchestrationAction, action_id)
            if action is None:
                return
            try:
                await self.mark_action_failed(write_db, action, error)
            except HTTPException as exc:
                if exc.status_code != 409:
                    raise
            await write_db.commit()

    async def _fail_reserved_action(
        self,
        db: AsyncSession,
        action: OrchestrationAction,
        error: str,
    ) -> None:
        await self._persist_reserved_action_failure(db, action.id, error)
        await db.refresh(action)
        if action.status == "reserved":
            await self.mark_action_failed(db, action, error)
            await db.flush()
            await db.refresh(action)

    async def _fail_reserved_action_in_transaction(
        self,
        db: AsyncSession,
        action: OrchestrationAction,
        error: str,
    ) -> None:
        await self.mark_action_failed(db, action, error)
        await db.flush()
        await db.refresh(action)

    async def _fail_reserved_action_for_current_flow(
        self,
        db: AsyncSession,
        action: OrchestrationAction,
        error: str,
    ) -> None:
        if db.in_nested_transaction():
            await self._fail_reserved_action_in_transaction(db, action, error)
            return
        await self._fail_reserved_action(db, action, error)

    async def _validate_delegation_targets(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: Mapping[str, Any],
    ) -> tuple[uuid.UUID, Agent, uuid.UUID | None]:
        result = await db.execute(select(OrchestrationRun.status).where(OrchestrationRun.id == run_id))
        run_status = result.scalar_one_or_none()
        if run_status is None:
            raise HTTPException(status_code=404, detail="Orchestration run not found")
        if run_status not in ACTIVE_RUN_STATUSES:
            raise HTTPException(status_code=409, detail=f"Orchestration run is {run_status}")

        agent_id = self._required_uuid(request.get("agent_id"), "agent_id")
        agent = await db.get(Agent, agent_id)
        if agent is None or not agent.is_active:
            raise HTTPException(status_code=404, detail="Agent not found")

        inherited = self._json_object_or_empty(request.get("orchestrator_context"))
        team = inherited.get("team")
        allowed = self._roadmap_team_agent_ids(team)
        if allowed is not None and str(agent_id) not in allowed:
            raise HTTPException(status_code=409, detail="Roadmap delegation agent is outside the inherited team")

        project_id = await self._project_id_for_run(db, run_id)
        parent_task_id = self._optional_uuid(request.get("parent_task_id"), "parent_task_id")
        if request.get("work_function") == "follow_up":
            parent_task_id = self._required_uuid(request.get("parent_task_id"), "parent_task_id")
        parent = (
            await TaskService().get(db, project_id, parent_task_id)
            if parent_task_id is not None
            else None
        )
        if parent_task_id is not None and parent is None:
            raise HTTPException(status_code=404, detail="Parent task not found")
        if request.get("work_function") == "follow_up":
            parent_action = await db.scalar(
                select(OrchestrationAction.id).where(
                    OrchestrationAction.run_id == run_id,
                    OrchestrationAction.action_type == "create_delegation_task",
                    OrchestrationAction.target_type == "task",
                    OrchestrationAction.target_id == parent_task_id,
                    OrchestrationAction.status == "completed",
                )
            )
            if parent_action is None:
                raise HTTPException(status_code=409, detail="Follow-up parent not in this run")
            if parent.assigned_to != agent_id:
                raise HTTPException(status_code=409, detail="Follow-up must reuse the parent's agent")
            source_session_id = self._optional_uuid(request.get("source_session_id"), "source_session_id")
            if source_session_id is not None:
                marker = await self._existing_action_for_key(
                    db, run_id, f"run:{run_id}:kind:report_consumed:task:{parent_task_id}"
                )
                marker_request = self._json_object_or_empty(marker.request if marker else {})
                if marker_request.get("session_id") != str(source_session_id):
                    raise HTTPException(status_code=409, detail="Follow-up source session was not the consumed report")

        return project_id, agent, parent_task_id

    @staticmethod
    def _roadmap_team_agent_ids(team: Mapping[str, Any] | None) -> set[str] | None:
        """Extract agent identities from an accepted team; ``None`` means no contract."""
        if not isinstance(team, Mapping):
            return None
        found: set[str] = set()
        def add(value: Any) -> None:
            if isinstance(value, str):
                try:
                    found.add(str(uuid.UUID(value)))
                except ValueError:
                    pass

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, child in value.items():
                    if key in {"verifier_id", "verifier_agent_id"}:
                        add(child)
                    elif key == "role_to_agent" and isinstance(child, Mapping):
                        for agent_id in child.values():
                            add(agent_id)
                    elif key == "assignments" and isinstance(child, list):
                        for assignment in child:
                            if isinstance(assignment, Mapping):
                                add(assignment.get("agent_ref"))
                    elif key in {"agent_ids", "agents", "contributors", "reviewers", "validators", "team_leads", "specialists", "verifier_ids"} and isinstance(child, list):
                        for agent_id in child:
                            add(agent_id)
                    elif key == "manager" and isinstance(child, Mapping):
                        if child.get("kind") == "agent":
                            add(child.get("id"))
                    elif key == "candidate_agents":
                        continue
                    else:
                        visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        visit(team)
        return found

    @staticmethod
    def _task_matches_orchestration_action(
        task: Task,
        action: OrchestrationAction,
        *,
        run_id: uuid.UUID | None = None,
        work_function: str | None = None,
    ) -> bool:
        task_metadata = OrchestrationService._json_object_or_empty(task.metadata_)
        orchestration = OrchestrationService._json_object_or_empty(task_metadata.get("orchestration"))
        if orchestration.get("action_id") != str(action.id):
            return False
        if run_id is not None and orchestration.get("run_id") != str(run_id):
            return False
        if work_function is not None and orchestration.get("work_function") != work_function:
            return False
        return True

    async def _find_task_for_action(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        action: OrchestrationAction,
        *,
        run_id: uuid.UUID | None = None,
        work_function: str | None = None,
    ) -> Task | None:
        candidate_ids: list[uuid.UUID] = []
        if action.target_type == "task" and action.target_id is not None:
            candidate_ids.append(action.target_id)
        if action.id not in candidate_ids:
            candidate_ids.append(action.id)

        for candidate_id in candidate_ids:
            task = await TaskService().get(db, project_id, candidate_id)
            if task is not None and self._task_matches_orchestration_action(
                task,
                action,
                run_id=run_id,
                work_function=work_function,
            ):
                return task

        query = select(Task).where(
            Task.project_id == project_id,
            Task.metadata_["orchestration"]["action_id"].as_string() == str(action.id),
        )
        if run_id is not None:
            query = query.where(Task.metadata_["orchestration"]["run_id"].as_string() == str(run_id))
        if work_function is not None:
            query = query.where(Task.metadata_["orchestration"]["work_function"].as_string() == work_function)
        result = await db.execute(
            query
            .order_by(Task.created_at.desc(), Task.id.desc())
        )
        for task in result.scalars().all():
            if self._task_matches_orchestration_action(task, action, run_id=run_id, work_function=work_function):
                return task
        return None

    async def execute_create_delegation_task_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
        skip_baseline_gate: bool = False,
    ) -> OrchestrationAction:
        gate_run = await db.get(OrchestrationRun, run_id)
        if gate_run is None:
            raise HTTPException(status_code=404, detail="Orchestration run not found")
        project_id = await self._project_id_for_run(db, run_id)
        await ProjectService().lock_workspace_boundary(db, project_id)
        await ProjectService().require_runnable_project(db, project_id)
        if not skip_baseline_gate:
            # Authority-decision delegation tasks (_sync_agent_authority_decisions)
            # pass skip_baseline_gate=True: they exist specifically to deliver the
            # decision that a parked baseline process (e.g. team_hierarchy in
            # "waiting_decision") is itself blocking on, so gating them on baseline
            # readiness would deadlock -- the decision could never be delegated to
            # the agent who needs to answer it.
            await self._ensure_baseline_processes_ready(db, gate_run.goal_id)
        request_to_store = self._canonical_delegation_task_request(request)
        if request_to_store["work_function"] == "follow_up":
            source_session_id = await self._follow_up_source_session_id(db, run_id, request_to_store)
            request_to_store["source_session_id"] = str(source_session_id) if source_session_id else None
            idempotency_key = self._follow_up_delegation_key(run_id, request_to_store)
            await self._validate_delegation_targets(db, run_id, request_to_store)
        idempotency_key, steering, versions, expected_active_ids, goal_for_fence, run_for_fence = (
            await self._steering_action_fence(db, run_id, idempotency_key, decision_id)
        )
        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        # Terminal replays reuse their original action after the baseline gate.
        if existing is not None and existing.status != "reserved":
            return await self._fence_steering_action(
                db, existing, steering, versions, expected_active_ids, goal_for_fence, run_for_fence,
                newly_reserved=False,
            )
        request_to_store = existing.request if existing is not None else request_to_store
        agent = None
        parent_task_id = None
        if existing is None:
            _, agent, parent_task_id = await self._validate_delegation_targets(db, run_id, request_to_store)
            authority = await self.roadmap_pre_release_authority(
                db, run_id, agent,
                self._json_object_or_empty(request_to_store.get("orchestrator_context")),
            )
            if authority.get("status") == "pending":
                raise HTTPException(status_code=409, detail="budget_wait")
            if authority.get("status") == "rejected":
                raise HTTPException(status_code=409, detail="needs_attention")
            if authority:
                context = self._json_object_or_empty(request_to_store.get("orchestrator_context"))
                roadmap = self._json_object_or_empty(context.get("roadmap"))
                request_to_store = {
                    **request_to_store,
                    "orchestrator_context": {
                        **context,
                        "roadmap": {**roadmap, "budget_approval": authority},
                    },
                }

        action = await self.reserve_action(
            db,
            run_id=run_id,
            idempotency_key=idempotency_key,
            action_type="create_delegation_task",
            request=request_to_store,
            decision_id=decision_id,
        )
        if action.status != "reserved":
            return action

        existing_task = await self._find_task_for_action(db, project_id, action, run_id=run_id)
        if existing_task is not None:
            await self._emit_delegation_task_created(db, project_id, action, existing_task)
            return await self._mark_action_completed(db, action, target_type="task", target_id=existing_task.id)

        stored_request = action.request or {}
        try:
            project_id, agent, parent_task_id = await self._validate_delegation_targets(db, run_id, stored_request)
        except HTTPException as exc:
            if exc.status_code == 404 and exc.detail in {"Agent not found", "Parent task not found"}:
                await self._fail_reserved_action_for_current_flow(db, action, str(exc.detail))
            raise

        agent_id = agent.id
        task_service = TaskService()

        run = await db.get(OrchestrationRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Orchestration run not found")
        goal = await db.get(OrchestrationGoal, run.goal_id)
        if goal is None:
            raise HTTPException(status_code=404, detail="Orchestration goal not found")

        contract = OrchestrationDelegationContract(
            goal_id=goal.id,
            run_id=run.id,
            action_id=action.id,
            agent_id=agent_id,
            work_function=self._required_string(stored_request.get("work_function"), "work_function"),
            scope=self._required_string(stored_request.get("scope"), "scope"),
            inputs=self._string_list(stored_request.get("inputs")),
            deliverable=self._required_string(stored_request.get("deliverable"), "deliverable"),
            forbidden_work=self._string_list(stored_request.get("forbidden_work")),
            success_evidence=self._string_list(stored_request.get("success_evidence")),
            budget=self._json_object_or_empty(stored_request.get("budget")),
            report_schema=self._json_object_or_empty(stored_request.get("report_schema")),
            parent_task_id=parent_task_id,
            orchestrator_context=self._json_object_or_empty(stored_request.get("orchestrator_context")),
        )
        contract_payload = contract.model_dump(mode="json")
        budget_amounts = self._json_object_or_empty(contract.budget)
        continuous = self._json_object_or_empty(
            self._json_object_or_empty(contract.orchestrator_context).get("continuous")
        )
        protected = (
            contract.work_function in {"verification", "final_summary", "report_clarification"}
            or ":verify_gate:" in idempotency_key or ":final_summary" in idempotency_key
        )
        # Discovery capacity is already reserved once per cycle by its durable
        # parent reservation; an action ledger would count that same hold twice.
        discovery_reservation = await db.scalar(select(OrchestrationBudgetReservation.id).where(
            OrchestrationBudgetReservation.discovery_run_id == run.id,
            OrchestrationBudgetReservation.parent_goal_id == goal.id,
            OrchestrationBudgetReservation.status == "active",
        )) if continuous.get("discovery_run_id") == str(run.id) else None
        if discovery_reservation is not None:
            budget_amounts = {}
        elif protected:
            from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService

            budget_amounts = await OrchestrationBudgetService().protected_action_allocation(db, goal, run)
        else:
            from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService, canonical_amounts

            caps = (await OrchestrationBudgetService().snapshot_for_run(db, goal, run))["caps"]
            budget_amounts = canonical_amounts(budget_amounts, allowed=set(caps))
            if caps and (not budget_amounts or not any(Decimal(value) > 0 for value in budget_amounts.values())):
                raise HTTPException(status_code=409, detail="Budgeted delegation requires a nonzero supported budget")
        if budget_amounts and not action.budget_ledger:
            from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService

            await OrchestrationBudgetService().reserve_action_budget(
                db, goal, run, action, budget_amounts, enforceable=agent.adapter_type in {"api", "cli"},
                closeout=protected,
            )
        task = await task_service.create(
            db,
            project_id,
            TaskCreate(
                title=self._delegation_task_title(contract.work_function, contract.deliverable),
                description=self._delegation_task_description(goal, contract),
                assigned_to=agent_id,
                parent_id=parent_task_id,
                metadata={
                    "orchestration": {
                        "goal_id": str(goal.id),
                        "run_id": str(run.id),
                        "action_id": str(action.id),
                        "work_function": contract.work_function,
                    },
                    "orchestration_contract": contract_payload,
                },
            ),
            task_id=action.id,
        )
        await self._emit_delegation_task_created(db, project_id, action, task)
        return await self._mark_action_completed(db, action, target_type="task", target_id=task.id)

    async def execute_request_final_summary_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        stable_key = self._final_summary_request_key(run_id)
        if idempotency_key != stable_key:
            raise HTTPException(
                status_code=409,
                detail=f"Final summary idempotency key must be '{stable_key}'",
            )
        existing = await self._existing_action_for_key(db, run_id, stable_key)
        if existing is not None and existing.status != "reserved":
            return existing
        run = await db.get(OrchestrationRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Orchestration run not found")
        if run.status not in LLM_DECISION_RUN_STATUSES:
            raise HTTPException(status_code=409, detail=f"Orchestration run is {run.status}")
        goal = await db.get(OrchestrationGoal, run.goal_id)
        if goal is None:
            raise HTTPException(status_code=404, detail="Orchestration goal not found")
        await self._ensure_baseline_processes_ready(db, goal.id)
        if goal.status not in {"active", "blocked"}:
            raise HTTPException(status_code=409, detail=f"Orchestration goal is {goal.status}")
        gates, evidence = await self._accepted_non_summary_manifest(db, run.id)
        roadmap_version_id = None
        if goal.goal_type == "roadmap":
            from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService
            version = await OrchestrationRoadmapService(self).current_version(db, goal.id)
            roadmap_version_id = str(version.id) if version is not None else None
        request_to_store = self._canonical_final_summary_request(
            existing.request if existing is not None else request, gates, evidence, goal, roadmap_version_id
        )
        fits = await OrchestrationRosterMapper().rank_agents(
            db,
            goal.project_id,
            FINAL_SUMMARY_WORK_FUNCTION,
            required_capabilities=[FINAL_SUMMARY_WORK_FUNCTION],
        )
        await db.refresh(run)
        await db.refresh(goal)
        if run.status not in LLM_DECISION_RUN_STATUSES:
            raise HTTPException(status_code=409, detail=f"Orchestration run is {run.status}")
        if goal.status not in {"active", "blocked"}:
            raise HTTPException(status_code=409, detail=f"Orchestration goal is {goal.status}")
        await ProjectService().lock_workspace_boundary(db, goal.project_id)
        await ProjectService().require_runnable_project(db, goal.project_id)
        if not fits or fits[0].weak:
            await self.handle_weak_roster_fit(
                db,
                run.id,
                FINAL_SUMMARY_WORK_FUNCTION,
                required_capabilities=[FINAL_SUMMARY_WORK_FUNCTION],
                reason="No strong summarization agent fit exists.",
                decision_id=decision_id,
            )
            raise HTTPException(status_code=409, detail="No strong summarization agent fit")
        action = await self.reserve_action(
            db,
            run_id=run.id,
            idempotency_key=stable_key,
            action_type="request_final_summary",
            request=request_to_store,
            decision_id=decision_id,
        )
        if action.status != "reserved":
            return action
        gate = await self._ensure_final_summary_gate(db, run.id)
        try:
            delegation = await self.execute_create_delegation_task_action(
                db,
                run_id=run.id,
                request=self._final_summary_delegation_request(
                    goal,
                    run,
                    fits[0].agent_id,
                    gates,
                    evidence,
                    request_to_store["criterion_evidence_manifest"],
                ),
                idempotency_key=self._final_summary_delegation_key(run.id),
                decision_id=decision_id,
            )
        except Exception as exc:
            if isinstance(exc, HTTPException) and self._is_project_not_runnable(exc):
                raise
            error = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
            await self._fail_reserved_action_for_current_flow(db, action, error)
            raise
        if delegation.status != "completed" or delegation.target_id is None:
            await self._fail_reserved_action_for_current_flow(
                db, action, "Final summary delegation did not complete"
            )
            raise HTTPException(status_code=409, detail="Final summary delegation did not complete")
        task = await db.get(Task, delegation.target_id)
        if task is None:
            await self._fail_reserved_action_for_current_flow(db, action, "Summary task not found")
            raise HTTPException(status_code=404, detail="Summary task not found")
        self._set_final_summary_task_metadata(
            task,
            gate,
            {
                "request_action_id": str(action.id),
                "gate_id": str(gate.id),
                "accepted_gate_ids": [str(item.id) for item in gates],
                "accepted_evidence_ids": [str(item.id) for item in evidence],
            },
        )
        await emit_event_once(
            db,
            goal.project_id,
            FINAL_SUMMARY_REQUESTED_EVENT_TYPE,
            {
                "goal_id": str(goal.id),
                "run_id": str(run.id),
                "action_id": str(action.id),
                "task_id": str(task.id),
                "agent_id": str(task.assigned_to),
                "gate_id": str(gate.id),
                "work_function": FINAL_SUMMARY_WORK_FUNCTION,
            },
            source="orchestrator",
            dedup_key=f"{FINAL_SUMMARY_REQUESTED_EVENT_TYPE}:action:{action.id}",
        )
        return await self._mark_action_completed(
            db, action, target_type="task", target_id=task.id
        )

    async def execute_complete_run_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        stable_key = self._complete_run_key(run_id)
        if idempotency_key != stable_key:
            raise HTTPException(
                status_code=409,
                detail=f"Complete run idempotency key must be '{stable_key}'",
            )
        existing = await self._existing_action_for_key(db, run_id, stable_key)
        if existing is not None and existing.status != "reserved":
            return existing
        request_to_store = self._canonical_complete_run_request(
            existing.request if existing is not None else request
        )
        run = await db.get(OrchestrationRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Orchestration run not found")
        if run.status not in LLM_DECISION_RUN_STATUSES:
            raise HTTPException(status_code=409, detail=f"Orchestration run is {run.status}")
        goal = await db.get(OrchestrationGoal, run.goal_id)
        if goal is None:
            raise HTTPException(status_code=404, detail="Orchestration goal not found")
        await self._ensure_baseline_processes_ready(db, goal.id)
        if goal.status not in {"active", "blocked"}:
            raise HTTPException(status_code=409, detail=f"Orchestration goal is {goal.status}")
        manifest = await self._completion_manifest(db, goal, run)
        await db.refresh(run)
        await db.refresh(goal)
        if run.status not in LLM_DECISION_RUN_STATUSES:
            raise HTTPException(status_code=409, detail=f"Orchestration run is {run.status}")
        if goal.status not in {"active", "blocked"}:
            raise HTTPException(status_code=409, detail=f"Orchestration goal is {goal.status}")
        # Recheck readiness under lock to close TOCTOU window
        await self._ensure_baseline_processes_ready(db, goal.id)
        action = await self.reserve_action(
            db,
            run_id=run.id,
            idempotency_key=stable_key,
            action_type="complete_run",
            request=request_to_store,
            decision_id=decision_id,
        )
        if action.status != "reserved":
            return action
        completed_at = _utcnow()
        run.status = "completed"
        run.phase = "completed"
        run.completed_at = completed_at
        run.active_blockers = []
        goal.status = "completed"
        await emit_event_once(
            db,
            goal.project_id,
            RUN_COMPLETED_EVENT_TYPE,
            {
                "goal_id": str(goal.id),
                "run_id": str(run.id),
                "action_id": str(action.id),
                "reason": request_to_store["reason"],
                "completed_at": completed_at.isoformat(),
                **manifest,
            },
            source="orchestrator",
            dedup_key=f"{RUN_COMPLETED_EVENT_TYPE}:run:{run.id}",
        )
        return await self._mark_action_completed(
            db, action, target_type="run", target_id=run.id
        )

    async def execute_request_plan_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        gate_run = await db.get(OrchestrationRun, run_id)
        if gate_run is None:
            raise HTTPException(status_code=404, detail="Orchestration run not found")
        idempotency_key, steering, versions, expected_active_ids, goal_for_fence, run_for_fence = (
            await self._steering_action_fence(db, run_id, idempotency_key, decision_id)
        )
        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        # If terminal action already exists, return it without the readiness gate
        if existing is not None and existing.status != "reserved":
            return await self._fence_steering_action(
                db, existing, steering, versions, expected_active_ids, goal_for_fence, run_for_fence,
                newly_reserved=False,
            )
        # New actions must pass the baseline readiness gate; a replay of an
        # already-reserved action was authorized at reservation time and must
        # complete idempotently without re-gating (mirrors the existing
        # `if existing is None` guard on delegation-target revalidation below).
        if existing is None:
            await self._ensure_baseline_processes_ready(db, gate_run.goal_id)
        if existing is None:
            inherited_goal = await db.get(OrchestrationGoal, gate_run.goal_id)
            inherited_context = self._json_object_or_empty(inherited_goal.orchestrator_context)
            if self._has_delegable_orchestrator_context(inherited_context):
                request = {**request, "orchestrator_context": deepcopy(inherited_context)}
        request_to_store = existing.request if existing is not None else self._canonical_plan_request(request)
        agent = None
        run = None
        if existing is None:
            run = await self._run_for_plan_action(db, run_id)
            await self._ensure_plan_request_allowed(db, run)
            _, agent, _ = await self._validate_delegation_targets(
                db,
                run_id,
                {**request_to_store, "parent_task_id": None},
            )
            authority = await self.roadmap_pre_release_authority(
                db, run_id, agent,
                self._json_object_or_empty(request_to_store.get("orchestrator_context")),
            )
            if authority.get("status") == "pending":
                raise HTTPException(status_code=409, detail="budget_wait")
            if authority.get("status") == "rejected":
                raise HTTPException(status_code=409, detail="needs_attention")
            if authority:
                context = self._json_object_or_empty(request_to_store.get("orchestrator_context"))
                roadmap = self._json_object_or_empty(context.get("roadmap"))
                request_to_store = {
                    **request_to_store,
                    "orchestrator_context": {
                        **context,
                        "roadmap": {**roadmap, "budget_approval": authority},
                    },
                }

        action = await self.reserve_action(
            db,
            run_id=run_id,
            idempotency_key=idempotency_key,
            action_type="request_plan",
            request=request_to_store,
            decision_id=decision_id,
        )
        if action.status != "reserved":
            return action

        project_id = await self._project_id_for_run(db, run_id)
        if run is None:
            run = await db.get(OrchestrationRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Orchestration run not found")
        goal = await db.get(OrchestrationGoal, run.goal_id)
        if goal is None:
            raise HTTPException(status_code=404, detail="Orchestration goal not found")

        try:
            await self._ensure_plan_request_allowed(db, run, check_readiness=existing is None)
        except HTTPException as exc:
            if exc.status_code == 409:
                await self._fail_reserved_action(db, action, str(exc.detail))
            raise
        task = await self._find_task_for_action(
            db,
            project_id,
            action,
            run_id=run_id,
            work_function=PLAN_WORK_FUNCTION,
        )
        if task is None:
            stored_request = action.request or {}
            if agent is None or stored_request != request_to_store:
                try:
                    _, agent, _ = await self._validate_delegation_targets(
                        db,
                        run_id,
                        {**stored_request, "parent_task_id": None},
                    )
                except HTTPException as exc:
                    if exc.status_code == 404 and exc.detail == "Agent not found":
                        await self._persist_reserved_action_failure(db, action.id, str(exc.detail))
                        await db.refresh(action)
                    raise

            contract = self._planning_contract(goal, run, action, stored_request)
            task = await TaskService().create(
                db,
                project_id,
                TaskCreate(
                    title=self._delegation_task_title(contract.work_function, contract.deliverable),
                    description=self._delegation_task_description(goal, contract),
                    assigned_to=agent.id,
                    metadata={
                        "orchestration": {
                            "goal_id": str(goal.id),
                            "run_id": str(run.id),
                            "action_id": str(action.id),
                            "work_function": contract.work_function,
                        },
                        "orchestration_plan": {
                            "status": "requested",
                        },
                        "orchestration_contract": contract.model_dump(mode="json"),
                    },
                ),
                task_id=action.id,
            )

        await self._cancel_superseded_planning_task(db, run, action)
        gate = await self._ensure_plan_gate(db, run_id)
        self._set_plan_requested_state(run, action, task, gate)
        await self._emit_plan_requested(
            db,
            project_id,
            goal,
            run,
            action,
            task,
            task.assigned_to or self._required_uuid((action.request or {}).get("agent_id"), "agent_id"),
            gate,
        )
        return await self._mark_action_completed(db, action, target_type="task", target_id=task.id)

    async def execute_request_plan_revision_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        gate_run = await db.get(OrchestrationRun, run_id)
        if gate_run is None:
            raise HTTPException(status_code=404, detail="Orchestration run not found")
        idempotency_key, steering, versions, expected_active_ids, goal_for_fence, run_for_fence = (
            await self._steering_action_fence(db, run_id, idempotency_key, decision_id)
        )
        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        # If terminal action already exists, return it without the readiness gate
        if existing is not None and existing.status != "reserved":
            return await self._fence_steering_action(
                db, existing, steering, versions, expected_active_ids, goal_for_fence, run_for_fence,
                newly_reserved=False,
            )
        await self._ensure_baseline_processes_ready(db, gate_run.goal_id)
        request_to_store = existing.request if existing is not None else self._canonical_plan_revision_request(request)
        run = None
        planning_task = None
        if existing is None:
            run = await self._run_for_plan_action(db, run_id)
            await self._ensure_plan_revision_allowed(db, run)
            planning_task = await self._planning_task_for_run(
                db,
                run,
                self._required_uuid(request_to_store.get("plan_task_id"), "plan_task_id"),
            )
            agent = await db.get(Agent, planning_task.assigned_to) if planning_task.assigned_to else None
            if agent is None:
                raise HTTPException(status_code=409, detail="Roadmap planning agent is invalid")
            inherited_goal = await db.get(OrchestrationGoal, run.goal_id)
            authority = await self.roadmap_pre_release_authority(
                db, run_id, agent, inherited_goal.orchestrator_context if inherited_goal else None,
            )
            if authority.get("status") == "pending":
                raise HTTPException(status_code=409, detail="budget_wait")
            if authority.get("status") == "rejected":
                raise HTTPException(status_code=409, detail="needs_attention")
            if authority:
                context = self._json_object_or_empty((inherited_goal.orchestrator_context or {}) if inherited_goal else {})
                roadmap = self._json_object_or_empty(context.get("roadmap"))
                request_to_store = {
                    **request_to_store,
                    "orchestrator_context": {
                        **context,
                        "roadmap": {**roadmap, "budget_approval": authority},
                    },
                }

        action = await self.reserve_action(
            db,
            run_id=run_id,
            idempotency_key=idempotency_key,
            action_type="request_plan_revision",
            request=request_to_store,
            decision_id=decision_id,
        )
        if action.status != "reserved":
            return action

        if run is None:
            run = await self._run_for_plan_action(db, run_id)
        try:
            await self._ensure_plan_revision_allowed(db, run)
        except HTTPException as exc:
            if exc.status_code == 409:
                await self._fail_reserved_action(db, action, str(exc.detail))
            raise
        if planning_task is None:
            planning_task = await self._planning_task_for_run(
                db,
                run,
                self._required_uuid(action.request.get("plan_task_id") if action.request else None, "plan_task_id"),
            )

        project_id = await self._project_id_for_run(db, run_id)
        stored_request = action.request or {}
        revision_request = self._required_string(stored_request.get("revision_request"), "revision_request")
        self._append_plan_revision(run, action, planning_task.id, revision_request)
        await emit_event_once(
            db,
            project_id,
            PLAN_REVISION_REQUESTED_EVENT_TYPE,
            {
                "run_id": str(run.id),
                "action_id": str(action.id),
                "plan_task_id": str(planning_task.id),
                "revision_request": revision_request,
            },
            source="orchestrator",
            dedup_key=f"{PLAN_REVISION_REQUESTED_EVENT_TYPE}:action:{action.id}",
        )
        return await self._mark_action_completed(db, action, target_type="task", target_id=planning_task.id)

    async def execute_request_roadmap_replan_action(
        self, db: AsyncSession, run_id: uuid.UUID, request: dict, idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService

        run = await self._run_for_plan_action(db, run_id)
        goal = await db.get(OrchestrationGoal, run.goal_id)
        if goal is None or goal.goal_type != "roadmap":
            raise HTTPException(status_code=409, detail="Roadmap replan requires a Roadmap goal")
        version = await OrchestrationRoadmapService(self).current_version(db, goal.id)
        state = self._json_object_or_empty(run.plan_state)
        if version is None or state.get("status") != "accepted" or state.get("roadmap_version_id") != str(version.id) or state.get("roadmap_version") != version.version:
            raise HTTPException(status_code=409, detail="Roadmap replan requires an accepted current version")
        expected_key = f"run:{run.id}:kind:request_roadmap_replan:version:{version.version}"
        expected_key, steering, versions, expected_active_ids, goal_for_fence, run_for_fence = (
            await self._steering_action_fence(db, run_id, expected_key, decision_id)
        )
        if idempotency_key != expected_key:
            raise HTTPException(status_code=409, detail="Roadmap replan key does not match current version")
        existing = await self._existing_action_for_key(db, run_id, expected_key)
        if existing is not None and existing.status != "reserved":
            return await self._fence_steering_action(
                db, existing, steering, versions, expected_active_ids, goal_for_fence, run_for_fence,
                newly_reserved=False,
            )
        stored = existing.request if existing is not None else self._canonical_roadmap_replan_request(request)
        if state.get("pending_replan"):
            raise HTTPException(status_code=409, detail="Roadmap replan is already pending")
        action = await self.reserve_action(
            db, run_id=run_id, idempotency_key=idempotency_key, action_type="request_roadmap_replan",
            request=stored, decision_id=decision_id,
        )
        if action.status != "reserved":
            return action
        project_id, agent, _ = await self._validate_delegation_targets(
            db, run_id, {**stored, "work_function": PLAN_WORK_FUNCTION, "parent_task_id": None},
        )
        task = await self._find_task_for_action(db, project_id, action, run_id=run_id, work_function=PLAN_WORK_FUNCTION)
        if task is None:
            released = list((await db.scalars(select(OrchestrationRoadmapItem).where(
                OrchestrationRoadmapItem.goal_id == goal.id
            ))).all())
            contract = self._planning_contract(goal, run, action, {**stored, "work_function": PLAN_WORK_FUNCTION})
            contract = contract.model_copy(update={
                "inputs": [*contract.inputs, f"Current immutable Roadmap version: {json.dumps(version.snapshot, sort_keys=True)}",
                           f"Released immutable item snapshots: {json.dumps([row.item_snapshot for row in released], sort_keys=True)}"],
                "forbidden_work": [*contract.forbidden_work, "Do not change or remove any released Roadmap item."],
            })
            task = await TaskService().create(
                db, project_id,
                TaskCreate(title="Planning: Roadmap replan", description="Produce a replacement plan for unstarted Roadmap work.",
                    assigned_to=agent.id, metadata={
                        "orchestration": {"goal_id": str(goal.id), "run_id": str(run.id), "action_id": str(action.id), "work_function": PLAN_WORK_FUNCTION},
                        "orchestration_plan": {"status": "replan_requested"},
                        "orchestration_contract": contract.model_dump(mode="json"),
                        "roadmap_replan": {"current_version": version.snapshot,
                            "released_items": [row.item_snapshot for row in released],
                            "forbidden_work": "Do not change or remove any released Roadmap item."},
                    }), task_id=action.id,
            )
        run.plan_state = {**state, "pending_replan": {
            "task_id": str(task.id), "action_id": str(action.id), "version_id": str(version.id),
            "planner_agent_id": str(agent.id),
        }}
        await db.flush()
        return await self._mark_action_completed(db, action, target_type="task", target_id=task.id)

    async def execute_accept_plan_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        request_to_store = existing.request if existing is not None else self._canonical_accept_plan_request(request)
        run = None
        artifact = None
        if existing is None:
            run = await self._run_for_plan_action(db, run_id)
            await self._ensure_plan_acceptance_pending(db, run)
            artifact = await self._plan_artifact_for_run(
                db,
                run,
                self._required_uuid(request_to_store.get("plan_artifact_id"), "plan_artifact_id"),
            )
            goal = await db.get(OrchestrationGoal, run.goal_id)
            if goal is not None and goal.goal_type == "roadmap" and self._json_object_or_empty(run.plan_state).get("pending_replan"):
                from huddleroom.services.orchestration_roadmap_service import (
                    OrchestrationRoadmapService,
                    RoadmapReplanMeasurementAttention,
                )
                try:
                    await OrchestrationRoadmapService(self).prepare_replan_acceptance(db, goal, run, artifact)
                except RoadmapReplanMeasurementAttention:
                    await db.flush()
                    raise

        action = await self.reserve_action(
            db,
            run_id=run_id,
            idempotency_key=idempotency_key,
            action_type="accept_plan",
            request=request_to_store,
            decision_id=decision_id,
        )
        if action.status != "reserved":
            return action

        if run is None:
            run = await self._run_for_plan_action(db, run_id)
        try:
            await self._ensure_plan_acceptance_pending(db, run)
        except HTTPException as exc:
            if exc.status_code == 409:
                await self._fail_reserved_action(db, action, str(exc.detail))
            raise
        stored_request = action.request or {}
        if artifact is None:
            try:
                artifact = await self._plan_artifact_for_run(
                    db,
                    run,
                    self._required_uuid(stored_request.get("plan_artifact_id"), "plan_artifact_id"),
                )
            except HTTPException as exc:
                if exc.status_code in {404, 409}:
                    await self._fail_reserved_action(db, action, str(exc.detail))
                raise

        goal = await db.get(OrchestrationGoal, run.goal_id)
        try:
            if goal is None:
                raise HTTPException(status_code=404, detail="Orchestration goal not found")
            if goal.goal_type == "roadmap":
                from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService

                async with db.begin_nested():
                    version = await OrchestrationRoadmapService(self).accept_version(
                        db, goal, run, artifact, None
                    )
                    plan_state = self._json_object_or_empty(run.plan_state)
                    pending = self._json_object_or_empty(plan_state.get("pending_replan"))
                    plan_task_id = self._required_uuid(
                        pending.get("task_id") if pending else plan_state.get("planning_task_id"), "planning_task_id"
                    )
                    gate = await self._accept_plan_gate(db, run, artifact, action)
                    run.plan_state = {
                        **plan_state,
                        "status": "accepted",
                        "planning_task_id": str(plan_task_id),
                        "accepted_artifact_id": str(artifact.id),
                        "accept_action_id": str(action.id),
                        "plan_gate_id": str(gate.id),
                        "roadmap_version_id": str(version.id),
                        "roadmap_version": version.version,
                        "accepted_plan_fingerprint": version.fingerprint,
                        **({} if not pending else {"pending_replan": None}),
                    }
                    await db.flush()
                items = None
            else:
                items = self._plan_items_from_artifact(artifact)
                self._validate_plan_criterion_links(goal, items)
                version = None
        except HTTPException as exc:
            # Savepoint rollback expires a projection that was already flushed.
            await db.refresh(run, ["plan_state"])
            if not self._json_object_or_empty(run.plan_state).get("pending_replan"):
                self._set_plan_revision_required(run, str(exc.detail))
                await db.flush()
            await self._fail_reserved_action(db, action, str(exc.detail))
            raise

        if version is None:
            plan_state = self._json_object_or_empty(run.plan_state)
            plan_task_id = self._required_uuid(plan_state.get("planning_task_id"), "planning_task_id")
            gate = await self._accept_plan_gate(db, run, artifact, action)
            run.plan_state = {
                **plan_state,
                "status": "accepted",
                "planning_task_id": str(plan_task_id),
                "accepted_artifact_id": str(artifact.id),
                "accept_action_id": str(action.id),
                "plan_gate_id": str(gate.id),
                "accepted_plan_snapshot": self._accepted_plan_snapshot(artifact.id, items),
            }
        project_id = await self._project_id_for_run(db, run_id)
        await emit_event_once(
            db,
            project_id,
            PLAN_ACCEPTED_EVENT_TYPE,
            {
                "run_id": str(run.id),
                "action_id": str(action.id),
                "plan_task_id": str(plan_task_id),
                "plan_artifact_id": str(artifact.id),
                "gate_id": str(gate.id),
                "accepted_plan_fingerprint": (
                    version.fingerprint if version is not None
                    else run.plan_state["accepted_plan_snapshot"]["fingerprint"]
                ),
            },
            source="orchestrator",
            dedup_key=f"{PLAN_ACCEPTED_EVENT_TYPE}:action:{action.id}",
        )
        return await self._mark_action_completed(db, action, target_type="artifact", target_id=artifact.id)

    async def expand_accepted_plan(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        decision_id: uuid.UUID | None = None,
    ) -> list[OrchestrationAction]:
        run = await self._run_for_plan_action(db, run_id)
        goal = await db.get(OrchestrationGoal, run.goal_id)
        try:
            items = await self._accepted_plan_items(db, run)
        except HTTPException as exc:
            self._block_plan_integrity(goal, run, str(exc.detail))
            raise
        artifact = await self._accepted_plan_artifact(db, run)
        actions: list[OrchestrationAction] = []
        nested = await db.begin_nested()
        try:
            for item in items:
                action = await self.execute_expand_plan_item_action(
                    db,
                    run_id=run.id,
                    request={
                        "action_type": "expand_plan_item",
                        "plan_item_id": item.id,
                        "work_function": item.work_function,
                    },
                    idempotency_key=self._expand_plan_item_idempotency_key(run.id, item.id),
                    decision_id=decision_id,
                    run=run,
                    artifact=artifact,
                    item=item,
                )
                if action.status != "completed":
                    raise HTTPException(status_code=409, detail=f"Plan item '{item.id}' expansion did not complete")
                actions.append(action)
        except Exception:
            await nested.rollback()
            raise
        await nested.commit()
        return actions

    async def execute_expand_plan_item_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
        *,
        run: OrchestrationRun | None = None,
        artifact: Artifact | None = None,
        item: OrchestrationPlanItem | None = None,
    ) -> OrchestrationAction:
        if not idempotency_key:
            raise HTTPException(status_code=400, detail="Action idempotency key is required")
        request_to_store = self._canonical_expand_plan_item_request(request)
        run = run or await self._run_for_plan_action(db, run_id)
        # The accepted snapshot is the only post-acceptance plan authority.
        goal = await db.get(OrchestrationGoal, run.goal_id)
        try:
            items = await self._accepted_plan_items(db, run)
        except HTTPException as exc:
            self._block_plan_integrity(goal, run, str(exc.detail))
            raise
        artifact = await self._accepted_plan_artifact(db, run)
        item = next((candidate for candidate in items if candidate.id == request_to_store["plan_item_id"]), None)
        if item is None:
            raise HTTPException(status_code=404, detail=f"Plan item '{request_to_store['plan_item_id']}' not found")
        if item.id != request_to_store["plan_item_id"] or item.work_function != request_to_store["work_function"]:
            raise HTTPException(status_code=409, detail="Parsed plan item does not match expand request")
        expanded_items = self._json_object_or_empty(run.plan_state).get("expanded_items")
        expanded_ids = {
            entry.get("plan_item_id")
            for entry in expanded_items if isinstance(entry, Mapping) and isinstance(entry.get("plan_item_id"), str)
        } if isinstance(expanded_items, list) else set()
        discretionary_divisor = max(1, sum(candidate.id not in expanded_ids for candidate in items))
        stable_idempotency_key = self._expand_plan_item_idempotency_key(run_id, item.id)
        if idempotency_key != stable_idempotency_key:
            raise HTTPException(
                status_code=409,
                detail=f"Expand plan item idempotency key must be '{stable_idempotency_key}'",
            )

        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        delegation_request = None
        if existing is None:
            project_id = await self._project_id_for_run(db, run_id)
            delegation_request = await self._plan_item_delegation_request(
                db, run_id, project_id, artifact, item, discretionary_divisor=discretionary_divisor,
            )
            # For accepted child plans, authority must be resolved before the
            # outer expansion reservation or its gate exists.  A budget wait is
            # therefore replay-safe and leaves no partially expanded lineage.
            _, agent, _ = await self._validate_delegation_targets(db, run_id, delegation_request)
            authority = await self.roadmap_pre_release_authority(
                db, run_id, agent,
                self._json_object_or_empty(delegation_request.get("orchestrator_context")),
            )
            if authority.get("status") == "pending":
                raise HTTPException(status_code=409, detail="budget_wait")
            if authority.get("status") == "rejected":
                raise HTTPException(status_code=409, detail="needs_attention")
            if authority:
                context = self._json_object_or_empty(delegation_request.get("orchestrator_context"))
                roadmap = self._json_object_or_empty(context.get("roadmap"))
                delegation_request = {
                    **delegation_request,
                    "orchestrator_context": {
                        **context,
                        "roadmap": {**roadmap, "budget_approval": authority},
                    },
                }

        action = await self.reserve_action(
            db,
            run_id=run_id,
            idempotency_key=idempotency_key,
            action_type="expand_plan_item",
            request=existing.request if existing is not None else request_to_store,
            decision_id=decision_id,
        )
        if action.status != "reserved":
            return action
        stored_request = action.request or {}

        project_id = await self._project_id_for_run(db, run_id)
        goal = await db.get(OrchestrationGoal, run.goal_id)
        if goal is None:
            raise HTTPException(status_code=404, detail="Orchestration goal not found")
        if len(self._declared_success_criterion_keys(goal)) == 1 and not item.success_criterion_keys:
            item.success_criterion_keys = [self._declared_success_criterion_keys(goal)[0]]
        if delegation_request is None:
            try:
                delegation_request = await self._plan_item_delegation_request(
                    db, run_id, project_id, artifact, item, discretionary_divisor=discretionary_divisor,
                )
            except HTTPException as exc:
                if exc.status_code in {400, 404, 409}:
                    await self._fail_reserved_action_for_current_flow(db, action, str(exc.detail))
                raise

        gate = await self._ensure_plan_item_gate(db, run.id, item)
        delegation_action = await self.execute_create_delegation_task_action(
            db,
            run_id=run.id,
            request=delegation_request,
            idempotency_key=self._plan_item_delegation_idempotency_key(run.id, item.id),
            decision_id=decision_id,
        )
        if delegation_action.status != "completed" or delegation_action.target_id is None:
            await self._fail_reserved_action_for_current_flow(db, action, "Plan item delegation did not complete")
            raise HTTPException(status_code=409, detail="Plan item delegation did not complete")

        task = await db.get(Task, delegation_action.target_id)
        if task is None:
            await self._fail_reserved_action_for_current_flow(db, action, "Delegated task not found")
            raise HTTPException(status_code=404, detail="Delegated task not found")

        self._attach_plan_item_metadata(task, artifact, item, action, gate)
        self._set_plan_item_expanded_state(run, item, action, delegation_action, task, gate)
        await self._emit_plan_item_expanded(db, project_id, goal, run, item, action, delegation_action, task, gate)
        return await self._mark_action_completed(db, action, target_type="task", target_id=task.id)

    async def execute_ask_human_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        request_to_store = existing.request if existing is not None else self._canonical_ask_human_request(request)
        action = await self.reserve_action(
            db,
            run_id=run_id,
            idempotency_key=idempotency_key,
            action_type="ask_human",
            request=request_to_store,
            decision_id=decision_id,
        )
        stored_request = action.request or {}
        question = self._required_string(stored_request.get("question"), "question")
        run = await db.get(OrchestrationRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Orchestration run not found")
        try:
            subject = self._optional_string(stored_request.get("subject")) or idempotency_key
            authority = self._optional_string(stored_request.get("authority")) or "human"
            contract_version = self._optional_string(stored_request.get("contract_version")) or f"legacy:{action.id}"
            identity = runtime_decision_identity(run_id, subject, authority, contract_version)
            existed = await db.scalar(select(OrchestrationAuthorityDecision.id).where(
                OrchestrationAuthorityDecision.runtime_identity == identity
            )) is not None
            runtime_decision = await OrchestrationAuthorityDecisionService().create_runtime_question(
                db,
                run.goal_id,
                run_id=run_id,
                subject=subject, authority=authority, contract_version=contract_version,
                continuation=self._json_object_or_empty(stored_request.get("continuation")) or {"action_type": "continue"},
                question=question,
                options=stored_request.get("options") or [{"key": "acknowledge"}],
            )
        except ValueError as exc:
            await self._fail_reserved_action_for_current_flow(db, action, str(exc))
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if action.status != "reserved":
            if action.target_type != "authority_decision" or action.target_id != runtime_decision.id:
                action.target_type, action.target_id = "authority_decision", runtime_decision.id
                await db.flush()
            return action
        if existed or runtime_decision.status != "pending":
            return await self._mark_action_completed(
                db, action, target_type="authority_decision", target_id=runtime_decision.id
            )
        project_id = await self._project_id_for_run(db, run_id)
        payload = {
            "run_id": str(run_id),
            "action_id": str(action.id),
            "decision_id": str(runtime_decision.id),
            "question": question,
            "work_function": self._optional_string(stored_request.get("work_function")),
            "required_capabilities": self._string_list(stored_request.get("required_capabilities")),
            "candidate_agent_ids": self._string_list(stored_request.get("candidate_agent_ids")),
            "gate_id": self._optional_string(stored_request.get("gate_id")),
            "reason": self._optional_string(stored_request.get("reason")),
        }
        await emit_event_once(
            db,
            project_id,
            ASK_HUMAN_EVENT_TYPE,
            payload,
            source="orchestrator",
            dedup_key=f"{ASK_HUMAN_EVENT_TYPE}:runtime:{runtime_decision.runtime_identity}",
        )
        return await self._mark_action_completed(
            db, action, target_type="authority_decision", target_id=runtime_decision.id
        )

    async def execute_noop_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        """Record an explicit completed wait instead of silently dropping noop."""
        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        request_to_store = existing.request if existing is not None else self._canonical_noop_request(request)
        action = await self.reserve_action(
            db,
            run_id=run_id,
            idempotency_key=idempotency_key,
            action_type="noop",
            request=request_to_store,
            decision_id=decision_id,
        )
        if action.status != "reserved":
            return action

        project_id = await self._project_id_for_run(db, run_id)
        await emit_event_once(
            db,
            project_id,
            NOOP_WAIT_EVENT_TYPE,
            {"run_id": str(run_id), "action_id": str(action.id), "reason": action.request.get("reason")},
            source="orchestrator",
            dedup_key=f"{NOOP_WAIT_EVENT_TYPE}:action:{action.id}",
        )
        return await self._mark_action_completed(db, action, target_type="wait", target_id=None)

    async def execute_request_human_decision_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        request_to_store = existing.request if existing is not None else dict(request)
        action = await self.reserve_action(
            db,
            run_id=run_id,
            idempotency_key=idempotency_key,
            action_type="request_human_decision",
            request=request_to_store,
            decision_id=decision_id,
        )
        if action.status != "reserved":
            return action

        stored = action.request or {}
        run = await db.get(OrchestrationRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Orchestration run not found")
        title = await self._require_stored_string(db, action, stored.get("title"), "title")
        question = await self._require_stored_string(db, action, stored.get("question"), "question")
        try:
            pending = await OrchestrationAuthorityDecisionService().create_pending(
                db,
                run.goal_id,
                decision_key=f"authority_interview:{action.id}",
                title=title,
                question=question,
                authority="human",
                options=stored.get("options") or [],
                context=self._optional_string(stored.get("context")),
                recommendation=self._optional_string(stored.get("recommendation")),
                consequences=self._optional_string(stored.get("consequences")),
                run_id=run_id,
            )
        except ValueError as exc:
            await self._fail_reserved_action_for_current_flow(db, action, str(exc))
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return await self._mark_action_completed(db, action, target_type="authority_decision", target_id=pending.id)

    async def execute_request_manager_decision_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        request_to_store = existing.request if existing is not None else dict(request)
        action = await self.reserve_action(
            db,
            run_id=run_id,
            idempotency_key=idempotency_key,
            action_type="request_manager_decision",
            request=request_to_store,
            decision_id=decision_id,
        )
        if action.status != "reserved":
            return action

        stored = action.request or {}
        run = await db.get(OrchestrationRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Orchestration run not found")
        goal = await db.get(OrchestrationGoal, run.goal_id)
        if goal is None:
            raise HTTPException(status_code=404, detail="Orchestration goal not found")
        if goal.authority_model != "agent_manager" or goal.manager_agent_id is None:
            error = "no active agent manager is selected for this goal"
            await self._fail_reserved_action_for_current_flow(db, action, error)
            raise HTTPException(status_code=409, detail=error)
        manager = await db.get(Agent, goal.manager_agent_id)
        if manager is None or not manager.is_active:
            error = "selected manager agent is inactive"
            await self._fail_reserved_action_for_current_flow(db, action, error)
            raise HTTPException(status_code=409, detail=error)

        title = await self._require_stored_string(db, action, stored.get("title"), "title")
        question = await self._require_stored_string(db, action, stored.get("question"), "question")
        try:
            pending = await OrchestrationAuthorityDecisionService().create_pending(
                db,
                goal.id,
                decision_key=f"authority_interview:{action.id}",
                title=title,
                question=question,
                authority="manager",
                authority_agent_id=goal.manager_agent_id,
                options=stored.get("options") or [],
                context=self._optional_string(stored.get("context")),
                recommendation=self._optional_string(stored.get("recommendation")),
                consequences=self._optional_string(stored.get("consequences")),
                run_id=run_id,
            )
        except ValueError as exc:
            await self._fail_reserved_action_for_current_flow(db, action, str(exc))
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return await self._mark_action_completed(db, action, target_type="authority_decision", target_id=pending.id)

    async def execute_record_authority_decision_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        request_to_store = existing.request if existing is not None else dict(request)
        action = await self.reserve_action(
            db,
            run_id=run_id,
            idempotency_key=idempotency_key,
            action_type="record_authority_decision",
            request=request_to_store,
            decision_id=decision_id,
        )
        if action.status != "reserved":
            return action

        stored = action.request or {}
        authority_decision_id = await self._require_stored_uuid(db, action, stored.get("decision_id"), "decision_id")
        decision = await db.get(OrchestrationAuthorityDecision, authority_decision_id)
        run = await db.get(OrchestrationRun, run_id)
        if decision is None or run is None or decision.goal_id != run.goal_id:
            error = "authority decision not found for this run's goal"
            await self._fail_reserved_action_for_current_flow(db, action, error)
            raise HTTPException(status_code=404, detail=error)

        if decision.related_action_id is None:
            error = "decision has no linked delegation task to validate the report against"
            await self._fail_reserved_action_for_current_flow(db, action, error)
            raise HTTPException(status_code=409, detail=error)
        delegation_action = await db.get(OrchestrationAction, decision.related_action_id)
        if delegation_action is None or delegation_action.target_type != "task":
            error = "decision's linked delegation task could not be found"
            await self._fail_reserved_action_for_current_flow(db, action, error)
            raise HTTPException(status_code=409, detail=error)

        artifact_id = await self._require_stored_uuid(db, action, stored.get("artifact_id"), "artifact_id")
        artifact = await db.get(Artifact, artifact_id)
        if (
            artifact is None
            or artifact.artifact_type != "decision_report"
            or artifact.linked_task_id is None
            or artifact.linked_task_id != delegation_action.target_id
            or artifact.created_by_agent != decision.authority_agent_id
        ):
            error = "decision_report artifact is missing, not created by the awaiting agent, or not linked to this decision's own delegation task"
            await self._fail_reserved_action_for_current_flow(db, action, error)
            raise HTTPException(status_code=409, detail=error)

        metadata = self._json_object_or_empty(artifact.metadata_)
        selected_option = await self._require_stored_string(db, action, metadata.get("selected_option"), "selected_option")
        reason = self._optional_string(metadata.get("reason"))
        try:
            await OrchestrationAuthorityDecisionService().answer_decision(
                db,
                decision,
                selected_option=selected_option,
                reason=reason,
                decided_by_agent_id=decision.authority_agent_id,
            )
        except ValueError as exc:
            await self._fail_reserved_action_for_current_flow(db, action, str(exc))
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return await self._mark_action_completed(db, action, target_type="authority_decision", target_id=decision.id)

    async def execute_cancel_pending_decision_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        request_to_store = existing.request if existing is not None else dict(request)
        action = await self.reserve_action(
            db,
            run_id=run_id,
            idempotency_key=idempotency_key,
            action_type="cancel_pending_decision",
            request=request_to_store,
            decision_id=decision_id,
        )
        if action.status != "reserved":
            return action

        stored = action.request or {}
        authority_decision_id = await self._require_stored_uuid(db, action, stored.get("decision_id"), "decision_id")
        decision = await db.get(OrchestrationAuthorityDecision, authority_decision_id)
        run = await db.get(OrchestrationRun, run_id)
        if decision is None or run is None or decision.goal_id != run.goal_id:
            error = "authority decision not found for this run's goal"
            await self._fail_reserved_action_for_current_flow(db, action, error)
            raise HTTPException(status_code=404, detail=error)

        reason = await self._require_stored_string(db, action, stored.get("reason"), "reason")
        try:
            await OrchestrationAuthorityDecisionService().cancel_decision(db, decision, reason=reason)
        except ValueError as exc:
            await self._fail_reserved_action_for_current_flow(db, action, str(exc))
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return await self._mark_action_completed(db, action, target_type="authority_decision", target_id=decision.id)

    async def execute_record_warning_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        request_to_store = (
            existing.request if existing is not None else self._canonical_record_warning_request(request)
        )
        action = await self.reserve_action(
            db,
            run_id=run_id,
            idempotency_key=idempotency_key,
            action_type="record_warning",
            request=request_to_store,
            decision_id=decision_id,
        )
        if action.status != "reserved":
            return action

        stored = action.request or {}
        run = await db.get(OrchestrationRun, run_id)
        if run is None:
            error = "Orchestration run not found"
            await self._fail_reserved_action_for_current_flow(db, action, error)
            raise HTTPException(status_code=404, detail=error)
        warning_type = await self._require_stored_string(db, action, stored.get("warning_type"), "warning_type")
        severity = await self._require_stored_string(db, action, stored.get("severity"), "severity")
        message = await self._require_stored_string(db, action, stored.get("message"), "message")
        try:
            async with self._lock_goal_for_baseline_transition(db, run.goal_id):
                warning = await OrchestrationWarningService().create_warning(
                    db,
                    run.goal_id,
                    warning_type=warning_type,
                    severity=severity,
                    message=message,
                    run_id=self._optional_uuid(stored.get("run_id"), "run_id") or run_id,
                    source_process_run_id=self._optional_uuid(
                        stored.get("source_process_run_id"), "source_process_run_id"
                    ),
                    related_gate_id=self._optional_uuid(stored.get("related_gate_id"), "related_gate_id"),
                    related_action_id=self._optional_uuid(stored.get("related_action_id"), "related_action_id"),
                    related_agent_id=self._optional_uuid(stored.get("related_agent_id"), "related_agent_id"),
                )
        except ValueError as exc:
            await self._fail_reserved_action_for_current_flow(db, action, str(exc))
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except HTTPException as exc:
            await self._fail_reserved_action_for_current_flow(db, action, str(exc.detail))
            raise

        return await self._mark_action_completed(db, action, target_type="warning", target_id=warning.id)

    async def execute_acknowledge_warning_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        request_to_store = existing.request if existing is not None else dict(request)
        action = await self.reserve_action(
            db,
            run_id=run_id,
            idempotency_key=idempotency_key,
            action_type="acknowledge_warning",
            request=request_to_store,
            decision_id=decision_id,
        )
        if action.status != "reserved":
            return action

        stored = action.request or {}
        warning_id = await self._require_stored_uuid(db, action, stored.get("warning_id"), "warning_id")
        await self._require_stored_string(db, action, stored.get("reason"), "reason")
        decided_by_user_id = await self._require_stored_uuid(
            db, action, stored.get("decided_by_user_id"), "decided_by_user_id"
        )
        warning = await db.get(OrchestrationWarning, warning_id)
        run = await db.get(OrchestrationRun, run_id)
        if warning is None or run is None or warning.goal_id != run.goal_id:
            error = "warning not found for this run's goal"
            await self._fail_reserved_action_for_current_flow(db, action, error)
            raise HTTPException(status_code=404, detail=error)
        acknowledging_user = await db.get(User, decided_by_user_id)
        if acknowledging_user is None or not acknowledging_user.is_active:
            error = "active acknowledging user not found"
            await self._fail_reserved_action_for_current_flow(db, action, error)
            raise HTTPException(status_code=404, detail=error)
        try:
            await OrchestrationWarningService().acknowledge_warning(
                db, warning, acknowledged_by=f"human:{decided_by_user_id}"
            )
        except ValueError as exc:
            await self._fail_reserved_action_for_current_flow(db, action, str(exc))
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return await self._mark_action_completed(db, action, target_type="warning", target_id=warning.id)

    async def execute_resolve_warning_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        request_to_store = existing.request if existing is not None else dict(request)
        action = await self.reserve_action(
            db,
            run_id=run_id,
            idempotency_key=idempotency_key,
            action_type="resolve_warning",
            request=request_to_store,
            decision_id=decision_id,
        )
        if action.status != "reserved":
            return action

        stored = action.request or {}
        warning_id = await self._require_stored_uuid(db, action, stored.get("warning_id"), "warning_id")
        reason = await self._require_stored_string(db, action, stored.get("reason"), "reason")
        warning = await db.get(OrchestrationWarning, warning_id)
        run = await db.get(OrchestrationRun, run_id)
        if warning is None or run is None or warning.goal_id != run.goal_id:
            error = "warning not found for this run's goal"
            await self._fail_reserved_action_for_current_flow(db, action, error)
            raise HTTPException(status_code=404, detail=error)
        try:
            async with self._lock_goal_for_baseline_transition(db, warning.goal_id):
                await OrchestrationWarningService().resolve_warning(
                    db, warning, resolved_by="orchestrator", reason=reason
                )
        except ValueError as exc:
            await self._fail_reserved_action_for_current_flow(db, action, str(exc))
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return await self._mark_action_completed(db, action, target_type="warning", target_id=warning.id)

    async def _agent_decision_report(
        self, db: AsyncSession, decision: OrchestrationAuthorityDecision
    ) -> Artifact | None:
        if decision.related_action_id is None:
            return None
        action = await db.get(OrchestrationAction, decision.related_action_id)
        if action is None or action.status != "completed" or action.target_type != "task":
            return None
        result = await db.execute(
            select(Artifact)
            .where(
                Artifact.artifact_type == "decision_report",
                Artifact.linked_task_id == action.target_id,
                Artifact.created_by_agent == decision.authority_agent_id,
            )
            .order_by(Artifact.created_at.desc(), Artifact.id.desc())
        )
        return result.scalars().first()

    async def _escalate_authority_decision(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        decision: OrchestrationAuthorityDecision,
        *,
        reason: str,
    ) -> None:
        authority_service = OrchestrationAuthorityDecisionService()
        await authority_service.cancel_decision(db, decision, reason=reason)
        await OrchestrationWarningService().create_warning(
            db,
            goal.id,
            warning_type="agent_authority_decision_escalated",
            severity="warning",
            message=f"Decision '{decision.title}' could not be answered by its agent and was escalated to human: {reason}",
            run_id=run.id,
            source_process_run_id=decision.source_process_run_id,
            related_agent_id=decision.authority_agent_id,
            related_authority_decision_id=decision.id,
        )
        # Reusing the same decision_key is safe: cancel_decision just made it
        # terminal, and the partial unique index only enforces uniqueness
        # among *pending* rows (spec 6.3) -- a fresh pending row under the
        # same key is exactly what lets the original owning process (if any)
        # find it on its next advance() call, unchanged.
        await authority_service.create_pending(
            db,
            goal.id,
            decision_key=decision.decision_key,
            title=decision.title,
            question=decision.question,
            authority="human",
            options=decision.options,
            context=decision.context,
            recommendation=decision.recommendation,
            consequences=decision.consequences,
            run_id=run.id,
            source_process_run_id=decision.source_process_run_id,
        )

    async def _sync_agent_authority_decisions(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
    ) -> dict:
        """Cross-cutting D (spec 8.3.1, 10.6): deliver every pending
        non-human authority decision to its named agent as a delegation
        task, and record the agent's report once it lands. Runs every tick
        regardless of which process (or none) raised the decision -- this
        is what actually lets a manager ever answer a decision that
        `TeamHierarchyProcess` (or any future process) parks on.
        """
        authority_service = OrchestrationAuthorityDecisionService()
        pending = await authority_service.list_decisions(db, goal.id, status="pending")
        delegated = recorded = escalated = 0
        for decision in pending:
            if decision.authority == "human":
                continue
            if decision.authority_agent_id is None:
                await self._escalate_authority_decision(
                    db, goal, run, decision,
                    reason="authority_agent_id was cleared before a decision task could be created",
                )
                escalated += 1
                continue

            authority_agent = await db.get(Agent, decision.authority_agent_id)
            if authority_agent is None or not authority_agent.is_active:
                await self._escalate_authority_decision(
                    db, goal, run, decision,
                    reason="assigned authority agent is missing or inactive",
                )
                escalated += 1
                continue

            if decision.related_action_id is not None:
                action = await db.get(OrchestrationAction, decision.related_action_id)
                task = await db.get(Task, action.target_id) if action is not None else None
                if task is not None:
                    metadata = self._json_object_or_empty(task.metadata_)
                    orchestration = self._json_object_or_empty(metadata.get("orchestration"))
                    task.metadata_ = {
                        **metadata,
                        "orchestration": {
                            **orchestration,
                            "authority_decision_id": str(decision.id),
                        },
                    }

            if decision.related_action_id is None:
                idempotency_key = f"authority_decision_delegate:{decision.id}"
                existing_action = await self._existing_action_for_key(db, run.id, idempotency_key)
                try:
                    action = await self.execute_create_delegation_task_action(
                        db,
                        run_id=run.id,
                        request={
                            "action_type": "create_delegation_task",
                            "agent_id": str(decision.authority_agent_id),
                            "work_function": "decision",
                            "scope": decision.question,
                            "deliverable": "A decision report selecting one of the offered options.",
                            "inputs": (
                                [f"Context: {decision.context}"] if decision.context else []
                            ) + [f"Options: {json.dumps(decision.options, sort_keys=True)}"],
                            "success_evidence": [
                                "An artifact of type 'decision_report' linked to this task, "
                                "created by the assigned agent, with a selected_option "
                                "matching one of the offered option keys."
                            ],
                            "report_schema": {"selected_option": "str", "reason": "str"},
                        },
                        idempotency_key=idempotency_key,
                        # The decision being delivered here is frequently the exact
                        # thing a parked baseline process (team_hierarchy in
                        # "waiting_decision", etc.) is blocking on -- gating this
                        # call on baseline readiness would deadlock forever, since
                        # that readiness can never become true until this decision
                        # is answered. See Task 8.5.
                        skip_baseline_gate=True,
                    )
                except HTTPException as exc:
                    if self._is_project_not_runnable(exc):
                        raise
                    await self._escalate_authority_decision(
                        db, goal, run, decision,
                        reason=f"could not delegate decision task: {exc.detail}",
                    )
                    escalated += 1
                    continue
                task = await db.get(Task, action.target_id)
                if task is not None:
                    metadata = self._json_object_or_empty(task.metadata_)
                    orchestration = self._json_object_or_empty(metadata.get("orchestration"))
                    task.metadata_ = {
                        **metadata,
                        "orchestration": {
                            **orchestration,
                            "authority_decision_id": str(decision.id),
                        },
                    }
                await authority_service.link_delegation_action(db, decision, action_id=action.id)
                if existing_action is None:
                    delegated += 1
                continue

            report = await self._agent_decision_report(db, decision)
            if report is None:
                continue
            try:
                idempotency_key = f"record_authority_decision:{decision.id}"
                existing_action = await self._existing_action_for_key(db, run.id, idempotency_key)
                await self.execute_record_authority_decision_action(
                    db,
                    run_id=run.id,
                    request={
                        "action_type": "record_authority_decision",
                        "decision_id": str(decision.id),
                        "artifact_id": str(report.id),
                    },
                    idempotency_key=idempotency_key,
                )
                if existing_action is None:
                    recorded += 1
            except HTTPException as exc:
                await self._escalate_authority_decision(
                    db, goal, run, decision,
                    reason=f"agent decision report was invalid: {exc.detail}",
                )
                escalated += 1
        return {"delegated": delegated, "recorded": recorded, "escalated": escalated}

    async def execute_suggest_agent_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        request_to_store = (
            existing.request if existing is not None else self._canonical_suggest_agent_request(request)
        )
        action = await self.reserve_action(
            db,
            run_id=run_id,
            idempotency_key=idempotency_key,
            action_type="suggest_agent",
            request=request_to_store,
            decision_id=decision_id,
        )
        if action.status != "reserved":
            return action

        stored_request = action.request or {}
        missing_work_function = self._required_string(
            stored_request.get("missing_work_function"),
            "missing_work_function",
        )
        reason = self._required_string(stored_request.get("reason"), "reason")
        project_id = await self._project_id_for_run(db, run_id)
        role = self._optional_string(stored_request.get("suggested_role")) or self._suggested_role(
            missing_work_function
        )
        capabilities = self._string_list(stored_request.get("suggested_capabilities")) or [missing_work_function]
        # action.id doubles as the suggestion primary key on purpose: one action produces
        # at most one suggestion, and replay/race recovery can reselect it by action id.
        suggestion = await db.get(OrchestrationAgentSuggestion, action.id)
        if suggestion is None:
            suggestion = OrchestrationAgentSuggestion(
                id=action.id,
                run_id=run_id,
                missing_work_function=missing_work_function,
                reason=reason,
                suggested_role=role,
                suggested_capabilities=capabilities,
                suggested_adapter_type=self._optional_string(stored_request.get("suggested_adapter_type")) or "api",
                suggested_model=self._optional_string(stored_request.get("suggested_model")),
                suggested_system_prompt_outline=(
                    self._optional_string(stored_request.get("suggested_system_prompt_outline"))
                    or self._suggested_system_prompt_outline(role, missing_work_function, capabilities)
                ),
            )
            nested = await db.begin_nested()
            try:
                db.add(suggestion)
                await db.flush()
            except IntegrityError:
                await nested.rollback()
                session = object_session(suggestion)
                if session is not None:
                    session.expunge(suggestion)
                suggestion = await db.get(OrchestrationAgentSuggestion, action.id)
                if suggestion is None:
                    raise
            else:
                await nested.commit()

        await emit_event_once(
            db,
            project_id,
            AGENT_SUGGESTED_EVENT_TYPE,
            {
                "run_id": str(run_id),
                "action_id": str(action.id),
                "suggestion_id": str(suggestion.id),
                "missing_work_function": suggestion.missing_work_function,
                "reason": suggestion.reason,
                "suggested_role": suggestion.suggested_role,
                "suggested_capabilities": list(suggestion.suggested_capabilities),
                "suggested_adapter_type": suggestion.suggested_adapter_type,
                "suggested_model": suggestion.suggested_model,
            },
            source="orchestrator",
            dedup_key=f"{AGENT_SUGGESTED_EVENT_TYPE}:action:{action.id}",
        )
        return await self._mark_action_completed(
            db,
            action,
            target_type="agent_suggestion",
            target_id=suggestion.id,
        )

    async def execute_schedule_meeting_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        request_to_store = (
            existing.request
            if existing is not None
            else self._canonical_schedule_meeting_request(request)
        )
        action = await self.reserve_action(
            db,
            run_id=run_id,
            idempotency_key=idempotency_key,
            action_type="schedule_meeting",
            request=request_to_store,
            decision_id=decision_id,
        )
        if action.status != "reserved":
            return action

        try:
            await self._require_unmetered_control_provider_budget(db, run_id)
            stored_request = self._json_object_or_empty(action.request)
            task_id = await self._meeting_task_id(db, run_id, stored_request)
            requested_gate_id = self._optional_uuid(stored_request.get("gate_id"), "gate_id")
            project_id, task, gate = await self._coordination_task_and_gate(
                db,
                run_id,
                task_id,
                requested_gate_id,
            )
            participant_ids = [
                self._required_uuid(value, "participant_agent_ids")
                for value in self._string_list(stored_request.get("participant_agent_ids"))
            ]
            active_ids = set(
                (
                    await db.execute(
                        select(Agent.id).where(
                            Agent.id.in_(participant_ids),
                            Agent.is_active.is_(True),
                        )
                    )
                ).scalars().all()
            )
            if active_ids != set(participant_ids):
                raise HTTPException(
                    status_code=409,
                    detail="Meeting participants must be active agents",
                )
            organizer_agent_id = self._required_uuid(
                stored_request.get("organizer_agent_id"),
                "organizer_agent_id",
            )
            if organizer_agent_id not in active_ids:
                raise HTTPException(
                    status_code=409,
                    detail="Meeting organizer must be an active participant",
                )
            existing_meeting = (
                await db.execute(
                    select(Meeting)
                    .where(
                        Meeting.source_task_id == task.id,
                        Meeting.status.in_(["scheduled", "preparing", "active", "concluding"]),
                    )
                    .order_by(Meeting.created_at.asc(), Meeting.id.asc())
                )
            ).scalars().first()
            if existing_meeting is not None:
                meeting = existing_meeting
            else:
                topic = self._required_string(stored_request.get("topic"), "topic")
                meeting = await MeetingService().create_meeting(
                    db=db,
                    project_id=project_id,
                    title=topic,
                    meeting_type="decision",
                    participant_agent_ids=[str(agent_id) for agent_id in participant_ids],
                    agenda_items=[
                        {
                            "order": 1,
                            "title": topic,
                            "question": topic,
                            "max_rounds": 2,
                        }
                    ],
                    auto_start=True,
                    created_by_trigger=True,
                    trigger_reason=f"Orchestration run {run_id} gate {gate.id}",
                    source_task_id=task.id,
                    organizer_agent_id=organizer_agent_id,
                )
                await emit_event_once(
                    db,
                    project_id,
                    "meeting.scheduled",
                    {
                        "meeting_id": str(meeting.id),
                        "auto_start": meeting.auto_start,
                        "run_id": str(run_id),
                        "gate_id": str(gate.id),
                    },
                    source="orchestrator",
                    dedup_key=f"meeting.scheduled:orchestration_action:{action.id}",
                )
            await emit_event_once(
                db,
                project_id,
                MEETING_SCHEDULED_EVENT_TYPE,
                {
                    "run_id": str(run_id),
                    "action_id": str(action.id),
                    "meeting_id": str(meeting.id),
                    "task_id": str(task.id),
                    "gate_id": str(gate.id),
                },
                source="orchestrator",
                dedup_key=f"{MEETING_SCHEDULED_EVENT_TYPE}:action:{action.id}",
            )
        except Exception as exc:
            error = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
            await self._fail_reserved_action_for_current_flow(db, action, error)
            raise
        return await self._mark_action_completed(
            db,
            action,
            target_type="meeting",
            target_id=meeting.id,
        )

    async def execute_start_protocol_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        request_to_store = (
            existing.request if existing is not None else self._canonical_start_protocol_request(request)
        )
        project_id = await self._project_id_for_run(db, run_id)
        await ProjectService().lock_workspace_boundary(db, project_id)
        await ProjectService().require_runnable_project(db, project_id)

        action = await self.reserve_action(
            db,
            run_id=run_id,
            idempotency_key=idempotency_key,
            action_type="start_protocol",
            request=request_to_store,
            decision_id=decision_id,
        )
        if action.status != "reserved":
            return action

        try:
            await self._require_unmetered_control_provider_budget(db, run_id)
            stored_request = self._json_object_or_empty(action.request)
            protocol_id = self._required_uuid(stored_request.get("protocol_id"), "protocol_id")
            subject_type = self._required_string(stored_request.get("subject_type"), "subject_type")
            subject_id = self._required_uuid(stored_request.get("subject_id"), "subject_id")
            project_id, task, gate, artifact_id = await self._protocol_subject(
                db,
                run_id,
                subject_type,
                subject_id,
            )
            protocol = (
                await db.execute(
                    select(Protocol).where(
                        Protocol.id == protocol_id,
                        Protocol.is_active.is_(True),
                        or_(Protocol.project_id == project_id, Protocol.project_id.is_(None)),
                    )
                )
            ).scalar_one_or_none()
            if protocol is None:
                raise HTTPException(status_code=404, detail="Active protocol not found")

            event = BusEvent(
                id=action.id,
                project_id=project_id,
                event_type="orchestration.protocol_start_requested",
                payload={
                    "run_id": str(run_id),
                    "action_id": str(action.id),
                    "gate_id": str(gate.id),
                    "task_id": str(task.id),
                    "artifact_id": str(artifact_id) if artifact_id is not None else None,
                },
                source="orchestrator",
            )
            instance = await ProtocolEngineService().start_protocol(db, protocol, event)
            await emit_event_once(
                db,
                project_id,
                PROTOCOL_STARTED_EVENT_TYPE,
                {
                    "run_id": str(run_id),
                    "action_id": str(action.id),
                    "protocol_id": str(protocol.id),
                    "protocol_instance_id": str(instance.id),
                    "task_id": str(task.id),
                    "gate_id": str(gate.id),
                },
                source="orchestrator",
                dedup_key=f"{PROTOCOL_STARTED_EVENT_TYPE}:action:{action.id}",
            )
        except Exception as exc:
            error = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
            await self._fail_reserved_action_for_current_flow(db, action, error)
            raise
        return await self._mark_action_completed(
            db,
            action,
            target_type="protocol_instance",
            target_id=instance.id,
        )

    async def execute_retry_task_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
        exact_source_session_id: uuid.UUID | None = None,
        recovery_disposition: str | None = None,
        source_session_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        if existing is not None and existing.status != "reserved" and exact_source_session_id is not None:
            source = await db.get(Session, exact_source_session_id)
            contract = self._json_object_or_empty(existing.dispatch_contract)
            if not (
                existing.status == "completed" and source is not None
                and source.status in {"pending", "running"}
                and existing.target_type == "session" and existing.target_id == source.id
                and contract.get("owner") == "orchestration_recovery"
                and contract.get("origin") == "recovery_resume"
                and contract.get("recovery_disposition") == "resume_exact"
                and contract.get("run_id") == str(run_id)
                and contract.get("task_id") == str(source.task_id)
                and contract.get("source_session_id") == str(source.id)
                and contract.get("session_id") == str(source.id)
                and contract.get("provider_session_id") == source.provider_session_id
                and contract.get("current_runner_task_id") == source.runner_task_id
                and (source.metadata_ or {}).get("orchestration", {}).get("action_id") == str(existing.id)
            ):
                raise HTTPException(status_code=409, detail="Exact continuation replay is not provable")
        request_to_store = existing.request if existing is not None else self._canonical_retry_task_request(request)
        if exact_source_session_id is not None and existing is None:
            task_id = self._required_uuid(request_to_store.get("task_id"), "task_id")
            error = await self._exact_resume_proof(db, run_id, task_id, exact_source_session_id)
            if error:
                raise HTTPException(status_code=409, detail=error)
        action = await self.reserve_action(
            db,
            run_id=run_id,
            idempotency_key=idempotency_key,
            action_type="retry_task",
            request=request_to_store,
            decision_id=decision_id,
        )
        if action.status != "reserved":
            return action

        project_id = await self._project_id_for_run(db, run_id)
        task_id = self._required_uuid(action.request.get("task_id"), "task_id")
        task = await TaskService().get(db, project_id, task_id)
        if task is None:
            await self._fail_reserved_action_for_current_flow(db, action, "Task not found")
            raise HTTPException(status_code=404, detail="Task not found")
        if task.status != "failed":
            await self._fail_reserved_action_for_current_flow(db, action, f"Task is {task.status}")
            raise HTTPException(status_code=409, detail=f"Task is {task.status}")
        if exact_source_session_id is not None:
            proof_error = await self._exact_resume_proof(db, run_id, task.id, exact_source_session_id)
            if proof_error:
                await self._fail_reserved_action_for_current_flow(db, action, proof_error)
                return action
            session = await db.get(Session, exact_source_session_id)
            run = await db.get(OrchestrationRun, run_id)
            goal = await db.get(OrchestrationGoal, run.goal_id) if run is not None else None
            agent = await db.get(Agent, session.agent_id) if session is not None else None
            metadata = session.metadata_ if session is not None and isinstance(session.metadata_, dict) else {}
            attempt = metadata.get("attempt") if isinstance(metadata.get("attempt"), dict) else {}
            expected_launch = await self._current_exact_launch_fingerprint(db, session, task, agent)
            source = self._json_object_or_empty(self._json_object_or_empty(task.metadata_).get("orchestration"))
            source_action_id = source.get("action_id")
            if (session is None or run is None or goal is None or agent is None or session.task_id != task.id
                    or goal.status != "active" or run.status != "running" or session.adapter_type != "cli"
                    or (agent.config.get("cli_runtime", agent.cli_runtime or "claude_code") if agent is not None else None) != "claude_code"
                    or metadata.get("_launch_fingerprint") != expected_launch
                    or session.status != "failed" or not session.resumable or not session.provider_session_id
                    or metadata.get("token_usage_complete") is not True
                    or attempt.get("effect_state") != "started" or attempt.get("usage_complete") is not True
                    or attempt.get("provider_session_id") not in (None, session.provider_session_id)
                    or str(source.get("run_id")) != str(run_id) or not source_action_id):
                await self._fail_reserved_action_for_current_flow(db, action, "Exact continuation is not provable")
                return action
            try:
                await self._reserve_recovery_task_budget(db, run_id, action, task)
            except HTTPException as exc:
                await self._fail_reserved_action_for_current_flow(db, action, str(exc.detail))
                return action
            task_metadata_before = task.metadata_
            self._set_task_budget_action(task, action.id)
            from huddleroom.services.session_service import SessionService
            try:
                source_runner_task_id = session.runner_task_id
                resumed = await SessionService().resume(db, session.id)
            except HTTPException as exc:
                task.metadata_ = task_metadata_before
                await self._release_recovery_action_budget(db, goal, run, action, "resume_refused")
                await self._fail_reserved_action_for_current_flow(db, action, str(exc.detail))
                return action
            if action.budget_ledger:
                from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService
                try:
                    await OrchestrationBudgetService().commit_action_budget(db, goal, run, action)
                except HTTPException as exc:
                    # The after-commit runner carries the old id; fencing it makes
                    # the queued task a no-op before it reaches an adapter.
                    resumed.status = "failed"
                    resumed.resumable = True
                    resumed.runner_task_id = None
                    task.status = "failed"
                    task.metadata_ = task_metadata_before
                    await self._release_recovery_action_budget(db, goal, run, action, "budget_commit_refused")
                    await self._fail_reserved_action_for_current_flow(db, action, str(exc.detail))
                    return action
                allocation = self._json_object_or_empty(action.budget_ledger.get("allocation"))
                config = dict((resumed.metadata_ or {}).get("_run_config", {}))
                if "max_tokens" in allocation:
                    config["max_tokens"] = int(allocation["max_tokens"])
                if "max_hours" in allocation:
                    config["timeout"] = int(Decimal(str(allocation["max_hours"])) * Decimal("3600"))
                resumed.metadata_ = {**(resumed.metadata_ or {}), "_run_config": config}
            resumed_launch = await self._current_exact_launch_fingerprint(db, resumed, task, agent)
            resumed.metadata_ = {
                **(resumed.metadata_ or {}), "_recovery_exact_launch_fingerprint": resumed_launch,
            }
            resumed.metadata_ = {**resumed.metadata_, "orchestration": {"action_id": str(action.id)}}
            task.status = "in_progress"
            action.dispatch_contract = {
                "owner": "orchestration_recovery", "origin": "recovery_resume",
                "contract_version": "recovery_resume:1", "idempotency_key": action.idempotency_key,
                "authority_basis": {"goal_status": goal.status, "run_status": run.status,
                                    "source_action_id": str(source_action_id)},
                "budget_basis": action.budget_ledger or {"mode": "unbudgeted"},
                "retry_ceiling": {"maximum": 1, "attempt": 1},
                "expected_result": "canonical_work_report", "recovery_disposition": "resume_exact",
                "goal_id": str(goal.id), "run_id": str(run.id), "action_id": str(action.id),
                "source_action_id": str(source_action_id), "source_session_id": str(session.id), "task_id": str(task.id),
                "session_id": str(resumed.id), "provider_session_id": resumed.provider_session_id,
                "source_runner_task_id": source_runner_task_id, "current_runner_task_id": resumed.runner_task_id,
                "runner_task_id": resumed.runner_task_id,
                "proof": {"launch_fingerprint": expected_launch, "attempt": attempt,
                          "proved_at": datetime.now(timezone.utc).isoformat()},
            }
            self._remember_task_recovery(run, task.id, "resume_exact", action.id)
            return await self._mark_action_completed(db, action, target_type="session", target_id=resumed.id)

        source_orchestration = self._json_object_or_empty(
            self._json_object_or_empty(task.metadata_).get("orchestration")
        )
        source_session = await db.get(Session, source_session_id) if source_session_id is not None else None
        await self._reserve_recovery_task_budget(db, run_id, action, task)
        self._set_task_budget_action(task, action.id)

        try:
            _, session_id = await TaskService().run(
                db,
                project_id,
                task.id,
                timeout=action.request.get("timeout"),
                max_tokens=action.request.get("max_tokens"),
            )
        except HTTPException as exc:
            from huddleroom.services.session_service import SessionClaimAttention, SessionService
            if isinstance(exc, SessionClaimAttention):
                await SessionService.persist_claim_attention(db, exc)
                await self._fail_reserved_action_for_current_flow(db, action, str(exc.detail))
                return action
            await self._fail_reserved_action_for_current_flow(db, action, str(exc.detail))
            raise
        run = await db.get(OrchestrationRun, run_id)
        if run is not None:
            self._remember_task_recovery(run, task.id, "retry_task", action.id)
        goal = await db.get(OrchestrationGoal, run.goal_id) if run is not None else None
        resumed = await db.get(Session, session_id)
        action.dispatch_contract = {
            "owner": "orchestration_recovery", "origin": "recovery_retry",
            "contract_version": "recovery_retry:1", "idempotency_key": action.idempotency_key,
            "recovery_disposition": recovery_disposition or "retry_safe",
            "authority_basis": {"goal_status": goal.status if goal else None, "run_status": run.status if run else None,
                                "source_action_id": source_orchestration.get("action_id")},
            "budget_basis": action.budget_ledger or {"mode": "unbudgeted"},
            "expected_result": "canonical_work_report", "goal_id": str(goal.id) if goal else None,
            "run_id": str(run.id) if run else str(run_id), "action_id": str(action.id),
            "source_action_id": source_orchestration.get("action_id"), "task_id": str(task.id),
            "source_session_id": str(source_session.id) if source_session else None,
            "source_runner_task_id": source_session.runner_task_id if source_session else None,
            "source_provider_session_id": source_session.provider_session_id if source_session else None,
            "session_id": str(resumed.id) if resumed else str(session_id),
            "current_runner_task_id": resumed.runner_task_id if resumed else None,
            "runner_task_id": resumed.runner_task_id if resumed else None,
            "provider_session_id": resumed.provider_session_id if resumed else None,
        }
        await emit_event_once(
            db,
            project_id,
            RECOVERY_TASK_RETRIED_EVENT_TYPE,
            {
                "run_id": str(run_id),
                "action_id": str(action.id),
                "task_id": str(task.id),
                "session_id": str(session_id),
            },
            source="orchestrator",
            dedup_key=f"{RECOVERY_TASK_RETRIED_EVENT_TYPE}:action:{action.id}",
        )
        return await self._mark_action_completed(db, action, target_type="session", target_id=session_id)

    async def execute_reassign_task_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        request_to_store = existing.request if existing is not None else self._canonical_reassign_task_request(request)
        action = await self.reserve_action(
            db,
            run_id=run_id,
            idempotency_key=idempotency_key,
            action_type="reassign_task",
            request=request_to_store,
            decision_id=decision_id,
        )
        if action.status != "reserved":
            return action

        project_id = await self._project_id_for_run(db, run_id)
        task_id = self._required_uuid(action.request.get("task_id"), "task_id")
        agent_id = self._required_uuid(action.request.get("agent_id"), "agent_id")
        task = await TaskService().get(db, project_id, task_id)
        agent = await db.get(Agent, agent_id)
        if task is None:
            await self._fail_reserved_action_for_current_flow(db, action, "Task not found")
            raise HTTPException(status_code=404, detail="Task not found")
        if agent is None or not agent.is_active:
            await self._fail_reserved_action_for_current_flow(db, action, "Agent not found")
            raise HTTPException(status_code=404, detail="Agent not found")
        if task.status != "failed":
            await self._fail_reserved_action_for_current_flow(db, action, f"Task is {task.status}")
            raise HTTPException(status_code=409, detail=f"Task is {task.status}")

        await self._reserve_recovery_task_budget(db, run_id, action, task)
        self._set_task_budget_action(task, action.id)

        previous_agent_id = task.assigned_to
        task.assigned_to = agent.id
        await db.flush()
        try:
            _, session_id = await TaskService().run(db, project_id, task.id)
        except Exception as exc:
            task.assigned_to = previous_agent_id
            await db.flush()
            error = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
            from huddleroom.services.session_service import SessionClaimAttention, SessionService
            if isinstance(exc, SessionClaimAttention):
                await SessionService.persist_claim_attention(db, exc)
            await self._fail_reserved_action_for_current_flow(db, action, error)
            if isinstance(exc, SessionClaimAttention):
                return action
            raise
        run = await db.get(OrchestrationRun, run_id)
        if run is not None:
            self._remember_task_recovery(run, task.id, "reassign_task", action.id)
        await emit_event_once(
            db,
            project_id,
            RECOVERY_TASK_REASSIGNED_EVENT_TYPE,
            {
                "run_id": str(run_id),
                "action_id": str(action.id),
                "task_id": str(task.id),
                "previous_agent_id": str(previous_agent_id) if previous_agent_id else None,
                "agent_id": str(agent.id),
                "session_id": str(session_id),
            },
            source="orchestrator",
            dedup_key=f"{RECOVERY_TASK_REASSIGNED_EVENT_TYPE}:action:{action.id}",
        )
        return await self._mark_action_completed(db, action, target_type="session", target_id=session_id)

    async def _reserve_recovery_task_budget(
        self, db: AsyncSession, run_id: uuid.UUID, action: OrchestrationAction, task: Task,
    ) -> None:
        """Charge a retry to its own durable action, never the original attempt."""
        from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService

        run = await db.get(OrchestrationRun, run_id)
        goal = await db.get(OrchestrationGoal, run.goal_id) if run is not None else None
        contract = self._json_object_or_empty(
            self._json_object_or_empty(task.metadata_).get("orchestration_contract")
        )
        amounts = self._json_object_or_empty(contract.get("budget"))
        amounts = self._json_object_or_empty(amounts.get("caps")) or amounts
        if goal is None or not amounts or action.budget_ledger:
            return
        await OrchestrationBudgetService().reserve_action_budget(
            db, goal, run, action, amounts, enforceable=True,
        )

    async def _release_recovery_action_budget(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun,
        action: OrchestrationAction, reason: str,
    ) -> None:
        """Settle an undispatched reservation at zero; never strand retry capacity."""
        if not action.budget_ledger:
            return
        from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService
        await OrchestrationBudgetService().settle_action_budget(
            db, goal, run, action, {}, measurement_complete=True,
            observation_id=f"action:{action.id}:{reason}",
        )

    async def _exact_resume_proof(
        self, db: AsyncSession, run_id: uuid.UUID, task_id: uuid.UUID, session_id: uuid.UUID,
    ) -> str | None:
        """Facts required before reserving an exact provider continuation."""
        run = await db.get(OrchestrationRun, run_id, populate_existing=True)
        goal = await db.get(OrchestrationGoal, run.goal_id, populate_existing=True) if run is not None else None
        task = await db.get(Task, task_id, populate_existing=True)
        session = await db.get(Session, session_id, populate_existing=True)
        agent = await db.get(Agent, session.agent_id, populate_existing=True) if session is not None else None
        metadata = session.metadata_ if session is not None and isinstance(session.metadata_, dict) else {}
        attempt = metadata.get("attempt") if isinstance(metadata.get("attempt"), dict) else {}
        runtime = agent.config.get("cli_runtime", agent.cli_runtime or "claude_code") if agent is not None else None
        expected_launch = await self._current_exact_launch_fingerprint(db, session, task, agent)
        source = self._json_object_or_empty(self._json_object_or_empty(task.metadata_ if task else {}).get("orchestration"))
        source_action = await db.get(OrchestrationAction, uuid.UUID(str(source["action_id"])), populate_existing=True) if source.get("action_id") else None
        counters = (attempt.get("token_count_in"), attempt.get("token_count_out"))
        if (run is None or goal is None or task is None or session is None or agent is None
                or goal.status != "active" or run.status != "running" or session.task_id != task.id
                or task.status != "failed" or task.assigned_to != session.agent_id or session.adapter_type != "cli"
                or runtime != "claude_code" or metadata.get("_launch_fingerprint") != expected_launch
                or session.status != "failed" or not session.resumable or not session.provider_session_id
                or metadata.get("token_usage_complete") is not True or attempt.get("effect_state") != "started"
                or attempt.get("usage_complete") is not True or attempt.get("result_status") != "failed"
                or attempt.get("provider_session_id") != session.provider_session_id
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 for value in counters)
                or str(source.get("run_id")) != str(run_id) or not source_action
                or source_action.run_id != run_id or source_action.status != "completed"):
            return "Exact continuation is not provable"
        return None

    @staticmethod
    async def _current_exact_launch_fingerprint(
        db: AsyncSession, session: Session | None, task: Task | None, agent: Agent | None,
    ) -> str | None:
        if session is None or task is None or agent is None:
            return None
        roadmap = (session.input_context or {}).get("orchestrator_context", {}).get("roadmap", {})
        try:
            if isinstance(roadmap, dict) and roadmap.get("mutates_shared_state") and roadmap.get("staging_boundary"):
                workspace = await ProjectService().require_frozen_roadmap_workspace(
                    db, session.project_id, roadmap["staging_boundary"], (session.metadata_ or {}).get("_roadmap_workspace", ""),
                )
            else:
                workspace = await ProjectService().require_runnable_project(db, session.project_id)
        except HTTPException:
            return None
        project = await db.get(Project, session.project_id, populate_existing=True)
        if project is None:
            return None
        from huddleroom.adapters.cli_adapter import CliAdapter
        return CliAdapter.launch_fingerprint(session, task, agent, project, workspace)

    def _set_task_budget_action(self, task: Task, action_id: uuid.UUID) -> None:
        metadata = self._json_object_or_empty(task.metadata_)
        task.metadata_ = {
            **metadata,
            "orchestration": {**self._json_object_or_empty(metadata.get("orchestration")), "action_id": str(action_id)},
        }

    async def _require_unmetered_control_provider_budget(self, db: AsyncSession, run_id: uuid.UUID) -> None:
        """Meeting/protocol providers cannot report the durable measurements a cap requires."""
        from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService

        run = await db.get(OrchestrationRun, run_id)
        goal = await db.get(OrchestrationGoal, run.goal_id) if run is not None else None
        if goal is not None and run is not None and (await OrchestrationBudgetService().snapshot_for_run(db, goal, run))["caps"]:
            raise HTTPException(status_code=409, detail="Budgeted control provider requires measured session telemetry")

    async def execute_request_verification_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        request_to_store = (
            existing.request if existing is not None else self._canonical_request_verification_request(request)
        )
        action = await self.reserve_action(
            db,
            run_id=run_id,
            idempotency_key=idempotency_key,
            action_type="request_verification",
            request=request_to_store,
            decision_id=decision_id,
        )
        if action.status != "reserved":
            return action

        gate_id = self._required_uuid(action.request.get("gate_id"), "gate_id")
        work_function = self._required_string(action.request.get("work_function"), "work_function")
        gate = await db.get(OrchestrationGate, gate_id)
        if gate is None or gate.run_id != run_id:
            await self._fail_reserved_action_for_current_flow(db, action, "Orchestration gate not found")
            raise HTTPException(status_code=404, detail="Orchestration gate not found")

        project_id = await self._project_id_for_run(db, run_id)
        source_task = await self._source_task_for_gate(db, project_id, gate)
        run = await db.get(OrchestrationRun, run_id)
        goal = await db.get(OrchestrationGoal, run.goal_id) if run is not None else None
        if gate.gate_type == "roadmap_plan_approval" and run is not None:
            source_task = await self._planning_task_for_gate(db, run, gate)
        if goal is not None and goal.goal_type in {"outcome", "roadmap"} and gate.gate_type == PLAN_ITEM_GATE_TYPE:
            source_task = await self._plan_item_producer_task(db, gate)
        if gate.gate_type != "roadmap_integration" and (source_task is None or source_task.assigned_to is None):
            await self._fail_reserved_action_for_current_flow(db, action, "Verification source producer is missing")
            raise HTTPException(status_code=409, detail="Verification source producer is missing")
        action.request = {
            **action.request,
            "source_task_id": str(source_task.id) if source_task is not None else None,
            "producer_agent_id": str(source_task.assigned_to) if source_task is not None else None,
            "roadmap_version_id": self._json_object_or_empty(gate.required_evidence).get("roadmap_version_id"),
        }
        excluded_agent_ids = await self._producer_agent_ids_for_gate(db, gate)
        source_contract = self._json_object_or_empty(
            self._json_object_or_empty(source_task.metadata_ if source_task else {}).get("orchestration_contract")
        )
        source_context = self._json_object_or_empty(source_contract.get("orchestrator_context"))
        verification_context = source_context or (deepcopy(goal.orchestrator_context or {}) if goal else {})
        allowed_agent_ids = self._roadmap_team_agent_ids(
            verification_context.get("team")
        )
        fit = await self._best_recovery_agent(
            db,
            project_id,
            work_function,
            required_capabilities=[work_function],
            exclude_agent_ids=excluded_agent_ids,
            allowed_agent_ids=allowed_agent_ids,
        )
        if fit is None:
            await self._fail_reserved_action_for_current_flow(db, action, "No strong verification agent fit")
            raise HTTPException(status_code=409, detail="No strong verification agent fit")

        try:
            from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService

            verification_budget = await OrchestrationBudgetService().protected_action_allocation(db, goal, run)
            delegation_action = await self.execute_create_delegation_task_action(
                db,
                run_id=run_id,
                request={
                    "action_type": "create_delegation_task",
                    "agent_id": str(fit.agent_id),
                    "work_function": work_function,
                    "scope": (
                        "Verify the latest work for the stale orchestration gate. "
                        "Report fresh evidence only; do not edit artifacts."
                    ),
                    "inputs": [
                        f"Gate: {gate.id}",
                        f"Failure reason: {gate.failure_reason}",
                    ],
                    "deliverable": "Fresh verification evidence for the stale gate.",
                    "forbidden_work": [
                        "Do not edit project artifacts.",
                        "Do not mark orchestration gates complete.",
                    ],
                    "success_evidence": [
                        "A task report or session output that is newer than the work being verified.",
                    ],
                    "budget": verification_budget,
                    "report_schema": {
                        "status": "done|blocked|failed",
                        "verdict": "accepted|rejected",
                        "evidence": "list[str]",
                        "notes": "str",
                    },
                    "parent_task_id": str(source_task.id) if source_task is not None else None,
                    "orchestrator_context": verification_context,
                },
                idempotency_key=f"run:{run_id}:kind:create_delegation_task:verify_gate:{gate.id}",
                decision_id=decision_id,
            )
        except Exception as exc:
            error = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
            await self._fail_reserved_action_for_current_flow(db, action, error)
            raise
        if delegation_action.status != "completed" or delegation_action.target_id is None:
            await self._fail_reserved_action_for_current_flow(db, action, "Verification delegation did not complete")
            raise HTTPException(status_code=409, detail="Verification delegation did not complete")

        task = await db.get(Task, delegation_action.target_id)
        if task is not None:
            action.request = {
                **action.request,
                "verifier_agent_id": str(task.assigned_to),
            }
            metadata = self._json_object_or_empty(task.metadata_)
            orchestration = self._json_object_or_empty(metadata.get("orchestration"))
            metadata["orchestration"] = {
                **orchestration,
                "plan_item_gate_id": str(gate.id),
                "recovery_kind": "stale_gate_verification",
                "verification_action_id": str(action.id),
                "verification_gate_id": str(gate.id),
                "source_task_id": str(source_task.id) if source_task is not None else None,
                "producer_agent_id": str(source_task.assigned_to) if source_task is not None else None,
                "verifier_agent_id": str(task.assigned_to),
            }
            task.metadata_ = metadata
        await emit_event_once(
            db,
            project_id,
            RECOVERY_VERIFICATION_REQUESTED_EVENT_TYPE,
            {
                "run_id": str(run_id),
                "action_id": str(action.id),
                "gate_id": str(gate.id),
                "task_id": str(delegation_action.target_id),
            },
            source="orchestrator",
            dedup_key=f"{RECOVERY_VERIFICATION_REQUESTED_EVENT_TYPE}:action:{action.id}",
        )
        return await self._mark_action_completed(db, action, target_type="task", target_id=delegation_action.target_id)

    async def execute_pause_run_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        request_to_store = existing.request if existing is not None else self._canonical_pause_run_request(request)
        action = await self.reserve_action(
            db,
            run_id=run_id,
            idempotency_key=idempotency_key,
            action_type="pause_run",
            request=request_to_store,
            decision_id=decision_id,
        )
        if action.status != "reserved":
            return action

        run = await db.get(OrchestrationRun, run_id)
        if run is None:
            await self._fail_reserved_action_for_current_flow(db, action, "Orchestration run not found")
            raise HTTPException(status_code=404, detail="Orchestration run not found")
        goal = await db.get(OrchestrationGoal, run.goal_id)
        if goal is None:
            await self._fail_reserved_action_for_current_flow(db, action, "Orchestration goal not found")
            raise HTTPException(status_code=404, detail="Orchestration goal not found")
        if run.status not in LLM_DECISION_RUN_STATUSES:
            await self._fail_reserved_action_for_current_flow(db, action, f"Run is {run.status}")
            raise HTTPException(status_code=409, detail=f"Run is {run.status}")

        reason = self._required_string(action.request.get("reason"), "reason")
        goal.status = "paused"
        run.status = "paused"
        self._upsert_active_blocker(
            run,
            {
                "kind": "repeated_failure",
                "reason": reason,
            },
        )
        await emit_event_once(
            db,
            goal.project_id,
            RECOVERY_RUN_PAUSED_EVENT_TYPE,
            {
                "goal_id": str(goal.id),
                "run_id": str(run.id),
                "action_id": str(action.id),
                "reason": reason,
            },
            source="orchestrator",
            dedup_key=f"{RECOVERY_RUN_PAUSED_EVENT_TYPE}:action:{action.id}",
        )
        return await self._mark_action_completed(db, action, target_type="run", target_id=run.id)

    async def handle_weak_roster_fit(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        work_function: str,
        required_capabilities: list[str] | None = None,
        reason: str | None = None,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        capabilities = self._string_list(required_capabilities)
        idempotency_key = self._weak_fit_idempotency_key(run_id, work_function, capabilities)
        existing = (
            await db.execute(
                select(OrchestrationAction).where(
                    OrchestrationAction.run_id == run_id,
                    OrchestrationAction.idempotency_key == idempotency_key,
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            return existing

        project_id = await self._project_id_for_run(db, run_id)
        fits = await OrchestrationRosterMapper().rank_agents(
            db,
            project_id,
            work_function,
            required_capabilities=capabilities,
        )
        reason_text = reason or f"No strong {work_function} roster fit exists."
        if fits and not fits[0].weak:
            raise HTTPException(status_code=409, detail="Roster fit is not weak")

        request = {
            "action_type": "suggest_agent",
            "missing_work_function": work_function,
            "reason": reason_text,
            "suggested_capabilities": capabilities or [work_function],
        }
        return await self.execute_suggest_agent_action(
            db,
            run_id=run_id,
            request=request,
            idempotency_key=idempotency_key,
            decision_id=decision_id,
        )

    def _weak_fit_idempotency_key(
        self,
        run_id: uuid.UUID,
        work_function: str,
        required_capabilities: list[str],
    ) -> str:
        return (
            f"run:{run_id}:kind:suggest_agent:work_function:{work_function}:"
            f"need:{self._stable_need_hash(work_function, required_capabilities)}"
        )

    @staticmethod
    def _stable_hash(value: Mapping[str, Any]) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    @staticmethod
    def _stable_need_hash(work_function: str, required_capabilities: list[str]) -> str:
        capabilities = {
            OrchestrationRosterMapper._normalize_token(capability)
            for capability in required_capabilities
            if OrchestrationRosterMapper._normalize_token(capability)
        }
        stable_need = {
            "work_function": OrchestrationRosterMapper._normalize_token(work_function),
            "required_capabilities": sorted(capabilities),
        }
        encoded = json.dumps(stable_need, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    async def _mark_action_completed(
        self,
        db: AsyncSession,
        action: OrchestrationAction,
        target_type: str,
        target_id: uuid.UUID | None,
    ) -> OrchestrationAction:
        action.target_type = target_type
        action.target_id = target_id
        action.status = "completed"
        action.error = None
        await db.flush()
        return action

    async def _project_id_for_run(self, db: AsyncSession, run_id: uuid.UUID) -> uuid.UUID:
        result = await db.execute(
            select(OrchestrationGoal.project_id)
            .join(OrchestrationRun, OrchestrationRun.goal_id == OrchestrationGoal.id)
            .where(OrchestrationRun.id == run_id)
        )
        project_id = result.scalar_one_or_none()
        if project_id is None:
            raise HTTPException(status_code=404, detail="Orchestration run not found")
        return project_id

    async def ensure_run_in_project(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        run_id: uuid.UUID,
    ) -> None:
        if await self._project_id_for_run(db, run_id) != project_id:
            raise HTTPException(status_code=404, detail="Orchestration run not found")

    @staticmethod
    def _required_string(value: Any, field_name: str) -> str:
        text_value = str(value).strip() if value is not None else ""
        if not text_value:
            raise HTTPException(status_code=400, detail=f"{field_name} is required")
        return text_value

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        if value is None:
            return None
        text_value = str(value).strip()
        return text_value or None

    def canonical_decision_request(
        self, action_type: str, request: Any, *, run_id: uuid.UUID | None = None
    ) -> dict[str, Any]:
        """Normalize a dispatchable decision before deriving its replay key."""
        if action_type == "record_warning":
            return self._canonical_record_warning_request(request, run_id=run_id)
        canonicalizers = {
            "noop": self._canonical_noop_request,
            "request_plan": self._canonical_plan_request,
            "request_roadmap_replan": self._canonical_roadmap_replan_request,
            "request_plan_revision": self._canonical_plan_revision_request,
            "accept_plan": self._canonical_accept_plan_request,
            "create_delegation_task": self._canonical_delegation_task_request,
            "request_verification": self._canonical_request_verification_request,
            "retry_task": self._canonical_retry_task_request,
            "reassign_task": self._canonical_reassign_task_request,
            "schedule_meeting": self._canonical_schedule_meeting_request,
            "start_protocol": self._canonical_start_protocol_request,
            "ask_human": self._canonical_ask_human_request,
            "pause_run": self._canonical_pause_run_request,
            "suggest_agent": self._canonical_suggest_agent_request,
        }
        return canonicalizers[action_type](request)

    def _canonical_noop_request(self, request: Any) -> dict[str, Any]:
        return {
            "action_type": "noop",
            "reason": self._optional_string(self._json_object_or_empty(request).get("reason")),
        }

    def _canonical_record_warning_request(
        self, request: Any, *, run_id: uuid.UUID | None = None
    ) -> dict[str, Any]:
        request_obj = self._json_object_or_empty(request)
        optional_ids = (
            "run_id", "source_process_run_id", "related_gate_id", "related_action_id", "related_agent_id",
        )
        optional_id_values = {
            field: self._optional_uuid(request_obj.get(field), field)
            for field in optional_ids
        }
        warning_run_id = optional_id_values["run_id"] or run_id
        return {
            "action_type": "record_warning",
            "warning_type": self._required_string(request_obj.get("warning_type"), "warning_type"),
            "severity": self._required_string(request_obj.get("severity"), "severity"),
            "message": self._required_string(request_obj.get("message"), "message"),
            "run_id": str(warning_run_id) if warning_run_id is not None else None,
            **{
                field: str(value) if value is not None else None
                for field, value in optional_id_values.items()
                if field != "run_id"
            },
        }

    def _canonical_delegation_task_request(self, request: Any) -> dict[str, Any]:
        request_obj = self._json_object_or_empty(request)
        parent_task_id = self._optional_uuid(request_obj.get("parent_task_id"), "parent_task_id")
        source_session_id = self._optional_uuid(
            request_obj.get("source_session_id"), "source_session_id"
        )
        context = self._json_object_or_empty(request_obj.get("orchestrator_context"))
        canonical = {
            "action_type": "create_delegation_task",
            "agent_id": str(self._required_uuid(request_obj.get("agent_id"), "agent_id")),
            "work_function": self._required_string(request_obj.get("work_function"), "work_function"),
            "scope": self._required_string(request_obj.get("scope"), "scope"),
            "inputs": self._string_list(request_obj.get("inputs")),
            "deliverable": self._required_string(request_obj.get("deliverable"), "deliverable"),
            "forbidden_work": self._string_list(request_obj.get("forbidden_work")),
            "success_evidence": self._string_list(request_obj.get("success_evidence")),
            "budget": self._json_object_or_empty(request_obj.get("budget")),
            "report_schema": self._json_object_or_empty(request_obj.get("report_schema")),
            "parent_task_id": str(parent_task_id) if parent_task_id is not None else None,
            "source_session_id": str(source_session_id) if source_session_id is not None else None,
        }
        if context:
            canonical["orchestrator_context"] = context
        return canonical

    def _follow_up_delegation_key(self, run_id: uuid.UUID, request: Mapping[str, Any]) -> str:
        parent_task_id = self._required_uuid(request.get("parent_task_id"), "parent_task_id")
        source_session_id = self._optional_uuid(request.get("source_session_id"), "source_session_id")
        question = " ".join(self._required_string(request.get("scope"), "scope").split())
        return (
            f"run:{run_id}:kind:create_delegation_task:follow_up:parent:{parent_task_id}:"
            f"report:{source_session_id or 'none'}:question:{self._stable_hash({'question': question})}"
        )

    async def _follow_up_source_session_id(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: Mapping[str, Any],
    ) -> uuid.UUID | None:
        parent_task_id = self._required_uuid(request.get("parent_task_id"), "parent_task_id")
        requested_session_id = self._optional_uuid(request.get("source_session_id"), "source_session_id")
        marker = await self._existing_action_for_key(
            db, run_id, f"run:{run_id}:kind:report_consumed:task:{parent_task_id}"
        )
        marker_session_id = self._optional_uuid(
            self._json_object_or_empty(marker.request if marker else {}).get("session_id"),
            "source_session_id",
        )
        if marker_session_id is not None:
            if requested_session_id is not None and requested_session_id != marker_session_id:
                raise HTTPException(status_code=409, detail="Follow-up source session was not the consumed report")
            return marker_session_id
        return requested_session_id

    def _canonical_plan_request(self, request: Any) -> dict[str, Any]:
        request_obj = self._json_object_or_empty(request)
        context = self._json_object_or_empty(request_obj.get("orchestrator_context"))
        canonical = {
            "action_type": "request_plan",
            "agent_id": str(self._required_uuid(request_obj.get("agent_id"), "agent_id")),
            "work_function": self._required_plan_work_function(request_obj.get("work_function")),
            "scope": self._required_string(request_obj.get("scope"), "scope"),
        }
        if context:
            canonical["orchestrator_context"] = context
        return canonical

    @staticmethod
    def _has_delegable_orchestrator_context(context: Mapping[str, Any]) -> bool:
        """Ignore empty baseline bookkeeping when dispatching ordinary plans."""
        operational = {
            key: value
            for key, value in context.items()
            if key not in {"assumptions", "merged_decision_keys"}
        }
        return any(value not in (None, "", [], {}) for value in operational.values())

    def _canonical_roadmap_replan_request(self, request: Any) -> dict[str, Any]:
        request_obj = self._json_object_or_empty(request)
        return {
            "action_type": "request_roadmap_replan",
            "agent_id": str(self._required_uuid(request_obj.get("agent_id"), "agent_id")),
            "scope": self._required_string(request_obj.get("scope"), "scope"),
            "reason": self._required_string(request_obj.get("reason"), "reason"),
        }

    async def roadmap_replan_action_key(self, db, run: OrchestrationRun) -> str:
        from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService

        goal = await db.get(OrchestrationGoal, run.goal_id)
        version = await OrchestrationRoadmapService(self).current_version(db, goal.id) if goal else None
        state = self._json_object_or_empty(run.plan_state)
        if goal is None or version is None or state.get("roadmap_version_id") != str(version.id) or state.get("roadmap_version") != version.version:
            raise HTTPException(status_code=409, detail="Roadmap replan requires an accepted current version")
        return f"run:{run.id}:kind:request_roadmap_replan:version:{version.version}"

    def _required_plan_work_function(self, value: Any) -> str:
        work_function = self._required_string(value, "work_function")
        if work_function != PLAN_WORK_FUNCTION:
            raise HTTPException(
                status_code=400,
                detail=f"request_plan work_function must be '{PLAN_WORK_FUNCTION}'",
            )
        return PLAN_WORK_FUNCTION

    def _canonical_plan_revision_request(self, request: Any) -> dict[str, Any]:
        request_obj = self._json_object_or_empty(request)
        return {
            "action_type": "request_plan_revision",
            "plan_task_id": str(self._required_uuid(request_obj.get("plan_task_id"), "plan_task_id")),
            "revision_request": self._required_string(request_obj.get("revision_request"), "revision_request"),
        }

    def _canonical_final_summary_request(
        self,
        request: Any,
        gates: list[OrchestrationGate],
        evidence: list[OrchestrationEvidence],
        goal: OrchestrationGoal | None = None,
        roadmap_version_id: str | None = None,
    ) -> dict[str, Any]:
        request_obj = self._json_object_or_empty(request)
        work_function = self._required_string(request_obj.get("work_function"), "work_function")
        if work_function != FINAL_SUMMARY_WORK_FUNCTION:
            raise HTTPException(
                status_code=400,
                detail=f"work_function must be '{FINAL_SUMMARY_WORK_FUNCTION}'",
            )
        return {
            "action_type": "request_final_summary",
            "work_function": work_function,
            "criterion_evidence_manifest": self._criterion_evidence_manifest(
                gates, evidence, goal, roadmap_version_id
            ),
        }

    def _canonical_complete_run_request(self, request: Any) -> dict[str, Any]:
        request_obj = self._json_object_or_empty(request)
        return {
            "action_type": "complete_run",
            "reason": self._required_string(request_obj.get("reason"), "reason"),
        }

    @staticmethod
    def _final_summary_request_key(run_id: uuid.UUID) -> str:
        return f"run:{run_id}:kind:request_final_summary"

    @staticmethod
    def _final_summary_delegation_key(run_id: uuid.UUID) -> str:
        return f"run:{run_id}:kind:create_delegation_task:final_summary"

    @staticmethod
    def _final_summary_replacement_key(
        run_id: uuid.UUID,
        failed_evidence_id: uuid.UUID,
    ) -> str:
        return f"run:{run_id}:kind:create_delegation_task:final_summary:replacement:{failed_evidence_id}"

    @staticmethod
    def _complete_run_key(run_id: uuid.UUID) -> str:
        return f"run:{run_id}:kind:complete_run"

    def _canonical_expand_plan_item_request(self, request: Any) -> dict[str, Any]:
        request_obj = self._json_object_or_empty(request)
        return {
            "action_type": "expand_plan_item",
            "plan_item_id": self._required_string(request_obj.get("plan_item_id"), "plan_item_id"),
            "work_function": self._required_string(request_obj.get("work_function"), "work_function"),
        }

    def _canonical_accept_plan_request(self, request: Any) -> dict[str, Any]:
        request_obj = self._json_object_or_empty(request)
        return {
            "action_type": "accept_plan",
            "plan_artifact_id": str(self._required_uuid(request_obj.get("plan_artifact_id"), "plan_artifact_id")),
        }

    async def _run_for_plan_action(self, db: AsyncSession, run_id: uuid.UUID) -> OrchestrationRun:
        run = await db.get(OrchestrationRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Orchestration run not found")
        if run.status not in ACTIVE_RUN_STATUSES:
            raise HTTPException(status_code=409, detail=f"Orchestration run is {run.status}")
        return run

    async def _planning_task_for_run(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        plan_task_id: uuid.UUID,
    ) -> Task:
        project_id = await self._project_id_for_run(db, run.id)
        task = await TaskService().get(db, project_id, plan_task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Planning task not found")

        task_metadata = self._json_object_or_empty(task.metadata_)
        orchestration = self._json_object_or_empty(task_metadata.get("orchestration"))
        if orchestration.get("run_id") != str(run.id):
            raise HTTPException(status_code=409, detail="Planning task does not belong to this run")
        if orchestration.get("work_function") != PLAN_WORK_FUNCTION:
            raise HTTPException(status_code=409, detail="Task is not a planning task")
        plan_state = self._json_object_or_empty(run.plan_state)
        pending = self._json_object_or_empty(plan_state.get("pending_replan"))
        current_plan_task_id = self._required_uuid(
            pending.get("task_id") if pending else plan_state.get("planning_task_id"), "planning_task_id"
        )
        request_action_id = self._required_uuid(
            pending.get("action_id") if pending else plan_state.get("request_action_id"), "request_action_id"
        )
        if task.id != current_plan_task_id or orchestration.get("action_id") != str(request_action_id):
            raise HTTPException(status_code=409, detail="Planning task is not the current outstanding request_plan task")
        return task

    async def _planning_task_for_gate(
        self, db: AsyncSession, run: OrchestrationRun, gate: OrchestrationGate
    ) -> Task:
        required = self._json_object_or_empty(gate.required_evidence)
        task_id = self._required_uuid(required.get("planning_task_id"), "planning_task_id")
        artifact_id = self._required_uuid(required.get("plan_artifact_id"), "plan_artifact_id")
        project_id = await self._project_id_for_run(db, run.id)
        task = await TaskService().get(db, project_id, task_id)
        artifact = await db.get(Artifact, artifact_id)
        orchestration = self._json_object_or_empty(
            self._json_object_or_empty(task.metadata_ if task else {}).get("orchestration")
        )
        if (
            task is None or artifact is None or artifact.project_id != project_id
            or artifact.linked_task_id != task.id or artifact.created_by_agent != task.assigned_to
            or orchestration.get("run_id") != str(run.id)
            or orchestration.get("work_function") != PLAN_WORK_FUNCTION
        ):
            raise HTTPException(status_code=409, detail="Planning task does not match approval gate")
        pending = self._json_object_or_empty(self._json_object_or_empty(run.plan_state).get("pending_replan"))
        if pending and (
            pending.get("task_id") != str(task.id)
            or pending.get("action_id") != orchestration.get("action_id")
            or pending.get("planner_agent_id") != str(task.assigned_to)
            or pending.get("artifact_id") != str(artifact.id)
            or pending.get("fingerprint") != gate.success_criterion_key.removeprefix("roadmap_plan:")
            or pending.get("gate_id") != str(gate.id)
        ):
            raise HTTPException(status_code=409, detail="Planning task does not match current pending Roadmap replan")
        return task

    async def _plan_artifact_for_run(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        artifact_id: uuid.UUID,
    ) -> Artifact:
        artifact = await db.get(Artifact, artifact_id)
        project_id = await self._project_id_for_run(db, run.id)
        if artifact is None or artifact.project_id != project_id:
            raise HTTPException(status_code=404, detail="Plan artifact not found")
        if artifact.artifact_type != "plan":
            raise HTTPException(status_code=409, detail="Artifact must have artifact_type 'plan'")

        plan_state = self._json_object_or_empty(run.plan_state)
        pending = self._json_object_or_empty(plan_state.get("pending_replan"))
        plan_task_id = self._required_uuid(
            pending.get("task_id") if pending else plan_state.get("planning_task_id"), "planning_task_id"
        )
        if artifact.linked_task_id != plan_task_id:
            raise HTTPException(status_code=409, detail="Plan artifact is not linked to the current planning task")
        planning_task = await self._planning_task_for_run(db, run, plan_task_id)
        if artifact.created_by_agent is None or artifact.created_by_agent != planning_task.assigned_to:
            raise HTTPException(status_code=409, detail="Plan artifact must be created by the assigned planning agent")
        return artifact

    def _plan_items_from_artifact(self, artifact: Artifact) -> list[OrchestrationPlanItem]:
        metadata = self._json_object_or_empty(artifact.metadata_)
        return self._plan_items_from_raw(metadata.get("plan_items"))

    def _plan_items_from_raw(self, raw_items: Any) -> list[OrchestrationPlanItem]:
        if not isinstance(raw_items, list) or not raw_items:
            raise HTTPException(status_code=409, detail="Accepted plan artifact must include metadata.plan_items")

        items = []
        seen_ids = set()
        for index, raw_item in enumerate(raw_items):
            try:
                item = OrchestrationPlanItem.model_validate(raw_item)
            except ValidationError as exc:
                error = exc.errors()[0]
                location = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid plan item at index {index} field {location}: {error.get('msg', 'Invalid plan item')}",
                ) from exc

            if item.id in seen_ids:
                raise HTTPException(status_code=409, detail=f"Duplicate plan item id '{item.id}'")
            seen_ids.add(item.id)
            items.append(item)

        dependencies = {
            item.id: [
                dependency.removeprefix("plan_item:")
                for dependency in item.depends_on
            ]
            for item in items
        }
        for item_id, item_dependencies in dependencies.items():
            for dependency in item_dependencies:
                if dependency not in dependencies:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Plan item '{item_id}' depends on unknown plan item '{dependency}'",
                    )
                if dependency == item_id:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Plan item '{item_id}' cannot depend on itself",
                    )

        try:
            tuple(TopologicalSorter(dependencies).static_order())
        except CycleError as exc:
            raise HTTPException(status_code=409, detail="Plan dependency cycle detected") from exc
        return items

    @staticmethod
    def _accepted_plan_fingerprint(items: list[dict[str, Any]]) -> str:
        canonical = json.dumps(items, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _accepted_plan_snapshot(self, artifact_id: uuid.UUID, items: list[OrchestrationPlanItem]) -> dict[str, Any]:
        normalized = [item.model_dump(mode="json") for item in items]
        return {
            "version": ACCEPTED_PLAN_SNAPSHOT_VERSION,
            "artifact_id": str(artifact_id),
            "items": normalized,
            "fingerprint": self._accepted_plan_fingerprint(normalized),
        }

    async def _accepted_plan_items(self, db: AsyncSession, run: OrchestrationRun) -> list[OrchestrationPlanItem]:
        plan_state = self._json_object_or_empty(run.plan_state)
        if plan_state.get("status") != "accepted":
            raise HTTPException(status_code=409, detail="Plan must be accepted before expansion")
        snapshot = self._json_object_or_empty(plan_state.get("accepted_plan_snapshot"))
        if snapshot.get("version") != ACCEPTED_PLAN_SNAPSHOT_VERSION:
            raise HTTPException(status_code=409, detail="Accepted plan snapshot is missing or unsupported")
        artifact_id = self._required_uuid(plan_state.get("accepted_artifact_id"), "accepted_artifact_id")
        snapshot_artifact_id = self._required_uuid(snapshot.get("artifact_id"), "accepted_plan_snapshot.artifact_id")
        if snapshot_artifact_id != artifact_id:
            raise HTTPException(status_code=409, detail="Accepted plan snapshot artifact does not match accepted artifact")
        artifact = await db.get(Artifact, artifact_id)
        if artifact is None or artifact.project_id != await self._project_id_for_run(db, run.id):
            raise HTTPException(status_code=409, detail="Accepted plan snapshot artifact is invalid")
        if artifact.artifact_type != "plan":
            raise HTTPException(status_code=409, detail="Accepted plan snapshot artifact is invalid")
        raw_items = snapshot.get("items")
        if not isinstance(raw_items, list):
            raise HTTPException(status_code=409, detail="Accepted plan snapshot items are invalid")
        fingerprint = snapshot.get("fingerprint")
        expected = self._accepted_plan_fingerprint(raw_items)
        if not isinstance(fingerprint, str) or not hmac.compare_digest(fingerprint, expected):
            raise HTTPException(status_code=409, detail="Accepted plan snapshot fingerprint is invalid")
        items = self._plan_items_from_raw(raw_items)
        self._validate_plan_criterion_links(await db.get(OrchestrationGoal, run.goal_id), items)
        return items

    @staticmethod
    def _declared_success_criterion_keys(goal: OrchestrationGoal | None) -> list[str]:
        if goal is None:
            raise HTTPException(status_code=404, detail="Orchestration goal not found")
        keys: list[str] = []
        for criterion in goal.success_criteria:
            key = criterion.get("key") if isinstance(criterion, Mapping) else None
            if not isinstance(key, str) or not key.strip() or key != key.strip() or key in keys:
                raise HTTPException(status_code=409, detail="Declared success criteria keys are invalid")
            keys.append(key)
        return keys

    def _validate_plan_criterion_links(
        self, goal: OrchestrationGoal | None, items: list[OrchestrationPlanItem]
    ) -> None:
        declared = self._declared_success_criterion_keys(goal)
        declared_set = set(declared)
        covered: set[str] = set()
        for item in items:
            if len(declared) == 1 and not item.success_criterion_keys:
                item.success_criterion_keys = [declared[0]]
            unknown = set(item.success_criterion_keys) - declared_set
            if unknown:
                raise HTTPException(
                    status_code=409,
                    detail=f"Plan item '{item.id}' has unknown success criterion key '{sorted(unknown)[0]}'",
                )
            covered.update(item.success_criterion_keys)
        if declared_set - covered:
            raise HTTPException(status_code=409, detail="Plan does not cover every declared success criterion")

    async def _backfill_plan_criterion_links(
        self, db: AsyncSession, run: OrchestrationRun, items: list[OrchestrationPlanItem]
    ) -> None:
        """Persist deterministic sole-criterion defaults on already-expanded legacy work."""
        for item in items:
            gate = await db.scalar(
                select(OrchestrationGate).where(
                    OrchestrationGate.run_id == run.id,
                    OrchestrationGate.success_criterion_key == self._plan_item_gate_key(item.id),
                    OrchestrationGate.gate_type == PLAN_ITEM_GATE_TYPE,
                )
            )
            if gate is None:
                continue
            required = self._json_object_or_empty(gate.required_evidence)
            if required.get("success_criterion_keys") != item.success_criterion_keys:
                gate.required_evidence = {**required, "success_criterion_keys": list(item.success_criterion_keys)}
            task = await self._plan_item_producer_task(db, gate)
            if task is None:
                continue
            metadata = self._json_object_or_empty(task.metadata_)
            orchestration = self._json_object_or_empty(metadata.get("orchestration"))
            plan_item = self._json_object_or_empty(metadata.get("orchestration_plan_item"))
            if (
                orchestration.get("success_criterion_keys") != item.success_criterion_keys
                or plan_item.get("success_criterion_keys") != item.success_criterion_keys
            ):
                task.metadata_ = {
                    **metadata,
                    "orchestration": {**orchestration, "success_criterion_keys": list(item.success_criterion_keys)},
                    "orchestration_plan_item": {
                        **plan_item, "success_criterion_keys": list(item.success_criterion_keys)
                    },
                }

    async def _accepted_plan_artifact(self, db: AsyncSession, run: OrchestrationRun) -> Artifact:
        plan_state = self._json_object_or_empty(run.plan_state)
        if plan_state.get("status") != "accepted":
            raise HTTPException(status_code=409, detail="Plan must be accepted before expansion")
        artifact_id = self._required_uuid(plan_state.get("accepted_artifact_id"), "accepted_artifact_id")
        artifact = await db.get(Artifact, artifact_id)
        if artifact is None or artifact.project_id != await self._project_id_for_run(db, run.id):
            raise HTTPException(status_code=404, detail="Accepted plan artifact not found")
        if artifact.artifact_type != "plan":
            raise HTTPException(status_code=409, detail="Accepted plan artifact is invalid")
        return artifact

    def _block_plan_integrity(self, goal: OrchestrationGoal | None, run: OrchestrationRun, reason: str) -> None:
        if goal is not None:
            self._mark_run_blocked(goal, run)
        self._upsert_active_blocker(run, {
            "kind": "plan_criterion_integrity",
            "reason": reason,
            "recommended_action": "Replace or supersede before authorization, or request human intervention.",
        })

    def _plan_item_from_artifact(
        self,
        artifact: Artifact,
        plan_item_id: str,
        work_function: str,
    ) -> OrchestrationPlanItem:
        for item in self._plan_items_from_artifact(artifact):
            if item.id != plan_item_id:
                continue
            if item.work_function != work_function:
                raise HTTPException(
                    status_code=409,
                    detail=f"Plan item '{plan_item_id}' work_function is '{item.work_function}'",
                )
            return item
        raise HTTPException(status_code=404, detail=f"Plan item '{plan_item_id}' not found")

    async def _ensure_plan_item_gate(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        item: OrchestrationPlanItem,
    ) -> OrchestrationGate:
        gate_key = self._plan_item_gate_key(item.id)
        result = await db.execute(
            select(OrchestrationGate)
            .where(
                OrchestrationGate.run_id == run_id,
                OrchestrationGate.success_criterion_key == gate_key,
                OrchestrationGate.gate_type == PLAN_ITEM_GATE_TYPE,
            )
            .order_by(OrchestrationGate.created_at.asc(), OrchestrationGate.id.asc())
            .limit(1)
        )
        gate = result.scalar_one_or_none()
        if gate is not None:
            return gate

        required_evidence = self._json_object_or_empty(item.required_evidence)
        run = await db.get(OrchestrationRun, run_id)
        goal = await db.get(OrchestrationGoal, run.goal_id) if run is not None else None
        if goal is not None and goal.goal_type == "outcome":
            required_evidence["required_source_types"] = ["task", "verification"]
            required_evidence["min_count"] = 2
            required_evidence["requires_independent_agent"] = True
        required_evidence["plan_item_id"] = item.id
        required_evidence["success_criterion_keys"] = list(item.success_criterion_keys)
        gate = OrchestrationGate(
            id=uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"rally:orchestration:gate:{run_id}:{PLAN_ITEM_GATE_TYPE}:{gate_key}",
            ),
            run_id=run_id,
            success_criterion_key=gate_key,
            gate_type=PLAN_ITEM_GATE_TYPE,
            required_evidence=required_evidence,
        )
        nested = await db.begin_nested()
        try:
            db.add(gate)
            await db.flush()
        except IntegrityError:
            await nested.rollback()
            session = object_session(gate)
            if session is not None:
                session.expunge(gate)
            result = await db.execute(
                select(OrchestrationGate)
                .where(
                    OrchestrationGate.run_id == run_id,
                    OrchestrationGate.success_criterion_key == gate_key,
                    OrchestrationGate.gate_type == PLAN_ITEM_GATE_TYPE,
                )
                .order_by(OrchestrationGate.created_at.asc(), OrchestrationGate.id.asc())
                .limit(1)
            )
            gate = result.scalar_one_or_none()
            if gate is not None:
                return gate
            raise
        await nested.commit()
        return gate

    @staticmethod
    def _plan_item_gate_key(plan_item_id: str) -> str:
        return f"plan_item:{plan_item_id}"

    async def _plan_item_delegation_request(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        project_id: uuid.UUID,
        artifact: Artifact,
        item: OrchestrationPlanItem,
        orchestrator_context: dict[str, Any] | None = None,
        *,
        discretionary_divisor: int,
    ) -> dict[str, Any]:
        run = await db.get(OrchestrationRun, run_id)
        goal = await db.get(OrchestrationGoal, run.goal_id) if run else None
        from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService, protected_quantum

        snapshot = await OrchestrationBudgetService().snapshot_for_run(db, goal, run) if goal and run else None
        budget = {
            dimension: format(
                max(Decimal("0"), Decimal(amount) - protected_quantum(
                    dimension, Decimal(snapshot["caps"][dimension])
                ))
                / discretionary_divisor,
                "f",
            )
            for dimension, amount in snapshot["remaining"].items()
        } if snapshot else {}
        request = self._canonical_delegation_task_request(
            {
                "action_type": "create_delegation_task",
                "agent_id": str(await self._plan_item_agent_id(
                    db, project_id, item,
                    allowed_agent_ids=self._roadmap_team_agent_ids(
                        self._json_object_or_empty(orchestrator_context or {}).get("team")
                    ),
                )),
                "work_function": item.work_function,
                "scope": item.scope,
                "inputs": [
                    f"Accepted plan artifact: {artifact.id}",
                    f"Plan item: {item.id}",
                    *item.inputs,
                ],
                "deliverable": item.deliverable,
                "forbidden_work": item.forbidden_work
                or [
                    "Do not expand other plan items.",
                    "Do not mark orchestration gates complete.",
                ],
                "success_evidence": item.success_evidence
                or [
                    "Complete this task and report concrete output or artifact evidence.",
                ],
                "budget": budget,
                "report_schema": {
                    "status": "done|blocked|failed",
                    "evidence": "list[str]",
                    "artifact_ids": "list[uuid]",
                    "notes": "str",
                },
                "parent_task_id": None,
                "orchestrator_context": deepcopy(orchestrator_context) if orchestrator_context is not None else (deepcopy(goal.orchestrator_context or {}) if goal else {}),
            }
        )
        await self._validate_delegation_targets(db, run_id, request)
        return request

    async def roadmap_pre_release_authority(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        agent: Agent,
        context: dict[str, Any] | None,
        *,
        adapter_override: str | None = None,
    ) -> dict[str, Any]:
        """Authorize CLI work from immutable Roadmap ownership before creating work."""
        roadmap = self._json_object_or_empty(self._json_object_or_empty(context).get("roadmap"))
        if not roadmap:
            return {}
        run = await db.get(OrchestrationRun, run_id)
        goal = await db.get(OrchestrationGoal, run.goal_id) if run else None
        if run is None or goal is None:
            raise HTTPException(status_code=409, detail="Roadmap budget lineage is invalid")
        parent = await db.get(OrchestrationGoal, goal.parent_goal_id) if goal.parent_goal_id else goal
        version_id = goal.roadmap_version_id if goal.parent_goal_id else roadmap.get("roadmap_version_id")
        item_key = goal.roadmap_item_key if goal.parent_goal_id else roadmap.get("roadmap_item_key")
        if parent is None or version_id is None or not isinstance(item_key, str):
            raise HTTPException(status_code=409, detail="Roadmap budget lineage is invalid")
        try:
            version = await db.get(OrchestrationRoadmapVersion, uuid.UUID(str(version_id)))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail="Roadmap budget lineage is invalid") from exc
        if version is None or version.run_id is None:
            raise HTTPException(status_code=409, detail="Roadmap budget lineage is invalid")
        owner_run = await db.get(OrchestrationRun, version.run_id)
        if owner_run is None or owner_run.goal_id != parent.id:
            raise HTTPException(status_code=409, detail="Roadmap budget lineage is invalid")
        caps = self._json_object_or_empty((run.budget_state or {}).get("caps")) if goal.parent_goal_id else self._json_object_or_empty((goal.budget or {}).get("caps"))
        adapter_type = adapter_override or agent.adapter_type
        unsupported = sorted(key for key in ("max_tokens",) if key in caps and adapter_type == "cli")
        if not unsupported:
            return {}
        dimensions = ",".join(unsupported)
        key = f"roadmap_budget_adapter:{version.id}:{item_key}:{agent.id}:{adapter_type}:{dimensions}"
        decision = await db.scalar(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.goal_id == parent.id,
            OrchestrationAuthorityDecision.run_id == owner_run.id,
            OrchestrationAuthorityDecision.decision_key == key,
        ).order_by(OrchestrationAuthorityDecision.created_at.desc()).limit(1))
        if decision is None:
            decision = await OrchestrationAuthorityDecisionService().create_pending(
                db, parent.id, decision_key=key, title="Approve capped CLI Roadmap work",
                question=(f"Approve CLI execution for Roadmap item '{item_key}' despite unsupported "
                          f"budget dimensions: {dimensions}?"), authority="human",
                options=[{"key": "approve"}, {"key": "reject"}], run_id=owner_run.id,
                context=(f"Immutable Roadmap version={version.id}; item={item_key}; agent={agent.id}; "
                         f"adapter={adapter_type}; unsupported_dimensions={dimensions}."),
            )
            return {"status": "pending"}
        # An existing request is still a wait, not a rejection.  In particular,
        # never create a durable rejection blocker before a human has answered.
        if decision.status == "pending":
            return {"status": "pending"}
        # Only a real answer can establish a durable rejection.  Forged or
        # terminal states are never approvals, but must not manufacture one.
        if decision.status != "answered":
            return {"status": "rejected", "decision_key": key, "unsupported_dimensions": unsupported}
        authorized_user = parent.manager_user_id or parent.created_by_user_id
        approved = (
            decision.status == "answered" and decision.authority == "human"
            and decision.selected_option == "approve" and authorized_user is not None
            and decision.decided_by_user_id == authorized_user
            and decision.decided_by_agent_id is None
        )
        if not approved:
            self._upsert_active_blocker(owner_run, {
                "kind": "budget_integrity", "scope": f"budget_authority:{key}",
                "decision_key": key, "item_key": item_key,
                "reason": "Human rejected capped CLI Roadmap work.",
            })
            await db.flush()
            return {"status": "rejected", "decision_key": key, "unsupported_dimensions": unsupported}
        return {
            "status": "approved", "decision_id": str(decision.id),
            "unsupported_dimensions": unsupported, "agent_id": str(agent.id),
            "adapter_type": adapter_type,
        }

    async def _plan_item_agent_id(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        item: OrchestrationPlanItem,
        *,
        allowed_agent_ids: set[str] | None = None,
    ) -> uuid.UUID:
        if item.agent_id is not None:
            if allowed_agent_ids is not None and str(item.agent_id) not in allowed_agent_ids:
                raise HTTPException(status_code=409, detail=f"Plan item '{item.id}' agent is outside the accepted team")
            return item.agent_id

        fits = await OrchestrationRosterMapper().rank_agents(
            db,
            project_id,
            item.work_function,
            required_capabilities=item.required_capabilities,
        )
        fits = [fit for fit in fits if allowed_agent_ids is None or str(fit.agent_id) in allowed_agent_ids]
        if not fits or fits[0].weak:
            raise HTTPException(status_code=409, detail=f"No strong roster fit for plan item '{item.id}'")
        return fits[0].agent_id

    @staticmethod
    def _expand_plan_item_idempotency_key(run_id: uuid.UUID, plan_item_id: str) -> str:
        return f"run:{run_id}:kind:expand_plan_item:plan_item:{plan_item_id}"

    @staticmethod
    def _plan_item_delegation_idempotency_key(run_id: uuid.UUID, plan_item_id: str) -> str:
        return f"run:{run_id}:kind:create_delegation_task:plan_item:{plan_item_id}"

    @staticmethod
    def _attach_plan_item_metadata(
        task: Task,
        artifact: Artifact,
        item: OrchestrationPlanItem,
        action: OrchestrationAction,
        gate: OrchestrationGate,
    ) -> None:
        metadata = OrchestrationService._json_object_or_empty(task.metadata_)
        orchestration = OrchestrationService._json_object_or_empty(metadata.get("orchestration"))
        metadata["orchestration"] = {
            **orchestration,
            "plan_item_id": item.id,
            "plan_item_gate_id": str(gate.id),
            "success_criterion_keys": list(item.success_criterion_keys),
            "expand_action_id": str(action.id),
        }
        metadata["orchestration_plan_item"] = {
            **item.model_dump(mode="json"),
            "accepted_plan_artifact_id": str(artifact.id),
        }
        task.metadata_ = metadata

    @staticmethod
    def _set_plan_item_expanded_state(
        run: OrchestrationRun,
        item: OrchestrationPlanItem,
        action: OrchestrationAction,
        delegation_action: OrchestrationAction,
        task: Task,
        gate: OrchestrationGate,
    ) -> None:
        plan_state = OrchestrationService._json_object_or_empty(run.plan_state)
        expanded_items = plan_state.get("expanded_items")
        if not isinstance(expanded_items, list):
            expanded_items = []
        entry = {
            "plan_item_id": item.id,
            "work_function": item.work_function,
            "expand_action_id": str(action.id),
            "delegation_action_id": str(delegation_action.id),
            "task_id": str(task.id),
            "gate_id": str(gate.id),
        }
        run.plan_state = {
            **plan_state,
            "expanded_items": [
                existing
                for existing in expanded_items
                if not isinstance(existing, Mapping) or existing.get("plan_item_id") != item.id
            ]
            + [entry],
        }

    async def _emit_plan_item_expanded(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        item: OrchestrationPlanItem,
        action: OrchestrationAction,
        delegation_action: OrchestrationAction,
        task: Task,
        gate: OrchestrationGate,
    ) -> None:
        await emit_event_once(
            db,
            project_id,
            PLAN_ITEM_EXPANDED_EVENT_TYPE,
            {
                "goal_id": str(goal.id),
                "run_id": str(run.id),
                "action_id": str(action.id),
                "plan_item_id": item.id,
                "work_function": item.work_function,
                "delegation_action_id": str(delegation_action.id),
                "task_id": str(task.id),
                "gate_id": str(gate.id),
            },
            source="orchestrator",
            dedup_key=f"{PLAN_ITEM_EXPANDED_EVENT_TYPE}:action:{action.id}",
        )

    @staticmethod
    def _ensure_plan_not_accepted(run: OrchestrationRun) -> None:
        plan_state = OrchestrationService._json_object_or_empty(run.plan_state)
        if plan_state.get("status") == "accepted":
            raise HTTPException(status_code=409, detail="Accepted plan cannot be revised")

    def _process_terminal(self, baseline_process: dict | None) -> bool:
        return (
            baseline_process is not None
            and baseline_process["status"] in ("completed", "skipped")
        )

    async def _baseline_processes_ready_for_goal(
        self, db: AsyncSession, goal_id: uuid.UUID, allow_heal: bool = True
    ) -> bool:
        """Check if all baseline processes are ready for a downstream transition.

        Thin wrapper over _baseline_readiness_reason: ready iff there is no reason.
        """
        goal = await db.get(OrchestrationGoal, goal_id)
        run = await self.get_active_run_for_goal(db, goal.project_id, goal.id) if goal else None
        delta = self._json_object_or_empty(
            self._json_object_or_empty(run.plan_state).get("child_delta_baseline")
        ) if run else {}
        if goal is not None and goal.parent_goal_id is not None and delta.get("status") == "accepted":
            return True
        return await self._baseline_readiness_reason(db, goal_id, allow_heal=allow_heal) is None

    async def _baseline_readiness_reason(
        self, db: AsyncSession, goal_id: uuid.UUID, allow_heal: bool = True
    ) -> str | None:
        """Return why the baseline processes are not ready, or None when ready.

        "Not ready" is more than "not terminal": a predecessor can be terminal yet
        stale (manager removed, agent roster changed since review, team changed since
        the hierarchy was accepted). Each returns a distinct, actionable reason so
        callers stop reporting a blanket "must all be terminal" message (which is a
        lie when the real cause is a stale accepted process).

        Wrapped in goal lock to ensure fresh reads and atomic heal operations (fix #3+#4).
        If allow_heal is False, reports stale answer without writing (fix #15).
        """
        async with self._lock_goal_for_baseline_transition(db, goal_id):
            process_service = OrchestrationProcessService()
            # Reload goal with populate_existing inside lock (fix #3+#4).
            goal = await db.execute(
                select(OrchestrationGoal)
                .where(OrchestrationGoal.id == goal_id)
                .execution_options(populate_existing=True)
            )
            goal = goal.scalar_one_or_none()
            if goal is None:
                return "goal not found"
            manager_selection_current = None
            for process_type in (
                "goal_definition",
                "manager_selection",
                "agent_definition_review",
                "team_hierarchy",
            ):
                # Reload current process with populate_existing inside lock (fix #3+#4).
                current = await process_service.get_current(db, goal_id, process_type)
                if current is None or current.status not in ("completed", "skipped"):
                    return f"'{process_type}' is not yet complete"
                if process_type == "manager_selection":
                    manager_selection_current = current
                # Pre-Phase-6 skipped manager_selection rows may not have backfilled
                # authority_model=no_manager. Heal them here (only if allow_heal=True),
                # as they may arrive via planning/delegation endpoints without going
                # through tick(). Re-verify status before calling handle_skip (fix #3+#4).
                if (
                    allow_heal
                    and process_type == "manager_selection"
                    and current.status == "skipped"
                    and (
                        goal.authority_model != "no_manager"
                        or goal.manager_agent_id is not None
                        or goal.manager_user_id is not None
                    )
                ):
                    await ManagerSelectionProcess().handle_skip(
                        db, goal, skipped_run=current
                    )
            # After confirming manager_selection is terminal, check if the selected
            # manager is still valid (not removed/inactive). Re-verify status (fix #3+#4).
            # This 409 safety net only covers `_stale_manager_reason` (manager
            # removed/inactive) -- the other spec 8.1 rerun triggers (roster
            # gained candidates, weight tier changed) have no equivalent
            # readiness check here and rely solely on the stale-inputs
            # suggestion to prompt a human rerun.
            if manager_selection_current is not None:
                stale_reason = await ManagerSelectionProcess()._stale_manager_reason(
                    db, goal, manager_selection_current
                )
                if stale_reason is not None and not await OrchestrationWarningService().stale_inputs_dismissed(
                    db, goal_id, process_type="manager_selection", process_run_id=manager_selection_current.id
                ):
                    return f"manager_selection is stale: {stale_reason}"
            agent_review_current = await process_service.get_current(
                db, goal_id, "agent_definition_review"
            )
            if agent_review_current is not None and agent_review_current.status == "completed":
                active_run = await self.get_active_run_for_goal(db, goal.project_id, goal.id)
                if active_run is not None:
                    coverage_fingerprint = (
                        await AgentDefinitionReviewProcess().current_coverage_fingerprint(
                            db,
                            goal,
                            active_run,
                            covered_target_ids=set(
                                agent_review_current.outputs.get("target_ids", [])
                            ),
                        )
                    )
                    if agent_review_current.outputs.get(
                        "coverage_fingerprint"
                    ) != coverage_fingerprint and not await OrchestrationWarningService().stale_inputs_dismissed(
                        db,
                        goal_id,
                        process_type="agent_definition_review",
                        process_run_id=agent_review_current.id,
                    ):
                        return (
                            "agent_definition_review is stale: the agent roster changed "
                            "since it was accepted — re-run agent_definition_review"
                        )
            hierarchy_current = await process_service.get_current(
                db, goal_id, "team_hierarchy"
            )
            if hierarchy_current is not None and hierarchy_current.status == "completed":
                active_run = await self.get_active_run_for_goal(
                    db, goal.project_id, goal.id
                )
                if active_run is not None:
                    current_fingerprint = await TeamHierarchyProcess().current_fingerprint(
                        db, goal, active_run
                    )
                    if hierarchy_current.outputs.get(
                        "fingerprint"
                    ) != current_fingerprint and not await OrchestrationWarningService().stale_inputs_dismissed(
                        db, goal_id, process_type="team_hierarchy", process_run_id=hierarchy_current.id
                    ):
                        return (
                            "team_hierarchy is stale: the team changed since it was "
                            "accepted — re-run team_hierarchy"
                        )
            return None

    async def _ensure_baseline_processes_ready(
        self, db: AsyncSession, goal_id: uuid.UUID, allow_heal: bool = True
    ) -> None:
        """Shared gate (Phase 5 Finding 1, extended through Phase 8):
        every place that can move a run's plan_state forward or create a
        delegation/finish action must call this first. Goal definition,
        manager selection, agent definition review, and team hierarchy must all
        be terminal.

        If allow_heal is False, reports stale answer without writing (fix #15).
        """
        if not await self._baseline_processes_ready_for_goal(db, goal_id, allow_heal=allow_heal):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Goal definition, manager selection, agent definition review, "
                    "and team hierarchy must complete before planning or delegation"
                ),
            )

    async def _ensure_plan_revision_allowed(self, db: AsyncSession, run: OrchestrationRun) -> None:
        await self._ensure_baseline_processes_ready(db, run.goal_id)
        self._ensure_plan_not_accepted(run)
        plan_state = OrchestrationService._json_object_or_empty(run.plan_state)
        if plan_state.get("status") == "failed":
            raise HTTPException(status_code=409, detail="Terminal plan cannot be revised for this run")

        result = await db.execute(
            select(OrchestrationGate.status)
            .where(
                OrchestrationGate.run_id == run.id,
                OrchestrationGate.success_criterion_key == PLAN_GATE_KEY,
                OrchestrationGate.gate_type == PLAN_GATE_TYPE,
            )
            .order_by(desc(OrchestrationGate.created_at), desc(OrchestrationGate.id))
            .limit(1)
        )
        gate_status = result.scalar_one_or_none()
        if gate_status == "accepted":
            raise HTTPException(status_code=409, detail="Accepted plan cannot be revised")
        if gate_status == "failed":
            raise HTTPException(status_code=409, detail="Terminal plan cannot be revised for this run")

    async def _ensure_plan_request_allowed(self, db: AsyncSession, run: OrchestrationRun, check_readiness: bool = True) -> None:
        if check_readiness:
            await self._ensure_baseline_processes_ready(db, run.goal_id)
        plan_state = OrchestrationService._json_object_or_empty(run.plan_state)
        if plan_state.get("status") in {"accepted", "failed"}:
            raise HTTPException(status_code=409, detail="Terminal plan cannot be re-requested for this run")
        result = await db.execute(
            select(OrchestrationGate.status)
            .where(
                OrchestrationGate.run_id == run.id,
                OrchestrationGate.success_criterion_key == PLAN_GATE_KEY,
                OrchestrationGate.gate_type == PLAN_GATE_TYPE,
            )
            .order_by(desc(OrchestrationGate.created_at), desc(OrchestrationGate.id))
            .limit(1)
        )
        gate_status = result.scalar_one_or_none()
        if gate_status in {"accepted", "failed"}:
            raise HTTPException(status_code=409, detail="Terminal plan cannot be re-requested for this run")

    async def _ensure_plan_acceptance_pending(self, db: AsyncSession, run: OrchestrationRun) -> None:
        await self._ensure_baseline_processes_ready(db, run.goal_id)
        plan_state = OrchestrationService._json_object_or_empty(run.plan_state)
        if plan_state.get("status") not in {"requested", "revision_requested"} and not plan_state.get("pending_replan"):
            raise HTTPException(
                status_code=409,
                detail="Plan can only be accepted from requested or revision_requested state",
            )

    def _planning_contract(
        self,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        action: OrchestrationAction,
        request: Mapping[str, Any],
    ) -> OrchestrationDelegationContract:
        from huddleroom.services.orchestration_goal_definition import goal_objective_with_clarifications

        return OrchestrationDelegationContract(
            goal_id=goal.id,
            run_id=run.id,
            action_id=action.id,
            agent_id=self._required_uuid(request.get("agent_id"), "agent_id"),
            work_function=PLAN_WORK_FUNCTION,
            scope=self._required_string(request.get("scope"), "scope"),
            inputs=[
                f"Goal objective: {goal_objective_with_clarifications(goal)}",
                f"Success criteria: {json.dumps(goal.success_criteria, sort_keys=True)}",
                f"Constraints: {json.dumps(goal.constraints, sort_keys=True)}",
            ],
            deliverable="Agent-produced implementation plan.",
            forbidden_work=[
                "Do not implement the plan.",
                "Do not edit project artifacts.",
                "Do not create tasks, meetings, protocols, rules, hooks, or automations.",
            ],
            success_evidence=[
                "A plan artifact linked to this planning task.",
                "Plan items are independently actionable and verifiable.",
            ],
            budget=self._json_object_or_empty(goal.budget),
            report_schema=(
                {
                    "artifact_id": "uuid",
                    "plan_items": (
                        "list[{item_key, unit_type, title, depends_on, mutates_shared_state, staging_boundary, "
                        "work_function, scope, deliverable, agent_id, inputs, forbidden_work, success_evidence, "
                        "required_capabilities, objective, success_criteria, constraints, allocation}]"
                    ),
                }
                if goal.goal_type == "roadmap"
                else {
                    "artifact_id": "uuid",
                    "plan_items": "list[{id, title, work_function, scope, deliverable, success_evidence, success_criterion_keys}]",
                }
            ),
            parent_task_id=None,
            orchestrator_context=deepcopy(goal.orchestrator_context or {}),
        )

    def _final_summary_delegation_request(
        self,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        agent_id: uuid.UUID,
        gates: list[OrchestrationGate],
        evidence: list[OrchestrationEvidence],
        criterion_evidence_manifest: Mapping[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        from huddleroom.services.orchestration_goal_definition import goal_objective_with_clarifications

        accepted_gate_ids = [str(gate.id) for gate in gates]
        accepted_evidence_ids = [str(item.id) for item in evidence]
        criterion_evidence_manifest = dict(
            criterion_evidence_manifest or self._criterion_evidence_manifest(gates, evidence)
        )
        return {
            "action_type": "create_delegation_task",
            "agent_id": str(agent_id),
            "work_function": FINAL_SUMMARY_WORK_FUNCTION,
            "scope": (
                "Summarize the completed orchestration run from the accepted gate manifest. "
                "Return JSON that maps every declared success criterion key exactly once to "
                "accepted evidence IDs from the manifest, and report unresolved gaps; do not "
                "change project artifacts."
            ),
            "inputs": [
                f"Goal objective: {goal_objective_with_clarifications(goal)}",
                f"Declared success criteria: {json.dumps(goal.success_criteria, sort_keys=True)}",
                f"Accepted gate IDs: {json.dumps(accepted_gate_ids)}",
                f"Accepted evidence IDs: {json.dumps(accepted_evidence_ids)}",
                f"Criterion-scoped accepted verification evidence: {json.dumps(criterion_evidence_manifest, sort_keys=True)}",
            ],
            "deliverable": "Final outcome summary grounded in the accepted gate/evidence manifest.",
            "forbidden_work": [
                "Do not edit project artifacts.",
                "Do not create implementation, tests, reviews, or validation results.",
                "Do not mark orchestration gates or runs complete.",
            ],
            "success_evidence": [
                "A non-empty agent session JSON output containing the final summary.",
                "Every declared success criterion key maps to one or more accepted evidence IDs.",
            ],
            "budget": self._json_object_or_empty(run.budget_state),
            "report_schema": {
                "summary": "str",
                "criteria": "list[{criterion_key: str, evidence_ids: list[uuid]}]",
                "unresolved_gaps": "list[str]",
            },
            "parent_task_id": None,
        }

    def _criterion_evidence_manifest(
        self,
        gates: list[OrchestrationGate],
        evidence: list[OrchestrationEvidence],
        goal: OrchestrationGoal | None = None,
        roadmap_version_id: str | None = None,
    ) -> dict[str, list[str]]:
        """Map each criterion to accepted verifier evidence from plan-item gates only."""
        manifest: dict[str, list[str]] = {}
        if goal is not None and goal.goal_type == "roadmap":
            for gate in gates:
                if (
                    gate.gate_type != "roadmap_integration" or gate.status != "accepted"
                    or self._json_object_or_empty(gate.required_evidence).get("roadmap_version_id") != roadmap_version_id
                ):
                    continue
                evidence_ids = [
                    str(item.id) for item in evidence
                    if item.gate_id == gate.id and item.source_type == "verification" and item.verdict == "accepted"
                ]
                for key in self._string_list(
                    self._json_object_or_empty(gate.required_evidence).get("success_criterion_keys")
                ):
                    manifest.setdefault(key, []).extend(evidence_ids)
            return manifest
        for gate in gates:
            if gate.gate_type != PLAN_ITEM_GATE_TYPE:
                continue
            evidence_ids = [
                str(item.id)
                for item in evidence
                if item.gate_id == gate.id and item.source_type == "verification"
            ]
            for key in self._string_list(
                self._json_object_or_empty(gate.required_evidence).get("success_criterion_keys")
            ):
                manifest.setdefault(key, []).extend(evidence_ids)
        return manifest

    def _set_final_summary_task_metadata(
        self,
        task: Task,
        gate: OrchestrationGate,
        final_summary_metadata: Mapping[str, Any],
    ) -> None:
        metadata = self._json_object_or_empty(task.metadata_)
        orchestration = self._json_object_or_empty(metadata.get("orchestration"))
        metadata["orchestration"] = {
            **orchestration,
            "gate_id": str(gate.id),
            "final_summary": True,
        }
        metadata["orchestration_final_summary"] = deepcopy(dict(final_summary_metadata))
        task.metadata_ = metadata

    async def _ensure_plan_gate(self, db: AsyncSession, run_id: uuid.UUID) -> OrchestrationGate:
        result = await db.execute(
            select(OrchestrationGate)
            .where(
                OrchestrationGate.run_id == run_id,
                OrchestrationGate.success_criterion_key == PLAN_GATE_KEY,
                OrchestrationGate.gate_type == PLAN_GATE_TYPE,
            )
            .order_by(OrchestrationGate.created_at.asc(), OrchestrationGate.id.asc())
            .limit(1)
        )
        gate = result.scalar_one_or_none()
        if gate is not None:
            return gate

        gate = OrchestrationGate(
            run_id=run_id,
            success_criterion_key=PLAN_GATE_KEY,
            gate_type=PLAN_GATE_TYPE,
            required_evidence={"required_source_types": ["artifact"], "min_count": 1},
        )
        db.add(gate)
        await db.flush()
        return gate

    async def _current_run_gates(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
    ) -> list[OrchestrationGate]:
        gates = list(
            (
                await db.execute(
                    select(OrchestrationGate)
                    .where(OrchestrationGate.run_id == run_id)
                    .order_by(OrchestrationGate.created_at.asc(), OrchestrationGate.id.asc())
                )
            ).scalars().all()
        )
        run = await db.get(OrchestrationRun, run_id)
        goal = await db.get(OrchestrationGoal, run.goal_id) if run is not None else None
        if goal is not None and goal.goal_type == "roadmap":
            from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService
            version = await OrchestrationRoadmapService(self).current_version(db, goal.id)
            integration_id = uuid.uuid5(
                uuid.NAMESPACE_URL, f"rally:roadmap-integration:{run_id}:{version.id}"
            ) if version is not None else None
            gates = [gate for gate in gates if gate.gate_type != "roadmap_integration" or (
                gate.id == integration_id
                and self._json_object_or_empty(gate.required_evidence).get("roadmap_version_id") == str(version.id)
            )]
        return gates

    async def _accepted_non_summary_manifest(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
    ) -> tuple[list[OrchestrationGate], list[OrchestrationEvidence]]:
        gates = [gate for gate in await self._current_run_gates(db, run_id)
                 if gate.gate_type != FINAL_SUMMARY_GATE_TYPE]
        if not gates or any(gate.status != "accepted" for gate in gates):
            raise HTTPException(
                status_code=409,
                detail="All non-summary gates must be accepted and have accepted evidence",
            )
        evidence = list(
            (
                await db.execute(
                    select(OrchestrationEvidence)
                    .where(
                        OrchestrationEvidence.run_id == run_id,
                        OrchestrationEvidence.gate_id.in_([gate.id for gate in gates]),
                        OrchestrationEvidence.verdict == "accepted",
                    )
                    .order_by(OrchestrationEvidence.created_at.asc(), OrchestrationEvidence.id.asc())
                )
            ).scalars().all()
        )
        evidence_gate_ids = {item.gate_id for item in evidence}
        if any(gate.id not in evidence_gate_ids for gate in gates):
            raise HTTPException(
                status_code=409,
                detail="All non-summary gates must be accepted and have accepted evidence",
            )
        return gates, evidence

    async def _closeout_preconditions_manifest(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
    ) -> dict[str, Any]:
        active_warnings = await OrchestrationWarningService().list_warnings(
            db, goal.id, active_only=True
        )
        for warning in active_warnings:
            if warning.severity in ("blocker", "hard_stop"):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Active {warning.severity} warning "
                        f"'{warning.warning_type}' must be resolved before completion"
                    ),
                )
            if warning.acknowledged_at is None:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Active warning '{warning.warning_type}' must be acknowledged "
                        "or resolved before completion"
                    ),
                )
        gates = await self._current_run_gates(db, run.id)
        final_gates = [gate for gate in gates if gate.gate_type == FINAL_SUMMARY_GATE_TYPE]
        if len(final_gates) != 1 or any(gate.status != "accepted" for gate in gates):
            raise HTTPException(status_code=409, detail="All orchestration gates must be accepted")
        evidence = list(
            (
                await db.execute(
                    select(OrchestrationEvidence)
                    .where(
                        OrchestrationEvidence.run_id == run.id,
                        OrchestrationEvidence.gate_id.in_([gate.id for gate in gates]),
                        OrchestrationEvidence.verdict == "accepted",
                    )
                    .order_by(OrchestrationEvidence.created_at.asc(), OrchestrationEvidence.id.asc())
                )
            ).scalars().all()
        )
        evidence_by_gate = {
            gate.id: [item for item in evidence if item.gate_id == gate.id]
            for gate in gates
        }
        if any(not evidence_by_gate[gate.id] for gate in gates):
            raise HTTPException(status_code=409, detail="Accepted gate evidence is missing")
        from huddleroom.services.orchestration_budget_service import (
            BudgetMeasurementError, OrchestrationBudgetService,
        )
        try:
            budget_snapshot = await OrchestrationBudgetService().snapshot_for_run(db, goal, run)
        except BudgetMeasurementError as exc:
            raise HTTPException(status_code=409, detail="Orchestration budget measurement is incomplete") from exc
        if self._budget_is_exhausted(run) or any(
            Decimal(value) < 0 for value in budget_snapshot["remaining"].values()
        ):
            raise HTTPException(status_code=409, detail="Orchestration budget is exceeded")
        final_gate = final_gates[0]
        final_evidence = next(
            (
                item
                for item in evidence_by_gate[final_gate.id]
                if item.source_type == "session" and item.source_id is not None
            ),
            None,
        )
        if final_evidence is None:
            raise HTTPException(status_code=409, detail="Final summary evidence is missing")
        summary_payload, summary_failure = await self._validated_final_summary_payload(
            db,
            final_gate,
            [final_evidence],
        )
        if summary_failure is not None or summary_payload is None:
            raise HTTPException(
                status_code=409,
                detail=summary_failure or "Final summary evidence is invalid",
            )
        session = await db.get(Session, final_evidence.source_id)
        if session is None or session.task_id is None:
            raise HTTPException(status_code=409, detail="Final summary session is missing")
        non_summary_gates = [gate for gate in gates if gate.id != final_gate.id]
        accepted_non_summary = [
            {
                "gate_id": str(gate.id),
                "success_criterion_key": gate.success_criterion_key,
                "gate_type": gate.gate_type,
                "accepted_evidence_ids": [str(item.id) for item in evidence_by_gate[gate.id]],
            }
            for gate in non_summary_gates
        ]
        return {
            "declared_success_criteria": deepcopy(goal.success_criteria),
            "criterion_evidence": deepcopy(summary_payload["criteria"]),
            "accepted_non_summary_gates": accepted_non_summary,
            "final_summary": {
                "gate_id": str(final_gate.id),
                "evidence_id": str(final_evidence.id),
                "task_id": str(session.task_id),
                "session_id": str(session.id),
                "agent_id": str(session.agent_id),
            },
            "warning_disposition": [
                {
                    "warning_id": str(warning.id),
                    "type": warning.warning_type,
                    "severity": warning.severity,
                    "acknowledged_by": warning.acknowledged_by,
                }
                for warning in active_warnings
            ],
            "overridden_gates": [
                {
                    "gate_id": str(item.gate_id),
                    "evidence_id": str(item.id),
                    "reason": self._json_object_or_empty(item.evidence_metadata).get("reason"),
                    "user_id": self._json_object_or_empty(item.evidence_metadata).get("user_id"),
                }
                for item in evidence
                if item.source_type == "human_override" and item.verdict == "accepted"
            ],
        }

    async def _completion_manifest(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
    ) -> dict[str, Any]:
        manifest = await self._closeout_preconditions_manifest(db, goal, run)
        closeout = await GoalCloseoutProcess().completion_authorization(db, goal, run)
        return {
            **manifest,
            "goal_closeout": closeout,
            "gates": {"closeout_completed": closeout["completion_authorized"]},
        }

    async def _run_ready_for_completion(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
    ) -> bool:
        if goal.goal_type == "roadmap":
            from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService
            if not await OrchestrationRoadmapService(self).integration_gate_accepted(db, goal, run):
                return False
        try:
            await self._completion_manifest(db, goal, run)
        except HTTPException as exc:
            if exc.status_code == 409:
                return False
            raise
        return True

    async def _run_ready_for_final_summary_request(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
    ) -> bool:
        if goal.goal_type == "roadmap":
            from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService
            if not await OrchestrationRoadmapService(self).integration_gate_accepted(db, goal, run):
                return False
        final_gate_id = await db.scalar(
            select(OrchestrationGate.id)
            .where(
                OrchestrationGate.run_id == run.id,
                OrchestrationGate.gate_type == FINAL_SUMMARY_GATE_TYPE,
            )
            .limit(1)
        )
        if final_gate_id is not None:
            return False
        try:
            gates, evidence = await self._accepted_non_summary_manifest(db, run.id)
        except HTTPException as exc:
            if exc.status_code == 409:
                return False
            raise
        version = await OrchestrationRoadmapService(self).current_version(db, goal.id) if goal.goal_type == "roadmap" else None
        manifest = self._criterion_evidence_manifest(gates, evidence, goal, str(version.id) if version else None)
        return all(manifest.get(key) for key in self._declared_success_criterion_keys(goal))

    async def _ensure_final_summary_gate(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
    ) -> OrchestrationGate:
        gate = (
            await db.execute(
                select(OrchestrationGate)
                .where(
                    OrchestrationGate.run_id == run_id,
                    OrchestrationGate.success_criterion_key == FINAL_SUMMARY_GATE_KEY,
                    OrchestrationGate.gate_type == FINAL_SUMMARY_GATE_TYPE,
                )
                .order_by(OrchestrationGate.created_at.asc(), OrchestrationGate.id.asc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if gate is not None:
            return gate
        gate = OrchestrationGate(
            id=uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"rally:orchestration:gate:{run_id}:{FINAL_SUMMARY_GATE_TYPE}",
            ),
            run_id=run_id,
            success_criterion_key=FINAL_SUMMARY_GATE_KEY,
            gate_type=FINAL_SUMMARY_GATE_TYPE,
            required_evidence={"required_source_types": ["session"], "min_count": 1},
        )
        nested = await db.begin_nested()
        try:
            db.add(gate)
            await db.flush()
        except IntegrityError:
            await nested.rollback()
            session = object_session(gate)
            if session is not None:
                session.expunge(gate)
            gate = (
                await db.execute(
                    select(OrchestrationGate)
                    .where(
                        OrchestrationGate.run_id == run_id,
                        OrchestrationGate.success_criterion_key == FINAL_SUMMARY_GATE_KEY,
                        OrchestrationGate.gate_type == FINAL_SUMMARY_GATE_TYPE,
                    )
                    .order_by(OrchestrationGate.created_at.asc(), OrchestrationGate.id.asc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if gate is None:
                raise
            return gate
        await nested.commit()
        return gate

    async def _cancel_superseded_planning_task(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        action: OrchestrationAction,
    ) -> None:
        plan_state = self._json_object_or_empty(run.plan_state)
        if plan_state.get("status") not in {"requested", "revision_requested"}:
            return
        if plan_state.get("request_action_id") == str(action.id):
            return

        plan_task_id = self._optional_uuid(plan_state.get("planning_task_id"), "planning_task_id")
        if plan_task_id is None:
            return

        project_id = await self._project_id_for_run(db, run.id)
        planning_task = await TaskService().get(db, project_id, plan_task_id)
        if planning_task is None:
            return

        session_result = await db.execute(
            select(Session)
            .where(
                Session.task_id == planning_task.id,
                Session.status.in_(("pending", "running")),
            )
            .order_by(Session.created_at.desc(), Session.id.desc())
            .limit(1)
        )
        active_session = session_result.scalar_one_or_none()
        if active_session is not None:
            await SessionService().cancel(db, active_session.id)

        if "cancelled" not in Task.VALID_TRANSITIONS.get(planning_task.status, set()):
            return

        await TaskService().cancel(db, project_id, planning_task.id)

    async def _accept_plan_gate(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        artifact: Artifact,
        action: OrchestrationAction,
    ) -> OrchestrationGate:
        gate = await self._ensure_plan_gate(db, run.id)
        gate.status = "accepted"
        gate.failure_reason = None
        gate.accepted_at = gate.accepted_at or _utcnow()
        gate.failed_at = None

        result = await db.execute(
            select(OrchestrationEvidence).where(
                OrchestrationEvidence.run_id == run.id,
                OrchestrationEvidence.gate_id == gate.id,
                OrchestrationEvidence.source_type == "artifact",
                OrchestrationEvidence.source_id == artifact.id,
                OrchestrationEvidence.observed_event_id.is_(None),
            )
        )
        evidence = result.scalar_one_or_none()
        if evidence is None:
            db.add(
                OrchestrationEvidence(
                    run_id=run.id,
                    gate_id=gate.id,
                    source_type="artifact",
                    source_id=artifact.id,
                    observed_event_id=None,
                    producer_agent_id=artifact.created_by_agent,
                    verdict="accepted",
                    evidence_metadata={"action_id": str(action.id), "gate_type": PLAN_GATE_TYPE},
                )
            )
        await db.flush()
        return gate

    @staticmethod
    def _set_plan_requested_state(
        run: OrchestrationRun,
        action: OrchestrationAction,
        task: Task,
        gate: OrchestrationGate,
    ) -> None:
        plan_state = OrchestrationService._json_object_or_empty(run.plan_state)
        preserved_state = {
            key: value
            for key, value in plan_state.items()
            if key not in {
                "status",
                "planning_task_id",
                "request_action_id",
                "work_function",
                "plan_gate_id",
                "revision_requests",
                "accepted_artifact_id",
                "accept_action_id",
            }
        }
        revision_requests = []
        if plan_state.get("request_action_id") == str(action.id):
            existing_revision_requests = plan_state.get("revision_requests")
            if isinstance(existing_revision_requests, list):
                revision_requests = existing_revision_requests
        run.plan_state = {
            **preserved_state,
            "status": "requested",
            "planning_task_id": str(task.id),
            "request_action_id": str(action.id),
            "work_function": PLAN_WORK_FUNCTION,
            "plan_gate_id": str(gate.id),
            "revision_requests": revision_requests,
        }

    @staticmethod
    def _append_plan_revision(
        run: OrchestrationRun,
        action: OrchestrationAction,
        plan_task_id: uuid.UUID,
        revision_request: str,
    ) -> None:
        plan_state = OrchestrationService._json_object_or_empty(run.plan_state)
        revision_requests = plan_state.get("revision_requests")
        if not isinstance(revision_requests, list):
            revision_requests = []
        action_id = str(action.id)
        if not any(
            isinstance(item, Mapping) and item.get("action_id") == action_id
            for item in revision_requests
        ):
            revision_requests = [
                *revision_requests,
                {
                    "action_id": action_id,
                    "plan_task_id": str(plan_task_id),
                    "revision_request": revision_request,
                },
            ]

        run.plan_state = {
            **plan_state,
            "status": "revision_requested",
            "planning_task_id": str(plan_task_id),
            "revision_requests": revision_requests,
        }

    @staticmethod
    def _set_plan_revision_required(run: OrchestrationRun, reason: str) -> None:
        run.plan_state = {
            **OrchestrationService._json_object_or_empty(run.plan_state),
            "status": "revision_required",
            "revision_reason": reason,
        }

    async def _emit_plan_requested(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        action: OrchestrationAction,
        task: Task,
        agent_id: uuid.UUID,
        gate: OrchestrationGate,
    ) -> None:
        await emit_event_once(
            db,
            project_id,
            PLAN_REQUESTED_EVENT_TYPE,
            {
                "goal_id": str(goal.id),
                "run_id": str(run.id),
                "action_id": str(action.id),
                "task_id": str(task.id),
                "agent_id": str(agent_id),
                "gate_id": str(gate.id),
                "work_function": PLAN_WORK_FUNCTION,
            },
            source="orchestrator",
            dedup_key=f"{PLAN_REQUESTED_EVENT_TYPE}:action:{action.id}",
        )

    def _required_uuid(self, value: Any, field_name: str) -> uuid.UUID:
        uuid_value = self._optional_uuid(value, field_name)
        if uuid_value is None:
            raise HTTPException(status_code=400, detail=f"{field_name} is required")
        return uuid_value

    async def _require_stored_string(
        self, db: AsyncSession, action: OrchestrationAction, value: Any, field_name: str
    ) -> str:
        try:
            return self._required_string(value, field_name)
        except HTTPException as exc:
            await self._fail_reserved_action_for_current_flow(db, action, str(exc))
            raise

    async def _require_stored_uuid(
        self, db: AsyncSession, action: OrchestrationAction, value: Any, field_name: str
    ) -> uuid.UUID:
        try:
            return self._required_uuid(value, field_name)
        except HTTPException as exc:
            await self._fail_reserved_action_for_current_flow(db, action, str(exc))
            raise

    @staticmethod
    def _optional_uuid(value: Any, field_name: str) -> uuid.UUID | None:
        text_value = str(value).strip() if value is not None else ""
        if not text_value:
            return None
        try:
            return uuid.UUID(text_value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"{field_name} must be a valid UUID") from exc

    @staticmethod
    def _delegation_task_title(work_function: str, deliverable: str) -> str:
        title = f"{work_function.replace('_', ' ').title()}: {deliverable.strip()}"
        return shorten(title, width=80, placeholder="...")

    @staticmethod
    def _delegation_task_description(goal: OrchestrationGoal, contract: OrchestrationDelegationContract) -> str:
        from huddleroom.services.orchestration_goal_definition import goal_objective_with_clarifications

        sections = [
            f"Objective: {goal_objective_with_clarifications(goal)}",
            f"Scope: {contract.scope}",
            f"Deliverable: {contract.deliverable}",
        ]
        if contract.inputs:
            sections.append("Inputs:\n" + "\n".join(f"- {item}" for item in contract.inputs))
        if contract.forbidden_work:
            sections.append("Forbidden work:\n" + "\n".join(f"- {item}" for item in contract.forbidden_work))
        if contract.success_evidence:
            sections.append("Success evidence:\n" + "\n".join(f"- {item}" for item in contract.success_evidence))
        roadmap = OrchestrationService._json_object_or_empty(contract.orchestrator_context).get("roadmap")
        if isinstance(roadmap, dict):
            sections.append("Roadmap execution policy:\n" + json.dumps(roadmap, sort_keys=True))
        return "\n\n".join(sections)

    async def _emit_delegation_task_created(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        action: OrchestrationAction,
        task: Task,
    ) -> None:
        stored_request = action.request or {}
        task_metadata = self._json_object_or_empty(task.metadata_)
        contract = self._json_object_or_empty(task_metadata.get("orchestration_contract"))
        await emit_event_once(
            db,
            project_id,
            DELEGATION_TASK_CREATED_EVENT_TYPE,
            {
                "goal_id": self._optional_string(contract.get("goal_id")),
                "run_id": str(action.run_id),
                "action_id": str(action.id),
                "task_id": str(task.id),
                "agent_id": self._optional_string(contract.get("agent_id"))
                or self._optional_string(stored_request.get("agent_id")),
                "work_function": self._optional_string(contract.get("work_function"))
                or self._optional_string(stored_request.get("work_function")),
            },
            source="orchestrator",
            dedup_key=f"{DELEGATION_TASK_CREATED_EVENT_TYPE}:action:{action.id}",
        )

    def _canonical_start_protocol_request(self, request: Any) -> dict[str, Any]:
        request_obj = self._json_object_or_empty(request)
        subject_type = self._required_string(request_obj.get("subject_type"), "subject_type")
        if subject_type not in {"task", "artifact"}:
            raise HTTPException(status_code=400, detail="subject_type must be 'task' or 'artifact'")
        return {
            "action_type": "start_protocol",
            "protocol_id": str(self._required_uuid(request_obj.get("protocol_id"), "protocol_id")),
            "subject_type": subject_type,
            "subject_id": str(self._required_uuid(request_obj.get("subject_id"), "subject_id")),
        }

    def _canonical_schedule_meeting_request(self, request: Any) -> dict[str, Any]:
        request_obj = self._json_object_or_empty(request)
        participant_ids: list[str] = []
        for value in self._string_list(request_obj.get("participant_agent_ids")):
            participant_id = str(self._required_uuid(value, "participant_agent_ids"))
            if participant_id not in participant_ids:
                participant_ids.append(participant_id)
        if not participant_ids:
            raise HTTPException(status_code=400, detail="participant_agent_ids is required")
        task_id = self._optional_uuid(request_obj.get("task_id"), "task_id")
        gate_id = self._optional_uuid(request_obj.get("gate_id"), "gate_id")
        organizer_agent_id = self._optional_uuid(
            request_obj.get("organizer_agent_id"),
            "organizer_agent_id",
        )
        return {
            "action_type": "schedule_meeting",
            "topic": self._required_string(request_obj.get("topic"), "topic"),
            "participant_agent_ids": participant_ids,
            "task_id": str(task_id) if task_id is not None else None,
            "gate_id": str(gate_id) if gate_id is not None else None,
            "organizer_agent_id": (
                str(organizer_agent_id) if organizer_agent_id is not None else participant_ids[0]
            ),
        }

    async def _coordination_task_and_gate(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        task_id: uuid.UUID,
        gate_id: uuid.UUID | None,
    ) -> tuple[uuid.UUID, Task, OrchestrationGate]:
        run = await db.get(OrchestrationRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Orchestration run not found")
        if run.status not in LLM_DECISION_RUN_STATUSES:
            raise HTTPException(status_code=409, detail=f"Orchestration run is {run.status}")
        project_id = await self._project_id_for_run(db, run_id)
        task = await TaskService().get(db, project_id, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Orchestration task not found")
        gate_ref = await self._task_gate_for_run(db, run_id, task)
        if gate_ref is None:
            raise HTTPException(status_code=409, detail="Task is not linked to this orchestration run")
        gate = gate_ref[0]
        if gate_id is not None and gate.id != gate_id:
            raise HTTPException(status_code=409, detail="Task is not linked to the requested gate")
        return project_id, task, gate

    async def _protocol_subject(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        subject_type: str,
        subject_id: uuid.UUID,
    ) -> tuple[uuid.UUID, Task, OrchestrationGate, uuid.UUID | None]:
        if subject_type == "task":
            project_id, task, gate = await self._coordination_task_and_gate(
                db,
                run_id,
                subject_id,
                None,
            )
            return project_id, task, gate, None

        project_id = await self._project_id_for_run(db, run_id)
        artifact = await db.get(Artifact, subject_id)
        if artifact is None or artifact.project_id != project_id:
            raise HTTPException(status_code=404, detail="Protocol subject artifact not found")
        if artifact.linked_task_id is None:
            raise HTTPException(
                status_code=409,
                detail="Protocol subject artifact must link to an orchestration task",
            )
        project_id, task, gate = await self._coordination_task_and_gate(
            db,
            run_id,
            artifact.linked_task_id,
            None,
        )
        return project_id, task, gate, artifact.id

    async def _meeting_task_id(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        stored_request: Mapping[str, Any],
    ) -> uuid.UUID:
        requested = self._optional_uuid(stored_request.get("task_id"), "task_id")
        if requested is not None:
            return requested
        run = await db.get(OrchestrationRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Orchestration run not found")
        blocker_task_ids = {
            task_id
            for blocker in (run.active_blockers or [])
            if (task_id := self._optional_uuid(blocker.get("task_id"), "task_id")) is not None
        }
        if len(blocker_task_ids) != 1:
            raise HTTPException(
                status_code=400,
                detail="task_id is required unless the run has exactly one task blocker",
            )
        return blocker_task_ids.pop()

    def _canonical_ask_human_request(self, request: Any) -> dict[str, Any]:
        request_obj = self._json_object_or_empty(request)
        canonical = {
            "action_type": "ask_human",
            "question": self._required_string(request_obj.get("question"), "question"),
            "work_function": self._optional_string(request_obj.get("work_function")),
            "required_capabilities": self._string_list(request_obj.get("required_capabilities")),
            "candidate_agent_ids": self._string_list(request_obj.get("candidate_agent_ids")),
            "gate_id": self._optional_string(request_obj.get("gate_id")),
            "reason": self._optional_string(request_obj.get("reason")),
        }
        for key in ("subject", "authority", "contract_version"):
            if value := self._optional_string(request_obj.get(key)):
                canonical[key] = value
        if isinstance(request_obj.get("continuation"), dict):
            canonical["continuation"] = self._json_object_or_empty(request_obj["continuation"])
        if isinstance(request_obj.get("options"), list):
            canonical["options"] = request_obj["options"]
        return canonical

    def _canonical_suggest_agent_request(self, request: Any) -> dict[str, Any]:
        request_obj = self._json_object_or_empty(request)
        return {
            "action_type": "suggest_agent",
            "missing_work_function": self._required_string(
                request_obj.get("missing_work_function"),
                "missing_work_function",
            ),
            "reason": self._required_string(request_obj.get("reason"), "reason"),
            "suggested_role": self._optional_string(request_obj.get("suggested_role")),
            "suggested_capabilities": self._string_list(request_obj.get("suggested_capabilities")),
            "suggested_adapter_type": self._optional_string(request_obj.get("suggested_adapter_type")),
            "suggested_model": self._optional_string(request_obj.get("suggested_model")),
            "suggested_system_prompt_outline": self._optional_string(
                request_obj.get("suggested_system_prompt_outline")
            ),
        }

    def _canonical_retry_task_request(self, request: Any) -> dict[str, Any]:
        request_obj = self._json_object_or_empty(request)
        canonical = {
            "action_type": "retry_task",
            "task_id": str(self._required_uuid(request_obj.get("task_id"), "task_id")),
        }
        for key in ("timeout", "max_tokens"):
            if request_obj.get(key) is not None:
                value = request_obj[key]
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise HTTPException(status_code=400, detail=f"{key} must be a positive integer")
                canonical[key] = value
        return canonical

    async def consume_terminal_task_output(
        self, db: AsyncSession, run: OrchestrationRun, task: Task, expected_session_id: uuid.UUID | None,
    ) -> tuple[uuid.UUID | None, str | None]:
        key = f"run:{run.id}:kind:report_consumed:task:{task.id}"
        marker = await self._existing_action_for_key(db, run.id, key)
        if marker is not None:
            session_id = self._optional_uuid(self._json_object_or_empty(marker.request).get("session_id"), "session_id")
            if session_id != expected_session_id:
                raise HTTPException(status_code=409, detail="Consumed task output lineage is invalid")
            if session_id is None:
                metadata = self._json_object_or_empty(task.metadata_)
                return None, self._optional_string(self._json_object_or_empty(metadata.get("orchestration")).get("terminal_output"))
            session = await db.get(Session, session_id)
            if session is None or session.task_id != task.id or session.status != "completed":
                raise HTTPException(status_code=409, detail="Consumed task output lineage is invalid")
            return session.id, session.output
        if task.status != "done":
            return None, None
        session = await db.get(Session, expected_session_id) if expected_session_id else None
        if expected_session_id is not None:
            if session is None:
                raise HTTPException(status_code=409, detail="Terminal task session is not canonical")
            if session.task_id != task.id or session.status != "completed":
                raise HTTPException(status_code=409, detail="Terminal task session is not canonical")
            await self.reserve_action(db, run.id, key, "report_consumed", {
                "task_id": str(task.id), "session_id": str(session.id),
            })
            return session.id, session.output
        metadata = self._json_object_or_empty(task.metadata_)
        output = self._optional_string(self._json_object_or_empty(metadata.get("orchestration")).get("terminal_output"))
        await self.reserve_action(db, run.id, key, "report_consumed", {
            "task_id": str(task.id), "session_id": None,
        })
        return None, output

    async def consume_canonical_report(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        task: Task,
        expected_session_id: uuid.UUID | None = None,
        event_id: uuid.UUID | None = None,
    ) -> tuple[uuid.UUID | None, "WorkReport | None"]:
        """Consume the one terminal report selected by task/session lineage."""
        from huddleroom.services.orchestration_work_report import (
            WorkReport,
            curate_report_memory,
            persisted_report_validation,
            validate_work_report,
            work_report_payload,
        )
        from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService

        key = f"run:{run.id}:kind:report_consumed:task:{task.id}"
        marker = await self._existing_action_for_key(db, run.id, key)
        if marker is not None and expected_session_id is None:
            expected_session_id = self._optional_uuid(
                self._json_object_or_empty(marker.request).get("session_id"), "session_id"
            )
        if marker is not None and marker.status == "completed":
            session_id = self._optional_uuid(
                self._json_object_or_empty(marker.request).get("session_id"), "session_id"
            )
            if session_id != expected_session_id:
                raise HTTPException(status_code=409, detail="Consumed task output lineage is invalid")
            if session_id is not None:
                canonical_session_id = await db.scalar(select(Session.id).where(
                    Session.id == session_id, Session.task_id == task.id, Session.status == "completed",
                ))
                if canonical_session_id is None:
                    raise HTTPException(status_code=409, detail="Consumed task output lineage is invalid")
            validation = persisted_report_validation(
                self._json_object_or_empty(marker.request).get("report"),
                self._json_object_or_empty(marker.request).get("report_validation_errors", []),
            )
            return session_id, validation.report
        if marker is None and expected_session_id is None:
            sessions = list(await db.scalars(select(Session).where(
                Session.task_id == task.id, Session.status == "completed",
            )))
            if len(sessions) > 1:
                raise HTTPException(status_code=409, detail="Terminal task session is not canonical")
            expected_session_id = sessions[0].id if sessions else None
        session_id, output = await self.consume_terminal_task_output(db, run, task, expected_session_id)
        validation = validate_work_report(output)
        if marker is not None and marker.status == "completed":
            return session_id, validation.report

        marker = await self._existing_action_for_key(db, run.id, key)
        assert marker is not None
        marker.request = {
            **self._json_object_or_empty(marker.request),
            "report_validation_errors": list(validation.errors),
            "clarification_count": 0,
            "report": work_report_payload(validation.report),
        }
        if validation.report is not None:
            state = self._json_object_or_empty(run.supervision_state)
            candidates = list(state.get("verification_candidates", []))
            candidates.extend(item for item in validation.report.candidate_evidence if item not in candidates)
            state["verification_candidates"] = candidates
            state.setdefault("verified_progress", [])
            run.supervision_state = state
            goal = await db.get(OrchestrationGoal, run.goal_id)
            if goal is not None:
                session = await db.get(Session, session_id) if session_id else None
                await curate_report_memory(
                    db, OrchestrationMemoryService(), goal, run, validation.report,
                    report_id=str(session_id or task.id),
                    producer_agent_id=session.agent_id if session else task.assigned_to,
                    event_id=event_id,
                    task_id=task.id,
                    session_id=session_id,
                )
        await self._mark_action_completed(db, marker, "task", task.id)
        if validation.report is None:
            await self._handle_malformed_canonical_report(db, run, task, session_id, marker)
        return session_id, validation.report

    async def _handle_malformed_canonical_report(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        task: Task,
        session_id: uuid.UUID | None,
        marker: OrchestrationAction,
    ) -> None:
        """Give a non-discovery producer one bounded chance to repair its report."""
        metadata = self._json_object_or_empty(task.metadata_)
        contract = self._json_object_or_empty(metadata.get("orchestration_contract"))
        context = self._json_object_or_empty(contract.get("orchestrator_context"))
        continuous = self._json_object_or_empty(context.get("continuous"))
        if continuous.get("discovery_run_id") == str(run.id):
            return
        report_schema = self._json_object_or_empty(contract.get("report_schema"))
        if contract.get("expected_result") != "canonical_work_report" and set(report_schema) != {
            "status", "changes", "evidence", "criterion_progress", "decisions",
            "risks", "open_questions", "next_step", "collaboration_need",
        }:
            return

        parent_task_id = self._optional_uuid(
            metadata.get("report_clarification_parent_task_id"), "report_clarification_parent_task_id"
        ) or task.id
        parent_marker = marker
        if parent_task_id != task.id:
            parent_marker = await self._existing_action_for_key(
                db, run.id, f"run:{run.id}:kind:report_consumed:task:{parent_task_id}"
            )
            if parent_marker is None:
                return
        parent_request = self._json_object_or_empty(parent_marker.request)
        clarification_count = int(parent_request.get("clarification_count", 0))
        if clarification_count:
            key = f"run:{run.id}:kind:ask_human:report_clarification:task:{parent_task_id}"
            ask = await self._existing_action_for_key(db, run.id, key)
            if ask is None:
                ask = await self.execute_ask_human_action(
                    db, run.id,
                    {
                        "action_type": "ask_human",
                        "question": f"Task report remained invalid after clarification: {parent_task_id}",
                        "work_function": self._task_work_function(task),
                        "required_capabilities": [], "candidate_agent_ids": [],
                        "reason": "Canonical work report was invalid twice.",
                    }, key,
                )
            goal = await db.get(OrchestrationGoal, run.goal_id)
            if goal is not None:
                self._mark_run_blocked(goal, run)
            self._upsert_active_blocker(run, {
                "kind": "report_clarification_required", "task_id": str(parent_task_id),
                "reason": "Canonical work report was invalid twice.",
                "decision_id": str(ask.target_id),
            })
            return

        goal = await db.get(OrchestrationGoal, run.goal_id)
        if goal is None or task.assigned_to is None:
            return
        from huddleroom.services.orchestration_budget_service import (
            BudgetMeasurementError,
            OrchestrationBudgetService,
            parent_caps,
        )

        if not parent_caps(goal):
            return

        try:
            budget = await OrchestrationBudgetService().protected_action_allocation(db, goal, run)
        except BudgetMeasurementError as exc:
            self._upsert_active_blocker(run, {
                "kind": "budget_measurement", "dimension": exc.dimension,
                "session_id": str(exc.session_id), "scope": f"report_clarification:{parent_task_id}",
            })
            self._upsert_active_blocker(run, {
                "kind": "report_clarification_required", "task_id": str(parent_task_id),
                "reason": f"Cannot fund report clarification: missing {exc.dimension} measurement.",
            })
            return
        if not any(Decimal(amount) for amount in budget.values()):
            self._upsert_active_blocker(run, {"kind": "budget_exhausted"})
            self._upsert_active_blocker(run, {
                "kind": "report_clarification_required", "task_id": str(parent_task_id),
                "reason": "Cannot fund report clarification: budget capacity is exhausted.",
            })
            return
        if not await OrchestrationBudgetService().can_dispatch(db, goal, run, budget):
            if self._has_active_blocker(run, "budget_measurement"):
                self._upsert_active_blocker(run, {
                    "kind": "report_clarification_required", "task_id": str(parent_task_id),
                    "reason": "Cannot fund report clarification until budget telemetry is complete.",
                })
                return
            self._upsert_active_blocker(run, {"kind": "budget_exhausted"})
            self._upsert_active_blocker(run, {
                "kind": "report_clarification_required", "task_id": str(parent_task_id),
                "reason": "Cannot fund report clarification: budget capacity is exhausted.",
            })
            return
        key = f"run:{run.id}:kind:report_clarification:task:{parent_task_id}"
        action = await self.execute_create_delegation_task_action(
            db, run.id,
            {
                "action_type": "create_delegation_task", "agent_id": str(task.assigned_to),
                "work_function": "report_clarification",
                "scope": "Return only a corrected canonical report for the prior completed task.",
                "inputs": [
                    f"Prior task: {parent_task_id}",
                    f"Invalid report errors: {parent_request.get('report_validation_errors', [])}",
                ],
                "deliverable": "A valid canonical work report.",
                "forbidden_work": ["Do not change project artifacts or repeat the task work."],
                "success_evidence": ["A valid canonical work report."],
                "budget": budget,
                "report_schema": {
                    "status": "str", "changes": "list[str]", "evidence": "list[str]",
                    "criterion_progress": "object", "decisions": "list[str]", "risks": "list[str]",
                    "open_questions": "list[str]", "next_step": "str", "collaboration_need": "str|null",
                },
                "parent_task_id": str(parent_task_id),
                "source_session_id": str(session_id) if session_id else None,
            }, key,
        )
        if action.status == "completed" and action.target_id is not None:
            clarification = await db.get(Task, action.target_id)
            if clarification is not None:
                clarification.metadata_ = {
                    **self._json_object_or_empty(clarification.metadata_),
                    "report_clarification_parent_task_id": str(parent_task_id),
                }
            parent_marker.request = {**parent_request, "clarification_count": 1}

    def _canonical_reassign_task_request(self, request: Any) -> dict[str, Any]:
        request_obj = self._json_object_or_empty(request)
        return {
            "action_type": "reassign_task",
            "task_id": str(self._required_uuid(request_obj.get("task_id"), "task_id")),
            "agent_id": str(self._required_uuid(request_obj.get("agent_id"), "agent_id")),
        }

    def _canonical_request_verification_request(self, request: Any) -> dict[str, Any]:
        request_obj = self._json_object_or_empty(request)
        return {
            "action_type": "request_verification",
            "gate_id": str(self._required_uuid(request_obj.get("gate_id"), "gate_id")),
            "work_function": self._required_string(request_obj.get("work_function"), "work_function"),
        }

    def _canonical_pause_run_request(self, request: Any) -> dict[str, Any]:
        request_obj = self._json_object_or_empty(request)
        return {
            "action_type": "pause_run",
            "reason": self._required_string(request_obj.get("reason"), "reason"),
        }

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if not isinstance(value, (list, tuple, set)):
            return []
        return [item_text for item in value if (item_text := str(item).strip())]

    @staticmethod
    def _suggested_role(work_function: str) -> str:
        return {
            "planning": "planner",
            "investigation": "investigator",
            "implementation": "developer",
            "review": "reviewer",
            "validation": "validator",
            "summarization": "summarizer",
        }.get(work_function, work_function.replace("_", " "))

    @staticmethod
    def _suggested_system_prompt_outline(
        role: str,
        work_function: str,
        capabilities: list[str],
    ) -> str:
        capability_text = ", ".join(capabilities)
        return (
            f"Act as a {role} for the orchestrator. "
            f"Handle {work_function} using these capabilities: {capability_text}. "
            "Produce only requested deliverables and report results without changing unrelated artifacts."
        )

    def _authorize_execution_key(self, run_id: uuid.UUID) -> str:
        return f"run:{run_id}:kind:authorize_execution"

    async def authorize_baseline(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
        *,
        actor: str,
    ) -> tuple[OrchestrationGoal, OrchestrationRun]:
        """Explicit human authorization to start/continue baseline processing.

        Idempotent (no-op once already authorized). Requires the goal's active
        run to still be in the 'baseline' phase.
        """
        goal = await self.get_goal(db, project_id, goal_id)
        if goal is None:
            raise HTTPException(status_code=404, detail="Orchestration goal not found")
        run = await self.get_active_run_for_goal(db, project_id, goal_id)
        if run is None:
            raise self._start_conflict("goal_not_runnable", "Goal has no active run")
        if run.phase != "baseline":
            raise self._start_conflict("baseline_complete", f"Run phase is {run.phase}")
        if not run.baseline_authorized:
            run.baseline_authorized = True
            await db.flush()
        return goal, run

    async def start_run(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
        *,
        actor: str,
    ) -> tuple[OrchestrationGoal, OrchestrationRun]:
        """Explicit human authorization of spend + execution (spec: Start work).
        Idempotent. The scheduler never executes a run whose phase != authorized."""
        transaction = db.sync_session.get_transaction()
        caller_owns_transaction = (
            transaction is not None and transaction.origin != SessionTransactionOrigin.AUTOBEGIN
        )
        async with self._lock_goal_for_baseline_transition(db, goal_id):
            goal = await db.scalar(
                select(OrchestrationGoal)
                .where(OrchestrationGoal.id == goal_id, OrchestrationGoal.project_id == project_id)
                .execution_options(populate_existing=True)
            )
            if goal is None:
                raise HTTPException(status_code=404, detail="Orchestration goal not found")
            run = await self.get_active_run_for_goal(db, project_id, goal_id)
            if run is None:
                raise self._start_conflict("goal_not_runnable", "Goal has no active run")

            if goal.goal_type == "continuous" and (goal.continuous_state or {}).get("started_at"):
                return goal, run

            # Idempotent replay: already authorized -> return as-is.
            existing = await self._existing_action_for_key(
                db, run.id, self._authorize_execution_key(run.id)
            )
            if existing is not None and run.phase in {"authorized", "completed"}:
                return goal, run

            if goal.status not in {"active", "blocked"} or run.status not in {"running", "blocked"}:
                raise self._start_conflict("goal_not_runnable", f"Goal is {goal.status}")
            if goal.goal_type not in {"outcome", "roadmap", "continuous"}:
                raise self._start_conflict("goal_not_runnable", f"goal_type '{goal.goal_type}' is not runnable")
            if goal.goal_type == "continuous" and not goal.continuous_policy:
                raise self._start_conflict("goal_not_runnable", "Continuous policy is required")
            if goal.goal_type == "continuous":
                from huddleroom.schemas.orchestration import OrchestrationContinuousPolicy

                try:
                    OrchestrationContinuousPolicy.model_validate({
                        key: value for key, value in goal.continuous_policy.items() if key != "version"
                    })
                except ValidationError as exc:
                    raise self._start_conflict("goal_not_runnable", "Continuous policy is invalid") from exc
            if run.phase != "ready":
                if run.phase == "baseline":
                    raise self._start_conflict("baseline_not_ready", "Baseline is not terminal yet")
                raise self._start_conflict("already_started", f"Run phase is {run.phase}")

            await ProjectService().lock_workspace_boundary(db, project_id)
            await ProjectService().require_runnable_project(db, project_id)
            action = await self.reserve_action(
                db,
                run_id=run.id,
                idempotency_key=self._authorize_execution_key(run.id),
                action_type="authorize_execution",
                request={
                    "action_type": "authorize_execution",
                    "actor": actor,
                    "baseline_version": self._json_object_or_empty(goal.orchestrator_context).get(
                        "baseline_version"
                    ),
                    "authorized_at": _utcnow().isoformat(),
                },
            )
            if action.status == "reserved":
                await self._mark_action_completed(db, action, target_type="run", target_id=run.id)
            if goal.goal_type == "continuous":
                from huddleroom.services.orchestration_continuous_service import OrchestrationContinuousService

                await OrchestrationContinuousService(self).initialize_after_start(db, goal, run, _utcnow())
            else:
                run.phase = "authorized"
            await db.flush()
            if not caller_owns_transaction:
                await db.commit()
        return goal, run

    def _start_conflict(self, conflict: str, message: str) -> HTTPException:
        return HTTPException(status_code=409, detail={"conflict": conflict, "message": message})

    async def pause_goal(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
    ) -> tuple[OrchestrationGoal, OrchestrationRun | None]:
        transaction = db.sync_session.get_transaction()
        caller_owns_transaction = transaction is not None and transaction.origin != SessionTransactionOrigin.AUTOBEGIN
        async with self._lock_goal_for_baseline_transition(db, goal_id):
            goal = await self.get_goal(db, project_id, goal_id)
            if goal is None:
                raise HTTPException(status_code=404, detail="Orchestration goal not found")
            if goal.status not in ("active", "blocked"):
                raise HTTPException(status_code=409, detail=f"Cannot pause goal in status '{goal.status}'")
            goal.status = "paused"
            run = await self.get_active_run_for_goal(db, project_id, goal_id)
            if run is not None and run.status in ("running", "blocked"):
                run.status = "paused"
                if run.phase == "baseline":
                    run.baseline_authorized = False
            await db.flush()
            if not caller_owns_transaction:
                await db.commit()
            return goal, run

    async def resume_goal(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
    ) -> tuple[OrchestrationGoal, OrchestrationRun | None]:
        goal = await self.get_goal(db, project_id, goal_id)
        if goal is None:
            raise HTTPException(status_code=404, detail="Orchestration goal not found")
        await ProjectService().lock_workspace_boundary(db, project_id)
        await ProjectService().require_runnable_project(db, project_id)
        if goal.status != "paused":
            raise HTTPException(status_code=409, detail=f"Cannot resume goal in status '{goal.status}'")
        goal.status = "active"
        run = await self.get_active_run_for_goal(db, project_id, goal_id)
        # ponytail: resume always returns the run to 'running'; blocker re-detection
        # is the tick loop's job (later phase), not this endpoint's.
        if run is not None and run.status == "paused":
            run.status = "running"
        await db.flush()
        return goal, run

    async def cancel_goal(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
        *,
        cancelled_by: str = "human:anonymous",
    ) -> tuple[OrchestrationGoal, OrchestrationRun | None]:
        transaction = db.sync_session.get_transaction()
        caller_owns_transaction = (
            transaction is not None and transaction.origin != SessionTransactionOrigin.AUTOBEGIN
        )
        async with self._lock_goal_for_baseline_transition(db, goal_id):
            goal = await db.scalar(
                select(OrchestrationGoal)
                .where(
                    OrchestrationGoal.id == goal_id,
                    OrchestrationGoal.project_id == project_id,
                )
                .execution_options(populate_existing=True)
            )
            if goal is None:
                raise HTTPException(status_code=404, detail="Orchestration goal not found")
            if goal.status in ("completed", "cancelled"):
                raise HTTPException(status_code=409, detail=f"Cannot cancel goal in status '{goal.status}'")
            run = await db.scalar(
                select(OrchestrationRun)
                .join(OrchestrationGoal, OrchestrationGoal.id == OrchestrationRun.goal_id)
                .where(
                    OrchestrationGoal.project_id == project_id,
                    OrchestrationRun.goal_id == goal_id,
                    OrchestrationRun.status.in_(ACTIVE_RUN_STATUSES),
                )
                .order_by(OrchestrationRun.started_at.desc(), OrchestrationRun.id.desc())
                .limit(1)
                .execution_options(populate_existing=True)
            )
            await self._cancel_goal_no_commit(db, goal, run, cancelled_by=cancelled_by)
            if not caller_owns_transaction:
                await db.commit()
        return goal, run

    async def _cancel_goal_no_commit(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun | None,
        *,
        cancelled_by: str,
    ) -> None:
        """Cancel an already locked, fresh goal without committing the caller's transaction."""
        if goal.goal_type == "roadmap":
            from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService
            await OrchestrationRoadmapService(self).cascade_cancel(db, goal, cancelled_by=cancelled_by)
        if goal.goal_type == "continuous":
            from huddleroom.services.orchestration_continuous_service import OrchestrationContinuousService
            await OrchestrationContinuousService(self).cancel_children(db, goal, cancelled_by=cancelled_by)
        await GoalCloseoutProcess().close_cancelled(db, goal, run, cancelled_by=cancelled_by)
        if run is not None:
            for task in await self._orchestrated_tasks_for_run(
                db, run.id, statuses=["backlog", "ready", "in_progress", "blocked"]
            ):
                active_sessions = await db.scalars(
                    select(Session).where(
                        Session.task_id == task.id,
                        Session.status.in_(("pending", "running")),
                    )
                )
                for session in active_sessions:
                    await SessionService().cancel(db, session.id)
                await TaskService().cancel(db, goal.project_id, task.id)
        goal.status = "cancelled"
        if run is not None:
            run.status = "cancelled"
            run.completed_at = _utcnow()
        await db.flush()

    async def reset_goal(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
    ) -> tuple[OrchestrationGoal, OrchestrationRun]:
        """Reset a goal to fresh state, wiping all run history and orchestration progress.

        Status lines / timeline / logs are UI renderings of the wiped tables — no
        separate store to clear.
        """
        transaction = db.sync_session.get_transaction()
        caller_owns_transaction = (
            transaction is not None and transaction.origin != SessionTransactionOrigin.AUTOBEGIN
        )
        async with self._lock_goal_for_baseline_transition(db, goal_id):
            # Fetch goal fresh and verify it belongs to project_id
            goal = await db.scalar(
                select(OrchestrationGoal)
                .where(
                    OrchestrationGoal.id == goal_id,
                    OrchestrationGoal.project_id == project_id,
                )
                .execution_options(populate_existing=True)
            )
            if goal is None:
                raise HTTPException(status_code=404, detail="Orchestration goal not found")
            if goal.status in ("completed", "cancelled"):
                raise HTTPException(status_code=409, detail=f"Cannot reset goal in status '{goal.status}'")
            child_lineage = (
                goal.parent_goal_id is not None
                or await db.scalar(select(OrchestrationRoadmapItem.id).where(
                    OrchestrationRoadmapItem.child_goal_id == goal.id
                )) is not None
                or await db.scalar(select(OrchestrationBudgetReservation.id).where(
                    OrchestrationBudgetReservation.child_goal_id == goal.id,
                    OrchestrationBudgetReservation.status == "active",
                )) is not None
            )
            owned_roadmap_lineage = goal.goal_type == "roadmap" and (
                await db.scalar(select(OrchestrationRoadmapVersion.id).where(
                    OrchestrationRoadmapVersion.goal_id == goal.id
                )) is not None
                or await db.scalar(select(OrchestrationRoadmapItem.id).where(
                    OrchestrationRoadmapItem.goal_id == goal.id
                )) is not None
            )
            owned_continuous_lineage = goal.goal_type == "continuous" and (
                bool((goal.continuous_state or {}).get("started_at"))
                or await db.scalar(select(OrchestrationGoal.id).where(
                    OrchestrationGoal.parent_goal_id == goal.id,
                    OrchestrationGoal.continuous_origin_key.is_not(None),
                ).limit(1)) is not None
                or await db.scalar(select(OrchestrationBudgetReservation.id).where(
                    OrchestrationBudgetReservation.parent_goal_id == goal.id,
                    OrchestrationBudgetReservation.continuous_origin_key.is_not(None),
                ).limit(1)) is not None
            )
            if child_lineage or owned_roadmap_lineage or owned_continuous_lineage:
                raise HTTPException(status_code=409, detail="Cannot reset goal with immutable execution lineage")

            # Bulk-delete goal-keyed tables (goal-keyed deletes first, then the run)
            await db.execute(delete(OrchestrationProcessRun).where(OrchestrationProcessRun.goal_id == goal_id))
            await db.execute(delete(OrchestrationWarning).where(OrchestrationWarning.goal_id == goal_id))
            await db.execute(delete(OrchestrationAuthorityDecision).where(
                OrchestrationAuthorityDecision.goal_id == goal_id
            ))
            await db.execute(delete(OrchestrationAgentReview).where(OrchestrationAgentReview.goal_id == goal_id))
            await db.execute(delete(OrchestrationMemorySection).where(OrchestrationMemorySection.goal_id == goal_id))

            # Delete goal's run row(s) — CASCADEs run-keyed children (decisions/actions/gates/evidence/agent_suggestions)
            await db.execute(delete(OrchestrationRun).where(OrchestrationRun.goal_id == goal_id))

            # Recreate ONE fresh run
            run = OrchestrationRun(
                goal_id=goal.id,
                budget_state=deepcopy(goal.budget),
                baseline_authorized=False,
            )
            db.add(run)

            # Reset goal columns to fresh state
            goal.orchestrator_context = {}
            goal.manager_agent_id = None
            goal.manager_user_id = None
            goal.authority_model = None
            goal.weight_overridden_by = None
            if goal.goal_type == "continuous":
                goal.continuous_state = {}
            goal.weight = classify_goal_weight(
                goal.success_criteria,
                goal.constraints,
                goal.budget,
                goal.objective,
                explicit_multi_work_function=goal.explicit_multi_work_function,
            )
            goal.status = "active"

            await db.flush()
            # Commit while still holding the lock so a concurrent request cannot
            # acquire the lock and read stale pre-reset state (mirrors tick/cancel).
            if not caller_owns_transaction:
                await db.commit()
        return goal, run

    async def recover_goal_definition(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
        *,
        mode: str,
    ) -> dict[str, Any]:
        """Recover from goal_definition_clarification_limit with non-destructive options.

        mode='proceed': finalize with current understanding (preserves all answers).
        mode='another_round': increment clarification_round_bonus and ask another round.
        """
        if mode not in ("proceed", "another_round"):
            raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")

        transaction = db.sync_session.get_transaction()
        caller_owns_transaction = (
            transaction is not None and transaction.origin != SessionTransactionOrigin.AUTOBEGIN
        )

        async with self._lock_goal_for_baseline_transition(db, goal_id):
            goal = await db.scalar(
                select(OrchestrationGoal)
                .where(
                    OrchestrationGoal.id == goal_id,
                    OrchestrationGoal.project_id == project_id,
                )
                .execution_options(populate_existing=True)
            )
            if goal is None:
                raise HTTPException(status_code=404, detail="Orchestration goal not found")
            if goal.status not in ("active", "blocked"):
                raise HTTPException(
                    status_code=409, detail=f"Cannot recover goal in status '{goal.status}'"
                )

            run = await self.get_active_run_for_goal(db, project_id, goal_id)
            if run is None:
                raise HTTPException(status_code=409, detail="No active run for goal")
            await db.refresh(run)
            if run.status not in LLM_DECISION_RUN_STATUSES:
                raise HTTPException(status_code=409, detail=f"Cannot recover run in status '{run.status}'")

            current = await OrchestrationProcessService().get_current(db, goal.id, "goal_definition")
            if (
                current is None
                or current.status != "completed"
                or not (current.outputs or {}).get("clarification_limit_reached")
            ):
                raise HTTPException(
                    status_code=409,
                    detail="Goal definition not blocked at clarification limit",
                )

            blocker = next(
                (
                    item
                    for item in run.active_blockers
                    if isinstance(item, Mapping)
                    and item.get("kind") == "goal_definition_clarification_limit"
                ),
                None,
            )
            if blocker is None:
                raise HTTPException(
                    status_code=409, detail="No goal_definition_clarification_limit blocker found"
                )

            self._remove_active_blocker_by_kind(run, "goal_definition_clarification_limit")
            if not run.active_blockers:
                if goal.status == "blocked":
                    goal.status = "active"
                if run.status == "blocked":
                    run.status = "running"

            await ProjectService().lock_workspace_boundary(db, project_id)
            await ProjectService().require_runnable_project(db, project_id)

            nested = await db.begin_nested()
            try:
                new_row = await OrchestrationProcessService().start_process(
                    db,
                    goal.id,
                    process_type="goal_definition",
                    trigger_reason=f"human requested: goal_definition recovery ({mode})",
                    run_id=run.id,
                    input_snapshot=deepcopy(current.input_snapshot or {}),
                    process_version=current.process_version,
                )
                context = deepcopy(goal.orchestrator_context or {})

                if mode == "proceed":
                    summary = await GoalDefinitionProcess()._finalize_goal_definition(
                        db, goal, run, context, new_row
                    )
                else:  # mode == "another_round"
                    context["clarification_round_bonus"] = int(
                        context.get("clarification_round_bonus", 0)
                    ) + 1
                    goal.orchestrator_context = context
                    summary = await GoalDefinitionProcess().advance(db, goal, run)

                await nested.commit()
            except Exception:
                await nested.rollback()
                raise

            await db.flush()
            if not caller_owns_transaction:
                await db.commit()

            return {
                "goal_id": goal.id,
                "run_id": run.id,
                "process_type": "goal_definition",
                "process": summary,
            }

    async def override_gate(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
        *,
        gate_id: uuid.UUID,
        decision: str,
        reason: str,
        user_id: uuid.UUID | None,
        evidence_metadata: Mapping[str, Any] | None = None,
    ) -> tuple[OrchestrationGoal, OrchestrationRun]:
        goal = await self.get_goal(db, project_id, goal_id)
        if goal is None:
            raise HTTPException(status_code=404, detail="Orchestration goal not found")
        run = await self.get_run_for_goal(db, project_id, goal_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Orchestration run not found")
        if run.status not in ACTIVE_RUN_STATUSES:
            raise HTTPException(status_code=409, detail=f"Orchestration run is {run.status}")

        gate = await db.get(OrchestrationGate, gate_id)
        if gate is None or gate.run_id != run.id:
            raise HTTPException(status_code=404, detail="Orchestration gate not found")

        if decision not in {"accept", "reject"}:
            raise HTTPException(status_code=400, detail="decision must be accept or reject")
        reason_text = self._required_string(reason, "reason")

        gate_changed = (
            gate.status != "accepted" or gate.failure_reason is not None
            if decision == "accept"
            else gate.status != "failed" or gate.failure_reason != reason_text
        )
        evidence, transition_id = await self._upsert_human_override_evidence(
            db,
            run_id=run.id,
            gate_id=gate.id,
            decision=decision,
            reason=reason_text,
            user_id=user_id,
            details=evidence_metadata,
            gate_changed=gate_changed,
        )
        if decision == "accept":
            if gate_changed:
                gate.accepted_at = _utcnow()
                gate.failed_at = None
            gate.status = "accepted"
            gate.failure_reason = None
        else:
            if gate_changed:
                gate.failed_at = _utcnow()
                gate.accepted_at = None
            gate.status = "failed"
            gate.failure_reason = reason_text

        payload = {
            "goal_id": str(goal.id),
            "run_id": str(run.id),
            "gate_id": str(gate.id),
            "evidence_id": str(evidence.id),
            "decision": decision,
            "reason": reason_text,
            "user_id": str(user_id) if user_id else None,
        }
        if transition_id is not None:
            payload["transition_id"] = transition_id
        await emit_event_once(
            db,
            project_id,
            "orchestration.gate_overridden",
            payload,
            source="orchestrator",
            dedup_key=f"orchestration.gate_overridden:{self._stable_hash(payload)}",
        )
        await db.flush()
        return goal, run

    async def _upsert_human_override_evidence(
        self,
        db: AsyncSession,
        *,
        run_id: uuid.UUID,
        gate_id: uuid.UUID,
        decision: str,
        reason: str,
        user_id: uuid.UUID | None,
        details: Mapping[str, Any] | None,
        gate_changed: bool,
    ) -> tuple[OrchestrationEvidence, str | None]:
        query = select(OrchestrationEvidence).where(
            OrchestrationEvidence.run_id == run_id,
            OrchestrationEvidence.gate_id == gate_id,
            OrchestrationEvidence.source_type == "human_override",
            OrchestrationEvidence.observed_event_id.is_(None),
        )
        if user_id is None:
            query = query.where(OrchestrationEvidence.source_id.is_(None))
        else:
            query = query.where(OrchestrationEvidence.source_id == user_id)

        metadata = {
            "decision": decision,
            "reason": reason,
            "user_id": str(user_id) if user_id else None,
            "details": self._json_object_or_empty(details),
        }
        verdict = "accepted" if decision == "accept" else "rejected"
        evidence = (await db.execute(query)).scalar_one_or_none()
        if evidence is None:
            transition_id = str(uuid.uuid4())
            evidence = OrchestrationEvidence(
                run_id=run_id,
                gate_id=gate_id,
                source_type="human_override",
                source_id=user_id,
                observed_event_id=None,
                producer_agent_id=None,
                verdict=verdict,
                evidence_metadata={**metadata, "override_transition_id": transition_id},
            )
            db.add(evidence)
        else:
            current_metadata = deepcopy(evidence.evidence_metadata or {})
            transition_id = current_metadata.pop("override_transition_id", None)
            evidence_changed = evidence.verdict != verdict or current_metadata != metadata
            if evidence_changed or gate_changed:
                transition_id = str(uuid.uuid4())
                metadata = {**metadata, "override_transition_id": transition_id}
                if evidence.verdict != verdict:
                    evidence.verdict = verdict
                if evidence.evidence_metadata != metadata:
                    evidence.evidence_metadata = deepcopy(metadata)
        await db.flush()
        return evidence, transition_id

    async def _new_events(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        cursor: int | None,
        limit: int = TICK_EVENT_BATCH_LIMIT,
    ) -> list[EventLog]:
        """Project events with seq > cursor, excluding orchestration.* events.

        Filtering out orchestration.* keeps the tick from feeding on its own output.
        cursor=None returns all non-orchestration events for the project.

        Do NOT reuse for the Phase 7 LLM context builder — that needs orchestration
        events included.
        """
        query = (
            select(EventLog)
            .where(
                EventLog.project_id == project_id,
                EventLog.event_type.notlike("orchestration.%"),
            )
            .order_by(EventLog.seq)
            .limit(max(1, limit))
        )
        if cursor is not None:
            query = query.where(EventLog.seq > cursor)
        result = await db.execute(query)
        return list(result.scalars().all())

    @staticmethod
    def _event_uuid(value: Any) -> uuid.UUID | None:
        if value is None:
            return None
        text_value = str(value).strip()
        if not text_value:
            return None
        try:
            return uuid.UUID(text_value)
        except ValueError:
            return None

    async def _task_gate_for_run(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        task: Task,
    ) -> tuple[OrchestrationGate, dict[str, Any]] | None:
        task_metadata = self._json_object_or_empty(task.metadata_)
        orchestration = self._json_object_or_empty(task_metadata.get("orchestration"))
        if orchestration.get("run_id") != str(run_id):
            return None

        gate_id = self._event_uuid(orchestration.get("gate_id"))
        if gate_id is None:
            gate_id = self._event_uuid(orchestration.get("plan_item_gate_id"))
        if gate_id is None:
            plan_item_metadata = self._json_object_or_empty(task_metadata.get("orchestration_plan_item"))
            gate_id = self._event_uuid(plan_item_metadata.get("plan_item_gate_id"))
        if gate_id is None:
            return None

        gate = await db.get(OrchestrationGate, gate_id)
        if gate is None or gate.run_id != run_id:
            return None
        return gate, orchestration

    async def _record_evidence_once(
        self,
        db: AsyncSession,
        *,
        run_id: uuid.UUID,
        gate_id: uuid.UUID,
        source_type: str,
        source_id: uuid.UUID | None,
        observed_event: EventLog,
        producer_agent_id: uuid.UUID | None,
        verdict: str,
        evidence_metadata: dict[str, Any],
    ) -> int:
        query = select(OrchestrationEvidence).where(
            OrchestrationEvidence.run_id == run_id,
            OrchestrationEvidence.gate_id == gate_id,
            OrchestrationEvidence.source_type == source_type,
            OrchestrationEvidence.observed_event_id == observed_event.id,
        )
        if source_id is None:
            query = query.where(OrchestrationEvidence.source_id.is_(None))
        else:
            query = query.where(OrchestrationEvidence.source_id == source_id)

        existing = (await db.execute(query)).scalar_one_or_none()
        if existing is not None:
            return 0

        db.add(
            OrchestrationEvidence(
                run_id=run_id,
                gate_id=gate_id,
                source_type=source_type,
                source_id=source_id,
                observed_event_id=observed_event.id,
                producer_agent_id=producer_agent_id,
                verdict=verdict,
                evidence_metadata=deepcopy(evidence_metadata),
            )
        )
        await db.flush()
        return 1

    def _base_evidence_metadata(
        self,
        event: EventLog,
        gate: OrchestrationGate,
        orchestration: Mapping[str, Any],
        source_status: str | None,
    ) -> dict[str, Any]:
        return {
            "event_type": event.event_type,
            "event_seq": event.seq,
            "event_id": str(event.id),
            "emitted_at": event.emitted_at.isoformat(),
            "source_status": source_status,
            "work_function": self._optional_string(orchestration.get("work_function")),
            "plan_item_id": self._optional_string(orchestration.get("plan_item_id")),
            "gate_type": gate.gate_type,
        }

    async def _ingest_evidence_from_events(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        events: list[EventLog],
    ) -> int:
        created = 0
        for event in events:
            if event.event_type not in EVIDENCE_EVENT_TYPES:
                continue
            created += await self._ingest_evidence_from_event(db, run, event)
        return created

    async def _ingest_evidence_from_event(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        event: EventLog,
    ) -> int:
        if run.status not in ACTIVE_RUN_STATUSES:
            return 0
        if event.event_type in TASK_EVIDENCE_EVENT_TYPES:
            return await self._ingest_task_evidence(db, run, event)
        if event.event_type in SESSION_EVIDENCE_EVENT_TYPES:
            return await self._ingest_session_evidence(db, run, event)
        if event.event_type in REVIEW_EVIDENCE_EVENT_TYPES:
            return await self._ingest_review_evidence(db, run, event)
        if event.event_type in PROTOCOL_EVIDENCE_EVENT_TYPES:
            return await self._ingest_protocol_evidence(db, run, event)
        if event.event_type in MEETING_EVIDENCE_EVENT_TYPES:
            return await self._ingest_meeting_evidence(db, run, event)
        return 0

    async def _ingest_task_evidence(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        event: EventLog,
    ) -> int:
        status = self._optional_string(event.payload.get("status"))
        if status not in {"done", "failed"}:
            return 0
        task_id = self._event_uuid(event.payload.get("task_id"))
        if task_id is None:
            return 0
        task = await db.get(Task, task_id)
        if task is None:
            return 0

        gate_ref = await self._task_gate_for_run(db, run.id, task)
        if gate_ref is None:
            return 0
        gate, orchestration = gate_ref
        if await self._is_outcome_verification_task(db, run, gate, task, orchestration):
            return 0
        metadata = {
            **self._base_evidence_metadata(event, gate, orchestration, status),
            "task_id": str(task.id),
            "previous_status": self._optional_string(event.payload.get("previous_status")),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        }
        created = await self._record_evidence_once(
            db,
            run_id=run.id,
            gate_id=gate.id,
            source_type="task",
            source_id=task.id,
            observed_event=event,
            producer_agent_id=task.assigned_to,
            verdict="candidate" if status == "done" else "rejected",
            evidence_metadata=metadata,
        )
        if created and status == "done":
            sessions = list(await db.scalars(select(Session).where(
                Session.task_id == task.id, Session.status == "completed",
            )))
            if len(sessions) <= 1:
                await self.consume_canonical_report(
                    db, run, task, sessions[0].id if sessions else None, event.id
                )
        return created

    async def _ingest_session_evidence(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        event: EventLog,
    ) -> int:
        session_id = self._event_uuid(event.payload.get("session_id"))
        if session_id is None:
            return 0
        session = await db.get(Session, session_id)
        if session is None or session.task_id is None:
            return 0
        task = await db.get(Task, session.task_id)
        if task is None:
            return 0
        gate_ref = await self._task_gate_for_run(db, run.id, task)
        if gate_ref is None:
            return 0

        gate, orchestration = gate_ref
        if await self._is_outcome_verification_task(db, run, gate, task, orchestration):
            return await self._ingest_outcome_verification_evidence(db, run, event, gate, task, session)
        status = self._optional_string(session.status)
        metadata = {
            **self._base_evidence_metadata(event, gate, orchestration, status),
            "task_id": str(task.id),
            "session_id": str(session.id),
            "protocol_instance_id": str(session.protocol_instance_id) if session.protocol_instance_id else None,
            "origin": session.origin,
            "adapter_type": session.adapter_type,
            "error": session.error,
            "output_snippet": shorten(session.output or "", width=500, placeholder="[truncated]"),
            "session_metadata": self._json_object_or_empty(session.metadata_),
        }
        created = await self._record_evidence_once(
            db,
            run_id=run.id,
            gate_id=gate.id,
            source_type="session",
            source_id=session.id,
            observed_event=event,
            producer_agent_id=session.agent_id,
            verdict="candidate" if event.event_type == "session.completed" else "rejected",
            evidence_metadata=metadata,
        )
        if (
            created
            and event.event_type == "session.completed"
            and gate.gate_type == FINAL_SUMMARY_GATE_TYPE
            and gate.status == "failed"
        ):
            gate.status = "open"
            gate.failure_reason = None
            gate.failed_at = None
            gate.accepted_at = None

        # INVARIANT: this keys off task.status == "done" being already committed
        # by the time this event is processed, which relies on every
        # session.completed emitter calling sync_task_from_session(task->done)
        # in the same transaction as the emit (see api_adapter.py, cli_adapter.py).
        # A future emitter that skips that call silently and permanently drops
        # the report here -- no error, no retry.
        if created and event.event_type == "session.completed" and task.status == "done":
            marker = await self._existing_action_for_key(
                db, run.id, f"run:{run.id}:kind:report_consumed:task:{task.id}",
            )
            if marker is None:
                await self.consume_canonical_report(db, run, task, session.id, event.id)

        return created

    async def _is_outcome_plan_item_gate(
        self, db: AsyncSession, run: OrchestrationRun, gate: OrchestrationGate
    ) -> bool:
        if (
            gate.gate_type != PLAN_ITEM_GATE_TYPE
            or self._json_object_or_empty(run.plan_state).get("status") != "accepted"
        ):
            return False
        goal = await db.get(OrchestrationGoal, run.goal_id)
        if goal is None:
            return False
        if goal.goal_type == "outcome":
            return True
        required_evidence = self._json_object_or_empty(gate.required_evidence)
        return (
            goal.goal_type == "roadmap"
            and required_evidence.get("roadmap_item_key") is not None
            and required_evidence.get("roadmap_version_id") is not None
        )

    async def _is_outcome_verification_task(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        gate: OrchestrationGate,
        task: Task,
        orchestration: Mapping[str, Any],
    ) -> bool:
        return await self._outcome_verification_action_for_task(db, run, gate, task) is not None

    async def _outcome_verification_action_for_task(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        gate: OrchestrationGate,
        task: Task,
    ) -> OrchestrationAction | None:
        if not (
            await self._is_outcome_plan_item_gate(db, run, gate)
            or gate.gate_type in {"roadmap_plan_approval", "roadmap_integration"}
        ):
            return None
        action = await db.scalar(
            select(OrchestrationAction)
            .where(
                OrchestrationAction.run_id == run.id,
                OrchestrationAction.action_type == "request_verification",
                OrchestrationAction.status == "completed",
                OrchestrationAction.target_type == "task",
                OrchestrationAction.target_id == task.id,
            )
            .order_by(desc(OrchestrationAction.created_at), desc(OrchestrationAction.id))
            .limit(1)
        )
        if action is None or self._event_uuid(action.request.get("gate_id")) != gate.id:
            return None
        return action

    @staticmethod
    def _verification_report(output: str | None) -> str | None:
        try:
            report = json.loads(output or "")
        except (TypeError, ValueError):
            return None
        if not isinstance(report, Mapping) or report.get("status") != "done":
            return None
        verdict = report.get("verdict")
        if verdict not in {"accepted", "rejected"}:
            return None
        evidence = report.get("evidence")
        if not isinstance(evidence, list) or any(not isinstance(item, str) for item in evidence):
            return None
        if verdict == "accepted" and not any(item.strip() for item in evidence):
            return None
        return verdict

    async def _bound_outcome_verification_binding(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        gate: OrchestrationGate,
        task: Task,
        session: Session,
    ) -> tuple[OrchestrationAction, Task] | None:
        """Return Phase-A's durable verifier binding, or reject the proof."""
        action = await self._outcome_verification_action_for_task(db, run, gate, task)
        if gate.gate_type == "roadmap_plan_approval":
            try:
                source_task = await self._planning_task_for_gate(db, run, gate)
            except HTTPException:
                return None
        else:
            source_task = await self._plan_item_producer_task(db, gate)
        if gate.gate_type == "roadmap_integration":
            configured_producers = {
                producer for producer in (
                    self._event_uuid(value)
                    for value in self._string_list(
                        self._json_object_or_empty(gate.required_evidence).get("work_producer_agent_ids")
                    )
                ) if producer is not None
            }
            if (
                action is None
                or self._event_uuid(action.request.get("roadmap_version_id"))
                != self._event_uuid(self._json_object_or_empty(gate.required_evidence).get("roadmap_version_id"))
                or task.assigned_to != self._event_uuid(action.request.get("verifier_agent_id"))
                or session.agent_id != task.assigned_to
                or session.agent_id in configured_producers
                or task.status != "done" or session.status != "completed"
            ):
                return None
            latest = await db.scalar(
                select(Session.id)
                .where(Session.task_id == task.id, Session.status.in_(("completed", "failed")))
                .order_by(desc(Session.created_at), desc(Session.id))
                .limit(1)
            )
            if latest != session.id:
                return None
            if task.protocol_instance_id is not None or session.protocol_instance_id is not None:
                if task.protocol_instance_id != session.protocol_instance_id:
                    return None
                instance = await db.get(ProtocolInstance, session.protocol_instance_id)
                if instance is None or instance.linked_task_id != task.id:
                    return None
            return action, task
        if (
            action is None
            or source_task is None
            or source_task.assigned_to is None
            or source_task.id != self._event_uuid(action.request.get("source_task_id"))
            or source_task.assigned_to != self._event_uuid(action.request.get("producer_agent_id"))
            or task.assigned_to != self._event_uuid(action.request.get("verifier_agent_id"))
            or session.agent_id != task.assigned_to
            or session.agent_id == source_task.assigned_to
            or task.status != "done"
            or session.status != "completed"
        ):
            return None
        latest = await db.scalar(
            select(Session.id)
            .where(Session.task_id == task.id, Session.status.in_(("completed", "failed")))
            .order_by(desc(Session.created_at), desc(Session.id))
            .limit(1)
        )
        if latest != session.id:
            return None
        if task.protocol_instance_id is not None or session.protocol_instance_id is not None:
            if task.protocol_instance_id != session.protocol_instance_id:
                return None
            instance = await db.get(ProtocolInstance, session.protocol_instance_id)
            if instance is None or instance.linked_task_id != task.id:
                return None
        return action, source_task

    async def _ingest_outcome_verification_evidence(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        event: EventLog,
        gate: OrchestrationGate,
        task: Task,
        session: Session,
    ) -> int:
        binding = await self._bound_outcome_verification_binding(db, run, gate, task, session)
        if event.event_type != "session.completed" or binding is None:
            return 0
        action, source_task = binding
        verdict = self._verification_report(session.output)
        if verdict is None:
            return 0
        return await self._record_evidence_once(
            db,
            run_id=run.id,
            gate_id=gate.id,
            source_type="verification",
            source_id=session.id,
            observed_event=event,
            producer_agent_id=session.agent_id,
            verdict="candidate" if verdict == "accepted" else "rejected",
            evidence_metadata={
                "event_type": event.event_type,
                "event_seq": event.seq,
                "task_id": str(task.id),
                "session_id": str(session.id),
                "verification_action_id": str(action.id),
                "source_task_id": str(source_task.id),
                "protocol_instance_id": str(session.protocol_instance_id) if session.protocol_instance_id else None,
            },
        )

    async def _has_bound_outcome_verification_evidence(
        self, db: AsyncSession, gate: OrchestrationGate
    ) -> bool:
        """Accept only the durable verifier/session binding Phase A ingests."""
        run = await db.get(OrchestrationRun, gate.run_id)
        if run is None:
            return False
        for item in await self._evidence_for_gate(db, gate):
            if item.source_type != "verification" or item.verdict not in {"candidate", "accepted"}:
                continue
            session = await db.get(Session, item.source_id) if item.source_id else None
            metadata = self._json_object_or_empty(item.evidence_metadata)
            verifier_task = await db.get(Task, session.task_id) if session and session.task_id else None
            binding = (
                await self._bound_outcome_verification_binding(db, run, gate, verifier_task, session)
                if verifier_task is not None and session is not None
                else None
            )
            if (
                binding is None
                or self._verification_report(session.output) != "accepted"
            ):
                continue
            action, source_task = binding
            if (
                self._event_uuid(metadata.get("verification_action_id")) != action.id
                or (
                    gate.gate_type != "roadmap_integration"
                    and self._event_uuid(metadata.get("source_task_id")) != source_task.id
                )
            ):
                continue
            return True
        return False

    async def _ingest_review_evidence(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        event: EventLog,
    ) -> int:
        session_id = self._event_uuid(event.payload.get("session_id"))
        if session_id is None:
            return 0
        task_id = self._event_uuid(event.payload.get("task_id"))
        session = await db.get(Session, session_id)
        if session is None:
            return 0
        if session.task_id is None:
            return 0
        if task_id is not None and task_id != session.task_id:
            return 0
        protocol_instance_id = self._event_uuid(event.payload.get("protocol_instance_id"))
        if protocol_instance_id is not None and protocol_instance_id != session.protocol_instance_id:
            return 0
        task_id = session.task_id
        task = await db.get(Task, task_id)
        if task is None:
            return 0
        gate_ref = await self._task_gate_for_run(db, run.id, task)
        if gate_ref is None:
            return 0

        gate, orchestration = gate_ref
        if await self._is_outcome_plan_item_gate(db, run, gate):
            return 0
        review_outcome = self._json_object_or_empty(event.payload.get("review_outcome"))
        review_verdict = self._optional_string(review_outcome.get("verdict"))
        metadata = {
            **self._base_evidence_metadata(event, gate, orchestration, review_verdict),
            "task_id": str(task.id),
            "session_id": str(session.id),
            "protocol_instance_id": str(session.protocol_instance_id) if session.protocol_instance_id else None,
            "artifact_id": self._optional_string(event.payload.get("artifact_id")),
            "review_verdict": review_verdict,
            "review_outcome": review_outcome,
        }
        return await self._record_evidence_once(
            db,
            run_id=run.id,
            gate_id=gate.id,
            source_type="review",
            source_id=session.id,
            observed_event=event,
            producer_agent_id=session.agent_id,
            verdict="candidate" if event.event_type == "review.approved" else "rejected",
            evidence_metadata=metadata,
        )

    async def _ingest_protocol_evidence(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        event: EventLog,
    ) -> int:
        protocol_instance_id = self._event_uuid(event.payload.get("protocol_instance_id"))
        if protocol_instance_id is None:
            return 0
        instance = await db.get(ProtocolInstance, protocol_instance_id)
        if instance is None or instance.linked_task_id is None:
            return 0
        task = await db.get(Task, instance.linked_task_id)
        if task is None:
            return 0
        gate_ref = await self._task_gate_for_run(db, run.id, task)
        if gate_ref is None:
            return 0

        gate, orchestration = gate_ref
        status = self._optional_string(instance.status)
        is_transition_event = event.event_type == "protocol.state_transitioned"
        has_transition_id = "protocol_transition_id" in event.payload
        transition_id = self._event_uuid(event.payload.get("protocol_transition_id"))
        transition = None
        if is_transition_event or has_transition_id:
            if transition_id is None:
                return 0
            transition = await db.get(ProtocolTransition, transition_id)
            if transition is None or transition.protocol_instance_id != instance.id:
                return 0

        source_type = (
            "protocol_transition"
            if is_transition_event
            else "protocol_instance"
        )
        source_id = transition.id if is_transition_event else instance.id
        metadata = {
            **self._base_evidence_metadata(event, gate, orchestration, status),
            "task_id": str(task.id),
            "protocol_instance_id": str(instance.id),
            "protocol_transition_id": str(transition_id) if transition_id is not None else None,
            "protocol_name": self._optional_string(event.payload.get("protocol_name")),
            "current_state": instance.current_state,
            "transition_name": self._optional_string(event.payload.get("transition_name")),
            "from_state": self._optional_string(event.payload.get("from_state")),
            "to_state": self._optional_string(event.payload.get("to_state")),
        }
        return await self._record_evidence_once(
            db,
            run_id=run.id,
            gate_id=gate.id,
            source_type=source_type,
            source_id=source_id,
            observed_event=event,
            producer_agent_id=None,
            verdict="rejected" if event.event_type == "protocol.failed" else "candidate",
            evidence_metadata=metadata,
        )

    async def _ingest_meeting_evidence(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        event: EventLog,
    ) -> int:
        if event.event_type == "meeting.decision_recorded":
            return await self._ingest_meeting_decision_recorded_evidence(db, run, event)
        if event.event_type == "meeting.concluded":
            return await self._ingest_meeting_concluded_evidence(db, run, event)
        return 0

    async def _ingest_meeting_decision_recorded_evidence(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        event: EventLog,
    ) -> int:
        meeting_id = self._event_uuid(event.payload.get("meeting_id"))
        decision_id = self._event_uuid(event.payload.get("decision_id"))
        gate_id = self._event_uuid(event.payload.get("gate_id"))
        if meeting_id is None or decision_id is None or gate_id is None:
            return 0

        meeting = await db.get(Meeting, meeting_id)
        decision = await db.get(MeetingDecision, decision_id)
        gate = await db.get(OrchestrationGate, gate_id)
        if meeting is None or decision is None or gate is None or gate.run_id != run.id:
            return 0
        if decision.meeting_id != meeting.id:
            return 0
        if meeting.source_task_id is None:
            return 0
        task = await db.get(Task, meeting.source_task_id)
        if task is None:
            return 0
        gate_ref = await self._task_gate_for_run(db, run.id, task)
        if gate_ref is None or gate_ref[0].id != gate.id:
            return 0

        metadata = {
            "event_type": event.event_type,
            "event_seq": event.seq,
            "event_id": str(event.id),
            "emitted_at": event.emitted_at.isoformat(),
            "source_status": "recorded",
            "meeting_id": str(meeting.id),
            "decision_id": str(decision.id),
            "agenda_item_id": str(decision.agenda_item_id),
            "chosen_option": decision.chosen_option,
            "decided_by": decision.decided_by,
            "gate_type": gate.gate_type,
        }
        return await self._record_evidence_once(
            db,
            run_id=run.id,
            gate_id=gate.id,
            source_type="meeting_decision",
            source_id=decision.id,
            observed_event=event,
            producer_agent_id=meeting.organizer_agent_id or meeting.created_by_agent_id,
            verdict="rejected" if decision.is_vetoed else "candidate",
            evidence_metadata=metadata,
        )

    async def _ingest_meeting_concluded_evidence(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        event: EventLog,
    ) -> int:
        meeting_id = self._event_uuid(event.payload.get("meeting_id"))
        if meeting_id is None:
            return 0
        meeting = await db.get(Meeting, meeting_id)
        if meeting is None or meeting.source_task_id is None:
            return 0
        task = await db.get(Task, meeting.source_task_id)
        if task is None:
            return 0
        gate_ref = await self._task_gate_for_run(db, run.id, task)
        if gate_ref is None:
            return 0

        gate, orchestration = gate_ref
        result = await db.execute(
            select(MeetingDecision)
            .where(
                MeetingDecision.meeting_id == meeting.id,
                MeetingDecision.is_vetoed.is_(False),
            )
            .order_by(MeetingDecision.created_at.asc(), MeetingDecision.id.asc())
        )
        decisions = list(result.scalars().all())
        producer_agent_id = meeting.organizer_agent_id or meeting.created_by_agent_id
        if not decisions:
            return await self._record_evidence_once(
                db,
                run_id=run.id,
                gate_id=gate.id,
                source_type="meeting",
                source_id=meeting.id,
                observed_event=event,
                producer_agent_id=producer_agent_id,
                verdict="candidate",
                evidence_metadata={
                    **self._base_evidence_metadata(event, gate, orchestration, meeting.status),
                    "task_id": str(task.id),
                    "meeting_id": str(meeting.id),
                    "decision_id": None,
                },
            )

        created = 0
        for decision in decisions:
            created += await self._record_evidence_once(
                db,
                run_id=run.id,
                gate_id=gate.id,
                source_type="meeting_decision",
                source_id=decision.id,
                observed_event=event,
                producer_agent_id=producer_agent_id,
                verdict="candidate",
                evidence_metadata={
                    **self._base_evidence_metadata(event, gate, orchestration, meeting.status),
                    "task_id": str(task.id),
                    "meeting_id": str(meeting.id),
                    "decision_id": str(decision.id),
                    "agenda_item_id": str(decision.agenda_item_id),
                    "chosen_option": decision.chosen_option,
                    "decided_by": decision.decided_by,
                    "confidence": decision.confidence,
                },
            )
        return created

    async def recover_run(self, db: AsyncSession, run_id: uuid.UUID, baseline_ready: bool = False) -> int:
        run = await db.get(OrchestrationRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Orchestration run not found")
        if run.status not in LLM_DECISION_RUN_STATUSES:
            return 0
        goal = await db.get(OrchestrationGoal, run.goal_id)
        if goal is None:
            raise HTTPException(status_code=404, detail="Orchestration goal not found")
        if goal.status not in {"active", "blocked"}:
            return 0
        if not baseline_ready:
            return 0

        created = 0
        created += await self._recover_failed_tasks(db, goal, run)
        if run.status in LLM_DECISION_RUN_STATUSES and goal.status in {"active", "blocked"}:
            created += await self._recover_blocked_tasks(db, goal, run)
        if run.status in LLM_DECISION_RUN_STATUSES and goal.status in {"active", "blocked"}:
            created += await self._recover_failed_gates(db, goal, run, baseline_ready=baseline_ready)
        await db.flush()
        return created

    async def _recover_failed_tasks(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
    ) -> int:
        created = 0
        failed_tasks = await self._orchestrated_tasks_for_run(db, run.id, statuses=["failed"])
        # Ownership recovery is the only path allowed to interpret ambiguous
        # runner/effect state.  Keep ordinary age/failure recovery for legacy
        # and manual work only.
        from huddleroom.services.orchestration_recovery_service import OrchestrationRecoveryService
        ownership_recovery = OrchestrationRecoveryService()
        owned_task_ids = set()
        failure_counts = {}
        for task in failed_tasks:
            failed_session_ids = list(await db.scalars(select(Session.id).where(
                Session.task_id == task.id, Session.status == "failed",
            ).order_by(Session.created_at.desc(), Session.id.desc())))
            if failed_session_ids:
                current_session = await db.get(Session, failed_session_ids[0])
                current_metadata = current_session.metadata_ if current_session is not None and isinstance(current_session.metadata_, dict) else {}
                current_attempt = current_metadata.get("attempt") if isinstance(current_metadata.get("attempt"), dict) else {}
                current_owned = await ownership_recovery.resolve_owned_session(db, failed_session_ids[0])
                # A task-link alone predates Task 9 and is not a durable worker
                # claim.  Only the current, fenced Task 9 attempt takes recovery
                # ownership away from the legacy retry/reassign ladder.
                if (current_owned is not None
                        and current_attempt.get("claimed_runner_task_id") == current_session.runner_task_id
                        and isinstance(current_attempt.get("attempt_version"), int)):
                    owned_task_ids.add(task.id)
                    continue
            failure_counts[task.id] = max(1, len(failed_session_ids))
        for task in failed_tasks:
            if task.id in owned_task_ids:
                continue
            failure_count = failure_counts[task.id]
            if failure_count >= RECOVERY_REPEATED_FAILURE_SESSION_LIMIT:
                created += await self._escalate_repeated_failure(db, goal, run, task)
                return created
        for task in failed_tasks:
            if task.id in owned_task_ids:
                continue
            failure_count = failure_counts[task.id]
            if failure_count <= RECOVERY_RETRY_FAILED_SESSION_LIMIT:
                created += await self._retry_failed_task_once(db, goal, run, task)
                continue
            if failure_count >= RECOVERY_REASSIGN_FAILED_SESSION_LIMIT:
                created += await self._reassign_or_ask_human(db, goal, run, task)
        return created

    async def _recover_blocked_tasks(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
    ) -> int:
        created = 0
        for task in await self._orchestrated_tasks_for_run(db, run.id, statuses=["blocked"]):
            key = f"run:{run.id}:kind:ask_human:blocked_task:{task.id}"
            if await self._existing_action_for_key(db, run.id, key) is not None:
                continue
            gate_id = await self._gate_id_for_task(db, run.id, task)
            nested = await db.begin_nested()
            try:
                self._mark_run_blocked(goal, run)
                ask = await self.execute_ask_human_action(
                    db,
                    run_id=run.id,
                    request={
                        "action_type": "ask_human",
                        "question": f"Task is blocked and needs clarification: {task.title}",
                        "work_function": self._task_work_function(task),
                        "required_capabilities": [],
                        "candidate_agent_ids": [],
                        "gate_id": str(gate_id) if gate_id else None,
                        "reason": "Blocked orchestration task.",
                    },
                    idempotency_key=key,
                )
                self._upsert_active_blocker(run, {
                    "kind": "task_blocked", "task_id": str(task.id),
                    "gate_id": str(gate_id) if gate_id else None,
                    "reason": "Task is blocked and needs human clarification.",
                    "decision_id": str(ask.target_id),
                })
            except Exception:
                await nested.rollback()
                raise
            await nested.commit()
            created += 1
        return created

    async def _recover_failed_gates(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        baseline_ready: bool = False,
    ) -> int:
        result = await db.execute(
            select(OrchestrationGate)
            .where(
                OrchestrationGate.run_id == run.id,
                OrchestrationGate.status == "failed",
            )
            .order_by(OrchestrationGate.updated_at.asc(), OrchestrationGate.id.asc())
        )
        created = 0
        for gate in result.scalars().all():
            if gate.gate_type == FINAL_SUMMARY_GATE_TYPE and (gate.failure_reason or "").startswith(
                "Final summary "
            ):
                if not baseline_ready:
                    continue
                latest_session_evidence = next(
                    (
                        item
                        for item in reversed(await self._evidence_for_gate(db, gate))
                        if item.source_type == "session"
                    ),
                    None,
                )
                if latest_session_evidence is None or latest_session_evidence.verdict != "candidate":
                    continue
                key = self._final_summary_replacement_key(run.id, latest_session_evidence.id)
                if await self._existing_action_for_key(db, run.id, key) is not None:
                    continue
                source_session = (
                    await db.get(Session, latest_session_evidence.source_id)
                    if latest_session_evidence.source_id is not None
                    else None
                )
                source_task = (
                    await db.get(Task, source_session.task_id)
                    if source_session is not None and source_session.task_id is not None
                    else None
                )
                if source_task is None:
                    continue
                source_metadata = self._json_object_or_empty(source_task.metadata_)
                final_summary_metadata = self._json_object_or_empty(
                    source_metadata.get("orchestration_final_summary")
                )
                fits = await OrchestrationRosterMapper().rank_agents(
                    db,
                    goal.project_id,
                    FINAL_SUMMARY_WORK_FUNCTION,
                    required_capabilities=[FINAL_SUMMARY_WORK_FUNCTION],
                )
                if not fits or fits[0].weak:
                    continue
                gates, evidence = await self._accepted_non_summary_manifest(db, run.id)
                replacement = await self.execute_create_delegation_task_action(
                    db,
                    run_id=run.id,
                    request=self._final_summary_delegation_request(
                        goal,
                        run,
                        fits[0].agent_id,
                        gates,
                        evidence,
                    ),
                    idempotency_key=key,
                )
                if replacement.status != "completed" or replacement.target_id is None:
                    continue
                replacement_task = await db.get(Task, replacement.target_id)
                if replacement_task is None:
                    continue
                self._set_final_summary_task_metadata(
                    replacement_task,
                    gate,
                    final_summary_metadata,
                )
                created += 1
                continue
            if gate.failure_reason == "Evidence is stale":
                if not baseline_ready:
                    continue
                created += await self._request_stale_gate_verification(db, goal, run, gate)
                continue
            if not baseline_ready:
                continue
            latest_rejected = await self._latest_rejected_evidence_for_gate(db, gate)
            if latest_rejected is not None and latest_rejected.source_type == "review":
                created += await self._create_rejected_review_fix_task(db, goal, run, gate, latest_rejected)
        return created

    async def _retry_failed_task_once(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        task: Task,
    ) -> int:
        work_function = self._task_work_function(task)
        key = f"run:{run.id}:kind:retry_task:task:{task.id}"
        if await self._existing_action_for_key(db, run.id, key) is not None:
            return 0
        try:
            await self.execute_retry_task_action(
                db,
                run_id=run.id,
                request={"action_type": "retry_task", "task_id": str(task.id)},
                idempotency_key=key,
            )
            return 1
        except HTTPException:
            gate_id = await self._gate_id_for_task(db, run.id, task)
            ask_key = f"run:{run.id}:kind:ask_human:retry_task:{task.id}"
            if await self._existing_action_for_key(db, run.id, ask_key) is not None:
                return 0
            ask = await self.execute_ask_human_action(
                db,
                run_id=run.id,
                request={
                    "action_type": "ask_human",
                    "question": (
                        "Task retry could not start and needs human help: "
                        f"{task.title}"
                    ),
                    "work_function": work_function,
                    "required_capabilities": [work_function],
                    "candidate_agent_ids": [],
                    "gate_id": str(gate_id) if gate_id else None,
                    "reason": "Task retry failed to start during recovery.",
                },
                idempotency_key=ask_key,
            )
            self._mark_run_blocked(goal, run)
            self._upsert_active_blocker(
                run,
                {
                    "kind": "retry_required",
                    "task_id": str(task.id),
                    "gate_id": str(gate_id) if gate_id else None,
                    "reason": "Task retry failed to start during recovery.",
                    "decision_id": str(ask.target_id),
                },
            )
            return 1

    async def _reassign_or_ask_human(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        task: Task,
    ) -> int:
        gate_id = await self._gate_id_for_task(db, run.id, task)
        work_function = self._task_work_function(task)
        fit = await self._best_recovery_agent(
            db,
            goal.project_id,
            work_function,
            required_capabilities=[work_function],
            exclude_agent_ids={task.assigned_to} if task.assigned_to else set(),
        )
        if fit is not None:
            key = f"run:{run.id}:kind:reassign_task:task:{task.id}:agent:{fit.agent_id}"
            existing = await self._existing_action_for_key(db, run.id, key)
            if existing is None:
                try:
                    await self.execute_reassign_task_action(
                        db,
                        run_id=run.id,
                        request={
                            "action_type": "reassign_task",
                            "task_id": str(task.id),
                            "agent_id": str(fit.agent_id),
                        },
                        idempotency_key=key,
                    )
                    return 1
                except HTTPException:
                    # Reassign could not execute (agent went inactive, task raced,
                    # session failed to start). Escalate to a human below instead of
                    # aborting the whole recovery sweep.
                    pass
            elif existing.status != "failed":
                return 0
            # else: a prior reassign attempt failed -> fall through to human escalation.

        key = f"run:{run.id}:kind:ask_human:reassign_task:{task.id}"
        if await self._existing_action_for_key(db, run.id, key) is not None:
            return 0
        ask = await self.execute_ask_human_action(
            db,
            run_id=run.id,
            request={
                "action_type": "ask_human",
                "question": f"Task failed after retry limit. Which agent should take over: {task.title}?",
                "work_function": work_function,
                "required_capabilities": [work_function],
                "candidate_agent_ids": [],
                "gate_id": str(gate_id) if gate_id else None,
                "reason": "No strong alternate agent fit exists.",
            },
            idempotency_key=key,
        )
        self._mark_run_blocked(goal, run)
        self._upsert_active_blocker(
            run,
            {
                "kind": "reassign_required",
                "task_id": str(task.id),
                "gate_id": str(gate_id) if gate_id else None,
                "reason": "Task failed after retry limit and no strong alternate agent exists.",
                "decision_id": str(ask.target_id),
            },
        )
        return 1

    async def _escalate_repeated_failure(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        task: Task,
    ) -> int:
        key = f"run:{run.id}:kind:ask_human:repeated_failure:task:{task.id}"
        if await self._existing_action_for_key(db, run.id, key) is not None:
            return 0
        gate_id = await self._gate_id_for_task(db, run.id, task)
        failed_session_ids = [
            str(session_id)
            for session_id in (await db.scalars(
                select(Session.id)
                .where(Session.task_id == task.id, Session.status == "failed")
                .order_by(Session.id)
            )).all()
        ]
        reason = "Task failed repeatedly after retry and reassignment attempts."
        ask = await self.execute_ask_human_action(
            db,
            run_id=run.id,
            request={
                "action_type": "ask_human",
                "question": f"Task failed repeatedly and needs a human decision: {task.title}",
                "work_function": self._task_work_function(task),
                "required_capabilities": [self._task_work_function(task)],
                "candidate_agent_ids": [],
                "gate_id": str(gate_id) if gate_id else None,
                "reason": reason,
            },
            idempotency_key=key,
        )
        self._mark_run_blocked(goal, run)
        self._upsert_active_blocker(
            run,
            {
                "kind": "repeated_failure",
                "task_id": str(task.id),
                "gate_id": str(gate_id) if gate_id else None,
                "attempt_count": len(failed_session_ids),
                "failed_session_ids": failed_session_ids,
                "owner": "human",
                "reason": reason,
                "recommended_action": "Review the failed attempts and choose how to proceed.",
                "decision_id": str(ask.target_id),
            },
        )
        return 1

    async def _request_stale_gate_verification(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        gate: OrchestrationGate,
    ) -> int:
        key = f"run:{run.id}:kind:request_verification:stale_gate:{gate.id}"
        if await self._existing_action_for_key(db, run.id, key) is not None:
            return 0
        try:
            await self.execute_request_verification_action(
                db,
                run_id=run.id,
                request={
                    "action_type": "request_verification",
                    "gate_id": str(gate.id),
                    "work_function": "validation",
                },
                idempotency_key=key,
            )
            return 1
        except HTTPException as exc:
            if exc.status_code != 409 or exc.detail != "No strong verification agent fit":
                raise
            ask_key = f"run:{run.id}:kind:ask_human:verify_gate:{gate.id}"
            if await self._existing_action_for_key(db, run.id, ask_key) is not None:
                return 0
            ask = await self.execute_ask_human_action(
                db,
                run_id=run.id,
                request={
                    "action_type": "ask_human",
                    "question": f"Gate needs fresh verification but no strong validator is available: {gate.id}",
                    "work_function": "validation",
                    "required_capabilities": ["validation"],
                    "candidate_agent_ids": [],
                    "gate_id": str(gate.id),
                    "reason": "Stale gate recovery needs a verifier.",
                },
                idempotency_key=ask_key,
            )
            self._mark_run_blocked(goal, run)
            self._upsert_active_blocker(
                run,
                {
                    "kind": "verification_required",
                    "gate_id": str(gate.id),
                    "reason": "Stale gate has no strong verification agent.",
                    "decision_id": str(ask.target_id),
                },
            )
            return 1

    async def _create_rejected_review_fix_task(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        gate: OrchestrationGate,
        rejected: OrchestrationEvidence,
    ) -> int:
        key = f"run:{run.id}:kind:create_delegation_task:review_fix:gate:{gate.id}"
        if await self._existing_action_for_key(db, run.id, key) is not None:
            return 0
        source_task = await self._source_task_for_gate(db, goal.project_id, gate)
        if source_task is None or source_task.assigned_to is None:
            ask_key = f"run:{run.id}:kind:ask_human:review_fix:gate:{gate.id}"
            ask = await self._existing_action_for_key(db, run.id, ask_key)
            if ask is not None:
                return 0
            ask = await self.execute_ask_human_action(
                db,
                run_id=run.id,
                request={
                    "action_type": "ask_human",
                    "question": f"Rejected review needs a fix but no source task or assignee was found for gate: {gate.id}",
                    "work_function": "implementation",
                    "required_capabilities": ["implementation"],
                    "candidate_agent_ids": [],
                    "gate_id": str(gate.id),
                    "reason": "Rejected-review fix has no resolvable source task.",
                },
                idempotency_key=ask_key,
            )
            self._mark_run_blocked(goal, run)
            self._upsert_active_blocker(
                run,
                {
                    "kind": "review_fix_required",
                    "gate_id": str(gate.id),
                    "reason": "Rejected review needs a fix task but its source task could not be resolved.",
                    "decision_id": str(ask.target_id),
                },
            )
            return 1
        action = await self.execute_create_delegation_task_action(
            db,
            run_id=run.id,
            request={
                "action_type": "create_delegation_task",
                "agent_id": str(source_task.assigned_to),
                "work_function": self._task_work_function(source_task),
                "scope": (
                    "Fix the rejected review findings for the linked orchestration gate. "
                    "Keep the change scoped to the parent task deliverable."
                ),
                "inputs": [
                    f"Parent task: {source_task.id}",
                    f"Gate: {gate.id}",
                    f"Rejected review evidence: {rejected.id}",
                ],
                "deliverable": "A corrected work item ready for fresh review.",
                "forbidden_work": [
                    "Do not mark orchestration gates complete.",
                    "Do not edit unrelated artifacts.",
                ],
                "success_evidence": [
                    "Complete the fix task and provide concrete evidence for a new review.",
                ],
                "budget": {},
                "report_schema": {
                    "status": "done|blocked|failed",
                    "changes": "list[str]",
                    "evidence": "list[str]",
                    "notes": "str",
                },
                "parent_task_id": str(source_task.id),
            },
            idempotency_key=key,
        )
        if action.status == "completed" and action.target_id is not None:
            task = await db.get(Task, action.target_id)
            if task is not None:
                metadata = self._json_object_or_empty(task.metadata_)
                orchestration = self._json_object_or_empty(metadata.get("orchestration"))
                metadata["orchestration"] = {
                    **orchestration,
                    "plan_item_gate_id": str(gate.id),
                    "recovery_kind": "review_rejected_fix",
                }
                task.metadata_ = metadata
        return 1

    async def _orchestrated_tasks_for_run(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        *,
        statuses: list[str],
    ) -> list[Task]:
        result = await db.execute(
            select(Task)
            .where(
                Task.status.in_(statuses),
                Task.metadata_["orchestration"]["run_id"].as_string() == str(run_id),
            )
            .order_by(Task.updated_at.asc(), Task.id.asc())
        )
        return list(result.scalars().all())

    async def _failed_session_count(self, db: AsyncSession, task: Task) -> int:
        return await db.scalar(
            select(func.count(Session.id)).where(  # pylint: disable=not-callable
                Session.task_id == task.id,
                Session.status == "failed",
            )
        ) or 0

    async def _gate_id_for_task(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        task: Task,
    ) -> uuid.UUID | None:
        gate_ref = await self._task_gate_for_run(db, run_id, task)
        if gate_ref is None:
            return None
        return gate_ref[0].id

    async def _source_task_for_gate(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        gate: OrchestrationGate,
    ) -> Task | None:
        evidence = await self._evidence_for_gate(db, gate)
        for row in reversed(evidence):
            if row.source_type == "task" and row.source_id is not None:
                task = await TaskService().get(db, project_id, row.source_id)
                if task is not None:
                    return task
            if row.source_type in {"session", "review"} and row.source_id is not None:
                session = await db.get(Session, row.source_id)
                if session is not None and session.task_id is not None:
                    task = await TaskService().get(db, project_id, session.task_id)
                    if task is not None:
                        return task
        return None

    async def _producer_agent_ids_for_gate(
        self,
        db: AsyncSession,
        gate: OrchestrationGate,
    ) -> set[uuid.UUID]:
        evidence = await self._evidence_for_gate(db, gate)
        producers = {row.producer_agent_id for row in evidence if row.producer_agent_id is not None}
        configured = self._event_uuid(
            self._json_object_or_empty(gate.required_evidence).get("work_producer_agent_id")
        )
        if configured is not None:
            producers.add(configured)
        producers.update(
            producer for producer in (
                self._event_uuid(value)
                for value in self._string_list(
                    self._json_object_or_empty(gate.required_evidence).get("work_producer_agent_ids")
                )
            ) if producer is not None
        )
        return producers

    async def _latest_rejected_evidence_for_gate(
        self,
        db: AsyncSession,
        gate: OrchestrationGate,
    ) -> OrchestrationEvidence | None:
        evidence = await self._evidence_for_gate(db, gate)
        for row in reversed(evidence):
            if row.verdict == "rejected":
                return row
        return None

    async def _best_recovery_agent(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        work_function: str,
        *,
        required_capabilities: list[str],
        exclude_agent_ids: set[uuid.UUID],
        allowed_agent_ids: set[str] | None = None,
    ):
        fits = await OrchestrationRosterMapper().rank_agents(
            db,
            project_id,
            work_function,
            required_capabilities=required_capabilities,
        )
        for fit in fits:
            if fit.agent_id in exclude_agent_ids:
                continue
            if allowed_agent_ids is not None and str(fit.agent_id) not in allowed_agent_ids:
                continue
            if not fit.weak:
                return fit
        return None

    @staticmethod
    def _task_work_function(task: Task) -> str:
        metadata = OrchestrationService._json_object_or_empty(task.metadata_)
        orchestration = OrchestrationService._json_object_or_empty(metadata.get("orchestration"))
        return OrchestrationService._optional_string(orchestration.get("work_function")) or "implementation"

    @staticmethod
    def _mark_run_blocked(goal: OrchestrationGoal, run: OrchestrationRun) -> None:
        if goal.status == "active":
            goal.status = "blocked"
        if run.status == "running":
            run.status = "blocked"

    @staticmethod
    def _upsert_active_blocker(run: OrchestrationRun, blocker: Mapping[str, Any]) -> None:
        blocker_obj = {
            key: value
            for key, value in dict(blocker).items()
            if value is not None
        }
        key_parts = [
            str(blocker_obj.get("kind", "")),
            str(blocker_obj.get("task_id", "")),
            str(blocker_obj.get("gate_id", "")),
            str(blocker_obj.get("scope", "")),
        ]
        blocker_key = ":".join(key_parts)
        blockers = [
            item
            for item in run.active_blockers
            if not isinstance(item, Mapping)
            or ":".join(
                [
                    str(item.get("kind", "")),
                    str(item.get("task_id", "")),
                    str(item.get("gate_id", "")),
                    str(item.get("scope", "")),
                ]
            )
            != blocker_key
        ]
        run.active_blockers = blockers + [blocker_obj]

    @staticmethod
    def _remove_active_blocker_by_kind(run: OrchestrationRun, kind: str) -> None:
        run.active_blockers = [
            item
            for item in run.active_blockers
            if not (isinstance(item, Mapping) and str(item.get("kind", "")) == kind)
        ]

    @staticmethod
    def _has_active_blocker(run: OrchestrationRun, kind: str) -> bool:
        return any(
            isinstance(item, Mapping) and item.get("kind") == kind
            for item in run.active_blockers
        )

    @staticmethod
    def _remember_task_recovery(
        run: OrchestrationRun,
        task_id: uuid.UUID,
        recovery_kind: str,
        action_id: uuid.UUID,
    ) -> None:
        retry_state = OrchestrationService._json_object_or_empty(run.retry_state)
        tasks = retry_state.get("tasks")
        if not isinstance(tasks, Mapping):
            tasks = {}
        task_state = OrchestrationService._json_object_or_empty(tasks.get(str(task_id)))
        run.retry_state = {
            **retry_state,
            "tasks": {
                **dict(tasks),
                str(task_id): {
                    **task_state,
                    "last_recovery": recovery_kind,
                    "last_action_id": str(action_id),
                },
            },
        }

    async def validate_open_gates(self, db: AsyncSession, run_id: uuid.UUID) -> int:
        result = await db.execute(
            select(OrchestrationGate)
            .where(
                OrchestrationGate.run_id == run_id,
                OrchestrationGate.status == "open",
            )
            .order_by(OrchestrationGate.created_at.asc(), OrchestrationGate.id.asc())
        )
        changed = 0
        for gate in result.scalars().all():
            changed += 1 if await self._validate_gate(db, gate) else 0
        await db.flush()
        return changed

    async def _reconcile_outcome_gate_acceptance(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun
    ) -> int:
        """Reopen active Outcome gates accepted without a bound verifier session."""
        if goal.goal_type != "outcome" or run.status not in (*LLM_DECISION_RUN_STATUSES, "paused"):
            return 0
        gates = list((await db.scalars(
            select(OrchestrationGate).where(
                OrchestrationGate.run_id == run.id,
                OrchestrationGate.status.in_(("open", "accepted")),
            )
        )).all())
        repaired = 0
        for gate in gates:
            if not await self._is_outcome_plan_item_gate(db, run, gate):
                continue
            await self._required_evidence_for_gate(db, gate)
            if gate.status != "accepted" or await self._has_bound_outcome_verification_evidence(db, gate):
                continue
            for item in await self._evidence_for_gate(db, gate):
                if item.verdict == "accepted" and item.source_type != "verification":
                    item.verdict = "candidate"
                    item.evidence_metadata = {
                        **self._json_object_or_empty(item.evidence_metadata),
                        "repair_reason": "missing_bound_verification",
                    }
            gate.status = "open"
            gate.accepted_at = None
            gate.failure_reason = "Reopened: missing bound verification evidence"
            await emit_event_once(
                db,
                goal.project_id,
                GATE_REPAIRED_EVENT_TYPE,
                {"run_id": str(run.id), "gate_id": str(gate.id), "reason": "missing_bound_verification"},
                source="orchestrator",
                dedup_key=f"orchestration.gate-repair:run:{run.id}:gate:{gate.id}",
            )
            repaired += 1
        if repaired:
            await db.flush()
        return repaired

    async def _evidence_for_gate(self, db: AsyncSession, gate: OrchestrationGate) -> list[OrchestrationEvidence]:
        result = await db.execute(
            select(OrchestrationEvidence).where(
                OrchestrationEvidence.run_id == gate.run_id,
                OrchestrationEvidence.gate_id == gate.id,
            )
        )
        return sorted(result.scalars().all(), key=self._evidence_order_key)

    async def _validated_final_summary_payload(
        self,
        db: AsyncSession,
        gate: OrchestrationGate,
        evidence: list[OrchestrationEvidence],
    ) -> tuple[dict[str, Any] | None, str | None]:
        summary_evidence = next(
            (item for item in evidence if item.source_type == "session"),
            None,
        )
        if summary_evidence is None or summary_evidence.source_id is None:
            return None, "Final summary session evidence is missing"
        session = await db.get(Session, summary_evidence.source_id)
        if session is None:
            return None, "Final summary session is missing"
        if session.status != "completed":
            return None, "Final summary session is not completed"
        if not (session.output or "").strip():
            return None, "Final summary output is missing"
        if session.task_id is None or session.agent_id is None:
            return None, "Final summary session attribution is missing"
        task = await db.get(Task, session.task_id)
        if task is None:
            return None, "Final summary task is missing"
        task_metadata = self._json_object_or_empty(task.metadata_)
        orchestration = self._json_object_or_empty(task_metadata.get("orchestration"))
        if (
            orchestration.get("run_id") != str(gate.run_id)
            or orchestration.get("gate_id") != str(gate.id)
            or orchestration.get("work_function") != FINAL_SUMMARY_WORK_FUNCTION
            or orchestration.get("final_summary") is not True
        ):
            return None, "Final summary task linkage is invalid"
        if task.assigned_to != session.agent_id or summary_evidence.producer_agent_id != session.agent_id:
            return None, "Final summary session attribution is invalid"

        try:
            payload = json.loads(session.output)
        except (TypeError, json.JSONDecodeError):
            return None, "Final summary output must be valid JSON"
        if not isinstance(payload, Mapping):
            return None, "Final summary output must be a JSON object"
        raw_summary = payload.get("summary")
        if not isinstance(raw_summary, str) or not raw_summary.strip():
            return None, "Final summary text must be a non-empty string"
        summary = raw_summary.strip()

        run = await db.get(OrchestrationRun, gate.run_id)
        if run is None:
            return None, "Final summary run is missing"
        goal = await db.get(OrchestrationGoal, run.goal_id)
        if goal is None:
            return None, "Final summary goal is missing"
        declared_keys: list[str] = []
        for criterion in goal.success_criteria:
            if not isinstance(criterion, Mapping):
                return None, "Declared success criteria keys are invalid"
            raw_criterion_key = criterion.get("key")
            if not isinstance(raw_criterion_key, str) or not raw_criterion_key.strip():
                return None, "Declared success criteria keys are invalid"
            if raw_criterion_key != raw_criterion_key.strip():
                return None, "Declared success criteria keys must not contain surrounding whitespace"
            criterion_key = raw_criterion_key
            if criterion_key in declared_keys:
                return None, "Declared success criteria keys are invalid"
            declared_keys.append(criterion_key)

        raw_mappings = payload.get("criteria")
        if not isinstance(raw_mappings, list):
            return None, "Final summary criteria must be a list"
        mappings: dict[str, list[uuid.UUID]] = {}
        for raw_mapping in raw_mappings:
            if not isinstance(raw_mapping, Mapping):
                return None, "Final summary criterion mapping must be an object"
            raw_criterion_key = raw_mapping.get("criterion_key")
            if not isinstance(raw_criterion_key, str) or not raw_criterion_key.strip():
                return None, "Final summary criterion_key must be a non-empty string"
            if raw_criterion_key != raw_criterion_key.strip():
                return None, "Final summary criterion_key must not contain surrounding whitespace"
            criterion_key = raw_criterion_key
            if criterion_key in mappings:
                return None, "Final summary criterion mapping contains duplicate criterion_key"
            raw_evidence_ids = raw_mapping.get("evidence_ids")
            if not isinstance(raw_evidence_ids, list) or not raw_evidence_ids:
                return None, "Final summary evidence_ids must be a non-empty list"
            if any(not isinstance(value, str) for value in raw_evidence_ids):
                return None, "Final summary evidence IDs are invalid"
            evidence_ids = [self._event_uuid(value) for value in raw_evidence_ids]
            if any(evidence_id is None for evidence_id in evidence_ids):
                return None, "Final summary evidence IDs are invalid"
            mappings[criterion_key] = [
                evidence_id for evidence_id in evidence_ids if evidence_id is not None
            ]

        if set(mappings) != set(declared_keys):
            return None, "Final summary criterion coverage is incomplete"
        accepted_evidence = {
            item.id: item
            for item in (
                await db.scalars(
                    select(OrchestrationEvidence).join(
                        OrchestrationGate, OrchestrationGate.id == OrchestrationEvidence.gate_id
                    ).where(
                        OrchestrationEvidence.run_id == gate.run_id,
                        OrchestrationEvidence.verdict == "accepted",
                        OrchestrationEvidence.source_type == "verification",
                        OrchestrationGate.gate_type.in_((PLAN_ITEM_GATE_TYPE, "roadmap_integration")),
                    )
                )
            ).all()
        }
        if goal.goal_type == "roadmap":
            integration_ids = {
                current.id for current in await self._current_run_gates(db, run.id)
                if current.gate_type == "roadmap_integration" and current.status == "accepted"
            }
            accepted_evidence = {
                evidence_id: item for evidence_id, item in accepted_evidence.items()
                if item.gate_id in integration_ids
            }
        for criterion_key, evidence_ids in mappings.items():
            for evidence_id in evidence_ids:
                item = accepted_evidence.get(evidence_id)
                source_gate = await db.get(OrchestrationGate, item.gate_id) if item else None
                linked_keys = self._string_list(
                    self._json_object_or_empty(source_gate.required_evidence).get("success_criterion_keys")
                ) if source_gate else []
                if criterion_key not in linked_keys:
                    return None, "Final summary evidence must be accepted verification linked to its criterion"

        unresolved_gaps = payload.get("unresolved_gaps", [])
        if not isinstance(unresolved_gaps, list) or any(
            not isinstance(item, str) for item in unresolved_gaps
        ):
            return None, "Final summary unresolved_gaps must be a list of strings"
        return {
            "summary": summary,
            "criteria": [
                {
                    "criterion_key": criterion_key,
                    "evidence_ids": [str(evidence_id) for evidence_id in mappings[criterion_key]],
                }
                for criterion_key in declared_keys
            ],
            "unresolved_gaps": unresolved_gaps,
        }, None

    async def _final_summary_evidence_failure(
        self,
        db: AsyncSession,
        gate: OrchestrationGate,
        evidence: list[OrchestrationEvidence],
    ) -> str | None:
        _, failure = await self._validated_final_summary_payload(db, gate, evidence)
        return failure

    async def _validate_gate(self, db: AsyncSession, gate: OrchestrationGate) -> bool:
        evidence = await self._evidence_for_gate(db, gate)
        latest_evidence = self._latest_evidence_by_source(evidence)
        rejected = next((item for item in latest_evidence if item.verdict == "rejected"), None)
        if rejected is not None:
            self._fail_gate(gate, f"Rejected evidence from {rejected.source_type}")
            return True

        required_evidence = await self._required_evidence_for_gate(db, gate)
        configured_source_types = required_evidence.get("required_source_types", [])
        required_source_types = [
            source_type
            for source_type in (self._optional_string(value) for value in configured_source_types)
            if source_type is not None
        ]
        configured_min_count = required_evidence.get("min_count")
        min_count = (
            configured_min_count if isinstance(configured_min_count, int) and configured_min_count > 0 else None
        )
        min_count = max(min_count or len(required_source_types) or 1, len(required_source_types))
        selected, missing = self._select_required_evidence(
            [item for item in latest_evidence if item.verdict in {"candidate", "accepted"}],
            required_source_types,
            min_count,
        )
        if missing:
            gate.failure_reason = f"Missing evidence: {', '.join(missing)}"
            return False
        if len(selected) < min_count:
            gate.failure_reason = "Missing evidence"
            return False

        run = await db.get(OrchestrationRun, gate.run_id)
        if (
            run is not None
            and (
                await self._is_outcome_plan_item_gate(db, run, gate)
                or gate.gate_type == "roadmap_integration"
            )
            and not await self._has_bound_outcome_verification_evidence(db, gate)
        ):
            gate.failure_reason = "Independent verification evidence is missing"
            return False

        if gate.gate_type == FINAL_SUMMARY_GATE_TYPE:
            summary_failure = await self._final_summary_evidence_failure(db, gate, selected)
            if summary_failure is not None:
                self._fail_gate(gate, summary_failure)
                return True

        stale_reason = self._stale_evidence_reason(selected)
        if stale_reason is not None:
            self._fail_gate(gate, stale_reason)
            return True

        independence_reason = await self._independence_failure_reason(db, gate, selected)
        if independence_reason is not None:
            if gate.gate_type == "roadmap_integration":
                gate.failure_reason = independence_reason
                return False
            run = await db.get(OrchestrationRun, gate.run_id)
            if run is not None and await self._is_outcome_plan_item_gate(db, run, gate):
                goal = await db.get(OrchestrationGoal, run.goal_id)
                if goal is not None and goal.goal_type == "roadmap":
                    gate.failure_reason = independence_reason
                    return False
            self._fail_gate(gate, independence_reason)
            return True

        self._accept_gate(gate, selected)
        return True

    def _select_required_evidence(
        self,
        evidence: list[OrchestrationEvidence],
        required_source_types: list[str],
        min_count: int,
    ) -> tuple[list[OrchestrationEvidence], list[str]]:
        selected: list[OrchestrationEvidence] = []
        missing: list[str] = []
        latest_evidence = list(reversed(evidence))
        for source_type in required_source_types:
            match = next(
                (item for item in latest_evidence if item.source_type == source_type and item not in selected),
                None,
            )
            if match is None:
                missing.append(source_type)
            else:
                selected.append(match)
        for item in latest_evidence:
            if len(selected) >= min_count:
                break
            if item not in selected:
                selected.append(item)
        return selected, missing

    def _stale_evidence_reason(self, evidence: list[OrchestrationEvidence]) -> str | None:
        work_evidence = [item for item in evidence if item.source_type in {"artifact", "task"}]
        if not work_evidence:
            return None
        work_seq = max((seq for item in work_evidence if (seq := self._evidence_event_seq(item)) is not None), default=None)
        work_created_at = max((item.created_at for item in work_evidence), default=None)
        for item in evidence:
            if item.source_type in {"artifact", "task"}:
                continue
            seq = self._evidence_event_seq(item)
            if work_seq is not None and seq is not None:
                if seq <= work_seq:
                    return "Evidence is stale"
                continue
            if work_created_at is not None and item.created_at <= work_created_at:
                return "Evidence is stale"
        return None

    def _latest_evidence_by_source(self, evidence: list[OrchestrationEvidence]) -> list[OrchestrationEvidence]:
        latest_by_source: dict[tuple[str, uuid.UUID], OrchestrationEvidence] = {}
        for item in evidence:
            latest_by_source[(item.source_type, item.source_id)] = item
        return sorted(latest_by_source.values(), key=self._evidence_order_key)

    def _evidence_order_key(self, evidence: OrchestrationEvidence) -> tuple[bool, int, datetime, str]:
        seq = self._evidence_event_seq(evidence)
        return (seq is None, seq or 0, evidence.created_at, str(evidence.id))

    async def _required_evidence_for_gate(
        self, db: AsyncSession, gate: OrchestrationGate
    ) -> dict[str, Any]:
        required_evidence = self._json_object_or_empty(gate.required_evidence)
        if gate.gate_type != PLAN_ITEM_GATE_TYPE:
            return required_evidence
        run = await db.get(OrchestrationRun, gate.run_id)
        if run is None or self._json_object_or_empty(run.plan_state).get("status") != "accepted":
            return required_evidence
        goal = await db.get(OrchestrationGoal, run.goal_id) if run is not None else None
        if goal is None or goal.goal_type != "outcome":
            return required_evidence
        normalized = {
            **required_evidence,
            "required_source_types": ["task", "verification"],
            "min_count": 2,
            "requires_independent_agent": True,
        }
        if normalized != required_evidence:
            gate.required_evidence = normalized
        return normalized

    async def _independence_failure_reason(
        self,
        db: AsyncSession,
        gate: OrchestrationGate,
        evidence: list[OrchestrationEvidence],
    ) -> str | None:
        required_evidence = self._json_object_or_empty(gate.required_evidence)
        if required_evidence.get("requires_independent_agent") is not True:
            return None
        if gate.gate_type == "roadmap_integration":
            producer_ids = {
                self._event_uuid(value)
                for value in self._string_list(required_evidence.get("work_producer_agent_ids"))
            }
            producer_ids.discard(None)
            verifier_evidence = [item for item in evidence if item.source_type == "verification"]
            if not verifier_evidence:
                return "Independent verification evidence is missing"
            if any(item.producer_agent_id is None for item in verifier_evidence):
                return "Independent verification producer is missing"
            if any(item.producer_agent_id in producer_ids for item in verifier_evidence):
                return "Independent integration verification must come from a different agent"
            return None
        run = await db.get(OrchestrationRun, gate.run_id)
        if run is not None and await self._is_outcome_plan_item_gate(db, run, gate):
            producer_agent_id = await self._plan_item_producer_agent_id(db, gate)
            if producer_agent_id is None:
                return "Outcome plan item producer is missing"
            producer_evidence = [
                item for item in evidence
                if item.source_type == "task" and item.producer_agent_id == producer_agent_id
            ]
            reviewer_evidence = [item for item in evidence if item.source_type == "verification"]
            if not producer_evidence:
                return "Outcome plan item producer evidence is missing"
            if not reviewer_evidence:
                return "Independent verification evidence is missing"
            if any(item.producer_agent_id is None for item in reviewer_evidence):
                return "Independent verification producer is missing"
            if not any(item.producer_agent_id != producer_agent_id for item in reviewer_evidence):
                return "Independent verification must come from a different agent"
            return None
        work_evidence = [item for item in evidence if item.source_type in {"artifact", "task"}]
        verifier_evidence = [item for item in evidence if item.source_type not in {"artifact", "task"}]
        configured_work_agent_id = self._event_uuid(required_evidence.get("work_producer_agent_id"))
        if not verifier_evidence:
            return "Independent verification evidence is missing"
        if any(item.producer_agent_id is None for item in verifier_evidence):
            return "Independent verification producer is missing"
        if any(item.producer_agent_id is None for item in work_evidence):
            return "Independent verification producer is missing"
        work_agent_ids = {item.producer_agent_id for item in work_evidence}
        if configured_work_agent_id is not None:
            work_agent_ids.add(configured_work_agent_id)
        if not work_agent_ids:
            return "Independent verification evidence is missing"
        verifier_agent_ids = {item.producer_agent_id for item in verifier_evidence}
        if work_agent_ids & verifier_agent_ids:
            return "Independent verification must come from a different agent"
        return None

    async def _plan_item_producer_agent_id(
        self, db: AsyncSession, gate: OrchestrationGate
    ) -> uuid.UUID | None:
        task = await self._plan_item_producer_task(db, gate)
        return task.assigned_to if task is not None else None

    async def _plan_item_producer_task(
        self, db: AsyncSession, gate: OrchestrationGate
    ) -> Task | None:
        tasks = await self._orchestrated_tasks_for_run(
            db,
            gate.run_id,
            statuses=["backlog", "ready", "in_progress", "blocked", "done", "failed", "cancelled"],
        )
        for task in tasks:
            orchestration = self._json_object_or_empty(
                self._json_object_or_empty(task.metadata_).get("orchestration")
            )
            if (
                orchestration.get("plan_item_gate_id") == str(gate.id)
                and self._optional_string(orchestration.get("plan_item_id")) is not None
            ):
                return task
        return None

    @staticmethod
    def _accept_gate(gate: OrchestrationGate, evidence: list[OrchestrationEvidence]) -> None:
        gate.status = "accepted"
        gate.failure_reason = None
        gate.accepted_at = gate.accepted_at or _utcnow()
        gate.failed_at = None
        for item in evidence:
            if item.verdict == "candidate":
                item.verdict = "accepted"

    @staticmethod
    def _fail_gate(gate: OrchestrationGate, reason: str) -> None:
        gate.status = "failed"
        gate.failure_reason = reason
        gate.failed_at = gate.failed_at or _utcnow()
        gate.accepted_at = None

    @staticmethod
    def _evidence_event_seq(evidence: OrchestrationEvidence) -> int | None:
        metadata = OrchestrationService._json_object_or_empty(evidence.evidence_metadata)
        seq = metadata.get("event_seq")
        if isinstance(seq, int):
            return seq
        if isinstance(seq, str):
            try:
                return int(seq)
            except ValueError:
                return None
        return None

    @staticmethod
    def _tick_noop_result(
        run: OrchestrationRun,
        effectiveness_review_process: dict | None = None,
        goal_closeout_process: dict | None = None,
    ) -> dict:
        return {
            "run_id": run.id,
            "status": run.status,
            "processed_events": 0,
            "evidence_created": 0,
            "gates_validated": 0,
            "recoveries_created": 0,
            "final_summary_action_id": None,
            "run_completed": False,
            "completion_action_id": None,
            "tick_emitted": False,
            "event_cursor": run.event_cursor,
            "baseline_process": None,
            "manager_selection_process": None,
            "agent_definition_review_process": None,
            "team_hierarchy_process": None,
            "effectiveness_review_process": effectiveness_review_process,
            "goal_closeout_process": goal_closeout_process,
            "authority_interview": None,
        }

    @staticmethod
    async def _hard_stop_active(db: AsyncSession, goal_id: uuid.UUID) -> bool:
        hard_stop_id = await db.scalar(
            select(OrchestrationWarning.id)
            .where(
                OrchestrationWarning.goal_id == goal_id,
                OrchestrationWarning.severity == "hard_stop",
                OrchestrationWarning.active.is_(True),
            )
            .limit(1)
        )
        return hard_stop_id is not None

    async def _reject_stale_steering_decision(self, db, decision_id: uuid.UUID) -> None:
        """Durably reject stale work without racing SQLite's sole writer."""
        statement = (
            update(OrchestrationDecision)
            .where(OrchestrationDecision.id == decision_id)
            .values(validator_status="rejected", rejection_reason="stale_steering_versions")
        )
        if db.bind is not None and db.bind.dialect.name == "sqlite":
            # SQLite cannot safely use a second writer while the caller owns a
            # transaction.  Mirror _persist_reserved_action_failure: only an
            # AUTOBEGIN transaction belongs to this flow and may be committed.
            await db.execute(statement)
            transaction = db.sync_session.get_transaction()
            caller_owns_transaction = (
                transaction is not None
                and transaction.origin != SessionTransactionOrigin.AUTOBEGIN
            )
            if caller_owns_transaction or db.in_nested_transaction():
                await db.flush()
            else:
                await db.commit()
            return

        session_factory = AsyncSessionLocal
        if db.bind is not None:
            session_factory = async_sessionmaker(
                db.bind,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        async with session_factory() as audit_db:
            result = await audit_db.execute(statement)
            await audit_db.commit()
            if result.rowcount:
                return

        # A newly-recorded decision can still belong only to the caller's
        # transaction, so an independent audit session cannot see it yet.
        await db.execute(statement)
        await db.flush()

    async def _dispatch_execution_decision(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        decision: OrchestrationDecision,
        *,
        stale_retry: bool = False,
    ) -> OrchestrationAction | None:
        """Dispatch one runtime decision, recording a wait when it cannot act."""
        from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher

        dispatcher = OrchestrationDecisionDispatcher(self)
        try:
            action = await dispatcher.dispatch(db, run, decision)
        except Exception as exc:
            from huddleroom.services.orchestration_steering import SteeringVersionsChanged

            if not isinstance(exc, SteeringVersionsChanged):
                raise
            await self._reject_stale_steering_decision(db, decision.id)
            if stale_retry:
                return None
            refreshed = await self.request_llm_decision(db, run.id)
            return await self._dispatch_execution_decision(
                db, run, refreshed, stale_retry=True,
            )
        if action is not None:
            return action

        parsed = self._json_object_or_empty(decision.parsed_decision)
        reason = (
            decision.rejection_reason
            or self._optional_string(parsed.get("reason"))
            or "No executable coordination action is available."
        )
        wait_identity = {
            "validator_status": decision.validator_status,
            **{key: value for key, value in parsed.items() if key != "reason"},
        }
        return await self.execute_noop_action(
            db,
            run_id=run.id,
            request={"action_type": "noop", "reason": reason},
            idempotency_key=dispatcher.action_key(run.id, "noop", wait_identity),
            decision_id=decision.id,
        )

    async def _advance_authorized_execution(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun
    ) -> dict:
        """One forward step of the authorized Outcome runner. Exactly one action
        (or one explicit wait) per call. Code releases deterministic work; the
        LLM only chooses when code cannot."""
        from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher

        if self._has_active_blocker(run, "plan_criterion_integrity"):
            return {"step": "needs_attention"}
        if await self._sync_measured_budget_exhaustion(db, goal, run):
            await db.flush()
            return {"step": "budget_exhausted"}
        if self._has_active_blocker(run, "budget_measurement"):
            await db.flush()
            return {"step": "needs_attention"}

        plan_state = self._json_object_or_empty(run.plan_state)
        plan_status = plan_state.get("status")

        # (1)/(2) plan not yet accepted -> ask + dispatch one decision.
        if plan_status != "accepted":
            decision = await self.request_llm_decision(db, run.id)
            decision_action_type = self._json_object_or_empty(decision.parsed_decision).get("action_type")
            if goal.goal_type == "roadmap" and decision_action_type == "accept_plan":
                from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService

                request = self._json_object_or_empty(decision.parsed_decision)
                artifact = await self._plan_artifact_for_run(
                    db, run, self._required_uuid(request.get("plan_artifact_id"), "plan_artifact_id")
                )
                roadmap = OrchestrationRoadmapService(self)
                try:
                    approval = await roadmap.ensure_plan_approval(db, goal, run, artifact)
                except HTTPException as exc:
                    if exc.status_code not in {409, 422}:
                        raise
                    self._set_plan_revision_required(run, str(exc.detail))
                    await db.flush()
                    return {"step": "plan_revision_required", "reason": str(exc.detail)}
                wait_state = {
                    "artifact_id": str(artifact.id),
                    "fingerprint": self._accepted_plan_fingerprint([
                        item.model_dump(mode="json") for item in roadmap.parse_items(artifact, goal)
                    ]),
                    "approval_id": str(approval.id),
                    "approval_kind": "gate" if isinstance(approval, OrchestrationGate) else "authority_decision",
                }
                run.plan_state = {**self._json_object_or_empty(run.plan_state), "roadmap_authority_wait": wait_state}
                if isinstance(approval, OrchestrationAuthorityDecision):
                    if approval.status == "pending":
                        return {"step": "waiting_plan_authority"}
                    if approval.selected_option != "approve":
                        self._set_plan_revision_required(run, "Roadmap plan approval was rejected")
                        await db.flush()
                        return {"step": "plan_revision_required"}
                    try:
                        await roadmap._require_plan_approval(db, goal, run, wait_state["fingerprint"], None)
                    except HTTPException as exc:
                        self._set_plan_revision_required(run, str(exc.detail))
                        await db.flush()
                        return {"step": "plan_revision_required"}
                elif approval.status != "accepted":
                    action = await self.execute_request_verification_action(
                        db, run.id,
                        {"action_type": "request_verification", "gate_id": str(approval.id), "work_function": "validation"},
                        f"run:{run.id}:kind:request_verification:roadmap_plan:{approval.id}", decision_id=decision.id,
                    )
                    return {"step": "waiting_plan_authority", "action_id": str(action.id)}
            try:
                action = await self._dispatch_execution_decision(db, run, decision)
            except HTTPException as exc:
                from huddleroom.services.orchestration_roadmap_service import RoadmapReplanMeasurementAttention
                if isinstance(exc, RoadmapReplanMeasurementAttention):
                    return {"step": "waiting", "reason": "needs_attention"}
                if exc.status_code == 409 and exc.detail in {"budget_wait", "needs_attention"}:
                    return {"step": "waiting", "reason": exc.detail}
                if (
                    decision_action_type != "accept_plan"
                    or plan_status not in {"requested", "revision_requested"}
                ):
                    raise
                await db.refresh(run)
                revised_plan = self._json_object_or_empty(run.plan_state)
                if revised_plan.get("status") != "revision_required":
                    raise
                recovery_key = OrchestrationDecisionDispatcher.action_key(
                    run.id,
                    decision_action_type,
                    self.canonical_decision_request(
                        decision_action_type,
                        self._json_object_or_empty(decision.parsed_decision),
                        run_id=run.id,
                    ),
                )
                recovery_key, recovery_steering, recovery_versions, recovery_active_ids, recovery_goal, recovery_run = (
                    await self._steering_action_fence(db, run.id, recovery_key, decision.id)
                )
                failed_action = await self._existing_action_for_key(db, run.id, recovery_key)
                if failed_action is None or failed_action.status != "failed":
                    raise
                await self._fence_steering_action(
                    db, failed_action, recovery_steering, recovery_versions, recovery_active_ids,
                    recovery_goal, recovery_run, newly_reserved=False,
                )
                return {
                    "step": "plan_revision_required",
                    "reason": revised_plan.get("revision_reason"),
                }
            if action is None:
                return {"step": "waiting", "reason": "stale_steering_versions"}
            return {"step": "plan_decision", "action_id": str(action.id)}

        # (3) plan accepted -> release dependency-ready work under the two-task cap.
        # expand_accepted_plan is NOT called here: it expands every plan item
        # eagerly in one shot, which would bypass the cap entirely. Expansion is
        # driven one item per free slot, only through _release_ready_work.
        try:
            released = await self._release_ready_work(db, goal, run)
        except HTTPException as exc:
            if exc.status_code == 409 and exc.detail in {"budget_wait", "needs_attention"}:
                return {"step": "waiting", "reason": exc.detail}
            raise
        if self._has_active_blocker(run, "plan_criterion_integrity") or run.status not in LLM_DECISION_RUN_STATUSES:
            return {"step": "needs_attention"}
        if released:
            return {"step": "release_work", "released": released}

        # (4) nothing deterministic -> one LLM next-action (verification / follow-up /
        # meeting / wait). noop/ask_human/pause_run IS the explicit wait/escalation.
        decision = await self.request_llm_decision(db, run.id)
        action = await self._dispatch_execution_decision(db, run, decision)
        if action is None:
            return {"step": "waiting", "reason": "stale_steering_versions"}
        return {"step": "next_action", "action_id": str(action.id)}

    RELEASE_TWO_TASK_CAP = 2

    async def _release_ready_work(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun
    ) -> int:
        try:
            items = await self._accepted_plan_items(db, run)
        except HTTPException as exc:
            self._block_plan_integrity(goal, run, str(exc.detail))
            return 0
        try:
            self._validate_plan_criterion_links(goal, items)
        except HTTPException as exc:
            self._block_plan_integrity(goal, run, str(exc.detail))
            return 0
        await self._backfill_plan_criterion_links(db, run, items)

        in_flight = await self._orchestrated_tasks_for_run(
            db, run.id, statuses=["backlog", "ready", "in_progress", "blocked"]
        )
        work_in_flight = [
            t for t in in_flight
            if self._task_work_function(t) not in (PLAN_WORK_FUNCTION, FINAL_SUMMARY_WORK_FUNCTION)
        ]
        slots = self.RELEASE_TWO_TASK_CAP - len(work_in_flight)
        if slots <= 0:
            return 0

        released = 0
        for item in items:
            if released >= slots:
                break
            key = f"run:{run.id}:kind:release_item:{item.id}"
            if await self._existing_action_for_key(db, run.id, key) is not None:
                continue
            if not await self._item_dependencies_accepted(db, run, item):
                continue
            await self.execute_expand_plan_item_action(
                db,
                run_id=run.id,
                request={"action_type": "expand_plan_item", "plan_item_id": item.id,
                         "work_function": item.work_function},
                idempotency_key=self._expand_plan_item_idempotency_key(run.id, item.id),
            )
            # expand_plan_item creates the delegation task for this item; mark the
            # release so the cap accounting and idempotency hold across ticks.
            await self.reserve_action(
                db, run_id=run.id, idempotency_key=key,
                action_type="release_item", request={"plan_item_id": item.id},
            )
            released += 1
        return released

    async def _item_dependencies_accepted(
        self, db: AsyncSession, run: OrchestrationRun, item: "OrchestrationPlanItem"
    ) -> bool:
        for dep_key in (item.depends_on or []):
            gate_key = dep_key if str(dep_key).startswith("plan_item:") else self._plan_item_gate_key(dep_key)
            result = await db.execute(
                select(OrchestrationGate).where(
                    OrchestrationGate.run_id == run.id,
                    OrchestrationGate.success_criterion_key == gate_key,
                )
            )
            gates = list(result.scalars().all())
            if not gates or not any(g.status == "accepted" for g in gates):
                return False
        return True

    async def tick(self, db: AsyncSession, run_id: uuid.UUID, *, local_only: bool = False) -> dict:
        """No-op control-loop tick: consume new events, advance cursor, emit once."""
        # Preserve a caller's explicit transaction, but own SQLAlchemy's implicit
        # AUTOBEGIN transaction (including route preflight reads) so commit stays
        # inside the goal lock.
        transaction = db.sync_session.get_transaction()
        caller_owns_transaction = (
            transaction is not None
            and transaction.origin != SessionTransactionOrigin.AUTOBEGIN
        )

        # Fetch goal_id cheaply first to acquire lock before trusting run/goal status (fix #5).
        run_temp = await db.get(OrchestrationRun, run_id)
        if run_temp is None:
            raise HTTPException(status_code=404, detail="Orchestration run not found")
        goal_id = run_temp.goal_id

        # Lock goal-level to serialize concurrent ticks and force-starts on the same goal.
        # This prevents: (a) baseline_ready stale-gate race; (b) concurrent completion race.
        async with self._lock_goal_for_baseline_transition(db, goal_id):
            # Reload run and goal inside lock with populate_existing=True (fix #5).
            run = await db.execute(
                select(OrchestrationRun)
                .where(OrchestrationRun.id == run_id)
                .execution_options(populate_existing=True)
            )
            run = run.scalar_one_or_none()
            if run is None:
                raise HTTPException(status_code=404, detail="Orchestration run not found")

            goal = await db.execute(
                select(OrchestrationGoal)
                .where(OrchestrationGoal.id == goal_id)
                .execution_options(populate_existing=True)
            )
            goal = goal.scalar_one_or_none()
            if goal is None:
                raise HTTPException(status_code=404, detail="Orchestration goal not found")
            from huddleroom.services.orchestration_steering import OrchestrationSteeringService
            steering_processing = await OrchestrationSteeringService().process_pending(db, goal, run)
            if local_only:
                local_liveness = await self.supervision.reconcile_local(
                    db, goal, run, allow_release=False
                )
                state = dict(run.supervision_state or {})
                state["evaluated_at"] = _utcnow().isoformat()
                state["judgment_dirty"] = bool(state.get("judgment_dirty", False))
                from huddleroom.services.orchestration_supervision_context import OrchestrationSupervisionContextBuilder
                state["context_fingerprint"] = await OrchestrationSupervisionContextBuilder().fingerprint(db, goal, run)
                state["needs_judgment"] = local_liveness["outcome"] == "continue"
                run.supervision_state = state
                await db.flush()
                if not caller_owns_transaction:
                    await db.commit()
                return {
                    **self._tick_noop_result(run), "local_liveness": local_liveness, "local_only": True,
                    "steering": {
                        "processed_request_ids": [str(item) for item in steering_processing.processed_request_ids],
                        "applied_request_ids": [str(item) for item in steering_processing.applied_request_ids],
                    },
                }
            project_id = goal.project_id
            await ProjectService().lock_workspace_boundary(db, project_id)
            await ProjectService().require_runnable_project(db, project_id)
            from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService
            await OrchestrationRoadmapService(self).reconcile_claim_blockers(db, goal, run)

            # NOT ACTIVE_RUN_STATUSES: that set includes "paused" because a paused run
            # still holds the one-active-per-goal slot. Tickable != active — spec §6.3
            # step 4 stops on paused. Do not collapse this into the constant.
            if run.status not in LLM_DECISION_RUN_STATUSES:
                await self._reconcile_outcome_gate_acceptance(db, goal, run)
                effectiveness_review_process = None
                goal_closeout_process = None
                current_closeout = await OrchestrationProcessService().get_current(
                    db, run.goal_id, "goal_closeout"
                )
                if current_closeout is not None and current_closeout.status == "completed":
                    goal_closeout_process = GoalCloseoutProcess._completed_summary(current_closeout)
                elif current_closeout is not None and current_closeout.status == "skipped":
                    goal_closeout_process = {
                        "status": "skipped",
                        "mode": "completion",
                        "completion_authorized": False,
                    }
                if run.status == "paused":
                    current_review = await OrchestrationProcessService().get_current(
                        db, run.goal_id, "effectiveness_review"
                    )
                    if (
                        current_review is not None
                        and current_review.status in {"running", "waiting_decision"}
                    ):
                        await ProjectService().lock_workspace_boundary(db, project_id)
                        await ProjectService().require_runnable_project(db, project_id)
                        effectiveness_review_process = (
                            await EffectivenessReviewProcess().advance(db, goal, run)
                        )
                        await db.flush()
                if not caller_owns_transaction:
                    await db.commit()
                return self._tick_noop_result(
                    run, effectiveness_review_process, goal_closeout_process
                )

            if await self._hard_stop_active(db, goal.id):
                return self._tick_noop_result(run)

            if goal.goal_type == "continuous" and run.phase in {"waiting_activation", "authorized"}:
                if run.phase == "waiting_activation":
                    if not caller_owns_transaction:
                        await db.commit()
                    return self._tick_noop_result(run)
                from huddleroom.services.orchestration_continuous_service import OrchestrationContinuousService
                authorized_execution = await OrchestrationContinuousService(self).advance(db, goal, run)
                await db.flush()
                if not caller_owns_transaction:
                    await db.commit()
                return {
                    **self._tick_noop_result(run),
                    "authorized_execution": authorized_execution,
                    "run_completed": run.status == "completed",
                }

            if self._has_active_blocker(run, "repeated_failure"):
                if not caller_owns_transaction:
                    await db.commit()
                return self._tick_noop_result(run)

            if any(
                isinstance(blocker, Mapping)
                and blocker.get("kind") == "goal_definition_analyzer_error"
                for blocker in run.active_blockers
            ):
                return self._tick_noop_result(run)

            if run.phase == "authorized" and await self._sync_budget_exhaustion_attention(db, goal, run):
                # An exhausted authorized run keeps ingesting terminal evidence and
                # validating gates, but must not spend on baseline/recovery/review
                # work while a human decides whether to override the budget stop.
                new_events = await self._new_events(db, project_id, run.event_cursor)
                new_cursor = new_events[-1].seq if new_events else run.event_cursor
                cursor_token = str(new_cursor) if new_cursor is not None else "genesis"
                evidence_created = await self._ingest_evidence_from_events(db, run, new_events)
                await self._reconcile_outcome_gate_acceptance(db, goal, run)
                gates_validated = await self.validate_open_gates(db, run.id)
                await db.refresh(run)
                await db.refresh(goal)
                _, created = await emit_event_once(
                    db,
                    project_id,
                    "orchestration.tick",
                    {
                        "run_id": str(run_id),
                        "goal_id": str(run.goal_id),
                        "processed_events": len(new_events),
                        "evidence_created": evidence_created,
                        "gates_validated": gates_validated,
                        "recoveries_created": 0,
                        "final_summary_action_id": None,
                        "completion_action_id": None,
                        "event_cursor": cursor_token,
                        "authorized_execution": {"step": "budget_exhausted"},
                    },
                    source="orchestrator",
                    dedup_key=f"orchestration.tick:run:{run_id}:cursor:{cursor_token}",
                )
                run.event_cursor = new_cursor
                await db.flush()
                if not caller_owns_transaction:
                    await db.commit()
                return {
                    "run_id": run_id,
                    "status": run.status,
                    "processed_events": len(new_events),
                    "evidence_created": evidence_created,
                    "gates_validated": gates_validated,
                    "recoveries_created": 0,
                    "final_summary_action_id": None,
                    "run_completed": False,
                    "completion_action_id": None,
                    "tick_emitted": created,
                    "event_cursor": new_cursor,
                    "baseline_process": None,
                    "manager_selection_process": None,
                    "agent_definition_review_process": None,
                    "team_hierarchy_process": None,
                    "effectiveness_review_process": None,
                    "goal_closeout_process": None,
                    "authority_interview": None,
                    "authorized_execution": {"step": "budget_exhausted"},
                }

            # Spec 6.3: deterministic baseline process steps advance before any
            # LLM coordination decision, in order -- B (manager_selection) only
            # once A (goal_definition) is terminal. Never blocks the tick loop:
            # a parked process returns its summary and the bookkeeping below
            # still runs. It DOES block this goal's own forward progress -- see
            # the can_finish gate below.
            await ProjectService().lock_workspace_boundary(db, project_id)
            await ProjectService().require_runnable_project(db, project_id)
            child_delta_ready = (
                goal.parent_goal_id is not None
                and self._json_object_or_empty(
                    self._json_object_or_empty(run.plan_state).get("child_delta_baseline")
                ).get("status") == "accepted"
            )
            baseline_process = {"status": "skipped"} if child_delta_ready else await GoalDefinitionProcess().advance(db, goal, run)
            goal_definition_ready = (
                self._process_terminal(baseline_process)
                and not baseline_process.get("clarification_limit_reached", False)
            )
            manager_selection_process = {"status": "skipped"} if child_delta_ready else None
            if goal_definition_ready and not child_delta_ready:
                await ProjectService().lock_workspace_boundary(db, project_id)
                await ProjectService().require_runnable_project(db, project_id)
                manager_selection_process = await ManagerSelectionProcess().advance(db, goal, run)
            manager_selection_ready = self._process_terminal(manager_selection_process)
            agent_definition_review_process = {"status": "skipped"} if child_delta_ready else None
            if manager_selection_ready and not child_delta_ready:
                if any(
                    isinstance(blocker, Mapping)
                    and blocker.get("kind") == "agent_definition_review_analyzer_error"
                    for blocker in run.active_blockers
                ):
                    return self._tick_noop_result(run)

                await ProjectService().lock_workspace_boundary(db, project_id)
                await ProjectService().require_runnable_project(db, project_id)
                agent_definition_review_process = await AgentDefinitionReviewProcess().advance(
                    db, goal, run
                )
            agent_definition_review_ready = self._process_terminal(
                agent_definition_review_process
            )
            team_hierarchy_process = {"status": "skipped"} if child_delta_ready else None
            if agent_definition_review_ready and not child_delta_ready:
                await ProjectService().lock_workspace_boundary(db, project_id)
                await ProjectService().require_runnable_project(db, project_id)
                team_hierarchy_process = await TeamHierarchyProcess().advance(
                    db, goal, run
                )

            # Cross-cutting D (spec 6.3, 8.3.1, 10.6): deliver every pending
            # non-human decision to its agent and record any report that has
            # landed, then write the human checkpoint's deferred questions to
            # memory. Runs every tick, independent of baseline_ready --
            # authority_interview is cross-cutting, not a gated process.
            await ProjectService().lock_workspace_boundary(db, project_id)
            await ProjectService().require_runnable_project(db, project_id)
            authority_interview_summary = await self._sync_agent_authority_decisions(db, goal, run)
            pending_for_checkpoint = await OrchestrationAuthorityDecisionService().list_decisions(
                db, goal.id, status="pending"
            )
            _, deferred_questions = build_checkpoint(
                pending_for_checkpoint,
                max_questions=settings.orchestration_checkpoint_max_questions,
            )
            await sync_deferred_questions_memory(db, goal, run, deferred_questions)

            baseline_ready = (
                agent_definition_review_ready
                and self._process_terminal(team_hierarchy_process)
            )

            # Authorization gate (spec: "Goal Baseline establishes that a goal
            # is safe and ready; it does not authorize work."). When baseline
            # first becomes terminal, park the run in `ready` and wait for an
            # explicit Start. The reconciler must never execute an unauthorized
            # run, so no plan / summary / closeout / completion runs below until
            # phase == "authorized".
            if baseline_ready and run.phase == "baseline":
                run.phase = "ready"
                await db.flush()

            new_events = await self._new_events(db, project_id, run.event_cursor)
            new_cursor = new_events[-1].seq if new_events else run.event_cursor
            cursor_token = str(new_cursor) if new_cursor is not None else "genesis"
            evidence_created = await self._ingest_evidence_from_events(db, run, new_events)
            await self._reconcile_outcome_gate_acceptance(db, goal, run)
            gates_validated = await self.validate_open_gates(db, run.id)
            # Keep deterministic evidence ingestion above, but stop before any
            # recovery, review, release, or LLM work if accepted authority is bad.
            if (
                goal.goal_type != "roadmap"
                and run.phase == "authorized"
                and self._json_object_or_empty(run.plan_state).get("status") == "accepted"
            ):
                try:
                    await self._accepted_plan_items(db, run)
                except HTTPException as exc:
                    self._block_plan_integrity(goal, run, str(exc.detail))
            if self._has_active_blocker(run, "plan_criterion_integrity"):
                await db.flush()
                if not caller_owns_transaction:
                    await db.commit()
                return self._tick_noop_result(run)
            recoveries_created = await self.recover_run(db, run.id, baseline_ready=baseline_ready)
            await db.refresh(run)
            await db.refresh(goal)
            effectiveness_review_process = None
            if not self._has_active_blocker(run, "repeated_failure"):
                await ProjectService().lock_workspace_boundary(db, project_id)
                await ProjectService().require_runnable_project(db, project_id)
                effectiveness_review_process = (
                    await EffectivenessReviewProcess().advance(db, goal, run)
                )
            await db.refresh(run)
            await db.refresh(goal)

            authorized_execution = None
            local_liveness = None
            if (
                run.phase == "authorized"
                and run.status in LLM_DECISION_RUN_STATUSES
                and baseline_ready
                and not self._has_active_blocker(run, "repeated_failure")
            ):
                # Local liveness is deliberately provider-free and runs before
                # the goal executor.  A durable wait/attention/release consumes
                # this tick; Roadmap owns its own release loop.
                if (
                    goal.goal_type == "outcome"
                    and await self._sync_measured_budget_exhaustion(db, goal, run)
                ):
                    await db.flush()
                    authorized_execution = {"step": "budget_exhausted"}
                elif goal.goal_type == "outcome" and self._has_active_blocker(run, "budget_measurement"):
                    await db.flush()
                    authorized_execution = {"step": "needs_attention"}
                else:
                    local_liveness = await self.supervision.reconcile_local(
                        db, goal, run, events=new_events, allow_release=goal.goal_type != "roadmap"
                    )
                    if local_liveness["outcome"] == "released":
                        authorized_execution = {"step": "release_work", "released": local_liveness["count"]}
                    elif local_liveness["outcome"] in {
                        "waiting", "due_fallback", "released", "budget_wait", "needs_attention",
                        "dependency_cycle", "orphaned_ownership", "durable_source_active", "closeout_ready",
                    }:
                        authorized_execution = {"step": "local_liveness", **local_liveness}
                    elif goal.goal_type == "roadmap":
                        from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService

                        authorized_execution = await OrchestrationRoadmapService(self).advance(db, goal, run)
                    else:
                        authorized_execution = await self._advance_authorized_execution(db, goal, run)
                await db.refresh(run)
                await db.refresh(goal)

            can_finish = (
                run.status in LLM_DECISION_RUN_STATUSES
                and goal.status in {"active", "blocked"}
                and run.phase == "authorized"          # NEW: no completion before Start
                and baseline_ready
                and not any(
                    isinstance(blocker, Mapping) and blocker.get("kind") != "everyone_idle"
                    for blocker in run.active_blockers
                )
                and (
                    effectiveness_review_process is None
                    or effectiveness_review_process["status"] != "waiting_decision"
                )
            )

            final_summary_action = None
            if can_finish and await self._run_ready_for_final_summary_request(db, goal, run):
                try:
                    final_summary_action = await self.execute_request_final_summary_action(
                        db,
                        run_id=run.id,
                        request={
                            "action_type": "request_final_summary",
                            "work_function": FINAL_SUMMARY_WORK_FUNCTION,
                        },
                        idempotency_key=self._final_summary_request_key(run.id),
                    )
                except HTTPException as exc:
                    if exc.status_code != 409 or exc.detail != "No strong summarization agent fit":
                        raise

            goal_closeout_process = None
            if can_finish:
                try:
                    closeout_preconditions = await self._closeout_preconditions_manifest(
                        db, goal, run
                    )
                except HTTPException as exc:
                    if exc.status_code != 409:
                        raise
                    closeout_preconditions = None
                current_closeout = await OrchestrationProcessService().get_current(
                    db, goal.id, "goal_closeout"
                )
                if closeout_preconditions is not None or current_closeout is not None:
                    await ProjectService().lock_workspace_boundary(db, project_id)
                    await ProjectService().require_runnable_project(db, project_id)
                    goal_closeout_process = await GoalCloseoutProcess().advance(
                        db, goal, run, preconditions=closeout_preconditions
                    )

            completion_action = None
            if can_finish and await self._run_ready_for_completion(db, goal, run):
                completion_action = await self.execute_complete_run_action(
                    db,
                    run_id=run.id,
                    request={
                        "action_type": "complete_run",
                        "reason": "Goal closeout authorized completion.",
                    },
                    idempotency_key=self._complete_run_key(run.id),
                )
            run_completed = (
                completion_action is not None
                and completion_action.status == "completed"
                and run.status == "completed"
                and goal.status == "completed"
            )

            # Move emit_event_once and run.event_cursor inside lock (fix #9).
            _, created = await emit_event_once(
                db,
                project_id,
                "orchestration.tick",
                {
                    "run_id": str(run_id),
                    "goal_id": str(run.goal_id),
                    "processed_events": len(new_events),
                    "evidence_created": evidence_created,
                    "gates_validated": gates_validated,
                    "recoveries_created": recoveries_created,
                    "final_summary_action_id": (
                        str(final_summary_action.id) if final_summary_action is not None else None
                    ),
                    "completion_action_id": (
                        str(completion_action.id) if completion_action is not None else None
                    ),
                    "event_cursor": cursor_token,
                    "authorized_execution": authorized_execution,
                    "local_liveness": local_liveness,
                },
                source="orchestrator",
                dedup_key=f"orchestration.tick:run:{run_id}:cursor:{cursor_token}",
            )

            run.event_cursor = new_cursor
            await db.flush()
            # MEDIUM/HIGH fix: Commit while still holding the lock to ensure another
            # concurrent request cannot acquire the lock and read uncommitted data.
            # In production (via get_db()), this commits and releases lock atomically.
            # In test contexts (async with session.begin()), the test transaction manager
            # controls commit/rollback, so we skip commit here to avoid InvalidRequestError.
            if not caller_owns_transaction:
                # Caller did not pre-begin transaction; safe to commit before releasing lock
                await db.commit()
            # else: caller owns the transaction context; skip explicit commit

        return {
            "run_id": run_id,
            "status": run.status,
            "processed_events": len(new_events),
            "evidence_created": evidence_created,
            "gates_validated": gates_validated,
            "recoveries_created": recoveries_created,
            "final_summary_action_id": (
                final_summary_action.id if final_summary_action is not None else None
            ),
            "run_completed": run_completed,
            "completion_action_id": completion_action.id if completion_action is not None else None,
            "tick_emitted": created,
            "event_cursor": new_cursor,
            "baseline_process": baseline_process,
            "manager_selection_process": manager_selection_process,
            "agent_definition_review_process": agent_definition_review_process,
            "team_hierarchy_process": team_hierarchy_process,
            "effectiveness_review_process": effectiveness_review_process,
            "goal_closeout_process": goal_closeout_process,
            "authority_interview": authority_interview_summary,
            "authorized_execution": authorized_execution,
            "steering": {
                "processed_request_ids": [str(item) for item in steering_processing.processed_request_ids],
                "applied_request_ids": [str(item) for item in steering_processing.applied_request_ids],
            },
        }

    async def _run_exists(self, db: AsyncSession, run_id: uuid.UUID) -> bool:
        result = await db.execute(select(OrchestrationRun.id).where(OrchestrationRun.id == run_id))
        return result.scalar_one_or_none() is not None
