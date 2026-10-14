from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import re
from copy import deepcopy
from decimal import Decimal
from graphlib import CycleError, TopologicalSorter
import hashlib
import logging
import hmac
import json
import logging
from textwrap import shorten
import uuid
import weakref
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

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
from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingDecision
from huddleroom.services.orchestration_meeting_commitments import (
    link_meeting_action_items, meeting_commitments, waive_meeting_action_item, waiver_reason,
)
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
    compute_goal_contract_fingerprint,
    delegable_context,
)
from huddleroom.models.graph import Graph, GraphRun, GraphRunStep
from huddleroom.models.project import Project
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.models.user import User
from huddleroom.models.base import _utcnow
from huddleroom.schemas.orchestration import (
    AuthorizationScope,
    OrchestrationDelegationContract,
    OrchestrationGoalCreate,
    OrchestrationPlanItem,
    planning_envelope_report_schema,
)
from huddleroom.schemas.orchestration_recovery import BaselineRepairAdmissionV1, CorrectiveDisposition
from huddleroom.services.orchestration_work_report import CANONICAL_WORK_REPORT_SCHEMA
from huddleroom.services.orchestration_plan_ingestion import (
    BUDGET_REQUIRED_CODE, BUDGET_REQUIRED_MESSAGE,
    INGESTION_SOURCE,
    PlanIngestionError,
    ingest,
    ingested_plan_hash,
    linked_correction_session,
    plan_artifact_id,
)
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
from huddleroom.models.orchestration_conversation import ConversationMessage
from huddleroom.models.orchestration_steering import OrchestrationSteeringProposal, OrchestrationSteeringRequest
from huddleroom.services.orchestration_authority_interview import build_checkpoint, sync_deferred_questions_memory
from huddleroom.services.orchestration_authority_service import (
    OrchestrationAuthorityDecisionService,
    effective_authorization,
    reauthorize_execution_key,
    runtime_decision_identity,
)
from huddleroom.services.orchestration_start_text_check import find_start_contradictions
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService
from huddleroom.services.orchestration_goal_shape import OrchestrationGoalShapeService
from huddleroom.services.orchestration_goal_definition import (
    GOAL_WEIGHT_ORDER,
    GoalDefinitionProcess,
    classify_goal_weight,
    required_decision_keys_for_weight,
)
from huddleroom.services.orchestration_manager_selection import ManagerSelectionProcess
from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess
from huddleroom.services.orchestration_team_hierarchy import TeamHierarchyProcess, plan_mode
from huddleroom.services.orchestration_effectiveness_review import EffectivenessReviewProcess
from huddleroom.services.orchestration_goal_closeout import GoalCloseoutProcess
from huddleroom.services.orchestration_capacity import (
    activate_ready, available_slots, dependency_counts, queued_tasks, rank_queued, root_goal_id,
)
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_llm_decision_adapter import LLMDecisionAdapter, OrchestrationDecisionAdapter
from huddleroom.services.orchestration_memory_preface import OrchestrationMemoryPrefaceBuilder
from huddleroom.services.orchestration_roster_mapper import OrchestrationRosterMapper
from huddleroom.services.graph_engine import GraphEngineService
from huddleroom.services.project_service import ProjectService
from huddleroom.services.session_service import SessionService
from huddleroom.services.task_service import TaskService
from huddleroom.services.orchestration_decision_validator import SELF_GRANT_ERROR, independent_work_rejection
from huddleroom.services.orchestration_result_contract import (
    CORRECTABLE_KINDS, PLANNING_ENVELOPE, WORK_REPORT, build_correction_brief, expected_result_kind,
    failure_fingerprint, next_step, normalize_result_output, stable_hash,
)
from huddleroom.services.orchestration_dependencies import (
    ADHOC_GATE_TYPE,
    NON_BLOCKING_GATE_TYPES,
    ADHOC_GATED_WORK_FUNCTIONS,
    RETAINED_GATE_TYPE,
    _uuid,
    accepted_plan_items,
    criterion_state,
    evidence_semantics,
    first_unmet_dependency,
    non_final_gate,
    task_gate_id,
    validate_dependencies,
)


LLM_DECISION_RUN_STATUSES = frozenset({"running", "blocked"})
RESULT_CORRECTION_ASK_OPTIONS = (
    {"key": "retry", "label": "Retry the correction"}, {"key": "dismiss", "label": "Stop correcting"},
)
RECOVERY_DIAGNOSIS_ERROR_BACKOFF_SECONDS = 60  # ponytail: fixed backoff; make it exponential if outages run long
# Actions that hand off to a human or pause: `_act_until_wait` must not keep acting after them.
# recover_blocker / recovery_trigger: one tick never acts past a newly started recovery lineage.
_LOOP_STOP_ACTIONS = frozenset({"ask_human", "pause_run", "recover_blocker", "recovery_trigger"})
# Spec "Corrective-action ownership": these never repair state (accept_plan has its own approval contract).
_NON_MUTATING_ACTIONS = frozenset({
    "noop", "suggest_agent", "record_warning", "pause_run", "ask_human", "accept_plan", "recover_blocker",
})
ASK_HUMAN_EVENT_TYPE = "orchestration.human_input_required"
GOAL_RESUMED_EVENT_TYPE = "orchestration.goal_resumed"
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
PLAN_KINDS = ("roadmap", "delivery", "implementation")


def plan_kind_for(goal, run=None) -> str:
    mode = plan_mode(goal, run) if run is not None else getattr(goal, "planning_mode", None)
    if mode in PLAN_KINDS:
        return mode
    return "roadmap" if getattr(goal, "goal_type", None) == "roadmap" else "implementation"
FINAL_SUMMARY_GATE_KEY = "final_summary"
FINAL_SUMMARY_GATE_TYPE = "final_summary_accepted"
ATTEMPT_SOURCE_TYPES = frozenset({"task", "session"})
ATTEMPT_REJECTION_REASONS = frozenset(
    {"Rejected evidence from task", "Rejected evidence from session"}
)
# Authorization denials a different agent/decision can resolve for one task; the rest are run-wide.
TASK_SCOPED_AUTH_DENIALS = frozenset({
    "agent_outside_accepted_team", "model_spend_not_granted", "external_write_not_granted",
    "live_send_not_granted", "workspace_mutation_exceeds_policy",
})
FINAL_SUMMARY_WORK_FUNCTION = "summarization"
FINAL_SUMMARY_REQUESTED_EVENT_TYPE = "orchestration.final_summary_requested"
# Run work a plan can never adopt (and need not disposition): coordination, not discovery.
NON_ADOPTABLE_WORK_FUNCTIONS = frozenset(
    {PLAN_WORK_FUNCTION, FINAL_SUMMARY_WORK_FUNCTION, "report_clarification", "decision", "verification"}
)
RUN_COMPLETED_EVENT_TYPE = "orchestration.run_completed"
DECISION_CONTEXT_EVENT_LIMIT = 20
TICK_EVENT_BATCH_LIMIT = 500
TASK_EVIDENCE_EVENT_TYPES = frozenset({"task.status_changed"})
SESSION_EVIDENCE_EVENT_TYPES = frozenset({"session.completed", "session.failed"})
REVIEW_EVIDENCE_EVENT_TYPES = frozenset({"review.approved", "review.changes_requested"})
GRAPH_EVIDENCE_EVENT_TYPES = frozenset({"graph.run_completed", "graph.run_failed", "graph.run_advanced"})
MEETING_EVIDENCE_EVENT_TYPES = frozenset({"meeting.concluded", "meeting.decision_recorded"})
EVIDENCE_EVENT_TYPES = (
    TASK_EVIDENCE_EVENT_TYPES
    | SESSION_EVIDENCE_EVENT_TYPES
    | REVIEW_EVIDENCE_EVENT_TYPES
    | GRAPH_EVIDENCE_EVENT_TYPES
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
GRAPH_STARTED_EVENT_TYPE = "orchestration.graph_started"


class PlanGenerationChanged(Exception):
    """A server-snapshot planning generation is no longer current."""


def _plan_generation_seal(run_id: Any, plan_state: Any) -> dict[str, Any]:
    if not isinstance(plan_state, Mapping):
        raise PlanGenerationChanged("Malformed planning generation")
    try:
        run_id = str(uuid.UUID(str(run_id)))
        version = plan_state.get("planning_version", 0)
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise ValueError("Invalid planning version")
        status = plan_state.get("status")
        if status is not None and not isinstance(status, str):
            raise ValueError("Invalid planning status")
        identities = {
            field: str(uuid.UUID(str(plan_state[field]))) if plan_state.get(field) is not None else None
            for field in ("planning_task_id", "request_action_id")
        }
    except (ValueError, TypeError, AttributeError):
        raise PlanGenerationChanged("Malformed planning generation") from None
    return {"run_id": run_id, "status": status, "planning_version": version, **identities}

logger = logging.getLogger(__name__)

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

# Separate from goal locks: recovery may acquire the roster lock before goal locks.
_RECOVERY_LIFECYCLE_LOCKS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
# Admission only for public ticks in the same process/event loop; persistence
# still decides the winner for direct callers and other processes.
_RECOVERY_DIAGNOSES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

# A baseline-repair admission (passed only by the dispatcher from a validated approve_fix recovery context,
# never read from request JSON) is honored only for that approved call's own recovery effect key.
_RECOVERY_APPLY_EFFECT_KEY = re.compile(r"recovery:[0-9a-f-]{36}:apply:[^:]+:effect")
# Sealed delegate_remediation@1 contract keys kept verbatim on the canonical delegation request. The model
# decision validator rejects them as unknown keys, so only the recovery broker's synthetic decision has them.
REMEDIATION_REQUEST_KEYS = (
    "objective", "expected_result", "requested_scope", "actual_execution_scope", "execution_evidence",
    "deadline", "wait_events", "verifier", "retry_target",
)



class OrchestrationService:
    _plan_generation_seal = staticmethod(_plan_generation_seal)

    async def _assert_current_plan_generation(self, db, run_id: uuid.UUID, expected: dict) -> OrchestrationRun:
        run = await db.get(OrchestrationRun, run_id, populate_existing=True)
        if run is None or _plan_generation_seal(run.id, run.plan_state) != expected:
            raise PlanGenerationChanged("Planning generation changed")
        return run

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
            # An authorized run reports waiting_work; this default never
            # claims work exists.
            return "waiting_work"
        return "working"

    async def create_goal(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        data: OrchestrationGoalCreate,
        created_by_user_id: uuid.UUID | None,
    ) -> tuple[OrchestrationGoal, OrchestrationRun]:
        # ponytail: shape before lock; non-runnable project costs one call
        decision = await OrchestrationGoalShapeService().resolve_creation_shape(project_id, data)
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
            orchestrator_context={
                "expansion_policy": {
                    "max_concurrent_tasks": settings.orchestration_max_concurrent_tasks,
                    "max_goal_depth": settings.orchestration_max_goal_depth,
                    "max_descendant_goals": settings.orchestration_max_descendant_goals,
                    "depth": 0,
                }
            },
            **decision.model_dump(),
        )
        db.add(goal)
        await db.flush()

        run = OrchestrationRun(
            goal_id=goal.id,
            budget_state=deepcopy(budget),
            baseline_authorized=False,
            goal_type_snapshot=goal.goal_type,
            planning_mode_snapshot=goal.planning_mode,
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
                OrchestrationGate.gate_type != RETAINED_GATE_TYPE,
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

    @staticmethod
    async def _commit_tick_owned_transaction(db: AsyncSession) -> bool:
        """Release a tick-owned SQLite writer before a model round trip; True when committed."""
        # A tick-owned SQLite write transaction must release the sole writer
        # before isolated sessions read decision context and write the result.
        if (
            db.info.get("orchestration_tick_owns_transaction")
            and db.bind is not None
            and db.bind.dialect.name == "sqlite"
        ):
            transaction = db.sync_session.get_transaction()
            if (
                transaction is not None
                and transaction.origin == SessionTransactionOrigin.AUTOBEGIN
                and not db.in_nested_transaction()
            ):
                await db.commit()
                return True
        return False

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

        current_run = await db.get(OrchestrationRun, run_id)
        if current_run is not None:
            self._require_roadmap_child_budget(current_run)

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
        # ponytail: the finite-tree root lock stays held across this model call (SQLite asyncio lock outlives the commit; PG holds FOR UPDATE until tick commit); drop root+leaf with post-call revalidation if tree throughput matters.
        await self._commit_tick_owned_transaction(db)
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
    async def _lock_goal_for_baseline_transition(
        self,
        db: AsyncSession,
        goal_id: uuid.UUID,
        *,
        tick_owns_transaction: bool = False,
    ):
        """Lock the goal's finite-tree root first, then the goal (root->leaf, one fixed order).

        Sibling/cousin ticks share root-wide slots, depth, descendant count and
        budget, so they must serialize on the root; a single fixed order means two
        goals of one tree never wait on each other in reverse. Continuous cases are
        their own roots. Same task / same transaction re-entry stays reentrant. A
        Postgres commit inside the section drops both row locks together (same
        precedent as the goal lock); the SQLite asyncio locks span commits, so
        _commit_tick_owned_transaction keeps the tree serialized.
        """
        # ponytail: one lock per finite tree serializes sibling ticks; finer locks if throughput matters.
        if db.in_transaction() or db.new or db.bind is None:
            root_id = await root_goal_id(db, goal_id)
        else:
            # Keep a fresh session fresh: callers may db.begin() under the lock (supersession).
            # It holds no uncommitted rows, so an independent read sees the same tree.
            async with AsyncSession(db.bind) as probe:
                root_id = await root_goal_id(probe, goal_id)
        async with AsyncExitStack() as stack:
            if root_id != goal_id:
                await stack.enter_async_context(self._lock_single_goal(db, root_id))
            await stack.enter_async_context(
                self._lock_single_goal(db, goal_id, tick_owns_transaction=tick_owns_transaction)
            )
            yield

    @asynccontextmanager
    async def _lock_single_goal(
        self,
        db: AsyncSession,
        goal_id: uuid.UUID,
        *,
        tick_owns_transaction: bool = False,
    ):
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
                if not tick_owns_transaction:
                    yield
                    return
                marker = "orchestration_tick_owns_transaction"
                missing = object()
                previous = db.info.get(marker, missing)
                db.info[marker] = True
                try:
                    yield
                finally:
                    if previous is missing:
                        db.info.pop(marker, None)
                    else:
                        db.info[marker] = previous
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
            # Recovery analyzer exchanges are not orchestrator loop decisions.
            .where(OrchestrationDecision.run_id == run_id, OrchestrationDecision.decision_type != "recovery")
            .order_by(OrchestrationDecision.created_at.desc(), OrchestrationDecision.id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    _VOLATILE_SUPERVISION_KEYS = frozenset({
        "at", "judgment_in_flight", "judgment_dirty", "judgment_failures",
        "last_event_id", "context_fingerprint", "last_pass", "judgment_claim_token",
    })

    @staticmethod
    def _stable_supervision_state(state: Any) -> Any:
        # ponytail: strips timestamps and bookkeeping counters so two idle ticks produce the same decision context.
        # The scheduler fingerprint is a separate hash and is untouched by this.
        if isinstance(state, dict):
            return {
                key: OrchestrationService._stable_supervision_state(value)
                for key, value in state.items()
                if not (isinstance(key, str) and (
                    key.endswith("_at") or key in OrchestrationService._VOLATILE_SUPERVISION_KEYS
                ))
            }
        if isinstance(state, list):
            return [OrchestrationService._stable_supervision_state(item) for item in state]
        return deepcopy(state)

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
                "supervision_state": self._stable_supervision_state(run.supervision_state),
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
                    OrchestrationGate.gate_type.not_in(NON_BLOCKING_GATE_TYPES),
                ).order_by(OrchestrationGate.created_at, OrchestrationGate.id)))
            ],
            "memory_preface": memory_preface,
            "roster": roster,
            "plan_author": await self._plan_author_hint(db, goal, run),
        }

    @staticmethod
    async def _plan_author_hint(db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun) -> dict | None:
        from huddleroom.services.orchestration_team_hierarchy import plan_mode, select_plan_author

        mode = plan_mode(goal, run)
        status = (run.plan_state or {}).get("status")
        # accepted Roadmap plans advertise the author for the replan path
        if mode == "continuous_policy" or status == "failed" or (status == "accepted" and mode != "roadmap"):
            return None
        try:
            author, error = await select_plan_author(db, goal, run), None
        except HTTPException as exc:
            author, error = None, str(exc.detail)
        except (ValueError, KeyError, TypeError) as exc:
            author, error = None, f"plan_author_unavailable: malformed team data ({exc})"
        return {"agent_id": str(author) if author else None, "planning_mode": mode, "error": error}

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

    _SESSION_SIDE_EFFECT_KEYS = ("pending_bus_events", "pending_session_dispatches")

    @classmethod
    def _side_effect_marks(cls, db) -> dict[str, int]:
        """Lengths of the after-commit queues, so a savepoint rollback can drop its own entries."""
        info = getattr(getattr(db, "sync_session", None), "info", None)
        if info is None:
            return {}
        return {key: len(info.get(key, ())) for key in cls._SESSION_SIDE_EFFECT_KEYS}

    async def _after_savepoint_rollback(self, db, run, marks: dict[str, int]) -> None:
        """Queued bus events/dispatches outlive a savepoint rollback; drop them and reload expired state."""
        info = db.sync_session.info
        for key, size in marks.items():
            if key in info:
                del info[key][size:]
        try:
            await db.refresh(run)
            await db.get(OrchestrationGoal, run.goal_id, populate_existing=True)
        except Exception:  # never mask the exception being handled
            logger.exception("Could not refresh run/goal after savepoint rollback")

    def _is_genuine_rejection(self, exc: HTTPException) -> bool:
        """Only a plain 4xx rolls its work back; waits and server errors persist state on purpose."""
        return 400 <= exc.status_code < 500 and self._waiting_step_for_http_error(exc) is None

    async def _failed_action_for_key(
        self, db: AsyncSession, run_id: uuid.UUID, key: str,
    ) -> OrchestrationAction | None:
        action = await self._existing_action_for_key(db, run_id, key)
        if action is None:
            return None
        await db.refresh(action)
        return action if action.status == "failed" else None

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
        ad_hoc: bool = False,
        baseline_repair_admission: BaselineRepairAdmissionV1 | None = None,
        gate_bound: bool = False,
        expected_result: str | None = None,
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
            # An approved remediation may bypass only its one named failed process, and only for its own effect.
            admission = (baseline_repair_admission
                         if _RECOVERY_APPLY_EFFECT_KEY.fullmatch(idempotency_key or "") else None)
            await self._ensure_baseline_processes_ready(db, gate_run.goal_id, admission=admission)
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
            await validate_dependencies(
                db, gate_run, [uuid.UUID(d) for d in request_to_store.get("depends_on") or []]
            )
            missing = await self._first_missing_artifact(db, project_id, request_to_store.get("artifact_ids") or [])
            if missing is not None:
                raise HTTPException(status_code=409, detail=f"artifact_invalid: artifact {missing} not found in project")
            if ad_hoc:
                run_goal = await db.get(OrchestrationGoal, gate_run.goal_id)
                rejection = independent_work_rejection(
                    request_to_store,
                    [c.get("key") for c in run_goal.success_criteria or [] if isinstance(c, Mapping)],
                    gate_run.plan_state,
                )
                if rejection is not None:
                    raise HTTPException(status_code=409, detail=rejection)
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
            await link_meeting_action_items(db, run_id, action, existing_task)
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

        report_schema = self._json_object_or_empty(stored_request.get("report_schema"))
        # The caller says when it created the gate (plan-item/roadmap expansion links it right after);
        # ad-hoc gated work functions get their gate below.
        work_function = self._required_string(stored_request.get("work_function"), "work_function")
        has_gate = gate_bound or (ad_hoc and work_function in ADHOC_GATED_WORK_FUNCTIONS)
        contract = OrchestrationDelegationContract(
            goal_id=goal.id,
            run_id=run.id,
            action_id=action.id,
            agent_id=agent_id,
            # Server-only override: a result correction keeps the original task's kind.
            expected_result=expected_result or expected_result_kind({"report_schema": report_schema}, has_gate),
            work_function=work_function,
            scope=self._required_string(stored_request.get("scope"), "scope"),
            inputs=self._string_list(stored_request.get("inputs")),
            deliverable=self._required_string(stored_request.get("deliverable"), "deliverable"),
            forbidden_work=self._string_list(stored_request.get("forbidden_work")),
            success_evidence=self._string_list(stored_request.get("success_evidence")),
            budget=self._json_object_or_empty(stored_request.get("budget")),
            report_schema=report_schema,
            parent_task_id=parent_task_id,
            orchestrator_context=self._json_object_or_empty(stored_request.get("orchestrator_context")),
            title=stored_request.get("title") or "",
            **{
                key: stored_request[key]
                for key in ("depends_on", "artifact_ids", "why_runnable", "priority", "decision_use", "authorization_scope")
                if key in stored_request
            },
        )
        contract_payload = contract.model_dump(mode="json")
        budget_amounts = self._json_object_or_empty(contract.budget)
        continuous = self._json_object_or_empty(
            self._json_object_or_empty(contract.orchestrator_context).get("continuous")
        )
        protected = (
            contract.work_function in {"verification", "final_summary", "report_clarification"}
            or ":verify_gate:" in idempotency_key or ":final_summary" in idempotency_key
            # The approved final-summary recovery retry (server-built marker, never model-authored) is the same
            # protected closeout work as the legacy replacement.
            or (idempotency_key.startswith("recovery:")
                and self._json_object_or_empty(contract.orchestrator_context).get("final_summary_recovery") is True)
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
        orchestration_metadata = {
            "goal_id": str(goal.id),
            "run_id": str(run.id),
            "action_id": str(action.id),
            "work_function": contract.work_function,
        }
        if ad_hoc and contract.work_function in ADHOC_GATED_WORK_FUNCTIONS:
            # Task id == action id, so the gate key is replay-stable. Completion alone never accepts it.
            adhoc_gate = await self._ensure_keyed_gate(
                db, run.id, ADHOC_GATE_TYPE, f"adhoc_task:{action.id}",
                {
                    "required_source_types": ["task"], "min_count": 1,
                    "evidence_role": ADHOC_GATED_WORK_FUNCTIONS[contract.work_function],
                    "review_stage": "preliminary", "task_id": str(action.id),
                },
            )
            orchestration_metadata["gate_id"] = str(adhoc_gate.id)
        task = await task_service.create(
            db,
            project_id,
            TaskCreate(
                title=self._delegation_task_title(contract.work_function, contract.deliverable, contract.title),
                description=self._delegation_task_description(goal, contract),
                assigned_to=agent_id,
                parent_id=parent_task_id,
                depends_on=contract.depends_on,
                priority=contract.priority,
                metadata={
                    "orchestration": orchestration_metadata,
                    "orchestration_contract": contract_payload,
                },
            ),
            task_id=action.id,
        )
        await self._emit_delegation_task_created(db, project_id, action, task)
        await link_meeting_action_items(db, run_id, action, task)
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
            existing.request if existing is not None else request, gates, evidence, goal, roadmap_version_id,
            accepted_plan_items(run.plan_state),
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

    @staticmethod
    async def _require_plan_author(db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun, agent_id) -> None:
        """Reject (never override) a plan request whose agent is not the selected plan author."""
        from huddleroom.services.orchestration_team_hierarchy import select_plan_author

        author = await select_plan_author(db, goal, run)
        if author is not None and str(author) != str(agent_id or "").strip():
            raise HTTPException(
                status_code=409,
                detail=f"plan_author_mismatch: request_plan agent_id must be the selected plan author {author}",
            )

    async def execute_request_plan_action(
        self, db: AsyncSession, run_id: uuid.UUID, request: dict, idempotency_key: str,
        decision_id: uuid.UUID | None = None, plan_dispatch_request: dict | None = None,
        expected_plan_generation: dict | None = None,
    ) -> OrchestrationAction:
        pending_writes = bool(db.new or db.dirty or db.deleted)
        flushed_writes = await self._sqlite_transaction_has_writes(db)
        with db.no_autoflush:
            run = await db.get(OrchestrationRun, run_id)
            lookup_key = await self._plan_action_lookup_key(db, idempotency_key, decision_id)
            existing = await self._existing_action_for_key(db, run_id, lookup_key)
            old_session = await self._sqlite_cli_planning_replacement_session(db, run) if run else None
        if existing is not None and existing.status != "reserved":
            replay_request = (existing.dispatch_contract or {}).get("plan_dispatch_request", existing.request)
            if replay_request == self._canonical_plan_request(request):
                return existing
            return await self._execute_request_plan_action(
                db, run_id, request, idempotency_key, decision_id, plan_dispatch_request,
            )
        if old_session is not None:
            if pending_writes or flushed_writes:
                raise HTTPException(status_code=409, detail="planning_replacement_transaction_conflict")
            await self._assert_sqlite_replacement_transaction(db)
            expected = expected_plan_generation or _plan_generation_seal(run.id, run.plan_state)
            goal_id = run.goal_id
            async with self._lock_goal_for_baseline_transition(db, goal_id):
                run = await self._assert_current_plan_generation(db, run_id, expected)
                # Admission checks precede destructive cancellation, just as
                # they precede reservation in the original executor.
                await self._ensure_baseline_processes_ready(db, goal_id)
                await self._ensure_plan_request_allowed(db, run)
                canonical = self._canonical_plan_request(request)
                goal = await db.get(OrchestrationGoal, goal_id)
                inherited_context = self._json_object_or_empty(goal.orchestrator_context)
                if self._has_delegable_orchestrator_context(inherited_context):
                    canonical = {**canonical, "orchestrator_context": delegable_context(inherited_context)}
                await self._require_plan_author(db, goal, run, canonical.get("agent_id"))
                _, agent, _ = await self._validate_delegation_targets(db, run_id, {**canonical, "parent_task_id": None})
                authority = await self.roadmap_pre_release_authority(
                    db, run_id, agent, self._json_object_or_empty(canonical.get("orchestrator_context")),
                )
                if authority.get("status") in {"pending", "rejected"}:
                    raise HTTPException(
                        status_code=409, detail="budget_wait" if authority["status"] == "pending" else "needs_attention",
                    )
                await self._terminalize_sqlite_planning_replacement(db, run_id, expected, old_session)
                return await self._execute_request_plan_action(
                    db, run_id, request, idempotency_key, decision_id, plan_dispatch_request,
                )
        return await self._execute_request_plan_action(
            db, run_id, request, idempotency_key, decision_id, plan_dispatch_request,
        )

    async def _terminalize_sqlite_planning_replacement(self, db, run_id, expected, old_session) -> None:
        session_id, task_id = old_session.id, old_session.task_id
        await self._assert_sqlite_replacement_transaction(db)
        await db.commit()
        await SessionService().cancel(db, session_id, sync_task=False)
        await self._assert_current_plan_generation(db, run_id, expected)
        old_task = await db.get(Task, task_id, populate_existing=True)
        if old_task is not None and "cancelled" in Task.VALID_TRANSITIONS.get(old_task.status, set()):
            await TaskService().cancel(db, old_task.project_id, task_id)
        await db.commit()  # old facts survive every later replacement failure
        await self._assert_current_plan_generation(db, run_id, expected)

    @staticmethod
    async def _plan_action_lookup_key(db, key, decision_id) -> str:
        """Derive exact replay identity without checking mutable steering state."""
        from huddleroom.services.orchestration_steering import (
            OrchestrationSteeringService, steering_versions_from_snapshot,
        )
        decision = await db.get(OrchestrationDecision, decision_id) if decision_id else None
        versions = steering_versions_from_snapshot(decision.input_snapshot or {}) if decision else None
        if versions is not None:
            suffix = f":steering:{OrchestrationSteeringService.version_digest(versions)}"
            if not key.endswith(suffix):
                key = key[:255 - len(suffix)] + suffix
        return key

    @staticmethod
    async def _assert_sqlite_replacement_transaction(db: AsyncSession) -> None:
        transaction = db.sync_session.get_transaction()
        if (db.new or db.dirty or db.deleted or db.in_nested_transaction()
                or transaction is None or transaction.origin != SessionTransactionOrigin.AUTOBEGIN
                or await OrchestrationService._sqlite_transaction_has_writes(db)):
            raise HTTPException(status_code=409, detail="planning_replacement_transaction_conflict")

    @staticmethod
    async def _sqlite_transaction_has_writes(db: AsyncSession) -> bool:
        if db.get_bind().dialect.name != "sqlite" or not db.in_transaction():
            return False
        connection = await db.connection()
        raw = await connection.get_raw_connection()
        # sqlite3/aiosqlite starts its native transaction at the first DML.
        # ORM dirty sets alone miss writes that the caller already flushed.
        return bool(raw.driver_connection.in_transaction)

    async def _sqlite_cli_planning_replacement_session(self, db, run) -> Session | None:
        if db.get_bind().dialect.name != "sqlite" or run is None:
            return None
        state = self._json_object_or_empty(run.plan_state)
        if state.get("status") not in {"requested", "revision_requested"}:
            return None
        task_id = self._optional_uuid(state.get("planning_task_id"), "planning_task_id")
        if task_id is None:
            return None
        return await db.scalar(select(Session).where(
            Session.task_id == task_id, Session.adapter_type == "cli",
            Session.status.in_(("pending", "running")),
        ).order_by(Session.created_at.desc(), Session.id.desc()).limit(1))

    async def _execute_request_plan_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
        plan_dispatch_request: dict | None = None,
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
                request = {**request, "orchestrator_context": delegable_context(inherited_context)}
        request_to_store = existing.request if existing is not None else self._canonical_plan_request(request)
        agent = None
        run = None
        if existing is None:
            run = await self._run_for_plan_action(db, run_id)
            await self._ensure_plan_request_allowed(db, run)
            await self._require_plan_author(db, inherited_goal, run, request_to_store.get("agent_id"))
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

        if plan_dispatch_request is not None:
            contract = dict(action.dispatch_contract or {})
            contract.setdefault("plan_dispatch_request", deepcopy(plan_dispatch_request))
            action.dispatch_contract = contract

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

            pre_plan_tasks = await self._pre_plan_tasks(db, run)
            planning_version = int(self._json_object_or_empty(run.plan_state).get("planning_version") or 0) + 1
            contract = self._planning_contract(
                goal, run, action, stored_request, pre_plan_lines=await self._pre_plan_lines(db, run, pre_plan_tasks)
            )
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
                            "planning_version": planning_version,
                            "pre_plan_task_ids": [str(pre.id) for pre in pre_plan_tasks],
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
        self, db: AsyncSession, run_id: uuid.UUID, request: dict, idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        with db.no_autoflush:
            run = await db.get(OrchestrationRun, run_id)
            lookup_key = await self._plan_action_lookup_key(db, idempotency_key, decision_id)
            existing = await self._existing_action_for_key(db, run_id, lookup_key)
            old_session = await self._sqlite_cli_planning_replacement_session(db, run)
            pending_replan = (self._json_object_or_empty(
                self._json_object_or_empty(run.plan_state).get("pending_replan"),
            ) if run else {})
        if (existing is not None and existing.status != "reserved"
                and existing.request == self._canonical_plan_revision_request(request)):
            return existing
        if old_session is not None and not pending_replan and (existing is None or existing.status == "reserved"):
            await self._assert_sqlite_replacement_transaction(db)
            goal_id = run.goal_id
            expected = _plan_generation_seal(run.id, run.plan_state)
            async with self._lock_goal_for_baseline_transition(db, goal_id):
                run = await self._assert_current_plan_generation(db, run_id, expected)
                await self._ensure_plan_revision_allowed(db, run)
                canonical = self._canonical_plan_revision_request(request)
                planning_task = await self._planning_task_for_run(
                    db, run, self._required_uuid(canonical.get("plan_task_id"), "plan_task_id"),
                )
                agent = await db.get(Agent, planning_task.assigned_to) if planning_task.assigned_to else None
                if agent is None:
                    raise HTTPException(status_code=409, detail="Roadmap planning agent is invalid")
                goal = await db.get(OrchestrationGoal, goal_id)
                authority = await self.roadmap_pre_release_authority(
                    db, run_id, agent, goal.orchestrator_context if goal else None,
                )
                if authority.get("status") in {"pending", "rejected"}:
                    raise HTTPException(
                        status_code=409, detail="budget_wait" if authority["status"] == "pending" else "needs_attention",
                    )
                await self._terminalize_sqlite_planning_replacement(db, run_id, expected, old_session)
                return await self._execute_request_plan_revision_action(
                    db, run_id, request, idempotency_key, decision_id,
                )
        return await self._execute_request_plan_revision_action(db, run_id, request, idempotency_key, decision_id)

    async def _execute_request_plan_revision_action(
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
                context = delegable_context(self._json_object_or_empty(inherited_goal.orchestrator_context if inherited_goal else {}))
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
        pending = self._json_object_or_empty(self._json_object_or_empty(run.plan_state).get("pending_replan"))
        if pending:
            # A rejected Roadmap replan is retried as a new replan task; the accepted version stays current.
            original = await db.get(OrchestrationAction, uuid.UUID(pending["action_id"]))
            replan = await self.reserve_action(
                db, run_id=run.id, idempotency_key=f"run:{run.id}:kind:request_roadmap_replan:revision:{action.id}",
                action_type="request_roadmap_replan", request=original.request,
            )
            if replan.status == "reserved":
                await self._dispatch_roadmap_replan_task(
                    db, await db.get(OrchestrationGoal, run.goal_id), run, replan, original.request,
                    await db.get(OrchestrationRoadmapVersion, uuid.UUID(pending["version_id"])),
                    extra_inputs=[
                        f"Plan revision request: {revision_request}",
                        f"Plan revision reason: {pending.get('ingestion_error')}",
                    ],
                )
        else:
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
        if pending:
            return await self._mark_action_completed(db, action, target_type="task", target_id=planning_task.id)
        contract = self._json_object_or_empty(self._json_object_or_empty(planning_task.metadata_).get("orchestration_contract"))
        plan_action = await self.execute_request_plan_action(
            db,
            run.id,
            {
                "action_type": "request_plan",
                "agent_id": self._json_object_or_empty(run.plan_state).get("plan_author_agent_id")
                or str(planning_task.assigned_to),
                "work_function": PLAN_WORK_FUNCTION,
                "scope": contract.get("scope"),
            },
            f"run:{run.id}:kind:request_plan:revision:{action.id}",
        )
        if plan_action.status != "completed":
            await self._fail_reserved_action(db, action, plan_action.error or "Plan revision could not start a planning task")
            return action
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
        if existing is None:
            await self._require_plan_author(db, goal, run, stored.get("agent_id"))
        action = await self.reserve_action(
            db, run_id=run_id, idempotency_key=idempotency_key, action_type="request_roadmap_replan",
            request=stored, decision_id=decision_id,
        )
        if action.status != "reserved":
            return action
        return await self._dispatch_roadmap_replan_task(db, goal, run, action, stored, version)

    async def _dispatch_roadmap_replan_task(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun, action: OrchestrationAction,
        stored: dict, version: OrchestrationRoadmapVersion, extra_inputs: Sequence[str] = (),
    ) -> OrchestrationAction:
        self._require_roadmap_child_budget(run)
        run_id = run.id
        state = self._json_object_or_empty(run.plan_state)
        project_id, agent, _ = await self._validate_delegation_targets(
            db, run_id, {**stored, "work_function": PLAN_WORK_FUNCTION, "parent_task_id": None},
        )
        task = await self._find_task_for_action(db, project_id, action, run_id=run_id, work_function=PLAN_WORK_FUNCTION)
        if task is None:
            planning_version = int(state.get("planning_version") or 0) + 1
            released = list((await db.scalars(select(OrchestrationRoadmapItem).where(
                OrchestrationRoadmapItem.goal_id == goal.id
            ))).all())
            contract = self._planning_contract(goal, run, action, {**stored, "work_function": PLAN_WORK_FUNCTION})
            contract = contract.model_copy(update={
                "inputs": [*contract.inputs, *extra_inputs, f"Current immutable Roadmap version: {json.dumps(version.snapshot, sort_keys=True)}",
                           f"Released immutable item snapshots: {json.dumps([row.item_snapshot for row in released], sort_keys=True)}"],
                "forbidden_work": [*contract.forbidden_work, "Do not change or remove any released Roadmap item."],
                # A Roadmap replan always returns a Roadmap plan, whatever the run's planning-mode snapshot says.
                "report_schema": planning_envelope_report_schema("roadmap"),
            })
            task = await TaskService().create(
                db, project_id,
                TaskCreate(title="Planning: Roadmap replan", description="Produce a replacement plan for unstarted Roadmap work.",
                    assigned_to=agent.id, metadata={
                        "orchestration": {"goal_id": str(goal.id), "run_id": str(run.id), "action_id": str(action.id), "work_function": PLAN_WORK_FUNCTION},
                        "orchestration_plan": {
                            "status": "replan_requested", "planning_version": planning_version, "pre_plan_task_ids": [],
                        },
                        "orchestration_contract": contract.model_dump(mode="json"),
                        "roadmap_replan": {"current_version": version.snapshot,
                            "released_items": [row.item_snapshot for row in released],
                            "forbidden_work": "Do not change or remove any released Roadmap item."},
                    }), task_id=action.id,
            )
        planning_version = self._json_object_or_empty(
            self._json_object_or_empty(task.metadata_).get("orchestration_plan")
        ).get("planning_version") or int(state.get("planning_version") or 0) + 1
        run.plan_state = {**state, "planning_version": planning_version, "pending_replan": {
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
            accepted = self._json_object_or_empty(run.plan_state)
            if accepted.get("status") == "accepted" and not accepted.get("pending_replan") and accepted.get(
                "accepted_artifact_id"
            ) == str(
                self._required_uuid(request_to_store.get("plan_artifact_id"), "plan_artifact_id")
            ):
                try:
                    prior_id = uuid.UUID(str(accepted.get("accept_action_id")))
                except ValueError:
                    prior_id = None
                prior = await db.get(OrchestrationAction, prior_id) if prior_id else None
                if (
                    prior is not None
                    and prior.run_id == run_id
                    and prior.action_type == "accept_plan"
                    and prior.status == "completed"
                ):
                    return prior  # same plan already accepted under another key
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

                await self._validate_plan_adoptions(db, run, OrchestrationRoadmapService(self).parse_items(artifact, goal))
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
                await self._validate_plan_adoptions(db, run, items)
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
                    idempotency_key=self._expand_plan_item_idempotency_key(run.id, item.id, self._current_plan_fingerprint(run)),
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
        stable_idempotency_key = self._expand_plan_item_idempotency_key(run_id, item.id, self._current_plan_fingerprint(run))
        if idempotency_key == self._expand_plan_item_idempotency_key(run_id, item.id):
            idempotency_key = stable_idempotency_key
        if idempotency_key != stable_idempotency_key:
            raise HTTPException(
                status_code=409,
                detail=f"Expand plan item idempotency key must be '{self._expand_plan_item_idempotency_key(run_id, item.id)}'",
            )
        if item.adopt_task_id is not None:
            return await self._expand_adopted_item(
                db, run, artifact, item, request_to_store, idempotency_key, decision_id
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
            gate_bound=True,
        )
        if delegation_action.status != "completed" or delegation_action.target_id is None:
            await self._fail_reserved_action_for_current_flow(db, action, "Plan item delegation did not complete")
            raise HTTPException(status_code=409, detail="Plan item delegation did not complete")

        task = await db.get(Task, delegation_action.target_id)
        if task is None:
            await self._fail_reserved_action_for_current_flow(db, action, "Delegated task not found")
            raise HTTPException(status_code=404, detail="Delegated task not found")

        self._validate_bound_plan_item(task, run, item)
        self._attach_plan_item_metadata(task, artifact, item, action, gate)
        task.priority = item.priority
        self._set_plan_item_expanded_state(run, item, action, delegation_action, task, gate)
        await self._emit_plan_item_expanded(db, project_id, goal, run, item, action, delegation_action, task, gate)
        return await self._mark_action_completed(db, action, target_type="task", target_id=task.id)

    async def _expand_adopted_item(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        artifact: Artifact,
        item: OrchestrationPlanItem,
        request_to_store: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None,
    ) -> OrchestrationAction:
        """Bind the adopted task to its plan item: no delegation, no new task, its session untouched."""
        action = await self.reserve_action(
            db, run_id=run.id, idempotency_key=idempotency_key, action_type="expand_plan_item",
            request=request_to_store, decision_id=decision_id,
        )
        if action.status != "reserved":
            return action
        task = await db.get(Task, item.adopt_task_id)
        delegation_action = await db.get(OrchestrationAction, item.adopt_task_id)
        if task is None or delegation_action is None:
            await self._fail_reserved_action_for_current_flow(db, action, "Adopted task not found")
            raise HTTPException(status_code=409, detail=f"adoption_invalid: item {item.id}: adopted task not found")
        gate = await self._ensure_plan_item_gate(db, run.id, item)
        try:
            await self._bind_adopted_task(db, run, task, artifact, item, action, gate)
        except HTTPException as exc:
            await self._fail_reserved_action_for_current_flow(db, action, str(exc.detail))
            raise
        goal = await db.get(OrchestrationGoal, run.goal_id)
        self._set_plan_item_expanded_state(run, item, action, delegation_action, task, gate, adopted=True)
        await self._emit_plan_item_expanded(
            db, goal.project_id, goal, run, item, action, delegation_action, task, gate, adopted=True,
        )
        return await self._mark_action_completed(db, action, target_type="task", target_id=task.id)

    async def _bind_adopted_task(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        task: Task,
        artifact: Artifact,
        item: OrchestrationPlanItem,
        action: OrchestrationAction,
        gate: OrchestrationGate,
    ) -> None:
        """Bind once per task (DB-unique marker), then replay its evidence onto the item gate."""
        state = self._json_object_or_empty(run.plan_state)
        fingerprint = state.get("accepted_plan_fingerprint") or self._json_object_or_empty(
            state.get("accepted_plan_snapshot")
        ).get("fingerprint")
        self._validate_bound_plan_item(task, run, item)
        request = {"plan_item_id": item.id, "adopt_task_id": str(task.id), "accepted_plan_fingerprint": fingerprint}
        marker = await self.reserve_action(
            db, run_id=run.id, idempotency_key=f"run:{run.id}:kind:adopt_task:task:{task.id}",
            action_type="adopt_task", request=request,
        )
        if any(marker.request.get(key) != request[key] for key in ("plan_item_id", "adopt_task_id")):
            raise HTTPException(
                status_code=409, detail=f"adoption_conflict: task {task.id} is already adopted by another plan item"
            )
        orchestration = self._json_object_or_empty(self._json_object_or_empty(task.metadata_).get("orchestration"))
        current_gate_id = task_gate_id(task, run.id)
        current_gate = await db.get(OrchestrationGate, current_gate_id) if current_gate_id else None
        if current_gate is not None and current_gate.gate_type == ADHOC_GATE_TYPE:
            # The item gate takes over; the ad-hoc gate and its evidence stay as history.
            task.metadata_ = {
                **self._json_object_or_empty(task.metadata_),
                "orchestration": {
                    **{k: v for k, v in orchestration.items() if k != "gate_id"}, "adhoc_gate_id": str(current_gate.id),
                },
            }
        self._attach_plan_item_metadata(task, artifact, item, action, gate)
        task.priority = item.priority
        await db.flush()
        await self._replay_task_evidence(db, run, task)
        if marker.status == "reserved":
            await self._mark_action_completed(db, marker, target_type="task", target_id=task.id)

    async def _replay_task_evidence(self, db: AsyncSession, run: OrchestrationRun, task: Task) -> int:
        """Re-ingest a task's already-processed events now that it has a gate (dedup is per observed event)."""
        session_ids = [str(sid) for sid in await db.scalars(select(Session.id).where(Session.task_id == task.id))]
        events = await db.scalars(
            select(EventLog)
            .where(
                EventLog.project_id == task.project_id,
                or_(
                    and_(
                        EventLog.event_type.in_(TASK_EVIDENCE_EVENT_TYPES),
                        EventLog.payload["task_id"].as_string() == str(task.id),
                    ),
                    and_(
                        EventLog.event_type.in_(SESSION_EVIDENCE_EVENT_TYPES),
                        EventLog.payload["session_id"].as_string().in_(session_ids),
                    ),
                ),
            )
            .order_by(EventLog.seq)
        )
        created = 0
        for event in events.all():
            created += await self._ingest_evidence_from_event(db, run, event)
        return created

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
                context=self._optional_string(stored_request.get("reason")),
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
                        # is answered.
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

    async def execute_start_graph_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        request_to_store = (
            existing.request if existing is not None else self._canonical_start_graph_request(request)
        )
        project_id = await self._project_id_for_run(db, run_id)
        await ProjectService().lock_workspace_boundary(db, project_id)
        await ProjectService().require_runnable_project(db, project_id)

        action = await self.reserve_action(
            db,
            run_id=run_id,
            idempotency_key=idempotency_key,
            action_type="start_graph",
            request=request_to_store,
            decision_id=decision_id,
        )
        if action.status != "reserved":
            return action

        try:
            await self._require_unmetered_control_provider_budget(db, run_id)
            stored_request = self._json_object_or_empty(action.request)
            graph_id = self._required_uuid(stored_request.get("graph_id"), "graph_id")
            subject_type = self._required_string(stored_request.get("subject_type"), "subject_type")
            subject_id = self._required_uuid(stored_request.get("subject_id"), "subject_id")
            project_id, task, gate, artifact_id = await self._graph_subject(
                db,
                run_id,
                subject_type,
                subject_id,
            )
            graph = (
                await db.execute(
                    select(Graph).where(
                        Graph.id == graph_id,
                        Graph.is_active.is_(True),
                        or_(Graph.project_id == project_id, Graph.project_id.is_(None)),
                    )
                )
            ).scalar_one_or_none()
            if graph is None:
                raise HTTPException(status_code=404, detail="Active graph not found")

            event = BusEvent(
                id=action.id,
                project_id=project_id,
                event_type="orchestration.graph_start_requested",
                payload={
                    "run_id": str(run_id),
                    "action_id": str(action.id),
                    "gate_id": str(gate.id),
                    "task_id": str(task.id),
                    "artifact_id": str(artifact_id) if artifact_id is not None else None,
                },
                source="orchestrator",
            )
            graph_run = await GraphEngineService().start_run(db, graph, event)
            await emit_event_once(
                db,
                project_id,
                GRAPH_STARTED_EVENT_TYPE,
                {
                    "run_id": str(run_id),
                    "action_id": str(action.id),
                    "graph_id": str(graph.id),
                    "graph_run_id": str(graph_run.id),
                    "task_id": str(task.id),
                    "gate_id": str(gate.id),
                },
                source="orchestrator",
                dedup_key=f"{GRAPH_STARTED_EVENT_TYPE}:action:{action.id}",
            )
        except Exception as exc:
            error = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
            await self._fail_reserved_action_for_current_flow(db, action, error)
            raise
        return await self._mark_action_completed(
            db,
            action,
            target_type="graph_run",
            target_id=graph_run.id,
        )

    async def execute_retry_task_action(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        request: dict,
        idempotency_key: str,
        decision_id: uuid.UUID | None = None,
        recovery_disposition: str | None = None,
        source_session_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        existing = await self._existing_action_for_key(db, run_id, idempotency_key)
        request_to_store = existing.request if existing is not None else self._canonical_retry_task_request(request)
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
        if task.status not in {"failed", "blocked"}:
            await self._fail_reserved_action_for_current_flow(db, action, f"Task is {task.status}")
            raise HTTPException(status_code=409, detail=f"Task is {task.status}")
        if await self._task_has_active_session(db, task.id):
            await self._fail_reserved_action_for_current_flow(db, action, "Task already has an active session")
            raise HTTPException(status_code=409, detail="Task already has an active session")
        if recovery_disposition is None:
            reason = await self._non_agent_failure_reason(db, task.id)
            if reason:
                await self._fail_reserved_action_for_current_flow(db, action, reason)
                raise HTTPException(status_code=409, detail=reason)
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
            from huddleroom.services.session_service import SessionService
            if await SessionService.handle_claim_refusal(db, exc):
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
        if task.status not in {"failed", "blocked"}:
            await self._fail_reserved_action_for_current_flow(db, action, f"Task is {task.status}")
            raise HTTPException(status_code=409, detail=f"Task is {task.status}")
        if await self._task_has_active_session(db, task.id):
            await self._fail_reserved_action_for_current_flow(db, action, "Task already has an active session")
            raise HTTPException(status_code=409, detail="Task already has an active session")

        await self._reserve_recovery_task_budget(db, run_id, action, task)
        self._set_task_budget_action(task, action.id)

        previous_agent_id = task.assigned_to
        previous_status = task.status
        task.assigned_to = agent.id
        await db.flush()
        try:
            _, session_id = await TaskService().run(db, project_id, task.id)
        except Exception as exc:
            task.assigned_to = previous_agent_id
            task.status = previous_status
            await db.flush()
            error = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
            from huddleroom.services.session_service import SessionService
            claim_refused = await SessionService.handle_claim_refusal(db, exc)
            await self._fail_reserved_action_for_current_flow(db, action, error)
            if claim_refused:
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

    def _set_task_budget_action(self, task: Task, action_id: uuid.UUID) -> None:
        metadata = self._json_object_or_empty(task.metadata_)
        task.metadata_ = {
            **metadata,
            "orchestration": {**self._json_object_or_empty(metadata.get("orchestration")), "action_id": str(action_id)},
        }

    async def _require_unmetered_control_provider_budget(self, db: AsyncSession, run_id: uuid.UUID) -> None:
        """Meeting/graph providers cannot report the durable measurements a cap requires."""
        from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService

        run = await db.get(OrchestrationRun, run_id)
        goal = await db.get(OrchestrationGoal, run.goal_id) if run is not None else None
        if goal is not None and run is not None and (await OrchestrationBudgetService().snapshot_for_run(db, goal, run))["caps"]:
            raise HTTPException(status_code=409, detail="Budgeted control provider requires measured session telemetry")

    # ---- recovery triggers (SPR 180/184) ---------------------------------------------------

    async def create_recovery_trigger(
        self, db: AsyncSession, run: OrchestrationRun, *, source_kind: str, source_id: uuid.UUID,
        blocker_code: str, summary: str, retry_target: dict[str, Any], fingerprint_basis: Any,
    ) -> tuple[OrchestrationAction | None, bool]:
        """Server-built ``RecoveryTriggerV1``; idempotent. Returns (trigger, newly created); (None, False) when
        the source belongs to recovery's own lineage (never a recursive coordinator)."""
        from huddleroom.schemas.orchestration_recovery import RecoveryTriggerV1
        from huddleroom.services.orchestration_blocker_recovery import OrchestrationBlockerRecoveryService, _trigger_key

        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_basis, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        trigger = RecoveryTriggerV1.model_validate({
            "schema_version": 1, "goal_id": str(run.goal_id), "run_id": str(run.id), "source_kind": source_kind,
            "source_id": str(source_id), "blocker_code": blocker_code[:255], "summary": summary[:4_000],
            "source_fingerprint": fingerprint, "retry_target": retry_target,
        })
        existed = await self._existing_action_for_key(db, run.id, _trigger_key(trigger)) is not None
        try:
            action = await OrchestrationBlockerRecoveryService().create_trigger(db, trigger)
        except ValueError:
            return None, False
        return action, not existed

    async def open_recovery_trigger(
        self, db: AsyncSession, run_id: uuid.UUID, source_fingerprint: str,
    ) -> OrchestrationAction | None:
        """A trigger with this fingerprint whose lineage is still open (awaiting diagnosis or approval/apply)."""
        for row in (await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run_id, OrchestrationAction.action_type == "recovery_trigger",
        ))).all():
            if (row.request or {}).get("source_fingerprint") != source_fingerprint:
                continue
            if row.status == "reserved":
                return row
            result = await db.get(OrchestrationAction, row.target_id) if row.target_id else None
            if result is not None and result.status == "reserved":
                return row
        return None

    async def recovery_holds_decision_loop(self, db: AsyncSession, run_id: uuid.UUID) -> bool:
        """An approved proposal is applying/verifying/retrying toward a sealed retry other than the decision loop:
        that retry owns progress this tick. A ``decision_loop`` retry never holds the loop it waits on."""
        from huddleroom.services.orchestration_blocker_recovery import (
            OrchestrationBlockerRecoveryService, RecoveryLifecycleStatus,
        )

        service = OrchestrationBlockerRecoveryService()
        for proposal in (await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run_id, OrchestrationAction.action_type == "recovery_proposal",
            OrchestrationAction.status == "reserved",
        ))).all():
            target = ((proposal.request or {}).get("retry") or {}).get("target") or {}
            if target.get("kind") in (None, "decision_loop"):
                continue
            if await service.project_status(db, proposal) in {
                RecoveryLifecycleStatus.APPLYING, RecoveryLifecycleStatus.VERIFYING, RecoveryLifecycleStatus.RETRYING,
            }:
                return True
        return False

    async def open_recovery_trigger_for_action(
        self, db: AsyncSession, run_id: uuid.UUID, action_type: str, request: dict[str, Any],
    ) -> OrchestrationAction | None:
        """An open lineage whose sealed retry re-runs this blocked action family (planning <-> request_plan; an
        action target <-> the same action and request): a corrective re-selection joins it, no new lineage."""
        def bare(value: Any) -> dict:
            return {k: v for k, v in (value or {}).items() if k != "reason"}

        for row in (await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run_id, OrchestrationAction.action_type == "recovery_trigger",
        ).order_by(OrchestrationAction.created_at, OrchestrationAction.id))).all():
            target = (row.request or {}).get("retry_target") or {}
            if target.get("kind") == "planning":
                if action_type != "request_plan":
                    continue
            elif target.get("kind") == "action":
                source = await db.get(OrchestrationAction, uuid.UUID(str(target.get("source_action_id"))))
                if source is None or source.action_type != action_type or bare(source.request) != bare(request):
                    continue
            else:
                continue
            if row.status == "reserved":
                return row
            result = await db.get(OrchestrationAction, row.target_id) if row.target_id else None
            if result is not None and result.id != row.id and result.status == "reserved":
                return row
        return None

    async def open_recovery_lineage(
        self, db: AsyncSession, run_id: uuid.UUID, source_kind: str, blocker_code: str | None = None,
    ) -> OrchestrationAction | None:
        """An open lineage (trigger awaiting diagnosis, or its outcome still awaiting approval/apply) of this kind."""
        for row in (await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run_id, OrchestrationAction.action_type == "recovery_trigger",
        ).order_by(OrchestrationAction.created_at, OrchestrationAction.id))).all():
            request = row.request or {}
            if request.get("source_kind") != source_kind or blocker_code not in (None, request.get("blocker_code")):
                continue
            if row.status == "reserved":
                return row
            result = await db.get(OrchestrationAction, row.target_id) if row.target_id else None
            if result is not None and result.id != row.id and result.status == "reserved":
                return row
        return None

    async def _failure_source(
        self, db: AsyncSession, run: OrchestrationRun, key: str, request: dict[str, Any],
        target_type: str, target_id: uuid.UUID, error: str,
    ) -> OrchestrationAction:
        """Durable failed record of an observed failure that has no failed action of its own (a failed task
        session, a failed gate); it is the ``failed_action`` source of the legacy-path recovery trigger."""
        marker = await self.reserve_action(db, run.id, key, "failure_observed", request)
        if marker.status == "reserved":
            marker.target_type, marker.target_id = target_type, target_id
            await self._fail_reserved_action_in_transaction(db, marker, error[:4_000])
        return marker

    async def _task_produced_by_recovery(self, db: AsyncSession, task: Task) -> bool:
        """A delegated remediation task fails inside its own recovery lineage, never a new one."""
        orchestration = self._json_object_or_empty(self._json_object_or_empty(task.metadata_).get("orchestration"))
        try:
            action_id = uuid.UUID(str(orchestration.get("action_id")))
        except ValueError:
            return False
        producer = await db.get(OrchestrationAction, action_id)
        return producer is not None and producer.idempotency_key.startswith("recovery:")

    async def _task_recovery_trigger(
        self, db: AsyncSession, run: OrchestrationRun, task: Task, *, blocker_code: str, summary: str,
        closes_when_task_healthy: bool = True,
    ) -> int:
        """Legacy retry/reassign replacement: one trigger per failed-attempt episode, nothing mutated."""
        if await self._task_produced_by_recovery(db, task):
            return 0
        session_id = await db.scalar(select(Session.id).where(Session.task_id == task.id, Session.status == "failed")
                                     .order_by(Session.created_at.desc(), Session.id.desc()).limit(1))
        effect = await self._recovery_effect_for_attempt(db, run, task)
        if effect is not None and await self.route_failure_to_recovery_lineage(db, run, effect, summary):
            return 0
        episode = f"session:{session_id}" if session_id else f"status:{task.status}"
        source = await self._failure_source(
            db, run, f"run:{run.id}:kind:failure_observed:task:{task.id}:{episode}",
            {"task_id": str(task.id), "session_id": str(session_id) if session_id else None,
             "blocker_code": blocker_code, "closes_when_task_healthy": closes_when_task_healthy},
            "task", task.id, summary,
        )
        _trigger, created = await self.create_recovery_trigger(
            db, run, source_kind="failed_action", source_id=source.id, blocker_code=blocker_code, summary=summary,
            retry_target={"kind": "decision_loop"}, fingerprint_basis={"source": str(source.id)},
        )
        return int(created)

    async def _gate_recovery_trigger(
        self, db: AsyncSession, run: OrchestrationRun, gate: OrchestrationGate, evidence_id: uuid.UUID,
        *, blocker_code: str, summary: str, retry_action: dict[str, Any] | None = None,
        link: dict[str, Any] | None = None,
    ) -> int:
        """Failed-gate replacement/fix work becomes one trigger per rejected evidence, nothing mutated.

        The gate-linked creation the legacy path would have run is sealed on the source record as the
        approved retry (``retry_action`` + ``link``): after approval the one retry dispatches it under the
        recovery effect key and ``link_recovery_retry_effect`` links the new task to the gate."""
        request = {"gate_id": str(gate.id), "evidence_id": str(evidence_id), "blocker_code": blocker_code}
        if retry_action is not None:
            request.update({"retry_action": retry_action, "link": link or {}})
        source = await self._failure_source(
            db, run, f"run:{run.id}:kind:failure_observed:gate:{gate.id}:evidence:{evidence_id}",
            request, "gate", gate.id, summary,
        )
        retry_target = ({"kind": "action", "source_action_id": str(source.id)}
                        if (source.request or {}).get("retry_action") else {"kind": "decision_loop"})
        _trigger, created = await self.create_recovery_trigger(
            db, run, source_kind="failed_action", source_id=source.id, blocker_code=blocker_code, summary=summary,
            retry_target=retry_target, fingerprint_basis={"source": str(source.id)},
        )
        return int(created)

    async def final_summary_retry_request(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun,
    ) -> dict[str, Any]:
        """Build the approved replacement summary request now (fit + accepted manifest). Unavailable ->
        ValueError, which fails the attempt into its lineage as evidence instead of crashing a tick."""
        fits = await OrchestrationRosterMapper().rank_agents(
            db, goal.project_id, FINAL_SUMMARY_WORK_FUNCTION, required_capabilities=[FINAL_SUMMARY_WORK_FUNCTION],
        )
        if not fits or fits[0].weak:
            raise ValueError("final_summary_unavailable: no strong summarization agent fit")
        try:
            gates, evidence = await self._accepted_non_summary_manifest(db, run.id)
        except HTTPException as exc:
            raise ValueError(f"final_summary_unavailable: {exc.detail}") from None
        request = self._final_summary_delegation_request(goal, run, fits[0].agent_id, gates, evidence)
        request["orchestrator_context"] = {
            **self._json_object_or_empty(request.get("orchestrator_context")), "final_summary_recovery": True,
        }
        return request

    async def link_recovery_retry_effect(
        self, db: AsyncSession, source: OrchestrationAction, effect: OrchestrationAction,
    ) -> None:
        """Link the task created by an approved gate retry to its gate, exactly as the legacy path did."""
        link = self._json_object_or_empty((source.request or {}).get("link"))
        if not link or effect.status != "completed" or effect.target_type != "task" or effect.target_id is None:
            return
        task = await db.get(Task, effect.target_id)
        gate = await db.get(OrchestrationGate, uuid.UUID(str(link["gate_id"])))
        if task is None or gate is None:
            return
        if link.get("kind") == "final_summary":
            self._set_final_summary_task_metadata(task, gate, self._json_object_or_empty(link.get("final_summary")))
        elif link.get("kind") == "review_fix":
            metadata = self._json_object_or_empty(task.metadata_)
            metadata["orchestration"] = {
                **self._json_object_or_empty(metadata.get("orchestration")),
                "plan_item_gate_id": str(gate.id), "recovery_kind": "review_rejected_fix",
            }
            task.metadata_ = metadata
        await db.flush()

    async def route_failure_to_recovery_lineage(
        self, db: AsyncSession, run: OrchestrationRun, effect: OrchestrationAction, error: str,
    ) -> bool:
        """A failed attempt produced by an approved recovery effect belongs to that lineage: fail its still-active
        attempt (feeding the lineage's next diagnosis) instead of starting another lineage. False when the
        lineage already resolved (a new failure is a new blocker) or the proposal is unknown."""
        parts = effect.idempotency_key.split(":")
        if len(parts) < 3 or parts[0] != "recovery":
            return False
        proposal = await self._existing_action_for_key(db, run.id, f"recovery:{parts[1]}:proposal")
        if proposal is None or proposal.status == "completed":
            return False
        if proposal.status == "reserved":
            from huddleroom.schemas.orchestration_recovery import EvidenceV1
            from huddleroom.services.orchestration_blocker_recovery import OrchestrationBlockerRecoveryService

            goal = await db.get(OrchestrationGoal, run.goal_id)
            await OrchestrationBlockerRecoveryService()._fail_attempt(  # pylint: disable=protected-access
                db, proposal, run, goal, error, [EvidenceV1(
                    label="The approved attempt failed.", value={"error": error[:300]},
                    source_kind="action", source_id=effect.id)],
            )
        return True  # failed proposals already re-entered diagnosis in their lineage

    async def _recovery_effect_for_attempt(
        self, db: AsyncSession, run: OrchestrationRun, task: Task,
    ) -> OrchestrationAction | None:
        """The approved recovery effect (retry/reassign) that started this task's LATEST failed session, if any; an
        older failure of a recovery attempt never swallows a later ordinary failure."""
        rows = (await db.execute(select(Session.id, Session.created_at).where(
            Session.task_id == task.id, Session.status == "failed"))).all()
        if not rows:
            return None
        latest = max(created for _id, created in rows)
        failed = [session_id for session_id, created in rows if created == latest]  # ties: same-instant writes
        rows = (await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id, OrchestrationAction.target_id.in_(failed),
            OrchestrationAction.idempotency_key.startswith("recovery:"),
        ).order_by(OrchestrationAction.created_at.desc(), OrchestrationAction.id.desc()))).all()
        return next((row for row in rows if row.idempotency_key.endswith(":effect")
                     and str((row.request or {}).get("task_id")) == str(task.id)), None)

    async def planning_preflight_trigger(self, db: AsyncSession, run_id: uuid.UUID, detail: str) -> None:
        """A stable planning/preflight blocker (e.g. plan_author_unavailable) starts one recovery lineage per
        team-configuration state; a recovery-applied configuration change is a new state."""
        run = await db.get(OrchestrationRun, run_id)
        if run is None or run.status not in ACTIVE_RUN_STATUSES:
            return
        from huddleroom.services.orchestration_process_service import OrchestrationProcessService

        processes = OrchestrationProcessService()
        state = [str(getattr(await processes.get_current(db, run.goal_id, kind), "id", None))
                 for kind in ("agent_definition_review", "team_hierarchy")]
        await self.create_recovery_trigger(
            db, run, source_kind="planning_preflight", source_id=run.id,
            blocker_code=detail.split(":", 1)[0].strip(), summary=f"Planning cannot start: {detail}",
            retry_target={"kind": "planning"}, fingerprint_basis={"detail": detail, "configuration": state},
        )

    async def live_work_refs(self, db: AsyncSession, run: OrchestrationRun, source) -> list:
        """Server-built live producers of this run for ``OrchestrationWakeSourceService``; model output never
        contributes. The service decides relevance to ``source`` and whether each producer can progress."""
        from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
        from huddleroom.models.orchestration_steering import OrchestrationSteeringRequest
        from huddleroom.services.orchestration_wake_source import LiveWorkRefV1

        def ref(kind: str, row_id: uuid.UUID) -> LiveWorkRefV1:
            return LiveWorkRefV1(kind, row_id, source.kind, source.id)

        tasks = await self._orchestrated_tasks_for_run(
            db, run.id, statuses=["backlog", "ready", "in_progress", "blocked", "failed"])
        task_ids = [task.id for task in tasks]
        refs = [ref("task", task.id) for task in tasks if task.status in {"backlog", "ready", "in_progress"}]
        if task_ids:
            refs += [ref("session", row) for row in await db.scalars(select(Session.id).where(
                Session.task_id.in_(task_ids), Session.status.in_(("pending", "running"))))]
            refs += [ref("meeting", row) for row in await db.scalars(select(Meeting.id).where(
                Meeting.source_task_id.in_(task_ids),
                Meeting.status.in_(("scheduled", "preparing", "active", "concluding"))))]
            refs += [ref("graph_run", row) for row in await db.scalars(select(GraphRun.id).where(
                GraphRun.linked_task_id.in_(task_ids), GraphRun.status == "active"))]
        refs += [ref("steering_request", row) for row in await db.scalars(select(OrchestrationSteeringRequest.id).where(
            OrchestrationSteeringRequest.goal_id == run.goal_id, OrchestrationSteeringRequest.submitted_run_id == run.id,
            OrchestrationSteeringRequest.status.in_(("pending", "being_considered"))))]
        refs += [ref("authority_decision", row) for row in await db.scalars(select(OrchestrationAuthorityDecision.id).where(
            OrchestrationAuthorityDecision.goal_id == run.goal_id, OrchestrationAuthorityDecision.run_id == run.id,
            OrchestrationAuthorityDecision.status == "pending"))]
        return refs

    async def _eventless_noop_trigger(self, db: AsyncSession, run: OrchestrationRun, decision: OrchestrationDecision) -> None:
        """SPR 184: an accepted noop whose wake has no relevant live producer becomes a recovery trigger. The
        verdict comes from server-built live work only, never from the noop's prose."""
        from huddleroom.services.orchestration_wake_source import OrchestrationWakeSourceService, SourceRef

        wake_when = self._json_object_or_empty(decision.parsed_decision).get("wake_when")
        if not isinstance(wake_when, Mapping):
            return
        source = SourceRef("run", run.id)
        verdict = await OrchestrationWakeSourceService().evaluate(
            db, run, blocked_source=source, wake_when=dict(wake_when),
            live_work_refs=await self.live_work_refs(db, run, source),
        )
        if verdict.verdict == "live" or await self.open_recovery_lineage(db, run.id, "eventless_noop") is not None:
            return  # one open eventless lineage per run: a stuck loop's repeated noops do not stack lineages
        await self.create_recovery_trigger(
            db, run, source_kind="eventless_noop", source_id=decision.id, blocker_code="eventless_noop",
            summary=f"The orchestrator chose to wait, but no live work can end the wait ({verdict.reason_code}).",
            retry_target={"kind": "decision_loop"}, fingerprint_basis={"decision": str(decision.id)},
        )

    async def execute_resolve_meeting_action_item_action(
        self, db: AsyncSession, run_id: uuid.UUID, request: dict, idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        run = await db.get(OrchestrationRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Orchestration run not found")
        from huddleroom.services.orchestration_baseline_runtime import require_baseline_available
        async with self._lock_goal_for_baseline_transition(db, run.goal_id):
            require_baseline_available(run.goal_id)
            # Hold a database write lock through the effect, including human PATCH and reset races.
            await db.execute(update(OrchestrationRun).where(OrchestrationRun.id == run_id)
                             .values(status=OrchestrationRun.status, updated_at=OrchestrationRun.updated_at)
                             .execution_options(synchronize_session=False))
            run = await db.get(OrchestrationRun, run_id, populate_existing=True)
            if run is None:
                raise HTTPException(status_code=409, detail="Orchestration run discarded by reset")
            goal = await db.get(OrchestrationGoal, run.goal_id, populate_existing=True)
            current = await self.get_active_run_for_goal(db, goal.project_id, goal.id) if goal else None
            if (goal is None or goal.status not in {"active", "blocked"} or current is None or current.id != run.id
                    or run.phase != "authorized" or run.status not in {"running", "blocked"}):
                raise HTTPException(status_code=409, detail="Waiver requires the current authorized active run")
            existing = await self._existing_action_for_key(db, run_id, idempotency_key)
            stored = existing.request if existing is not None else self._canonical_resolve_meeting_action_item_request(request)
            action = await self.reserve_action(db, run_id, idempotency_key, "resolve_meeting_action_item", stored,
                                               decision_id=decision_id)
            if action.status != "reserved":
                return action
            item, _changed = await waive_meeting_action_item(
                db, item_id=self._required_uuid(action.request.get("item_id"), "item_id"),
                reason=action.request.get("reason"), run_id=run_id, action_id=action.id,
            )
            return await self._mark_action_completed(db, action, "meeting_action_item", item.id)

    async def execute_recover_blocker_action(
        self, db: AsyncSession, run_id: uuid.UUID, request: dict, idempotency_key: str,
        decision_id: uuid.UUID | None = None,
    ) -> OrchestrationAction:
        """Non-mutating: records the reported blocker and starts its recovery lineage. Diagnosis runs later,
        outside the decision path (tick), so no analyzer call happens here."""
        action = await self.reserve_action(db, run_id, idempotency_key, "recover_blocker", deepcopy(request),
                                           decision_id=decision_id)
        if action.status != "reserved":
            return action
        run = await db.get(OrchestrationRun, run_id)
        reason = self._optional_string(request.get("reason")) or "The orchestrator reported a blocker."
        trigger = await self.open_recovery_lineage(db, run_id, "reported_blocker", "reported_blocker")
        if trigger is not None:  # repeated reports while a lineage is open join it
            await self._mark_action_completed(db, action, "orchestration_action", trigger.id)
            return action
        trigger, _created = await self.create_recovery_trigger(
            db, run, source_kind="reported_blocker", source_id=decision_id or action.id, blocker_code="reported_blocker",
            summary=reason, retry_target={"kind": "decision_loop"},
            fingerprint_basis={"evidence_refs": request.get("evidence_refs") or []},
        )
        await self._mark_action_completed(db, action, "orchestration_action", trigger.id if trigger else action.id)
        return action

    async def classify_corrective_action(
        self, db: AsyncSession, run: OrchestrationRun, action_type: str, request: dict[str, Any], *, source: str,
    ) -> CorrectiveDisposition:
        """Spec "Corrective-action ownership". Read-only; the disposition comes from durable task/gate/action
        state and effect identity only, so ``source`` (the selecting path) and reason text never change it.

        A corrective action may only resume the identical APPROVED pending effect; an ordinary action may
        also resume its identical pending ordinary row (transport/provider retry)."""
        del source
        if action_type == "resolve_meeting_action_item":
            return CorrectiveDisposition.ORDINARY  # Waives the obligation only; never retries or repairs work.
        if action_type in _NON_MUTATING_ACTIONS:
            return CorrectiveDisposition.ORDINARY  # records or asks; never a fix (spec: not mutation tools)
        from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher

        identity = OrchestrationDecisionDispatcher.action_key(run.id, action_type, request)
        ordinary_key = identity
        if action_type == "request_roadmap_replan":
            try:
                ordinary_key = await self.roadmap_replan_action_key(db, run)
            except HTTPException:
                ordinary_key = None  # no accepted version: no ordinary row can exist under that key
        rows = (await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == action_type,
            OrchestrationAction.status.in_(("reserved", "failed", "completed")),
        ))).all()
        same = [row for row in rows if row.status != "completed"
                and OrchestrationDecisionDispatcher.action_key(run.id, action_type, row.request or {}) == identity]
        effects = await self._recovery_effect_lineages(db, run, action_type, [
            row for row in rows if row.status == "completed"
            and row.idempotency_key.startswith("recovery:") and row.idempotency_key.endswith(":effect")])

        def ordinary_row(row: OrchestrationAction) -> bool:
            return ordinary_key is not None and (
                row.idempotency_key == ordinary_key or row.idempotency_key.startswith(ordinary_key + ":"))

        def recovered(row: OrchestrationAction) -> bool:
            # B1/F1: a later completed approved effect recovered THIS failure only when its lineage names this row
            # (sealed retry source / trigger source), it re-ran this exact identity, or it is the planning retry.
            return any(created_at >= row.created_at and (row.id in sources or effect_identity == identity or planning)
                       for created_at, sources, effect_identity, planning in effects)

        corrective = action_type in {"retry_task", "reassign_task"} or any(
            row.status == "failed" and ordinary_row(row)  # a new attempt after a durable failure
            and not recovered(row) for row in same
        ) or await self._has_corrective_source(db, run, action_type, request)
        for row in same:
            if row.status != "reserved":
                continue
            if row.idempotency_key.startswith("recovery:") and row.idempotency_key.endswith(":effect"):
                wrapper = await self._existing_action_for_key(db, run.id, row.idempotency_key.removesuffix(":effect"))
                if wrapper is not None and (wrapper.dispatch_contract or {}).get("effect_action_id") == str(row.id):
                    return CorrectiveDisposition.SAME_ATTEMPT_RECONCILIATION
            elif not corrective and ordinary_row(row):
                return CorrectiveDisposition.SAME_ATTEMPT_RECONCILIATION
        return CorrectiveDisposition.REQUIRES_RECOVERY if corrective else CorrectiveDisposition.ORDINARY

    async def _recovery_effect_lineages(
        self, db: AsyncSession, run: OrchestrationRun, action_type: str, effects: list[OrchestrationAction],
    ) -> list[tuple[datetime, set[uuid.UUID], str, bool]]:
        """(created_at, lineage source ids, canonical identity, planning retry) per completed recovery effect.
        The proposal is found by the ``proposal_id`` embedded in the effect key ``recovery:<proposal_id>:...``."""
        from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher

        if not effects:
            return []
        proposals = {str((row.request or {}).get("proposal_id")): row.request or {}
                     for row in (await db.scalars(select(OrchestrationAction).where(
                         OrchestrationAction.run_id == run.id,
                         OrchestrationAction.action_type == "recovery_proposal"))).all()}
        lineages = []
        for effect in effects:
            proposal = proposals.get(effect.idempotency_key.split(":")[1], {})
            target = (proposal.get("retry") or {}).get("target") or {}
            sources = set()
            for value in (target.get("source_action_id"), (proposal.get("source") or {}).get("id")):
                try:
                    sources.add(uuid.UUID(str(value)))
                except ValueError:
                    pass
            lineages.append((effect.created_at, sources,
                             OrchestrationDecisionDispatcher.action_key(run.id, action_type, effect.request or {}),
                             target.get("kind") == "planning" and action_type == "request_plan"))
        return lineages

    async def _has_corrective_source(
        self, db: AsyncSession, run: OrchestrationRun, action_type: str, request: dict[str, Any],
    ) -> bool:
        """The action targets a failed/blocked task or a failed gate. Recovery lineage alone is not a corrective
        source: healthy work on a task produced by an approved recovery effect is ordinary ;
        once that target durably fails or blocks, the status checks below make the next attempt corrective."""
        task_ids, gate_ids = [], []
        if action_type == "request_verification":
            gate_ids.append(request.get("gate_id"))
        elif action_type == "schedule_meeting":
            task_ids.append(request.get("task_id"))
            gate_ids.append(request.get("gate_id"))
        elif action_type == "start_graph" and request.get("subject_type") == "task":
            task_ids.append(request.get("subject_id"))
        elif action_type == "request_plan_revision":
            task_ids.append(request.get("plan_task_id"))
        elif action_type == "create_delegation_task":
            task_ids.append(request.get("parent_task_id"))  # a fix/replacement for its failed parent
        elif action_type == "request_roadmap_replan":
            return bool(await self._orchestrated_tasks_for_run(db, run.id, statuses=["failed"]))
        task_ids = [uuid.UUID(str(value)) for value in task_ids if value]
        gate_ids = [uuid.UUID(str(value)) for value in gate_ids if value]
        if task_ids and await db.scalar(select(func.count(Task.id)).where(  # pylint: disable=not-callable
            Task.id.in_(task_ids), Task.status.in_(("failed", "blocked")),
            Task.metadata_["orchestration"]["run_id"].as_string() == str(run.id),
        )):
            return True
        return bool(gate_ids) and bool(await db.scalar(select(func.count(OrchestrationGate.id)).where(  # pylint: disable=not-callable
            OrchestrationGate.id.in_(gate_ids), OrchestrationGate.run_id == run.id, OrchestrationGate.status == "failed",
        )))

    async def verification_attempt_suffix(self, db: AsyncSession, run_id: uuid.UUID, gate_id: Any) -> str:
        """Key suffix that makes a verification retryable: one attempt per prior FAILED
        request_verification for the gate. Attempt 0 has no suffix (legacy keys unchanged)."""
        rows = (await db.execute(
            select(OrchestrationAction.request).where(
                OrchestrationAction.run_id == run_id,
                OrchestrationAction.action_type == "request_verification",
                OrchestrationAction.status == "failed",
            )
        )).scalars().all()
        failed = sum(1 for req in rows if isinstance(req, dict) and str(req.get("gate_id")) == str(gate_id))
        return f":attempt:{failed}" if failed else ""

    async def _task_has_active_session(self, db: AsyncSession, task_id: uuid.UUID) -> bool:
        result = await db.execute(
            select(Session.id).where(Session.task_id == task_id, Session.status.in_(["pending", "running"])).limit(1)
        )
        return result.first() is not None

    async def _non_agent_failure_reason(self, db: AsyncSession, task_id: uuid.UUID) -> str | None:
        """Reason the latest failed session never launched the agent (so a retry is pointless), else None."""
        result = await db.execute(
            select(Session).where(Session.task_id == task_id)
            .order_by(Session.created_at.desc(), Session.id.desc()).limit(1)
        )
        session = result.scalar_one_or_none()
        if session is None or session.status != "failed":
            return None
        metadata = session.metadata_ if isinstance(session.metadata_, dict) else {}
        attempt = metadata.get("attempt") if isinstance(metadata.get("attempt"), dict) else {}
        if attempt.get("retired") == "interrupted_safe":
            return None  # the runner died before effects; a fresh (approved) attempt is meaningful
        # effect_state is only written for orchestrated sessions; manual sessions carry no attempt record.
        never_ran = attempt.get("effect_state") != "started" if attempt else session.started_at is None
        if never_ran or (session.error or "").startswith("command_not_found"):
            return f"Not retryable: agent never ran ({session.error or 'unknown error'}); fix the cause before retrying"
        return None

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
        if (
            gate.gate_type == PLAN_GATE_TYPE and run is not None
            and self._json_object_or_empty(run.plan_state).get("plan_artifact_id")
        ):
            try:
                source_task = await self._ingested_planning_task_for_gate(db, run, gate)
            except HTTPException as exc:
                await self._fail_reserved_action_for_current_flow(db, action, str(exc.detail))
                raise
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
        if source_task is not None and source_task.assigned_to is not None:
            excluded_agent_ids.add(source_task.assigned_to)
        source_contract = self._json_object_or_empty(
            self._json_object_or_empty(source_task.metadata_ if source_task else {}).get("orchestration_contract")
        )
        source_context = self._json_object_or_empty(source_contract.get("orchestrator_context"))
        verification_context = source_context or (delegable_context(goal.orchestrator_context) if goal else {})
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
                        "Report fresh evidence only; do not edit artifacts. "
                        "Final evidence must fit each criterion: live observation where the criterion "
                        "requires it; otherwise deterministic unit tests, document review, static "
                        "analysis, or another accepted method. Preliminary readiness reports never "
                        "satisfy a final criterion."
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
                idempotency_key=(
                    f"run:{run_id}:kind:create_delegation_task:verify_gate:{gate.id}"
                    + await self.verification_attempt_suffix(db, run_id, gate.id)
                ),
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
        return stable_hash(value)

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
            "resolve_meeting_action_item": self._canonical_resolve_meeting_action_item_request,
            "retry_task": self._canonical_retry_task_request,
            "reassign_task": self._canonical_reassign_task_request,
            "schedule_meeting": self._canonical_schedule_meeting_request,
            "start_graph": self._canonical_start_graph_request,
            "ask_human": self._canonical_ask_human_request,
            "pause_run": self._canonical_pause_run_request,
            "suggest_agent": self._canonical_suggest_agent_request,
            "recover_blocker": self._canonical_recover_blocker_request,
        }
        return canonicalizers[action_type](request)

    def _canonical_resolve_meeting_action_item_request(self, request: Any) -> dict[str, Any]:
        request_obj = self._json_object_or_empty(request)
        reason = waiver_reason(request_obj.get("reason"))
        return {"action_type": "resolve_meeting_action_item",
                "item_id": str(self._required_uuid(request_obj.get("item_id"), "item_id")), "reason": reason}

    def _canonical_recover_blocker_request(self, request: Any) -> dict[str, Any]:
        request_obj = self._json_object_or_empty(request)
        refs = request_obj.get("evidence_refs") if isinstance(request_obj.get("evidence_refs"), list) else []
        return {
            "action_type": "recover_blocker",
            "reason": self._optional_string(request_obj.get("reason")),
            "evidence_refs": [{"kind": str(item.get("kind")), "id": str(item.get("id"))}
                              for item in refs if isinstance(item, Mapping)],
        }

    def _canonical_noop_request(self, request: Any) -> dict[str, Any]:
        return {
            "action_type": "noop",  # action request, not an LLM decision: wake_when lives on the decision, not here
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
        canonical.update({key: request_obj[key] for key in REMEDIATION_REQUEST_KEYS if key in request_obj})
        title = request_obj.get("title")
        if isinstance(title, str) and title.strip():
            canonical["title"] = title.strip()
        raw_deps = request_obj.get("depends_on")
        if raw_deps is not None and not isinstance(raw_deps, list):
            raise HTTPException(status_code=422, detail="depends_on must be a list of task UUIDs")
        if raw_deps:  # absent/empty keeps legacy action keys byte-identical
            deps = [str(self._required_uuid(d, "depends_on")) for d in raw_deps]
            if len(set(deps)) != len(deps):
                raise HTTPException(status_code=422, detail="depends_on must not contain duplicates")
            canonical["depends_on"] = deps
        # Plan-item passthrough fields: kept only when non-default so legacy action keys stay stable.
        priority = request_obj.get("priority")
        if priority is not None:
            if isinstance(priority, bool) or not isinstance(priority, int) or not 0 <= priority <= 100:
                raise HTTPException(status_code=422, detail="priority must be an integer from 0 to 100")
            if priority != 50:
                canonical["priority"] = priority
        raw_artifacts = request_obj.get("artifact_ids")
        if raw_artifacts is not None and not isinstance(raw_artifacts, list):
            raise HTTPException(status_code=422, detail="artifact_ids must be a list of artifact UUIDs")
        if raw_artifacts:
            canonical["artifact_ids"] = [str(self._required_uuid(a, "artifact_ids")) for a in raw_artifacts]
        why_runnable = request_obj.get("why_runnable")
        if why_runnable is not None and not isinstance(why_runnable, str):
            raise HTTPException(status_code=422, detail="why_runnable must be a string")
        if why_runnable and why_runnable.strip():
            canonical["why_runnable"] = why_runnable.strip()
        decision_use = request_obj.get("decision_use")
        if isinstance(decision_use, str) and decision_use.strip():
            canonical["decision_use"] = decision_use.strip()
        raw_scope = request_obj.get("authorization_scope")
        if raw_scope is not None:
            try:
                scope = AuthorizationScope.model_validate(raw_scope)
            except ValidationError:
                raise HTTPException(status_code=422, detail="authorization_scope is not a valid authorization scope")
            if scope.external_writes.allowed or scope.live_sends.allowed:
                raise HTTPException(status_code=422, detail=SELF_GRANT_ERROR)
            canonical["authorization_scope"] = scope.model_dump(mode="json")
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
            if key not in {"assumptions", "merged_decision_keys", "expansion_policy"}
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
        plan_items: list[dict] | None = None,
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
                gates, evidence, goal, roadmap_version_id, plan_items
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

    async def _ingested_planning_task_for_gate(
        self, db: AsyncSession, run: OrchestrationRun, gate: OrchestrationGate
    ) -> Task:
        """Resolve pre-acceptance verification from the current ingested plan's lineage."""
        state = self._json_object_or_empty(run.plan_state)
        if gate.run_id != run.id or state.get("plan_gate_id") != str(gate.id):
            raise HTTPException(status_code=409, detail="Plan gate does not match the current planning request")
        artifact = await self._plan_artifact_for_run(
            db, run, self._required_uuid(state.get("plan_artifact_id"), "plan_artifact_id")
        )
        task = await self._planning_task_for_run(db, run, artifact.linked_task_id)
        metadata = self._json_object_or_empty(artifact.metadata_)
        task_plan = self._json_object_or_empty(self._json_object_or_empty(task.metadata_).get("orchestration_plan"))
        if (
            metadata.get("source") != INGESTION_SOURCE
            or metadata.get("planning_task_id") != str(task.id)
            or metadata.get("planning_version") != state.get("planning_version")
            or task_plan.get("planning_version") != state.get("planning_version")
            or metadata.get("plan_author_agent_id") != str(task.assigned_to)
            or state.get("plan_author_agent_id") != str(task.assigned_to)
            or task.status != "done"
        ):
            raise HTTPException(status_code=409, detail="Plan artifact does not match current planning provenance")
        session_id = self._event_uuid(metadata.get("winning_session_id"))
        session = await db.get(Session, session_id) if session_id is not None else None
        corrected = session is not None and session.task_id != task.id and await linked_correction_session(
            db, run, task, session, state.get("planning_version"),
        )
        if (
            session is None or (session.task_id != task.id and not corrected) or session.agent_id != task.assigned_to
            or session.project_id != artifact.project_id or session.status != "completed"
        ):
            raise HTTPException(status_code=409, detail="Plan artifact winning session does not match its producer")
        marker = await self._existing_action_for_key(
            db, run.id, f"run:{run.id}:kind:report_consumed:task:{task.id}"
        )
        if (
            marker is None or marker.status != "completed"
            or marker.target_type != "artifact" or marker.target_id != artifact.id
            or self._event_uuid(marker.request.get("task_id")) != task.id
            or (self._event_uuid(marker.request.get("session_id")) != session.id and not corrected)
        ):
            raise HTTPException(status_code=409, detail="Plan artifact session does not match the consumed planning report")
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
        metadata = self._json_object_or_empty(artifact.metadata_)
        # The server-held plan_artifact_id keeps the binding even if the Artifact's metadata is rewritten.
        if metadata.get("source") == INGESTION_SOURCE or str(artifact.id) == plan_state.get("plan_artifact_id"):
            if metadata.get("planning_version") != plan_state.get("planning_version"):
                raise HTTPException(status_code=409, detail="Plan artifact is from a superseded planning version")
            if ingested_plan_hash(metadata) != plan_state.get("plan_output_hash"):
                raise HTTPException(status_code=409, detail="Plan artifact content does not match the ingested planning output")
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

    @staticmethod
    async def _first_missing_artifact(db: AsyncSession, project_id: uuid.UUID, artifact_ids: Sequence[Any]) -> uuid.UUID | None:
        wanted = {uuid.UUID(str(a)) for a in artifact_ids}
        if not wanted:
            return None
        found = set((await db.execute(
            select(Artifact.id).where(Artifact.id.in_(wanted), Artifact.project_id == project_id)
        )).scalars())
        return min(wanted - found, key=str) if wanted - found else None

    async def _validate_plan_adoptions(self, db: AsyncSession, run: OrchestrationRun, items: Sequence[Any]) -> None:
        """Each adopt_task_id must name this run's own delegated discovery work, adopted by one item only.

        Items already bound (Roadmap lineage row / expanded Outcome entry) are immutable and skipped."""
        goal = await db.get(OrchestrationGoal, run.goal_id)
        expanded = self._json_object_or_empty(run.plan_state).get("expanded_items")
        bound_keys = {
            entry.get("plan_item_id") for entry in expanded if isinstance(entry, Mapping)
        } if isinstance(expanded, list) else set()
        if goal is not None:
            bound_keys |= set(await db.scalars(select(OrchestrationRoadmapItem.item_key).where(
                OrchestrationRoadmapItem.goal_id == goal.id
            )))
        seen: set[uuid.UUID] = set()
        for item in items:
            roadmap = hasattr(item, "unit_type")
            key = item.item_key if roadmap else item.id
            if goal is not None:
                missing = await self._first_missing_artifact(db, goal.project_id, getattr(item, "artifact_ids", None) or [])
                if missing is not None:
                    raise HTTPException(
                        status_code=409, detail=f"artifact_invalid: item {key}: artifact {missing} not found in project"
                    )
            task_id = getattr(item, "adopt_task_id", None)
            if task_id is None:
                binding = next((entry for entry in expanded or [] if isinstance(entry, Mapping)
                                and entry.get("plan_item_id") == key), None)
                if binding is not None:
                    bound_task = await db.get(Task, uuid.UUID(binding["task_id"]))
                    if bound_task is None:
                        raise HTTPException(status_code=409, detail=f"adoption_invalid: item {key}: bound task missing")
                    self._validate_bound_plan_item(bound_task, run, item)
                continue

            def invalid(reason: str) -> HTTPException:
                return HTTPException(status_code=409, detail=f"adoption_invalid: item {key}: {reason}")

            if task_id in seen:
                raise invalid(f"task {task_id} is adopted by more than one item")
            seen.add(task_id)
            if roadmap and (item.unit_type != "task" or item.mutates_shared_state):
                raise invalid("only Roadmap task items with mutates_shared_state false can adopt work")
            if item.depends_on:
                raise invalid("an adopting item cannot have depends_on")
            task = await db.get(Task, task_id)
            orchestration = self._json_object_or_empty(
                self._json_object_or_empty(task.metadata_ if task is not None else None).get("orchestration")
            )
            if (
                task is None or goal is None or task.project_id != goal.project_id
                or orchestration.get("run_id") != str(run.id) or orchestration.get("goal_id") != str(run.goal_id)
            ):
                raise invalid(f"task {task_id} is not work of this run")
            if key in bound_keys:
                binding = next((entry for entry in expanded or [] if isinstance(entry, Mapping)
                                and entry.get("plan_item_id") == key), None)
                roadmap_binding = await db.scalar(select(OrchestrationRoadmapItem).where(
                    OrchestrationRoadmapItem.goal_id == run.goal_id, OrchestrationRoadmapItem.item_key == key,
                )) if roadmap else None
                bound_task_id = str(roadmap_binding.task_id) if roadmap_binding else (binding or {}).get("task_id")
                if bound_task_id != str(task_id):
                    raise invalid("task binding changed")
                metadata_view = None
                if roadmap:
                    from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService
                    metadata_view = await OrchestrationRoadmapService(self)._released_task_metadata(
                        db, run, item, roadmap_binding, task
                    )
                self._validate_bound_plan_item(task, run, item, metadata_view)
            if task.status == "cancelled":
                raise invalid(f"task {task_id} is cancelled")
            if (
                not self._json_object_or_empty(task.metadata_).get("orchestration_contract")
                or await db.scalar(select(OrchestrationAction.id).where(
                    OrchestrationAction.id == task_id,
                    OrchestrationAction.action_type == "create_delegation_task",
                    OrchestrationAction.run_id == run.id,
                )) is None
            ):
                raise invalid(f"task {task_id} was not created by a delegation")
            work_function = self._task_work_function(task)
            if work_function in NON_ADOPTABLE_WORK_FUNCTIONS:
                raise invalid(f"task {task_id} has non-adoptable work_function '{work_function}'")
            if work_function != item.work_function:
                raise invalid(f"task {task_id} work_function '{work_function}' does not match '{item.work_function}'")
            if item.agent_id is not None and item.agent_id != task.assigned_to:
                raise invalid(f"task {task_id} is assigned to another agent")
            gate_id = task_gate_id(task, run.id)
            if gate_id is not None:
                gate = await db.get(OrchestrationGate, gate_id)
                gate_key = f"roadmap_item:{key}" if roadmap else self._plan_item_gate_key(key)
                own_adhoc = (
                    gate is not None and gate.gate_type == ADHOC_GATE_TYPE
                    and self._json_object_or_empty(gate.required_evidence).get("task_id") == str(task.id)
                )
                if not own_adhoc and (
                    gate is None or gate.run_id != run.id or gate.gate_type != PLAN_ITEM_GATE_TYPE
                    or gate.success_criterion_key != gate_key
                ):
                    raise invalid(f"task {task_id} is already bound to another gate")

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
        required_evidence = self._json_object_or_empty(item.required_evidence)
        run = await db.get(OrchestrationRun, run_id)
        goal = await db.get(OrchestrationGoal, run.goal_id) if run is not None else None
        if goal is not None and goal.goal_type == "outcome":
            required_evidence["required_source_types"] = ["task", "verification"]
            required_evidence["min_count"] = 2
            required_evidence["requires_independent_agent"] = True
        required_evidence["plan_item_id"] = item.id
        required_evidence["success_criterion_keys"] = list(item.success_criterion_keys)
        return await self._ensure_keyed_gate(db, run_id, PLAN_ITEM_GATE_TYPE, gate_key, required_evidence)

    async def _ensure_keyed_gate(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        gate_type: str,
        gate_key: str,
        required_evidence: dict[str, Any],
    ) -> OrchestrationGate:
        """Get-or-create the deterministic gate for (run, type, key); concurrent creators converge."""
        lookup = (
            select(OrchestrationGate)
            .where(
                OrchestrationGate.run_id == run_id,
                OrchestrationGate.success_criterion_key == gate_key,
                OrchestrationGate.gate_type == gate_type,
            )
            .order_by(OrchestrationGate.created_at.asc(), OrchestrationGate.id.asc())
            .limit(1)
        )
        gate = (await db.execute(lookup)).scalar_one_or_none()
        if gate is not None:
            return gate

        gate = OrchestrationGate(
            id=uuid.uuid5(uuid.NAMESPACE_URL, f"rally:orchestration:gate:{run_id}:{gate_type}:{gate_key}"),
            run_id=run_id,
            success_criterion_key=gate_key,
            gate_type=gate_type,
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
            gate = (await db.execute(lookup)).scalar_one_or_none()
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
                "title": item.title,
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
                "orchestrator_context": delegable_context(orchestrator_context if orchestrator_context is not None else (goal.orchestrator_context if goal else None)),
                "priority": item.priority,
                "artifact_ids": [str(artifact_id) for artifact_id in item.artifact_ids],
                "why_runnable": item.why_runnable,
                "depends_on": self._expanded_predecessor_task_ids(run, item),
            }
        )
        await self._validate_delegation_targets(db, run_id, request)
        return request

    def _expanded_predecessor_task_ids(self, run: OrchestrationRun | None, item: OrchestrationPlanItem) -> list[str]:
        """Task ids of the item's already-expanded predecessors (released only after their gates accepted)."""
        expanded = self._json_object_or_empty(run.plan_state if run else None).get("expanded_items")
        task_by_item = {
            entry.get("plan_item_id"): entry.get("task_id")
            for entry in expanded if isinstance(entry, Mapping)
        } if isinstance(expanded, list) else {}
        # ponytail: Roadmap item keys are not in expanded_items, so Roadmap task items get no depends_on here.
        return [
            task_by_item[key] for dep in item.depends_on
            if (key := dep.removeprefix("plan_item:")) in task_by_item and task_by_item[key]
        ]

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
        from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService

        parent = await OrchestrationBudgetService().cli_authority_owner(
            db, goal, roadmap.get("roadmap_version_id")) or goal
        child = parent.id != goal.id
        version_id = goal.roadmap_version_id if child else roadmap.get("roadmap_version_id")
        item_key = goal.roadmap_item_key if child else roadmap.get("roadmap_item_key")
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
        caps = self._json_object_or_empty((run.budget_state or {}).get("caps")) if child else self._json_object_or_empty((goal.budget or {}).get("caps"))
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
    def _current_plan_fingerprint(run: OrchestrationRun) -> str | None:
        state = OrchestrationService._json_object_or_empty(run.plan_state)
        return state.get("accepted_plan_fingerprint") or OrchestrationService._json_object_or_empty(
            state.get("accepted_plan_snapshot")
        ).get("fingerprint")

    @staticmethod
    def _validate_bound_plan_item(
        task: Task, run: OrchestrationRun, item: Any, metadata_view: dict | None = None,
    ) -> None:
        metadata = OrchestrationService._json_object_or_empty(task.metadata_ if metadata_view is None else metadata_view)
        orchestration = OrchestrationService._json_object_or_empty(metadata.get("orchestration"))
        previous = OrchestrationService._json_object_or_empty(metadata.get("orchestration_plan_item"))
        key = getattr(item, "item_key", None) or item.id
        candidate = item.model_dump(mode="json")
        if hasattr(item, "unit_type"):
            candidate["id"] = candidate.pop("item_key")
            candidate = {k: candidate.get(k, v) for k, v in previous.items()
                         if not k.startswith("accepted_plan_")}
        frozen = lambda value: {k: v for k, v in value.items() if k not in {
            "priority", "artifact_ids", "accepted_plan_artifact_id", "accepted_plan_fingerprint",
        }}
        if previous and (orchestration.get("plan_item_id") != key or frozen(previous) != frozen(candidate)):
            raise HTTPException(status_code=409, detail=f"adoption_invalid: item {key}: bound work contract changed")
        if previous and metadata.get("orchestration_plan_contract", metadata.get("orchestration_contract")) != metadata.get("orchestration_contract"):
            raise HTTPException(status_code=409, detail=f"adoption_invalid: item {key}: actual work contract changed")
        if previous and (orchestration.get("run_id") != str(run.id)
                         or orchestration.get("goal_id") != str(run.goal_id)):
            raise HTTPException(status_code=409, detail=f"adoption_invalid: item {key}: task binding changed")

    @staticmethod
    def _expand_plan_item_idempotency_key(run_id: uuid.UUID, plan_item_id: str, fingerprint: str | None = None) -> str:
        key = f"run:{run_id}:kind:expand_plan_item:plan_item:{plan_item_id}"
        return f"{key}:version:{fingerprint}" if fingerprint else key

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
        raw_items = [
            raw for raw in OrchestrationService._json_object_or_empty(artifact.metadata_).get("plan_items") or []
            if isinstance(raw, dict)
        ]
        plan_index = next(
            (i for i, raw in enumerate(raw_items) if item.id in (raw.get("id"), raw.get("item_key"))), None
        )
        plan_unlocks = sum(
            1 for raw in raw_items
            if item.id in {str(dep).removeprefix("plan_item:") for dep in raw.get("depends_on") or []}
        )
        metadata["orchestration"] = {
            **orchestration,
            **({"plan_index": plan_index} if plan_index is not None else {}),
            "plan_unlocks": plan_unlocks,
            "plan_item_id": item.id,
            "plan_item_gate_id": str(gate.id),
            "success_criterion_keys": list(item.success_criterion_keys),
            "expand_action_id": str(action.id),
        }
        metadata.setdefault("orchestration_plan_contract", deepcopy(metadata.get("orchestration_contract")))
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
        adopted: bool = False,
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
            "accepted_plan_fingerprint": OrchestrationService._current_plan_fingerprint(run),
            "task_id": str(task.id),
            "gate_id": str(gate.id),
            **({"adopted": True} if adopted else {}),
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
        adopted: bool = False,
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
                **({"adopted": True} if adopted else {}),
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
        self, db: AsyncSession, goal_id: uuid.UUID, allow_heal: bool = True,
        admission: BaselineRepairAdmissionV1 | None = None,
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
        return await self._baseline_readiness_reason(db, goal_id, allow_heal=allow_heal, admission=admission) is None

    @staticmethod
    def baseline_gate_fingerprint(process: OrchestrationProcessRun) -> str:
        """Identity of one blocked baseline process run, as named by a baseline-repair admission."""
        # Outputs (gates, decisions) carry the failure itself, so a changed failure is a different gate.
        value = {"id": str(process.id), "process_type": process.process_type, "status": process.status,
                 "input_fingerprint": (process.input_snapshot or {}).get("fingerprint"), "outputs": process.outputs}
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

    async def _baseline_readiness_reason(
        self, db: AsyncSession, goal_id: uuid.UUID, allow_heal: bool = True,
        admission: BaselineRepairAdmissionV1 | None = None,
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
                if (admission is not None and process_type == admission.process_type and current is not None
                        and current.status not in ("completed", "skipped")):
                    # Only a blocked (waiting_decision) run is admissible; a running/pending one is not a failure.
                    # The one approved remediation admits exactly this failed run; any change rejects it.
                    if (current.status != "waiting_decision" or current.id != admission.gate_id
                            or self.baseline_gate_fingerprint(current) != admission.gate_fingerprint):
                        return f"'{process_type}' changed since the approved repair admission"
                    continue
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
            _phase_run = await self.get_active_run_for_goal(db, goal.project_id, goal.id)
            if manager_selection_current is not None and (
                _phase_run is None or _phase_run.phase in ("baseline", "ready")
            ):
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
                # Past Start, reruns are user-initiated only: stale review/hierarchy
                # no longer blocks delegation (suggestion still surfaces).
                if active_run is not None and active_run.phase in ("baseline", "ready"):
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
                # Past Start, reruns are user-initiated only: stale review/hierarchy
                # no longer blocks delegation (suggestion still surfaces).
                if active_run is not None and active_run.phase in ("baseline", "ready"):
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
        self, db: AsyncSession, goal_id: uuid.UUID, allow_heal: bool = True,
        admission: BaselineRepairAdmissionV1 | None = None,
    ) -> None:
        """Shared gate (Phase 5 Finding 1, extended through Phase 8):
        every place that can move a run's plan_state forward or create a
        delegation/finish action must call this first. Goal definition,
        manager selection, agent definition review, and team hierarchy must all
        be terminal.

        If allow_heal is False, reports stale answer without writing (fix #15).
        """
        if not await self._baseline_processes_ready_for_goal(db, goal_id, allow_heal=allow_heal, admission=admission):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Goal definition, manager selection, agent definition review, "
                    "and team hierarchy must complete before planning or delegation"
                ),
            )

    async def _ensure_plan_revision_allowed(self, db: AsyncSession, run: OrchestrationRun) -> None:
        self._require_roadmap_child_budget(run)
        await self._ensure_baseline_processes_ready(db, run.goal_id)
        pending = self._json_object_or_empty(self._json_object_or_empty(run.plan_state).get("pending_replan"))
        if pending:
            if not pending.get("ingestion_error"):
                raise HTTPException(status_code=409, detail="Pending Roadmap replan can only be revised after its output is rejected")
            return
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

    @staticmethod
    def _require_roadmap_child_budget(run: OrchestrationRun) -> None:
        if OrchestrationService._json_object_or_empty(run.plan_state).get("budget_required"):
            raise HTTPException(status_code=409, detail=BUDGET_REQUIRED_CODE)

    async def _ensure_plan_request_allowed(self, db: AsyncSession, run: OrchestrationRun, check_readiness: bool = True) -> None:
        self._require_roadmap_child_budget(run)
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
        pre_plan_lines: Sequence[str] = (),
    ) -> OrchestrationDelegationContract:
        from huddleroom.services.orchestration_goal_definition import goal_objective_with_clarifications

        from huddleroom.services.orchestration_team_hierarchy import plan_mode

        deliverable = {
            "roadmap": "Roadmap plan: a finite graph of distinct outcomes, defined progressively.",
            "delivery": "Delivery plan: one bounded plan coordinating every work function this outcome needs.",
        }.get(
            plan_mode(goal, run),
            "Implementation plan: a leaf technical plan for this outcome.",
        )
        plan_state = self._json_object_or_empty(run.plan_state)
        extra_inputs = []
        if pre_plan_lines:
            extra_inputs.append(
                "Existing run work — adopt each via adopt_task_id or say in summary why it is not used:\n"
                + "\n".join(pre_plan_lines)
            )
        if plan_state.get("status") in {"revision_requested", "revision_required"}:
            extra_inputs.extend(
                f"Plan revision request: {item['revision_request']}"
                for item in plan_state.get("revision_requests") or []
                if isinstance(item, Mapping) and item.get("revision_request")
            )
            if plan_state.get("revision_reason"):
                extra_inputs.append(f"Plan revision reason: {plan_state['revision_reason']}")
        eligible = self._json_object_or_empty(self._json_object_or_empty(goal.orchestrator_context).get("eligible_evidence"))
        artifact_ids = [str(a) for a in eligible.get("artifact_ids", [])]
        if eligible:
            evidence_ids = [str(e) for e in eligible.get("evidence_ids", [])]
            extra_inputs.append(
                "Eligible parent evidence (references only; re-verify before relying on them for any final "
                f"criterion): artifacts {', '.join(artifact_ids) or 'none'}; evidence {', '.join(evidence_ids) or 'none'}."
            )
        retained = [e for e in plan_state.get("retained_evidence_manifest") or [] if isinstance(e, Mapping)]
        if retained:
            extra_inputs.append(
                "Retained evidence from superseded run (re-verify; reuse only via retained gates):\n"
                + "\n".join(
                    f"{e.get('source_gate_id')} | task {e.get('source_task_id')} | "
                    f"artifact {e.get('artifact_id') or 'none'} | "
                    f"{'re-verify' if e.get('revalidation_required') else 'contract unchanged'}"
                    for e in retained
                )
            )
            artifact_ids += [str(e["artifact_id"]) for e in retained if e.get("artifact_id")]
            artifact_ids = list(dict.fromkeys(artifact_ids))
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
                "Optional specialist inputs (only when a named risk or missing decision benefits; inputs to this "
                "plan, never co-authors): product strategist — sequencing; tester — qualification and evidence; "
                "UI/UX designer — interaction and accessibility.",
                *extra_inputs,
            ],
            deliverable=deliverable,
            forbidden_work=[
                "Do not implement the plan.",
                "Do not edit project artifacts.",
                "Do not create tasks, meetings, graphs, rules, hooks, or automations.",
            ],
            success_evidence=[
                "One PlanningEnvelope JSON object as your final output; HuddleRoom registers the plan Artifact.",
                "Plan items are independently actionable and verifiable.",
                "You are the sole plan author: one plan that synthesizes any specialist inputs.",
            ],
            budget=self._json_object_or_empty(goal.budget),
            report_schema=planning_envelope_report_schema(plan_kind_for(goal, run)),
            expected_result=PLANNING_ENVELOPE,
            artifact_ids=[uuid.UUID(a) for a in artifact_ids],
            parent_task_id=None,
            orchestrator_context=delegable_context(goal.orchestrator_context),
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
            criterion_evidence_manifest or self._criterion_evidence_manifest(gates, evidence, plan_items=accepted_plan_items(run.plan_state))
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
        plan_items: list[dict] | None = None,
    ) -> dict[str, list[str]]:
        """Map each criterion to accepted verifier evidence from plan-item gates only.

        With accepted plan items, a criterion is listed only when the shared acceptance predicate holds."""
        manifest: dict[str, list[str]] = {}
        non_final = non_final_gate

        if goal is not None and goal.goal_type == "roadmap":
            for gate in gates:
                if non_final(gate) or (
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
        else:
            for gate in gates:
                if gate.gate_type != PLAN_ITEM_GATE_TYPE or non_final(gate):
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
        if plan_items:
            manifest = {k: v for k, v in manifest.items() if criterion_state(k, plan_items, gates) == "accepted"}
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
        # Ad-hoc audit gates are advisory: only an accepted one joins closeout and the summary manifest.
        # Retained (type-correction) gates never do: integration reads them via item_gate_satisfied and they carry no evidence.
        gates = [gate for gate in gates if gate.gate_type != RETAINED_GATE_TYPE
                 and (gate.gate_type not in NON_BLOCKING_GATE_TYPES or gate.status == "accepted")]
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
        from huddleroom.services.orchestration_progress_view import unreleased_plan_item_ids
        unreleased_items = unreleased_plan_item_ids(run)
        if unreleased_items:
            raise HTTPException(status_code=409, detail="Accepted plan items are unreleased: " + ", ".join(unreleased_items))
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
        unresolved_commitments = [c for c in await meeting_commitments(db, run) if not c["fulfilled"]]
        if unresolved_commitments:
            raise HTTPException(status_code=409, detail="Unresolved meeting commitments: " + ", ".join(
                f"{c['id']} ({c['state']}: {c['reason']})" for c in unresolved_commitments
            ))
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
            "unresolved_meeting_commitments": [],
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
        manifest = self._criterion_evidence_manifest(
            gates, evidence, goal, str(version.id) if version else None, accepted_plan_items(run.plan_state)
        )
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
            await SessionService().cancel(db, active_session.id, sync_task=False)

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
                "plan_author_agent_id",
                "plan_artifact_id",
                "plan_output_hash",
                "planning_version",
                "revision_reason",
            }
        }
        current_version = int(plan_state.get("planning_version") or 0)
        recorded_version = OrchestrationService._json_object_or_empty(
            OrchestrationService._json_object_or_empty(task.metadata_).get("orchestration_plan")
        ).get("planning_version")
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
            "planning_version": (
                current_version or recorded_version or 1
                if plan_state.get("request_action_id") == str(action.id)
                else recorded_version or current_version + 1
            ),
        }
        author_id = task.assigned_to or (action.request or {}).get("agent_id")
        if author_id:
            run.plan_state["plan_author_agent_id"] = str(author_id)

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
        planning_version = int(plan_state.get("planning_version") or 0)
        if not any(
            isinstance(item, Mapping) and item.get("action_id") == action_id
            for item in revision_requests
        ):
            planning_version += 1
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
            "planning_version": planning_version,
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
    def _delegation_task_title(work_function: str, deliverable: str, title: str = "") -> str:
        if title.strip():
            title = title.strip()
            short = shorten(title, width=80, placeholder="...")
            return short if short != "..." else title[:77] + "..."
        title = f"{work_function.replace('_', ' ').title()}: {deliverable.strip()}"
        return shorten(title, width=80, placeholder="...")

    @staticmethod
    def _render_criterion_input(goal: OrchestrationGoal, item: str) -> str:
        """Append the goal's criterion description to a ``criterion:<key>`` input."""
        if item.startswith("criterion:"):
            criteria = getattr(goal, "success_criteria", None)
            for criterion in criteria if isinstance(criteria, list) else []:
                if isinstance(criterion, dict) and criterion.get("key") == item[len("criterion:"):]:
                    description = criterion.get("description")
                    if isinstance(description, str) and description.strip():
                        return f"{item} — {description.strip()}"
        return item

    @staticmethod
    def _delegation_task_description(goal: OrchestrationGoal, contract: OrchestrationDelegationContract) -> str:
        from huddleroom.services.orchestration_goal_definition import goal_objective_with_clarifications

        sections = [
            f"Objective: {goal_objective_with_clarifications(goal)}",
            f"Scope: {contract.scope}",
            f"Deliverable: {contract.deliverable}",
        ]
        if contract.why_runnable:
            sections.append(f"Why runnable: {contract.why_runnable}")
        if contract.decision_use:
            sections.append(f"Decision use: {contract.decision_use}")
        if contract.inputs:
            sections.append("Inputs:\n" + "\n".join(
                f"- {OrchestrationService._render_criterion_input(goal, item)}" for item in contract.inputs
            ))
        if contract.depends_on:
            sections.append(
                "Dependencies (accepted predecessor task IDs; use their reports as inputs):\n"
                + "\n".join(f"- {item}" for item in contract.depends_on)
            )
        if contract.artifact_ids:
            sections.append(
                "Artifacts (existing Artifact IDs to read):\n" + "\n".join(f"- {item}" for item in contract.artifact_ids)
            )
        if contract.forbidden_work:
            sections.append("Forbidden work:\n" + "\n".join(f"- {item}" for item in contract.forbidden_work))
        if contract.success_evidence:
            sections.append("Success evidence:\n" + "\n".join(f"- {item}" for item in contract.success_evidence))
        sections.append(OrchestrationService._authorization_scope_text(contract.authorization_scope))
        roadmap = OrchestrationService._json_object_or_empty(contract.orchestrator_context).get("roadmap")
        if isinstance(roadmap, dict):
            sections.append("Roadmap execution policy:\n" + json.dumps(roadmap, sort_keys=True))
        report_schema = contract.report_schema or {}
        strict = True
        kind = expected_result_kind({"expected_result": contract.expected_result, "report_schema": report_schema}, False)
        schema = OrchestrationService._result_schema(goal, kind, report_schema)
        if kind == PLANNING_ENVELOPE:
            rules = [
                "Planning rules:",
                "Adoption: use only a supplied existing task ID from the same run, in one item only, with a "
                "compatible work function and assignment. Every adopted item must have depends_on: []. "
                "An adopted preliminary audit cannot become a future final review; create a separate final review item.",
                "Dependencies: name only immediate necessary predecessors whose gates must be accepted. "
                "Put artifact and report references in inputs/artifact_ids, not redundant transitive scheduling edges. "
                "Pattern: adopted preliminary audit (no dependencies) → new implementation → new final review; "
                "the final review depends only on implementation and reads the audit as input.",
                "Do not implement the plan: this restriction applies to your planning assignment. Future child goals "
                "own implementation plus verification of their delivered outcome under inherited parent constraints. "
                "Every required candidate needs a named producer in the plan; do not make delivery children read_only "
                "unless the requested outcome itself is read-only, and never relax actual parent restrictions.",
                "The type-labelled field inventory below describes allowed fields, not values to copy or a demand "
                "to populate every field. Omit optional/default fields when unused.",
            ]
            task_example = {
                "id": "audit", "work_function": "investigation", "scope": "Inspect the current behavior.",
                "deliverable": "Audit findings and evidence references.", "depends_on": [],
                "why_runnable": "Initial discovery needs no predecessor.",
            }
            if schema["plan_kind"] == "roadmap":
                from huddleroom.services.orchestration_budget_service import parent_caps

                caps = parent_caps(goal)
                rules.extend([
                    "Roadmap adoption: unit_type: task and mutates_shared_state: false are required for adopted work.",
                    "Task variant: require work_function, scope, deliverable; objective and goal_type must be null "
                    "or absent; success_criteria, constraints, allocation must be empty or absent. Goal variant: "
                    "require objective, success_criteria and complete nonzero allocation; work_function, scope, "
                    "deliverable, agent_id must be null or absent. Shared inputs, evidence and priority are allowed.",
                    "Staging: declare mutates_shared_state truthfully for the future work. A nonnull staging_boundary "
                    "has {type: git_branch|git_worktree|transaction|draft, identifier: nonblank, reversible: true}. "
                    "The local workspace adapter resolves only an existing active branch or registered worktree; "
                    "schema validity does not establish adapter support. Never invent a path, branch, transaction "
                    "or draft. Null is legal and requires the existing mutation-authority wait before execution.",
                    "Measured parent caps: " + json.dumps(caps, sort_keys=True) + ". Child allocation must cover "
                    "exactly every parent dimension, use finite nonnegative numbers with at least one positive "
                    "amount, and summed child allocations must fit each parent cap and remaining capacity. "
                    "Explicit measurable parent caps are required before creating child goals. Missing caps mean "
                    "Budget required; do not invent limits or propose zero-only reservations. Zero/exhausted caps "
                    "require the existing budget-exhaustion path.",
                ])
                task_example.pop("id")
                task_example.update(item_key="audit", unit_type="task", title="Audit current behavior",
                                    mutates_shared_state=False)
                if caps and any(Decimal(amount) > 0 for amount in caps.values()):
                    child_example = {
                        "item_key": "delivery", "unit_type": "goal", "title": "Deliver bounded outcome",
                        "objective": "Implement the bounded outcome and independently verify its delivered behavior.",
                        "success_criteria": [{"key": "delivered", "description": "Implemented behavior has verification evidence."}],
                        "allocation": {key: format(Decimal(amount) / 2, "f") for key, amount in caps.items()},
                        "mutates_shared_state": True, "staging_boundary": None, "depends_on": [],
                        "why_runnable": "Delivery starts after normal release and mutation authority are granted.",
                    }
                    rules.append("Valid goal example: " + json.dumps({
                        "schema_version": 1, "plan_kind": "roadmap", "summary": "Bounded delivery with authority wait.",
                        "plan_items": [child_example],
                    }, sort_keys=True))
            rules.append("Valid task example: " + json.dumps({
                "schema_version": 1, "plan_kind": schema["plan_kind"], "summary": "Initial evidence collection.",
                "plan_items": [task_example],
            }, sort_keys=True))
            sections.append("\n".join(rules))
        elif kind != WORK_REPORT:
            strict = False
        field_instruction = (
            "include required fields and omit unused optional/default fields; no extra keys (values show types). "
            if kind == PLANNING_ENVELOPE
            else "no missing or extra keys (values show types). "
        )
        output = (
            "Required output:\nWhen done, return exactly one JSON object as your final message: no prose, "
            f"no Markdown fences, {field_instruction}"
            "Read-only work reports its findings in this JSON.\n" + json.dumps(schema, indent=2)
        )
        if strict:
            output += (
                "\nNever invent task, agent, or Artifact UUIDs; plan item ids/item_keys and their depends_on "
                "references are yours to define; do not register Artifacts for this output."
                if kind == PLANNING_ENVELOPE
                else "\nUse only IDs given in this task; never invent IDs, and do not register Artifacts for this output."
            )
        sections.append(output)
        return "\n\n".join(sections)

    @staticmethod
    def _result_schema(goal: OrchestrationGoal, kind: str, report_schema: Mapping[str, Any]) -> dict:
        """The exact schema a result of this kind is validated against, as shown in its brief."""
        if kind == PLANNING_ENVELOPE:
            plan_kind = report_schema.get("plan_kind")
            return planning_envelope_report_schema(plan_kind if plan_kind in PLAN_KINDS else plan_kind_for(goal))
        return CANONICAL_WORK_REPORT_SCHEMA if kind == WORK_REPORT else dict(report_schema)

    @staticmethod
    def _authorization_scope_text(scope) -> str:
        if scope is None:
            return (
                "Authorization scope: not separately declared; goal constraints and owner-approval rules apply. "
                "Do not spend, write to external systems, send live messages, publish, or run destructive "
                "operations unless the scope explicitly authorizes it."
            )
        spend, writes, sends = scope.model_spend, scope.external_writes, scope.live_sends
        remaining = ", ".join(f"{k}={v}" for k, v in spend.remaining_by_dimension.items())
        return "\n".join([
            "Authorization scope:",
            "- Model spend: " + (f"allowed (remaining: {remaining})" if spend.allowed and remaining else "allowed" if spend.allowed else "not allowed"),
            "- External writes: " + (f"allowed to {', '.join(writes.targets)}" if writes.allowed and writes.targets else "allowed" if writes.allowed else "not allowed"),
            "- Live sends: " + (f"allowed on {', '.join(sends.channels)}" if sends.allowed and sends.channels else "allowed" if sends.allowed else "not allowed"),
            f"- Workspace mutation: {scope.workspace_mutation}",
            "- Tool constraints: " + (", ".join(scope.tool_constraints) or "none"),
        ])

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

    def _canonical_start_graph_request(self, request: Any) -> dict[str, Any]:
        request_obj = self._json_object_or_empty(request)
        subject_type = self._required_string(request_obj.get("subject_type"), "subject_type")
        if subject_type not in {"task", "artifact"}:
            raise HTTPException(status_code=400, detail="subject_type must be 'task' or 'artifact'")
        return {
            "action_type": "start_graph",
            "graph_id": str(self._required_uuid(request_obj.get("graph_id"), "graph_id")),
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

    async def _graph_subject(
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
            raise HTTPException(status_code=404, detail="Graph subject artifact not found")
        if artifact.linked_task_id is None:
            raise HTTPException(
                status_code=409,
                detail="Graph subject artifact must link to an orchestration task",
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

    async def _ingest_planning_output(
        self, db: AsyncSession, run: OrchestrationRun, task: Task, expected_session_id: uuid.UUID | None,
    ) -> None:
        """Register the winning planner output as the run's plan Artifact, or require a plan revision."""
        metadata = self._json_object_or_empty(task.metadata_)
        orchestration = self._json_object_or_empty(metadata.get("orchestration"))
        if (
            task.status != "done"
            or orchestration.get("work_function") != PLAN_WORK_FUNCTION
            or orchestration.get("run_id") != str(run.id)
        ):
            return
        key = f"run:{run.id}:kind:report_consumed:task:{task.id}"
        marker = await self._existing_action_for_key(db, run.id, key)
        if marker is not None and marker.status == "completed":
            return
        if expected_session_id is None:
            if marker is not None:
                expected_session_id = self._optional_uuid(
                    self._json_object_or_empty(marker.request).get("session_id"), "session_id"
                )
            else:
                sessions = list(await db.scalars(select(Session.id).where(
                    Session.task_id == task.id, Session.status == "completed",
                )))
                if len(sessions) > 1:
                    return
                expected_session_id = sessions[0] if sessions else None
        try:
            session_id, output = await self.consume_terminal_task_output(db, run, task, expected_session_id)
        except HTTPException as exc:
            if exc.status_code == 409:
                return  # a nonwinning session never becomes the plan Artifact
            raise
        marker = await self._existing_action_for_key(db, run.id, key)
        await self._land_plan_output(db, run, task, marker, task, session_id, output)

    async def _land_plan_output(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        task: Task,
        marker: OrchestrationAction,
        source: Task,
        session_id: uuid.UUID | None,
        output: str | None,
    ) -> None:
        """Ingest ``output`` as ``task``'s plan; ``source`` produced it (the planning task or its linked correction).

        Correctable rejections start an output-only correction; stale and provenance rejections keep the revision path."""
        metadata = self._json_object_or_empty(task.metadata_)
        plan_state = self._json_object_or_empty(run.plan_state)
        version = self._json_object_or_empty(metadata.get("orchestration_plan")).get("planning_version")
        if version is None:
            version = plan_state.get("planning_version")
        goal = await db.get(OrchestrationGoal, run.goal_id)
        try:
            plan_text, how = normalize_result_output(output)
        except ValueError:
            plan_text, how = output, None  # parse_envelope reports the precise rejection
        if how and source is task:  # a correction never overwrites the original's own record
            marker.request = {**self._json_object_or_empty(marker.request), "normalized": how}
        try:
            artifact = await ingest(db, goal, run, task, session_id, version, plan_text)
        except PlanIngestionError as exc:
            # ponytail: legacy self-registering planners keep working; drop this check once they are gone.
            manual_plan = await db.scalar(select(Artifact.id).where(
                Artifact.linked_task_id == task.id,
                Artifact.artifact_type == "plan",
                Artifact.id != plan_artifact_id(task.id, version),
            ).limit(1))
            # Planner prose with no JSON object never emitted an envelope: a missing plan, which a revision regenerates.
            # ponytail: "has a brace" is the envelope test, so brace-bearing prose goes to correction; parse if that bites.
            correct = exc.correctable and manual_plan is None and (source is not task or "{" in (output or ""))
            if source is task:
                marker.request = {**self._json_object_or_empty(marker.request), "plan_ingestion_error": exc.reason}
            elif not correct:  # a linked correction failing a fence ends the correction; nothing re-runs it
                self._set_result_correction_status(marker, "rejected", reason=exc.reason)
            if exc.code == BUDGET_REQUIRED_CODE:
                details = {
                    "code": exc.code, "title": "Budget required", "message": BUDGET_REQUIRED_MESSAGE,
                    "raw_output": output, "diagnostics": exc.diagnostics,
                }
                if source is task:
                    marker.request = {**marker.request, "plan_ingestion_error_code": exc.code,
                                      "plan_validation_diagnostics": exc.diagnostics, "rejected_output": output}
                run.plan_state = {**plan_state, "status": "budget_required", "budget_required": details}
                self._mark_run_blocked(goal, run)
                self._upsert_active_blocker(run, {
                    "kind": BUDGET_REQUIRED_CODE, "reason": f"Budget required: {BUDGET_REQUIRED_MESSAGE}",
                    "details": details,
                })
                ask = await self.execute_ask_human_action(
                    db, run.id, {
                        "action_type": "ask_human", "question": BUDGET_REQUIRED_MESSAGE,
                        "subject": BUDGET_REQUIRED_CODE, "contract_version": "explicit_parent_caps:v1",
                        "reason": json.dumps(details, ensure_ascii=False),
                        "options": [{"key": "acknowledge", "label": "Acknowledge"}],
                    }, f"run:{run.id}:kind:ask_human:{BUDGET_REQUIRED_CODE}",
                )
                question = await db.get(OrchestrationAuthorityDecision, ask.target_id)
                question.title = "Budget required"
                await self._mark_action_completed(db, marker, "task", task.id)
                return
            if correct:
                if source is task:
                    marker.request = {**marker.request, "rejected_output": output}
                await self._mark_action_completed(db, marker, "task", task.id)
                await self._request_result_correction(
                    db, run, source, session_id, PLANNING_ENVELOPE, exc.diagnostics or [exc.reason],
                )
                if self._json_object_or_empty(marker.request).get("result_correction"):
                    return
                # No correction applies to this contract (legacy or exempt): the revision path below decides.
            if not exc.stale and manual_plan is None:
                self._plan_output_rejected(run, f"planning_output_rejected: {exc.reason}")
            await self._mark_action_completed(db, marker, "task", task.id)
            return
        await self._mark_action_completed(db, marker, "artifact", artifact.id)
        if source is not task:
            self._set_result_correction_status(marker, "corrected")

    def _plan_output_rejected(self, run: OrchestrationRun, reason: str) -> None:
        plan_state = self._json_object_or_empty(run.plan_state)
        pending = self._json_object_or_empty(plan_state.get("pending_replan"))
        if pending:
            # A pending Roadmap replan sits on an accepted plan; revision_required would strand it,
            # so surface the rejection to the decision context instead.
            run.plan_state = {**plan_state, "pending_replan": {**pending, "ingestion_error": reason}}
        else:
            self._set_plan_revision_required(run, reason)

    def _plan_correction_fallback(self, run: OrchestrationRun, ledger: dict[str, Any]) -> None:
        """A planning correction that stopped without a plan falls back to the plan revision path, once per cycle."""
        status = ledger.get("status")
        if (ledger.get("kind") != PLANNING_ENVELOPE or status not in {"exhausted", "unavailable", "owner_declined"}
                or ledger.get("revision_fallback")):
            return
        ledger["revision_fallback"] = True
        errors = next((item["errors"] for item in reversed(ledger.get("attempts") or []) if item.get("errors")), [])
        self._plan_output_rejected(
            run, f"planning_output_rejected: result correction {status}: " + "; ".join(errors or [ledger.get("detail") or ""]),
        )

    def _set_result_correction_status(self, marker: OrchestrationAction, status: str, **fields: Any) -> None:
        request = self._json_object_or_empty(marker.request)
        ledger = self._json_object_or_empty(request.get("result_correction"))
        marker.request = {**request, "result_correction": {**ledger, "status": status, **fields}}

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
            if validation.report is None:
                await self._request_result_correction(
                    db, run, task, session_id, WORK_REPORT, list(validation.errors),
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
            **({"normalized": validation.normalized} if validation.normalized else {}),
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
            await self._request_result_correction(db, run, task, session_id, WORK_REPORT, list(validation.errors))
        return session_id, validation.report

    async def _request_result_correction(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        task: Task,
        session_id: uuid.UUID | None,
        kind: str,
        errors: Sequence[str],
    ) -> None:
        """Bounded, output-only correction of a result that failed validation, tracked on the original's ledger.

        ``task`` is the original task or its current correction task; ``kind`` names the validator that failed."""
        metadata = self._json_object_or_empty(task.metadata_)
        contract = self._json_object_or_empty(metadata.get("orchestration_contract"))
        context = self._json_object_or_empty(contract.get("orchestrator_context"))
        if self._json_object_or_empty(context.get("continuous")).get("discovery_run_id") == str(run.id):
            return
        if kind not in CORRECTABLE_KINDS or expected_result_kind(contract, task_gate_id(task, run.id) is not None) != kind:
            return
        parent_task_id = self._optional_uuid(
            metadata.get("result_correction_parent_task_id"), "result_correction_parent_task_id"
        ) or task.id
        parent = task if parent_task_id == task.id else await db.get(Task, parent_task_id)
        goal = await db.get(OrchestrationGoal, run.goal_id)
        if parent is None or goal is None:
            return
        key = f"run:{run.id}:kind:report_consumed:task:{parent.id}"
        marker = await self._existing_action_for_key(db, run.id, key)
        if marker is None:
            if parent is not task:
                return
            marker = await self.reserve_action(db, run.id, key, "report_consumed", {
                "task_id": str(task.id), "session_id": str(session_id) if session_id else None,
            })
            await self._mark_action_completed(db, marker, "task", task.id)
        ledger = dict(self._json_object_or_empty(self._json_object_or_empty(marker.request).get("result_correction")))
        if parent is not task and ledger.get("correction_task_id") != str(task.id):
            return  # an unlinked or superseded correction never steers the original's ledger
        version = self._result_correction_version(goal, run)
        ledger = ledger or {
            "kind": kind, "cycle": 1, "version": version, "attempts": [],
            "correction_task_id": None, "correction_task_ids": [], "status": "pending",
        }
        try:
            await self._advance_result_correction(db, goal, run, parent, marker, ledger, task, session_id, errors, version)
            self._plan_correction_fallback(run, ledger)
        finally:
            marker.request = {**self._json_object_or_empty(marker.request), "result_correction": ledger}

    async def _advance_result_correction(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        parent: Task,
        marker: OrchestrationAction,
        ledger: dict[str, Any],
        task: Task,
        session_id: uuid.UUID | None,
        errors: Sequence[str],
        version: dict[str, Any],
    ) -> None:
        stale = await self._result_correction_stale(db, run)
        if stale is not None:
            ledger["stale"] = stale  # neither dispatched nor consumed onto a paused/superseded run
            return
        ledger.pop("stale", None)
        if ledger.get("status") in {"corrected", "owner_declined", "superseded", "rejected"}:
            return
        if ledger["kind"] == PLANNING_ENVELOPE and self._json_object_or_empty(ledger.get("version")).get(
            "planning_version", version["planning_version"]
        ) != version["planning_version"]:
            # A newer planning version replaces this plan; its revision carries on, not a correction of the old one.
            await self._new_result_correction_cycle(db, run, parent, ledger, version)
            ledger["status"] = "superseded"
            return
        if ledger.get("version", version) != version:
            # A new owner contract or plan version resets the count.
            await self._new_result_correction_cycle(db, run, parent, ledger, version)
        if ledger.get("status") in {"exhausted", "unavailable"}:
            decision_id = self._optional_uuid(ledger.get("decision_id"), "decision_id")
            decision = await db.get(OrchestrationAuthorityDecision, decision_id) if decision_id else None
            if decision is None or decision.status == "pending":
                return  # ticks and replays never re-arm; only an owner answer does
            if decision.status != "answered" or decision.selected_option == "dismiss":
                ledger["status"] = "owner_declined"
                return
            await self._new_result_correction_cycle(db, run, parent, ledger, version)
        attempts = list(ledger.get("attempts") or [])
        if ledger.get("status") == "correcting":
            correction = await db.get(Task, uuid.UUID(ledger["correction_task_id"]))
            if correction is not None and correction.status in {"failed", "cancelled"} and not any(
                item.get("task_id") == str(correction.id) for item in attempts
            ):  # a failed correction run is a failed attempt with no output to carry
                attempts.append({
                    "task_id": str(correction.id), "session_id": None, "task_failed": True,
                    "fingerprint": f"task_{correction.status}:{correction.id}",
                    "errors": [f"correction task {correction.status}"],
                })
                ledger.update(attempts=attempts, status="exhausted" if len(attempts) > 2 else "pending")
        identity = {"task_id": str(task.id), "session_id": str(session_id) if session_id else None}
        if not any(item.get("task_id") == identity["task_id"] and item.get("session_id") == identity["session_id"]
                   for item in attempts):
            output, _reason = await self._stored_result_output(db, task, session_id)
            try:
                normalized = normalize_result_output(output or "")[0]
            except ValueError:
                normalized = output or ""
            fingerprint = failure_fingerprint(ledger["kind"], normalized, errors)
            step = next_step(attempts, fingerprint, len(errors))
            attempts.append({**identity, "fingerprint": fingerprint, "errors": [str(error) for error in errors]})
            ledger.update(attempts=attempts, status="exhausted" if step == "stop" else "pending")
        subject = f"task:{parent.id}:result_correction" + (f":cycle:{ledger['cycle']}" if ledger.get("cycle", 1) > 1 else "")
        history = [
            f"task {item['task_id']} session {item['session_id']}: {len(item['errors'])} error(s): "
            + "; ".join(item["errors"])
            for item in attempts
        ]
        if ledger["status"] == "exhausted":
            if not ledger.get("decision_id"):
                ask = await self._escalate_owner(
                    db, goal, run, "result_correction_exhausted", subject, history,
                    f"The result of task {parent.id} still fails validation after {len(attempts) - 1} correction(s). "
                    "Retry starts a new correction cycle; dismiss stops correcting.",
                    options=RESULT_CORRECTION_ASK_OPTIONS,
                )
                ledger["decision_id"] = str(ask.target_id) if ask.target_id else None
            return
        if ledger["status"] == "correcting":
            correction = await db.get(Task, uuid.UUID(ledger["correction_task_id"]))
            if correction is not None and correction.status not in {"failed", "cancelled"}:
                await self._reopen_invalid_report_gate(db, run, parent, marker)
            return
        if ledger["status"] != "pending":
            return
        latest = next(item for item in reversed(attempts) if not item.get("task_failed"))
        source = task if latest["task_id"] == str(task.id) else await db.get(Task, uuid.UUID(latest["task_id"]))
        failed_output, unavailable = await self._stored_result_output(
            db, source, self._optional_uuid(latest["session_id"], "session_id"),
        ) if source is not None else (None, f"task {latest['task_id']} is missing")
        if failed_output is None:
            ledger.update(status="unavailable", reason="correction_input_unavailable", detail=unavailable)
            ask = await self._escalate_owner(
                db, goal, run, "capability_unavailable",
                f"task:{parent.id}:correction_input_unavailable" + subject.removeprefix(
                    f"task:{parent.id}:result_correction"
                ),
                history, f"Cannot correct the result of task {parent.id}: {unavailable}.",
                options=RESULT_CORRECTION_ASK_OPTIONS,
            )
            ledger["decision_id"] = str(ask.target_id) if ask.target_id else None
            return
        budget, shortfall = await self._result_correction_budget(goal, run, db, parent.id)
        if shortfall is not None:
            await self._escalate_owner(
                db, goal, run, "budget_exhausted", subject, history, f"Cannot fund a result correction: {shortfall}.",
            )
            return
        await self._dispatch_result_correction(
            db, goal, run, parent, ledger, failed_output, latest, budget, subject, history, version,
        )

    async def _new_result_correction_cycle(
        self, db: AsyncSession, run: OrchestrationRun, parent: Task, ledger: dict[str, Any], version: dict[str, Any],
    ) -> None:
        """Start a fresh bounded cycle; any in-flight correction is unlinked and never consumed.

        The old cycle's open owner asks are cancelled and its subject blockers dropped, so nothing stale waits."""
        from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

        cycle = int(ledger.get("cycle", 1))
        suffix = f":cycle:{cycle}" if cycle > 1 else ""
        old_scopes = {f"task:{parent.id}:result_correction{suffix}", f"task:{parent.id}:correction_input_unavailable{suffix}"}
        stale = [item for item in run.active_blockers
                 if isinstance(item, Mapping) and item.get("kind") == "owner_escalation" and item.get("scope") in old_scopes]
        run.active_blockers = [item for item in run.active_blockers if item not in stale]
        decision_ids = {ledger.get("decision_id"), *(item.get("decision_id") for item in stale)} - {None}
        for decision_id in decision_ids:
            decision = await db.get(OrchestrationAuthorityDecision, uuid.UUID(str(decision_id)))
            if decision is not None and decision.status == "pending":
                await OrchestrationAuthorityDecisionService().cancel_decision(
                    db, decision, reason="superseded by a new result correction cycle",
                )
        ledger.update(
            cycle=int(ledger.get("cycle", 1)) + 1, version=version, status="pending", decision_id=None,
            correction_task_id=None, correction_session_id=None, revision_fallback=False,
            previous_attempts=[*ledger.get("previous_attempts", []), *ledger.get("attempts", [])], attempts=[],
        )

    async def _sweep_result_corrections(self, db: AsyncSession, run: OrchestrationRun) -> None:
        """Re-enter open correction ledgers each tick: funding, resume, contract change, owner answer, failed task."""
        if run.status != "running":
            return
        # ponytail: oldest 50 open ledgers per tick; page by cursor if a run ever holds more
        markers = list(await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id,
            OrchestrationAction.action_type == "report_consumed",
            OrchestrationAction.request["result_correction"]["status"].as_string().in_(
                ("pending", "correcting", "exhausted", "unavailable")
            ),
        ).order_by(OrchestrationAction.created_at, OrchestrationAction.id).limit(50)))
        for marker in markers:
            await self._resume_result_correction(db, run, marker)

    async def _resume_result_correction(
        self, db: AsyncSession, run: OrchestrationRun, marker: OrchestrationAction,
    ) -> None:
        ledger = dict(self._json_object_or_empty(self._json_object_or_empty(marker.request).get("result_correction")))
        parent = await db.get(Task, marker.target_id) if marker.target_id else None
        goal = await db.get(OrchestrationGoal, run.goal_id)
        if not ledger or parent is None or goal is None:
            return
        version = self._result_correction_version(goal, run)
        if (ledger.get("status") == "correcting" and not ledger.get("correction_session_id")
                and ledger.get("version") == version):
            correction = await db.get(Task, uuid.UUID(ledger["correction_task_id"]))
            if correction is not None and correction.status == "done":  # finished while it could not be consumed
                await self._consume_result_correction(db, run, correction, None)
                return
        latest = next((item for item in reversed(ledger.get("attempts", [])) if not item.get("task_failed")), None)
        task = None
        if latest is not None:
            task = parent if latest["task_id"] == str(parent.id) else await db.get(Task, uuid.UUID(latest["task_id"]))
        if task is None:
            return
        try:
            await self._advance_result_correction(
                db, goal, run, parent, marker, ledger, task,
                self._optional_uuid(latest["session_id"], "session_id"), latest["errors"], version,
            )
            self._plan_correction_fallback(run, ledger)
        finally:
            marker.request = {**self._json_object_or_empty(marker.request), "result_correction": ledger}

    async def _dispatch_result_correction(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        parent: Task,
        ledger: dict[str, Any],
        failed_output: str,
        latest: Mapping[str, Any],
        budget: dict[str, str],
        subject: str,
        history: Sequence[str],
        version: dict[str, Any],
    ) -> None:
        contract = self._json_object_or_empty(self._json_object_or_empty(parent.metadata_).get("orchestration_contract"))
        kind = ledger["kind"]
        report_schema = self._json_object_or_empty(contract.get("report_schema"))
        brief = build_correction_brief(
            parent.description or "", failed_output, self._result_schema(goal, kind, report_schema), latest["errors"],
            {"task_id": str(parent.id), "run_id": str(run.id), **version},
        )
        request = {
            "action_type": "create_delegation_task", "agent_id": str(parent.assigned_to),
            "work_function": "report_clarification",
            "title": f"Correct result: {parent.title}",
            "scope": "Output-only correction: re-emit the prior task's result so it passes validation.",
            "inputs": [f"Prior task: {parent.id}"],
            "deliverable": "The corrected result.",
            "forbidden_work": [
                "Do not redo the work, repeat tool or external effects, invent evidence or widen scope.",
            ],
            "success_evidence": ["A result that passes validation."],
            "budget": budget,
            "report_schema": report_schema,
            "parent_task_id": str(parent.id),
            "source_session_id": latest["session_id"],
        }
        if contract.get("orchestrator_context"):
            request["orchestrator_context"] = contract["orchestrator_context"]
        if contract.get("authorization_scope"):
            request["authorization_scope"] = contract["authorization_scope"]
        dispatched = list(ledger.get("correction_task_ids") or [])
        key = f"run:{run.id}:kind:report_clarification:task:{parent.id}"
        if dispatched:
            key += f":correction:{len(dispatched) + 1}"
        try:
            if parent.assigned_to is None:
                raise HTTPException(status_code=409, detail="the original task has no assigned agent")
            # Output-only re-emission of already admitted work: the baseline gate already passed for the original.
            action = await self.execute_create_delegation_task_action(
                db, run.id, request, key, skip_baseline_gate=True, expected_result=kind,
            )
        except HTTPException as exc:
            if not 400 <= exc.status_code < 500:
                raise
            category = "budget_exhausted" if exc.detail == "budget_wait" else "capability_unavailable"
            await self._escalate_owner(
                db, goal, run, category, subject, history, f"Cannot dispatch a result correction: {exc.detail}.",
            )
            return
        correction = await db.get(Task, action.target_id) if action.status == "completed" and action.target_id else None
        if correction is None:
            return
        correction_metadata = self._json_object_or_empty(correction.metadata_)
        if "result_correction_parent_task_id" not in correction_metadata:
            correction.metadata_ = {**correction_metadata, "result_correction_parent_task_id": str(parent.id)}
            correction.description = f"{correction.description or ''}\n\n{brief}"
        ledger.update(
            status="correcting", correction_task_id=str(correction.id), correction_session_id=None,
            correction_task_ids=[*dispatched, str(correction.id)] if str(correction.id) not in dispatched else dispatched,
        )
        run.active_blockers = [  # funding returned: the subject's budget ask no longer blocks it
            item for item in run.active_blockers
            if not (isinstance(item, Mapping) and item.get("kind") == "owner_escalation"
                    and item.get("category") == "budget_exhausted" and item.get("scope") == subject)
        ]
        await self._reopen_invalid_report_gate(db, run, parent, await self._existing_action_for_key(
            db, run.id, f"run:{run.id}:kind:report_consumed:task:{parent.id}",
        ))

    async def _result_correction_budget(
        self, goal: OrchestrationGoal, run: OrchestrationRun, db: AsyncSession, parent_task_id: uuid.UUID,
    ) -> tuple[dict[str, str], str | None]:
        """(allowance, shortfall): capped goals use the protected allowance; uncapped goals need none."""
        from huddleroom.services.orchestration_budget_service import (
            BudgetMeasurementError,
            OrchestrationBudgetService,
            parent_caps,
        )

        if self._budget_is_exhausted(run):
            return {}, "budget capacity is exhausted"
        if not parent_caps(goal):
            return {}, None
        budgets = OrchestrationBudgetService()
        try:
            budget = await budgets.protected_action_allocation(db, goal, run)
        except BudgetMeasurementError as exc:
            self._upsert_active_blocker(run, {
                "kind": "budget_measurement", "dimension": exc.dimension,
                "session_id": str(exc.session_id), "scope": f"result_correction:{parent_task_id}",
            })
            return {}, f"missing {exc.dimension} measurement"
        if not any(Decimal(amount) for amount in budget.values()):
            self._upsert_active_blocker(run, {"kind": "budget_exhausted"})
            return budget, "budget capacity is exhausted"
        if not await budgets.can_dispatch(db, goal, run, budget):
            if self._has_active_blocker(run, "budget_measurement"):
                return budget, "budget telemetry is incomplete"
            self._upsert_active_blocker(run, {"kind": "budget_exhausted"})
            return budget, "budget capacity is exhausted"
        return budget, None

    @staticmethod
    def _result_correction_version(goal: OrchestrationGoal, run: OrchestrationRun) -> dict[str, Any]:
        # Budget is left out on purpose: raising it must fund a waiting correction, not invalidate it.
        return {
            "owner_contract": stable_hash({
                "objective": goal.objective, "success_criteria": goal.success_criteria,
                "constraints": goal.constraints,
            }),
            "planning_version": OrchestrationService._json_object_or_empty(run.plan_state).get("planning_version"),
        }

    async def _result_correction_stale(self, db: AsyncSession, run: OrchestrationRun) -> str | None:
        if run.status not in LLM_DECISION_RUN_STATUSES:
            return f"run_{run.status}"
        latest = await db.scalar(select(OrchestrationRun.id).where(
            OrchestrationRun.goal_id == run.goal_id
        ).order_by(OrchestrationRun.started_at.desc(), OrchestrationRun.id.desc()).limit(1))
        if latest != run.id:
            return "run_superseded"
        return None

    async def _stored_result_output(
        self, db: AsyncSession, task: Task, session_id: uuid.UUID | None,
    ) -> tuple[str | None, str | None]:
        """(complete stored output, None) or (None, exact reason it cannot be used for a correction)."""
        from huddleroom.services.tool_executor import TOOL_EXECUTED_NO_CONTENT

        if session_id is None:
            orchestration = self._json_object_or_empty(self._json_object_or_empty(task.metadata_).get("orchestration"))
            output = orchestration.get("terminal_output")
        else:
            session = await db.get(Session, session_id)
            if session is None or session.task_id != task.id or session.status != "completed":
                return None, f"session {session_id} is not a completed session of task {task.id}"
            output = session.output
        if not isinstance(output, str) or not output.strip():
            return None, f"the stored output of task {task.id} is absent"
        if output == TOOL_EXECUTED_NO_CONTENT:
            return None, f"the stored output of task {task.id} is a placeholder, not the agent's result"
        return output, None

    async def _reopen_invalid_report_gate(
        self, db: AsyncSession, run: OrchestrationRun, parent: Task, marker: OrchestrationAction | None,
    ) -> None:
        """Reopen the original's ad-hoc gate, failed only for its invalid report, while a correction is pending."""
        gate_id = task_gate_id(parent, run.id)
        gate = await db.get(OrchestrationGate, gate_id) if gate_id else None
        if (gate is not None and gate.status == "failed" and gate.gate_type == ADHOC_GATE_TYPE
                and gate.id == uuid.uuid5(uuid.NAMESPACE_URL,
                                         f"rally:orchestration:gate:{run.id}:{ADHOC_GATE_TYPE}:adhoc_task:{parent.id}")
                and gate.success_criterion_key == f"adhoc_task:{parent.id}"
                and gate.required_evidence.get("task_id") == str(parent.id)
                and gate.failure_reason == "Task report is invalid"
                and marker is not None and marker.status == "completed" and marker.target_id == parent.id):
            self._reopen_gate(gate)

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
        async with self._lock_goal_for_baseline_transition(db, goal_id):
            goal = await self.get_goal(db, project_id, goal_id)
            if goal is None:
                raise HTTPException(status_code=404, detail="Orchestration goal not found")
            if goal.status not in ("active", "blocked"):
                raise self._start_conflict(
                    "goal_not_runnable", f"Cannot start baseline for goal in status '{goal.status}'",
                )
            run = await self.get_active_run_for_goal(db, project_id, goal_id)
            if run is None:
                raise self._start_conflict("goal_not_runnable", "Goal has no active run")
            if run.phase != "baseline":
                raise self._start_conflict("baseline_complete", f"Run phase is {run.phase}")
            if not run.baseline_authorized:
                run.baseline_authorized = True
                await db.flush()
            return goal, run

    async def _record_reauthorization(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        actor: str,
        caller_owns_transaction: bool,
    ) -> None:
        """Re-Start after a contract change: one action per current fingerprint; the run's
        frozen fingerprint is never rewritten. Legacy NULL fingerprints are grandfathered."""
        fingerprint = compute_goal_contract_fingerprint(goal)
        if run.goal_contract_fingerprint in (None, fingerprint):
            return
        action = await self.reserve_action(
            db,
            run_id=run.id,
            idempotency_key=reauthorize_execution_key(run.id, fingerprint),
            action_type="reauthorize_execution",
            request={"actor": actor, "contract_fingerprint": fingerprint},
        )
        if action.status == "reserved":
            await self._mark_action_completed(db, action, target_type="run", target_id=run.id)
        await db.flush()
        if not caller_owns_transaction:
            await db.commit()

    async def start_run(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
        *,
        actor: str,
        commit: bool = True,
    ) -> tuple[OrchestrationGoal, OrchestrationRun]:
        """Explicit human authorization of spend + execution (spec: Start work).
        Idempotent. The scheduler never executes a run whose phase != authorized."""
        transaction = db.sync_session.get_transaction()
        caller_owns_transaction = not commit or (
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
                await self._record_reauthorization(db, goal, run, actor, caller_owns_transaction)
                return goal, run

            # Idempotent replay: already authorized -> return as-is.
            existing = await self._existing_action_for_key(
                db, run.id, self._authorize_execution_key(run.id)
            )
            if existing is not None and run.phase in {"authorized", "completed"}:
                await self._record_reauthorization(db, goal, run, actor, caller_owns_transaction)
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
            # Baseline processes may still rewrite contract fields while ready; Start freezes them.
            if run.goal_contract_fingerprint is None:
                run.goal_contract_fingerprint = compute_goal_contract_fingerprint(goal)
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
                    "contract_fingerprint": run.goal_contract_fingerprint,
                    "goal_type": goal.goal_type,
                    "planning_mode": goal.planning_mode,
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
            await self._warn_start_text_conflicts(db, goal, run)
            await db.flush()
            if not caller_owns_transaction:
                await db.commit()
        return goal, run

    @classmethod
    def _constraint_strings(cls, value) -> list[str]:
        if isinstance(value, str):
            return [value]
        items = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
        return [s for item in items for s in cls._constraint_strings(item)]

    async def _warn_start_text_conflicts(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun
    ) -> None:
        """Tell the owner when project/goal text says "don't start" but they pressed Start.
        Never blocks Start. run.phase is the source of truth; no flag is written to constraints."""
        # ponytail: phrase list is a fixed regex in orchestration_start_text_check; add phrases as they show up.
        try:
            project = await db.get(Project, goal.project_id)
            hits = find_start_contradictions({
                "project description": project.description if project else None,
                "goal objective": goal.objective,
                "goal constraints": " ".join(self._constraint_strings(goal.constraints)) or None,
            })
            if not hits:
                return
            by_source: dict[str, list[str]] = {}
            for source, phrase in hits:
                by_source.setdefault(source, []).append(f'"{phrase}"')
            parts = [
                f"the {source}{' (Settings → Project)' if source == 'project description' else ''} says "
                + ", ".join(phrases)
                for source, phrases in by_source.items()
            ]
            message = (
                f"You started this goal, but {'; and '.join(parts)}. "
                "The orchestrator reads this text in every decision; edit it so it matches."
            )
            # create_warning dedupes active rows per (goal, type, run), so repeats don't pile up.
            async with db.begin_nested():
                await OrchestrationWarningService().create_warning(
                    db, goal.id, warning_type="start_text_conflict", severity="recommendation",
                    run_id=run.id, message=message,
                )
        except Exception:  # noqa: BLE001 - advisory only
            logging.getLogger(__name__).warning("start text conflict check failed", exc_info=True)

    def _start_conflict(self, conflict: str, message: str) -> HTTPException:
        return HTTPException(status_code=409, detail={"conflict": conflict, "message": message})

    async def pause_goal(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
        *,
        commit: bool = True,
    ) -> tuple[OrchestrationGoal, OrchestrationRun | None]:
        transaction = db.sync_session.get_transaction()
        caller_owns_transaction = not commit or transaction is not None and transaction.origin != SessionTransactionOrigin.AUTOBEGIN
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
        async with self._lock_goal_for_baseline_transition(db, goal_id):
            goal = await db.scalar(
                select(OrchestrationGoal)
                .where(OrchestrationGoal.id == goal_id, OrchestrationGoal.project_id == project_id)
                .execution_options(populate_existing=True)
            )
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
            # Committed with the status change; the worker ticks after commit, never here.
            await emit_event_once(
                db,
                project_id,
                GOAL_RESUMED_EVENT_TYPE,
                {"goal_id": str(goal_id), "run_id": str(run.id) if run is not None else None},
            )
        return goal, run

    async def cancel_goal(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
        *,
        cancelled_by: str = "human:anonymous",
        commit: bool = True,
    ) -> tuple[OrchestrationGoal, OrchestrationRun | None]:
        transaction = db.sync_session.get_transaction()
        caller_owns_transaction = not commit or (
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
                    await SessionService().cancel(db, session.id, sync_task=False)
                await TaskService().cancel(db, goal.project_id, task.id)
        goal.status = "cancelled"
        if run is not None:
            run.status = "cancelled"
            run.completed_at = _utcnow()
        await db.flush()

    @staticmethod
    async def has_type_correction_lineage(db: AsyncSession, goal_id: uuid.UUID) -> bool:
        """True once any run of the goal is a type-correction successor or was retired by one."""
        return await db.scalar(select(OrchestrationRun.id).where(
            OrchestrationRun.goal_id == goal_id,
            or_(OrchestrationRun.supersedes_run_id.is_not(None), OrchestrationRun.terminal_reason.is_not(None)),
        ).limit(1)) is not None

    async def _require_resettable_goal(self, db: AsyncSession, project_id: uuid.UUID, goal_id: uuid.UUID):
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
        if (child_lineage or owned_roadmap_lineage or owned_continuous_lineage
                or await self.has_type_correction_lineage(db, goal.id)):
            raise HTTPException(status_code=409, detail="Cannot reset goal with immutable execution lineage")

        return goal

    async def reset_goal(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
    ) -> tuple[OrchestrationGoal, OrchestrationRun]:
        """Reset a goal to fresh state, wiping all run history and orchestration progress.

        Status lines / timeline / logs are UI renderings of the wiped tables — no
        separate store to clear. Also wipes the goal's steering requests and proposals.
        """
        transaction = db.sync_session.get_transaction()
        caller_owns_transaction = (
            transaction is not None and transaction.origin != SessionTransactionOrigin.AUTOBEGIN
        )
        from huddleroom.services.orchestration_baseline_runtime import reset_baseline
        await self._require_resettable_goal(db, project_id, goal_id)
        if not caller_owns_transaction:
            await db.rollback()
        async with reset_baseline(goal_id), self._lock_goal_for_baseline_transition(db, goal_id):
            goal = await self._require_resettable_goal(db, project_id, goal_id)

            # Bulk-delete goal-keyed tables (goal-keyed deletes first, then the run)
            await db.execute(delete(OrchestrationProcessRun).where(OrchestrationProcessRun.goal_id == goal_id))
            await db.execute(delete(OrchestrationWarning).where(OrchestrationWarning.goal_id == goal_id))
            await db.execute(delete(OrchestrationAuthorityDecision).where(
                OrchestrationAuthorityDecision.goal_id == goal_id
            ))
            await db.execute(delete(OrchestrationAgentReview).where(OrchestrationAgentReview.goal_id == goal_id))
            await db.execute(delete(OrchestrationMemorySection).where(OrchestrationMemorySection.goal_id == goal_id))
            # Steering requests (DB-cascades transitions + result_links, which otherwise RESTRICT
            # the run's decisions/actions) and proposals; steering state is kept.
            await db.execute(delete(OrchestrationSteeringRequest).where(OrchestrationSteeringRequest.goal_id == goal_id))
            # Conversation children (responses, feedback, reservations and proposals) cascade.
            await db.execute(delete(ConversationMessage).where(ConversationMessage.goal_id == goal_id))
            await db.execute(delete(OrchestrationSteeringProposal).where(OrchestrationSteeringProposal.goal_id == goal_id))

            # Delete goal's run row(s) — CASCADEs run-keyed children (decisions/actions/gates/evidence/agent_suggestions)
            await db.execute(delete(OrchestrationRun).where(OrchestrationRun.goal_id == goal_id))

            # Recreate ONE fresh run
            run = OrchestrationRun(
                goal_id=goal.id,
                budget_state=deepcopy(goal.budget),
                baseline_authorized=False,
                goal_type_snapshot=goal.goal_type,
                planning_mode_snapshot=goal.planning_mode,
            )
            db.add(run)

            # Reset goal columns to fresh state
            policy = (goal.orchestrator_context or {}).get("expansion_policy")
            goal.orchestrator_context = {"expansion_policy": policy} if policy else {}
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
        gate_id = task_gate_id(task, run_id)
        if gate_id is None:
            return None

        gate = await db.get(OrchestrationGate, gate_id)
        if gate is None or gate.run_id != run_id:
            return None
        orchestration = self._json_object_or_empty(self._json_object_or_empty(task.metadata_).get("orchestration"))
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
        if event.event_type in GRAPH_EVIDENCE_EVENT_TYPES:
            return await self._ingest_graph_evidence(db, run, event)
        if event.event_type in MEETING_EVIDENCE_EVENT_TYPES:
            return await self._ingest_meeting_evidence(db, run, event)
        return 0

    async def _ingest_ungated_output(
        self, db: AsyncSession, run: OrchestrationRun, task: Task, session_id: uuid.UUID | None,
        event_id: uuid.UUID | None = None,
    ) -> None:
        """Route an ungated planning-role task's terminal output by the result kind its contract expects."""
        metadata = self._json_object_or_empty(task.metadata_)
        if metadata.get("result_correction_parent_task_id"):
            await self._consume_result_correction(db, run, task, session_id, event_id)
            return
        orchestration = self._json_object_or_empty(metadata.get("orchestration"))
        if orchestration.get("work_function") != PLAN_WORK_FUNCTION or orchestration.get("run_id") != str(run.id):
            return
        kind = expected_result_kind(self._json_object_or_empty(metadata.get("orchestration_contract")), False)
        if kind == PLANNING_ENVELOPE:
            await self._ingest_planning_output(db, run, task, session_id)
        elif kind == WORK_REPORT and task.status == "done":
            await self.consume_canonical_report(db, run, task, session_id, event_id)

    async def _consume_result_correction(
        self, db: AsyncSession, run: OrchestrationRun, task: Task, session_id: uuid.UUID | None,
        event_id: uuid.UUID | None = None,
    ) -> None:
        """Consume a correction task's result only when the original's ledger names it as current, it points back
        to the original, and the run, owner contract and plan version are still the ones it was dispatched under."""
        if task.status != "done":
            return
        parent_id = self._optional_uuid(
            self._json_object_or_empty(task.metadata_).get("result_correction_parent_task_id"),
            "result_correction_parent_task_id",
        )
        marker = await self._existing_action_for_key(db, run.id, f"run:{run.id}:kind:report_consumed:task:{parent_id}")
        ledger = self._json_object_or_empty(self._json_object_or_empty(marker.request if marker else {}).get(
            "result_correction"
        ))
        if marker is None or ledger.get("correction_task_id") != str(task.id):
            return
        if session_id is None:
            sessions = list(await db.scalars(select(Session.id).where(
                Session.task_id == task.id, Session.status == "completed",
            )))
            if len(sessions) > 1:
                return
            session_id = sessions[0] if sessions else None
        session = await db.get(Session, session_id) if session_id is not None else None
        goal = await db.get(OrchestrationGoal, run.goal_id)
        if goal is None or (session_id is not None and (
            session is None or session.task_id != task.id or session.status != "completed"
        )):
            return
        if ledger.get("version") != self._result_correction_version(goal, run):
            await self._resume_result_correction(db, run, marker)  # new contract version: fresh cycle, not this result
            return
        stale = await self._result_correction_stale(db, run)
        recorded = {"stale": stale} if stale else {"correction_session_id": str(session_id) if session_id else None}
        marker.request = {**self._json_object_or_empty(marker.request), "result_correction": {
            **{key: value for key, value in ledger.items() if key != "stale"}, **recorded,
        }}
        if stale:
            return
        if ledger.get("kind") == PLANNING_ENVELOPE:
            if ledger.get("status") != "correcting":
                return  # a replayed completion after the correction settled lands nothing
            parent = await db.get(Task, parent_id)
            if session is None or parent is None:  # a plan only lands from a recorded session of the correction
                self._set_result_correction_status(marker, "rejected", reason="the correction has no recorded session")
                return
            await self._land_plan_output(db, run, parent, marker, task, session.id, session.output)
            return
        if ledger.get("kind") != WORK_REPORT:
            return  # other kinds land through their own consumers
        _, report = await self.consume_canonical_report(db, run, task, session_id, event_id)
        if report is not None:
            # The corrected report becomes the original's effective report; the failed candidate stays recorded.
            from huddleroom.services.orchestration_work_report import work_report_payload

            request = self._json_object_or_empty(marker.request)
            # `report` now holds the corrected content while `session_id` stays the original's; read
            # `effective_report` for the session that produced it.
            marker.request = {
                **request,
                "original_report": request.get("original_report", request.get("report")),
                "report": work_report_payload(report),
                "effective_report": {"task_id": str(task.id), "session_id": str(session_id) if session_id else None},
            }
            self._set_result_correction_status(marker, "corrected")

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
            if status == "done":
                await self._ingest_ungated_output(db, run, task, None, event.id)
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
        if (
            created
            and status == "done"
            and gate.status == "failed"
            and gate.failure_reason in ATTEMPT_REJECTION_REASONS
        ):
            self._reopen_gate(gate)
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
            if event.event_type == "session.completed" and task.status == "done":
                await self._ingest_ungated_output(db, run, task, session.id)
            return 0

        gate, orchestration = gate_ref
        if await self._is_outcome_verification_task(db, run, gate, task, orchestration):
            return await self._ingest_outcome_verification_evidence(db, run, event, gate, task, session)
        status = self._optional_string(session.status)
        metadata = {
            **self._base_evidence_metadata(event, gate, orchestration, status),
            "task_id": str(task.id),
            "session_id": str(session.id),
            "graph_run_id": str(session.graph_run_id) if session.graph_run_id else None,
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
            and gate.status == "failed"
            and (
                gate.gate_type == FINAL_SUMMARY_GATE_TYPE
                or gate.failure_reason in ATTEMPT_REJECTION_REASONS
            )
        ):
            self._reopen_gate(gate)

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
    def _verification_report(output: str | None) -> tuple[str | None, str | None]:
        """Return (verdict, how the output was normalized); verdict is None when the report is invalid."""
        try:
            text, how = normalize_result_output(output or "")
            report = json.loads(text)
        except (TypeError, ValueError):
            return None, None
        if not isinstance(report, Mapping) or report.get("status") != "done":
            return None, None
        verdict = report.get("verdict")
        if verdict not in {"accepted", "rejected"}:
            return None, None
        evidence = report.get("evidence")
        if not isinstance(evidence, list) or any(not isinstance(item, str) for item in evidence):
            return None, None
        if verdict == "accepted" and not any(item.strip() for item in evidence):
            return None, None
        return verdict, how

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
            if task.graph_run_id is not None or session.graph_run_id is not None:
                if task.graph_run_id != session.graph_run_id:
                    return None
                graph_run = await db.get(GraphRun, session.graph_run_id)
                if graph_run is None or graph_run.linked_task_id != task.id:
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
        if task.graph_run_id is not None or session.graph_run_id is not None:
            if task.graph_run_id != session.graph_run_id:
                return None
            graph_run = await db.get(GraphRun, session.graph_run_id)
            if graph_run is None or graph_run.linked_task_id != task.id:
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
        verdict, how = self._verification_report(session.output)
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
                "graph_run_id": str(session.graph_run_id) if session.graph_run_id else None,
                **({"normalized": how} if how else {}),
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
                or self._verification_report(session.output)[0] != "accepted"
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
        graph_run_id = self._event_uuid(event.payload.get("graph_run_id"))
        if graph_run_id is not None and graph_run_id != session.graph_run_id:
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
        if review_verdict == "commented":
            return 0
        metadata = {
            **self._base_evidence_metadata(event, gate, orchestration, review_verdict),
            "task_id": str(task.id),
            "session_id": str(session.id),
            "graph_run_id": str(session.graph_run_id) if session.graph_run_id else None,
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
            verdict=(
                "candidate"
                if event.event_type == "review.approved" and review_verdict in (None, "approved")
                else "rejected"
            ),
            evidence_metadata=metadata,
        )

    async def _ingest_graph_evidence(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        event: EventLog,
    ) -> int:
        graph_run_id = self._event_uuid(event.payload.get("graph_run_id"))
        if graph_run_id is None:
            return 0
        graph_run = await db.get(GraphRun, graph_run_id)
        if graph_run is None or graph_run.linked_task_id is None:
            return 0
        task = await db.get(Task, graph_run.linked_task_id)
        if task is None:
            return 0
        gate_ref = await self._task_gate_for_run(db, run.id, task)
        if gate_ref is None:
            return 0

        gate, orchestration = gate_ref
        status = self._optional_string(graph_run.status)
        is_step_event = event.event_type == "graph.run_advanced"
        has_transition_id = "graph_run_step_id" in event.payload
        transition_id = self._event_uuid(event.payload.get("graph_run_step_id"))
        transition = None
        if is_step_event or has_transition_id:
            if transition_id is None:
                return 0
            transition = await db.get(GraphRunStep, transition_id)
            if transition is None or transition.graph_run_id != graph_run.id:
                return 0

        source_type = (
            "graph_run_step"
            if is_step_event
            else "graph_run"
        )
        source_id = transition.id if is_step_event else graph_run.id
        metadata = {
            **self._base_evidence_metadata(event, gate, orchestration, status),
            "task_id": str(task.id),
            "graph_run_id": str(graph_run.id),
            "graph_run_step_id": str(transition_id) if transition_id is not None else None,
            "graph_name": self._optional_string(event.payload.get("graph_name")),
            "current_node": graph_run.current_node,
            "edge_name": self._optional_string(event.payload.get("edge_name")),
            "from_node": self._optional_string(event.payload.get("from_node")),
            "to_node": self._optional_string(event.payload.get("to_node")),
        }
        return await self._record_evidence_once(
            db,
            run_id=run.id,
            gate_id=gate.id,
            source_type=source_type,
            source_id=source_id,
            observed_event=event,
            producer_agent_id=None,
            verdict="rejected" if event.event_type == "graph.run_failed" else "candidate",
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

        review_verdict = await self._meeting_review_verdict(db, meeting, decision.agenda_item_id)
        if review_verdict is None:
            return 0
        verdict = "rejected" if decision.is_vetoed else review_verdict
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
            verdict=verdict,
            evidence_metadata=metadata,
        )

    @staticmethod
    async def _meeting_review_verdict(
        db: AsyncSession, meeting: Meeting, agenda_item_id: uuid.UUID | None = None
    ) -> str | None:
        """Evidence verdict for a review meeting's items: candidate, rejected, or None (no evidence yet).

        Non-review meetings are always candidate. Human-overridden items (resolved without a
        resolution kind) count as approved; unresolved, majority and deferred items yield no evidence.
        """
        if meeting.meeting_type != "review":
            return "candidate"
        query = select(MeetingAgendaItem).where(MeetingAgendaItem.meeting_id == meeting.id)
        if agenda_item_id is not None:
            query = query.where(MeetingAgendaItem.id == agenda_item_id)
        kinds: list[str | None] = []
        for item in (await db.execute(query)).scalars().all():
            if item.resolution_kind is None and item.status == "resolved":
                kinds.append("approved")
            else:
                kinds.append(item.resolution_kind)
        if any(k in {"rejected", "changes_requested", "verdict_missing"} for k in kinds):
            return "rejected"
        if all(k in {"approved", "approved_with_followups"} for k in kinds):
            return "candidate"
        return None

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
        verdict = await self._meeting_review_verdict(db, meeting)
        if verdict is None:
            return 0
        if not decisions:
            return await self._record_evidence_once(
                db,
                run_id=run.id,
                gate_id=gate.id,
                source_type="meeting",
                source_id=meeting.id,
                observed_event=event,
                producer_agent_id=producer_agent_id,
                verdict=verdict,
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
                verdict=verdict,
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
                # A task-link alone is not a durable worker claim. Only the
                # current, fenced attempt takes recovery
                # ownership away from the legacy retry/reassign ladder.
                if (current_owned is not None
                        and current_attempt.get("claimed_runner_task_id") == current_session.runner_task_id
                        and isinstance(current_attempt.get("attempt_version"), int)):
                    owned_task_ids.add(task.id)
                    continue
            failure_counts[task.id] = max(1, len(failed_session_ids))
        for task in failed_tasks:
            # An approved recovery attempt that failed again belongs to its own lineage (re-diagnosis there),
            # never to the legacy ladder or a second lineage.
            effect = await self._recovery_effect_for_attempt(db, run, task)
            if task.id not in owned_task_ids and effect is not None and await self.route_failure_to_recovery_lineage(
                    db, run, effect, f'Task "{task.title}" failed again after the approved fix.'):
                owned_task_ids.add(task.id)
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
        owner_only = []
        for task in await self._orchestrated_tasks_for_run(db, run.id, statuses=["blocked"]):
            blocked = self._json_object_or_empty(self._json_object_or_empty(task.metadata_).get("blocked"))
            episode_at = self._optional_string(blocked.get("at"))
            key = f"run:{run.id}:kind:ask_human:blocked_task:{task.id}" + (f":{episode_at}" if episode_at else "")
            if await self._existing_action_for_key(db, run.id, key) is not None:
                continue
            handled = await self._diagnose_blocked_task(db, goal, run, task, episode_at)
            if handled is not None:
                created += handled
            else:
                owner_only.append((task, blocked, key))
        # Owner-only input: the only branch that blocks the run, and only after all tasks are triaged.
        for task, blocked, key in owner_only:
            reason = self._optional_string(blocked.get("reason"))
            if reason is not None:
                reason = reason[:200]
            if reason is None:
                latest = await db.scalar(select(Session).where(
                    Session.task_id == task.id, (Session.error.is_not(None)) | (Session.output.is_not(None)),
                ).order_by(Session.created_at.desc(), Session.id.desc()).limit(1))
                excerpt = str(latest.error or latest.output or "").strip()[:200] if latest is not None else ""
                reason = excerpt or "no reason was recorded"
            gate_id = await self._gate_id_for_task(db, run.id, task)
            nested = await db.begin_nested()
            try:
                self._mark_run_blocked(goal, run)  # idempotent; inside the savepoint so a failed ask rolls it back
                ask = await self.execute_ask_human_action(
                    db,
                    run_id=run.id,
                    request={
                        "action_type": "ask_human",
                        "question": (
                            f'Task "{task.title}" is blocked: {reason}. '
                            "What input or decision do you need to supply to unblock it?"
                        ),
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

    async def _diagnose_blocked_task(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        task: Task,
        episode_at: str | None,
    ) -> int | None:
        """Deterministic blocked-task triage: bounded wait, then retry/reassign.
        Returns the number of things created, or None when only the owner can unblock."""
        from huddleroom.models.orchestration import OrchestrationWait

        since = None
        if episode_at:
            try:
                since = datetime.fromisoformat(episode_at)
            except ValueError:
                since = None
            if since is not None and since.tzinfo is None:
                since = since.replace(tzinfo=timezone.utc)

        def in_episode(row) -> bool:
            if since is None:
                return True
            stamp = row.created_at if row.created_at.tzinfo else row.created_at.replace(tzinfo=timezone.utc)
            return stamp >= since

        # 1. Bounded wait on a dependency or live provider.
        waits = list(await db.scalars(select(OrchestrationWait).where(
            OrchestrationWait.run_id == run.id,
            OrchestrationWait.wait_key.like(f"run:{run.id}:wait:task:{task.id}:blocked:%"),
        )))
        if any(wait.status == "open" for wait in waits):
            return 0
        awaiting = await self._blocked_task_awaiting(db, run, task)
        # ponytail: waiting is bounded by elapsed time (24 wake windows since the episode's first wait), not wait count.
        stamps = [w.created_at if w.created_at.tzinfo else w.created_at.replace(tzinfo=timezone.utc)
                  for w in waits if in_episode(w)]
        within_bound = not stamps or (
            datetime.now(timezone.utc) - min(stamps)
        ).total_seconds() < settings.orchestration_wake_max_seconds * 24
        if awaiting and within_bound:
            label, event_task_id = awaiting
            try:
                await self.supervision.create_wait(
                    db, run, origin=f"task:{task.id}:blocked:{len(waits)}",
                    owner={"type": "task", "id": str(task.id), "task_id": str(task.id)},
                    awaited_event={"event_type": "task.status_changed", "matcher": {"task_id": str(event_task_id)}},
                    recheck_seconds=settings.orchestration_wake_max_seconds,
                    fallback={"action_type": "continue", "reason": f"Blocked task still waiting on {label}."},
                    expected_result=f"The blocked task can proceed once {label} finishes.",
                )
                return 1
            except HTTPException:
                return 0  # rejected create_wait with a live dependency: retry next tick, do not ask

        # 2. Recoverable ownership/agent failure: reassign or retry, once per blocked episode.
        agent = await db.get(Agent, task.assigned_to) if task.assigned_to else None
        agent_bad = agent is None or not agent.is_active
        latest_status = await db.scalar(select(Session.status).where(Session.task_id == task.id).order_by(
            Session.created_at.desc(), Session.id.desc()).limit(1))
        if not (agent_bad or latest_status == "failed"):
            return None
        prior = (await db.execute(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id,
            OrchestrationAction.action_type.in_(["retry_task", "reassign_task"]),
        ))).scalars().all()
        if any(str((act.request or {}).get("task_id")) == str(task.id) and in_episode(act) for act in prior):
            return None
        if agent_bad:
            work_function = self._task_work_function(task)
            fit = await self._best_recovery_agent(
                db, goal.project_id, work_function, required_capabilities=[work_function],
                exclude_agent_ids={task.assigned_to} if task.assigned_to else set(),
            )
            if fit is None:
                return None
            code, summary = "task_blocked_agent_unavailable", "its agent is unavailable; reassignment needs approval"
        else:
            code, summary = "task_blocked_session_failed", "its latest attempt failed; a retry needs approval"
        # Retry/reassign are corrective fixes: one recovery lineage per blocked episode, nothing mutated.
        return await self._task_recovery_trigger(
            db, run, task, blocker_code=code, summary=f'Task "{task.title}" is blocked: {summary}.',
        )

    async def _blocked_task_awaiting(
        self, db: AsyncSession, run: OrchestrationRun, task: Task
    ) -> tuple[str, uuid.UUID] | None:
        """(label, task id whose status event ends the wait) for what a blocked task awaits, or None."""
        # ponytail: no gate-accepted event; a done dependency with an open gate is released by the wait recheck (orchestration_wake_max_seconds)
        unmet = await first_unmet_dependency(db, run, task)
        if unmet is not None:
            dep = await db.get(Task, unmet[0])
            gate_id = task_gate_id(dep, run.id) if dep is not None else None
            gate = await db.get(OrchestrationGate, gate_id) if gate_id is not None else None
            # A predecessor that can never satisfy its gate (cancelled, missing, or done with a final
            # non-attempt gate failure) is not a wait, so the owner is asked at once.
            dead = dep is None or dep.status == "cancelled" or (
                dep.status == "done" and gate is not None and gate.status == "failed"
                and gate.failure_reason not in ATTEMPT_REJECTION_REASONS)
            if not dead:
                return f"dependency task {unmet[0]} (gate {unmet[1]})", unmet[0]
        session_id = await self._first_id(db, select(Session.id).where(
            Session.task_id == task.id, Session.status.in_(["pending", "running"])))
        if session_id is not None:
            return f"live session {session_id}", task.id
        meeting_id = await self._first_id(db, select(Meeting.id).where(
            Meeting.source_task_id == task.id,
            Meeting.status.in_(("scheduled", "preparing", "active", "concluding"))))
        if meeting_id is not None:
            return f"meeting {meeting_id}", task.id
        run_id = await self._first_id(db, select(GraphRun.id).where(
            GraphRun.linked_task_id == task.id, GraphRun.status == "active"))
        return (f"graph run {run_id}", task.id) if run_id is not None else None

    @staticmethod
    async def _first_id(db: AsyncSession, stmt):
        return await db.scalar(stmt.limit(1))

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
                final_summary_metadata = self._json_object_or_empty(
                    self._json_object_or_empty(source_task.metadata_).get("orchestration_final_summary")
                )
                # A replacement summary task is a corrective fix: one recovery lineage per invalid summary. Only a
                # typed marker is sealed; the gate-linked request (fit + accepted manifest) is built at retry time.
                created += await self._gate_recovery_trigger(
                    db, run, gate, latest_session_evidence.id, blocker_code="final_summary_invalid",
                    summary=f"The final summary was rejected ({gate.failure_reason}); a replacement needs approval.",
                    retry_action={"kind": "final_summary"},
                    link={"kind": "final_summary", "gate_id": str(gate.id), "final_summary": final_summary_metadata},
                )
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
        """Every corrective retry needs approval: start one recovery lineage instead of retrying."""
        del goal
        return await self._task_recovery_trigger(
            db, run, task, blocker_code="task_failed",
            summary=f'Task "{task.title}" failed; a new attempt is a corrective fix that needs approval.',
        )

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
            # A reassignment is a corrective fix: recovery proposes it (or something better) for approval.
            return await self._task_recovery_trigger(
                db, run, task, blocker_code="task_failed_after_retry",
                summary=f'Task "{task.title}" failed after the retry limit; reassignment needs approval.',
            )

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

    _ESCALATION_CATEGORIES = (
        "owner_decision", "permission_required", "capability_unavailable", "budget_exhausted",
        "result_correction_exhausted", "revision_exhausted", "review_input_missing",
    )

    @staticmethod
    def escalation_subject(category: str, task_id: Any, detail: str) -> str:
        """Subject for per-task escalations, e.g. `task:<id>:permission:<tool>` or `task:<id>:capability:<name>`."""
        label = {"permission_required": "permission", "capability_unavailable": "capability"}.get(category, category)
        return f"task:{task_id}:{label}:{detail}"

    async def _escalate_owner(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        category: str,
        subject: str,
        history: Sequence[str],
        question: str,
        options: Sequence[Mapping[str, str]] | None = None,
    ) -> OrchestrationAction:
        """Ask the owner about one subject and block only that subject; the run keeps dispatching other work."""
        if category not in self._ESCALATION_CATEGORIES:
            raise ValueError(f"Unknown escalation category: {category}")
        attempts = [str(item) for item in history][-10:]  # ponytail: keep only the latest attempts
        lines = "\n".join(f"{index}. {item}" for index, item in enumerate(attempts, 1)) or "None recorded."
        text = f"[{category}] {subject}\n{question}\nAttempt history:\n{lines}"
        key_subject = subject if len(subject) <= 120 else hashlib.sha256(subject.encode()).hexdigest()
        action = await self.execute_ask_human_action(
            db,
            run.id,
            {"question": text, "subject": f"{category}:{subject}", "reason": category,
             **({"options": [dict(option) for option in options]} if options else {})},
            f"run:{run.id}:kind:ask_human:{category}:{key_subject}",
        )
        decision = await db.get(OrchestrationAuthorityDecision, action.target_id) if action.target_id else None
        if decision is None or decision.status == "pending":  # an answered ask must not resurrect its blocker
            # kind is not the category: budget_exhausted etc. are synced/cleared by other blocker owners.
            # scope keys the blocker per subject; decision_id lets an owner answer clear it.
            self._upsert_active_blocker(run, {
                "kind": "owner_escalation", "category": category, "scope": subject, "subject": subject,
                "reason": f"{category}: {subject}", "history": attempts, "decision_id": str(action.target_id) if action.target_id else None,
            })
        await db.flush()
        return action

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
        # A fix task for rejected review is a corrective fix: one recovery lineage per rejected review; the
        # gate-linked fix task is sealed as its approved retry.
        request = {
            "action_type": "create_delegation_task",
            "agent_id": str(source_task.assigned_to),
            "work_function": self._task_work_function(source_task),
            "scope": (
                "Fix the rejected review findings for the linked orchestration gate. "
                "Keep the change scoped to the parent task deliverable."
            ),
            "inputs": [f"Parent task: {source_task.id}", f"Gate: {gate.id}", f"Rejected review evidence: {rejected.id}"],
            "deliverable": "A corrected work item ready for fresh review.",
            "forbidden_work": ["Do not mark orchestration gates complete.", "Do not edit unrelated artifacts."],
            "success_evidence": ["Complete the fix task and provide concrete evidence for a new review."],
            "budget": {},
            "report_schema": {"status": "done|blocked|failed", "changes": "list[str]", "evidence": "list[str]", "notes": "str"},
            "parent_task_id": str(source_task.id),
        }
        return await self._gate_recovery_trigger(
            db, run, gate, rejected.id, blocker_code="review_rejected",
            summary=f'Review rejected the work of task "{source_task.title}"; a fix task needs approval.',
            retry_action={"action_type": "create_delegation_task", "request": request},
            link={"kind": "review_fix", "gate_id": str(gate.id)},
        )

    async def _pre_plan_lines(self, db: AsyncSession, run: OrchestrationRun, tasks: Sequence[Task]) -> list[str]:
        """One planner-facing line per pre-plan task: id | function | status | gate | title [| report: status: next step]."""
        from huddleroom.services.orchestration_work_report import persisted_report_validation

        lines = []
        for pre in tasks:
            gate_ref = await self._task_gate_for_run(db, run.id, pre)
            marker = await self._existing_action_for_key(db, run.id, f"run:{run.id}:kind:report_consumed:task:{pre.id}")
            request = self._json_object_or_empty(marker.request) if marker is not None else {}
            report = persisted_report_validation(request.get("report"), request.get("report_validation_errors", [])).report
            line = (
                f"{pre.id} | {self._task_work_function(pre)} | {pre.status} | "
                f"gate {gate_ref[0].status if gate_ref else 'none'} | {pre.title}"
            )
            if report is not None:
                line += f" | report: {report.status}: {(report.recommended_next_step or '')[:120]}"
            lines.append(line)
        return lines

    async def _pre_plan_tasks(self, db: AsyncSession, run: OrchestrationRun) -> list[Task]:
        """Run work that exists before a plan is requested; the plan must adopt or explain each."""
        tasks = await self._orchestrated_tasks_for_run(
            db, run.id, statuses=[status for status in Task.VALID_TRANSITIONS if status != "cancelled"],
        )
        return [
            task for task in tasks
            if self._task_work_function(task) not in NON_ADOPTABLE_WORK_FUNCTIONS
        ]

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

    async def _task_start_blocked(self, db: AsyncSession, run: OrchestrationRun, task: Task):
        """Truthy while the run is stopped, a predecessor gate is unaccepted, or a meeting / graph run / live session holds the task."""
        # ponytail: re-checks per task; cache per run if queues grow
        return (
            await self._run_activation_stopped(db, run)
            or await first_unmet_dependency(db, run, task)
            or await self._blocked_task_awaiting(db, run, task)
        )

    async def _activation_blocker(self, db: AsyncSession, run: OrchestrationRun, task: Task):
        """_task_start_blocked, then effective authorization; a denial is parked on the task as authorization_wait.

        Authorization is checked last so stop/dependency/awaiting (and capacity, owned by activate_ready)
        keep their meaning. Only activation uses this: _release_headroom keeps _task_start_blocked, so a
        run-wide denial (e.g. Start stale) still pins the slot; a task-scoped denial (TASK_SCOPED_AUTH_DENIALS)
        parked by a previous activation frees it. Headroom stays write-free.
        """
        blocked = await self._task_start_blocked(db, run, task)
        if blocked:
            stale = self._json_object_or_empty(task.metadata_.get("orchestration"))
            if "authorization_wait" in stale:  # a stop/dependency block supersedes a parked authorization wait
                task.metadata_ = {**task.metadata_, "orchestration": {
                    k: v for k, v in stale.items() if k != "authorization_wait"}}
            return blocked
        goal = await db.get(OrchestrationGoal, run.goal_id)
        try:
            auth = await effective_authorization(db, goal, run, task)
            wait = None if auth.allowed else {
                "reason_code": auth.reason_code, "required_decision": auth.required_decision,
                "authority_source": auth.authority_source,
            }
        except HTTPException as exc:
            # Corrupt roadmap lineage (409) is a denial, not a tick crash.
            wait = {
                "reason_code": "lineage_invalid", "required_decision": "ask_human",
                "authority_source": "parent_release" if goal.parent_goal_id else "root_start",
                "detail": exc.detail,
            }
        meta = self._json_object_or_empty(task.metadata_.get("orchestration"))
        if meta.get("authorization_wait") != wait:
            meta = {key: value for key, value in meta.items() if key != "authorization_wait"}
            task.metadata_ = {**task.metadata_, "orchestration": {**meta, **({"authorization_wait": wait} if wait else {})}}
        if wait is not None:
            return f"authorization:{wait['reason_code']}", run.id
        scope = f"authorization:{task.id}"  # the claim boundary's blocker; the next allowed activation clears it
        if any(isinstance(b, Mapping) and b.get("scope") == scope for b in run.active_blockers or []):
            run.active_blockers = [
                b for b in run.active_blockers if not (isinstance(b, Mapping) and b.get("scope") == scope)
            ]
        return None

    async def _run_activation_stopped(self, db: AsyncSession, run: OrchestrationRun) -> str | None:
        """Why this run's own tick would refuse to spend; None when it may start work."""
        if await self._hard_stop_active(db, run.goal_id):
            return "hard_stop"
        if self._budget_is_exhausted(run):
            return "budget_exhausted"
        goal = await db.get(OrchestrationGoal, run.goal_id)
        child_delta_ready = (
            goal is not None and goal.parent_goal_id is not None
            and self._json_object_or_empty(
                self._json_object_or_empty(run.plan_state).get("child_delta_baseline")
            ).get("status") == "accepted"
        )
        for kind in (
            BUDGET_REQUIRED_CODE,
            "repeated_failure", "goal_definition_analyzer_error", "agent_definition_review_analyzer_error",
            "plan_criterion_integrity", "budget_exhausted",
        ):
            if kind == "agent_definition_review_analyzer_error" and child_delta_ready:
                continue
            if self._has_active_blocker(run, kind):
                return kind
        return None

    async def _start_released_tasks(self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun) -> int:
        """Start never-started backlog tasks the orchestrator delegated (nothing else runs them)."""
        if goal.status != "active" or run.status != "running" or run.phase not in {"baseline", "authorized"}:
            return 0
        if run.phase == "authorized":
            # Best-ranked startable tasks of the whole root tree, into the root's free slots.
            result = await activate_ready(db, await root_goal_id(db, goal.id), is_blocked=self._activation_blocker)
            await db.flush()
            await db.refresh(run)  # a refused start rolled back a savepoint, which expires run/goal
            await db.refresh(goal)
            return len(result["started"])
        started = 0
        # ponytail: pre-Start baseline delivery is not capacity-gated (it is still counted by available_slots).
        for task in await self._orchestrated_tasks_for_run(db, run.id, statuses=["backlog"]):
            if task.assigned_to is None:
                continue
            # ponytail: only never-started tasks; recovery/retry/reassign own tasks that already had a session.
            if await db.scalar(select(Session.id).where(Session.task_id == task.id).limit(1)) is not None:
                continue
            if await first_unmet_dependency(db, run, task) is not None:  # predecessor gate not accepted
                continue
            if await self._blocked_task_awaiting(db, run, task) is not None:  # meeting / graph run
                continue
            await TaskService().auto_start(db, task)
            if task.status == "in_progress":
                started += 1
            else:  # refused start rolled back a savepoint, which expires run/goal; reload for tick's later reads
                await db.refresh(run)
                await db.refresh(goal)
        return started

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
            try:
                text, how = normalize_result_output(session.output)
            except ValueError:
                text, how = session.output, None  # strict parse below reports the precise error
            payload = json.loads(text)
        except (TypeError, ValueError):
            return None, "Final summary output must be valid JSON"
        if how and self._json_object_or_empty(summary_evidence.evidence_metadata).get("normalized") != how:
            summary_evidence.evidence_metadata = {
                **self._json_object_or_empty(summary_evidence.evidence_metadata), "normalized": how,
            }
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
        selected, failure = await self._gate_validation_result(db, gate, update_required_evidence=True)
        if failure is not None:
            reason, final = failure
            if final:
                self._fail_gate(gate, reason)
                return True
            gate.failure_reason = reason
            return False
        self._accept_gate(gate, selected)
        return True

    async def _gate_validation_result(
        self, db: AsyncSession, gate: OrchestrationGate, *, update_required_evidence: bool = False,
    ) -> tuple[list[OrchestrationEvidence], tuple[str, bool] | None]:
        """Evaluate the ordinary evidence predicate without accepting or failing rows."""
        if gate.gate_type == RETAINED_GATE_TYPE:
            # Provenance only: no evidence rows, and not a plan-item gate, so it never satisfies criteria.
            if failure := await self._retained_gate_failure(db, gate):
                return [], failure
            return [], None
        evidence = await self._evidence_for_gate(db, gate)
        latest_evidence = self._latest_evidence_by_source(evidence)
        rejected = next(
            (item for item in latest_evidence if item.verdict == "rejected" and not self._superseded_attempt(item, evidence)),
            None,
        )
        if rejected is not None:
            return [], (f"Rejected evidence from {rejected.source_type}", True)

        required_evidence = await self._required_evidence_for_gate(db, gate, update=update_required_evidence)
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
            return selected, (f"Missing evidence: {', '.join(missing)}", False)
        if len(selected) < min_count:
            return selected, ("Missing evidence", False)
        if gate.gate_type == ADHOC_GATE_TYPE and (failure := await self._adhoc_report_failure(db, gate)):
            return selected, failure

        run = await db.get(OrchestrationRun, gate.run_id)
        if (
            run is not None
            and (
                await self._is_outcome_plan_item_gate(db, run, gate)
                or gate.gate_type == "roadmap_integration"
            )
            and not await self._has_bound_outcome_verification_evidence(db, gate)
        ):
            return selected, ("Independent verification evidence is missing", False)

        if gate.gate_type == FINAL_SUMMARY_GATE_TYPE:
            summary_failure = await self._final_summary_evidence_failure(db, gate, selected)
            if summary_failure is not None:
                return selected, (summary_failure, True)

        stale_reason = self._stale_evidence_reason(selected)
        if stale_reason is not None:
            return selected, (stale_reason, True)

        independence_reason = await self._independence_failure_reason(db, gate, selected, required_evidence=required_evidence)
        if independence_reason is not None:
            if gate.gate_type == "roadmap_integration":
                return selected, (independence_reason, False)
            run = await db.get(OrchestrationRun, gate.run_id)
            if run is not None and await self._is_outcome_plan_item_gate(db, run, gate):
                goal = await db.get(OrchestrationGoal, run.goal_id)
                if goal is not None and goal.goal_type == "roadmap":
                    return selected, (independence_reason, False)
            return selected, (independence_reason, True)

        return selected, None

    async def _adhoc_report_failure(self, db: AsyncSession, gate: OrchestrationGate) -> tuple[str, bool] | None:
        """(reason, final) unless the done task's consumed report is valid with evidence.

        An unusable report is final (the gate fails) unless a result correction is still pending."""
        from huddleroom.services.orchestration_work_report import persisted_report_validation

        task_id = self._optional_uuid(self._json_object_or_empty(gate.required_evidence).get("task_id"), "task_id")
        task = await db.get(Task, task_id) if task_id is not None else None
        if task is None or task.status != "done":
            return "Task is not completed", False
        own = await self._existing_action_for_key(db, gate.run_id, f"run:{gate.run_id}:kind:report_consumed:task:{task.id}")
        ledger = self._json_object_or_empty(self._json_object_or_empty(own.request if own else {}).get("result_correction"))
        clarification = await self._existing_action_for_key(
            db, gate.run_id, f"run:{gate.run_id}:kind:report_clarification:task:{task.id}"
        )
        corrections = [self._optional_uuid(item, "correction_task_id") for item in ledger.get("correction_task_ids", [])]
        sources = [task.id, *(corrections or [clarification.target_id if clarification is not None else None])]
        markers = [
            await self._existing_action_for_key(db, gate.run_id, f"run:{gate.run_id}:kind:report_consumed:task:{source}")
            if source is not None else None
            for source in sources
        ]
        for index, marker in enumerate(markers):
            if index == 0 and ledger.get("status") in {"pending", "correcting"}:
                continue  # the original's report is under correction, so it is not the effective one yet
            request = self._json_object_or_empty(marker.request) if marker is not None else {}
            report = persisted_report_validation(request.get("report"), request.get("report_validation_errors", [])).report
            if report is not None:
                return None if report.candidate_evidence else ("Task report has no evidence", True)
        if markers[0] is None:
            return "Task report is missing", False
        if ledger:  # the correction ledger decides: final only once correction has stopped
            return "Task report is invalid", ledger.get("status") not in {"pending", "correcting"}
        repair = await db.get(Task, clarification.target_id) if clarification and clarification.target_id else None
        if repair is not None and repair.status in {"failed", "cancelled"}:
            dispatched = bool(self._json_object_or_empty(markers[0].request).get("clarification_count")) or (
                await db.scalar(select(Session.id).where(Session.task_id == repair.id).limit(1)) is not None
            )
            return "Task report is invalid", dispatched
        return "Task report is invalid", False if repair is None or markers[1] is None else True

    async def _retained_gate_failure(self, db: AsyncSession, gate: OrchestrationGate) -> tuple[str, bool] | None:
        """(reason, final) unless the retained source still holds.

        The source gate must still be accepted on an earlier run of this run's correction lineage, its task and
        Artifact must exist in the project, and the contract must differ from the source Start by type alone.
        Acceptance waits for this run's own Start so a baseline rerun cannot rewrite the contract afterwards."""
        from huddleroom.services.orchestration_budget_service import run_lineage

        required = self._json_object_or_empty(gate.required_evidence)
        run = await db.get(OrchestrationRun, gate.run_id)
        goal = await db.get(OrchestrationGoal, run.goal_id)
        source_gate_id, task_id, artifact_id = (
            _uuid(required.get(key)) for key in ("source_gate_id", "source_task_id", "artifact_id")
        )
        source_gate = await db.get(OrchestrationGate, source_gate_id) if source_gate_id is not None else None
        if source_gate is None or source_gate.status != "accepted" or str(source_gate.run_id) != required.get(
            "source_run_id"
        ):
            return "Retained source gate is no longer accepted", True
        source_run = next((item for item in (await run_lineage(db, run))[1:] if item.id == source_gate.run_id), None)
        if source_run is None:
            return "Retained source run is not superseded by this run", True
        task = await db.get(Task, task_id) if task_id is not None else None
        if task is None or task.project_id != goal.project_id:
            return "Retained source task is missing", True
        artifact = await db.get(Artifact, artifact_id) if artifact_id is not None else None
        if artifact_id is not None and (artifact is None or artifact.project_id != goal.project_id):
            return "Retained artifact is missing", True
        if (
            required.get("source_contract_fingerprint") != source_run.goal_contract_fingerprint
            or not OrchestrationGoalShapeService.contract_unchanged_since_start(goal, source_run)
        ):
            return "Goal requirements changed since the retained source Start", True
        if run.goal_contract_fingerprint is None:
            return "Waiting for Start to freeze the goal contract", False
        return None

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

    def _superseded_attempt(
        self, item: OrchestrationEvidence, evidence: list[OrchestrationEvidence]
    ) -> bool:
        """Rejected attempt evidence is superseded by a later successful attempt of the same task and source
        type (recovery retry/reassign/resume). Rows are never rewritten: recorded verdicts stay as recorded."""
        if item.verdict != "rejected" or item.source_type not in ATTEMPT_SOURCE_TYPES:
            return False
        task_id = self._json_object_or_empty(item.evidence_metadata).get("task_id")
        if not isinstance(task_id, str) or not task_id:
            return False
        key = self._evidence_order_key(item)
        return any(
            o is not item
            and o.source_type == item.source_type
            and o.verdict in {"candidate", "accepted"}
            and self._json_object_or_empty(o.evidence_metadata).get("task_id") == task_id
            and self._evidence_order_key(o) > key
            for o in evidence
        )

    def _evidence_order_key(self, evidence: OrchestrationEvidence) -> tuple[bool, int, datetime, str]:
        seq = self._evidence_event_seq(evidence)
        return (seq is None, seq or 0, evidence.created_at, str(evidence.id))

    async def _required_evidence_for_gate(
        self, db: AsyncSession, gate: OrchestrationGate, *, update: bool = True,
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
        if update and normalized != required_evidence:
            gate.required_evidence = normalized
        return normalized

    async def _independence_failure_reason(
        self,
        db: AsyncSession,
        gate: OrchestrationGate,
        evidence: list[OrchestrationEvidence],
        *, required_evidence: dict[str, Any] | None = None,
    ) -> str | None:
        if required_evidence is None:
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
    def _reopen_gate(gate: OrchestrationGate) -> None:
        gate.status = "open"
        gate.failure_reason = None
        gate.failed_at = None
        gate.accepted_at = None

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

    async def _reject_stale_steering_decision(
        self, db, decision_id: uuid.UUID, *, reason: str = "stale_steering_versions",
    ) -> None:
        """Durably reject stale work without racing SQLite's sole writer."""
        statement = (
            update(OrchestrationDecision)
            .where(OrchestrationDecision.id == decision_id)
            .values(validator_status="rejected", rejection_reason=reason)
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

            if not isinstance(exc, (SteeringVersionsChanged, PlanGenerationChanged)):
                raise
            await self._reject_stale_steering_decision(
                db, decision.id,
                reason="stale_plan_generation" if isinstance(exc, PlanGenerationChanged) else "stale_steering_versions",
            )
            if stale_retry:
                await self._create_noop_waits(db, run, decision)
                return None
            else:
                refreshed = await self.request_llm_decision(db, run.id)
                return await self._dispatch_execution_decision(
                    db, run, refreshed, stale_retry=True,
                )
        if action is not None:
            if action.action_type == "noop" or action.status == "failed":
                await self._create_noop_waits(db, run, decision)
            if (action.action_type == "noop" and decision.validator_status == "accepted"
                    and self._json_object_or_empty(decision.parsed_decision).get("action_type") == "noop"):
                await self._eventless_noop_trigger(db, run, decision)
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
        action = await self.execute_noop_action(
            db,
            run_id=run.id,
            request={"action_type": "noop", "reason": reason},
            idempotency_key=dispatcher.action_key(run.id, "noop", wait_identity),
            decision_id=decision.id,
        )
        await self._create_noop_waits(db, run, decision)
        return action

    async def _create_noop_waits(self, db, run, decision) -> None:
        """A noop waits on its wake_when; any other or unusable outcome gets the backstop wait."""
        parsed = self._json_object_or_empty(decision.parsed_decision)
        wake_when = (
            parsed.get("wake_when")
            if decision.validator_status == "accepted" and parsed.get("action_type") == "noop"
            else None
        )
        await self.supervision.create_orchestrator_waits(db, run, owner_id=decision.id, wake_when=wake_when)

    async def _advance_authorized_execution(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun
    ) -> dict:
        return await self._act_until_wait(
            db, goal, run, self._advance_authorized_execution_once,
            loop_steps=frozenset({"plan_decision", "next_action"}),
        )

    async def _act_until_wait(self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun, step_once, *, loop_steps) -> dict:
        """Repeat step_once while each step dispatched a real, completed, non-noop
        action and the run is still runnable, up to orchestration_max_actions_per_tick.
        Each iteration re-reads context via request_llm_decision. The returned `step`
        and extra keys come from the last iteration; `action_id` is the first
        dispatched id and `action_ids` the full list.
        # ponytail: on non-SQLite the isolated context read can't see the uncommitted
        # action, so iteration 2 reuses the decision -> same action id -> loop stops
        # (one action per tick). Upgrade path: commit before each re-decision.
        Errors from any iteration propagate so the caller's transaction rolls back (no partial
        writes of the failed iteration). On SQLite a tick-owned autobegin transaction is
        pre-committed by request_llm_decision, so earlier iterations stay durable; executors are
        idempotent by key, so a retry reuses their effects."""
        cap = max(1, settings.orchestration_max_actions_per_tick)
        action_ids: list[str] = []
        iterations = 0
        result: dict = {}
        hit_cap = False
        while iterations < cap:
            iterations += 1
            result = await step_once(db, goal, run)
            if result.get("reused"):
                break  # already-accepted plan re-dispatched: not progress
            aid = result.get("action_id")
            if aid in action_ids:
                break
            if aid:
                action_ids.append(aid)
            if result.get("step") not in loop_steps or not aid:
                break
            action = await db.get(OrchestrationAction, uuid.UUID(str(aid)))
            if action is None or action.action_type == "noop" or action.status != "completed":
                break
            if action.action_type in _LOOP_STOP_ACTIONS:
                break
            if action.action_type == "accept_plan" and goal.goal_type == "roadmap":
                # roadmap accept writes no snapshot; the roadmap stepper takes over next tick
                break
            await db.refresh(run)
            await db.refresh(goal)
            if (
                run.status not in LLM_DECISION_RUN_STATUSES
                or run.phase != "authorized"
                or goal.status not in {"active", "blocked"}
            ):
                break
        else:
            # ponytail: loop was still productive; parking a wait would stall it
            hit_cap = True
        if action_ids:
            if not hit_cap:
                await self._post_action_wait(db, goal, run, action_ids, result, loop_steps)
        result = {k: v for k, v in result.items() if k != "reused"}
        if iterations == 1 or not action_ids:
            return result
        return {**result, "action_id": action_ids[0], "action_ids": action_ids}

    async def _post_action_wait(self, db, goal, run, action_ids, result, loop_steps) -> None:
        """After a real action ended the loop, park a short orchestrator wait so an action that
        did not remove its trigger cannot re-call the LLM on every reconcile tick."""
        from huddleroom.models.orchestration import OrchestrationWait
        from huddleroom.services.orchestration_wake_when import ORCHESTRATOR_WAIT_OWNER_TYPE

        if result.get("step") not in loop_steps:
            return
        action = await db.get(OrchestrationAction, uuid.UUID(str(action_ids[-1])))
        if (
            action is None
            or action.status != "completed"
            or action.action_type == "noop"
            or action.action_type in _LOOP_STOP_ACTIONS
            or (action.action_type == "accept_plan" and goal.goal_type == "roadmap")
            or self._json_object_or_empty(run.plan_state).get("status") != "accepted"
        ):
            return
        waits = (await db.scalars(select(OrchestrationWait).where(
            OrchestrationWait.run_id == run.id, OrchestrationWait.status == "open"
        ))).all()
        from huddleroom.services.orchestration_supervision import SYSTEM_WAIT_OWNER_TYPES

        owner_types = {(w.owner or {}).get("type") for w in waits}
        if ORCHESTRATOR_WAIT_OWNER_TYPE in owner_types or not owner_types & SYSTEM_WAIT_OWNER_TYPES:
            return
        # ponytail: only on the decide-while-work path (open system waits); a task created this tick has no
        # system wait yet, so one extra LLM call can occur. One post-action wait per tick, fixed recheck =
        # reconcile interval; per-item tracking if needed
        await self.supervision.create_orchestrator_waits(
            db, run, owner_id=action.id,
            wake_when={
                "recheck_after_seconds": settings.orchestration_reconcile_interval_seconds,
                "expected_result": "Post-action recheck",
            },
        )

    @staticmethod
    def _waiting_step_for_http_error(exc: HTTPException) -> dict | None:
        """Map a recoverable dispatch/decision HTTP error to an explicit wait step."""
        from huddleroom.services.orchestration_roadmap_service import RoadmapReplanMeasurementAttention

        if isinstance(exc, RoadmapReplanMeasurementAttention):
            return {"step": "waiting", "reason": "needs_attention"}
        if exc.status_code == 409 and exc.detail in {"budget_wait", "needs_attention"}:
            return {"step": "waiting", "reason": exc.detail}
        return None

    async def _advance_authorized_execution_once(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun
    ) -> dict:
        """One forward step of the authorized Outcome runner. Exactly one action
        (or one explicit wait) per call. Code releases deterministic work; the
        LLM only chooses when code cannot."""
        from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher

        if self._has_active_blocker(run, BUDGET_REQUIRED_CODE):
            return {"step": "needs_attention", "reason": BUDGET_REQUIRED_CODE}

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
                    "fingerprint": await roadmap.plan_fingerprint(db, run, [
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
                if (waiting := self._waiting_step_for_http_error(exc)) is not None:
                    return waiting
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
        try:
            decision = await self.request_llm_decision(db, run.id)
            action = await self._dispatch_execution_decision(db, run, decision)
        except HTTPException as exc:
            if (waiting := self._waiting_step_for_http_error(exc)) is not None:
                return waiting
            if exc.status_code == 409 and exc.detail == "Orchestration run is paused":
                return {"step": "waiting", "reason": "paused"}  # concurrent pause during the LLM call
            raise
        if action is None:
            return {"step": "waiting", "reason": "stale_steering_versions"}
        if (
            action.action_type == "accept_plan"
            and action.status == "completed"
            and plan_state.get("accept_action_id") == str(action.id)
        ):
            # Plan was already accepted before this dispatch: replayed accept, not progress.
            await self.supervision.create_orchestrator_waits(
                db, run, owner_id=decision.id,
                wake_when={
                    "recheck_after_seconds": settings.orchestration_reconcile_interval_seconds,
                    "expected_result": "Replayed accept_plan recheck",
                },
            )
            return {"step": "next_action", "action_id": str(action.id), "reused": True}
        return {"step": "next_action", "action_id": str(action.id)}

    async def _release_headroom(self, db: AsyncSession, root_id: uuid.UUID) -> int:
        """Free root slots minus startable queued tasks; dependency-blocked backlog holds no slot."""
        startable = 0
        for task in await queued_tasks(db, root_id):
            run = await db.get(OrchestrationRun, uuid.UUID(task.metadata_["orchestration"]["run_id"]))
            if await self._task_start_blocked(db, run, task):
                continue
            # ponytail: uses the wait parked by the last activation; a fresh denial holds its slot for one tick.
            meta = self._json_object_or_empty(task.metadata_.get("orchestration"))
            wait = self._json_object_or_empty(meta.get("authorization_wait"))
            if wait.get("reason_code") not in TASK_SCOPED_AUTH_DENIALS:
                startable += 1
        return await available_slots(db, root_id) - startable

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

        # Adopted items bind already-running/finished work: no new task, so no slot is spent.
        bound = 0
        for item in items:
            if item.adopt_task_id is None:
                continue
            if await self._release_item_once(db, goal, run, item):
                bound += 1

        slots = await self._release_headroom(db, await root_goal_id(db, goal.id))
        if slots <= 0:
            return bound

        ready = [
            item for item in items
            if item.adopt_task_id is None and await self._item_dependencies_accepted(db, run, item)
        ]
        released = 0
        for item in rank_queued(
            ready, {item.id: index for index, item in enumerate(items)}, dependency_counts(items)
        ):
            if released >= slots:
                break
            if await self._release_item_once(db, goal, run, item):
                released += 1
        return bound + released

    async def _release_item_once(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun, item: OrchestrationPlanItem
    ) -> bool:
        key = f"run:{run.id}:kind:release_item:{item.id}"
        released = await self._existing_action_for_key(db, run.id, key) is not None
        if released:
            state = self._json_object_or_empty(run.plan_state)
            entry = next((entry for entry in state.get("expanded_items", []) if isinstance(entry, Mapping)
                          and entry.get("plan_item_id") == item.id), None)
            if entry is None or entry.get("accepted_plan_fingerprint") == self._current_plan_fingerprint(run):
                return False
            task_id = self._optional_uuid(entry.get("task_id"), "task_id")
            task = await db.get(Task, task_id) if task_id else None
            metadata = self._json_object_or_empty(self._json_object_or_empty(task.metadata_ if task else None).get(
                "orchestration_plan_item"
            ))
            if (not entry.get("accepted_plan_fingerprint")
                    and metadata.get("accepted_plan_artifact_id") == state.get("accepted_artifact_id")):
                return False
        expand_key = self._expand_plan_item_idempotency_key(run.id, item.id, self._current_plan_fingerprint(run))
        try:
            action = await self.execute_expand_plan_item_action(
                db,
                run_id=run.id,
                request={"action_type": "expand_plan_item", "plan_item_id": item.id,
                         "work_function": item.work_function},
                idempotency_key=expand_key,
            )
        except HTTPException as exc:
            if item.adopt_task_id is None or exc.status_code != 409:
                raise
            action = await self._existing_action_for_key(db, run.id, expand_key)
        if item.adopt_task_id is not None and (action is None or action.status != "completed"):
            # A binding that cannot complete (conflicting adoption, vanished task) needs a human, not a release.
            self._block_plan_integrity(
                goal, run, f"Adopted plan item '{item.id}' could not bind: {action.error if action else 'unknown'}"
            )
            return False
        if released:
            return False
        # expand_plan_item creates (or, for adoption, binds) this item's task; mark the
        # release so the cap accounting and idempotency hold across ticks.
        await self.reserve_action(
            db, run_id=run.id, idempotency_key=key,
            action_type="release_item", request={"plan_item_id": item.id},
        )
        return True

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

    async def _clear_obsolete_authorization_blockers(self, db: AsyncSession, run: OrchestrationRun) -> None:
        """Keep claim blockers only while their tasks are waiting to start."""
        scoped_tasks = {}
        for blocker in run.active_blockers or []:
            scope = blocker.get("scope") if isinstance(blocker, dict) else None
            if isinstance(scope, str) and scope.startswith("authorization:"):
                task_id = self._event_uuid(scope.removeprefix("authorization:"))
                if task_id is not None:
                    scoped_tasks[scope] = task_id
        if not scoped_tasks:
            return
        waiting_ids = set((await db.scalars(select(Task.id).where(
            Task.id.in_(scoped_tasks.values()), Task.status.in_(("backlog", "queued")),
        ))).all())
        run.active_blockers = [
            blocker for blocker in run.active_blockers or []
            if not isinstance(blocker, dict)
            or blocker.get("scope") not in scoped_tasks
            or scoped_tasks[blocker["scope"]] in waiting_ids
        ]

    async def tick(self, db: AsyncSession, run_id: uuid.UUID, *, local_only: bool = False) -> dict:
        from huddleroom.services.orchestration_roster_mutation import RosterHandoffRequired

        try:
            if not local_only:
                # B1: an approved fix progresses (apply/verify/retry) before the decision loop can re-select the
                # blocked action; new triggers are diagnosed after the tick, below.
                await self._drive_recovery(db, run_id, diagnose=False)
            try:
                result = await self._tick_baseline_impl(db, run_id, local_only=local_only)
            except RosterHandoffRequired as handoff:
                # The reservation was published under the goal lock, which is now
                # released: apply roster-first, then a fresh tick resumes the goal.
                from huddleroom.services.agent_service import complete_roster_handoff

                result = await complete_roster_handoff(
                    db, handoff.prepared,
                    lambda _prepared: self._tick_baseline_impl(db, run_id, local_only=local_only),
                )
        except HTTPException as exc:
            if exc.status_code == 409 and exc.detail == "Baseline operation discarded by reset":
                return {"run_id": str(run_id), "status": "interrupted", "interrupted": True}
            raise
        if not local_only:  # local-only passes are provider-free by contract
            await self._drive_recovery(db, run_id)
        return result

    @asynccontextmanager
    async def _lock_recovery_lifecycle(self, db: AsyncSession, run_id: uuid.UUID, owns: bool):
        """Publish a tick-owned recovery pass before another pass reads its proposals.

        Caller-owned transactions retain their existing savepoint/publication semantics;
        their caller remains responsible for serialization through its outer commit.
        This mutex is independent of baseline/roster locks and never covers diagnosis.
        """
        if not owns or db.get_bind().dialect.name != "sqlite":
            yield
            return
        goal_id = await db.scalar(select(OrchestrationRun.goal_id).where(OrchestrationRun.id == run_id))
        await db.commit()  # a waiter must not keep a SQLite writer open
        loop_locks = _RECOVERY_LIFECYCLE_LOCKS.setdefault(asyncio.get_running_loop(), {})
        lock = loop_locks.setdefault(goal_id, asyncio.Lock())
        async with lock:
            try:
                yield
            finally:
                await db.commit()  # lifecycle updates must publish before the mutex is released

    async def _drive_recovery(self, db: AsyncSession, run_id: uuid.UUID, *, diagnose: bool = True) -> None:
        """Outside the tick's goal lock: resume every active proposal (apply/verify/retry progress,
        delegated-remediation waits, in-wrapper retries), then diagnose reserved recovery triggers.
        Approved proposals go first so a parallel trigger is never diagnosed ahead of an approved fix.

        A tick-owned session commits before each step, so a trigger is durable and no SQLite write
        transaction is held across the analyzer call. A domain failure (flushed only) is committed; any
        other error rolls that step back and leaves the trigger/proposal for the next tick."""
        from huddleroom.services.orchestration_blocker_recovery import (
            OrchestrationBlockerRecoveryService, RecoveryDiagnosisFailed, RecoveryProposalStale, _RunFenced,
        )

        transaction = db.sync_session.get_transaction()
        owns = not db.in_nested_transaction() and (
            transaction is None or transaction.origin == SessionTransactionOrigin.AUTOBEGIN)

        async def ids(action_type: str) -> list[uuid.UUID]:
            return list(await db.scalars(select(OrchestrationAction.id).where(
                OrchestrationAction.run_id == run_id, OrchestrationAction.action_type == action_type,
                OrchestrationAction.status == "reserved",
            ).order_by(OrchestrationAction.created_at, OrchestrationAction.id)))

        async def step(call, on_error=None) -> None:
            if owns:
                await db.commit()
                try:
                    await call()
                except (RecoveryDiagnosisFailed, RecoveryProposalStale):
                    pass
                except _RunFenced:
                    await db.rollback()
                except Exception:  # noqa: BLE001 -- one recovery step never fails the tick
                    await db.rollback()
                    logger.warning("recovery step failed for run %s", run_id, exc_info=True)
                    if on_error is not None:
                        await on_error()
                await db.commit()
                return
            try:
                async with db.begin_nested():
                    try:
                        await call()
                    except (RecoveryDiagnosisFailed, RecoveryProposalStale):
                        pass
            except _RunFenced:
                pass
            except Exception:  # noqa: BLE001
                logger.warning("recovery step failed for run %s", run_id, exc_info=True)
                if on_error is not None:
                    await on_error()

        def naive(value):
            if value is None:
                return datetime.min
            return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value

        async def next_trigger() -> uuid.UUID | None:
            """Least recently attempted reserved trigger, so one that keeps failing never starves the others."""
            triggers = list(await db.scalars(select(OrchestrationAction).where(
                OrchestrationAction.run_id == run_id, OrchestrationAction.action_type == "recovery_trigger",
                OrchestrationAction.status == "reserved",
            )))
            if not triggers:
                return None
            attempted: dict[str, datetime] = {}
            for exchange in await db.scalars(select(OrchestrationDecision).where(
                    OrchestrationDecision.run_id == run_id, OrchestrationDecision.decision_type == "recovery")):
                key = str((exchange.input_snapshot or {}).get("trigger_action_id"))
                attempted[key] = max(attempted.get(key, datetime.min), naive(exchange.created_at))

            def error_at(row):
                value = (row.dispatch_contract or {}).get("last_diagnosis_error_at")
                return naive(datetime.fromisoformat(value)) if value else datetime.min

            def last_attempt(row):
                return max(attempted.get(str(row.id), datetime.min), error_at(row))
            # F3: back off a trigger whose diagnosis just errored (provider outage), instead of a call per tick.
            cutoff = naive(_utcnow()) - timedelta(seconds=RECOVERY_DIAGNOSIS_ERROR_BACKOFF_SECONDS)
            triggers = [row for row in triggers if error_at(row) <= cutoff]
            if not triggers:
                return None
            return min(triggers, key=lambda row: (last_attempt(row), naive(row.created_at), str(row.id))).id

        async def mark_diagnosis_error(trigger_id: uuid.UUID) -> None:
            row = await db.get(OrchestrationAction, trigger_id)
            if row is not None and row.status == "reserved":
                row.dispatch_contract = {**(row.dispatch_contract or {}),
                                         "last_diagnosis_error_at": _utcnow().isoformat()}
                await db.flush()

        try:
            service = OrchestrationBlockerRecoveryService()
            async with self._lock_recovery_lifecycle(db, run_id, owns):
                status = await db.scalar(select(OrchestrationRun.status).where(OrchestrationRun.id == run_id))
                if status not in LLM_DECISION_RUN_STATUSES:
                    return
                for proposal_id in await ids("recovery_proposal"):
                    await step(lambda proposal_id=proposal_id: service.resume(db, proposal_id))
            # Diagnosis waits for the provider without a lifecycle, goal, or roster lock.
            # Bounded: at most one analyzer diagnosis per run per tick, least recently attempted first.
            trigger_id = await next_trigger() if diagnose else None
            if trigger_id is not None:
                active = _RECOVERY_DIAGNOSES.setdefault(asyncio.get_running_loop(), set())
                # No await between checking and inserting: a losing tick skips,
                # rather than queuing another logical analysis behind this one.
                if trigger_id in active:
                    return
                active.add(trigger_id)
                try:
                    # Directly await diagnosis so cancellation cannot release
                    # admission while an analyzer is still unwinding.
                    await step(lambda: service.diagnose(db, trigger_id, tick_owns_transaction=owns),
                               on_error=lambda: mark_diagnosis_error(trigger_id))
                finally:
                    active.discard(trigger_id)
        finally:
            if owns:
                await db.commit()  # the tick leaves no transaction open behind it

    async def _tick_baseline_impl(self, db: AsyncSession, run_id: uuid.UUID, *, local_only: bool = False) -> dict:
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

        from huddleroom.services.orchestration_baseline_runtime import baseline_operation, require_baseline_available
        from huddleroom.services.orchestration_roster_mutation import commit_before_roster_handoff
        require_baseline_available(goal_id)
        # Lock goal-level to serialize concurrent ticks and force-starts on the same goal.
        # This prevents: (a) baseline_ready stale-gate race; (b) concurrent completion race.
        async with self._lock_goal_for_baseline_transition(
            db,
            goal_id,
            tick_owns_transaction=not caller_owns_transaction and not db.in_nested_transaction(),
        ), baseline_operation(
            db, goal_id, owns_transaction=not caller_owns_transaction and not db.in_nested_transaction(), tick=True
        ), commit_before_roster_handoff(
            db, not caller_owns_transaction and not db.in_nested_transaction()
        ):
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
            await self._clear_obsolete_authorization_blockers(db, run)
            if self._has_active_blocker(run, BUDGET_REQUIRED_CODE):
                return {**self._tick_noop_result(run), "authorized_execution": {
                    "step": "needs_attention", "reason": BUDGET_REQUIRED_CODE,
                }}
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

            if goal.parent_goal_id is not None and self._json_object_or_empty(
                self._json_object_or_empty(run.plan_state).get("child_delta_baseline")
            ).get("status") == "pending_renewal":
                if not caller_owns_transaction:
                    await db.commit()
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

            # Legacy authorized root runs (no frozen fingerprint) freeze the contract on first tick.
            if (
                run.phase == "authorized"
                and goal.parent_goal_id is None
                and run.goal_contract_fingerprint is None
            ):
                run.goal_contract_fingerprint = compute_goal_contract_fingerprint(goal)
                await db.flush()

            new_events = await self._new_events(db, project_id, run.event_cursor)
            new_cursor = new_events[-1].seq if new_events else run.event_cursor
            cursor_token = str(new_cursor) if new_cursor is not None else "genesis"
            evidence_created = await self._ingest_evidence_from_events(db, run, new_events)
            await self._sweep_result_corrections(db, run)
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
            if self._has_active_blocker(run, "plan_criterion_integrity") or self._has_active_blocker(run, BUDGET_REQUIRED_CODE):
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
                    elif await self.recovery_holds_decision_loop(db, run.id):
                        authorized_execution = {"step": "recovery_in_progress"}
                    elif goal.goal_type == "roadmap":
                        from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService

                        authorized_execution = await OrchestrationRoadmapService(self).advance(db, goal, run)
                    else:
                        authorized_execution = await self._advance_authorized_execution(db, goal, run)
                await db.flush()  # refresh below discards unflushed attribute writes (e.g. supervision_state)
                await db.refresh(run)
                await db.refresh(goal)

            can_finish = (
                run.status in LLM_DECISION_RUN_STATUSES
                and goal.status in {"active", "blocked"}
                and run.phase == "authorized"          # NEW: no completion before Start
                and baseline_ready
                and not any(
                    isinstance(blocker, Mapping)
                    and blocker.get("kind") not in {"everyone_idle", "supervision_judgment_failures"}
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

            await self._start_released_tasks(db, goal, run)

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
