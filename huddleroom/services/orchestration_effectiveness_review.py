"""Effectiveness-review value objects."""

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.config import settings
from huddleroom.models.agent import Agent
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationEvidence,
    OrchestrationGate,
    OrchestrationGoal,
    OrchestrationRun,
)
from huddleroom.models.orchestration_process import (
    OrchestrationAuthorityDecision,
    OrchestrationProcessRun,
    OrchestrationWarning,
)
from huddleroom.models.project import Project
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.models.user import User
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_effectiveness_analyzer import (
    EffectivenessAnalyzer,
    EffectivenessAnalysis,
)
from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService
from huddleroom.services.orchestration_agent_definition_analyzer import redact_semantic_payload
from huddleroom.services.orchestration_llm_decision_adapter import _safe_completion_error

PROCESS_TYPE = "effectiveness_review"
LIVE_TASK_STATUSES = ("backlog", "ready", "in_progress", "blocked")
MEMORY_SECTION_KEY = "effectiveness_review"
MEMORY_TOC_ORDER = 50
# Actions that do not count as substantive progress for the inactivity trigger.
NON_PROGRESS_ACTION_TYPES = (
    "noop",
    "record_warning",
    "acknowledge_warning",
    "resolve_warning",
    "pause_run",
    "ask_human",
    "suggest_agent",
    "decision_continuation",
)


@dataclass(frozen=True)
class TriggerReason:
    name: str
    token: str
    detail: str


@dataclass(frozen=True)
class Recommendation:
    disposition: str
    detail: str


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def serialize_triggers(triggers: Iterable[TriggerReason]) -> list[dict[str, str]]:
    return [{"name": trigger.name, "token": trigger.token, "detail": trigger.detail} for trigger in triggers]


def serialize_recommendations(recommendations: Iterable[Recommendation]) -> list[dict[str, str]]:
    return [{"disposition": recommendation.disposition, "detail": recommendation.detail} for recommendation in recommendations]


def serialize_checks(checks: Iterable[CheckResult]) -> list[dict[str, str | bool]]:
    return [{"name": check.name, "passed": check.passed, "detail": check.detail} for check in checks]


def _render_review_memory(
    triggers: Iterable[TriggerReason],
    checks: Iterable[CheckResult],
    recommendations: Iterable[Recommendation],
    decision: dict | None,
) -> str:
    return json.dumps(
        {
            "triggers": serialize_triggers(triggers),
            "checks": serialize_checks(checks),
            "recommendations": serialize_recommendations(recommendations),
            "decision": decision,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


async def upsert_review_memory(
    db: AsyncSession,
    goal: OrchestrationGoal,
    run: OrchestrationRun,
    triggers: Iterable[TriggerReason],
    checks: Iterable[CheckResult],
    recommendations: Iterable[Recommendation],
    *,
    decision: dict | None = None,
):
    triggers = list(triggers)
    checks = list(checks)
    recommendations = list(recommendations)
    summary = recommendations[0].detail if recommendations else "No effectiveness recommendation."
    return await OrchestrationMemoryService().upsert_section(
        db,
        goal.project_id,
        goal.id,
        section_key=MEMORY_SECTION_KEY,
        title="Effectiveness review",
        body=_render_review_memory(triggers, checks, recommendations, decision),
        summary=summary,
        toc_order=MEMORY_TOC_ORDER,
        run_id=run.id,
        created_by="orchestrator:effectiveness_review",
    )


def _task_gate_map(run: OrchestrationRun) -> dict[str, str]:
    items = run.plan_state.get("expanded_items", []) if isinstance(run.plan_state, dict) else []
    return {
        str(item["task_id"]): str(item["gate_id"])
        for item in items
        if isinstance(item, dict) and item.get("task_id") and item.get("gate_id")
    }


def _utc(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc) if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _stable_token(values: Iterable[str]) -> str:
    payload = json.dumps(sorted(values), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _trigger_tokens(triggers: Iterable[TriggerReason]) -> frozenset[str]:
    return frozenset(trigger.token for trigger in triggers)


async def _manager_check(db: AsyncSession, goal: OrchestrationGoal) -> CheckResult:
    agent_id = goal.manager_agent_id
    user_id = goal.manager_user_id
    if goal.authority_model is None and agent_id is None and user_id is None:
        return CheckResult(
            "manager_valid",
            True,
            "Manager authority has not been selected yet.",
        )
    if goal.authority_model == "no_manager" and agent_id is None and user_id is None:
        return CheckResult(
            "manager_valid",
            True,
            "authority_model=no_manager intentionally selects no manager.",
        )
    if goal.authority_model == "agent_manager" and agent_id is not None and user_id is None:
        agent = await db.get(Agent, agent_id)
        if agent is not None and agent.is_active:
            return CheckResult("manager_valid", True, f"Selected agent manager {agent_id} is active.")
    elif goal.authority_model == "human_manager" and user_id is not None and agent_id is None:
        user = await db.get(User, user_id)
        if user is not None and user.is_active:
            return CheckResult("manager_valid", True, f"Selected human manager {user_id} is active.")
    return CheckResult(
        "manager_valid",
        False,
        "Selected manager is inconsistent or inactive: "
        f"authority_model={goal.authority_model}, "
        f"agent_id={agent_id}, user_id={user_id}.",
    )


async def _stored_hierarchy(db: AsyncSession, goal_id: uuid.UUID) -> dict | None:
    current = await OrchestrationProcessService().get_current(db, goal_id, "team_hierarchy")
    if current is None:
        return None
    outputs = current.outputs if isinstance(current.outputs, dict) else {}
    stored = outputs.get("proposal", outputs)
    return stored if isinstance(stored, dict) else {}


async def _hierarchy_principals_check(db: AsyncSession, stored: dict | None) -> CheckResult:
    if stored is None:
        return CheckResult(
            "hierarchy_principals_valid",
            True,
            "No current team hierarchy principals to validate.",
        )
    role_to_agent = stored.get("role_to_agent", {})
    if not isinstance(role_to_agent, dict):
        return CheckResult(
            "hierarchy_principals_valid",
            False,
            "Stored team hierarchy role_to_agent is invalid.",
        )
    references: list[tuple[str, str, uuid.UUID | None]] = []
    agent_ids: set[uuid.UUID] = set()
    for role, value in role_to_agent.items():
        try:
            agent_id = uuid.UUID(str(value))
        except (TypeError, ValueError):
            agent_id = None
        references.append((str(role), str(value), agent_id))
        if agent_id is not None:
            agent_ids.add(agent_id)
    active_ids = (
        set(
            (
                await db.execute(
                    select(Agent.id).where(Agent.id.in_(agent_ids), Agent.is_active.is_(True))
                )
            ).scalars()
        )
        if agent_ids
        else set()
    )
    invalid = sorted(
        f"{role}=agent:{value}"
        for role, value, agent_id in references
        if agent_id not in active_ids
    )
    if invalid:
        return CheckResult(
            "hierarchy_principals_valid",
            False,
            f"Invalid stored team hierarchy principals: {', '.join(invalid)}.",
        )
    return CheckResult(
        "hierarchy_principals_valid",
        True,
        f"Stored team hierarchy principals are active ({len(references)} referenced).",
    )


async def _live_weak_fit_check(
    db: AsyncSession,
    run: OrchestrationRun,
    stored: dict | None,
) -> CheckResult:
    weak_fits = stored.get("weak_fits", []) if stored is not None else []
    weak_pairs: set[tuple[str, uuid.UUID]] = set()
    if isinstance(weak_fits, list):
        for fit in weak_fits:
            if not isinstance(fit, dict) or not isinstance(fit.get("work_function"), str):
                continue
            try:
                weak_pairs.add((fit["work_function"], uuid.UUID(str(fit.get("agent_id")))))
            except (TypeError, ValueError):
                continue
    items = run.plan_state.get("expanded_items", []) if isinstance(run.plan_state, dict) else []
    expanded: list[tuple[uuid.UUID, str]] = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("work_function"), str):
            continue
        try:
            expanded.append((uuid.UUID(str(item.get("task_id"))), item["work_function"]))
        except (TypeError, ValueError):
            continue
    task_ids = {task_id for task_id, _ in expanded}
    tasks = (
        {
            task.id: task
            for task in (
                await db.execute(
                    select(Task).where(
                        Task.id.in_(task_ids),
                        Task.status.in_(LIVE_TASK_STATUSES),
                    )
                )
            ).scalars()
        }
        if task_ids and weak_pairs
        else {}
    )
    matches = sorted({
        f"task:{task_id}:{work_function}:agent:{task.assigned_to}"
        for task_id, work_function in expanded
        if (task := tasks.get(task_id)) is not None
        and task.assigned_to is not None
        and (work_function, task.assigned_to) in weak_pairs
    })
    return CheckResult(
        "live_weak_fit_assignments",
        not matches,
        (
            f"Live weak-fit assignments: {', '.join(matches)}."
            if matches
            else "No live tasks match stored weak-fit assignments."
        ),
    )


async def _material_warning_check(db: AsyncSession, goal_id: uuid.UUID) -> CheckResult:
    rows = (
        await db.execute(
            select(OrchestrationWarning.id, OrchestrationWarning.related_gate_id)
            .join(OrchestrationGate, OrchestrationGate.id == OrchestrationWarning.related_gate_id)
            .where(
                OrchestrationWarning.goal_id == goal_id,
                OrchestrationWarning.active.is_(True),
                OrchestrationWarning.acknowledged_at.is_not(None),
                OrchestrationWarning.related_gate_id.is_not(None),
                OrchestrationGate.status == "failed",
            )
        )
    ).all()
    materialized = sorted(f"warning:{warning_id}:gate:{gate_id}" for warning_id, gate_id in rows)
    return CheckResult(
        "materialized_acknowledged_warnings",
        not materialized,
        (
            f"Materialized acknowledged warnings: {', '.join(materialized)}."
            if materialized
            else "No acknowledged active warnings are linked to failed gates."
        ),
    )


async def collect_checks(
    db: AsyncSession,
    goal: OrchestrationGoal,
    run: OrchestrationRun,
) -> list[CheckResult]:
    gap_fields = []
    if not goal.objective.strip():
        gap_fields.append("objective")
    if not goal.success_criteria:
        gap_fields.append("success_criteria")
    goal_check = CheckResult(
        "goal_complete",
        not gap_fields,
        (
            f"Goal completeness gaps: {', '.join(gap_fields)}."
            if gap_fields
            else "Objective and success criteria are complete."
        ),
    )
    stored = await _stored_hierarchy(db, goal.id)
    return [
        goal_check,
        await _manager_check(db, goal),
        await _hierarchy_principals_check(db, stored),
        await _live_weak_fit_check(db, run, stored),
        await _material_warning_check(db, goal.id),
    ]


def _apply_blocker_floor(disposition: str, checks: list[CheckResult]) -> str:
    """Apply the deterministic safety override: if disposition is 'continue' but any check failed, revise instead.

    # ponytail: LLM must not be asked to "fix" a safety override; any failed check already made
    # continue unreachable in the old engine. This is the safety gate, never a repair loop.
    """
    if disposition == "continue" and any(not c.passed for c in checks):
        return "revise"
    return disposition


async def previous_completed_review_tokens(db: AsyncSession, goal_id: uuid.UUID) -> frozenset[str]:
    previous = (
        await db.execute(
            select(OrchestrationProcessRun)
            .where(
                OrchestrationProcessRun.goal_id == goal_id,
                OrchestrationProcessRun.process_type == PROCESS_TYPE,
                OrchestrationProcessRun.status == "completed",
            )
            .order_by(
                OrchestrationProcessRun.completed_at.desc(),
                OrchestrationProcessRun.created_at.desc(),
                OrchestrationProcessRun.id.desc(),
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    if previous is None or not isinstance(previous.outputs, dict):
        return frozenset()
    triggers = previous.outputs.get("triggers", [])
    if not isinstance(triggers, list):
        return frozenset()
    return frozenset(
        trigger["token"]
        for trigger in triggers
        if isinstance(trigger, dict) and isinstance(trigger.get("name"), str) and isinstance(trigger.get("token"), str)
    )


async def has_new_trigger_evidence(
    db: AsyncSession,
    goal_id: uuid.UUID,
    triggers: Iterable[TriggerReason],
) -> bool:
    return not _trigger_tokens(triggers).issubset(await previous_completed_review_tokens(db, goal_id))


async def detect_triggers(
    db: AsyncSession,
    goal: OrchestrationGoal,
    run: OrchestrationRun,
) -> list[TriggerReason]:
    actions = (
        await db.execute(
            select(OrchestrationAction).where(
                OrchestrationAction.run_id == run.id,
                OrchestrationAction.action_type.in_(("retry_task", "reassign_task")),
            )
        )
    ).scalars()
    task_to_gate = _task_gate_map(run)
    recovery_counts: dict[str, int] = {}
    for action in actions:
        task_id = action.request.get("task_id") if isinstance(action.request, dict) else None
        gate_id = task_to_gate.get(str(task_id))
        if gate_id:
            recovery_counts[gate_id] = recovery_counts.get(gate_id, 0) + 1

    repeated_recoveries = [
        TriggerReason(
            "repeated_recovery",
            f"gate:{gate_id}:{count}",
            f"{count} recoveries for gate {gate_id}",
        )
        for gate_id, count in sorted(recovery_counts.items())
        if count >= settings.effectiveness_recovery_threshold
    ]

    task_ids: set[uuid.UUID] = set()
    for task_id in task_to_gate:
        try:
            task_ids.add(uuid.UUID(task_id))
        except ValueError:
            continue
    failed_session_ids: list[str] = []
    if task_ids:
        sessions = (
            await db.execute(
                select(Session)
                .where(Session.task_id.in_(task_ids))
                .order_by(Session.created_at.desc(), Session.id.desc())
            )
        ).scalars()
        for session in sessions:
            if session.status != "failed":
                break
            failed_session_ids.append(str(session.id))
    failed_sessions = len(failed_session_ids)
    consecutive_failures = (
        TriggerReason(
            "consecutive_failed_sessions",
            f"failed_sessions:{_stable_token(failed_session_ids)}",
            f"{failed_sessions} consecutive failed sessions",
        )
        if failed_sessions >= settings.effectiveness_failed_session_threshold
        else None
    )
    # Substantive progress only: noops/warnings/etc. must not reset inactivity.
    latest_progress_action = (
        await db.execute(
            select(OrchestrationAction.created_at)
            .where(
                OrchestrationAction.run_id == run.id,
                OrchestrationAction.status == "completed",
                ~OrchestrationAction.action_type.in_(NON_PROGRESS_ACTION_TYPES),
            )
            .order_by(OrchestrationAction.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    gates = (
        await db.execute(select(OrchestrationGate).where(OrchestrationGate.run_id == run.id))
    ).scalars().all()
    gate_decisions = [value for gate in gates for value in (gate.accepted_at, gate.failed_at)]
    # ponytail: scan bounded to the run lifetime (SQL started/completed >= run.started_at), run metadata matched in Python; add a JSON index if this gets hot.
    task_conditions = [Task.project_id == goal.project_id]
    if task_ids:
        task_conditions.append(Task.id.in_(task_ids))
    task_where = [or_(*task_conditions), or_(Task.started_at.is_not(None), Task.completed_at.is_not(None))]
    if run.started_at is not None:
        # Task columns are naive UTC on sqlite; values before run start can never raise last_progress (run.started_at is in the max).
        run_start = _utc(run.started_at).replace(tzinfo=None)
        task_where.append(or_(Task.started_at >= run_start, Task.completed_at >= run_start))
    task_rows = (
        await db.execute(select(Task.id, Task.started_at, Task.completed_at, Task.metadata_).where(*task_where))
    ).all()
    task_progress = [
        value
        for task_id, started_at, completed_at, metadata in task_rows
        if str(task_id) in task_to_gate
        or (
            isinstance(metadata, dict)
            and isinstance(metadata.get("orchestration"), dict)
            and metadata["orchestration"].get("run_id") == str(run.id)
        )
        for value in (started_at, completed_at)
    ]
    # ponytail: Evidence.updated_at bumps on any edit, so non-progress edits count as progress (accepted noise).
    evidence_max_created, evidence_max_updated = (
        await db.execute(
            select(func.max(OrchestrationEvidence.created_at), func.max(OrchestrationEvidence.updated_at)).where(
                OrchestrationEvidence.run_id == run.id
            )
        )
    ).one()
    last_progress = max(
        _utc(value)
        for value in (
            run.started_at,
            latest_progress_action,
            *gate_decisions,
            *task_progress,
            evidence_max_created,
            evidence_max_updated,
        )
        if value is not None
    )
    idle_hours = (_utc(datetime.now(timezone.utc)) - last_progress).total_seconds() / 3600
    inactivity_bucket = int(idle_hours // settings.effectiveness_inactivity_hours)
    inactivity = (
        TriggerReason(
            "inactivity",
            last_progress.isoformat(),
            f"No action or gate progress for {inactivity_bucket * settings.effectiveness_inactivity_hours} hours",
        )
        if inactivity_bucket >= 1
        else None
    )
    non_summary_gates = [gate for gate in gates if gate.gate_type != "final_summary_accepted"]
    pre_completion = (
        TriggerReason(
            "pre_completion",
            str(run.id),
            "All non-summary gates are accepted",
        )
        if non_summary_gates and all(gate.status == "accepted" for gate in non_summary_gates)
        else None
    )
    return [
        trigger
        for trigger in (*repeated_recoveries, consecutive_failures, inactivity, pre_completion)
        if trigger is not None
    ]


class EffectivenessReviewProcess:
    def __init__(self, analyzer: EffectivenessAnalyzer | None = None) -> None:
        self.process_service = OrchestrationProcessService()
        self.decision_service = OrchestrationAuthorityDecisionService()
        self.warning_service = OrchestrationWarningService()
        self.analyzer = analyzer or EffectivenessAnalyzer()

    async def _finalize_analysis(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        current: OrchestrationProcessRun,
        triggers: list[TriggerReason],
        checks: list[CheckResult],
        analysis: EffectivenessAnalysis,
    ) -> dict:
        """Apply the LLM analysis with the deterministic safety override and complete the process or park for decision."""
        # Clear recovery state on analyzer success: filter blockers and resolve companion warning
        checkpoint = (current.outputs or {}).get("_lm_retry")
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("warning_id"), str):
            warning_id = checkpoint["warning_id"]
            # Filter active_blockers to drop the matching effectiveness_review_analyzer_error blocker
            run.active_blockers = [
                item
                for item in run.active_blockers
                if not (
                    isinstance(item, dict)
                    and item.get("kind") == "effectiveness_review_analyzer_error"
                    and item.get("warning_id") == warning_id
                )
            ]
            # Resolve the companion warning
            for warning in await self.warning_service.list_warnings(db, goal.id, active_only=True):
                if str(warning.id) == warning_id:
                    await self.warning_service.resolve_warning(
                        db,
                        warning,
                        resolved_by="orchestrator:effectiveness_review",
                        reason="effectiveness review analysis succeeded after analyzer failure",
                    )
                    break
            # Pop recovery state from outputs so final outputs are clean
            outputs = dict(current.outputs or {})
            outputs.pop("_lm_retry", None)
            outputs.pop("error", None)
            outputs.pop("retryable", None)
            current.outputs = outputs

        disposition = _apply_blocker_floor(analysis.disposition, checks)
        detail = (
            analysis.rationale.strip()
            or "; ".join(f"{f['name']}: {f['detail']}" for f in analysis.findings)
            or "No rationale provided."
        )
        recommendations = [Recommendation(disposition, detail)]

        outputs = {
            "triggers": serialize_triggers(triggers),
            "checks": serialize_checks(checks),
            "recommendations": serialize_recommendations(recommendations),
            "recommended_disposition": disposition,
            "run_id": str(run.id),
            "decision_id": None,
            "decision_status": None,
            "selected_disposition": None,
            "decision_reason": None,
        }

        if disposition == "continue":
            await upsert_review_memory(db, goal, run, triggers, checks, recommendations)
            await self.process_service.complete_process(db, current, outputs=outputs)
            return self._summary("completed", recommended_disposition=disposition)

        authority = "human"
        authority_agent_id = None
        if goal.authority_model == "agent_manager" and goal.manager_agent_id is not None:
            manager = await db.get(Agent, goal.manager_agent_id)
            if manager is not None and manager.is_active:
                authority = "manager"
                authority_agent_id = manager.id

        decision_key = f"{PROCESS_TYPE}:disposition:{current.id}"
        decision = await self.decision_service.get_pending_by_key(db, goal.id, decision_key)
        questions_created = 0
        if decision is None:
            decision = await self.decision_service.create_pending(
                db,
                goal.id,
                decision_key=decision_key,
                title="Choose effectiveness review disposition",
                question="How should work proceed after this effectiveness review?",
                authority=authority,
                authority_agent_id=authority_agent_id,
                options=[
                    {"key": "continue"},
                    {"key": "revise"},
                    {"key": "split"},
                    {"key": "pause"},
                ],
                context=json.dumps({"recommended_disposition": disposition}, sort_keys=True),
                recommendation=disposition,
                consequences="The selected disposition is recorded but not executed.",
                run_id=run.id,
                source_process_run_id=current.id,
            )
            questions_created = 1

        decision_snapshot = {
            "id": str(decision.id),
            "status": decision.status,
            "selected_disposition": decision.selected_option,
            "reason": decision.reason,
        }
        outputs.update(
            {
                "decision_id": str(decision.id),
                "decision_status": decision.status,
                "selected_disposition": decision.selected_option,
                "decision_reason": decision.reason,
            }
        )
        current.outputs = outputs
        await db.flush()
        await upsert_review_memory(
            db,
            goal,
            run,
            triggers,
            checks,
            recommendations,
            decision=decision_snapshot,
        )
        await self.process_service.park_process(db, current)
        return self._summary(
            "waiting_decision",
            questions_created=questions_created,
            recommended_disposition=disposition,
        )

    async def advance(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
    ) -> dict:
        current = await self.process_service.get_current(db, goal.id, PROCESS_TYPE)
        if current is not None and current.status == "skipped":
            return self._summary("skipped")
        if current is None or current.status == "completed":
            triggers = await detect_triggers(db, goal, run)
            if not triggers or not await has_new_trigger_evidence(db, goal.id, triggers):
                disposition = (
                    current.outputs.get("recommended_disposition")
                    if current is not None and isinstance(current.outputs, dict)
                    else None
                )
                return self._summary(current.status if current is not None else "idle", recommended_disposition=disposition)
            current = await self.process_service.start_process(
                db,
                goal.id,
                process_type=PROCESS_TYPE,
                trigger_reason="automatic: " + ", ".join(trigger.name for trigger in triggers),
                run_id=run.id,
                input_snapshot={"triggers": serialize_triggers(triggers)},
            )
        if current.status == "waiting_decision":
            return await self._advance_waiting(db, goal, run, current)
        if current.status != "running":
            disposition = (
                current.outputs.get("recommended_disposition")
                if isinstance(current.outputs, dict)
                else None
            )
            return self._summary(current.status, recommended_disposition=disposition)
        return await self._advance_running(db, goal, run, current)

    async def _advance_running(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        current: OrchestrationProcessRun,
    ) -> dict:
        # Checkpoint guard: if the LLM request failed and is retryable, return immediately
        # without processing again. This prevents tick-driven retry storms.
        if isinstance((current.outputs or {}).get("_lm_retry"), dict):
            return {
                "process_type": PROCESS_TYPE,
                "status": current.status,
                "retryable": True,
                "error": (current.outputs or {}).get("error"),
            }

        snapshot = current.input_snapshot if isinstance(current.input_snapshot, dict) else {}
        stored = snapshot.get("triggers", [])
        triggers = [
            TriggerReason(item["name"], item["token"], item["detail"])
            for item in stored
            if isinstance(item, dict)
            and all(isinstance(item.get(key), str) for key in ("name", "token", "detail"))
        ] if isinstance(stored, list) else []
        if not triggers:
            triggers = [
                TriggerReason("manual", f"manual:{current.id}", current.trigger_reason),
                *await detect_triggers(db, goal, run),
            ]
            current.input_snapshot = {**snapshot, "triggers": serialize_triggers(triggers)}
            await db.flush()

        checks = await collect_checks(db, goal, run)

        # Build the payload for the analyzer
        payload = {
            "schema_version": 1,
            "goal": {
                "id": str(goal.id),
                "objective": goal.objective,
                "success_criteria": goal.success_criteria,
                "weight": goal.weight,
            },
            "triggers": serialize_triggers(triggers),
            "checks": serialize_checks(checks),
        }

        # Fetch project for request building
        project = await db.get(Project, goal.project_id)
        project_dict = {"name": project.name, "description": project.description} if project else None

        # Build the request so we can store it on failure
        request = self.analyzer.build_request(payload, project=project_dict)

        try:
            analysis = await self.analyzer.review_request(request, project_id=goal.project_id)
        except Exception as exc:
            safe_error = _safe_completion_error(exc)
            current.outputs = {
                "_lm_retry": {
                    "kind": "effectiveness_review",
                    "version": 1,
                    "request": redact_semantic_payload(request),
                },
                "retryable": True,
                "error": safe_error,
            }
            warning = await self.warning_service.create_warning(
                db,
                goal.id,
                warning_type="effectiveness_review_analyzer_error",
                severity="warning",
                message=f"Effectiveness review analysis failed: {safe_error}",
                run_id=run.id,
                source_process_run_id=current.id,
            )
            checkpoint = dict(current.outputs["_lm_retry"])
            checkpoint["warning_id"] = str(warning.id)
            current.outputs = {**current.outputs, "_lm_retry": checkpoint}
            from huddleroom.services.orchestration_service import OrchestrationService
            OrchestrationService._upsert_active_blocker(run, {
                "kind": "effectiveness_review_analyzer_error",
                "gate_id": str(warning.id),
                "warning_id": str(warning.id),
                "reason": f"Effectiveness review analysis failed: {safe_error}",
            })
            await db.flush()
            return {
                "process_type": PROCESS_TYPE,
                "status": current.status,
                "retryable": True,
                "error": safe_error,
            }

        return await self._finalize_analysis(db, goal, run, current, triggers, checks, analysis)

    async def _advance_waiting(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        current: OrchestrationProcessRun,
    ) -> dict:
        decision = (
            await db.execute(
                select(OrchestrationAuthorityDecision)
                .where(OrchestrationAuthorityDecision.source_process_run_id == current.id)
                .order_by(
                    OrchestrationAuthorityDecision.asked_at.desc(),
                    OrchestrationAuthorityDecision.created_at.desc(),
                    OrchestrationAuthorityDecision.id.desc(),
                )
                .limit(1)
            )
        ).scalar_one_or_none()
        disposition = (
            current.outputs.get("recommended_disposition")
            if isinstance(current.outputs, dict)
            else None
        )
        if decision is None:
            return self._summary(
                "waiting_decision",
                recommended_disposition=disposition,
                inconsistency="waiting effectiveness review has no linked decision",
            )
        if decision.status == "pending":
            return self._summary("waiting_decision", recommended_disposition=disposition)

        outputs = dict(current.outputs) if isinstance(current.outputs, dict) else {}
        selected = decision.selected_option if decision.status == "answered" else None
        reason = decision.reason if decision.status in {"answered", "cancelled"} else None
        outputs.update(
            {
                "decision_id": str(decision.id),
                "decision_status": decision.status,
                "selected_disposition": selected,
                "decision_reason": reason,
            }
        )
        triggers = [TriggerReason(**item) for item in outputs.get("triggers", [])]
        checks = [CheckResult(**item) for item in outputs.get("checks", [])]
        recommendations = [Recommendation(**item) for item in outputs.get("recommendations", [])]
        await upsert_review_memory(
            db,
            goal,
            run,
            triggers,
            checks,
            recommendations,
            decision={
                "id": str(decision.id),
                "status": decision.status,
                "selected_disposition": selected,
                "reason": reason,
            },
        )
        await self.process_service.resume_process(db, current)
        await self.process_service.complete_process(db, current, outputs=outputs)
        return self._summary(
            "completed",
            recommended_disposition=disposition,
            decision_id=str(decision.id),
            decision_status=decision.status,
            selected_disposition=selected,
        )

    async def retry_failed(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        current: OrchestrationProcessRun,
    ) -> dict:
        """Retry a failed effectiveness review analysis using the stored request."""
        checkpoint = (current.outputs or {}).get("_lm_retry")
        if not isinstance(checkpoint, dict) or checkpoint.get("kind") != "effectiveness_review":
            raise ValueError("effectiveness review has no valid retry checkpoint")

        try:
            analysis = await self.analyzer.review_request(checkpoint["request"], project_id=goal.project_id)
        except Exception as exc:
            safe_error = _safe_completion_error(exc)
            current.outputs = {
                key: value for key, value in (current.outputs or {}).items()
                if key not in {"_lm_retry", "retryable", "error"}
            }
            current.outputs["_lm_retry"] = {
                "kind": "effectiveness_review",
                "version": 1,
                "request": checkpoint["request"],
            }
            current.outputs["retryable"] = True
            current.outputs["error"] = safe_error
            warning = await self.warning_service.create_warning(
                db,
                goal.id,
                warning_type="effectiveness_review_analyzer_error",
                severity="warning",
                message=f"Effectiveness review analysis failed: {safe_error}",
                run_id=run.id,
                source_process_run_id=current.id,
            )
            checkpoint_dict = dict(current.outputs["_lm_retry"])
            checkpoint_dict["warning_id"] = str(warning.id)
            current.outputs = {**current.outputs, "_lm_retry": checkpoint_dict}
            from huddleroom.services.orchestration_service import OrchestrationService
            OrchestrationService._upsert_active_blocker(run, {
                "kind": "effectiveness_review_analyzer_error",
                "gate_id": str(warning.id),
                "warning_id": str(warning.id),
                "reason": f"Effectiveness review analysis failed: {safe_error}",
            })
            await db.flush()
            return {
                "process_type": PROCESS_TYPE,
                "status": current.status,
                "retryable": True,
                "error": safe_error,
            }

        # Extract triggers and checks from the stored request payload for replay
        # # ponytail: replay from checkpoint request, not recompute live. Ensures consistency
        # with the LLM's actual evidence window at failure time.
        payload = json.loads(checkpoint["request"]["messages"][1]["content"])
        triggers = [
            TriggerReason(item["name"], item["token"], item["detail"])
            for item in payload.get("triggers", [])
            if isinstance(item, dict)
            and all(isinstance(item.get(key), str) for key in ("name", "token", "detail"))
        ]
        checks = [
            CheckResult(
                item["name"],
                item["passed"],
                item["detail"],
            )
            for item in payload.get("checks", [])
            if isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and isinstance(item.get("passed"), bool)
            and isinstance(item.get("detail"), str)
        ]

        return await self._finalize_analysis(db, goal, run, current, triggers, checks, analysis)

    @staticmethod
    def _summary(
        status: str,
        *,
        questions_created: int = 0,
        recommended_disposition: str | None = None,
        decision_id: str | None = None,
        decision_status: str | None = None,
        selected_disposition: str | None = None,
        inconsistency: str | None = None,
    ) -> dict:
        summary = {
            "process_type": PROCESS_TYPE,
            "status": status,
            "questions_created": questions_created,
            "recommended_disposition": recommended_disposition,
        }
        if decision_id is not None:
            summary.update(
                {
                    "decision_id": decision_id,
                    "decision_status": decision_status,
                    "selected_disposition": selected_disposition,
                }
            )
        if inconsistency is not None:
            summary["inconsistency"] = inconsistency
        return summary
