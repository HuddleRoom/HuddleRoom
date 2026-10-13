import hashlib
import json
import logging
import uuid
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Literal
from uuid import UUID

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from huddleroom.config import settings
from huddleroom.models.artifact import Artifact
from huddleroom.models.orchestration import (
    EXPLICIT_PLANNING_MODE_BY_GOAL_TYPE,
    GOAL_CONTRACT_FINGERPRINT_FIELDS,
    OrchestrationAction,
    OrchestrationBudgetReservation,
    OrchestrationEvidence,
    OrchestrationGate,
    OrchestrationGoal,
    OrchestrationRun,
    OrchestrationWait,
    compute_goal_contract_fingerprint,
    explicit_planning_rationale,
)
from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import (
    GoalShapeDecision,
    GoalTypeCorrectionApplyRequest,
    GoalTypeCorrectionBlocker,
    GoalTypeCorrectionPreviewRequest,
    GoalTypeCorrectionPreviewResult,
    GoalTypeCorrectionRetainedEvidence,
    OrchestrationContinuousPolicy,
    OrchestrationGoalCreate,
)
from huddleroom.services.orchestration_dependencies import ADHOC_GATE_TYPE, RETAINED_GATE_TYPE, _obj, _uuid, task_gate_id
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

_TERMINAL_GOAL_STATUSES = ("completed", "cancelled")

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
    "For example, a user-visible reports export feature is outcome/delivery: it coordinates behavior, UI, "
    "safety, and qualification. A localized pagination-helper correction is outcome/implementation: "
    "one specialist can make a leaf implementation plan. These are representative structural examples, "
    "not an objective-text lookup; keep size, criteria, and budget independent of shape. "
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

    # --- Typed correction ---

    async def preview_correction(
        self, db, goal: OrchestrationGoal, request: GoalTypeCorrectionPreviewRequest, actor: str,
    ) -> GoalTypeCorrectionPreviewResult:
        """Read-only: what a goal-type correction would block on, require, retain, and cost. Writes nothing."""
        from huddleroom.services.orchestration_budget_service import BudgetMeasurementError, OrchestrationBudgetService
        from huddleroom.services.orchestration_service import OrchestrationService

        run = await OrchestrationService().get_run_for_goal(db, goal.project_id, goal.id)
        if run is None:
            raise HTTPException(status_code=409, detail="Goal has no run")
        fp = compute_goal_contract_fingerprint(goal)
        requested = request.requested_goal_type
        run_task = Task.metadata_["orchestration"]["run_id"].as_string() == str(run.id)
        blockers: list[GoalTypeCorrectionBlocker] = []

        def block(code, entity_type, entity_id, message, resolution):
            blockers.append(GoalTypeCorrectionBlocker(
                code=code, entity_type=entity_type, entity_id=entity_id, message=message, resolution=resolution,
            ))

        if request.expected_run_id != run.id:
            block("stale_run", "run", run.id, "The goal's current run changed since this request was prepared.",
                  "Reload the goal and preview again.")
        if request.expected_contract_fingerprint != fp:
            block("stale_contract", "goal", goal.id, "The goal contract changed since this request was prepared.",
                  "Reload the goal and preview again.")
        if goal.status in _TERMINAL_GOAL_STATUSES:
            block("goal_terminal", "goal", goal.id, f"The goal is {goal.status}.", "Create a new goal instead.")
        if requested == goal.goal_type:
            block("same_goal_type", "goal", goal.id, f"The goal is already {requested}.", "Choose a different type.")
        # Defense in depth: every current caller passes a human: actor; an agent-initiated correction path must not bypass an explicit choice.
        if not actor.startswith("human:") and goal.goal_type_source not in ("auto", "parent_auto"):
            block("explicit_choice_requires_user", "goal", goal.id,
                  f"The goal type was chosen explicitly ({goal.goal_type_source}); only a user may change it.",
                  "Ask the user to request the correction.")
        if goal.parent_goal_id is not None:
            if requested == "continuous":
                block("continuous_child_requires_continuous_root", "goal", goal.id,
                      "A child goal cannot become Continuous.", "Create a root Continuous goal instead.")
            if goal.goal_type_source == "parent_explicit":
                block("parent_explicit_requires_parent_plan", "goal", goal.parent_goal_id,
                      "The parent plan chose this child's type explicitly.",
                      "Revise the parent plan with a new item of the desired type.")
            reservation = await db.scalar(select(OrchestrationBudgetReservation).where(
                OrchestrationBudgetReservation.child_goal_id == goal.id,
            ))
            if reservation is None or reservation.status != "active":
                block("parent_reservation_inactive", "budget_reservation", reservation.id if reservation else None,
                      "The parent's budget reservation for this child is not active.",
                      "Correct the goal only while the parent still holds its reservation.")
        for session_id in (await db.scalars(select(Session.id).join(Task, Task.id == Session.task_id).where(
            Session.status.in_(("pending", "running")), run_task,
        ))).all():
            block("active_session", "session", session_id, "An agent session is still running for this run.",
                  "Pause the goal and let the session finish.")
        for task_id in (await db.scalars(select(Task.id).where(
            Task.status.in_(("ready", "in_progress")), run_task,
        ))).all():
            block("active_task", "task", task_id, "A task of this run is ready or in progress.",
                  "Pause the goal and let the task finish.")
        for action_id in (await db.scalars(select(OrchestrationAction.id).where(
            OrchestrationAction.run_id == run.id, OrchestrationAction.status == "reserved",
        ))).all():
            block("reserved_action", "action", action_id, "An orchestration action is still in flight.",
                  "Wait for the action to finish.")
        for child_id in (await db.scalars(select(OrchestrationGoal.id).where(
            OrchestrationGoal.parent_goal_id == goal.id, OrchestrationGoal.status.not_in(_TERMINAL_GOAL_STATUSES),
        ))).all():
            block("active_child_goal", "goal", child_id, "A child goal is still active.",
                  "Let the child goal finish or cancel it.")
        try:
            remaining = (await OrchestrationBudgetService().remaining(db, goal))["remaining"]
        except BudgetMeasurementError as exc:
            remaining = {}
            block("budget_measurement", "session", exc.session_id,
                  f"Missing {exc.dimension} measurement; remaining budget is unknown.",
                  "Wait for session telemetry or settle the session's spend.")
        blockers.sort(key=lambda item: (item.code, str(item.entity_id)))

        missing_policy = False
        if requested == "continuous":
            try:
                OrchestrationContinuousPolicy.model_validate({
                    key: value for key, value in (goal.continuous_policy or {}).items() if key != "version"
                })
            except ValidationError:
                missing_policy = True
        if goal.parent_goal_id is None:
            authority = ["baseline_rerun", "start"] + (["continuous_policy"] if missing_policy else [])
        elif goal.goal_type_source == "parent_explicit":
            authority = ["parent_plan", "authority_review"]
        else:
            authority = ["parent_release"]

        result = GoalTypeCorrectionPreviewResult(
            goal_id=goal.id, current_run_id=run.id,
            current_goal_type=goal.goal_type, proposed_goal_type=requested,
            current_planning_mode=goal.planning_mode,
            proposed_planning_mode=EXPLICIT_PLANNING_MODE_BY_GOAL_TYPE[requested],
            source="user", rationale=request.rationale, blockers=blockers,
            retained_evidence_manifest=await self._retained_manifest(db, goal, run, fp),
            remaining_budget=remaining, missing_recurring_policy=missing_policy,
            requires_new_start=goal.parent_goal_id is None, required_authority=authority,
            preview_fingerprint="0" * 64,
        )
        payload = {**result.model_dump(mode="json", exclude={"preview_fingerprint"}), "contract_fingerprint": fp}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
        return result.model_copy(update={"preview_fingerprint": digest})

    async def apply_correction(
        self, db, goal: OrchestrationGoal, request: GoalTypeCorrectionApplyRequest, actor: str,
    ) -> tuple[OrchestrationGoal, OrchestrationRun]:
        """Apply a previewed correction: retire the current run as reclassified and create one successor run."""
        from huddleroom.services.orchestration_service import OrchestrationService

        if not actor.startswith("human:"):
            raise HTTPException(status_code=403, detail={
                "conflict": "explicit_user_request_required", "message": "Only a user can correct a goal type.",
            })
        svc = OrchestrationService()
        key = (f"run:{request.expected_run_id}:kind:correct_goal_type:"
               f"{request.requested_goal_type}:{request.preview_fingerprint}")
        reload = select(OrchestrationGoal).where(OrchestrationGoal.id == goal.id).execution_options(
            populate_existing=True,
        )
        async with svc._lock_goal_for_baseline_transition(db, goal.id):
            goal = await db.scalar(reload)
            if (successor := await self._correction_replay(db, goal.id, request.expected_run_id, key)) is not None:
                return goal, successor
            preview = await self.preview_correction(db, goal, request, actor)
            if preview.preview_fingerprint != request.preview_fingerprint:
                raise HTTPException(status_code=409, detail={
                    "conflict": "stale_preview", "message": "The correction preview is out of date; review it again.",
                    "preview": preview.model_dump(mode="json"),
                })
            if preview.blockers:
                codes = {blocker.code for blocker in preview.blockers}
                raise HTTPException(status_code=409, detail={
                    "conflict": next((code for code in ("stale_run", "stale_contract") if code in codes),
                                     "correction_blocked"),
                    "message": "The goal type cannot be corrected yet.",
                    "blockers": [blocker.model_dump(mode="json") for blocker in preview.blockers],
                    "preview": preview.model_dump(mode="json"),
                })
            old = await db.scalar(select(OrchestrationRun).where(
                OrchestrationRun.id == preview.current_run_id,
            ).execution_options(populate_existing=True))
            try:
                async with db.begin_nested():
                    successor = await self._apply_locked(db, svc, goal, old, preview, actor, key)
            except IntegrityError:
                goal = await db.scalar(reload)
                if (successor := await self._correction_replay(db, goal.id, request.expected_run_id, key)) is not None:
                    return goal, successor
                raise HTTPException(status_code=409, detail={
                    "conflict": "stale_run", "message": "The goal's run changed concurrently; preview again.",
                })
        return goal, successor

    @staticmethod
    async def _correction_replay(db, goal_id: UUID, run_id: UUID, key: str) -> OrchestrationRun | None:
        action = await db.scalar(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run_id, OrchestrationAction.idempotency_key == key,
            OrchestrationAction.status == "completed",
        ))
        successor = await db.get(OrchestrationRun, action.target_id) if action is not None else None
        return successor if successor is not None and successor.goal_id == goal_id else None

    async def _apply_locked(
        self, db, svc, goal: OrchestrationGoal, old: OrchestrationRun, preview: GoalTypeCorrectionPreviewResult,
        actor: str, key: str,
    ) -> OrchestrationRun:
        from huddleroom.services.event_bus import emit_event_once
        from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
        from huddleroom.services.orchestration_continuous_service import OrchestrationContinuousService
        from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService
        from huddleroom.services.task_service import TaskService

        now = _utcnow()
        was_paused = goal.status == "paused"  # _stop_locked below sets the goal active
        requested = preview.proposed_goal_type
        cancellable = []
        for task in await svc._orchestrated_tasks_for_run(
            db, old.id, statuses=["backlog", "ready", "in_progress", "blocked", "failed"],
        ):
            statuses = set((await db.scalars(select(Session.status).where(Session.task_id == task.id))).all())
            if not statuses & {"pending", "running"}:  # preview blocks live sessions; never cancel one here
                cancellable.append((task, bool(statuses)))
        action = await svc.reserve_action(db, old.id, key, "correct_goal_type", {
            "action_type": "correct_goal_type", "requested_goal_type": requested,
            "previous_goal_type": goal.goal_type, "previous_planning_mode": goal.planning_mode,
            "rationale": preview.rationale, "preview_fingerprint": preview.preview_fingerprint, "actor": actor,
            "cancelled_tasks": [
                {"task_id": str(task.id), "previously_activated": activated} for task, activated in cancellable
            ],
        })
        for task, _ in cancellable:
            await TaskService().cancel(db, goal.project_id, task.id)
        decisions = OrchestrationAuthorityDecisionService()
        # Baseline-process decisions are goal-keyed baseline work and survive the correction.
        for decision in (await db.scalars(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.run_id == old.id, OrchestrationAuthorityDecision.status == "pending",
            OrchestrationAuthorityDecision.source_process_run_id.is_(None),
        ))).all():
            await decisions.cancel_decision(db, decision, reason="superseded by goal type correction")
        await db.execute(update(OrchestrationWait).where(
            OrchestrationWait.run_id == old.id, OrchestrationWait.status == "open",
        ).values(status="cleared", cleared_at=now))
        if goal.goal_type == "roadmap":
            # Finished children only (active ones block the preview); their reservations stop counting.
            await OrchestrationRoadmapService(svc).settle_finished_children(db, goal, old)
        elif goal.goal_type == "continuous" and _obj(goal.continuous_state).get("started_at"):
            # After the task loop: stop's discovery cleanup skips the tasks cancelled above (no double cancel).
            continuous = OrchestrationContinuousService(svc)
            await continuous._stop_locked(db, goal, old, actor=actor, reason="goal type corrected", now=now)
            await continuous.settle_terminal_children(db, goal, now)
            # A stale started_at would turn a later finite->Continuous Start into a no-op; the stop audit keeps the record.
            goal.continuous_state = {}

        if old.status in ("running", "blocked", "paused"):
            old.status, old.completed_at = "cancelled", now
        old.terminal_reason = "reclassified"
        await db.flush()  # free the one-active-run slot before the successor insert

        goal.goal_type, goal.goal_type_source = requested, "user"
        goal.goal_type_rationale = preview.rationale.strip()
        goal.planning_mode = preview.proposed_planning_mode
        goal.planning_rationale = explicit_planning_rationale(requested)
        goal.status = "paused" if was_paused else "active"
        manifest = [entry.model_dump(mode="json") for entry in preview.retained_evidence_manifest]
        plan_state: dict[str, Any] = {"retained_evidence_manifest": manifest}
        if goal.parent_goal_id is None:
            budget_state = deepcopy(goal.budget or {})
        else:
            budget_state = {"caps": deepcopy(_obj(old.budget_state).get("caps", _obj(goal.budget).get("caps", {})))}
            plan_state["child_delta_baseline"] = {
                "status": "pending_renewal", "parent_goal_id": str(goal.parent_goal_id),
                "roadmap_version_id": str(goal.roadmap_version_id) if goal.roadmap_version_id else None,
                "roadmap_item_key": goal.roadmap_item_key, "supersedes_run_id": str(old.id),
            }
        successor = OrchestrationRun(
            id=uuid.uuid5(uuid.NAMESPACE_URL, f"rally:type-correction-successor:{old.id}"),
            goal_id=goal.id, supersedes_run_id=old.id, phase="baseline", status="paused" if was_paused else "running",
            # ponytail: events before the old cursor belong to the old run; resume from it.
            event_cursor=old.event_cursor,
            baseline_authorized=old.baseline_authorized if goal.parent_goal_id is None else True,
            goal_type_snapshot=goal.goal_type, planning_mode_snapshot=goal.planning_mode,
            budget_state=budget_state, plan_state=plan_state,
        )
        db.add(successor)
        await db.flush()
        for entry in manifest:
            if not entry["revalidation_required"]:  # changed requirements stay reference-only
                await svc._ensure_keyed_gate(
                    db, successor.id, RETAINED_GATE_TYPE, entry["source_gate_id"], {**entry, "min_count": 0},
                )
        if goal.parent_goal_id is None:
            await self._rerun_baseline(db, goal, successor, f"goal type corrected to {requested}")
        await svc._mark_action_completed(db, action, target_type="run", target_id=successor.id)
        await emit_event_once(
            db, goal.project_id, "orchestration.goal_type_corrected", {
                "goal_id": str(goal.id), "old_run_id": str(old.id), "successor_run_id": str(successor.id),
                "from_type": preview.current_goal_type, "to_type": requested,
            }, source="orchestrator", dedup_key=key,
        )
        return successor

    @staticmethod
    async def _rerun_baseline(db, goal: OrchestrationGoal, run: OrchestrationRun, reason: str) -> None:
        """Restart each finished baseline process on the successor; answered decisions keep their keys."""
        from huddleroom.services.orchestration_process_service import OrchestrationProcessService
        from huddleroom.services.project_service import ProjectService

        processes = OrchestrationProcessService()
        finished = []
        for process_type in ("goal_definition", "manager_selection", "agent_definition_review", "team_hierarchy"):
            current = await processes.get_current(db, goal.id, process_type)
            if current is not None and current.status in ("completed", "skipped"):
                finished.append(process_type)
        if not finished:
            return
        await ProjectService().lock_workspace_boundary(db, goal.project_id)
        await ProjectService().require_runnable_project(db, goal.project_id)
        for process_type in finished:
            await processes.start_process(db, goal.id, process_type=process_type, trigger_reason=reason, run_id=run.id)

    @staticmethod
    def contract_unchanged_since_start(goal: OrchestrationGoal, run: OrchestrationRun) -> bool:
        """True only when the goal contract differs from the run's Start freeze by type/mode alone."""
        if None in (run.goal_contract_fingerprint, run.goal_type_snapshot, run.planning_mode_snapshot):
            return False
        as_started = SimpleNamespace(**{key: getattr(goal, key) for key in GOAL_CONTRACT_FINGERPRINT_FIELDS})
        as_started.goal_type, as_started.planning_mode = run.goal_type_snapshot, run.planning_mode_snapshot
        return compute_goal_contract_fingerprint(as_started) == run.goal_contract_fingerprint

    async def _retained_manifest(self, db, goal, run, fp) -> list[GoalTypeCorrectionRetainedEvidence]:
        from huddleroom.services.orchestration_service import PLAN_ITEM_GATE_TYPE, OrchestrationService

        # child_goal_completed gates are excluded: no source task to retain (ponytail).
        gates = list((await db.scalars(select(OrchestrationGate).where(
            OrchestrationGate.run_id == run.id,
            OrchestrationGate.status == "accepted",
            OrchestrationGate.gate_type.in_((PLAN_ITEM_GATE_TYPE, ADHOC_GATE_TYPE, RETAINED_GATE_TYPE)),
        ))).all())
        if not gates:
            return []
        producer: dict[UUID, UUID] = {}
        for task in await OrchestrationService()._orchestrated_tasks_for_run(
            db, run.id, statuses=list(Task.VALID_TRANSITIONS),
        ):
            adhoc_gate_id = _uuid(_obj(_obj(task.metadata_).get("orchestration")).get("adhoc_gate_id"))
            for gate_id in (task_gate_id(task, run.id), adhoc_gate_id):
                if gate_id is not None:
                    producer.setdefault(gate_id, task.id)
        artifact_by_gate: dict[UUID, UUID] = {}
        for evidence in (await db.scalars(select(OrchestrationEvidence).where(
            OrchestrationEvidence.gate_id.in_([gate.id for gate in gates]),
            OrchestrationEvidence.source_type == "artifact",
            OrchestrationEvidence.verdict == "accepted",
            OrchestrationEvidence.source_id.is_not(None),
        ).order_by(OrchestrationEvidence.created_at.asc(), OrchestrationEvidence.id.asc()))).all():
            artifact_by_gate.setdefault(evidence.gate_id, evidence.source_id)
        unchanged = self.contract_unchanged_since_start(goal, run)
        entries = []
        for gate in gates:
            if gate.gate_type == RETAINED_GATE_TYPE:
                # A retained gate carries its original source forward.
                source = _obj(gate.required_evidence)
                entry = {key: source.get(key) for key in (
                    "source_run_id", "source_task_id", "source_gate_id", "artifact_id", "source_contract_fingerprint",
                )}
            elif gate.id in producer:
                entry = {
                    "source_run_id": run.id, "source_task_id": producer[gate.id], "source_gate_id": gate.id,
                    "artifact_id": artifact_by_gate.get(gate.id),
                    "source_contract_fingerprint": run.goal_contract_fingerprint or fp,
                }
            else:
                continue
            artifact_id = _uuid(entry["artifact_id"])
            artifact = await db.get(Artifact, artifact_id) if artifact_id is not None else None
            artifact_ok = artifact_id is None or (artifact is not None and artifact.project_id == goal.project_id)
            revalidate = not (unchanged and artifact_ok)
            entries.append(GoalTypeCorrectionRetainedEvidence(
                **entry, revalidation_required=revalidate,
                reason="requirements changed / legacy run; re-verify before use" if revalidate
                else "contract unchanged since source Start; new run must validate provenance",
            ))
        return sorted(entries, key=lambda item: str(item.source_gate_id))
