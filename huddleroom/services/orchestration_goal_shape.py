import hashlib
import json
import logging
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Literal
from uuid import UUID

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select

from huddleroom.config import settings
from huddleroom.models.artifact import Artifact
from huddleroom.models.orchestration import (
    EXPLICIT_PLANNING_MODE_BY_GOAL_TYPE,
    GOAL_CONTRACT_FINGERPRINT_FIELDS,
    OrchestrationAction,
    OrchestrationBudgetReservation,
    OrchestrationContinuousCandidate,
    OrchestrationEvidence,
    OrchestrationGate,
    OrchestrationGoal,
    OrchestrationRoadmapVersion,
    OrchestrationRun,
    compute_goal_contract_fingerprint,
    explicit_planning_rationale,
)
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import (
    GoalShapeDecision,
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

    # --- Typed correction (Task 14) ---

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

    @staticmethod
    def contract_unchanged_since_start(goal: OrchestrationGoal, run: OrchestrationRun) -> bool:
        """True only when the goal contract differs from the run's Start freeze by type/mode alone (R14-3)."""
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

    async def has_type_specific_lineage(self, db, goal: OrchestrationGoal, run: OrchestrationRun) -> bool:
        """True once the goal holds plan, roadmap, Continuous, or child lineage that a correction must supersede."""
        plan_state = _obj(run.plan_state)
        if plan_state.get("planning_version") or plan_state.get("plan_artifact_id"):
            return True
        if _obj(goal.continuous_state).get("started_at"):
            return True
        # Roadmap items imply a version; continuous-origin children/reservations are covered by parent_goal_id.
        for probe in (
            select(OrchestrationRoadmapVersion.id).where(OrchestrationRoadmapVersion.goal_id == goal.id),
            select(OrchestrationContinuousCandidate.id).where(OrchestrationContinuousCandidate.parent_goal_id == goal.id),
            select(OrchestrationGoal.id).where(OrchestrationGoal.parent_goal_id == goal.id),
            select(OrchestrationBudgetReservation.id).where(OrchestrationBudgetReservation.parent_goal_id == goal.id),
        ):
            if await db.scalar(probe.limit(1)) is not None:
                return True
        return False

    async def is_pre_start(self, db, goal: OrchestrationGoal, run: OrchestrationRun) -> bool:
        """Pre-Start: never authorized, still in baseline/ready, and no type-specific lineage."""
        if run.phase not in ("baseline", "ready"):
            return False
        authorized = await db.scalar(select(OrchestrationAction.id).where(
            OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "authorize_execution",
        ).limit(1))
        return authorized is None and not await self.has_type_specific_lineage(db, goal, run)
