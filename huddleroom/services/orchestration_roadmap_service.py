import hashlib
import uuid
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from graphlib import CycleError, TopologicalSorter

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import func, inspect as sa_inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import object_session

from huddleroom.config import settings
from huddleroom.models.artifact import Artifact
from huddleroom.models.agent import Agent
from huddleroom.services.project_service import ProjectService
from huddleroom.models.base import _utcnow
from huddleroom.models.project import Project
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationBudgetReservation,
    OrchestrationEvidence,
    OrchestrationGate,
    OrchestrationGoal,
    OrchestrationRoadmapItem,
    OrchestrationRoadmapVersion,
    OrchestrationRun,
    EXPLICIT_PLANNING_MODE_BY_GOAL_TYPE,
    compute_goal_contract_fingerprint,
    explicit_planning_rationale,
)
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import (
    OrchestrationGoalCreate, OrchestrationPlanItem, OrchestrationRoadmapPlanItem, ROADMAP_BUDGET_DIMENSIONS,
)
from huddleroom.services.orchestration_capacity import (
    CAPACITY_WAIT, dependency_counts, rank_queued, root_and_depth, root_goal_id,
)
from huddleroom.services.orchestration_goal_shape import GoalShapeResolutionError, OrchestrationGoalShapeService
from huddleroom.services.orchestration_budget_service import (
    BudgetMeasurementError, OrchestrationBudgetService, canonical_amounts, parent_caps, validate_declared_allocations,
)
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_dependencies import RETAINED_GATE_TYPE
from huddleroom.services.event_bus import emit_event_once
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

def retained_gate_id(run_id, source_gate_id) -> uuid.UUID:
    """Successor gate retaining source_gate_id; equals _ensure_keyed_gate(run_id, RETAINED_GATE_TYPE, str(source_gate_id))."""
    return uuid.uuid5(uuid.NAMESPACE_URL, f"rally:orchestration:gate:{run_id}:{RETAINED_GATE_TYPE}:{source_gate_id}")


class ChildShapeRejected(Exception):
    """A child item resolved to a shape a finite Roadmap may not create."""


class RoadmapReplanMeasurementAttention(HTTPException):
    def __init__(self) -> None:
        super().__init__(status_code=409, detail="Roadmap budget measurement needs attention")


def parse_roadmap_items(raw: object, parent_dimensions: set[str]) -> list[OrchestrationRoadmapPlanItem]:
    if not isinstance(raw, list) or not raw:
        raise HTTPException(status_code=409, detail="Accepted Roadmap plan must include metadata.plan_items")
    items: list[OrchestrationRoadmapPlanItem] = []
    by_key: dict[str, OrchestrationRoadmapPlanItem] = {}
    for index, value in enumerate(raw):
        try:
            item = OrchestrationRoadmapPlanItem.model_validate(value)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid Roadmap item at index {index}: {exc.errors()[0]['msg']}") from exc
        if item.item_key in by_key:
            raise HTTPException(status_code=409, detail=f"Duplicate roadmap item_key '{item.item_key}'")
        for dimension, amount in item.allocation.items():
            if dimension not in ROADMAP_BUDGET_DIMENSIONS or dimension not in parent_dimensions:
                raise HTTPException(status_code=409, detail=f"Unsupported measured budget dimension '{dimension}'")
            try:
                parsed_amount = Decimal(str(amount))
            except (InvalidOperation, ValueError) as exc:
                raise HTTPException(status_code=422, detail=f"Invalid allocation for '{dimension}'") from exc
            if isinstance(amount, bool) or not parsed_amount.is_finite() or parsed_amount < 0:
                raise HTTPException(status_code=422, detail=f"Invalid allocation for '{dimension}'")
        if item.unit_type == "goal" and set(item.allocation) != parent_dimensions:
            raise HTTPException(status_code=409, detail="Roadmap child allocation must include every parent budget dimension")
        if item.unit_type == "goal" and not any(Decimal(str(amount)) > 0 for amount in item.allocation.values()):
            raise HTTPException(status_code=409, detail="Roadmap child allocation must reserve nonzero budget capacity")
        items.append(item)
        by_key[item.item_key] = item
    graph = {item.item_key: list(item.depends_on) for item in items}
    for key, dependencies in graph.items():
        for dependency in dependencies:
            if dependency not in graph:
                raise HTTPException(status_code=409, detail=f"Roadmap item '{key}' depends on unknown item '{dependency}'")
            if dependency == key:
                raise HTTPException(status_code=409, detail=f"Roadmap item '{key}' cannot depend on itself")
    try:
        tuple(TopologicalSorter(graph).static_order())
    except CycleError as exc:
        raise HTTPException(status_code=409, detail="Roadmap dependency cycle detected") from exc
    return items


async def expansion_scope(db, goal) -> tuple[OrchestrationGoal, int, dict]:
    """Finite root (nearest ancestor-or-self whose parent is None or Continuous), goal's depth below it, root limits."""
    root_id, depth = await root_and_depth(db, goal.id)
    root = goal if root_id == goal.id else await db.get(OrchestrationGoal, root_id)
    policy = (root.orchestrator_context or {}).get("expansion_policy") or {}
    limits = {
        "max_concurrent_tasks": settings.orchestration_max_concurrent_tasks,
        "max_goal_depth": settings.orchestration_max_goal_depth,
        "max_descendant_goals": settings.orchestration_max_descendant_goals,
    }
    limits.update({key: value for key, value in policy.items() if key in limits})
    return root, depth, limits


async def subtree_goal_ids(db, root_id) -> list[uuid.UUID]:
    ids, frontier = [root_id], [root_id]
    while frontier:
        frontier = list(await db.scalars(select(OrchestrationGoal.id).where(
            OrchestrationGoal.parent_goal_id.in_(frontier)
        )))
        ids.extend(frontier)
    return ids


async def descendant_count(db, root_id) -> int:
    return len(await subtree_goal_ids(db, root_id)) - 1


def expansion_limit_error(parent_depth: int, existing: int, new_children: int, limits: dict) -> str | None:
    if new_children <= 0:
        return None
    if parent_depth + 1 > limits["max_goal_depth"]:
        return (
            f"expansion_limit: child depth {parent_depth + 1} exceeds max_goal_depth {limits['max_goal_depth']}; "
            "replan: make this item an Outcome task or merge it into its parent"
        )
    if existing + new_children > limits["max_descendant_goals"]:
        return (
            f"expansion_limit: {existing + new_children} descendant goals exceed max_descendant_goals "
            f"{limits['max_descendant_goals']}; replan with fewer goal items"
        )
    return None


class OrchestrationRoadmapService:
    def __init__(self, orchestration_service):
        self.orchestration = orchestration_service

    async def _expansion_limit_error(self, db, goal, new_children: int) -> str | None:
        if new_children <= 0:
            return None
        root, depth, limits = await expansion_scope(db, goal)
        return expansion_limit_error(depth, await descendant_count(db, root.id), new_children, limits)

    async def _child_shape(self, goal, item) -> dict:
        if item.goal_type is None:
            return {
                "goal_type": "outcome", "goal_type_source": "parent_explicit",
                "goal_type_rationale": f"Roadmap item {item.item_key} runs as a leaf Outcome.",
                "planning_mode": "implementation",
                "planning_rationale": "A leaf roadmap item needs an implementation plan.",
            }
        if item.goal_type == "auto":
            shape = (await OrchestrationGoalShapeService().resolve_creation_shape(
                goal.project_id, OrchestrationGoalCreate(
                    goal_type="auto", objective=item.objective or item.title,
                    success_criteria=deepcopy(item.success_criteria),
                    constraints={**goal.constraints, **item.constraints},
                ),
            )).model_dump()
            if shape["goal_type"] == "continuous":
                raise ChildShapeRejected(
                    "continuous_child_requires_continuous_root: a finite Roadmap cannot create a Continuous child"
                )
            return {**shape, "goal_type_source": "parent_auto"}
        return {
            "goal_type": item.goal_type, "goal_type_source": "parent_explicit",
            "goal_type_rationale": f"Roadmap item {item.item_key} declares goal_type {item.goal_type}.",
            "planning_mode": EXPLICIT_PLANNING_MODE_BY_GOAL_TYPE[item.goal_type],
            "planning_rationale": explicit_planning_rationale(item.goal_type),
        }

    async def _require_expansion_limits(self, db, goal, items) -> None:
        released = set(await db.scalars(select(OrchestrationRoadmapItem.item_key).where(
            OrchestrationRoadmapItem.goal_id == goal.id
        )))
        reason = await self._expansion_limit_error(db, goal, sum(
            1 for item in items if item.unit_type == "goal" and item.item_key not in released
        ))
        if reason is not None:
            raise HTTPException(status_code=409, detail=reason)

    async def _child_expansion_context(self, db, goal, item, root, depth, limits, existing) -> dict:
        context = {"expansion_policy": {
            **limits, "depth": depth + 1, "root_goal_id": str(root.id),
            "remaining_descendant_goals": limits["max_descendant_goals"] - (existing + 1),
        }}
        evidence_ids = list(await db.scalars(select(OrchestrationEvidence.id).join(
            OrchestrationRoadmapItem, OrchestrationRoadmapItem.gate_id == OrchestrationEvidence.gate_id,
        ).where(
            OrchestrationRoadmapItem.goal_id == goal.id,
            OrchestrationRoadmapItem.item_key.in_(item.depends_on),
            OrchestrationEvidence.verdict == "accepted",
        ).order_by(OrchestrationEvidence.created_at, OrchestrationEvidence.id)))
        artifact_ids = [str(x) for x in item.artifact_ids]
        if artifact_ids or evidence_ids:
            context["eligible_evidence"] = {"artifact_ids": artifact_ids, "evidence_ids": [str(x) for x in evidence_ids]}
        return context

    async def current_version(self, db: AsyncSession, goal_id: uuid.UUID) -> OrchestrationRoadmapVersion | None:
        # Same ordering as get_run_for_goal: a corrected successor never inherits its predecessor's versions.
        latest_run = select(OrchestrationRun.id).where(OrchestrationRun.goal_id == goal_id).order_by(
            OrchestrationRun.started_at.desc(), OrchestrationRun.id.desc()
        ).limit(1).scalar_subquery()
        return await db.scalar(select(OrchestrationRoadmapVersion).where(
            OrchestrationRoadmapVersion.goal_id == goal_id,
            OrchestrationRoadmapVersion.run_id == latest_run,
        ).order_by(OrchestrationRoadmapVersion.version.desc()).limit(1))

    async def plan_fingerprint(self, db, run, items: list[dict]) -> str:
        """Successor runs salt the plan hash so versions and approvals never alias their predecessor's."""
        fingerprint = self.orchestration._accepted_plan_fingerprint(items)
        if sa_inspect(run).expired:
            await db.refresh(run)  # a rolled-back savepoint expired it; never lazy-load in async
        if run.supersedes_run_id is None:
            return fingerprint
        return hashlib.sha256(f"{run.id}:{fingerprint}".encode()).hexdigest()

    async def item_gate_satisfied(self, db, run, row) -> bool:
        """Released item accepted in this run, or a predecessor's item retained by an accepted successor gate."""
        gate = await db.get(OrchestrationGate, row.gate_id)
        if gate is not None and gate.run_id == run.id:
            metadata = self.orchestration._json_object_or_empty(gate.required_evidence)
            return (
                gate.status == "accepted"
                and gate.gate_type == ("work_completed" if row.unit_type == "task" else "child_goal_completed")
                and gate.success_criterion_key == f"roadmap_item:{row.item_key}"
                and metadata.get("roadmap_item_key") == row.item_key
                and metadata.get("roadmap_version_id") == str(row.first_version_id)
            )
        retained = await db.get(OrchestrationGate, retained_gate_id(run.id, row.gate_id))
        return retained is not None and retained.run_id == run.id and retained.status == "accepted"

    async def _require_historical_keys(self, db, goal, run, items) -> None:
        """A successor may reuse a predecessor's item key only unchanged and with accepted retained evidence."""
        candidate = {item.item_key: item.model_dump(mode="json") for item in items}
        rows = (await db.scalars(select(OrchestrationRoadmapItem).join(
            OrchestrationGate, OrchestrationRoadmapItem.gate_id == OrchestrationGate.id,
        ).where(
            OrchestrationRoadmapItem.goal_id == goal.id,
            OrchestrationRoadmapItem.item_key.in_(list(candidate)),
            OrchestrationGate.run_id != run.id,
        ))).all()
        for row in rows:
            if candidate[row.item_key] != row.item_snapshot or not await self.item_gate_satisfied(db, run, row):
                raise HTTPException(
                    status_code=409,
                    detail=f"Historical roadmap item '{row.item_key}' needs an unchanged snapshot and accepted retained evidence",
                )

    async def reconcile_claim_blockers(self, db, goal, run) -> None:
        """Clear only this run's recovered claim refusal, independent of its parent."""
        budget = OrchestrationBudgetService()
        parent = await budget.roadmap_parent(db, goal)
        if parent is None:
            return
        try:
            if goal.id == parent.id:
                remaining = (await budget.remaining(db, parent))["remaining"]
            else:
                caps = canonical_amounts(
                    (run.budget_state or {}).get("caps", goal.budget.get("caps", {})),
                    allowed=ROADMAP_BUDGET_DIMENSIONS,
                )
                remaining = await budget.run_remaining(db, run, caps)
        except BudgetMeasurementError:
            return
        if all(Decimal(value) > 0 for value in remaining.values()):
            run.active_blockers = [
                blocker for blocker in (run.active_blockers or [])
                if not (isinstance(blocker, dict) and blocker.get("scope") == f"claim:{goal.id}")
            ]

    async def remaining_or_block(self, db, goal, run):
        """A missing parent measurement is an attention state, never free budget."""
        try:
            summary = await OrchestrationBudgetService().remaining(db, goal)
        except BudgetMeasurementError as exc:
            self.orchestration._upsert_active_blocker(run, {
                "kind": "budget_measurement", "dimension": exc.dimension,
                "session_id": str(exc.session_id),
                "scope": f"parent:{goal.id}",
            })
            await db.flush()
            return None
        blockers = [item for item in (run.active_blockers or []) if not (
            isinstance(item, dict)
            and item.get("kind") == "budget_measurement"
            and item.get("scope") == f"parent:{goal.id}"
        )]
        if blockers != run.active_blockers:
            run.active_blockers = blockers
        await self.reconcile_claim_blockers(db, goal, run)
        return summary

    async def cascade_cancel(self, db, goal, *, cancelled_by: str) -> None:
        """Cancel unfinished descendants and conservatively settle their reservations."""
        children = list((await db.scalars(select(OrchestrationGoal).where(
            OrchestrationGoal.parent_goal_id == goal.id,
            OrchestrationGoal.status.not_in(("completed", "cancelled")),
        ))).all())
        for child in children:
            child_run = await self.orchestration.get_run_for_goal(db, child.project_id, child.id)
            await self.orchestration._cancel_goal_no_commit(db, child, child_run, cancelled_by=cancelled_by)
        rows = list((await db.scalars(select(OrchestrationRoadmapItem).where(
            OrchestrationRoadmapItem.goal_id == goal.id,
            OrchestrationRoadmapItem.unit_type == "goal",
        ))).all())
        budget = OrchestrationBudgetService()
        for row in rows:
            reservation = await db.scalar(select(OrchestrationBudgetReservation).where(
                OrchestrationBudgetReservation.roadmap_item_id == row.id
            ))
            child = await db.get(OrchestrationGoal, row.child_goal_id)
            if reservation is not None and child is not None:
                child_run = await self.orchestration.get_run_for_goal(db, child.project_id, child.id)
                if child_run is not None:
                    await budget.settle_child(db, reservation, child_run, reason="cancelled")
                    if not reservation.measurement_complete:
                        parent_run = await self.orchestration.get_run_for_goal(db, goal.project_id, goal.id)
                        if parent_run is not None:
                            self.orchestration._upsert_active_blocker(parent_run, {
                                "kind": "budget_integrity", "item_key": row.item_key,
                                "reason": "Cancelled child budget measurement is incomplete.",
                            })

    async def _item_producer_ids(self, db, lineage) -> set[uuid.UUID]:
        producer_ids: set[uuid.UUID] = set()
        for row in lineage:
            if row.unit_type == "task" and row.task_id is not None:
                task = await db.get(Task, row.task_id)
                if task is not None and task.assigned_to is not None:
                    producer_ids.add(task.assigned_to)
            elif row.unit_type == "goal" and row.child_goal_id is not None:
                child = await db.get(OrchestrationGoal, row.child_goal_id)
                if child is not None and child.manager_agent_id is not None:
                    producer_ids.add(child.manager_agent_id)
        return producer_ids

    async def ensure_integration_gate(self, db, goal, run, version):
        items = parse_roadmap_items(version.snapshot["items"], set(parent_caps(goal)))
        lineage = list((await db.scalars(select(OrchestrationRoadmapItem).where(
            OrchestrationRoadmapItem.goal_id == goal.id,
            OrchestrationRoadmapItem.item_key.in_([item.item_key for item in items]),
        ))).all())
        if len(lineage) != len(items):
            return None
        for row in lineage:
            if not await self.item_gate_satisfied(db, run, row):
                return None
        gate_id = uuid.uuid5(uuid.NAMESPACE_URL, f"rally:roadmap-integration:{run.id}:{version.id}")
        gate = await db.get(OrchestrationGate, gate_id)
        if gate is not None:
            return gate
        producer_ids = await self._item_producer_ids(db, lineage)
        gate = OrchestrationGate(
            id=gate_id, run_id=run.id, success_criterion_key="roadmap_integration",
            gate_type="roadmap_integration", required_evidence={
                "required_source_types": ["verification"], "min_count": 1,
                "requires_independent_agent": True,
                "work_producer_agent_ids": sorted(str(value) for value in producer_ids),
                "success_criterion_keys": self.orchestration._declared_success_criterion_keys(goal),
                "roadmap_version_id": str(version.id),
            },
        )
        db.add(gate)
        await db.flush()
        return gate

    async def integration_gate_accepted(self, db, goal, run) -> bool:
        version = await self.current_version(db, goal.id)
        if version is None:
            return False
        gate = await db.get(
            OrchestrationGate,
            uuid.uuid5(uuid.NAMESPACE_URL, f"rally:roadmap-integration:{run.id}:{version.id}"),
        )
        return gate is not None and gate.status == "accepted"

    def parse_items(self, artifact, goal):
        return parse_roadmap_items(
            self.orchestration._json_object_or_empty(artifact.metadata_).get("plan_items"),
            set(parent_caps(goal)),
        )

    async def ensure_plan_approval(self, db, goal, run, artifact):
        items = self.parse_items(artifact, goal)
        await self._require_expansion_limits(db, goal, items)
        await self._require_historical_keys(db, goal, run, items)
        fingerprint = await self.plan_fingerprint(db, run, [item.model_dump(mode="json") for item in items])
        key = f"roadmap_plan:{fingerprint}"
        if goal.authority_model == "no_manager":
            gate = await db.scalar(select(OrchestrationGate).where(
                OrchestrationGate.run_id == run.id,
                OrchestrationGate.success_criterion_key == key,
                OrchestrationGate.gate_type == "roadmap_plan_approval",
            ))
            if gate is None:
                gate = OrchestrationGate(
                    id=uuid.uuid5(uuid.NAMESPACE_URL, f"rally:orchestration:gate:{run.id}:roadmap_plan_approval:{fingerprint}"),
                    run_id=run.id, success_criterion_key=key, gate_type="roadmap_plan_approval",
                    required_evidence={
                        "required_source_types": ["verification"], "min_count": 1,
                        "requires_independent_agent": True,
                        "work_producer_agent_id": str(artifact.created_by_agent),
                        "planning_task_id": str(artifact.linked_task_id),
                        "plan_artifact_id": str(artifact.id),
                    },
                )
                try:
                    async with db.begin_nested():
                        db.add(gate)
                        await db.flush()
                except IntegrityError:
                    if object_session(gate) is not None:
                        object_session(gate).expunge(gate)
                    gate = await db.scalar(select(OrchestrationGate).where(OrchestrationGate.id == gate.id))
                    if gate is None:
                        raise
            return gate
        authority = "human" if goal.authority_model == "human_manager" else "manager"
        agent_id = None if authority == "human" else goal.manager_agent_id
        if (authority == "human" and goal.manager_user_id is None) or (authority != "human" and agent_id is None):
            raise HTTPException(status_code=409, detail="Roadmap plan approval authority is invalid")
        answered = await db.scalar(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.goal_id == goal.id,
            OrchestrationAuthorityDecision.run_id == run.id,
            OrchestrationAuthorityDecision.decision_key == key,
            OrchestrationAuthorityDecision.status == "answered",
        ).order_by(OrchestrationAuthorityDecision.decided_at.desc()).limit(1))
        if answered is not None:
            return answered
        return await OrchestrationAuthorityDecisionService().create_pending(
            db, goal.id, decision_key=key, title="Approve Roadmap plan",
            question="Approve this immutable Roadmap plan version?", authority=authority,
            authority_agent_id=agent_id, options=[{"key": "approve"}, {"key": "reject"}], run_id=run.id,
        )

    async def bind_pending_replan(self, db, goal, run, artifact, approval):
        """Pin the one candidate a pending replan may dispatch or ingest."""
        pending = self.orchestration._json_object_or_empty(
            self.orchestration._json_object_or_empty(run.plan_state).get("pending_replan")
        )
        version = await self.current_version(db, goal.id)
        task = await self.orchestration._planning_task_for_run(
            db, run, self.orchestration._required_uuid(pending.get("task_id"), "planning_task_id")
        )
        if (
            version is None or pending.get("version_id") != str(version.id)
            or task.assigned_to != artifact.created_by_agent
        ):
            raise HTTPException(status_code=409, detail="Pending Roadmap replan binding is stale")
        fingerprint = await self.plan_fingerprint(db, run, [
            item.model_dump(mode="json") for item in self.parse_items(artifact, goal)
        ])
        required = self.orchestration._json_object_or_empty(
            approval.required_evidence if isinstance(approval, OrchestrationGate) else {}
        )
        if isinstance(approval, OrchestrationGate) and (
            approval.run_id != run.id or approval.gate_type != "roadmap_plan_approval"
            or approval.success_criterion_key != f"roadmap_plan:{fingerprint}"
            or required.get("planning_task_id") != str(task.id)
            or required.get("plan_artifact_id") != str(artifact.id)
            or required.get("work_producer_agent_id") != str(task.assigned_to)
        ):
            raise HTTPException(status_code=409, detail="Pending Roadmap replan gate binding does not match")
        binding = {
            "planner_agent_id": str(task.assigned_to), "artifact_id": str(artifact.id),
            "fingerprint": fingerprint,
            **({"gate_id": str(approval.id)} if isinstance(approval, OrchestrationGate) else {}),
        }
        for key, value in binding.items():
            if pending.get(key) not in {None, value}:
                raise HTTPException(status_code=409, detail="Pending Roadmap replan binding does not match")
        run.plan_state = {
            **self.orchestration._json_object_or_empty(run.plan_state),
            "pending_replan": {**pending, **binding},
        }
        await db.flush()
        return approval

    async def has_active_bound_verifier(self, db, run) -> bool:
        actions = list((await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id,
            OrchestrationAction.action_type == "request_verification",
            OrchestrationAction.status == "completed",
            OrchestrationAction.target_type == "task",
        ))).all())
        for action in actions:
            task = await db.get(Task, action.target_id)
            gate = await db.get(OrchestrationGate, self.orchestration._event_uuid(
                self.orchestration._json_object_or_empty(action.request).get("gate_id")
            ))
            if (
                task is not None and gate is not None and gate.run_id == run.id and gate.status == "open"
                and (task.status in {"backlog", "ready", "in_progress", "blocked"}
                     or await self.orchestration._verdict_correction_open(db, run, task))
                and await self.orchestration._outcome_verification_action_for_task(db, run, gate, task) == action
            ):
                return True
        return False

    async def accept_version(self, db, goal, run, artifact, approval_reference):
        items = self.parse_items(artifact, goal)
        previous = await self.current_version(db, goal.id)
        if previous is None:
            validate_declared_allocations(goal, items)
        normalized = [item.model_dump(mode="json") for item in items]
        fingerprint = await self.plan_fingerprint(db, run, normalized)
        existing = await db.scalar(select(OrchestrationRoadmapVersion).where(
            OrchestrationRoadmapVersion.goal_id == goal.id,
            OrchestrationRoadmapVersion.fingerprint == fingerprint,
        ))
        if existing is not None:
            if previous is not None and existing.id != previous.id:
                raise HTTPException(status_code=409, detail="Historical Roadmap fingerprint cannot replace the current version")
            return existing
        await self._require_historical_keys(db, goal, run, items)
        if previous is not None:
            await self._validate_replan(db, goal, run, previous, items)
            await self._require_expansion_approval(db, goal, run, previous, items, fingerprint)
        approval_reference = await self._require_plan_approval(
            db, goal, run, fingerprint, approval_reference
        )
        project = await db.get(Project, goal.project_id)
        # The accepted hierarchy is a contract, not a roster suggestion.  A replan
        # must retain V1's team even if the live hierarchy has since changed.
        ctx = self.orchestration._json_object_or_empty(goal.orchestrator_context)
        inherited = ctx.get("team")
        hierarchy = None if previous is not None or (isinstance(inherited, dict) and inherited) else await OrchestrationProcessService().get_current(db, goal.id, "team_hierarchy")
        team = deepcopy(previous.snapshot.get("team") if previous is not None else (
            inherited if isinstance(inherited, dict) and inherited else (hierarchy.outputs if hierarchy is not None else None)
        ))
        agent_ids = self.orchestration._roadmap_team_agent_ids(team)
        outsiders = sorted(str(item.agent_id) for item in items if item.agent_id is not None and agent_ids is not None and str(item.agent_id) not in agent_ids)
        if outsiders:
            raise HTTPException(status_code=409, detail="Roadmap item agent is outside the accepted team")
        if agent_ids is not None:
            team = dict(team or {})
            team["agent_ids"] = sorted(agent_ids)
        previous_items = {
            item["item_key"]: item for item in previous.snapshot.get("items", [])
            if isinstance(item, dict) and isinstance(item.get("item_key"), str)
        } if previous is not None else {}
        previous_bindings = previous.snapshot.get("workspace_bindings", {}) if previous is not None else {}
        workspace_bindings = {}
        for item in items:
            if item.mutates_shared_state and item.staging_boundary is not None:
                item_snapshot = item.model_dump(mode="json")
                binding = previous_bindings.get(item.item_key) if isinstance(previous_bindings, dict) else None
                if previous_items.get(item.item_key) == item_snapshot and isinstance(binding, dict):
                    # A replan may not redirect unchanged unstarted mutable work
                    # through today's project configuration.
                    workspace_bindings[item.item_key] = deepcopy(binding)
                    continue
                try:
                    path = await ProjectService().require_roadmap_workspace(
                        db, goal.project_id, item.staging_boundary
                    )
                except HTTPException:
                    # Unsupported execution still reaches the exact human authority
                    # wait; only a successfully accepted concrete target is frozen.
                    continue
                workspace_bindings[item.item_key] = {
                    "boundary": deepcopy(item.staging_boundary), "path": str(path),
                }
        version = OrchestrationRoadmapVersion(
            goal_id=goal.id, run_id=run.id, version=1 + (await db.scalar(select(func.max(
                OrchestrationRoadmapVersion.version
            )).where(OrchestrationRoadmapVersion.goal_id == goal.id)) or 0),
            plan_artifact_id=artifact.id, snapshot={
                "schema_version": 1, "items": normalized, "team": team,
                "workspace_policy": deepcopy(previous.snapshot.get("workspace_policy", {})) if previous else (
                    # A nested Roadmap freezes its parent's policy, not today's project config.
                    deepcopy(ctx["workspace_policy"]) if goal.parent_goal_id is not None and "workspace_policy" in ctx
                    else deepcopy((project.config or {}).get("workspace_policy", {})) if project else {}
                ),
                "workspace_bindings": workspace_bindings,
            },
            fingerprint=fingerprint, approval_reference=approval_reference,
        )
        db.add(version)
        await db.flush()
        for item in items:
            row = await db.scalar(select(OrchestrationRoadmapItem).where(
                OrchestrationRoadmapItem.goal_id == goal.id, OrchestrationRoadmapItem.item_key == item.item_key,
            ))
            if row is not None and row.unit_type == "task":
                await self._refresh_released_task(db, run, version, item, row)
        return version

    async def _released_task_metadata(self, db, run, item, row, task):
        metadata = self.orchestration._json_object_or_empty(task.metadata_)
        current = self.orchestration._json_object_or_empty(metadata.get("orchestration_plan_item"))
        if item.adopt_task_id is not None and "adopt_task_id" not in current:
            raise HTTPException(status_code=409, detail="Released roadmap adoption metadata is missing")
        if current.get("adopt_task_id") is not None or item.adopt_task_id is None:
            return metadata
        first = await db.get(OrchestrationRoadmapVersion, row.first_version_id)
        marker = await self.orchestration._existing_action_for_key(
            db, run.id, f"run:{run.id}:kind:adopt_task:task:{task.id}"
        )
        orchestration = self.orchestration._json_object_or_empty(metadata.get("orchestration"))
        if (row.unit_type == "task" and row.goal_id == run.goal_id and row.item_key == item.item_key
                and row.task_id == task.id == item.adopt_task_id
                and row.item_snapshot.get("adopt_task_id") == str(task.id)
                and first is not None and first.run_id == run.id and first.goal_id == row.goal_id
                and any(raw.get("item_key") == row.item_key and raw.get("adopt_task_id") == str(task.id)
                        for raw in first.snapshot.get("items", []))
                and current.get("id") == row.item_key
                and orchestration.get("roadmap_item_key") == row.item_key
                and orchestration.get("roadmap_version_id") == str(first.id)
                and orchestration.get("run_id") == str(run.id)
                and orchestration.get("goal_id") == str(run.goal_id)
                and marker is not None and marker.status == "completed" and marker.target_type == "task"
                and marker.target_id == task.id and marker.request.get("plan_item_id") == row.item_key
                and marker.request.get("adopt_task_id") == str(task.id)):
            return {**metadata, "orchestration_plan_item": {**current, "adopt_task_id": str(task.id)}}
        return metadata

    async def _refresh_released_task(self, db, run, version, item, row):
        first_version = await db.get(OrchestrationRoadmapVersion, row.first_version_id)
        if first_version is not None and first_version.run_id != run.id:
            return  # retained predecessor work stays under its original run contract
        current_version = await self.current_version(db, row.goal_id)
        if current_version is None or current_version.id != version.id or version.run_id != run.id:
            return
        candidate = item.model_dump(mode="json")
        if {k: v for k, v in candidate.items() if k not in {"priority", "artifact_ids"}} != {
            k: v for k, v in row.item_snapshot.items() if k not in {"priority", "artifact_ids"}
        }:
            raise HTTPException(status_code=409, detail=f"Released roadmap item '{item.item_key}' cannot be changed")
        if item.adopt_task_id is not None and (row.task_id != item.adopt_task_id
                or row.item_snapshot.get("adopt_task_id") != str(row.task_id)):
            raise HTTPException(status_code=409, detail="Released roadmap adopted task binding changed")
        task = await db.get(Task, row.task_id)
        if task is None:
            raise HTTPException(status_code=409, detail="Released roadmap task is missing")
        metadata = await self._released_task_metadata(db, run, item, row, task)
        current = self.orchestration._json_object_or_empty(metadata.get("orchestration_plan_item"))
        self.orchestration._validate_bound_plan_item(task, run, item, metadata)
        if task.project_id != (await db.get(OrchestrationGoal, row.goal_id)).project_id:
            raise HTTPException(status_code=409, detail="Released roadmap task project changed")
        task.priority = item.priority
        task.metadata_ = {
            **metadata,
            "orchestration": {**self.orchestration._json_object_or_empty(metadata.get("orchestration")),
                              "roadmap_version_id": str(version.id)},
            "orchestration_plan_item": {**current, "priority": item.priority,
                                        "artifact_ids": [str(value) for value in item.artifact_ids],
                                        "accepted_plan_artifact_id": str(version.plan_artifact_id),
                                        "accepted_plan_fingerprint": version.fingerprint},
        }
        await db.flush()

    async def _validate_replan(self, db, goal, run, previous, candidate_items):
        # Released keys always stay in later versions, so previous's keys are this run's releases plus
        # reused history; a successor never has to carry predecessor items it did not reuse.
        released = {
            row.item_key: row
            for row in (await db.scalars(select(OrchestrationRoadmapItem).where(
                OrchestrationRoadmapItem.goal_id == goal.id,
                OrchestrationRoadmapItem.item_key.in_([item["item_key"] for item in previous.snapshot["items"]]),
            ))).all()
        }
        candidate = {item.item_key: item.model_dump(mode="json") for item in candidate_items}
        for key, row in released.items():
            if key not in candidate:
                raise HTTPException(status_code=409, detail=f"Released roadmap item '{key}' cannot be removed")
            refresh_fields = {"priority", "artifact_ids"} if row.unit_type == "task" else set()
            if {k: v for k, v in candidate[key].items() if k not in refresh_fields} != {
                k: v for k, v in row.item_snapshot.items() if k not in refresh_fields
            }:
                raise HTTPException(status_code=409, detail=f"Released roadmap item '{key}' cannot be changed")
        await self._validate_candidate_allocations(db, goal, run, candidate_items, released)

    async def _validate_candidate_allocations(self, db, goal, run, items, released):
        remaining = await self.remaining_or_block(db, goal, run)
        if remaining is None:
            raise RoadmapReplanMeasurementAttention()
        remaining = remaining["remaining"]
        totals = {key: Decimal("0") for key in remaining}
        for item in items:
            if item.unit_type == "goal" and item.item_key not in released:
                for key, amount in item.allocation.items():
                    totals[key] += Decimal(str(amount))
        if any(total > Decimal(remaining[key]) for key, total in totals.items()):
            raise HTTPException(status_code=409, detail="Roadmap child allocation exceeds remaining parent budget")

    async def _require_expansion_approval(self, db, goal, run, previous, items, fingerprint):
        # Limit rejection precedes any expansion authority decision.
        await self._require_expansion_limits(db, goal, items)
        prior = {item["item_key"]: item for item in previous.snapshot["items"]}
        expansion = any(
            item.unit_type == "goal" and item.item_key not in prior
            or any(Decimal(str(amount)) > Decimal(str(prior.get(item.item_key, {}).get("allocation", {}).get(key, 0)))
                   for key, amount in item.allocation.items())
            for item in items
        )
        if not expansion:
            return
        key = f"roadmap_expansion:{fingerprint}"
        decision = await db.scalar(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.goal_id == goal.id,
            OrchestrationAuthorityDecision.run_id == run.id,
            OrchestrationAuthorityDecision.decision_key == key,
        ).order_by(OrchestrationAuthorityDecision.created_at.desc()).limit(1))
        authorized_user = goal.manager_user_id or goal.created_by_user_id
        if authorized_user is None:
            raise HTTPException(status_code=409, detail="waiting_human_scope_budget_authority")
        if decision is not None and decision.authority == "human" and decision.status == "answered" and decision.selected_option == "approve" and decision.decided_by_user_id == authorized_user:
            return
        if decision is None:
            await OrchestrationAuthorityDecisionService().create_pending(
                db, goal.id, decision_key=key, title="Approve Roadmap expansion",
                question="Approve the new Roadmap scope or budget allocation?", authority="human",
                options=[{"key": "approve"}, {"key": "reject"}], run_id=run.id,
            )
        raise HTTPException(status_code=409, detail="waiting_human_scope_budget_authority")

    async def prepare_replan_acceptance(self, db, goal, run, artifact):
        """Persist authority waits before an accept action is reserved."""
        previous = await self.current_version(db, goal.id)
        items = self.parse_items(artifact, goal)
        if previous is None:
            return None
        fingerprint = await self.plan_fingerprint(db, run, [item.model_dump(mode="json") for item in items])
        await self._validate_replan(db, goal, run, previous, items)
        try:
            await self._require_expansion_approval(db, goal, run, previous, items, fingerprint)
        except HTTPException as exc:
            if exc.detail == "waiting_human_scope_budget_authority":
                await db.flush()
            raise
        approval = await self.bind_pending_replan(
            db, goal, run, artifact, await self.ensure_plan_approval(db, goal, run, artifact)
        )
        if isinstance(approval, OrchestrationAuthorityDecision) and approval.status != "answered":
            await db.flush()
            raise HTTPException(status_code=409, detail="waiting_plan_authority")
        if isinstance(approval, OrchestrationGate) and approval.status != "accepted":
            await db.flush()
            raise HTTPException(status_code=409, detail="waiting_plan_authority")
        await self._require_plan_approval(db, goal, run, fingerprint, None)
        return approval

    async def _require_plan_approval(self, db, goal, run, fingerprint, approval_reference):
        if goal.authority_model == "no_manager":
            gate = await db.scalar(select(OrchestrationGate).where(
                OrchestrationGate.run_id == run.id,
                OrchestrationGate.success_criterion_key == f"roadmap_plan:{fingerprint}",
                OrchestrationGate.gate_type == "roadmap_plan_approval",
                OrchestrationGate.status == "accepted",
            ))
            if gate is None:
                raise HTTPException(status_code=409, detail="Roadmap plan approval is required")
            evidence = list((await db.scalars(select(OrchestrationEvidence).where(
                OrchestrationEvidence.gate_id == gate.id,
                OrchestrationEvidence.verdict == "accepted",
                OrchestrationEvidence.source_type == "verification",
            ))).all())
            if await self.orchestration._independence_failure_reason(db, gate, evidence) is not None:
                raise HTTPException(status_code=409, detail="Roadmap plan approval is required")
            return {"kind": "gate", "id": str(gate.id)}

        decision = await db.scalar(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.goal_id == goal.id,
            OrchestrationAuthorityDecision.run_id == run.id,
            OrchestrationAuthorityDecision.decision_key == f"roadmap_plan:{fingerprint}",
        ).order_by(OrchestrationAuthorityDecision.created_at.desc()).limit(1))
        if decision is None or decision.status != "answered" or decision.selected_option != "approve":
            raise HTTPException(status_code=409, detail="Roadmap plan approval is required")
        if goal.authority_model == "human_manager":
            if goal.manager_user_id is None or decision.decided_by_user_id != goal.manager_user_id:
                raise HTTPException(status_code=409, detail="Roadmap plan approval is not from the selected authority")
        elif goal.authority_model == "agent_manager":
            if goal.manager_agent_id is None or decision.decided_by_agent_id != goal.manager_agent_id:
                raise HTTPException(status_code=409, detail="Roadmap plan approval is not from the selected authority")
        else:
            raise HTTPException(status_code=409, detail="Roadmap plan approval authority is invalid")
        return {"kind": "authority_decision", "id": str(decision.id)}

    async def _dependencies_accepted(self, db, goal_id, run, version, dependencies) -> bool:
        for key in dependencies:
            lineage = await db.scalar(select(OrchestrationRoadmapItem).where(
                OrchestrationRoadmapItem.goal_id == goal_id,
                OrchestrationRoadmapItem.item_key == key,
            ))
            if lineage is None or not await self.item_gate_satisfied(db, run, lineage):
                return False
        return True

    async def _mutable_item_active(self, db, goal_id, run) -> bool:
        return await db.scalar(select(OrchestrationRoadmapItem.id).join(
            OrchestrationGate, OrchestrationRoadmapItem.gate_id == OrchestrationGate.id,
        ).where(
            OrchestrationRoadmapItem.goal_id == goal_id,
            OrchestrationGate.run_id == run.id,
            OrchestrationRoadmapItem.item_snapshot["mutates_shared_state"].as_boolean().is_(True),
            OrchestrationGate.status != "accepted",
        ).limit(1)) is not None

    @staticmethod
    def _unstaged_mutation_key(version, item) -> str:
        return f"roadmap_unstaged_mutation:{version.id}:{item.item_key}"

    async def _mutation_authority(self, db, goal, run, version, item):
        key = self._unstaged_mutation_key(version, item)
        answered = await db.scalar(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.goal_id == goal.id,
            OrchestrationAuthorityDecision.run_id == run.id,
            OrchestrationAuthorityDecision.decision_key == key,
            OrchestrationAuthorityDecision.status == "answered",
        ).order_by(OrchestrationAuthorityDecision.decided_at.desc()).limit(1))
        if answered is not None:
            if answered.selected_option == "reject":
                self.orchestration._upsert_active_blocker(run, {
                    "kind": "staging_boundary",
                    "item_key": item.item_key,
                    "reason": "Human rejected unstaged mutable Roadmap work.",
                })
                await db.flush()
            return answered
        return await OrchestrationAuthorityDecisionService().create_pending(
            db, goal.id, decision_key=key, title="Approve unstaged Roadmap work",
            question=f"Approve mutable Roadmap item '{item.item_key}' without a reversible staging boundary?",
            authority="human", options=[{"key": "approve"}, {"key": "reject"}], run_id=run.id,
            context=(
                f"The adapter cannot stage or revert this work. Immutable item snapshot: "
                f"{item.model_dump_json()}"
            ),
        )

    async def _cli_budget_authority(self, db, goal, run, version, item):
        """Use the shared action-boundary authority before direct task lineage exists."""
        if item.unit_type != "task":
            return None
        try:
            task_item = OrchestrationPlanItem(
                id=item.item_key, title=item.title, work_function=item.work_function,
                scope=item.scope, deliverable=item.deliverable, agent_id=item.agent_id,
                inputs=item.inputs, forbidden_work=item.forbidden_work,
                success_evidence=item.success_evidence, required_capabilities=item.required_capabilities,
                depends_on=item.depends_on,
            )
            agent_id = await self.orchestration._plan_item_agent_id(
                db, goal.project_id, task_item,
                allowed_agent_ids=self.orchestration._roadmap_team_agent_ids(version.snapshot.get("team")),
            )
        except HTTPException:
            return None
        agent = await db.get(Agent, agent_id)
        if agent is None:
            return None
        result = await self.orchestration.roadmap_pre_release_authority(
            db, run.id, agent,
            {"roadmap": {"roadmap_version_id": str(version.id), "roadmap_item_key": item.item_key}},
        )
        return result or None

    async def first_releasable_item(self, db, goal, run, version):
        items = parse_roadmap_items(version.snapshot["items"], set(parent_caps(goal)))
        released = {
            row.item_key: row
            for row in (await db.scalars(select(OrchestrationRoadmapItem).where(
                OrchestrationRoadmapItem.goal_id == goal.id
            ))).all()
        }
        headroom = await self.orchestration._release_headroom(db, await root_goal_id(db, goal.id))
        for item in rank_queued(items, {item.item_key: index for index, item in enumerate(items)}, dependency_counts(items)):
            if item.item_key in released or not await self._dependencies_accepted(
                db, goal.id, run, version, item.depends_on
            ):
                continue
            if item.unit_type == "task" and headroom <= 0:
                continue
            if item.mutates_shared_state and await self._mutable_item_active(db, goal.id, run):
                continue
            if item.mutates_shared_state and await self._requires_mutation_authority(db, goal, version, item):
                if item.unit_type == "goal" and await self._expansion_limit_error(db, goal, 1):
                    return item  # release_goal_item records the expansion_limit blocker before any authority
                authority = await self._mutation_authority(db, goal, run, version, item)
                if authority.status == "answered" and authority.selected_option == "reject":
                    return None
                if authority.status != "answered" or authority.selected_option != "approve":
                    continue
            return item
        return None

    async def _requires_mutation_authority(self, db, goal, version, item) -> bool:
        if item.staging_boundary is None or item.unit_type != "task" or item.work_function is None:
            return True
        task_item = OrchestrationPlanItem(
            id=item.item_key, title=item.title, work_function=item.work_function,
            scope=item.scope, deliverable=item.deliverable, agent_id=item.agent_id,
            inputs=item.inputs, forbidden_work=item.forbidden_work,
            success_evidence=item.success_evidence, required_capabilities=item.required_capabilities,
            depends_on=item.depends_on,
        )
        try:
            agent_id = await self.orchestration._plan_item_agent_id(
                db, goal.project_id, task_item,
                allowed_agent_ids=self.orchestration._roadmap_team_agent_ids(version.snapshot.get("team")),
            )
        except HTTPException:
            return True
        agent = await db.get(Agent, agent_id)
        if agent is None or agent.adapter_type != "cli":
            return True
        try:
            binding = self._workspace_binding(version, item)
            if binding is None:
                return True
            await ProjectService().require_frozen_roadmap_workspace(
                db, goal.project_id, binding["boundary"], binding["path"]
            )
        except HTTPException:
            return True
        return False

    @staticmethod
    def _workspace_binding(version, item):
        binding = (version.snapshot.get("workspace_bindings", {}) or {}).get(item.item_key)
        if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
            return None
        return binding

    async def wait_reason(self, db, goal, run, version) -> str:
        if await db.scalar(select(OrchestrationRoadmapItem.id).join(
            OrchestrationGoal, OrchestrationRoadmapItem.child_goal_id == OrchestrationGoal.id,
        ).where(
            OrchestrationRoadmapItem.goal_id == goal.id,
            OrchestrationRoadmapItem.unit_type == "goal",
            OrchestrationRoadmapItem.completed_at.is_(None),
            OrchestrationGoal.status.not_in(("completed", "cancelled")),
        ).limit(1)) is not None:
            return "waiting_active_work"
        items = parse_roadmap_items(version.snapshot["items"], set(parent_caps(goal)))
        released = set((await db.scalars(select(OrchestrationRoadmapItem.item_key).where(
                OrchestrationRoadmapItem.goal_id == goal.id
            ))).all())
        headroom = await self.orchestration._release_headroom(db, await root_goal_id(db, goal.id))
        for item in rank_queued(items, {item.item_key: index for index, item in enumerate(items)}, dependency_counts(items)):
            if item.item_key in released:
                continue
            if not await self._dependencies_accepted(db, goal.id, run, version, item.depends_on):
                continue
            if item.unit_type == "task" and headroom <= 0:
                return CAPACITY_WAIT
            if item.mutates_shared_state and await self._mutable_item_active(db, goal.id, run):
                return "waiting_shared_workspace"
            if item.mutates_shared_state and await self._requires_mutation_authority(db, goal, version, item):
                authority = await self._mutation_authority(db, goal, run, version, item)
                if authority.status != "answered" or authority.selected_option != "approve":
                    return "waiting_unstaged_approval"
        return "waiting_dependencies"

    async def release_task_item(self, db, goal, run, version, item):
        existing = await db.scalar(select(OrchestrationRoadmapItem).where(
            OrchestrationRoadmapItem.goal_id == goal.id, OrchestrationRoadmapItem.item_key == item.item_key,
        ))
        if existing is not None:
            await self._refresh_released_task(db, run, version, item, existing)
            return {"step": "release_item", "item_key": item.item_key, "unit_type": "task"}
        release = await self.orchestration._existing_action_for_key(
            db, run.id, f"run:{run.id}:kind:release_roadmap_item:{item.item_key}"
        )
        if release is not None and release.status == "failed":
            return await self._adoption_failed(db, goal, run, release, item, release.error or "Adoption failed")
        unstaged = item.mutates_shared_state and await self._requires_mutation_authority(db, goal, version, item)
        if unstaged:
            authority = await self._mutation_authority(db, goal, run, version, item)
            if authority.status != "answered" or authority.selected_option != "approve":
                return {"step": "waiting", "reason": "waiting_unstaged_approval"}
        elif item.mutates_shared_state:
            # Resolve before reserving any task/gate/lineage state. The claim repeats
            # this check to close the time-of-check-to-launch gap.
            binding = self._workspace_binding(version, item)
            if binding is None:
                return {"step": "waiting", "reason": "waiting_unstaged_approval"}
            await ProjectService().require_frozen_roadmap_workspace(
                db, goal.project_id, binding["boundary"], binding["path"]
            )
        budget_authority = await self._cli_budget_authority(db, goal, run, version, item)
        if budget_authority and budget_authority["status"] == "pending":
            return {"step": "waiting", "reason": "budget_wait"}
        if budget_authority and budget_authority["status"] == "rejected":
            return {"step": "waiting", "reason": "needs_attention"}
        summary = await self.remaining_or_block(db, goal, run)
        if summary is None:
            return {"step": "waiting", "reason": "needs_attention"}
        exhausted = None if summary is None else next(
            (key for key, value in summary["remaining"].items() if Decimal(value) <= 0), None,
        )
        if exhausted is not None:
            self.orchestration._upsert_active_blocker(run, {
                "kind": "budget_integrity", "reason": f"Roadmap parent budget exhausted {exhausted}",
            })
            await db.flush()
            return {"step": "waiting", "reason": "needs_attention"}
        release_key = f"run:{run.id}:kind:release_roadmap_item:{item.item_key}"
        release = await self.orchestration.reserve_action(
            db, run.id, release_key, "release_roadmap_item", {"item_key": item.item_key, "unit_type": "task"},
        )
        gate_id = uuid.uuid5(
            uuid.NAMESPACE_URL, f"rally:orchestration:gate:{run.id}:roadmap_item:{version.id}:{item.item_key}"
        )
        gate = await db.get(OrchestrationGate, gate_id)
        task_item = OrchestrationPlanItem(
            id=item.item_key, title=item.title, work_function=item.work_function,
            scope=item.scope, deliverable=item.deliverable, agent_id=item.agent_id,
            inputs=item.inputs, forbidden_work=item.forbidden_work,
            success_evidence=item.success_evidence, required_capabilities=item.required_capabilities,
            depends_on=item.depends_on, priority=item.priority, artifact_ids=item.artifact_ids,
            why_runnable=item.why_runnable, adopt_task_id=item.adopt_task_id,
        )
        required_evidence = {
            "plan_item_id": item.item_key,
            "roadmap_item_key": item.item_key,
            "roadmap_version_id": str(version.id),
            "required_source_types": ["task", "verification"],
            "min_count": 2,
            "requires_independent_agent": True,
        }
        if gate is None:
            gate = OrchestrationGate(
                id=gate_id, run_id=run.id, success_criterion_key=f"roadmap_item:{item.item_key}",
                gate_type="work_completed", required_evidence=required_evidence,
            )
            db.add(gate)
            await db.flush()
        elif (
            gate.run_id != run.id or gate.gate_type != "work_completed"
            or gate.success_criterion_key != f"roadmap_item:{item.item_key}"
            or self.orchestration._json_object_or_empty(gate.required_evidence) != required_evidence
        ):
            raise HTTPException(status_code=409, detail="Roadmap item gate identity is invalid")
        artifact = await db.get(Artifact, version.plan_artifact_id)
        snapshot = await self._parent_contract_snapshot(db, goal, version, item)
        context = {
            "roadmap": {"roadmap_version_id": str(version.id), "roadmap_item_key": item.item_key,
                        "staging_boundary": None if unstaged else deepcopy(item.staging_boundary),
                        "mutates_shared_state": item.mutates_shared_state,
                        "no_publish_before_integration": True,
                        **({"workspace_path": self._workspace_binding(version, item)["path"]} if not unstaged and self._workspace_binding(version, item) else {}),
                        **({"unstaged_authority_decision_id": str(authority.id)} if unstaged else {})},
            "team": deepcopy(snapshot["team"]),
            "workspace_policy": deepcopy(snapshot["workspace_policy"]),
        }
        if budget_authority and budget_authority["status"] == "approved":
            context["roadmap"]["budget_approval"] = {
                "decision_id": budget_authority["decision_id"],
                "unsupported_dimensions": budget_authority["unsupported_dimensions"],
            }
        if item.adopt_task_id is not None:
            # Adopted discovery work keeps its task and session; no delegation is created.
            task = await db.get(Task, item.adopt_task_id)
            if task is None:
                return await self._adoption_failed(
                    db, goal, run, release, item, f"adoption_invalid: item {item.item_key}: adopted task not found"
                )
        else:
            released_task_keys = set((await db.scalars(select(OrchestrationRoadmapItem.item_key).where(
                OrchestrationRoadmapItem.goal_id == goal.id,
                OrchestrationRoadmapItem.unit_type == "task",
            ))).all())
            discretionary_divisor = max(1, sum(
                candidate.unit_type == "task" and candidate.item_key not in released_task_keys
                for candidate in parse_roadmap_items(version.snapshot["items"], set(parent_caps(goal)))
            ))
            delegation_request = await self.orchestration._plan_item_delegation_request(
                db, run.id, goal.project_id, artifact, task_item, orchestrator_context=context,
                discretionary_divisor=discretionary_divisor,
            )
            delegation = await self.orchestration.execute_create_delegation_task_action(
                db, run.id,
                delegation_request,
                f"run:{run.id}:kind:create_delegation_task:roadmap_item:{item.item_key}",
                gate_bound=True,
            )
            task = await db.get(Task, delegation.target_id)
        original_metadata = task.metadata_
        metadata = self.orchestration._json_object_or_empty(task.metadata_)
        orchestration = self.orchestration._json_object_or_empty(metadata.get("orchestration"))
        task.metadata_ = {**metadata, "orchestration": {
            **orchestration, "roadmap_version_id": str(version.id), "roadmap_item_key": item.item_key,
        }}
        if item.adopt_task_id is not None:
            try:
                await self.orchestration._bind_adopted_task(db, run, task, artifact, task_item, release, gate)
            except HTTPException as exc:
                task.metadata_ = original_metadata  # no half-bound task
                return await self._adoption_failed(db, goal, run, release, item, str(exc.detail))
        else:
            self.orchestration._attach_plan_item_metadata(task, artifact, task_item, release, gate)
        lineage = OrchestrationRoadmapItem(
            id=uuid.uuid5(uuid.NAMESPACE_URL, f"rally:roadmap-item:{goal.id}:{item.item_key}"),
            goal_id=goal.id, first_version_id=version.id, item_key=item.item_key,
            unit_type="task", item_snapshot=item.model_dump(mode="json"), task_id=task.id,
            child_goal_id=None, gate_id=gate.id,
        )
        db.add(lineage)
        await db.flush()
        await self.orchestration._mark_action_completed(db, release, target_type="task", target_id=task.id)
        return {"step": "release_item", "item_key": item.item_key, "unit_type": "task"}

    async def _adoption_failed(self, db, goal, run, release, item, reason):
        """Same as Outcome: fail the release and block on plan integrity instead of raising every tick."""
        version = await self.current_version(db, goal.id)
        if version is not None and version.run_id == run.id and item.adopt_task_id is not None:
            gate_id = uuid.uuid5(uuid.NAMESPACE_URL,
                                f"rally:orchestration:gate:{run.id}:roadmap_item:{version.id}:{item.item_key}")
            gate = await db.get(OrchestrationGate, gate_id)
            request = self.orchestration._json_object_or_empty(release.request)
            if (gate is not None and gate.status == "open" and gate.run_id == run.id
                    and gate.gate_type == "work_completed"
                    and gate.success_criterion_key == f"roadmap_item:{item.item_key}"
                    and gate.required_evidence.get("roadmap_version_id") == str(version.id)
                    and gate.required_evidence.get("plan_item_id") == item.item_key
                    and gate.required_evidence.get("task_id", str(item.adopt_task_id)) == str(item.adopt_task_id)
                    and any(raw.get("item_key") == item.item_key
                            and raw.get("adopt_task_id") == str(item.adopt_task_id)
                            for raw in version.snapshot.get("items", []))
                    and release.target_id in (None, item.adopt_task_id)
                    and request == {"item_key": item.item_key, "unit_type": "task"}
                    and release.idempotency_key == f"run:{run.id}:kind:release_roadmap_item:{item.item_key}"):
                self.orchestration._fail_gate(gate, reason)
        if release.status == "reserved":
            await self.orchestration._fail_reserved_action_for_current_flow(db, release, reason)
        self.orchestration._block_plan_integrity(
            goal, run, f"Adopted plan item '{item.item_key}' could not bind: {reason}"
        )
        await db.flush()
        return {"step": "needs_attention"}

    async def _parent_contract_snapshot(self, db, parent, version, item):
        team = deepcopy(version.snapshot.get("team"))
        agent_ids = self.orchestration._roadmap_team_agent_ids(team)
        if agent_ids is not None:
            team = dict(team or {})
            team["agent_ids"] = sorted(agent_ids)
        return {
            "snapshot_version": 1, "parent_goal_id": str(parent.id),
            "roadmap_version_id": str(version.id), "roadmap_item_key": item.item_key,
            "objective": parent.objective, "constraints": deepcopy(parent.constraints),
            "budget_policy": deepcopy(parent.budget),
            "authority": {
                "authority_model": parent.authority_model,
                "manager_agent_id": str(parent.manager_agent_id) if parent.manager_agent_id else None,
                "manager_user_id": str(parent.manager_user_id) if parent.manager_user_id else None,
            },
            "team": team,
            "success_criteria": deepcopy(parent.success_criteria),
            "workspace_policy": deepcopy(version.snapshot.get("workspace_policy", {})),
            **({"workspace_binding": deepcopy(self._workspace_binding(version, item))}
               if self._workspace_binding(version, item) else {}),
        }

    async def release_goal_item(self, db, goal, run, version, item, shape=None):
        existing = await db.scalar(select(OrchestrationRoadmapItem).where(
            OrchestrationRoadmapItem.goal_id == goal.id,
            OrchestrationRoadmapItem.item_key == item.item_key,
        ))
        if existing is not None:
            reservation = await db.scalar(select(OrchestrationBudgetReservation).where(
                OrchestrationBudgetReservation.roadmap_item_id == existing.id
            ))
            if reservation is None:
                raise HTTPException(status_code=409, detail="Roadmap child reservation is missing")
            return {"step": "release_item", "item_key": item.item_key, "unit_type": "goal"}
        from huddleroom.services.orchestration_service import LLM_DECISION_RUN_STATUSES
        current = await self.current_version(db, goal.id)
        if (goal.status not in {"active", "blocked"} or run.status not in LLM_DECISION_RUN_STATUSES
                or current is None or current.id != version.id):
            return {"step": "waiting", "reason": "child_shape_state_changed"}
        if not await self._dependencies_accepted(db, goal.id, run, version, item.depends_on):
            return {"step": "waiting", "reason": "waiting_dependencies"}
        summary = await self.remaining_or_block(db, goal, run)
        if summary is None:
            return {"step": "waiting", "reason": "needs_attention"}
        if shape is not None and any(Decimal(str(amount)) > Decimal(summary["remaining"].get(key, "0"))
               for key, amount in item.allocation.items()):
            return {"step": "waiting", "reason": "budget_wait"}
        # ponytail: backstop for cousin races and legacy versions; mirrors child_contract
        root, depth, limits = await expansion_scope(db, goal)
        existing_descendants = await descendant_count(db, root.id)
        reason = expansion_limit_error(depth, existing_descendants, 1, limits)
        if reason is not None:
            self.orchestration._upsert_active_blocker(run, {
                "kind": "expansion_limit", "item_key": item.item_key, "roadmap_version_id": str(version.id),
                "reason": reason, "required_decision": "request_roadmap_replan",
            })
            await db.flush()
            return {"step": "needs_attention"}
        for key, value in item.constraints.items():
            if key in goal.constraints and goal.constraints[key] != value:
                self.orchestration._upsert_active_blocker(run, {
                    "kind": "child_contract", "item_key": item.item_key, "roadmap_version_id": str(version.id),
                    "reason": f"Child constraint conflicts with parent: {key}",
                })
                await db.flush()
                return {"step": "needs_attention"}
        unstaged = item.mutates_shared_state and await self._requires_mutation_authority(db, goal, version, item)
        authority = None
        if unstaged:
            authority = await self._mutation_authority(db, goal, run, version, item)
            if authority.status != "answered" or authority.selected_option != "approve":
                return {"step": "waiting", "reason": "waiting_unstaged_approval"}
        elif item.mutates_shared_state:
            binding = self._workspace_binding(version, item)
            if binding is None:
                return {"step": "waiting", "reason": "waiting_unstaged_approval"}
            await ProjectService().require_frozen_roadmap_workspace(
                db, goal.project_id, binding["boundary"], binding["path"]
            )
        if shape is None:
            auto = item.goal_type == "auto"
            if auto and any(
                isinstance(blocker, dict) and blocker.get("kind") == "child_contract"
                and blocker.get("item_key") == item.item_key and blocker.get("roadmap_version_id") == str(version.id)
                and str(blocker.get("reason", "")).startswith("continuous_child_requires_continuous_root")
                for blocker in run.active_blockers
            ):
                return {"step": "needs_attention"}  # already rejected for this version; no re-classification
            # The classifier is a model round trip: release a tick-owned writer first (as request_llm_decision does).
            committed = auto and await self.orchestration._commit_tick_owned_transaction(db)
            try:
                shape = await self._child_shape(goal, item)
            except GoalShapeResolutionError as exc:
                return {"step": "waiting", "reason": f"child_shape_{exc.reason}"}
            except ChildShapeRejected as exc:
                if committed:
                    await db.refresh(run)
                self.orchestration._upsert_active_blocker(run, {
                    "kind": "child_contract", "item_key": item.item_key, "roadmap_version_id": str(version.id),
                    "reason": str(exc),
                })
                await db.flush()
                return {"step": "needs_attention"}
            if auto:
                from huddleroom.services.orchestration_service import LLM_DECISION_RUN_STATUSES
                # Other writers may have moved while the writer was released: re-read and re-run every check.
                if committed:
                    db.expire_all()  # ancestors, dependency gates and authority may also have changed
                    await db.refresh(goal)
                    await db.refresh(run)
                    await db.refresh(version)
                current = await self.current_version(db, goal.id)
                if run.status not in LLM_DECISION_RUN_STATUSES or current is None or current.id != version.id:
                    return {"step": "waiting", "reason": "child_shape_state_changed"}
                return await self.release_goal_item(db, goal, run, version, item, shape=shape)
        release = await self.orchestration.reserve_action(
            db, run.id, f"run:{run.id}:kind:release_roadmap_item:{item.item_key}",
            "release_roadmap_item", {"item_key": item.item_key, "unit_type": "goal"},
        )
        try:
            async with db.begin_nested():
                child_id = uuid.uuid5(uuid.NAMESPACE_URL, f"rally:roadmap-child:{goal.id}:{item.item_key}")
                child_run_id = uuid.uuid5(uuid.NAMESPACE_URL, f"rally:roadmap-child-run:{goal.id}:{item.item_key}")
                snapshot = await self._parent_contract_snapshot(db, goal, version, item)
                child = OrchestrationGoal(
                    id=child_id, project_id=goal.project_id,
                    objective=item.objective or item.title, original_request=item.objective or item.title,
                    success_criteria=deepcopy(item.success_criteria),
                    constraints={**deepcopy(goal.constraints), **deepcopy(item.constraints)},
                    budget={"caps": canonical_amounts(item.allocation, allowed=set(parent_caps(goal)))},
                    orchestrator_context={
                        "roadmap": {"roadmap_version_id": str(version.id), "roadmap_item_key": item.item_key,
                                    "staging_boundary": None if unstaged else deepcopy(item.staging_boundary),
                                    "mutates_shared_state": item.mutates_shared_state,
                                    "no_publish_before_integration": True,
                                    **({"workspace_path": self._workspace_binding(version, item)["path"]} if not unstaged and self._workspace_binding(version, item) else {}),
                                    **({"unstaged_authority_decision_id": str(authority.id)} if unstaged else {})},
                        "team": deepcopy(snapshot["team"]),
                        "workspace_policy": deepcopy(snapshot["workspace_policy"]),
                        **await self._child_expansion_context(
                            db, goal, item, root, depth, limits, existing_descendants,
                        ),
                    }, **shape,
                    parent_goal_id=goal.id, roadmap_version_id=version.id, roadmap_item_key=item.item_key,
                    parent_contract_snapshot=snapshot,
                    goal_delta={"objective": item.objective, "constraints": deepcopy(item.constraints),
                                "success_criteria": deepcopy(item.success_criteria)},
                    authority_model=goal.authority_model, manager_agent_id=goal.manager_agent_id,
                    manager_user_id=goal.manager_user_id,
                )
                child_run = OrchestrationRun(
                    id=child_run_id, goal_id=child.id, phase="authorized",
                    goal_type_snapshot=child.goal_type, planning_mode_snapshot=child.planning_mode,
                    goal_contract_fingerprint=compute_goal_contract_fingerprint(child),
                    budget_state={"caps": canonical_amounts(item.allocation, allowed=set(parent_caps(goal)))},
                    plan_state={"child_delta_baseline": {
                        "status": "accepted", "snapshot_version": 1, "parent_goal_id": str(goal.id),
                        "roadmap_version_id": str(version.id), "roadmap_item_key": item.item_key,
                        "approval_reference": deepcopy(version.approval_reference),
                    }},
                )
                db.add_all([child, child_run])
                await db.flush()
                gate = OrchestrationGate(
                    id=uuid.uuid5(uuid.NAMESPACE_URL, f"rally:orchestration:gate:{run.id}:roadmap_item:{version.id}:{item.item_key}"),
                    run_id=run.id, success_criterion_key=f"roadmap_item:{item.item_key}",
                    gate_type="child_goal_completed", required_evidence={
                        "roadmap_item_key": item.item_key, "roadmap_version_id": str(version.id),
                        "required_source_types": ["child_goal"], "min_count": 1,
                    },
                )
                lineage = OrchestrationRoadmapItem(
                    id=uuid.uuid5(uuid.NAMESPACE_URL, f"rally:roadmap-item:{goal.id}:{item.item_key}"),
                    goal_id=goal.id, first_version_id=version.id, item_key=item.item_key, unit_type="goal",
                    item_snapshot=item.model_dump(mode="json"), child_goal_id=child.id, task_id=None, gate_id=gate.id,
                )
                action = OrchestrationAction(
                    id=uuid.uuid5(uuid.NAMESPACE_URL, f"rally:authorize-child:{child_run.id}"), run_id=child_run.id,
                    idempotency_key=f"run:{child_run.id}:kind:authorize_execution", action_type="authorize_execution",
                    status="completed", request={"action_type": "authorize_execution", "actor": f"roadmap_parent:{goal.id}"},
                    target_type="run", target_id=child_run.id,
                )
                db.add_all([gate, lineage, action])
                await db.flush()
                await OrchestrationBudgetService().reserve_child(db, goal, lineage, item.allocation)
                await emit_event_once(
                    db, goal.project_id, "orchestration.roadmap_child_released",
                    {"parent_run_id": str(run.id), "roadmap_version_id": str(version.id),
                     "item_key": item.item_key, "child_goal_id": str(child.id), "child_run_id": str(child_run.id)},
                    source="orchestrator",
                    dedup_key=f"orchestration.roadmap_child_released:{run.id}:{version.id}:{item.item_key}",
                )
                await self.orchestration._mark_action_completed(db, release, target_type="goal", target_id=child.id)
        except Exception as exc:
            await self.orchestration._fail_reserved_action_for_current_flow(db, release, str(exc))
            raise
        return {"step": "release_item", "item_key": item.item_key, "unit_type": "goal"}

    async def advance(self, db, goal, run) -> dict:
        return await self.orchestration._act_until_wait(
            db, goal, run, self._advance_once,
            loop_steps=frozenset({"replan_decision", "terminal_item_decision", "integration_decision"}),
        )

    async def _advance_once(self, db, goal, run) -> dict:
        await self.renew_corrected_children(db, goal, run)
        version = await self.current_version(db, goal.id)
        if version is None:
            return await self.orchestration._advance_authorized_execution(db, goal, run)
        settled = await self.settle_finished_children(db, goal, run)
        summary = await self.remaining_or_block(db, goal, run)
        if summary is None:
            return {"step": "waiting", "reason": "needs_attention"}
        exhausted = next((key for key, value in summary["remaining"].items() if Decimal(value) <= 0), None)
        if exhausted is not None:
            self.orchestration._upsert_active_blocker(run, {
                "kind": "budget_integrity", "reason": f"Roadmap parent budget exhausted {exhausted}",
            })
            await db.flush()
            return {"step": "waiting", "reason": "needs_attention"}
        run.active_blockers = [blocker for blocker in run.active_blockers if not (
            isinstance(blocker, dict)
            and blocker.get("kind") == "budget_integrity"
            and str(blocker.get("reason", "")).startswith("Roadmap parent budget exhausted")
        )]
        run.active_blockers = [blocker for blocker in run.active_blockers if not (
            isinstance(blocker, dict)
            and blocker.get("kind") in ("expansion_limit", "child_contract")
            and blocker.get("roadmap_version_id") != str(version.id)
        )]
        if any(self.orchestration._has_active_blocker(run, kind) for kind in (
            "staging_boundary", "budget_measurement", "budget_integrity", "child_cancelled",
        )):
            return {"step": "waiting", "reason": "needs_attention"}
        if settled:
            return {"step": "settle_children", "count": settled}
        if await self.has_active_bound_verifier(db, run):
            return {"step": "waiting", "reason": "waiting_active_verification"}
        item = await self.first_releasable_item(db, goal, run, version)
        if self.orchestration._has_active_blocker(run, "staging_boundary"):
            return {"step": "waiting", "reason": "needs_attention"}
        if item is None:
            async def dispatch_or_attention(decision):
                try:
                    return await self.orchestration._dispatch_execution_decision(db, run, decision)
                except RoadmapReplanMeasurementAttention:
                    return None

            async def pending_replan_authority_wait(decision):
                parsed = self.orchestration._json_object_or_empty(decision.parsed_decision)
                if (
                    not self.orchestration._json_object_or_empty(run.plan_state).get("pending_replan")
                    or parsed.get("action_type") != "accept_plan"
                ):
                    return None
                artifact = await self.orchestration._plan_artifact_for_run(
                    db, run, self.orchestration._required_uuid(
                        parsed.get("plan_artifact_id"), "plan_artifact_id"
                    )
                )
                try:
                    await self._validate_replan(db, goal, run, version, self.parse_items(artifact, goal))
                except RoadmapReplanMeasurementAttention:
                    return None
                approval = await self.bind_pending_replan(
                    db, goal, run, artifact, await self.ensure_plan_approval(db, goal, run, artifact)
                )
                if isinstance(approval, OrchestrationAuthorityDecision):
                    if approval.status == "pending":
                        return {"step": "waiting", "reason": "waiting_plan_authority"}
                    return None
                if approval.status == "accepted":
                    return None
                action = await self.orchestration.execute_request_verification_action(
                    db, run.id,
                    {"action_type": "request_verification", "gate_id": str(approval.id), "work_function": "validation"},
                    await self.orchestration._verification_retry_key(
                        db, run.id, f"run:{run.id}:kind:request_verification:roadmap_plan:{approval.id}",
                    ), decision_id=decision.id,
                )
                return {"step": "waiting_plan_authority", "action_id": str(action.id)}

            async def dispatch_or_authority_wait(decision):
                authority_wait = await pending_replan_authority_wait(decision)
                if authority_wait is not None:
                    return authority_wait
                try:
                    return await dispatch_or_attention(decision)
                except HTTPException as exc:
                    if exc.status_code == 409 and exc.detail in {
                        "waiting_human_scope_budget_authority", "waiting_plan_authority",
                    }:
                        return {"step": "waiting", "reason": exc.detail}
                    raise

            if self.orchestration._json_object_or_empty(run.plan_state).get("pending_replan"):
                decision = await self.orchestration.request_llm_decision(db, run.id)
                action = await dispatch_or_authority_wait(decision)
                if isinstance(action, dict):
                    return action
                if action is None:
                    return {"step": "waiting", "reason": "needs_attention"}
                return {"step": "replan_decision", "action_id": str(action.id)}

            terminal_task_gate = await db.scalar(select(OrchestrationRoadmapItem.id).join(
                Task, OrchestrationRoadmapItem.task_id == Task.id,
            ).join(OrchestrationGate, OrchestrationRoadmapItem.gate_id == OrchestrationGate.id).where(
                OrchestrationRoadmapItem.goal_id == goal.id,
                OrchestrationRoadmapItem.unit_type == "task",
                Task.status.in_(("done", "failed", "cancelled")),
                OrchestrationGate.run_id == run.id,
                OrchestrationGate.status == "open",
            ).limit(1))
            if terminal_task_gate is not None:
                decision = await self.orchestration.request_llm_decision(db, run.id)
                action = await dispatch_or_authority_wait(decision)
                if isinstance(action, dict):
                    return action
                if action is None:
                    return {"step": "waiting", "reason": "needs_attention"}
                return {"step": "terminal_item_decision", "action_id": str(action.id)}
            integration_gate = await self.ensure_integration_gate(db, goal, run, version)
            if integration_gate is not None and integration_gate.status != "accepted":
                decision = await self.orchestration.request_llm_decision(db, run.id)
                action = await dispatch_or_authority_wait(decision)
                if isinstance(action, dict):
                    return action
                if action is None:
                    return {"step": "waiting", "reason": "needs_attention"}
                return {"step": "integration_decision", "action_id": str(action.id)}
            return {"step": "waiting", "reason": await self.wait_reason(db, goal, run, version)}
        if item.unit_type == "goal":
            remaining = summary["remaining"]
            if any(Decimal(str(amount)) > Decimal(remaining.get(key, "0")) for key, amount in item.allocation.items()):
                return {"step": "waiting", "reason": "budget_wait"}
            return await self.release_goal_item(db, goal, run, version, item)
        return await self.release_task_item(db, goal, run, version, item)

    async def project_child_completion(self, db, goal, run, row, child, child_run):
        if (
            child.parent_goal_id != goal.id
            or child.roadmap_version_id != row.first_version_id
            or child.roadmap_item_key != row.item_key
            or row.child_goal_id != child.id
            or child_run.goal_id != child.id
            or child_run.status != "completed"
            or child_run.phase != "completed"
        ):
            raise HTTPException(status_code=409, detail="Roadmap child completion lineage is invalid")
        gates, child_evidence = await self.orchestration._accepted_non_summary_manifest(db, child_run.id)
        gate = await db.get(OrchestrationGate, row.gate_id)
        if (
            gate is None or gate.run_id != run.id or gate.gate_type != "child_goal_completed"
            or gate.success_criterion_key != f"roadmap_item:{row.item_key}"
        ):
            raise HTTPException(status_code=409, detail="Roadmap child completion gate is invalid")
        evidence_id = uuid.uuid5(
            uuid.NAMESPACE_URL, f"rally:child-rollup:{run.id}:{row.item_key}:{child.id}"
        )
        evidence = await db.get(OrchestrationEvidence, evidence_id)
        metadata = {
            "child_run_id": str(child_run.id),
            "child_gate_ids": [str(item.id) for item in gates],
            "child_evidence_ids": [str(item.id) for item in child_evidence],
        }
        if evidence is None:
            evidence = OrchestrationEvidence(
                id=evidence_id, run_id=run.id, gate_id=gate.id, source_type="child_goal",
                source_id=child.id, verdict="accepted", evidence_metadata=metadata,
            )
            db.add(evidence)
            await db.flush()
        elif (
            evidence.run_id != run.id or evidence.gate_id != gate.id
            or evidence.source_type != "child_goal" or evidence.source_id != child.id
            or evidence.verdict != "accepted" or evidence.evidence_metadata != metadata
        ):
            raise HTTPException(status_code=409, detail="Roadmap child completion evidence is invalid")
        await self.orchestration._validate_gate(db, gate)
        return evidence

    async def renew_corrected_children(self, db, goal, run) -> int:
        """Re-release type-corrected children (pending_renewal successor) that still hold an active reservation."""
        children = list((await db.scalars(select(OrchestrationGoal).join(
            OrchestrationRoadmapItem, OrchestrationRoadmapItem.child_goal_id == OrchestrationGoal.id,
        ).join(
            OrchestrationBudgetReservation, OrchestrationBudgetReservation.roadmap_item_id == OrchestrationRoadmapItem.id,
        ).where(
            OrchestrationRoadmapItem.goal_id == goal.id,
            OrchestrationGoal.parent_goal_id == goal.id,
            OrchestrationGoal.status.not_in(("completed", "cancelled")),
            OrchestrationBudgetReservation.status == "active",
        ).order_by(OrchestrationRoadmapItem.item_key))).all())
        renewed = 0
        for child in children:
            child_run = await self.orchestration.get_run_for_goal(db, child.project_id, child.id)
            plan_state = self.orchestration._json_object_or_empty(child_run.plan_state) if child_run else {}
            delta = self.orchestration._json_object_or_empty(plan_state.get("child_delta_baseline"))
            if delta.get("status") != "pending_renewal":
                continue
            child_run.plan_state = {**plan_state, "child_delta_baseline": {
                **delta, "status": "accepted",
                "approval_reference": {"kind": "parent_renewal", "parent_run_id": str(run.id)},
            }}
            child_run.phase = "authorized"
            if child_run.goal_contract_fingerprint is None:
                child_run.goal_contract_fingerprint = compute_goal_contract_fingerprint(child)
            if await db.get(OrchestrationAction, action_id := uuid.uuid5(
                uuid.NAMESPACE_URL, f"rally:authorize-child:{child_run.id}",
            )) is None:
                db.add(OrchestrationAction(
                    id=action_id, run_id=child_run.id,
                    idempotency_key=f"run:{child_run.id}:kind:authorize_execution",
                    action_type="authorize_execution", status="completed",
                    request={"action_type": "authorize_execution", "actor": f"roadmap_parent:{goal.id}"},
                    target_type="run", target_id=child_run.id,
                ))
            renewed += 1
        if renewed:
            await db.flush()
        return renewed

    async def settle_finished_children(self, db, goal, run):
        rows = list((await db.scalars(select(OrchestrationRoadmapItem).join(
            OrchestrationBudgetReservation,
            OrchestrationBudgetReservation.roadmap_item_id == OrchestrationRoadmapItem.id,
        ).where(
            OrchestrationRoadmapItem.goal_id == goal.id,
            OrchestrationRoadmapItem.unit_type == "goal",
            OrchestrationRoadmapItem.completed_at.is_(None),
        ).order_by(OrchestrationRoadmapItem.item_key))).all())
        budget = OrchestrationBudgetService()
        settled = 0
        for row in rows:
            child = await db.get(OrchestrationGoal, row.child_goal_id)
            if child is None or child.status not in {"completed", "cancelled"}:
                continue
            reservation = await db.scalar(select(OrchestrationBudgetReservation).where(
                OrchestrationBudgetReservation.roadmap_item_id == row.id
            ))
            child_run = await self.orchestration.get_run_for_goal(db, child.project_id, child.id)
            if child_run is None:
                continue
            try:
                await budget.settle_child(
                    db, reservation, child_run, reason="cancelled" if child.status == "cancelled" else "completed"
                )
            except BudgetMeasurementError as exc:
                self.orchestration._upsert_active_blocker(run, {
                    "kind": "budget_measurement", "item_key": row.item_key,
                    "dimension": exc.dimension, "session_id": str(exc.session_id),
                    "scope": f"roadmap_item:{row.id}",
                })
                await db.flush()
                continue
            except HTTPException as exc:
                if exc.status_code != 409 or not str(exc.detail).startswith("Child exceeded reserved"):
                    raise
                self.orchestration._upsert_active_blocker(run, {
                    "kind": "budget_integrity", "item_key": row.item_key,
                    "reason": str(exc.detail),
                })
                await db.flush()
                continue
            run.active_blockers = [
                blocker for blocker in run.active_blockers
                if not (
                    isinstance(blocker, dict)
                    and blocker.get("kind") == "budget_measurement"
                    and blocker.get("scope") == f"roadmap_item:{row.id}"
                )
            ]
            if not reservation.measurement_complete:
                self.orchestration._upsert_active_blocker(run, {
                    "kind": "budget_integrity", "item_key": row.item_key,
                    "reason": "Cancelled child budget measurement is incomplete.",
                })
            exceeded_child = next(
                (key for key, amount in reservation.settled_spend.items()
                 if Decimal(amount) > Decimal(reservation.allocation.get(key, "0"))),
                None,
            )
            if exceeded_child is not None:
                self.orchestration._upsert_active_blocker(run, {
                    "kind": "budget_integrity", "item_key": row.item_key,
                    "reason": f"Child exceeded reserved {exceeded_child}",
                })
            if child.status == "completed" and child_run.status == child_run.phase == "completed":
                await self.project_child_completion(db, goal, run, row, child, child_run)
                row.completed_at = row.completed_at or _utcnow()
                settled += 1
            elif child.status == "cancelled":
                self.orchestration._upsert_active_blocker(run, {
                    "kind": "child_cancelled", "item_key": row.item_key,
                    "reason": "Roadmap child goal was cancelled.",
                })
                row.completed_at = row.completed_at or _utcnow()
                settled += 1
            await db.flush()
            try:
                summary = await budget.remaining(db, goal)
            except BudgetMeasurementError:
                # The terminal child is already durably projected.  Let advance()
                # record the parent-scoped measurement attention through its shared
                # remaining_or_block() path instead of rolling that projection back.
                continue
            exceeded = next((key for key, amount in summary["remaining"].items() if Decimal(amount) < 0), None)
            if exceeded is not None:
                self.orchestration._upsert_active_blocker(run, {
                    "kind": "budget_integrity", "item_key": row.item_key,
                    "reason": f"Parent budget exceeded {exceeded}",
                })
            await db.flush()
        return settled

    async def _settle_terminal_children(self, db, goal, run):
        """Return finished-child settlement in the legacy step-result format."""
        settled = await self.settle_finished_children(db, goal, run)
        return {"step": "settle_children", "count": settled} if settled else None
