"""Roster projections plus the single global agent-roster write coordinator.

Lock order: AgentRosterWriteLock first, then the affected goals' existing
baseline-transition locks (tree roots, then goals, each sorted by UUID). No
caller may take the roster lock while it holds a goal lock; goal-locked entry
points reserve a PreparedRosterMutation, commit, release, and hand off.
"""
from __future__ import annotations

import asyncio
import copy
import dataclasses
import hashlib
import json
import weakref
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Generic, Literal, Sequence, TypeVar
from types import SimpleNamespace
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import event, false, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import SessionTransactionOrigin

from huddleroom.models.agent import Agent
from huddleroom.models.orchestration import ACTIVE_RUN_STATUSES, OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision, OrchestrationProcessRun
from huddleroom.models.task import Task
from huddleroom.services.orchestration_agent_definition_review import (
    AgentDefinitionReviewProcess,
    coverage_fingerprint_from_projection,
    coverage_targets_from_projection,
)
from huddleroom.services.orchestration_team_hierarchy import (
    TeamHierarchyProcess,
    infer_required_work_functions,
    hierarchy_analysis_input_from_projection,
    hierarchy_fingerprint_from_projection,
    resolved_proposal_exclusions_from_decisions,
    trivial_hierarchy_fingerprint_from_projection,
)


@dataclass(frozen=True)
class CanonicalAgentState:
    id: UUID; name: str; role: str; description: str | None; provider: str; model: str
    system_prompt: str | None; adapter_type: str; cli_runtime: str | None
    capabilities: tuple[str, ...]; config: dict[str, Any]; is_active: bool


@dataclass(frozen=True)
class CanonicalRosterState:
    agents: tuple[CanonicalAgentState, ...]


@dataclass(frozen=True)
class ValidatedRosterDelta:
    creates: tuple[CanonicalAgentState, ...] = ()
    updates: tuple[CanonicalAgentState, ...] = ()
    deletes: tuple[UUID, ...] = ()


@dataclass(frozen=True)
class GoalRosterFingerprintInput:
    goal_id: UUID; run_id: UUID; project_id: UUID; weight: str; objective: str
    success_criteria: list[dict[str, Any]]; constraints: dict[str, Any]; authority_model: str
    manager_agent_id: UUID | None; manager_user_id: UUID | None
    assignments: tuple[tuple[UUID, UUID, str, bool], ...]
    hierarchy_analysis: dict[str, Any] | None
    resolved_proposal_exclusions: tuple[dict[str, Any], ...]
    trivial_hierarchy_proposal: dict[str, Any] | None
    objective_with_clarifications: str | None = None
    covered_target_ids: frozenset[str] | None = None
    required_work_functions: tuple[str, ...] = ()


@dataclass(frozen=True)
class GoalRosterFingerprintChange:
    goal_id: UUID; before_coverage_fingerprint: str; after_coverage_fingerprint: str
    before_hierarchy_fingerprint: str; after_hierarchy_fingerprint: str


@dataclass(frozen=True)
class RosterMutationPlan:
    before_roster: CanonicalRosterState; after_roster: CanonicalRosterState
    goal_fingerprints: tuple[GoalRosterFingerprintChange, ...]
    affected_goal_ids: tuple[UUID, ...]; origin_goal_id: UUID | None


def _canonical_agent(agent: Agent) -> CanonicalAgentState:
    return CanonicalAgentState(agent.id, agent.name, agent.role, agent.description, agent.provider, agent.model,
        agent.system_prompt, agent.adapter_type, agent.cli_runtime, tuple(agent.capabilities or ()),
        copy.deepcopy(agent.config or {}), bool(agent.is_active))


def _projection_goal(goal: GoalRosterFingerprintInput) -> SimpleNamespace:
    return SimpleNamespace(
        id=goal.goal_id, weight=goal.weight, objective=goal.objective,
        success_criteria=copy.deepcopy(goal.success_criteria), constraints=copy.deepcopy(goal.constraints),
        authority_model=goal.authority_model, manager_agent_id=goal.manager_agent_id,
        manager_user_id=goal.manager_user_id, orchestrator_context={},
    )


def _coverage(goal: GoalRosterFingerprintInput, roster: CanonicalRosterState) -> str:
    projection_goal = _projection_goal(goal)
    agents = {agent.id: agent for agent in roster.agents}
    assignment_rows = [
        (task_id, agent_id, work_function, action_tagged)
        for task_id, agent_id, work_function, action_tagged in goal.assignments
        if agent_id in agents
    ]
    assignments = [
        (SimpleNamespace(id=task_id), agents[agent_id], work_function)
        for task_id, agent_id, work_function, _action_tagged in assignment_rows
    ]
    targets = coverage_targets_from_projection(
        projection_goal,
        assignments,
        active_agents=list(roster.agents) if goal.weight == "substantial" else [],
        manager=agents.get(goal.manager_agent_id),
    )
    if goal.covered_target_ids is not None:
        baseline = [
            assignment for assignment, item in zip(assignments, assignment_rows)
            if not item[3]
        ]
        baseline_targets = coverage_targets_from_projection(
            projection_goal, baseline,
            active_agents=list(roster.agents) if goal.weight == "substantial" else [],
            manager=agents.get(goal.manager_agent_id),
        )
        sanitized = [
            (task, agent, "" if action_tagged else work_function)
            for (task, agent, work_function), (_task_id, _agent_id, _work_function, action_tagged)
            in zip(assignments, assignment_rows)
        ]
        sanitized_targets = coverage_targets_from_projection(
            projection_goal, sanitized,
            active_agents=list(roster.agents) if goal.weight == "substantial" else [],
            manager=agents.get(goal.manager_agent_id),
        )
        targets = AgentDefinitionReviewProcess._merge_coverage_targets(
            baseline_targets,
            [target for target in sanitized_targets if str(target[0].id) in goal.covered_target_ids],
        )
    return coverage_fingerprint_from_projection(projection_goal, targets)


def _hierarchy(goal: GoalRosterFingerprintInput, roster: CanonicalRosterState) -> str:
    projection_goal = _projection_goal(goal)
    if goal.weight == "trivial":
        return trivial_hierarchy_fingerprint_from_projection(
            projection_goal, goal.trivial_hierarchy_proposal or {},
        )
    payload = hierarchy_analysis_input_from_projection(
        projection_goal, list(roster.agents), exclusions=goal.resolved_proposal_exclusions,
        required_work_functions=goal.required_work_functions,
    )
    return hierarchy_fingerprint_from_projection(payload, goal.hierarchy_analysis)


def derive_roster_mutation_plan(current_roster: CanonicalRosterState, delta: ValidatedRosterDelta,
                                goal_inputs: Sequence[GoalRosterFingerprintInput], *, origin_goal_id: UUID | None) -> RosterMutationPlan:
    create_ids, update_ids, delete_ids = {a.id for a in delta.creates}, {a.id for a in delta.updates}, set(delta.deletes)
    if not (create_ids or update_ids or delete_ids) or len(create_ids) != len(delta.creates) or len(update_ids) != len(delta.updates) or len(delete_ids) != len(delta.deletes) or create_ids & update_ids or create_ids & delete_ids or update_ids & delete_ids:
        raise ValueError("invalid_roster_delta")
    old = {agent.id: agent for agent in current_roster.agents}
    if create_ids & old.keys() or not update_ids <= old.keys() or not delete_ids <= old.keys():
        raise ValueError("invalid_roster_delta")
    if any(old[item.id] == item for item in delta.updates):
        raise ValueError("no_op_roster_delta")
    after = {**old, **{item.id: item for item in delta.creates}, **{item.id: item for item in delta.updates}}
    for item in delete_ids: del after[item]
    before_state = CanonicalRosterState(tuple(sorted(old.values(), key=lambda item: str(item.id))))
    after_state = CanonicalRosterState(tuple(sorted(
        (agent for agent in after.values() if agent.is_active), key=lambda item: str(item.id)
    )))
    changes = tuple(GoalRosterFingerprintChange(item.goal_id, _coverage(item, before_state), _coverage(item, after_state),
        _hierarchy(item, before_state), _hierarchy(item, after_state)) for item in sorted(goal_inputs, key=lambda item: str(item.goal_id)))
    affected = {item.goal_id for item in changes if item.before_coverage_fingerprint != item.after_coverage_fingerprint or item.before_hierarchy_fingerprint != item.after_hierarchy_fingerprint}
    if origin_goal_id: affected.add(origin_goal_id)
    return RosterMutationPlan(before_state, after_state, changes, tuple(sorted(affected, key=str)), origin_goal_id)


async def preview_roster_delta(db: AsyncSession, delta: ValidatedRosterDelta, *, origin_goal_id: UUID | None) -> RosterMutationPlan:
    with db.no_autoflush:
        roster = CanonicalRosterState(tuple(sorted((_canonical_agent(agent) for agent in (await db.execute(select(Agent).where(Agent.is_active.is_(True)))).scalars()), key=lambda item: str(item.id))))
        goals = list((await db.execute(select(OrchestrationGoal))).scalars())
        runs = list((await db.execute(select(OrchestrationRun).where(
            OrchestrationRun.status.in_(ACTIVE_RUN_STATUSES)
        ))).scalars())
        processes = list((await db.execute(select(OrchestrationProcessRun).where(OrchestrationProcessRun.superseded_by_id.is_(None)))).scalars())
        by_goal: dict[UUID, dict[str, OrchestrationProcessRun]] = {}
        for process in processes:
            by_goal.setdefault(process.goal_id, {})[process.process_type] = process
        runs_by_goal = {run.goal_id: run for run in runs}
        definition_process = AgentDefinitionReviewProcess()
        hierarchy_process = TeamHierarchyProcess()
        inputs = []
        for goal in goals:
            run = runs_by_goal.get(goal.id)
            if run is None:
                continue
            assignments = await definition_process._current_run_assignments(db, goal.project_id, run.id)
            hierarchy_current = by_goal.get(goal.id, {}).get("team_hierarchy")
            analysis = (hierarchy_current.outputs or {}).get("analysis") if hierarchy_current else None
            proposal = await hierarchy_process.build_proposal(db, goal, run) if goal.weight == "trivial" else None
            exclusions = resolved_proposal_exclusions_from_decisions(
                await hierarchy_process.decision_service.list_decisions(db, goal.id)
            )
            definition_current = by_goal.get(goal.id, {}).get("agent_definition_review")
            covered_target_ids = (
                frozenset(str(item) for item in (definition_current.outputs or {}).get("target_ids", []))
                if definition_current is not None and definition_current.status == "completed"
                and isinstance((definition_current.outputs or {}).get("target_ids"), list)
                else None
            )
            from huddleroom.services.orchestration_goal_definition import goal_objective_with_clarifications
            inputs.append(GoalRosterFingerprintInput(
                goal.id, run.id, goal.project_id, goal.weight, goal.objective,
                copy.deepcopy(goal.success_criteria or []), copy.deepcopy(goal.constraints or {}),
                goal.authority_model, goal.manager_agent_id, goal.manager_user_id,
                tuple((task.id, agent.id, work_function, bool(definition_process._assignment_action_id(task)))
                      for task, agent, work_function in assignments),
                copy.deepcopy(analysis) if isinstance(analysis, dict) else None,
                tuple(copy.deepcopy(exclusions)), copy.deepcopy(proposal),
                goal_objective_with_clarifications(goal), covered_target_ids,
                tuple(infer_required_work_functions(goal)),
            ))
    return derive_roster_mutation_plan(roster, delta, inputs, origin_goal_id=origin_goal_id)


# --------------------------------------------------------------------------- write coordination

T = TypeVar("T")
TransactionOwner = Literal["coordinator", "caller"]
_HANDOFF_PROCESS_STATUSES = frozenset({"running", "waiting_decision"})
# The CHECK constraint on orchestration_process_runs.status has no "failed" value
# and this feature ships without a migration, so an invalidated accepted snapshot
# is superseded by a non-terminal row whose gates are all false.
INVALIDATED_PROCESS_STATUS = "running"
_INVALIDATION_GATES = {
    "agent_definition_review": {"agent_definitions_reviewed": False},
    "team_hierarchy": {
        "team_structure_reviewed": False,
        "required_work_functions_mapped": False,
        "agent_definitions_reviewed": False,
        "missing_capabilities_handled": False,
        "independent_verification_possible_or_overridden": False,
    },
}
COLLATERAL_WARNING_TYPE = "roster_change_invalidated_configuration"
_CALLER_SAFE_REFRESH = frozenset({Agent, OrchestrationGoal, OrchestrationRun, Task})

# ponytail: process-local lock (supported deployment is single-process SQLite);
# a database advisory lock is the upgrade path for multi-process deployments.
_ROSTER_LOCKS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = weakref.WeakKeyDictionary()


class AgentRosterWriteLock:
    """Process-wide write lock for the global agent roster (one per event loop)."""

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        loop = asyncio.get_running_loop()
        lock = _ROSTER_LOCKS.get(loop)
        if lock is None:
            lock = _ROSTER_LOCKS[loop] = asyncio.Lock()
        async with lock:
            yield


class RosterStateChanged(HTTPException):
    """The sealed roster plan no longer matches fresh state; nothing was written."""

    error = "state_changed"

    def __init__(self, detail: str = "agent roster state changed") -> None:
        super().__init__(status_code=409, detail=detail)


@dataclass(frozen=True)
class PreparedRosterMutation:
    """Server-authored roster effect reserved under the origin goal lock (ORM-free)."""

    origin_goal_id: UUID
    origin_run_id: UUID
    process_id: UUID
    decision_ids: tuple[UUID, ...]
    delta: ValidatedRosterDelta
    expected_plan: RosterMutationPlan


@dataclass(frozen=True)
class LockedRosterMutation:
    delta: ValidatedRosterDelta
    plan: RosterMutationPlan
    transaction_owner: str


@dataclass(frozen=True)
class RosterMutationResult(Generic[T]):
    value: T
    affected_goal_ids: tuple[UUID, ...]
    goal_fingerprints: tuple[GoalRosterFingerprintChange, ...]
    error: str | None = None


class RosterHandoffRequired(Exception):
    """Raised under a goal lock: commit, release the goal lock, then apply roster-first."""

    def __init__(self, prepared: PreparedRosterMutation) -> None:
        super().__init__("roster mutation must be applied after releasing the goal lock")
        self.prepared = prepared


@asynccontextmanager
async def commit_before_roster_handoff(db: AsyncSession, owns_transaction: bool) -> AsyncIterator[None]:
    """Publish the reservation while the origin goal lock is still held."""
    try:
        yield
    except RosterHandoffRequired as handoff:
        await publish_roster_reservation(db, handoff.prepared, owns_transaction)
        raise


async def publish_roster_reservation(
    db: AsyncSession, prepared: PreparedRosterMutation, owns_transaction: bool,
) -> None:
    """Commit the reservation under the origin goal lock unless a reset already
    claimed the goal; then discard it and fail with the reset 409."""
    from huddleroom.services.orchestration_baseline_runtime import _current, _resetting

    operation = _current.get()
    if prepared.origin_goal_id in _resetting or (operation is not None and operation.interrupted.is_set()):
        if owns_transaction:
            await db.rollback()
        raise HTTPException(status_code=409, detail="Baseline operation discarded by reset")
    if owns_transaction:
        await db.commit()


def require_lock_introspection(db: AsyncSession) -> None:
    """Fail closed where goal-lock ownership cannot be detected (non-SQLite)."""
    # ponytail: Postgres goal locks are row locks we cannot introspect; supporting it
    # needs an explicit held-lock registry. Supported deployment is SQLite.
    if db.get_bind().dialect.name != "sqlite":
        raise RuntimeError("agent roster coordination requires SQLite goal-lock introspection")


def holds_goal_lock() -> bool:
    """True when the current task owns any SQLite baseline-transition goal lock."""
    # ponytail: SQLite-only detection; Postgres row locks are not introspected.
    from huddleroom.services.orchestration_service import _SQLITE_GOAL_LOCKS

    task = asyncio.current_task()
    return any(lock._owner_task is task for lock in list(_SQLITE_GOAL_LOCKS.values()))  # pylint: disable=protected-access


def transaction_shape(db: AsyncSession) -> str:
    transaction = db.sync_session.get_transaction()
    if transaction is None:
        return "none"
    if db.in_nested_transaction():
        return "nested"
    return "autobegin" if transaction.origin == SessionTransactionOrigin.AUTOBEGIN else "explicit"


async def fresh_preview(
    db: AsyncSession, delta: ValidatedRosterDelta, *, origin_goal_id: UUID | None, refresh_all: bool = True,
) -> RosterMutationPlan:
    """Run the shared preview with ORM reads repopulated from the database.

    ``refresh_all=False`` (a session whose transaction the caller still owns)
    refreshes only roster/goal/run/assignment rows, leaving the caller's own
    process and decision objects as the caller holds them.
    """

    def repopulate(state) -> None:
        if state.is_select and (refresh_all or any(
            mapper.class_ in _CALLER_SAFE_REFRESH for mapper in state.all_mappers
        )):
            state.update_execution_options(populate_existing=True)

    event.listen(db.sync_session, "do_orm_execute", repopulate)
    try:
        return await preview_roster_delta(db, delta, origin_goal_id=origin_goal_id)
    finally:
        event.remove(db.sync_session, "do_orm_execute", repopulate)


async def _preview_or_stale(db, delta, origin_goal_id, refresh_all) -> RosterMutationPlan:
    try:
        return await fresh_preview(db, delta, origin_goal_id=origin_goal_id, refresh_all=refresh_all)
    except ValueError as exc:  # delta no longer applies to the current roster
        raise RosterStateChanged(str(exc)) from exc


async def _claim_sqlite_writer(db: AsyncSession) -> None:
    """Start SQLite's write transaction so reloads see committed state and a
    released savepoint cannot publish (legacy pysqlite begins only before DML)."""
    if db.get_bind().dialect.name == "sqlite":
        await db.execute(
            update(Agent).where(false()).values(id=Agent.id).execution_options(synchronize_session=False)
        )


async def _goal_lock_order(db: AsyncSession, goal_ids: Sequence[UUID]) -> list[UUID]:
    """Tree roots first (ticks lock root then goal), then goals, each sorted by UUID."""
    from huddleroom.services.orchestration_capacity import root_goal_id

    roots = {await root_goal_id(db, goal_id) for goal_id in goal_ids}
    return sorted(roots, key=str) + sorted(set(goal_ids) - roots, key=str)


async def _require_origin_unchanged(db: AsyncSession, origin_goal_id, origin_run_id) -> None:
    """Under the goal locks: the origin goal is not being reset and its run is the reserved one."""
    if origin_goal_id is None:
        return
    from huddleroom.services.orchestration_baseline_runtime import require_baseline_available

    try:
        require_baseline_available(origin_goal_id)
    except HTTPException as exc:
        raise RosterStateChanged("origin goal is being reset") from exc
    if origin_run_id is not None and await db.scalar(select(OrchestrationRun.id).where(
        OrchestrationRun.goal_id == origin_goal_id, OrchestrationRun.status.in_(ACTIVE_RUN_STATUSES),
    ).limit(1)) != origin_run_id:
        raise RosterStateChanged("origin run changed")


def _root_open(db: AsyncSession, root) -> bool:
    return root is not None and db.sync_session.get_transaction() is root and root.is_active


def roster_fingerprint(roster: CanonicalRosterState) -> str:
    return hashlib.sha256(json.dumps(
        dataclasses.asdict(roster), sort_keys=True, separators=(",", ":"), default=str
    ).encode()).hexdigest()


class AgentRosterMutationCoordinator:
    """Applies a validated roster delta under roster-first, sorted goal locks."""

    async def apply(
        self, db: AsyncSession, delta: ValidatedRosterDelta, expected_plan: RosterMutationPlan | None,
        mutation: Callable[[AsyncSession, ValidatedRosterDelta, RosterMutationPlan], Awaitable[T]], *,
        origin_goal_id: UUID | None, provenance: dict, origin_run_id: UUID | None = None,
    ) -> RosterMutationResult[T]:
        """Coordinator-owned transaction: commit or roll back before any lock is released.

        ``expected_plan=None`` (public agent routes) seals the plan from the
        initial preview taken under the roster lock.
        """
        async with self._guard(db, delta, expected_plan, origin_goal_id, "coordinator", origin_run_id) as locked:
            value = await mutation(db, locked.delta, locked.plan)
            await self.invalidate_collateral(db, locked.plan, provenance)
        affected = set(locked.plan.affected_goal_ids)
        return RosterMutationResult(
            value, tuple(sorted(affected, key=str)),
            tuple(item for item in locked.plan.goal_fingerprints if item.goal_id in affected),
        )

    @asynccontextmanager
    async def mutation_guard(
        self, db: AsyncSession, delta: ValidatedRosterDelta, expected_plan: RosterMutationPlan, *,
        origin_goal_id: UUID | None, transaction_owner: TransactionOwner,
    ) -> AsyncIterator[LockedRosterMutation]:
        """``caller`` requires an explicit root transaction that the body commits or
        rolls back before leaving; a savepoint release or flush is not publication."""
        async with self._guard(db, delta, expected_plan, origin_goal_id, transaction_owner) as locked:
            yield locked

    @asynccontextmanager
    async def outer_transaction_guard(
        self, db: AsyncSession, delta: ValidatedRosterDelta, expected_plan: RosterMutationPlan | None, *,
        origin_goal_id: UUID | None, origin_run_id: UUID | None = None,
    ) -> AsyncIterator[LockedRosterMutation]:
        """Compatibility for code running inside an explicit transaction it does not
        own (test fixtures, outer ``db.begin()``). The locks cover revalidation and
        the write; publication is the outer commit, serialized by the SQLite writer
        lock the guard claims and keeps until that commit or rollback. Request
        sessions (``get_db``) are AUTOBEGIN and always use ``apply``; no
        production path reaches this guard (compatibility for never-committed
        outer transactions only)."""
        async with self._guard(db, delta, expected_plan, origin_goal_id, "outer", origin_run_id) as locked:
            yield locked

    @asynccontextmanager
    async def _guard(
        self, db, delta, expected_plan, origin_goal_id, owner, origin_run_id=None,
    ) -> AsyncIterator[LockedRosterMutation]:
        from huddleroom.services.orchestration_service import OrchestrationService

        require_lock_introspection(db)
        shape = transaction_shape(db)
        allowed = {"coordinator": {"none", "autobegin"}, "caller": {"explicit"}, "outer": {"explicit", "nested"}}
        if shape not in allowed[owner]:
            raise RuntimeError(f"roster mutation owner '{owner}' does not support a '{shape}' transaction")
        if holds_goal_lock():
            raise RuntimeError("the agent roster lock must be acquired before any goal lock")
        if shape == "autobegin":
            await db.commit()  # the coordinator owns this implicit transaction
        root = db.sync_session.get_transaction()
        async with AgentRosterWriteLock().acquire():
            if owner != "coordinator":
                await db.flush()
            initial = await _preview_or_stale(db, delta, origin_goal_id, owner == "coordinator")
            if expected_plan is not None and initial != expected_plan:
                raise RosterStateChanged()
            if owner == "coordinator":
                await db.commit()  # end the preview read before waiting for goal locks
            entered = False
            async with AsyncExitStack() as goal_locks:
                for goal_id in await _goal_lock_order(db, initial.affected_goal_ids):
                    await goal_locks.enter_async_context(OrchestrationService()._lock_single_goal(db, goal_id))  # pylint: disable=protected-access
                try:
                    await _claim_sqlite_writer(db)
                    plan = await _preview_or_stale(db, delta, origin_goal_id, owner == "coordinator")
                    if plan != initial:  # includes an expanded lock set
                        raise RosterStateChanged()
                    await _require_origin_unchanged(db, origin_goal_id, origin_run_id)
                    entered = True
                    yield LockedRosterMutation(delta, plan, owner)
                    if owner == "coordinator":
                        await db.commit()
                    elif owner == "caller" and _root_open(db, root):
                        raise RuntimeError("mutation_guard caller must commit or roll back its root transaction")
                except BaseException as exc:
                    if owner == "coordinator" and not entered and isinstance(exc, RosterStateChanged):
                        # Nothing but the zero-row writer claim ran: end the transaction
                        # without a rollback, which would expire the caller's objects.
                        await db.commit()
                    elif owner == "coordinator" or (owner == "caller" and entered and _root_open(db, root)):
                        await db.rollback()
                    raise

    async def invalidate_collateral(self, db: AsyncSession, plan: RosterMutationPlan, provenance: dict) -> None:
        """Supersede other goals' accepted snapshots whose existing fingerprints changed.

        Only an approved recovery (``provenance["proposal_action_id"]``) invalidates
        collateral goals; legacy edits keep the existing stale-input suggestions.
        """
        proposal_action_id = provenance.get("proposal_action_id")
        if not proposal_action_id:
            return
        from huddleroom.services.orchestration_process_service import OrchestrationProcessService
        from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

        process_service = OrchestrationProcessService()
        after_roster = roster_fingerprint(plan.after_roster)
        for change in plan.goal_fingerprints:
            if change.goal_id == plan.origin_goal_id or change.goal_id not in plan.affected_goal_ids or (
                change.before_coverage_fingerprint == change.after_coverage_fingerprint
                and change.before_hierarchy_fingerprint == change.after_hierarchy_fingerprint
            ):
                continue
            run = (await db.execute(select(OrchestrationRun).where(
                OrchestrationRun.goal_id == change.goal_id, OrchestrationRun.status.in_(ACTIVE_RUN_STATUSES)
            ).limit(1))).scalars().first()
            run_id = run.id if run is not None else None
            # An accepted child delta baseline is the parent's approved snapshot: tick
            # never advances its own review/hierarchy, so invalidating them would wedge
            # readiness checks that read the rows. Warn only.
            child_delta = (run.plan_state or {}).get("child_delta_baseline") if run is not None else None
            accepted_child_delta = isinstance(child_delta, dict) and child_delta.get("status") == "accepted"
            evidence = {
                **{key: str(value) for key, value in provenance.items()},
                "roster_fingerprint": after_roster,
                "fingerprints": {key: value for key, value in dataclasses.asdict(change).items() if key != "goal_id"},
            }
            for process_type, gates in ({} if accepted_child_delta else _INVALIDATION_GATES).items():
                current = await process_service.get_current(db, change.goal_id, process_type)
                if current is None or current.status != "completed":
                    continue  # only an accepted snapshot needs invalidating
                row = await process_service.start_process(
                    db, change.goal_id, process_type=process_type,
                    trigger_reason="approved_recovery: global roster change invalidated accepted snapshot",
                    run_id=run_id, input_snapshot={"recovery_invalidation": evidence},
                    process_version=current.process_version,
                )
                row.status = INVALIDATED_PROCESS_STATUS
                row.outputs = {"recovery_invalidation": evidence, "gates": dict(gates)}
                await db.flush()
            await OrchestrationWarningService().create_warning(
                db, change.goal_id, warning_type=COLLATERAL_WARNING_TYPE, severity="warning",
                message=(
                    "An approved fix for another goal changed the global agent roster. This goal's "
                    "accepted agent review and team hierarchy must be re-established before planning."
                ),
                run_id=run_id, related_action_id=UUID(str(proposal_action_id)),
                cross_goal_recovery_action=True,
            )
            if run is not None and run.status in ("running", "blocked"):
                # Spec: an affected active run receives a recovery trigger, atomically with the invalidation.
                from huddleroom.services.orchestration_service import OrchestrationService

                await OrchestrationService().create_recovery_trigger(
                    db, run, source_kind="planning_preflight", source_id=run.id,
                    blocker_code=COLLATERAL_WARNING_TYPE,
                    summary=("An approved fix for another goal changed the global agent roster; this goal's accepted "
                             "agent review and team hierarchy must be re-established."),
                    retry_target={"kind": "baseline_process", "process_type": "team_hierarchy"},
                    fingerprint_basis={"proposal_action_id": str(proposal_action_id), "roster": after_roster},
                )


async def revalidate_roster_handoff(db: AsyncSession, prepared: PreparedRosterMutation) -> OrchestrationRun:
    """Resume check under the origin goal lock: reset fence, active-run identity,
    authorization (decisions still answered) and the process continuation."""
    from huddleroom.services.orchestration_baseline_runtime import require_baseline_available
    from huddleroom.services.orchestration_service import LLM_DECISION_RUN_STATUSES

    require_baseline_available(prepared.origin_goal_id)
    run = (await db.execute(select(OrchestrationRun).where(
        OrchestrationRun.goal_id == prepared.origin_goal_id, OrchestrationRun.status.in_(ACTIVE_RUN_STATUSES),
    ).execution_options(populate_existing=True))).scalars().first()
    if run is None or run.id != prepared.origin_run_id or run.status not in LLM_DECISION_RUN_STATUSES:
        raise HTTPException(status_code=409, detail="active run changed during the roster update")
    # Column reads are always fresh and leave the caller's ORM objects untouched.
    process = (await db.execute(select(
        OrchestrationProcessRun.status, OrchestrationProcessRun.superseded_by_id
    ).where(OrchestrationProcessRun.id == prepared.process_id))).one_or_none()
    if process is None or process.superseded_by_id is not None or process.status not in _HANDOFF_PROCESS_STATUSES:
        raise HTTPException(status_code=409, detail="process changed during the roster update")
    statuses = list((await db.execute(select(OrchestrationAuthorityDecision.status).where(
        OrchestrationAuthorityDecision.id.in_(prepared.decision_ids)
    ))).scalars())
    if len(statuses) != len(set(prepared.decision_ids)) or any(status != "answered" for status in statuses):
        raise HTTPException(status_code=409, detail="approval changed during the roster update")
    return run
