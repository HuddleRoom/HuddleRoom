"""Pure roster projections used by recovery preparation and later application."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Sequence
from types import SimpleNamespace
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.agent import Agent
from huddleroom.models.orchestration import ACTIVE_RUN_STATUSES, OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import OrchestrationProcessRun
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
