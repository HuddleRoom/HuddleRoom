import uuid

import pytest

from huddleroom.models.agent import Agent
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import OrchestrationProcessRun
from huddleroom.schemas.orchestration import (
    OrchestrationPlanItem,
    OrchestrationRoadmapPlanItem,
    PlanReviewIdentity,
    PlanReviewRecord,
)
from huddleroom.services.orchestration_plan_ownership import PlanOwnershipService
from huddleroom.services.orchestration_process_service import OrchestrationProcessService


def _agent(name: str, *, active: bool = True) -> Agent:
    return Agent(name=f"{name}-{uuid.uuid4()}", role="developer", provider="openai", model="x",
                 adapter_type="api", capabilities=["implementation"], config={}, is_active=active)


def _accepted_outputs(agent_id, *, coverage=None):
    return {
        "role_to_agent": {"implementation": str(agent_id)}, "approval": "approve",
        "approval_decision_id": str(uuid.uuid4()), "gates": {"team_structure_reviewed": True},
        "accepted_responsibility_gap_ids": [],
        "responsibility_coverage": coverage or [{
            "responsibility_id": "implementation", "work_function": "implementation", "agent_ref": str(agent_id),
            "status": "covered", "evidence_ref": "agent_review:accepted",
        }],
    }


async def _accepted_review(db_session, goal, agent_id):
    db_session.add(OrchestrationProcessRun(
        goal_id=goal.id, process_type="agent_definition_review", status="completed", trigger_reason="test",
        outputs={"eligibility": {str(agent_id): ["implementation"]}},
    ))


@pytest.mark.asyncio
async def test_g3_inventory_and_assignment_use_accepted_evidence(db_session, test_project):
    agent = _agent("accepted")
    db_session.add(agent)
    await db_session.flush()
    goal = OrchestrationGoal(project_id=test_project.id, objective="o", original_request="o")
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    db_session.add(run)
    await _accepted_review(db_session, goal, agent.id)
    db_session.add(OrchestrationProcessRun(
        goal_id=goal.id, process_type="team_hierarchy", status="completed", trigger_reason="test",
        outputs=_accepted_outputs(agent.id),
    ))
    await db_session.flush()
    item = OrchestrationPlanItem(id="work", work_function="implementation", scope="s", deliverable="d")
    result = await PlanOwnershipService().validate_plan_owners(db_session, goal, run, [item], bind=False, review=None)
    assert result == [{"item_id": "work", "status": "eligible", "agent_id": str(agent.id), "work_function": "implementation"}]
    with pytest.raises(ValueError, match="current PlanReviewRecord"):
        await PlanOwnershipService().validate_plan_owners(db_session, goal, run, [item], bind=True, review=None)


@pytest.mark.asyncio
async def test_g3_domain_owner_is_not_execution_membership(db_session, test_project):
    agent = _agent("inactive", active=False)
    db_session.add(agent)
    await db_session.flush()
    goal = OrchestrationGoal(project_id=test_project.id, objective="o", original_request="o")
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    db_session.add(run)
    await _accepted_review(db_session, goal, agent.id)
    db_session.add(OrchestrationProcessRun(
        goal_id=goal.id, process_type="team_hierarchy", status="completed", trigger_reason="test",
        outputs=_accepted_outputs(agent.id),
    ))
    await db_session.flush()
    item = OrchestrationPlanItem(id="work", work_function="implementation", scope="s", deliverable="d")
    result = await PlanOwnershipService().validate_plan_owners(db_session, goal, run, [item], bind=False, review=None)
    assert result[0]["status"] == "conflict"


@pytest.mark.asyncio
async def test_g3_coverage_gap_and_domain_owner_are_not_execution_assignments(db_session, test_project):
    agent = _agent("worker")
    db_session.add(agent)
    await db_session.flush()
    goal = OrchestrationGoal(project_id=test_project.id, objective="o", original_request="o")
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    db_session.add(run)
    await _accepted_review(db_session, goal, agent.id)
    db_session.add(OrchestrationProcessRun(
        goal_id=goal.id, process_type="team_hierarchy", status="completed", trigger_reason="test",
        outputs=_accepted_outputs(agent.id, coverage=[
            {"responsibility_id": "worker", "work_function": "implementation", "agent_ref": str(agent.id),
             "status": "covered", "evidence_ref": "agent_review:worker"},
            {"responsibility_id": "zero", "work_function": "implementation", "agent_ref": None,
             "status": "gap", "evidence_ref": "gap:zero"},
        ]) | {"approval": "approve_with_documented_gaps", "accepted_responsibility_gap_ids": ["zero"]},
    ))
    await db_session.flush()
    good = OrchestrationPlanItem(id="good", work_function="implementation", scope="s", deliverable="d",
                                 required_evidence={"responsibility_id": "worker"})
    gap = OrchestrationPlanItem(id="gap", work_function="implementation", scope="s", deliverable="d",
                                required_evidence={"responsibility_id": "zero"})
    result = await PlanOwnershipService().validate_plan_owners(db_session, goal, run, [good, gap], bind=False, review=None)
    assert result[0]["status"] == "eligible"
    assert result[1]["status"] == "conflict"
    assert "gap" in result[1]["reason"]


@pytest.mark.asyncio
async def test_g3_inherited_team_uses_ancestor_g2_records(db_session, test_project):
    agent = _agent("worker")
    db_session.add(agent)
    await db_session.flush()
    parent = OrchestrationGoal(project_id=test_project.id, objective="parent", original_request="parent")
    db_session.add(parent)
    await db_session.flush()
    parent_run = OrchestrationRun(goal_id=parent.id)
    db_session.add(parent_run)
    await _accepted_review(db_session, parent, agent.id)
    outputs = _accepted_outputs(agent.id)
    db_session.add(OrchestrationProcessRun(goal_id=parent.id, process_type="team_hierarchy", status="completed",
                                           trigger_reason="test", outputs=outputs))
    child = OrchestrationGoal(project_id=test_project.id, objective="child", original_request="child",
                              parent_goal_id=parent.id, continuous_origin_key="test-child",
                              parent_contract_snapshot={}, goal_delta={}, orchestrator_context={"team": outputs})
    db_session.add(child)
    await db_session.flush()
    child_run = OrchestrationRun(goal_id=child.id)
    db_session.add(child_run)
    await db_session.flush()
    item = OrchestrationPlanItem(id="work", work_function="implementation", scope="s", deliverable="d")
    result = await PlanOwnershipService().validate_plan_owners(db_session, child, child_run, [item], bind=False, review=None)
    assert result[0]["status"] == "eligible"


@pytest.mark.asyncio
async def test_g3_superseded_completed_hierarchy_cannot_authorize(db_session, test_project):
    agent = _agent("worker")
    db_session.add(agent)
    await db_session.flush()
    goal = OrchestrationGoal(project_id=test_project.id, objective="o", original_request="o")
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    db_session.add(run)
    await _accepted_review(db_session, goal, agent.id)
    old = OrchestrationProcessRun(goal_id=goal.id, process_type="team_hierarchy", status="completed",
                                  trigger_reason="old", outputs=_accepted_outputs(agent.id))
    db_session.add(old)
    await db_session.flush()
    current = await OrchestrationProcessService().start_process(
        db_session, goal.id, process_type="team_hierarchy", trigger_reason="replacement",
    )
    assert old.superseded_by_id == current.id
    item = OrchestrationPlanItem(id="work", work_function="implementation", scope="s", deliverable="d")
    result = await PlanOwnershipService().validate_plan_owners(db_session, goal, run, [item], bind=False, review=None)
    assert result[0]["status"] == "conflict"
    assert "hierarchy" in result[0]["reason"]


@pytest.mark.asyncio
async def test_g3_human_prerequisite_stays_blocked(db_session, test_project):
    goal = OrchestrationGoal(project_id=test_project.id, objective="o", original_request="o")
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    db_session.add(run)
    await db_session.flush()
    item = OrchestrationPlanItem.model_validate({
        "id": "consent", "owner_type": "human", "scope": "s", "deliverable": "d",
    })
    result = await PlanOwnershipService().validate_plan_owners(db_session, goal, run, [item], bind=False, review=None)
    assert result == [{"item_id": "consent", "status": "blocked", "reason": "human executable work requires SPR254 workflow"}]


@pytest.mark.asyncio
async def test_g3_goal_item_binds_after_accepted_g2_inventory(db_session, test_project):
    agent = _agent("worker")
    db_session.add(agent)
    await db_session.flush()
    goal = OrchestrationGoal(project_id=test_project.id, objective="o", original_request="o")
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    db_session.add(run)
    await _accepted_review(db_session, goal, agent.id)
    db_session.add(OrchestrationProcessRun(
        goal_id=goal.id, process_type="team_hierarchy", status="completed", trigger_reason="test",
        outputs=_accepted_outputs(agent.id),
    ))
    await db_session.flush()
    item = OrchestrationRoadmapPlanItem(
        item_key="child", unit_type="goal", title="Child", objective="o", success_criteria=[{"key": "done"}],
        allocation={"tokens": 1},
    )
    review = PlanReviewRecord.model_construct(
        identity=PlanReviewIdentity.model_construct(project_id=test_project.id, goal_id=goal.id, run_id=run.id),
    )
    result = await PlanOwnershipService().validate_plan_owners(db_session, goal, run, [item], bind=True, review=review)
    assert result == [{"item_id": "child", "status": "eligible", "unit_type": "goal"}]
    assert run.plan_state["accepted_plan_ownership"]["items"] == {"child": {"unit_type": "goal"}}


@pytest.mark.asyncio
async def test_g3_goal_item_requires_complete_accepted_g2_inventory(db_session, test_project):
    goal = OrchestrationGoal(project_id=test_project.id, objective="o", original_request="o")
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    db_session.add(run)
    await db_session.flush()
    item = OrchestrationRoadmapPlanItem(
        item_key="child", unit_type="goal", title="Child", objective="o", success_criteria=[{"key": "done"}],
        allocation={"tokens": 1},
    )
    result = await PlanOwnershipService().validate_plan_owners(db_session, goal, run, [item], bind=False, review=None)
    assert result == [{"item_id": "child", "status": "conflict", "reason": "current accepted team hierarchy is missing"}]


@pytest.mark.asyncio
async def test_g3_goal_item_rejects_incomplete_g2_inventory(db_session, test_project):
    agent = _agent("worker")
    db_session.add(agent)
    await db_session.flush()
    goal = OrchestrationGoal(project_id=test_project.id, objective="o", original_request="o")
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    db_session.add(run)
    db_session.add(OrchestrationProcessRun(
        goal_id=goal.id, process_type="team_hierarchy", status="completed", trigger_reason="test",
        outputs=_accepted_outputs(agent.id),
    ))
    await db_session.flush()
    item = OrchestrationRoadmapPlanItem(
        item_key="child", unit_type="goal", title="Child", objective="o", success_criteria=[{"key": "done"}],
        allocation={"tokens": 1},
    )
    result = await PlanOwnershipService().validate_plan_owners(db_session, goal, run, [item], bind=False, review=None)
    assert result == [{"item_id": "child", "status": "conflict", "reason": "current accepted agent review is missing"}]


async def _two_row_goal(db_session, test_project, *, second_status="covered"):
    agent = _agent("worker")
    db_session.add(agent)
    await db_session.flush()
    goal = OrchestrationGoal(project_id=test_project.id, objective="o", original_request="o")
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    db_session.add(run)
    await _accepted_review(db_session, goal, agent.id)
    gap = second_status == "gap"
    db_session.add(OrchestrationProcessRun(
        goal_id=goal.id, process_type="team_hierarchy", status="completed", trigger_reason="test",
        outputs=_accepted_outputs(agent.id, coverage=[
            {"responsibility_id": "a", "work_function": "implementation", "agent_ref": str(agent.id),
             "status": "covered", "evidence_ref": "agent_review:a"},
            {"responsibility_id": "b", "work_function": "implementation", "agent_ref": None if gap else str(agent.id),
             "status": second_status, "evidence_ref": "gap:b" if gap else "agent_review:b"},
        ]) | ({"approval": "approve_with_documented_gaps", "accepted_responsibility_gap_ids": ["b"]} if gap else {}),
    ))
    await db_session.flush()
    return agent, goal, run


@pytest.mark.asyncio
async def test_roadmap_item_with_multi_row_function_is_eligible(db_session, test_project):
    agent, goal, run = await _two_row_goal(db_session, test_project)
    item = OrchestrationRoadmapPlanItem(item_key="r", unit_type="task", title="t", work_function="implementation",
                                        scope="s", deliverable="d")
    result = await PlanOwnershipService().validate_plan_owners(db_session, goal, run, [item], bind=False, review=None)
    assert result[0]["status"] == "eligible"
    assert result[0]["agent_id"] == str(agent.id)


@pytest.mark.asyncio
async def test_outcome_item_with_empty_evidence_and_multi_row_stays_ambiguous(db_session, test_project):
    _, goal, run = await _two_row_goal(db_session, test_project)
    item = OrchestrationPlanItem(id="o", work_function="implementation", scope="s", deliverable="d", required_evidence={})
    result = await PlanOwnershipService().validate_plan_owners(db_session, goal, run, [item], bind=False, review=None)
    assert result[0]["status"] == "conflict"
    assert "ambiguous" in result[0]["reason"]


@pytest.mark.asyncio
async def test_roadmap_item_with_gap_row_still_conflicts(db_session, test_project):
    _, goal, run = await _two_row_goal(db_session, test_project, second_status="gap")
    item = OrchestrationRoadmapPlanItem(item_key="r", unit_type="task", title="t", work_function="implementation",
                                        scope="s", deliverable="d")
    result = await PlanOwnershipService().validate_plan_owners(db_session, goal, run, [item], bind=False, review=None)
    assert result[0]["status"] == "conflict"
    assert "gap" in result[0]["reason"] or "ambiguous" in result[0]["reason"]
