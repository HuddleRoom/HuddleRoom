import pytest

from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import OrchestrationProcessRun
from huddleroom.services.orchestration_plan_ownership import PlanOwnershipService
from tests.test_orchestration_plan_ownership import _accepted_outputs, _accepted_review, _agent


@pytest.mark.asyncio
async def test_team_prompt_renders_accepted_team_and_documented_gaps(db_session, test_project):
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
            {"responsibility_id": "implementation", "work_function": "implementation", "agent_ref": str(agent.id),
             "status": "covered", "evidence_ref": "agent_review:accepted"},
            {"responsibility_id": "zero", "work_function": "implementation", "agent_ref": None,
             "status": "gap", "evidence_ref": "gap:zero"},
        ]) | {"approval": "approve_with_documented_gaps", "accepted_responsibility_gap_ids": ["zero"]},
    ))
    await db_session.flush()

    text = await PlanOwnershipService().team_prompt(db_session, goal, run)

    assert f"- implementation: {agent.name} ({agent.id})" in text
    assert "Allowed work_function values: implementation" in text
    assert "Documented gaps (accepted with no owner; do not create plan items for them and never invent an agent; mention them in the summary): zero" in text
    assert "Set owner_type explicitly to agent for every task item." in text
    assert "Ignore work_function values shown in examples; use only the allowed values above." in text


@pytest.mark.asyncio
async def test_team_prompt_is_empty_without_accepted_hierarchy(db_session, test_project):
    goal = OrchestrationGoal(project_id=test_project.id, objective="o", original_request="o")
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    db_session.add(run)
    await db_session.flush()

    assert await PlanOwnershipService().team_prompt(db_session, goal, run) == ""


@pytest.mark.asyncio
async def test_planning_inventory_treats_dangling_parent_goal_as_no_team(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from uuid import uuid4

    from huddleroom.services.orchestration_plan_ownership import PlanOwnershipService
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    monkeypatch.setattr(OrchestrationProcessService, "get_current", AsyncMock(return_value=None))
    goal = SimpleNamespace(id=uuid4(), parent_goal_id=uuid4(), orchestrator_context={})
    db = SimpleNamespace(get=AsyncMock(return_value=None))

    inventory = await PlanOwnershipService().planning_inventory(db, goal, None)

    assert inventory["functions"] == {}
    assert inventory["conflicts"]
