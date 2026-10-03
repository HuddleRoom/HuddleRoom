import pytest

from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.models.project import Project
from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService
from huddleroom.services.orchestration_process_service import OrchestrationProcessService


@pytest.mark.asyncio
async def test_memory_overview_includes_bounded_orchestrator_preface(
    client, db_session, test_project
):
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Ship the release",
        success_criteria=[{"description": "Release is verified"}],
        constraints={},
        budget={},
        weight="substantial",
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running")
    db_session.add(run)
    await db_session.flush()
    await OrchestrationProcessService().start_process(
        db_session,
        goal.id,
        process_type="team_hierarchy",
        trigger_reason="dashboard test",
        run_id=run.id,
    )
    await OrchestrationMemoryService().upsert_section(
        db_session,
        test_project.id,
        goal.id,
        section_key="introduction",
        title="Introduction",
        body="Operator context",
        summary="Operator context",
        always_load=True,
        created_by="orchestrator:test",
    )

    response = await client.get(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/memory"
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["preface"]["objective"] == goal.objective
    assert payload["preface"]["goal_weight"] == "substantial"
    assert payload["preface"]["run_status"] == run.status
    assert payload["preface"]["current_process"] == {
        "process_type": "team_hierarchy",
        "status": "running",
    }
    assert payload["preface"]["introduction"] == "Operator context"
    assert payload["preface"]["toc"] == [
        {"section_key": "introduction", "title": "Introduction"}
    ]


@pytest.mark.asyncio
async def test_memory_overview_returns_404_for_goal_from_another_project(
    client, db_session, test_project
):
    other_project = Project(name="Other Project", description="", config={})
    db_session.add(other_project)
    await db_session.flush()
    goal = OrchestrationGoal(
        project_id=other_project.id,
        objective="Other project goal",
        success_criteria=[],
        constraints={},
        budget={},
    )
    db_session.add(goal)
    await db_session.flush()

    response = await client.get(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/memory"
    )

    assert response.status_code == 404, response.text
