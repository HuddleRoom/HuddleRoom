from pathlib import Path

import pytest
from pydantic import ValidationError
from fastapi import HTTPException

from huddleroom.models.meeting import Meeting
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.models.graph import Graph, GraphRun
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.project import ProjectCreate, ProjectUpdate
from huddleroom.services.project_service import ProjectService


def test_project_create_requires_workspace_path():
    with pytest.raises(ValidationError):
        ProjectCreate(name="Project")


def test_project_update_keeps_nullable_legacy_workspace_support():
    assert ProjectUpdate(workspace_path=None).workspace_path is None


def test_workspace_paths_are_canonicalized_for_create_and_update(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    noncanonical_path = workspace / ".." / workspace.name

    assert ProjectCreate(name="Project", workspace_path=str(noncanonical_path)).workspace_path == str(workspace.resolve())
    assert ProjectUpdate(workspace_path=str(noncanonical_path)).workspace_path == str(workspace.resolve())


@pytest.mark.parametrize("workspace_path", ["relative", "/does/not/exist"])
def test_workspace_paths_must_be_usable_directories(workspace_path: str):
    with pytest.raises(ValidationError):
        ProjectCreate(name="Project", workspace_path=workspace_path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("work_type", "status"),
    [
        ("session", "pending"),
        ("session", "running"),
        ("meeting", "preparing"),
        ("meeting", "active"),
        ("meeting", "concluding"),
        ("graph", "active"),
        ("graph", "paused"),
        ("run", "running"),
        ("run", "blocked"),
        ("run", "paused"),
    ],
)
async def test_workspace_change_is_rejected_while_project_work_is_active(
    db_session, legacy_project, test_agent, tmp_path: Path, work_type: str, status: str
):
    """Removing an active-work status check must permit an unsafe workspace switch."""
    if work_type == "session":
        db_session.add(Session(
            agent_id=test_agent.id, project_id=legacy_project.id, adapter_type="api", status=status,
        ))
    elif work_type == "meeting":
        db_session.add(Meeting(
            project_id=legacy_project.id, title="Active work", meeting_type="decision", status=status,
        ))
    elif work_type == "graph":
        graph = Graph(project_id=legacy_project.id, name="Graph", definition={}, triggers=[])
        db_session.add(graph)
        await db_session.flush()
        db_session.add(GraphRun(
            graph_id=graph.id, project_id=legacy_project.id, current_node="start", status=status,
        ))
    else:
        goal = OrchestrationGoal(project_id=legacy_project.id, objective="Finish safely")
        db_session.add(goal)
        await db_session.flush()
        db_session.add(OrchestrationRun(goal_id=goal.id, status=status))
    await db_session.flush()

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(HTTPException) as exc_info:
        await ProjectService().update(
            db_session, legacy_project.id, ProjectUpdate(workspace_path=str(workspace))
        )

    assert exc_info.value.status_code == 409
    assert legacy_project.workspace_path is None


@pytest.mark.asyncio
async def test_workspace_change_allows_tasks_and_scheduled_meetings_and_updates_legacy_project(
    db_session, legacy_project, tmp_path: Path
):
    """Treating queued planning rows as active work would block a safe legacy-project repair."""
    db_session.add_all([
        Task(project_id=legacy_project.id, title="Planned task"),
        Meeting(project_id=legacy_project.id, title="Scheduled", meeting_type="decision", status="scheduled"),
    ])
    await db_session.flush()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    updated = await ProjectService().update(
        db_session,
        legacy_project.id,
        ProjectUpdate(
            name="Renamed", description="Updated", config={"mode": "safe"}, workspace_path=str(workspace),
        ),
    )

    assert updated.workspace_path == str(workspace.resolve())
    assert updated.name == "Renamed"
    assert updated.description == "Updated"
    assert updated.config == {"mode": "safe"}


@pytest.mark.asyncio
async def test_unchanged_workspace_path_does_not_block_ordinary_updates(
    db_session, test_project, test_agent, tmp_path: Path
):
    """A no-op workspace value must not turn an unrelated edit into a conflict."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    test_project.workspace_path = str(workspace.resolve())
    db_session.add(Session(
        agent_id=test_agent.id, project_id=test_project.id, adapter_type="api", status="running",
    ))
    await db_session.flush()

    updated = await ProjectService().update(
        db_session, test_project.id, ProjectUpdate(name="Renamed", workspace_path=str(workspace))
    )

    assert updated.name == "Renamed"
