"""tick() starts never-started backlog tasks the orchestrator delegated."""
import pytest
from sqlalchemy import select

from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.services.orchestration_service import OrchestrationService
from tests.test_orchestration_runtime_e2e import _agent

pytestmark = pytest.mark.asyncio


async def _setup(db, project, phase, **task_kwargs):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db, project_id=project.id,
        data=OrchestrationGoalCreate(objective="autostart", success_criteria=[]), created_by_user_id=None,
    )
    agent = _agent("autostart", ["implementation"])
    db.add(agent)
    await db.flush()
    goal.status, run.status, run.phase = "active", "running", phase
    task = Task(project_id=project.id, title="delegated", status="backlog", assigned_to=agent.id,
                metadata_={"orchestration": {"run_id": str(run.id)}}, **task_kwargs)
    db.add(task)
    await db.flush()
    return service, run, task


async def _sessions(db, task):
    return list(await db.scalars(select(Session).where(Session.task_id == task.id)))


@pytest.mark.parametrize("phase", ["baseline", "authorized"])
async def test_tick_starts_delegated_backlog_task_once(db_session, test_project, phase):
    service, run, task = await _setup(db_session, test_project, phase)
    await service.tick(db_session, run.id)
    await db_session.refresh(task)
    assert task.status == "in_progress"
    sessions = await _sessions(db_session, task)
    assert [s.origin for s in sessions] == ["auto"]
    await service.tick(db_session, run.id)
    assert len(await _sessions(db_session, task)) == 1


async def test_tick_leaves_task_in_backlog_when_not_released(db_session, test_project):
    service, run, task = await _setup(db_session, test_project, "ready")
    await service.tick(db_session, run.id)
    await db_session.refresh(task)
    assert task.status == "backlog" and await _sessions(db_session, task) == []


async def test_tick_leaves_task_with_open_dependency_in_backlog(db_session, test_project):
    service, run, task = await _setup(db_session, test_project, "authorized")
    dep = Task(project_id=test_project.id, title="dep", status="in_progress")
    db_session.add(dep)
    await db_session.flush()
    task.depends_on = [str(dep.id)]
    await db_session.flush()
    await service.tick(db_session, run.id)
    await db_session.refresh(task)
    assert task.status == "backlog" and await _sessions(db_session, task) == []
