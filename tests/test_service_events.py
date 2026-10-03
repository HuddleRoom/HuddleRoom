import pytest
import uuid
from sqlalchemy import select, delete
from huddleroom.models.event_log import EventLog
from huddleroom.schemas.task import TaskCreate
from huddleroom.schemas.project import ProjectCreate
from huddleroom.schemas.agent import AgentCreate


@pytest.mark.asyncio
async def test_task_create_emits_event(db_session, test_project, test_agent):
    from huddleroom.services.task_service import TaskService
    service = TaskService()
    await service.create(db_session, test_project.id, TaskCreate(
        title="Test task", description="desc"
    ))
    result = await db_session.execute(
        select(EventLog).where(
            EventLog.project_id == test_project.id,
            EventLog.event_type == "task.created",
        )
    )
    rows = list(result.scalars().all())
    assert len(rows) == 1
    assert "task_id" in rows[0].payload


@pytest.mark.asyncio
async def test_task_status_change_emits_event(db_session, test_project, test_agent):
    from huddleroom.services.task_service import TaskService
    service = TaskService()
    task = await service.create(db_session, test_project.id, TaskCreate(
        title="Test task", description="desc", assigned_to=test_agent.id
    ))
    # Clear events from create
    await db_session.execute(
        delete(EventLog).where(EventLog.project_id == test_project.id)
    )
    await service.transition_status(db_session, test_project.id, task.id, "ready")
    result = await db_session.execute(
        select(EventLog).where(
            EventLog.project_id == test_project.id,
            EventLog.event_type == "task.status_changed",
        )
    )
    rows = list(result.scalars().all())
    assert len(rows) >= 1
    status_events = [r for r in rows if r.payload.get("status") == "ready"]
    assert len(status_events) == 1


@pytest.mark.asyncio
async def test_project_create_emits_event(db_session, tmp_path):
    from huddleroom.services.project_service import ProjectService
    service = ProjectService()
    project = await service.create(
        db_session,
        ProjectCreate(name="New Project", workspace_path=str(tmp_path)),
    )
    result = await db_session.execute(
        select(EventLog).where(
            EventLog.project_id == project.id,
            EventLog.event_type == "project.created",
        )
    )
    rows = list(result.scalars().all())
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_agent_create_emits_event(db_session):
    from huddleroom.services.agent_service import AgentService
    service = AgentService()
    agent = await service.create(db_session, AgentCreate(
        name="Test Agent",
        role="tester",
        provider="openai",
        model="gpt-4o",
        adapter_type="api",
    ))
    _GLOBAL_PROJECT = __import__("uuid").UUID("00000000-0000-0000-0000-000000000000")
    result = await db_session.execute(
        select(EventLog).where(
            EventLog.project_id == _GLOBAL_PROJECT,
            EventLog.event_type == "agent.created",
        )
    )
    rows = list(result.scalars().all())
    assert len(rows) == 1
    assert "agent_id" in rows[0].payload


@pytest.mark.asyncio
async def test_task_assign_emits_event(db_session, test_project, test_agent):
    from huddleroom.services.task_service import TaskService
    service = TaskService()
    task = await service.create(db_session, test_project.id, TaskCreate(
        title="Test task", description="desc"
    ))
    # Clear events from create
    await db_session.execute(
        delete(EventLog).where(EventLog.project_id == test_project.id)
    )
    await service.assign(db_session, test_project.id, task.id, test_agent.id)
    result = await db_session.execute(
        select(EventLog).where(
            EventLog.project_id == test_project.id,
            EventLog.event_type == "task.assigned",
        )
    )
    rows = list(result.scalars().all())
    assert len(rows) == 1
    assert "agent_id" in rows[0].payload
