import pytest
import uuid
from sqlalchemy import select
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.task import TaskCreate
from huddleroom.services.event_bus import BusEvent
from huddleroom.workers.consumers.rule_engine import evaluate_event_triggers
from datetime import datetime, timezone


@pytest.mark.asyncio
async def test_event_trigger_creates_session(db_session, test_project, test_agent):
    """Task with event trigger gets a session when matching event emitted."""
    from huddleroom.services.task_service import TaskService
    task_service = TaskService()

    task = await task_service.create(db_session, test_project.id, TaskCreate(
        title="Event-triggered task",
        assigned_to=test_agent.id,
        trigger={"type": "event", "event_type": "artifact.pr_opened"},
    ))
    task.status = "ready"
    await db_session.flush()

    event = BusEvent(
        id=uuid.uuid4(),
        project_id=test_project.id,
        event_type="artifact.pr_opened",
        payload={"pr_url": "https://github.com/foo/bar/pull/1"},
        source="system",
        emitted_at=datetime.now(timezone.utc),
    )
    await evaluate_event_triggers(db_session, event)

    result = await db_session.execute(
        select(Session).where(
            Session.task_id == task.id,
            Session.origin == "trigger",
        )
    )
    sessions = list(result.scalars().all())
    assert len(sessions) == 1


@pytest.mark.asyncio
async def test_task_status_trigger_creates_session(db_session, test_project, test_agent):
    """Task with task_status trigger fires when watched task reaches target status."""
    from huddleroom.services.task_service import TaskService
    task_service = TaskService()

    watched_task = await task_service.create(db_session, test_project.id, TaskCreate(
        title="Watched task",
    ))

    triggered_task = await task_service.create(db_session, test_project.id, TaskCreate(
        title="Triggered task",
        assigned_to=test_agent.id,
        trigger={
            "type": "task_status",
            "task_id": str(watched_task.id),
            "target_status": "done",
        },
    ))
    triggered_task.status = "ready"
    await db_session.flush()

    event = BusEvent(
        id=uuid.uuid4(),
        project_id=test_project.id,
        event_type="task.status_changed",
        payload={"task_id": str(watched_task.id), "status": "done"},
        source="system",
        emitted_at=datetime.now(timezone.utc),
    )
    await evaluate_event_triggers(db_session, event)

    result = await db_session.execute(
        select(Session).where(
            Session.task_id == triggered_task.id,
            Session.origin == "trigger",
        )
    )
    sessions = list(result.scalars().all())
    assert len(sessions) == 1


@pytest.mark.asyncio
async def test_event_trigger_skips_if_active_session_exists(db_session, test_project, test_agent):
    """Event trigger does not create duplicate session if one already active."""
    from huddleroom.services.task_service import TaskService
    task_service = TaskService()

    task = await task_service.create(db_session, test_project.id, TaskCreate(
        title="Event-triggered task",
        assigned_to=test_agent.id,
        trigger={"type": "event", "event_type": "artifact.pr_opened"},
    ))
    task.status = "ready"
    await db_session.flush()

    active_session = Session(
        task_id=task.id,
        agent_id=test_agent.id,
        project_id=test_project.id,
        adapter_type="api",
        status="running",
        origin="trigger",
    )
    db_session.add(active_session)
    await db_session.flush()

    event = BusEvent(
        id=uuid.uuid4(),
        project_id=test_project.id,
        event_type="artifact.pr_opened",
        payload={},
        source="system",
        emitted_at=datetime.now(timezone.utc),
    )
    await evaluate_event_triggers(db_session, event)

    result = await db_session.execute(
        select(Session).where(Session.task_id == task.id)
    )
    sessions = list(result.scalars().all())
    assert len(sessions) == 1  # still only the pre-existing one


@pytest.mark.asyncio
async def test_consumer_tasks_registry():
    """Consumer task registry is a dict (empty before lifespan starts in test context)."""
    from huddleroom.workers.consumers import get_consumer_tasks
    tasks = get_consumer_tasks()
    assert isinstance(tasks, dict)


@pytest.mark.asyncio
async def test_orchestration_health_endpoint(client, auth_headers):
    resp = await client.get("/api/v1/orchestration/health", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "consumers" in data
    assert "active_sessions" in data
    assert "event_bus_mode" in data
    assert "event_log_total" in data
    assert data["event_bus_mode"] in ("in_process", "redis_streams")
    for name in ("ws_hub", "rule_engine", "protocol_engine", "meeting_engine", "optimizer"):
        assert name in data["consumers"]


@pytest.mark.asyncio
async def test_orchestration_health_debug_disabled(client, auth_headers, monkeypatch):
    monkeypatch.setattr("huddleroom.routers.orchestration.settings.debug", False)
    resp = await client.get("/api/v1/orchestration/health", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["debug_enabled"] is False


@pytest.mark.asyncio
async def test_orchestration_health_debug_enabled(client, auth_headers, monkeypatch):
    monkeypatch.setattr("huddleroom.routers.orchestration.settings.debug", True)
    resp = await client.get("/api/v1/orchestration/health", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["debug_enabled"] is True
