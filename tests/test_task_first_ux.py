"""Tests for task-first UX features.

Coverage:
  Task model transitions (failed status)
  Session origin field
  sync_task_from_session
  _maybe_auto_session (auto-session on ready+assigned)
  TaskService.run()
  Router endpoints (POST /run, GET /runs)
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.session import Session as SessionModel
from huddleroom.models.task import Task
from huddleroom.schemas.session import SessionCreate, SessionResponse
from huddleroom.schemas.task import TaskCreate, TaskRunRequest
from huddleroom.services.session_service import SessionService, sync_task_from_session
from huddleroom.services.task_service import TaskService


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

async def _make_task(
    db: AsyncSession,
    project_id: uuid.UUID,
    status: str = "ready",
    assigned_to: uuid.UUID | None = None,
) -> Task:
    """Create a Task row with an explicit status."""
    svc = TaskService()
    task = await svc.create(
        db, project_id, TaskCreate(title="Test Task", assigned_to=assigned_to)
    )
    task.status = status
    await db.flush()
    return task


# ===========================================================================
# Task model transitions (failed status)
# ===========================================================================


@pytest.mark.asyncio
async def test_in_progress_to_failed_valid(db_session: AsyncSession, test_project):
    """in_progress → failed is a valid transition."""
    task = await _make_task(db_session, test_project.id, status="in_progress")
    svc = TaskService()
    updated = await svc.transition_status(db_session, test_project.id, task.id, "failed")
    assert updated.status == "failed"


@pytest.mark.asyncio
async def test_failed_to_in_progress_valid(db_session: AsyncSession, test_project):
    """failed → in_progress is a valid transition."""
    task = await _make_task(db_session, test_project.id, status="failed")
    svc = TaskService()
    updated = await svc.transition_status(db_session, test_project.id, task.id, "in_progress")
    assert updated.status == "in_progress"


@pytest.mark.asyncio
async def test_failed_to_cancelled_valid(db_session: AsyncSession, test_project):
    """failed → cancelled is a valid transition."""
    task = await _make_task(db_session, test_project.id, status="failed")
    svc = TaskService()
    updated = await svc.transition_status(db_session, test_project.id, task.id, "cancelled")
    assert updated.status == "cancelled"


@pytest.mark.asyncio
async def test_backlog_to_failed_invalid(db_session: AsyncSession, test_project):
    """backlog → failed is NOT valid; expect HTTPException 409."""
    task = await _make_task(db_session, test_project.id, status="backlog")
    svc = TaskService()
    with pytest.raises(HTTPException) as exc_info:
        await svc.transition_status(db_session, test_project.id, task.id, "failed")
    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_ready_to_failed_invalid(db_session: AsyncSession, test_project):
    """ready → failed is NOT valid; expect HTTPException 409."""
    task = await _make_task(db_session, test_project.id, status="ready")
    svc = TaskService()
    with pytest.raises(HTTPException) as exc_info:
        await svc.transition_status(db_session, test_project.id, task.id, "failed")
    assert exc_info.value.status_code == 409


# ===========================================================================
# Session origin field
# ===========================================================================


def test_session_create_defaults_origin_to_manual():
    """SessionCreate defaults origin to 'manual'."""
    data = SessionCreate(agent_id=uuid.uuid4(), project_id=uuid.uuid4())
    assert data.origin == "manual"


@pytest.mark.asyncio
async def test_session_created_with_trigger_origin_persists(
    db_session: AsyncSession, test_project, test_agent
):
    """Session created with origin='trigger' persists origin correctly."""
    session = SessionModel(
        agent_id=test_agent.id,
        project_id=test_project.id,
        adapter_type="api",
        status="pending",
        input_context={},
        metadata_={},
        origin="trigger",
    )
    db_session.add(session)
    await db_session.flush()
    await db_session.refresh(session)
    assert session.origin == "trigger"


@pytest.mark.asyncio
async def test_session_response_includes_origin(
    db_session: AsyncSession, test_project, test_agent
):
    """SessionResponse serialization includes origin field."""
    session = SessionModel(
        agent_id=test_agent.id,
        project_id=test_project.id,
        adapter_type="api",
        status="pending",
        input_context={},
        metadata_={},
        origin="manual",
    )
    db_session.add(session)
    await db_session.flush()
    await db_session.refresh(session)

    response = SessionResponse.model_validate(session)
    assert hasattr(response, "origin")
    assert response.origin == "manual"


# ===========================================================================
# sync_task_from_session
# ===========================================================================


@pytest.mark.asyncio
async def test_sync_task_session_completed_sets_done(
    db_session: AsyncSession, test_project, test_agent
):
    """When session.status='completed': task status → 'done', completed_at set."""
    task = await _make_task(db_session, test_project.id, status="in_progress")
    session = SessionModel(
        agent_id=test_agent.id,
        project_id=test_project.id,
        task_id=task.id,
        adapter_type="api",
        status="completed",
        input_context={},
        metadata_={},
    )
    db_session.add(session)
    await db_session.flush()

    await sync_task_from_session(db_session, session)

    await db_session.refresh(task)
    assert task.status == "done"
    assert task.completed_at is not None


@pytest.mark.asyncio
async def test_sync_task_session_failed_sets_failed(
    db_session: AsyncSession, test_project, test_agent
):
    """When session.status='failed': task status → 'failed'."""
    task = await _make_task(db_session, test_project.id, status="in_progress")
    session = SessionModel(
        agent_id=test_agent.id,
        project_id=test_project.id,
        task_id=task.id,
        adapter_type="api",
        status="failed",
        input_context={},
        metadata_={},
    )
    db_session.add(session)
    await db_session.flush()

    await sync_task_from_session(db_session, session)

    await db_session.refresh(task)
    assert task.status == "failed"


@pytest.mark.asyncio
async def test_sync_task_no_task_id_noop(
    db_session: AsyncSession, test_project, test_agent
):
    """When session has no task_id: no-op."""
    session = SessionModel(
        agent_id=test_agent.id,
        project_id=test_project.id,
        task_id=None,
        adapter_type="api",
        status="completed",
        input_context={},
        metadata_={},
    )
    db_session.add(session)
    await db_session.flush()

    # Should not raise and nothing happens
    await sync_task_from_session(db_session, session)


@pytest.mark.asyncio
async def test_sync_task_session_running_noop(
    db_session: AsyncSession, test_project, test_agent
):
    """When session.status='running': no-op (not terminal)."""
    task = await _make_task(db_session, test_project.id, status="in_progress")
    session = SessionModel(
        agent_id=test_agent.id,
        project_id=test_project.id,
        task_id=task.id,
        adapter_type="api",
        status="running",
        input_context={},
        metadata_={},
    )
    db_session.add(session)
    await db_session.flush()

    await sync_task_from_session(db_session, session)

    await db_session.refresh(task)
    assert task.status == "in_progress"


@pytest.mark.asyncio
async def test_sync_task_already_done_noop(
    db_session: AsyncSession, test_project, test_agent
):
    """When task.status='done' already: no-op (don't touch completed tasks)."""
    task = await _make_task(db_session, test_project.id, status="done")
    session = SessionModel(
        agent_id=test_agent.id,
        project_id=test_project.id,
        task_id=task.id,
        adapter_type="api",
        status="completed",
        input_context={},
        metadata_={},
    )
    db_session.add(session)
    await db_session.flush()

    await sync_task_from_session(db_session, session)

    await db_session.refresh(task)
    assert task.status == "done"


@pytest.mark.asyncio
async def test_sync_task_already_cancelled_noop(
    db_session: AsyncSession, test_project, test_agent
):
    """When task.status='cancelled': no-op."""
    task = await _make_task(db_session, test_project.id, status="cancelled")
    session = SessionModel(
        agent_id=test_agent.id,
        project_id=test_project.id,
        task_id=task.id,
        adapter_type="api",
        status="completed",
        input_context={},
        metadata_={},
    )
    db_session.add(session)
    await db_session.flush()

    await sync_task_from_session(db_session, session)

    await db_session.refresh(task)
    assert task.status == "cancelled"


# ===========================================================================
# _maybe_auto_session (auto-session on ready+assigned)
# ===========================================================================


@pytest.mark.asyncio
async def test_auto_session_created_on_ready_with_agent(
    db_session: AsyncSession, runnable_project, test_agent
):
    """Transitioning task to 'ready' with assigned agent → session auto-created, task → 'in_progress'."""
    task = await _make_task(
        db_session, runnable_project.id, status="backlog", assigned_to=test_agent.id
    )
    svc = TaskService()

    with patch(
        "huddleroom.workers.task_runner.dispatch_session",
        new=AsyncMock(return_value="fake-celery-id"),
    ):
        updated = await svc.transition_status(
            db_session, runnable_project.id, task.id, "ready"
        )

    await db_session.refresh(updated)
    assert updated.status == "in_progress"

    # Verify a session was created
    from sqlalchemy import select
    result = await db_session.execute(
        select(SessionModel).where(SessionModel.task_id == task.id)
    )
    sessions = result.scalars().all()
    assert len(sessions) >= 1


@pytest.mark.asyncio
async def test_auto_session_not_created_when_no_agent(
    db_session: AsyncSession, test_project
):
    """Transitioning task to 'ready' WITHOUT assigned agent → no session, stays 'ready'."""
    task = await _make_task(db_session, test_project.id, status="backlog")
    svc = TaskService()

    with patch(
        "huddleroom.workers.task_runner.dispatch_session",
        new=AsyncMock(return_value="fake-celery-id"),
    ):
        updated = await svc.transition_status(
            db_session, test_project.id, task.id, "ready"
        )

    await db_session.refresh(updated)
    assert updated.status == "ready"

    from sqlalchemy import select
    result = await db_session.execute(
        select(SessionModel).where(SessionModel.task_id == task.id)
    )
    sessions = result.scalars().all()
    assert len(sessions) == 0


@pytest.mark.asyncio
async def test_auto_session_on_assign_to_ready_task(
    db_session: AsyncSession, runnable_project, test_agent
):
    """Assigning agent to already-'ready' task → session auto-created."""
    task = await _make_task(db_session, runnable_project.id, status="ready")
    svc = TaskService()

    with patch(
        "huddleroom.workers.task_runner.dispatch_session",
        new=AsyncMock(return_value="fake-celery-id"),
    ):
        await svc.assign(db_session, runnable_project.id, task.id, test_agent.id)

    from sqlalchemy import select
    result = await db_session.execute(
        select(SessionModel).where(SessionModel.task_id == task.id)
    )
    sessions = result.scalars().all()
    assert len(sessions) >= 1


@pytest.mark.asyncio
async def test_auto_session_no_duplicate_if_pending_exists(
    db_session: AsyncSession, test_project, test_agent
):
    """If pending session already exists for task → no duplicate session created."""
    task = await _make_task(
        db_session, test_project.id, status="ready", assigned_to=test_agent.id
    )
    # Pre-create a pending session
    existing_session = SessionModel(
        agent_id=test_agent.id,
        project_id=test_project.id,
        task_id=task.id,
        adapter_type="api",
        status="pending",
        input_context={},
        metadata_={},
    )
    db_session.add(existing_session)
    await db_session.flush()

    svc = TaskService()
    with patch(
        "huddleroom.workers.task_runner.dispatch_session",
        new=AsyncMock(return_value="fake-celery-id"),
    ):
        await svc._maybe_auto_session(db_session, task)

    from sqlalchemy import select
    result = await db_session.execute(
        select(SessionModel).where(SessionModel.task_id == task.id)
    )
    sessions = result.scalars().all()
    assert len(sessions) == 1


# ===========================================================================
# TaskService.run()
# ===========================================================================


@pytest.mark.asyncio
async def test_run_no_agent_raises_400(db_session: AsyncSession, test_project):
    """run() on task with no agent → HTTPException 400."""
    task = await _make_task(db_session, test_project.id, status="ready")
    svc = TaskService()

    with pytest.raises(HTTPException) as exc_info:
        await svc.run(db_session, test_project.id, task.id)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_run_with_active_session_raises_409(
    db_session: AsyncSession, test_project, test_agent
):
    """run() on task with active (pending) session → HTTPException 409."""
    task = await _make_task(
        db_session, test_project.id, status="ready", assigned_to=test_agent.id
    )
    # Pre-create a pending session
    existing_session = SessionModel(
        agent_id=test_agent.id,
        project_id=test_project.id,
        task_id=task.id,
        adapter_type="api",
        status="pending",
        input_context={},
        metadata_={},
    )
    db_session.add(existing_session)
    await db_session.flush()

    svc = TaskService()
    with pytest.raises(HTTPException) as exc_info:
        await svc.run(db_session, test_project.id, task.id)
    assert exc_info.value.status_code == 409


@pytest.mark.parametrize(
    "starting_status",
    ["backlog", "ready", "failed", "blocked", "in_progress"],
)
@pytest.mark.asyncio
async def test_run_on_task_creates_session(
    starting_status: str, db_session: AsyncSession, runnable_project, test_agent
):
    """run() on assigned task in various statuses → session created, task in 'in_progress'."""
    task = await _make_task(
        db_session, runnable_project.id, status=starting_status, assigned_to=test_agent.id
    )
    svc = TaskService()

    with patch(
        "huddleroom.workers.task_runner.dispatch_session",
        new=AsyncMock(return_value="fake-celery-id"),
    ):
        result_task, session_id = await svc.run(
            db_session, runnable_project.id, task.id
        )

    assert result_task.status == "in_progress"
    assert session_id is not None


@pytest.mark.asyncio
async def test_run_session_has_origin_manual(
    db_session: AsyncSession, runnable_project, test_agent
):
    """Session created by run() has origin='manual'."""
    task = await _make_task(
        db_session, runnable_project.id, status="ready", assigned_to=test_agent.id
    )
    svc = TaskService()

    with patch(
        "huddleroom.workers.task_runner.dispatch_session",
        new=AsyncMock(return_value="fake-celery-id"),
    ):
        _, session_id = await svc.run(db_session, runnable_project.id, task.id)

    svc_session = SessionService()
    session = await svc_session.get(db_session, session_id)
    assert session is not None
    assert session.origin == "manual"


# ===========================================================================
# Router endpoints
# ===========================================================================


@pytest.mark.asyncio
async def test_router_run_task_returns_200(
    client: AsyncClient, auth_headers: dict, db_session: AsyncSession, runnable_project, test_agent
):
    """POST /api/v1/projects/{pid}/tasks/{tid}/run returns 200 with task+session_id."""
    # Create a ready task with agent assigned
    task = await _make_task(
        db_session, runnable_project.id, status="ready", assigned_to=test_agent.id
    )

    with patch(
        "huddleroom.workers.task_runner.dispatch_session",
        new=AsyncMock(return_value="fake-celery-id"),
    ):
        resp = await client.post(
            f"/api/v1/projects/{runnable_project.id}/tasks/{task.id}/run",
            json={},
            headers=auth_headers,
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "task" in data
    assert "session_id" in data
    assert data["task"]["status"] == "in_progress"


@pytest.mark.asyncio
async def test_router_run_task_no_agent_returns_400(
    client: AsyncClient, auth_headers: dict, db_session: AsyncSession, test_project
):
    """POST /api/v1/projects/{pid}/tasks/{tid}/run on task with no agent returns 400."""
    task = await _make_task(db_session, test_project.id, status="ready")

    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/tasks/{task.id}/run",
        json={},
        headers=auth_headers,
    )

    assert resp.status_code == 400, resp.text


@pytest.mark.asyncio
async def test_router_get_runs_returns_session_list(
    client: AsyncClient, auth_headers: dict, db_session: AsyncSession, test_project, test_agent
):
    """GET /api/v1/projects/{pid}/tasks/{tid}/runs returns session list."""
    task = await _make_task(
        db_session, test_project.id, status="in_progress", assigned_to=test_agent.id
    )
    # Create a session for the task directly (bypass dispatch)
    session = SessionModel(
        agent_id=test_agent.id,
        project_id=test_project.id,
        task_id=task.id,
        adapter_type="api",
        status="completed",
        input_context={},
        metadata_={},
        origin="manual",
    )
    db_session.add(session)
    await db_session.flush()

    resp = await client.get(
        f"/api/v1/projects/{test_project.id}/tasks/{task.id}/runs",
        headers=auth_headers,
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "items" in data
    assert len(data["items"]) >= 1
    item = data["items"][0]
    assert item["task_id"] == str(task.id)
    assert "origin" in item


@pytest.mark.asyncio
async def test_router_get_sessions_returns_same_project_session_output(
    client: AsyncClient, auth_headers: dict, db_session: AsyncSession, test_project, test_agent
):
    """GET /api/v1/projects/{pid}/tasks/{tid}/sessions returns same-project session output."""
    task = await _make_task(
        db_session, test_project.id, status="in_progress", assigned_to=test_agent.id
    )
    session = SessionModel(
        agent_id=test_agent.id,
        project_id=test_project.id,
        task_id=task.id,
        adapter_type="api",
        status="completed",
        output="same project output",
        input_context={},
        metadata_={},
        origin="manual",
    )
    db_session.add(session)
    await db_session.flush()

    resp = await client.get(
        f"/api/v1/projects/{test_project.id}/tasks/{task.id}/sessions",
        headers=auth_headers,
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["items"][0]["project_id"] == str(test_project.id)
    assert data["items"][0]["output"] == "same project output"
