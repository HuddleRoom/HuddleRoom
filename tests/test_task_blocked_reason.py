"""TaskService.transition_status persists the reason for a blocked transition."""
from __future__ import annotations

import uuid
from datetime import datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.task import Task
from huddleroom.schemas.task import TaskCreate
from huddleroom.services.task_service import TaskService


async def _make_task(db: AsyncSession, project_id: uuid.UUID, status: str) -> Task:
    task = await TaskService().create(db, project_id, TaskCreate(title="Blocked Task"))
    task.status = status
    await db.flush()
    return task


@pytest.mark.asyncio
async def test_blocked_transition_stores_reason(db_session: AsyncSession, test_project):
    task = await _make_task(db_session, test_project.id, "in_progress")
    svc = TaskService()
    updated = await svc.transition_status(
        db_session, test_project.id, task.id, "blocked", "waiting on API key"
    )
    assert updated.status == "blocked"
    blocked = updated.metadata_["blocked"]
    assert blocked["reason"] == "waiting on API key"
    datetime.fromisoformat(blocked["at"])  # valid ISO timestamp


@pytest.mark.asyncio
async def test_blocked_transition_without_reason_stores_none(db_session: AsyncSession, test_project):
    task = await _make_task(db_session, test_project.id, "in_progress")
    svc = TaskService()
    updated = await svc.transition_status(db_session, test_project.id, task.id, "blocked")
    assert updated.metadata_["blocked"]["reason"] is None


@pytest.mark.asyncio
async def test_blocked_transition_preserves_existing_metadata(db_session: AsyncSession, test_project):
    task = await _make_task(db_session, test_project.id, "in_progress")
    task.metadata_ = {"keep": 1}
    await db_session.flush()
    updated = await TaskService().transition_status(
        db_session, test_project.id, task.id, "blocked", "r"
    )
    assert updated.metadata_["keep"] == 1
    assert updated.metadata_["blocked"]["reason"] == "r"


@pytest.mark.asyncio
async def test_other_transitions_do_not_add_blocked_key(db_session: AsyncSession, test_project):
    task = await _make_task(db_session, test_project.id, "in_progress")
    updated = await TaskService().transition_status(
        db_session, test_project.id, task.id, "done", "some reason"
    )
    assert updated.status == "done"
    assert "blocked" not in (updated.metadata_ or {})


@pytest.mark.asyncio
async def test_failed_transition_does_not_add_blocked_key(db_session: AsyncSession, test_project):
    task = await _make_task(db_session, test_project.id, "in_progress")
    updated = await TaskService().transition_status(
        db_session, test_project.id, task.id, "failed", "boom"
    )
    assert updated.status == "failed"
    assert "blocked" not in (updated.metadata_ or {})
