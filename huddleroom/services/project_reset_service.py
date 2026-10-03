from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.project import Project
from huddleroom.services.project_service import ProjectService
from huddleroom.services.agent_response_relay import (
    clear_project_reset,
    publish_project_reset,
    wait_for_project_reset_monitors,
)
from huddleroom.workers import meeting_tasks, task_runner

logger = logging.getLogger(__name__)


# ponytail: SQLite reset ownership is process-local; use PostgreSQL for multi-process deployments.
_SQLITE_RESET_LOCKS: dict[tuple[int, uuid.UUID], asyncio.Lock] = {}


class ProjectResetService:
    """Coordinate a reset around a durable project execution fence."""

    @asynccontextmanager
    async def _lock_project_reset(self, db: AsyncSession, project_id: uuid.UUID):
        """Serialize one project's reset lifecycle on every supported database."""
        bind = db.bind
        if bind is not None and bind.dialect.name == "postgresql":
            project_lock_key = int.from_bytes(project_id.bytes[:8], "big", signed=True)
            async with bind.connect() as lock_db:
                await lock_db.execute(text("SELECT pg_advisory_lock(:project_lock_key)"), {
                    "project_lock_key": project_lock_key,
                })
                try:
                    yield
                finally:
                    try:
                        result = await lock_db.execute(text("SELECT pg_advisory_unlock(:project_lock_key)"), {
                            "project_lock_key": project_lock_key,
                        })
                        if result.scalar_one() is not True:
                            raise RuntimeError("Failed to release project reset lock")
                    except BaseException:
                        await lock_db.invalidate()
                        raise
            return

        lock = _SQLITE_RESET_LOCKS.setdefault((id(asyncio.get_running_loop()), project_id), asyncio.Lock())
        async with lock:
            yield

    async def reset(
        self, db: AsyncSession, project_id: uuid.UUID, confirm_name: str
    ) -> dict[str, int | dict[str, int]]:
        project_service = ProjectService()
        async with self._lock_project_reset(db, project_id):
            await project_service.lock_workspace_boundary(db, project_id)
            project = await db.get(Project, project_id, populate_existing=True)
            if project is None:
                raise HTTPException(status_code=404, detail="Project not found")
            if confirm_name != project.name:
                raise HTTPException(status_code=400, detail="Project name confirmation does not match")
            if project.status == "archived":
                raise HTTPException(status_code=409, detail="Archived projects cannot be reset")

            project.status = "resetting"
            await db.commit()
            generation = await publish_project_reset(project_id)
            await wait_for_project_reset_monitors(project_id, generation)

            try:
                cancelled_sessions = await task_runner.cancel_project_sessions(project_id)
                cancelled_meeting_tasks = await meeting_tasks.cancel_project_meeting_tasks(project_id)
                await meeting_tasks.revoke_and_await_project_meeting_tasks(project_id)
            except Exception:
                await db.rollback()
                raise

            try:
                deletions = await project_service.reset(db, project_id)
                await clear_project_reset(project_id, generation)
                project = await db.get(Project, project_id)
                project.status = "active"
                try:
                    await db.commit()
                except Exception as exc:
                    await db.rollback()
                    try:
                        replacement = await publish_project_reset(project_id)
                        logger.error("Restored reset marker for %s generation=%s", project_id, replacement)
                    except Exception as restore_exc:
                        logger.critical(
                            "Could not restore reset marker for %s after commit failure %r: %r",
                            project_id,
                            exc,
                            restore_exc,
                            exc_info=True,
                        )
                    raise exc
            except Exception:
                await db.rollback()
                raise

            return {
                "cancelled_sessions": cancelled_sessions,
                "cancelled_meeting_tasks": cancelled_meeting_tasks,
                "deletions": deletions,
            }
