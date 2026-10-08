from __future__ import annotations

import uuid
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException

from huddleroom.models.task import Task
from huddleroom.models.agent import Agent
from huddleroom.schemas.task import TaskCreate, TaskUpdate
from huddleroom.services.event_bus import emit_event


class TaskService:
    async def create(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        data: TaskCreate,
        task_id: uuid.UUID | None = None,
    ) -> Task:
        task_kwargs = {
            "project_id": project_id,
            "title": data.title,
            "description": data.description,
            "priority": data.priority,
            "assigned_to": data.assigned_to,
            "adapter_type_override": data.adapter_type_override,
            "trigger": data.trigger,
            "metadata_": data.metadata,
            "due_at": data.due_at,
            "parent_id": data.parent_id,
        }
        if task_id is not None:
            task_kwargs["id"] = task_id
        task = Task(**task_kwargs)
        db.add(task)
        await db.flush()
        await emit_event(db, project_id, "task.created", {
            "task_id": str(task.id),
            "title": task.title,
            "status": task.status,
            "assigned_to": str(task.assigned_to) if task.assigned_to else None,
            "project_id": str(project_id),
        })
        return task

    async def get(self, db: AsyncSession, project_id: uuid.UUID, task_id: uuid.UUID) -> Task | None:
        result = await db.execute(
            select(Task).where(Task.id == task_id, Task.project_id == project_id)
        )
        return result.scalar_one_or_none()

    async def get_or_404(self, db: AsyncSession, project_id: uuid.UUID, task_id: uuid.UUID) -> Task:
        task = await self.get(db, project_id, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task

    async def list(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        status: str | None = None,
        assigned_to: uuid.UUID | None = None,
        parent_id: uuid.UUID | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> tuple[list[Task], str | None]:
        from sqlalchemy import or_, and_
        query = (
            select(Task)
            .where(Task.project_id == project_id)
            .order_by(Task.created_at.desc(), Task.id.desc())
            .limit(limit + 1)
        )
        if status:
            query = query.where(Task.status == status)
        if assigned_to:
            query = query.where(Task.assigned_to == assigned_to)
        if parent_id is not None:
            query = query.where(Task.parent_id == parent_id)
        if cursor:
            try:
                cursor_dt_str, cursor_id_str = cursor.split("__", 1)
                cursor_dt = datetime.fromisoformat(cursor_dt_str)
                cursor_id = uuid.UUID(cursor_id_str)
            except (ValueError, AttributeError) as exc:
                raise HTTPException(status_code=400, detail="Invalid cursor") from exc
            query = query.where(
                or_(
                    Task.created_at < cursor_dt,
                    and_(Task.created_at == cursor_dt, Task.id < cursor_id),
                )
            )
        result = await db.execute(query)
        items = list(result.scalars().all())
        next_cursor = None
        if len(items) > limit:
            items = items[:limit]
            last = items[-1]
            next_cursor = f"{last.created_at.isoformat()}__{last.id}"
        return items, next_cursor

    async def update(self, db: AsyncSession, project_id: uuid.UUID, task_id: uuid.UUID, data: TaskUpdate) -> Task:
        task = await self.get_or_404(db, project_id, task_id)
        if data.title is not None:
            task.title = data.title
        if data.description is not None:
            task.description = data.description
        if data.priority is not None:
            task.priority = data.priority
        if data.assigned_to is not None:
            task.assigned_to = data.assigned_to
        if data.adapter_type_override is not None:
            task.adapter_type_override = data.adapter_type_override
        if data.trigger is not None:
            task.trigger = data.trigger
        if data.metadata is not None:
            task.metadata_ = data.metadata
        if data.due_at is not None:
            task.due_at = data.due_at
        await db.flush()
        return task

    async def transition_status(
        self, db: AsyncSession, project_id: uuid.UUID, task_id: uuid.UUID, new_status: str, reason: str | None = None
    ) -> Task:
        task = await self.get_or_404(db, project_id, task_id)
        current = task.status
        valid = Task.VALID_TRANSITIONS.get(current, set())
        if new_status not in valid:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot transition from '{current}' to '{new_status}'",
            )
        task.status = new_status
        now = datetime.now(timezone.utc)
        if new_status == "in_progress" and task.started_at is None:
            task.started_at = now
        if new_status == "done":
            task.completed_at = now
        if new_status == "blocked":
            # Copy + reassign so SQLAlchemy detects the JSON change.
            metadata = dict(task.metadata_ or {})
            metadata["blocked"] = {"reason": reason or None, "at": now.isoformat()}
            task.metadata_ = metadata
        await db.flush()
        await emit_event(db, project_id, "task.status_changed", {
            "task_id": str(task.id),
            "status": new_status,
            "previous_status": current,
            "project_id": str(project_id),
        })
        if new_status == "ready":
            await self._maybe_auto_session(db, task)
        return task

    async def assign(self, db: AsyncSession, project_id: uuid.UUID, task_id: uuid.UUID, agent_id: uuid.UUID) -> Task:
        task = await self.get_or_404(db, project_id, task_id)
        task.assigned_to = agent_id
        await db.flush()
        await emit_event(db, task.project_id, "task.assigned", {
            "task_id": str(task.id),
            "agent_id": str(agent_id),
            "project_id": str(task.project_id),
        })
        if task.status == "ready":
            await self._maybe_auto_session(db, task)
        return task

    async def list_subtasks(self, db: AsyncSession, project_id: uuid.UUID, parent_id: uuid.UUID) -> list[Task]:
        result = await db.execute(
            select(Task).where(Task.project_id == project_id, Task.parent_id == parent_id)
        )
        return list(result.scalars().all())

    async def cancel(self, db: AsyncSession, project_id: uuid.UUID, task_id: uuid.UUID) -> Task:
        return await self.transition_status(db, project_id, task_id, "cancelled")

    async def run(
        self, db: AsyncSession, project_id: uuid.UUID, task_id: uuid.UUID,
        adapter_type_override: str | None = None, context_override: dict | None = None,
        model_override: str | None = None, timeout: int | None = None, max_tokens: int | None = None,
    ) -> tuple:
        task = await self.get_or_404(db, project_id, task_id)
        if not task.assigned_to:
            raise HTTPException(status_code=400, detail="Task has no assigned agent")
        # Guard: no active session
        from huddleroom.models.session import Session
        existing = await db.execute(
            select(Session).where(
                Session.task_id == task.id,
                Session.status.in_(["pending", "running"]),
            )
        )
        if existing.scalar_one_or_none() is not None:
            raise HTTPException(status_code=409, detail="Task already has an active session")
        # Refuse a Roadmap claim before mutating task state or emitting events.
        from huddleroom.services.session_service import SessionService
        agent = await db.get(Agent, task.assigned_to)
        if agent is not None:
            await SessionService().preflight_claim(db, task, agent, adapter_type_override)
        # Transition to in_progress if not already
        now = datetime.now(timezone.utc)
        if task.status not in ("in_progress",):
            allowed = {"backlog", "ready", "failed", "blocked"}
            if task.status not in allowed:
                raise HTTPException(
                    status_code=409,
                    detail=f"Cannot run task in status '{task.status}'",
                )
            previous_status = task.status
            task.status = "in_progress"
            if task.started_at is None:
                task.started_at = now
            await db.flush()
            await emit_event(db, task.project_id, "task.status_changed", {
                "task_id": str(task.id),
                "status": "in_progress",
                "previous_status": previous_status,
                "project_id": str(task.project_id),
            })
        # Create session
        from huddleroom.services.session_service import SessionService
        from huddleroom.schemas.session import SessionCreate
        session_data = SessionCreate(
            agent_id=task.assigned_to,
            task_id=task.id,
            project_id=task.project_id,
            adapter_type_override=adapter_type_override,
            context_override=context_override or {},
            origin="manual",
            model_override=model_override,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        session = await SessionService().create(db, session_data)
        return task, session.id

    async def _maybe_auto_session(self, db: AsyncSession, task: Task) -> None:
        """Auto-create a session if task is ready and assigned.

        Checks if a pending/running session already exists for this task.
        If not, creates an auto session and transitions the task to in_progress.
        """
        if task.status != "ready" or task.assigned_to is None:
            return
        # Guard: skip if pending/running session already exists for this task
        from sqlalchemy import select
        from huddleroom.models.session import Session
        existing = await db.execute(
            select(Session).where(
                Session.task_id == task.id,
                Session.status.in_(["pending", "running"]),
            )
        )
        if existing.scalar_one_or_none() is not None:
            return
        # Create auto session
        from huddleroom.services.session_service import SessionService
        from huddleroom.schemas.session import SessionCreate
        session_data = SessionCreate(
            agent_id=task.assigned_to,
            task_id=task.id,
            project_id=task.project_id,
            origin="auto",
        )
        await SessionService().create(db, session_data)
        await self.transition_status(db, task.project_id, task.id, "in_progress")
