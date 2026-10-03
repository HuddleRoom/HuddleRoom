from __future__ import annotations

import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException

from huddleroom.models.channel import Channel
from huddleroom.schemas.channel import ChannelCreate


class ChannelService:
    async def create(self, db: AsyncSession, project_id: uuid.UUID, data: ChannelCreate) -> Channel:
        channel = Channel(
            project_id=project_id,
            name=data.name,
            channel_type=data.channel_type,
            task_id=data.task_id,
            members=data.members or [],
        )
        db.add(channel)
        await db.flush()
        return channel

    async def get(self, db: AsyncSession, channel_id: uuid.UUID) -> Channel | None:
        result = await db.execute(select(Channel).where(Channel.id == channel_id))
        return result.scalar_one_or_none()

    async def get_or_404(self, db: AsyncSession, channel_id: uuid.UUID) -> Channel:
        channel = await self.get(db, channel_id)
        if not channel:
            raise HTTPException(status_code=404, detail="Channel not found")
        return channel

    async def list(self, db: AsyncSession, project_id: uuid.UUID) -> list[Channel]:
        result = await db.execute(
            select(Channel).where(Channel.project_id == project_id).order_by(Channel.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_or_create_for_task(
        self, db: AsyncSession, project_id: uuid.UUID, task_id: uuid.UUID
    ) -> Channel:
        result = await db.execute(
            select(Channel).where(Channel.task_id == task_id, Channel.channel_type == "task")
        )
        channel = result.scalar_one_or_none()
        if channel:
            return channel
        channel = Channel(
            project_id=project_id,
            name=f"task-{task_id}",
            channel_type="task",
            task_id=task_id,
            members=[],
        )
        db.add(channel)
        await db.flush()
        return channel
