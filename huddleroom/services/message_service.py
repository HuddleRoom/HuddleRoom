from __future__ import annotations

import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.message import Message


class MessageService:
    async def create(
        self,
        db: AsyncSession,
        channel_id: uuid.UUID,
        content: str,
        sender_agent_id: uuid.UUID | None = None,
        sender_user_id: uuid.UUID | None = None,
        message_type: str = "text",
        metadata: dict | None = None,
    ) -> Message:
        msg = Message(
            channel_id=channel_id,
            content=content,
            sender_agent_id=sender_agent_id,
            sender_user_id=sender_user_id,
            message_type=message_type,
            metadata_=metadata or {},
        )
        db.add(msg)
        await db.flush()
        return msg

    async def list(
        self,
        db: AsyncSession,
        channel_id: uuid.UUID,
        cursor: str | None = None,
        limit: int = 50,
    ) -> tuple[list[Message], str | None]:
        query = (
            select(Message)
            .where(Message.channel_id == channel_id)
            .order_by(Message.created_at.desc())
            .limit(limit + 1)
        )
        result = await db.execute(query)
        items = list(result.scalars().all())
        next_cursor = None
        if len(items) > limit:
            items = items[:limit]
            next_cursor = items[-1].created_at.isoformat()
        return items, next_cursor
