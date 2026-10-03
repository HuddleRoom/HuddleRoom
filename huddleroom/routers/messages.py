import uuid
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user
from huddleroom.models.user import User
from huddleroom.schemas.message import MessageCreate, MessageResponse
from huddleroom.schemas.common import CursorPage
from huddleroom.services.channel_service import ChannelService
from huddleroom.services.message_service import MessageService

router = APIRouter()
channel_service = ChannelService()
message_service = MessageService()


@router.get("/channels/{channel_id}/messages", response_model=CursorPage[MessageResponse])
async def list_messages(
    channel_id: uuid.UUID,
    cursor: str | None = None,
    limit: int = 50,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await channel_service.get_or_404(db, channel_id)
    items, next_cursor = await message_service.list(db, channel_id, cursor=cursor, limit=limit)
    return CursorPage(items=items, next_cursor=next_cursor)


@router.post("/channels/{channel_id}/messages", response_model=MessageResponse, status_code=201)
async def create_message(
    channel_id: uuid.UUID,
    data: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await channel_service.get_or_404(db, channel_id)
    return await message_service.create(
        db,
        channel_id=channel_id,
        content=data.content,
        sender_user_id=current_user.id,
        message_type=data.message_type,
        metadata=data.metadata,
    )
