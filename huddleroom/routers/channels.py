import uuid
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user
from huddleroom.models.user import User
from huddleroom.schemas.channel import ChannelCreate, ChannelResponse
from huddleroom.services.channel_service import ChannelService

router = APIRouter()
service = ChannelService()


@router.get("/projects/{project_id}/channels", response_model=list[ChannelResponse])
async def list_channels(
    project_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.list(db, project_id)


@router.post("/projects/{project_id}/channels", response_model=ChannelResponse, status_code=201)
async def create_channel(
    project_id: uuid.UUID,
    data: ChannelCreate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.create(db, project_id, data)


@router.get("/channels/{channel_id}", response_model=ChannelResponse)
async def get_channel(
    channel_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.get_or_404(db, channel_id)
