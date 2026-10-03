from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ChannelCreate(BaseModel):
    name: str
    channel_type: str
    task_id: UUID | None = None
    members: list[UUID] = []


class ChannelResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    project_id: UUID | None
    name: str
    channel_type: str
    task_id: UUID | None
    members: list[UUID] | None
    created_at: datetime
