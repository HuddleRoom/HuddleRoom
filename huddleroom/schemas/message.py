from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class MessageCreate(BaseModel):
    content: str
    message_type: str = "text"
    metadata: dict = {}


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)
    id: UUID
    channel_id: UUID
    sender_agent_id: UUID | None
    sender_user_id: UUID | None
    content: str
    message_type: str
    metadata_: dict = Field(serialization_alias="metadata")
    created_at: datetime
