from __future__ import annotations

import uuid
from datetime import datetime
from pydantic import BaseModel, Field


class EventEmit(BaseModel):
    project_id: uuid.UUID
    event_type: str = Field(..., max_length=100)
    payload: dict = Field(default_factory=dict)
    source: str = Field(default="agent", max_length=50)


class EventResponse(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    event_type: str
    payload: dict
    source: str
    emitted_at: datetime

    model_config = {"from_attributes": True}
