from __future__ import annotations
import uuid
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class HookCreate(BaseModel):
    name: str
    trigger_event: str
    code: str
    description: str | None = None


class HookUpdate(BaseModel):
    name: str | None = None
    trigger_event: str | None = None
    code: str | None = None
    description: str | None = None
    status: str | None = None


class HookResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    name: str
    description: str | None
    code: str
    status: str
    trigger_event: str
    execution_count: int
    error_count: int
    created_at: datetime
    updated_at: datetime
