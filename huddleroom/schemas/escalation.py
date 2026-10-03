from __future__ import annotations
import uuid
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class EscalationChainCreate(BaseModel):
    name: str
    description: str | None = None
    definition: dict
    steps: list


class EscalationChainResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID | None
    name: str
    description: str | None
    steps: list
    is_active: bool
    created_at: datetime
    updated_at: datetime
