from __future__ import annotations
import uuid
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class RoutingRuleCreate(BaseModel):
    name: str
    on_event: str
    conditions: dict = {}
    actions: dict = {}
    description: str | None = None
    priority: int = 0
    enabled: bool = True


class RoutingRuleUpdate(BaseModel):
    name: str | None = None
    on_event: str | None = None
    conditions: dict | None = None
    actions: dict | None = None
    description: str | None = None
    priority: int | None = None
    enabled: bool | None = None


class RoutingRuleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    name: str
    description: str | None
    priority: int
    on_event: str
    conditions: dict
    actions: dict
    enabled: bool
    created_at: datetime
    updated_at: datetime
