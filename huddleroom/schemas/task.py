from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class TaskCreate(BaseModel):
    title: str
    description: str | None = None
    priority: int = 50
    assigned_to: UUID | None = None
    adapter_type_override: str | None = None
    trigger: dict | None = None
    metadata: dict = {}
    due_at: datetime | None = None
    parent_id: UUID | None = None


class TaskUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    priority: int | None = None
    assigned_to: UUID | None = None
    adapter_type_override: str | None = None
    trigger: dict | None = None
    metadata: dict | None = None
    due_at: datetime | None = None


class TaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)
    id: UUID
    project_id: UUID
    parent_id: UUID | None
    title: str
    description: str | None
    status: str
    priority: int
    assigned_to: UUID | None
    adapter_type_override: str | None
    trigger: dict | None
    metadata_: dict = Field(serialization_alias="metadata")
    due_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class StatusPatch(BaseModel):
    status: str
    reason: str | None = None


class TaskAssign(BaseModel):
    agent_id: UUID


class TaskRunRequest(BaseModel):
    adapter_type_override: str | None = None
    context_override: dict = {}
    model_override: str | None = None
    timeout: int | None = None
    max_tokens: int | None = None


class TaskRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    task: TaskResponse
    session_id: UUID
