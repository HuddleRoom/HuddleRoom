from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class SessionCreate(BaseModel):
    agent_id: UUID
    task_id: UUID | None = None
    project_id: UUID
    graph_run_id: UUID | None = None
    adapter_type_override: str | None = None
    context_override: dict = {}
    origin: Literal["manual", "auto", "trigger", "meeting", "graph"] = "manual"
    model_override: str | None = None
    timeout: int | None = None
    max_tokens: int | None = None


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)
    id: UUID
    task_id: UUID | None
    agent_id: UUID
    project_id: UUID
    graph_run_id: UUID | None
    adapter_type: str
    status: str
    input_context: dict
    output: str | None
    error: str | None
    runner_task_id: str | None
    sandbox_path: str | None
    metadata_: dict = Field(serialization_alias="metadata")
    origin: str
    started_at: datetime | None
    ended_at: datetime | None
    resumable: bool = False
    provider_session_id: str | None = None
    created_at: datetime


class SessionOutputResponse(BaseModel):
    session_id: UUID
    status: str
    adapter_type: str
    output: str | None
    error: str | None
    metadata: dict
