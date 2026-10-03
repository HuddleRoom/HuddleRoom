from __future__ import annotations
import uuid
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class ArtifactCreate(BaseModel):
    name: str
    artifact_type: str
    path: str | None = None
    url: str | None = None
    metadata: dict = Field(default_factory=dict)
    linked_task_id: uuid.UUID | None = None
    created_by_agent: uuid.UUID | None = None
    created_by_user: uuid.UUID | None = None


class ArtifactUpdate(BaseModel):
    name: str | None = None
    status: str | None = None
    path: str | None = None
    url: str | None = None
    metadata: dict | None = None


class ArtifactResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    project_id: uuid.UUID
    name: str
    artifact_type: str
    status: str
    path: str | None
    url: str | None
    content_hash: str | None
    metadata_: dict = Field(alias="metadata", serialization_alias="metadata")
    linked_task_id: uuid.UUID | None
    version: int
    is_breaking: bool
    created_at: datetime
    updated_at: datetime


class WatchRequest(BaseModel):
    watcher_kind: str  # agent | user
    watcher_id: uuid.UUID
    event_filter: list[str] | None = None


class WatcherResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    artifact_id: uuid.UUID
    watcher_kind: str
    watcher_id: uuid.UUID
    event_filter: list | None
    created_at: datetime
