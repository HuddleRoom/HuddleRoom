from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeCreate(BaseModel):
    title: str | None = None
    content: str
    content_type: str
    tags: list[str] = []
    provenance: dict | None = None
    supersedes: list[UUID] | None = None


class KnowledgeUpdate(BaseModel):
    title: str | None = None
    content: str | None = None
    tags: list[str] | None = None


class KnowledgeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)
    id: UUID
    project_id: UUID | None
    title: str | None
    content: str
    content_type: str
    tags: list[str] | None
    provenance_type: str
    provenance_protocol_instance_id: UUID | None = None
    is_superseded: bool
    version: int
    conflict_status: str
    metadata_: dict = Field(serialization_alias="metadata")
    created_at: datetime
    updated_at: datetime


class KnowledgeSearchRequest(BaseModel):
    query: str
    limit: int = 10
    content_type: str | None = None
    min_relevance_score: float = 0.7


class KnowledgeSearchResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    project_id: UUID | None
    title: str | None
    content: str
    content_type: str
    tags: list[str] | None
    relevance_score: float
