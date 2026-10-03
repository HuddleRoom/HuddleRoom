from datetime import datetime
from typing import Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class MemoryCreate(BaseModel):
    content: str
    tags: list[str] = []
    shared: bool = True
    scope: Literal["project", "global"] = "project"


class MemoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    agent_id: UUID
    project_id: UUID | None
    scope: str
    content: str
    tags: list[str]
    shared: bool
    created_at: datetime


class MemorySearchRequest(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=100)
    tags: list[str] | None = None


class MemorySearchResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    agent_id: UUID
    project_id: UUID | None
    scope: str
    content: str
    tags: list[str]
    shared: bool
    relevance_score: float
    created_at: datetime
