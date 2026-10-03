from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class AgentCreate(BaseModel):
    name: str
    role: str
    description: str | None = None
    provider: str
    model: str
    system_prompt: str | None = None
    adapter_type: Literal["api", "cli", "routine"] = "api"
    cli_runtime: str | None = None
    capabilities: list[str] = []
    config: dict = {}


class AgentUpdate(BaseModel):
    name: str | None = None
    role: str | None = None
    description: str | None = None
    provider: str | None = None
    model: str | None = None
    system_prompt: str | None = None
    adapter_type: Literal["api", "cli", "routine"] | None = None
    cli_runtime: str | None = None
    capabilities: list[str] | None = None
    config: dict | None = None
    is_active: bool | None = None


class AgentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    name: str
    role: str
    description: str | None
    provider: str
    model: str
    system_prompt: str | None
    adapter_type: str
    cli_runtime: str | None
    capabilities: list[str]
    config: dict
    is_active: bool
    created_at: datetime
    updated_at: datetime


class AgentTaskSummary(BaseModel):
    id: UUID
    title: str
    status: str
    priority: int


class AgentKnowledgeSummary(BaseModel):
    id: UUID
    title: str | None
    content: str
    relevance_score: float | None = None


class AgentContextResponse(BaseModel):
    agent: AgentResponse
    current_tasks: list[AgentTaskSummary] = []
    pending_meetings: list[dict] = []
    recent_knowledge: list[AgentKnowledgeSummary] = []
    active_protocol_instances: list[dict] = []
