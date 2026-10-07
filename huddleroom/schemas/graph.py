from __future__ import annotations
import uuid
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class GraphCreate(BaseModel):
    name: str
    version: str = "1.0"
    description: str | None = None
    definition: dict
    triggers: list
    escalation_chain: str | None = None
    loaded_from: str | None = None


class GraphUpdate(BaseModel):
    name: str | None = None
    definition: dict | None = None
    description: str | None = None
    version: str | None = None
    triggers: list | None = None
    escalation_chain: str | None = None


class GraphResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID | None
    name: str
    version: str
    description: str | None
    triggers: list
    definition: dict
    escalation_chain: str | None
    is_active: bool
    loaded_from: str | None
    created_at: datetime
    updated_at: datetime


class GraphRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    graph_id: uuid.UUID
    project_id: uuid.UUID
    linked_task_id: uuid.UUID | None
    artifact_id: uuid.UUID | None
    current_node: str
    status: str
    actor_assignments: dict
    context: dict
    escalation_step: int | None
    started_at: datetime
    last_stepped_at: datetime | None
    completed_at: datetime | None


class GraphRunStepResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    graph_run_id: uuid.UUID
    from_node: str
    to_node: str
    edge_name: str | None
    trigger_event_id: uuid.UUID | None
    trigger_reason: str | None
    actions_executed: list
    stepped_at: datetime


class GraphSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    version: str


class GraphRunSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    task_id: uuid.UUID | None
    agent_id: uuid.UUID
    status: str
    origin: str
    graph_run_id: uuid.UUID | None
    output: str | None
    error: str | None
    started_at: datetime | None
    ended_at: datetime | None
    created_at: datetime


class GraphRunTimeoutResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    graph_run_id: uuid.UUID
    node_name: str
    timeout_action: str
    expires_at: datetime
    resolved: bool
    resolved_at: datetime | None
    retry_count: int
    created_at: datetime


class GraphRunTaskSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    status: str
    parent_id: uuid.UUID | None
    graph_run_id: uuid.UUID | None


class GraphRunDetailResponse(BaseModel):
    run: GraphRunResponse
    graph: GraphSummaryResponse
    steps: list[GraphRunStepResponse]
    sessions: list[GraphRunSessionResponse]
    timeouts: list[GraphRunTimeoutResponse]
    tasks: list[GraphRunTaskSummaryResponse]


class ActorAssignRequest(BaseModel):
    kind: str  # agent | user
    id: uuid.UUID


class AdvanceRequest(BaseModel):
    to_node: str
    reason: str | None = None
