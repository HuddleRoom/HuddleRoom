from __future__ import annotations
import uuid
from datetime import datetime
from pydantic import BaseModel, ConfigDict, computed_field


class ProtocolCreate(BaseModel):
    name: str
    version: str = "1.0"
    description: str | None = None
    definition: dict
    triggers: list
    escalation_chain: str | None = None
    loaded_from: str | None = None


class ProtocolUpdate(BaseModel):
    name: str | None = None
    definition: dict | None = None
    description: str | None = None
    version: str | None = None
    triggers: list | None = None
    escalation_chain: str | None = None


class ProtocolResponse(BaseModel):
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


class ProtocolInstanceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    protocol_id: uuid.UUID
    project_id: uuid.UUID
    linked_task_id: uuid.UUID | None
    artifact_id: uuid.UUID | None
    current_state: str
    status: str
    actor_assignments: dict
    context: dict
    escalation_step: int | None
    started_at: datetime
    last_transitioned_at: datetime | None
    completed_at: datetime | None


class ProtocolTransitionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    protocol_instance_id: uuid.UUID
    from_state: str
    to_state: str
    transition_name: str | None
    trigger_event_id: uuid.UUID | None
    trigger_reason: str | None
    actions_executed: list
    transitioned_at: datetime

    # TODO: deprecate — remove once frontend no longer reads event_type on transitions (tracked in ISSUES.md)
    @computed_field
    @property
    def event_type(self) -> str | None:
        return self.transition_name

    # TODO: deprecate — remove once frontend no longer reads created_at on transitions (tracked in ISSUES.md)
    @computed_field
    @property
    def created_at(self) -> datetime:
        return self.transitioned_at


class ProtocolSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    version: str


class ProtocolInstanceSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    task_id: uuid.UUID | None
    agent_id: uuid.UUID
    status: str
    origin: str
    protocol_instance_id: uuid.UUID | None
    output: str | None
    error: str | None
    started_at: datetime | None
    ended_at: datetime | None
    created_at: datetime


class ProtocolTimeoutResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    protocol_instance_id: uuid.UUID
    state_name: str
    timeout_action: str
    expires_at: datetime
    resolved: bool
    resolved_at: datetime | None
    retry_count: int
    created_at: datetime


class ProtocolInstanceTaskSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    status: str
    parent_id: uuid.UUID | None
    protocol_instance_id: uuid.UUID | None


class ProtocolInstanceDetailResponse(BaseModel):
    instance: ProtocolInstanceResponse
    protocol: ProtocolSummaryResponse
    transitions: list[ProtocolTransitionResponse]
    sessions: list[ProtocolInstanceSessionResponse]
    timeouts: list[ProtocolTimeoutResponse]
    tasks: list[ProtocolInstanceTaskSummaryResponse]


class ActorAssignRequest(BaseModel):
    kind: str  # agent | user
    id: uuid.UUID


class AdvanceRequest(BaseModel):
    to_state: str
    reason: str | None = None
