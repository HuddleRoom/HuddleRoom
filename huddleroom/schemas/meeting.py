from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MeetingType = Literal["decision", "review", "standup", "escalation", "adhoc"]
MeetingStatus = Literal["scheduled", "preparing", "active", "concluding", "concluded", "cancelled"]
TurnStrategy = Literal["round_robin", "agenda_driven", "moderated", "organizer_controlled"]
DeadlockStrategy = Literal["human_intervention", "majority_rules", "table_item", "escalate"]


class AgendaItemCreate(BaseModel):
    order: int
    title: str
    description: str | None = None
    question: str | None = None
    options: list[str] | None = None
    artifact_url: str | None = None
    turn_order: list[uuid.UUID] | None = None
    max_rounds: int = 3
    requires_approval: bool = False
    creates_graph: bool = False


class AgendaItemResponse(BaseModel):
    id: uuid.UUID
    meeting_id: uuid.UUID
    order: int
    title: str
    description: str | None
    question: str | None
    options: list[str] | None
    max_rounds: int
    status: str
    is_deadlocked: bool
    current_round: int
    consensus_check_count: int
    artifact_id: uuid.UUID | None
    artifact_url: str | None
    turn_order: list | None
    requires_approval: bool
    creates_graph: bool
    resolution_kind: str | None
    resolution_summary: str | None
    required_followup: str | None
    participants_heard: list[str] | None
    started_at: datetime | None
    resolved_at: datetime | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MeetingCreate(BaseModel):
    title: str
    meeting_type: MeetingType
    participant_agent_ids: list[uuid.UUID] = Field(default_factory=list)
    participant_user_ids: list[uuid.UUID] = Field(default_factory=list)
    agenda_items: list[AgendaItemCreate] = Field(default_factory=list)
    scheduled_at: datetime | None = None
    max_duration_minutes: int = 30
    veto_window_hours: int | None = None
    turn_strategy: TurnStrategy = "round_robin"
    deadlock_strategy: DeadlockStrategy = "human_intervention"
    auto_start: bool = True
    source_task_id: uuid.UUID | None = None
    source_graph_run_id: uuid.UUID | None = None
    organizer_agent_id: uuid.UUID | None = None
    organizer_user_id: uuid.UUID | None = None
    planner_agent_id: uuid.UUID | None = None
    signal_check_enabled: bool = False


class MeetingCopyRequest(BaseModel):
    title: str | None = None
    scheduled_at: datetime | None = None
    auto_start: bool | None = None


class MeetingResponse(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    title: str
    meeting_type: MeetingType
    status: MeetingStatus
    turn_strategy: TurnStrategy
    deadlock_strategy: DeadlockStrategy
    participant_agent_ids: list[str]
    participant_user_ids: list[str]
    max_duration_minutes: int
    veto_window_hours: int
    auto_start: bool
    created_by_agent_id: uuid.UUID | None
    created_by_user_id: uuid.UUID | None
    created_by_trigger: bool
    trigger_reason: str | None
    source_graph_run_id: uuid.UUID | None
    source_task_id: uuid.UUID | None
    summary: str | None
    is_partial: bool
    scheduled_at: datetime | None
    preparing_started_at: datetime | None
    active_started_at: datetime | None
    concluding_started_at: datetime | None
    concluded_at: datetime | None
    cancelled_at: datetime | None
    cancelled_reason: str | None
    organizer_agent_id: uuid.UUID | None
    organizer_user_id: uuid.UUID | None
    pending_grant_agent_id: uuid.UUID | None
    planner_agent_id: uuid.UUID | None
    planner_summary: str | None
    signal_check_enabled: bool
    created_at: datetime
    updated_at: datetime
    agenda_items: list[AgendaItemResponse] = Field(default_factory=list)
    resume_state: dict = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class TurnResponse(BaseModel):
    id: uuid.UUID
    meeting_id: uuid.UUID
    agenda_item_id: uuid.UUID | None
    turn_number: int
    round_number: int
    speaker_agent_id: uuid.UUID | None
    speaker_user_id: uuid.UUID | None
    content: str
    references: list
    is_human_turn: bool
    is_override: bool
    moderator_note: str | None
    token_count: int | None
    model_used: str | None
    latency_ms: int | None
    prompt_messages: list | None = None
    raw_response: str | None = None
    organizer_selection: dict | None = None
    reasoning_content: str | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class HumanTurnCreate(BaseModel):
    content: str
    references: list = Field(default_factory=list)


class HumanOverrideCreate(BaseModel):
    agenda_item_id: uuid.UUID
    decision: str
    reason: str


class VetoDecisionCreate(BaseModel):
    decision_id: uuid.UUID
    reason: str


class FinalReviewResponse(BaseModel):
    reviewer_kind: Literal["organizer_agent", "organizer_user", "orchestrator"]
    reviewer_id: uuid.UUID | None
    decisions_made: bool
    decisions_clear: bool
    suggested_action_items: list[str]


class FinalReviewSubmit(BaseModel):
    decisions_made: bool
    decisions_clear: bool
    action_items_needed: bool
    action_items: list[str] = Field(default_factory=list)


class DecisionResponse(BaseModel):
    id: uuid.UUID
    meeting_id: uuid.UUID
    agenda_item_id: uuid.UUID
    title: str
    question: str | None
    chosen_option: str
    rationale: str
    alternatives_rejected: list
    participants_agreed: list
    dissent: list
    decided_by: str
    confidence: float | None
    is_partial: bool
    is_vetoed: bool
    veto_reason: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ActionItemResponse(BaseModel):
    id: uuid.UUID
    meeting_id: uuid.UUID
    depends_on_decision_id: uuid.UUID | None
    description: str
    assignee_agent_id: uuid.UUID | None
    assignee_user_id: uuid.UUID | None
    priority: int
    deadline_days: int | None
    deadline_at: datetime | None
    status: str
    task_id: uuid.UUID | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AgendaItemUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    question: str | None = None
    options: list[str] | None = None
    max_rounds: int | None = None
    order: int | None = None


class ActionItemUpdate(BaseModel):
    description: str | None = None
    assignee_agent_id: uuid.UUID | None = None
    assignee_user_id: uuid.UUID | None = None
    priority: int | None = None
    deadline_days: int | None = None
    status: str | None = None


class GrantTurnRequest(BaseModel):
    participant_agent_id: uuid.UUID


class SignalCreate(BaseModel):
    agent_id: uuid.UUID
    signal_type: str = "want_to_speak"
    message: str | None = None


class SignalResponse(BaseModel):
    id: uuid.UUID
    meeting_id: uuid.UUID
    agent_id: uuid.UUID
    signal_type: str
    message: str | None
    created_at: datetime
    acknowledged_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class AdvanceAgendaItemRequest(BaseModel):
    resolution: Literal["resolved", "unresolved", "tabled"] = "resolved"
    decided_by: str = "organizer"


class AgentMeetingRequestCreate(BaseModel):
    project_id: uuid.UUID
    title: str
    reason: str
    meeting_type: str = "decision"
    suggested_participant_agent_ids: list[uuid.UUID] = Field(default_factory=list)


class AgentMeetingRequestResponse(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    requesting_agent_id: uuid.UUID
    title: str
    reason: str
    meeting_type: str
    suggested_participant_agent_ids: list
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class VetoCreate(BaseModel):
    reason: str
