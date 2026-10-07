import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Index,
    Integer, JSON, String, text,
)
from sqlalchemy.orm import Mapped, mapped_column

from huddleroom.models.base import Base, TimestampMixin, _utcnow, _utcnow_naive


class Meeting(Base, TimestampMixin):
    __tablename__ = "meetings"
    __table_args__ = (
        Index("idx_meetings_project_status", "project_id", "status"),
        Index("idx_meetings_status_scheduled", "status", "scheduled_at"),
        Index("idx_meetings_source_task", "source_task_id"),
        Index("idx_meetings_project_created_id", "project_id", sa.text("created_at DESC"), sa.text("id DESC")),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    meeting_type: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, server_default="scheduled")
    turn_strategy: Mapped[str] = mapped_column(String, nullable=False, server_default="round_robin")
    deadlock_strategy: Mapped[str] = mapped_column(String, nullable=False, server_default="human_intervention")
    participant_agent_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    participant_user_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    max_duration_minutes: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("30"))
    veto_window_hours: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("24"))
    auto_start: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    created_by_agent_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    created_by_user_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    created_by_trigger: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    trigger_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    source_graph_run_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    source_task_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True)
    timeout_task_id: Mapped[str | None] = mapped_column(String, nullable=True)
    signal_check_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    organizer_agent_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    organizer_user_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    pending_grant_agent_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    planner_agent_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    planner_summary: Mapped[str | None] = mapped_column(String, nullable=True)
    participant_contexts: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    resume_state: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict, server_default=text("'{}'"))
    summary: Mapped[str | None] = mapped_column(String, nullable=True)
    is_partial: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    scheduled_at: Mapped[datetime | None] = mapped_column(nullable=True)
    preparing_started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    active_started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    concluding_started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    concluded_at: Mapped[datetime | None] = mapped_column(nullable=True)
    cancelled_at: Mapped[datetime | None] = mapped_column(nullable=True)
    cancelled_reason: Mapped[str | None] = mapped_column(String, nullable=True)


class MeetingAgendaItem(Base):
    __tablename__ = "meeting_agenda_items"
    __table_args__ = (
        Index("idx_mai_meeting_order", "meeting_id", "order"),
        Index("idx_mai_meeting_status", "meeting_id", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    meeting_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("meetings.id"), nullable=False)
    order: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    question: Mapped[str | None] = mapped_column(String, nullable=True)
    options: Mapped[list | None] = mapped_column(JSON, nullable=True)
    artifact_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    artifact_url: Mapped[str | None] = mapped_column(String, nullable=True)
    turn_order: Mapped[list | None] = mapped_column(JSON, nullable=True)
    max_rounds: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("3"))
    status: Mapped[str] = mapped_column(String, nullable=False, server_default="pending")
    is_deadlocked: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    current_round: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    consensus_check_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    requires_approval: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    creates_graph: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    resolution_kind: Mapped[str | None] = mapped_column(String, nullable=True)
    resolution_summary: Mapped[str | None] = mapped_column(String, nullable=True)
    required_followup: Mapped[str | None] = mapped_column(String, nullable=True)
    participants_heard: Mapped[list | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False, default=_utcnow_naive)


class MeetingTurn(Base):
    __tablename__ = "meeting_turns"
    __table_args__ = (
        Index("idx_mt_meeting_turn", "meeting_id", "turn_number"),
        Index("idx_mt_meeting_item_round", "meeting_id", "agenda_item_id", "round_number"),
        Index("idx_mt_speaker_agent", "speaker_agent_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    meeting_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("meetings.id"), nullable=False)
    agenda_item_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("meeting_agenda_items.id"), nullable=True
    )
    turn_number: Mapped[int] = mapped_column(Integer, nullable=False)
    round_number: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    speaker_agent_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    speaker_user_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    content: Mapped[str] = mapped_column(String, nullable=False)
    references: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    is_human_turn: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    is_override: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    moderator_note: Mapped[str | None] = mapped_column(String, nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_used: Mapped[str | None] = mapped_column(String, nullable=True)
    provider_used: Mapped[str | None] = mapped_column(String, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prompt_messages: Mapped[list | None] = mapped_column(JSON, nullable=True)
    raw_response: Mapped[str | None] = mapped_column(String, nullable=True)
    organizer_selection: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    reasoning_content: Mapped[str | None] = mapped_column(String, nullable=True)
    cli_session_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False, default=_utcnow_naive)


class MeetingDecision(Base):
    __tablename__ = "meeting_decisions"
    __table_args__ = (
        Index("idx_md_meeting", "meeting_id"),
        Index("idx_md_agenda_item", "agenda_item_id"),
        Index("idx_md_vetoed", "is_vetoed"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    meeting_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("meetings.id"), nullable=False)
    agenda_item_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("meeting_agenda_items.id"), nullable=False
    )
    title: Mapped[str] = mapped_column(String, nullable=False)
    question: Mapped[str | None] = mapped_column(String, nullable=True)
    chosen_option: Mapped[str] = mapped_column(String, nullable=False)
    rationale: Mapped[str] = mapped_column(String, nullable=False)
    alternatives_rejected: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    participants_agreed: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    dissent: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    decided_by: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_partial: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    is_vetoed: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    veto_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    vetoed_by_user_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    vetoed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    knowledge_item_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False, default=_utcnow_naive)


class MeetingActionItem(Base):
    __tablename__ = "meeting_action_items"
    __table_args__ = (
        Index("idx_mact_meeting", "meeting_id"),
        Index("idx_mact_assignee_status", "assignee_agent_id", "status"),
        Index("idx_mact_task", "task_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    meeting_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("meetings.id"), nullable=False)
    depends_on_decision_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("meeting_decisions.id"), nullable=True
    )
    description: Mapped[str] = mapped_column(String, nullable=False)
    assignee_agent_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    assignee_user_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("70"))
    deadline_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    deadline_at: Mapped[datetime | None] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, server_default="open")
    task_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True)
    creates_graph: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    graph_run_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    is_partial: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    created_at: Mapped[datetime] = mapped_column(nullable=False, default=_utcnow_naive)


class MeetingEvent(Base):
    __tablename__ = "meeting_events"
    __table_args__ = (
        Index("idx_mev_meeting", "meeting_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    meeting_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("meetings.id"), nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    actor_agent_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False, default=_utcnow_naive)


class MeetingParticipantSignal(Base):
    __tablename__ = "meeting_participant_signals"
    __table_args__ = (
        Index("idx_mps_meeting_ack", "meeting_id", "acknowledged_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    meeting_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("meetings.id"), nullable=False)
    agent_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    signal_type: Mapped[str] = mapped_column(String, nullable=False)
    message: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    acknowledged_at: Mapped[datetime | None] = mapped_column(nullable=True)


class MeetingRequest(Base):
    __tablename__ = "meeting_requests"
    __table_args__ = (
        Index("idx_mreq_project_status", "project_id", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    requesting_agent_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    reason: Mapped[str] = mapped_column(String, nullable=False)
    meeting_type: Mapped[str] = mapped_column(String, nullable=False, server_default="decision")
    suggested_participant_agent_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    status: Mapped[str] = mapped_column(String, nullable=False, server_default="pending_approval")
    approved_by_user_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(nullable=True)
    created_meeting_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("meetings.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
