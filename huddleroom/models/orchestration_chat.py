"""Durable project/actor index; original records remain authoritative."""
import uuid
from datetime import datetime
from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from huddleroom.models.base import Base, _utcnow

CHAT_STATUS_CHECK = "status IN ('pending', 'running', 'completed', 'failed', 'interrupted_unknown')"
CHAT_PROVIDER_STATUS_CHECK = "provider_status IN ('not_started', 'pending', 'running', 'completed', 'failed', 'interrupted_unknown')"
CHAT_SOURCE_CHECK = "source_kind IN ('project_advisor', 'goal_conversation', 'action', 'clarification')"

class ProjectChatTurn(Base):
    __tablename__ = "project_chat_turns"
    __table_args__ = (
        UniqueConstraint("project_id", "actor_id", "client_request_id", name="uq_project_chat_turns_request"),
        UniqueConstraint("provider_request_id", name="uq_project_chat_turns_provider_request"),
        Index("idx_project_chat_turns_thread", "project_id", "actor_id", "created_at"),
        Index("idx_project_chat_turns_recovery", "provider_status", "deadline_at"),
        CheckConstraint(CHAT_STATUS_CHECK, name="ck_project_chat_turns_status"),
        CheckConstraint(CHAT_PROVIDER_STATUS_CHECK, name="ck_project_chat_turns_provider_status"),
        CheckConstraint(CHAT_SOURCE_CHECK, name="ck_project_chat_turns_source"),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    actor_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    client_request_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    context_goal_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    run_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    frozen_context: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    source_kind: Mapped[str] = mapped_column(String(32), nullable=False, default="clarification")
    source_record_id: Mapped[uuid.UUID] = mapped_column(nullable=False, default=uuid.uuid4)
    source_references: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", server_default="pending")
    provider_request_id: Mapped[str] = mapped_column(String(64), nullable=False, default=lambda: str(uuid.uuid4()))
    provider_status: Mapped[str] = mapped_column(String(32), nullable=False, default="not_started", server_default="not_started")
    deadline_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    accounting_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    interpretation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    action_draft: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    action_receipt: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)
