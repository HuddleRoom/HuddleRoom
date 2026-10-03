import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column
from huddleroom.models.base import Base, TimestampMixin, _utcnow

HOOK_VALID_STATUSES = {"proposed", "active", "shadow", "disabled"}

HOOK_VALID_TRANSITIONS = {
    "proposed": {"active", "disabled"},
    "active": {"shadow", "disabled"},
    "shadow": {"active", "disabled"},
    "disabled": {"proposed"},
}


class Hook(Base, TimestampMixin):
    __tablename__ = "hooks"
    __table_args__ = (
        Index("idx_hooks_project_status", "project_id", "status"),
        CheckConstraint("status IN ('proposed', 'active', 'shadow', 'disabled')", name="ck_hooks_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    code: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, server_default=text("'proposed'"))
    trigger_event: Mapped[str] = mapped_column(String, nullable=False)
    execution_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    error_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )
