import uuid
from datetime import datetime
from sqlalchemy import ForeignKey, Index, JSON, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column
from huddleroom.models.base import Base, _utcnow_naive


class Session(Base):
    __tablename__ = "sessions"
    __table_args__ = (
        Index("idx_sessions_task_id", "task_id"),
        Index("idx_sessions_agent_id", "agent_id"),
        Index("idx_sessions_project_status", "project_id", "status"),
        Index("idx_sessions_runner_task_id", "runner_task_id"),
        Index("idx_sessions_meeting_id", "meeting_id"),
        Index("idx_sessions_origin", "origin"),
        Index(
            "uq_session_task_active",
            "task_id",
            unique=True,
            sqlite_where=text("status IN ('pending', 'running') AND task_id IS NOT NULL"),
            postgresql_where=text("status IN ('pending', 'running') AND task_id IS NOT NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True)
    agent_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agents.id", ondelete="RESTRICT"), nullable=False)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    graph_run_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)  # No database foreign key.
    meeting_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)  # No database foreign key.
    adapter_type: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, server_default="pending")
    input_context: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    output: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    runner_task_id: Mapped[str | None] = mapped_column(String, nullable=True)
    sandbox_path: Mapped[str | None] = mapped_column(String, nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)
    origin: Mapped[str] = mapped_column(String, nullable=False, server_default="manual")
    resumable: Mapped[bool] = mapped_column(nullable=False, server_default=text("0"))
    provider_session_id: Mapped[str | None] = mapped_column(String, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow_naive, nullable=False)
