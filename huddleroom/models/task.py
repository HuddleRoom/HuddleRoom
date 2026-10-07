import uuid
from datetime import datetime
from sqlalchemy import ForeignKey, Index, JSON, Integer, String, text
from sqlalchemy.orm import Mapped, mapped_column
from huddleroom.models.base import Base, TimestampMixin


class Task(Base, TimestampMixin):
    __tablename__ = "tasks"
    __table_args__ = (
        Index("idx_tasks_project_status", "project_id", "status"),
        Index("idx_tasks_assigned_to", "assigned_to"),
        Index("idx_tasks_parent_id", "parent_id"),
        Index("idx_tasks_graph_run", "graph_run_id"),
    )

    # State machine transitions
    VALID_TRANSITIONS: dict[str, set[str]] = {
        "backlog": {"ready", "cancelled"},
        "ready": {"in_progress", "cancelled"},
        "in_progress": {"blocked", "done", "cancelled", "failed"},
        "blocked": {"in_progress", "cancelled"},
        "done": set(),
        "cancelled": set(),
        "failed": {"in_progress", "cancelled"},
    }

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    parent_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True)
    graph_run_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)  # No database foreign key.
    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, server_default="backlog")
    priority: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("50"))
    assigned_to: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("agents.id", ondelete="SET NULL"), nullable=True)
    created_by_agent: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("agents.id", ondelete="SET NULL"), nullable=True)
    created_by_user: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    adapter_type_override: Mapped[str | None] = mapped_column(String, nullable=True)
    depends_on: Mapped[list[uuid.UUID] | None] = mapped_column(JSON, nullable=True)
    trigger: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)
    due_at: Mapped[datetime | None] = mapped_column(nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    created_by_meeting_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
