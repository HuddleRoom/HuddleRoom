import uuid
from datetime import datetime
from sqlalchemy import ForeignKey, Index, JSON, String, Integer, Boolean, text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from huddleroom.models.base import Base, TimestampMixin, _utcnow_naive


class Graph(Base, TimestampMixin):
    __tablename__ = "graphs"
    __table_args__ = (
        Index("idx_graphs_active", "project_id", "is_active"),
        UniqueConstraint("project_id", "name", "version", name="uq_graphs_project_name_version"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    version: Mapped[str] = mapped_column(String, nullable=False, server_default="1.0")
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    definition: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    triggers: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    escalation_chain: Mapped[str | None] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    loaded_from: Mapped[str | None] = mapped_column(String, nullable=True)


class GraphRun(Base, TimestampMixin):
    __tablename__ = "graph_runs"
    __table_args__ = (
        Index("idx_gr_project_status", "project_id", "status"),
        Index("idx_gr_graph", "graph_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    graph_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("graphs.id"), nullable=False)
    project_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    linked_task_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    artifact_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    current_node: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, server_default="active")
    actor_assignments: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    context: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    escalation_step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    triggering_event_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    started_at: Mapped[datetime] = mapped_column(nullable=False, default=_utcnow_naive)
    last_stepped_at: Mapped[datetime | None] = mapped_column(nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)


class GraphRunStep(Base):
    __tablename__ = "graph_run_steps"
    __table_args__ = (
        Index("idx_grs_run", "graph_run_id"),
        Index("idx_grs_run_ts", "graph_run_id", "stepped_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    graph_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("graph_runs.id", ondelete="CASCADE"), nullable=False
    )
    from_node: Mapped[str] = mapped_column(String, nullable=False, server_default="")
    to_node: Mapped[str] = mapped_column(String, nullable=False)
    edge_name: Mapped[str | None] = mapped_column(String, nullable=True)
    trigger_event_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    trigger_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    actor_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    actions_executed: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    guard_context: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    stepped_at: Mapped[datetime] = mapped_column(
        nullable=False, default=_utcnow_naive
    )


class GraphRunTimeout(Base):
    __tablename__ = "graph_run_timeouts"
    __table_args__ = (
        Index("idx_grt_expires", "expires_at"),
        Index("idx_grt_run", "graph_run_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    graph_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("graph_runs.id", ondelete="CASCADE"), nullable=False
    )
    node_name: Mapped[str] = mapped_column(String, nullable=False)
    timeout_action: Mapped[str] = mapped_column(String, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(nullable=False)
    resolved: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    resolved_at: Mapped[datetime | None] = mapped_column(nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, default=_utcnow_naive
    )
