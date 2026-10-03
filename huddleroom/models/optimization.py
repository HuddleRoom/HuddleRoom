import uuid
from datetime import date, datetime
from sqlalchemy import CheckConstraint, Date, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column
from huddleroom.models.base import Base, TimestampMixin, _utcnow

OPTIMIZATION_VALID_STATUSES = {
    "proposed", "requires_approval", "approved", "active", "shadow", "disabled", "rejected",
}

OPTIMIZATION_VALID_TRANSITIONS = {
    "proposed": {"requires_approval", "approved", "rejected"},
    "requires_approval": {"approved", "rejected"},
    "approved": {"active"},
    "active": {"shadow", "disabled"},
    "shadow": {"active", "disabled"},
    "disabled": {"proposed"},
}


class Pattern(Base):
    __tablename__ = "patterns"
    __table_args__ = (
        Index("idx_patterns_project", "project_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    pattern_type: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    context: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class Optimization(Base, TimestampMixin):
    __tablename__ = "optimizations"
    __table_args__ = (
        Index("idx_optimizations_project_status", "project_id", "status"),
        CheckConstraint("type IN ('hook', 'rule', 'shortcut')", name="ck_optimizations_type"),
        CheckConstraint(
            "status IN ('proposed', 'requires_approval', 'approved', 'active', 'shadow', 'disabled', 'rejected')",
            name="ck_optimizations_status",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    pattern_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("patterns.id", ondelete="SET NULL"), nullable=True)
    type: Mapped[str] = mapped_column(String, nullable=False)
    generated_code: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, server_default=text("'proposed'"))
    error_rate: Mapped[float] = mapped_column(Float, nullable=False, server_default=text("0.0"))
    fire_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class CostMetric(Base):
    __tablename__ = "cost_metrics"
    __table_args__ = (
        Index("idx_cost_metrics_project_date", "project_id", "date"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    optimization_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("optimizations.id", ondelete="SET NULL"), nullable=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    llm_calls_saved: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    estimated_cost_saved_usd: Mapped[float] = mapped_column(Float, nullable=False, server_default=text("0.0"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
