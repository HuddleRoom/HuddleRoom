import uuid
from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from huddleroom.models.base import Base, _utcnow

# Canonical section keys from Spec 5.2. Keys are not restricted to this
# tuple — processes may write custom sections — but known names live here so
# later phases (preface builder, baseline processes) reference one constant.
CANONICAL_SECTION_KEYS = (
    "introduction",
    "goal_definition",
    "success_criteria",
    "constraints_tradeoffs",
    "manager_authority",
    "team_hierarchy",
    "agent_definition_review",
    "human_decisions",
    "manager_decisions",
    "active_warnings",
    "ignored_recommendations",
    "open_questions",
    "recovery_history",
    "operating_assumptions",
    "completion_rationale",
    "lessons_learned",
)
MEMORY_FACT_STATUS_VALUES = ("unverified", "accepted", "superseded")
MEMORY_FACT_STATUS_CHECK = "fact_status IN ('unverified', 'accepted', 'superseded')"


class OrchestrationMemorySection(Base):
    """Orchestrator-only memory (Spec section 5).

    Agents never read or write this table. Code may copy selected excerpts
    into delegation contracts in later phases.
    """

    __tablename__ = "orchestration_memory_sections"
    __table_args__ = (
        UniqueConstraint("goal_id", "section_key", name="uq_orch_memory_sections_goal_key"),
        UniqueConstraint("project_id", "goal_id", "section_key", name="uq_orch_memory_sections_project_goal_key"),
        Index("idx_orch_memory_sections_goal_always", "goal_id", "always_load"),
        Index("idx_orch_memory_sections_project", "project_id"),
        CheckConstraint(MEMORY_FACT_STATUS_CHECK, name="ck_orch_memory_sections_fact_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    goal_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("orchestration_goals.id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orchestration_runs.id", ondelete="SET NULL"), nullable=True
    )
    section_key: Mapped[str] = mapped_column(String(100), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    section_type: Mapped[str] = mapped_column(String(50), nullable=False, default="text", server_default="text")
    body: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    always_load: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="0")
    toc_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    # Who last wrote the section: "orchestrator", "orchestrator:<process>",
    # "human:<user_id>", or "manager:<agent_id>". Full revision history is
    # deferred per spec section 21.
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    created_from_event_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("event_log.id", ondelete="SET NULL"), nullable=True
    )
    updated_from_event_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("event_log.id", ondelete="SET NULL"), nullable=True
    )
    fact_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="unverified", server_default="unverified"
    )
    provenance: Mapped[dict] = mapped_column(
        JSON, nullable=False, default=dict, server_default=text("'{}'")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )
