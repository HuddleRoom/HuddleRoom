import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from huddleroom.models.base import Base, _utcnow
from huddleroom.models.orchestration import _column_check

# Canonical process types (Spec 6.1). Recommendation/warning handling
# (Process E) is a cross-cutting subsystem, not a process run, so it has no
# key here. No DB CHECK on process_type: the service validates against this
# tuple so future process types can extend it without a migration.
PROCESS_TYPE_VALUES = (
    "goal_definition",
    "manager_selection",
    "team_hierarchy",
    "agent_definition_review",
    "authority_interview",
    "effectiveness_review",
    "goal_closeout",
)
PROCESS_TYPES = frozenset(PROCESS_TYPE_VALUES)

# waiting_decision is reserved for Phase 9 park/resume (spec 6.3); included
# now so the status CHECK does not need a migration later.
PROCESS_RUN_STATUS_VALUES = ("running", "waiting_decision", "completed", "skipped")
PROCESS_RUN_STATUSES = frozenset(PROCESS_RUN_STATUS_VALUES)
PROCESS_RUN_STATUS_CHECK = _column_check("status", PROCESS_RUN_STATUS_VALUES)

WARNING_SEVERITY_VALUES = ("recommendation", "warning", "blocker", "hard_stop")
WARNING_SEVERITIES = frozenset(WARNING_SEVERITY_VALUES)
WARNING_SEVERITY_CHECK = _column_check("severity", WARNING_SEVERITY_VALUES)

# `expired` is a legal status but nothing expires decisions yet (no phase
# defines expiry timing); included so the CHECK needs no migration later.
AUTHORITY_DECISION_STATUS_VALUES = ("pending", "answered", "cancelled", "expired")
AUTHORITY_DECISION_STATUSES = frozenset(AUTHORITY_DECISION_STATUS_VALUES)
AUTHORITY_DECISION_STATUS_CHECK = _column_check("status", AUTHORITY_DECISION_STATUS_VALUES)

AUTHORITY_VALUES = ("human", "manager", "team_lead", "agent")
AUTHORITIES = frozenset(AUTHORITY_VALUES)
AUTHORITY_CHECK = _column_check("authority", AUTHORITY_VALUES)


class OrchestrationProcessRun(Base):
    """Deterministic baseline process execution record (Spec 6, 15.2).

    goal_id (not just run_id) because processes outlive individual runs. A
    goal has at most one current (non-superseded) run per process_type;
    reruns link the prior run via superseded_by_id. Enforced by a partial
    unique index (goal_id, process_type) WHERE superseded_by_id IS NULL,
    which prevents race conditions in start_process/skip_process while
    allowing concurrent runs to coexist with proper supersedence links.
    """

    __tablename__ = "orchestration_process_runs"
    __table_args__ = (
        Index("idx_orch_process_runs_goal_type", "goal_id", "process_type"),
        Index("idx_orch_process_runs_run", "run_id"),
        Index(
            "uq_orch_process_runs_goal_type_current",
            "goal_id",
            "process_type",
            unique=True,
            sqlite_where=text("superseded_by_id IS NULL"),
            postgresql_where=text("superseded_by_id IS NULL"),
        ),
        CheckConstraint(PROCESS_RUN_STATUS_CHECK, name="ck_orchestration_process_runs_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    goal_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("orchestration_goals.id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orchestration_runs.id", ondelete="SET NULL"), nullable=True
    )
    process_type: Mapped[str] = mapped_column(String(100), nullable=False)
    process_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="running", server_default="running")
    trigger_reason: Mapped[str] = mapped_column(Text, nullable=False)
    input_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    outputs: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    # Attribution strings follow the Phase 1 convention: "human:<user_id>",
    # "manager:<agent_id>", "orchestrator" or "orchestrator:<process>".
    skipped_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    override_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    superseded_by_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orchestration_process_runs.id", ondelete="SET NULL"), nullable=True
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class OrchestrationWarning(Base):
    """Durable warning / ignored recommendation (Spec 11, 15.3).

    Lifecycle: created active -> optionally acknowledged (stays active) ->
    resolved (active becomes False). resolved_by == "system" marks
    auto-resolution (spec 11.8); resolved_reason records why for both the
    manual and auto paths. goal_id keeps warnings alive across runs;
    source_process_run_id is what makes re-evaluation on rerun deterministic.
    """

    __tablename__ = "orchestration_warnings"
    __table_args__ = (
        Index("idx_orch_warnings_goal_active", "goal_id", "active"),
        Index("idx_orch_warnings_run", "run_id"),
        Index("idx_orch_warnings_source_process", "source_process_run_id"),
        Index("idx_orch_warnings_source_agent_review", "source_agent_review_id"),
        Index("idx_orch_warnings_related_authority_decision", "related_authority_decision_id"),
        CheckConstraint(WARNING_SEVERITY_CHECK, name="ck_orchestration_warnings_severity"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    goal_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("orchestration_goals.id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orchestration_runs.id", ondelete="SET NULL"), nullable=True
    )
    warning_type: Mapped[str] = mapped_column(String(100), nullable=False)
    severity: Mapped[str] = mapped_column(String(50), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    source_process_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orchestration_process_runs.id", ondelete="SET NULL"), nullable=True
    )
    related_gate_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orchestration_gates.id", ondelete="SET NULL"), nullable=True
    )
    related_action_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orchestration_actions.id", ondelete="SET NULL"), nullable=True
    )
    related_agent_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("agents.id", ondelete="SET NULL"), nullable=True
    )
    source_agent_review_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orchestration_agent_reviews.id", ondelete="SET NULL"), nullable=True
    )
    related_authority_decision_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(
            "orchestration_authority_decisions.id",
            name="fk_orch_warnings_related_authority_decision_id",
            ondelete="SET NULL",
            use_alter=True,
        ),
        nullable=True,
    )
    acknowledged_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="1")
    resolved_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    resolved_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class OrchestrationAuthorityDecision(Base):
    """First-class authority decision record (Spec 10.4, 10.6, 15.5).

    Lifecycle: pending -> answered | cancelled | expired. decision_key is
    the idempotency key while a decision is pending: unique per goal among
    *pending* rows only (partial index), so a retried process step never
    duplicates a pending question. It is deliberately NOT unique across all
    time — spec 6.3 requires a rerun to be able to raise a fresh decision
    under the same logical key once the prior one went terminal
    (answered/cancelled/expired); a permanent constraint would forbid that
    forever after the first answer. `reason` holds the answer reason when
    answered, or the cancellation reason when cancelled. `consequences`
    records what happens as a result of the selected option (spec 10.4),
    distinct from `context` (background for the question) and `reason`
    (why this option was picked). `authority_agent_id` names the specific
    agent instance that must answer when `authority != "human"` — the
    `authority` string alone only names a role ("manager"/"team_lead"/
    "agent"), not a specific answerer, so the service enforces
    `decided_by_agent_id == authority_agent_id` rather than accepting any
    agent (spec 4.3, 10.4).
    """

    __tablename__ = "orchestration_authority_decisions"
    __table_args__ = (
        Index(
            "uq_orch_authority_decisions_goal_key_pending",
            "goal_id",
            "decision_key",
            unique=True,
            sqlite_where=text("status = 'pending'"),
            postgresql_where=text("status = 'pending'"),
        ),
        Index("idx_orch_authority_decisions_goal_status", "goal_id", "status"),
        Index("idx_orch_authority_decisions_run", "run_id"),
        CheckConstraint(AUTHORITY_DECISION_STATUS_CHECK, name="ck_orch_authority_decisions_status"),
        CheckConstraint(AUTHORITY_CHECK, name="ck_orch_authority_decisions_authority"),
        UniqueConstraint("runtime_identity", name="uq_orch_authority_decisions_runtime_identity"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    goal_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("orchestration_goals.id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orchestration_runs.id", ondelete="SET NULL"), nullable=True
    )
    decision_key: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending", server_default="pending")
    authority: Mapped[str] = mapped_column(String(50), nullable=False)
    runtime_identity: Mapped[str | None] = mapped_column(String(64), nullable=True)
    contract_version: Mapped[str | None] = mapped_column(String(255), nullable=True)
    continuation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    continuation_action_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orchestration_actions.id", ondelete="SET NULL"), nullable=True
    )
    continuation_applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    source_process_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orchestration_process_runs.id", ondelete="SET NULL"), nullable=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[str | None] = mapped_column(Text, nullable=True)
    # List of option strings, or dicts with a required "key" field
    # (e.g. {"key": "add_verifier", "description": "..."}).
    options: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    recommendation: Mapped[str | None] = mapped_column(Text, nullable=True)
    selected_option: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    decided_by_agent_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("agents.id", ondelete="SET NULL"), nullable=True
    )
    # Specific agent instance this decision awaits when authority != "human"
    # (spec 4.3, 10.4). "authority" alone only names a role; this pins the
    # answerer so answer_decision can check identity, not just agent-ness.
    authority_agent_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("agents.id", ondelete="SET NULL"), nullable=True
    )
    consequences: Mapped[str | None] = mapped_column(Text, nullable=True)
    overrides_recommendation: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0"
    )
    created_warning_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orchestration_warnings.id", ondelete="SET NULL"), nullable=True
    )
    related_gate_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orchestration_gates.id", ondelete="SET NULL"), nullable=True
    )
    related_action_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orchestration_actions.id", ondelete="SET NULL"), nullable=True
    )
    asked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class OrchestrationAgentReview(Base):
    """Agent-definition review record (Spec 9.5, 15.4).

    definition_snapshot captures the reviewed agent's configuration at
    review time so the review stays meaningful after the agent changes or
    is deleted — agent_id is SET NULL on delete; the snapshot is the
    durable record. Warnings raised by a review are rows in
    orchestration_warnings (related_agent_id + source_process_run_id), not
    a JSON copy here — same rationale as dropping the process-run warnings
    column (Phase 2 Deviation 2). source_process_run_id links the review
    to the agent_definition_review / team_hierarchy process run that
    produced it (spec 6.1); it is absent from spec 15.4 but mirrors the
    sibling tables (Phase 3 Deviation 1).
    """

    __tablename__ = "orchestration_agent_reviews"
    __table_args__ = (
        Index("idx_orch_agent_reviews_goal_agent", "goal_id", "agent_id"),
        Index("idx_orch_agent_reviews_run", "run_id"),
        Index("idx_orch_agent_reviews_source_process", "source_process_run_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    goal_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("orchestration_goals.id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orchestration_runs.id", ondelete="SET NULL"), nullable=True
    )
    # Nullable only so agent deletion can SET NULL without erasing review
    # history; the service requires it at creation (Phase 3 Deviation 4).
    agent_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("agents.id", ondelete="SET NULL"), nullable=True
    )
    source_process_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orchestration_process_runs.id", ondelete="SET NULL"), nullable=True
    )
    review_context: Mapped[str | None] = mapped_column(Text, nullable=True)
    proposed_work_functions: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    # No default: a review without a snapshot is meaningless, so a missing
    # one must fail loudly rather than persist as {}.
    definition_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    fit_summary: Mapped[str] = mapped_column(Text, nullable=False)
    strengths: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    risks: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    recommended_changes: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    approved_for_work_functions: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )
