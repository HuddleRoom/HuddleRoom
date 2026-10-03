import uuid
from datetime import datetime

from sqlalchemy import BigInteger, Boolean, CheckConstraint, DateTime, ForeignKey, Index, JSON, String, Text, UniqueConstraint, false, inspect, text, true
from sqlalchemy.orm import Mapped, mapped_column, validates

from huddleroom.models.base import Base, _utcnow


def _column_check(column_name: str, values: tuple[str, ...]) -> str:
    return f"{column_name} IN ({', '.join(repr(value) for value in values)})"


def _status_check(statuses: tuple[str, ...]) -> str:
    return _column_check("status", statuses)


def _validator_status_check(statuses: tuple[str, ...]) -> str:
    return _column_check("validator_status", statuses)


GOAL_STATUS_VALUES = ("active", "blocked", "paused", "completed", "cancelled")
RUN_STATUS_VALUES = ("running", "blocked", "paused", "completed", "cancelled")
ACTIVE_RUN_STATUS_VALUES = ("running", "blocked", "paused")
DECISION_VALIDATOR_STATUS_VALUES = ("pending", "accepted", "rejected")
ACTION_STATUS_VALUES = ("reserved", "completed", "failed")
GATE_STATUS_VALUES = ("open", "accepted", "failed")
EVIDENCE_VERDICT_VALUES = ("candidate", "accepted", "rejected")
AGENT_SUGGESTION_STATUS_VALUES = ("open", "accepted", "dismissed")
GOAL_STATUSES = frozenset(GOAL_STATUS_VALUES)
RUN_STATUSES = frozenset(RUN_STATUS_VALUES)
ACTIVE_RUN_STATUSES = frozenset(ACTIVE_RUN_STATUS_VALUES)
DECISION_VALIDATOR_STATUSES = frozenset(DECISION_VALIDATOR_STATUS_VALUES)
ACTION_STATUSES = frozenset(ACTION_STATUS_VALUES)
GATE_STATUSES = frozenset(GATE_STATUS_VALUES)
EVIDENCE_VERDICTS = frozenset(EVIDENCE_VERDICT_VALUES)
AGENT_SUGGESTION_STATUSES = frozenset(AGENT_SUGGESTION_STATUS_VALUES)
GOAL_STATUS_CHECK = _status_check(GOAL_STATUS_VALUES)
RUN_STATUS_CHECK = _status_check(RUN_STATUS_VALUES)
ACTIVE_RUN_STATUS_CHECK = _status_check(ACTIVE_RUN_STATUS_VALUES)
DECISION_VALIDATOR_STATUS_CHECK = _validator_status_check(DECISION_VALIDATOR_STATUS_VALUES)
ACTION_STATUS_CHECK = _status_check(ACTION_STATUS_VALUES)
GATE_STATUS_CHECK = _status_check(GATE_STATUS_VALUES)
EVIDENCE_VERDICT_CHECK = _column_check("verdict", EVIDENCE_VERDICT_VALUES)
AGENT_SUGGESTION_STATUS_CHECK = _status_check(AGENT_SUGGESTION_STATUS_VALUES)

GOAL_WEIGHT_VALUES = ("trivial", "standard", "substantial")
GOAL_WEIGHTS = frozenset(GOAL_WEIGHT_VALUES)
GOAL_WEIGHT_CHECK = _column_check("weight", GOAL_WEIGHT_VALUES)

AUTHORITY_MODEL_VALUES = ("agent_manager", "human_manager", "no_manager")
AUTHORITY_MODELS = frozenset(AUTHORITY_MODEL_VALUES)
AUTHORITY_MODEL_CHECK = (
    "authority_model IS NULL OR " + _column_check("authority_model", AUTHORITY_MODEL_VALUES)
)

GOAL_TYPE_VALUES = ("outcome", "roadmap", "continuous")
GOAL_TYPES = frozenset(GOAL_TYPE_VALUES)
GOAL_TYPE_CHECK = _column_check("goal_type", GOAL_TYPE_VALUES)

ROADMAP_UNIT_TYPE_VALUES = ("task", "goal")
ROADMAP_UNIT_TYPE_CHECK = _column_check("unit_type", ROADMAP_UNIT_TYPE_VALUES)
BUDGET_RESERVATION_STATUS_VALUES = ("active", "settled")
BUDGET_RESERVATION_STATUS_CHECK = _column_check("status", BUDGET_RESERVATION_STATUS_VALUES)
BUDGET_RESERVATION_LINEAGE_CHECK = (
    "(roadmap_item_id IS NOT NULL AND continuous_origin_key IS NULL AND discovery_run_id IS NULL "
    "AND child_goal_id IS NOT NULL) OR "
    "(roadmap_item_id IS NULL AND continuous_origin_key IS NOT NULL AND discovery_run_id IS NULL "
    "AND child_goal_id IS NOT NULL) OR "
    "(roadmap_item_id IS NULL AND continuous_origin_key IS NULL AND discovery_run_id IS NOT NULL "
    "AND child_goal_id IS NULL)"
)
BUDGET_RESERVATION_SETTLEMENT_CHECK = (
    "(status = 'active' AND settled_at IS NULL AND settlement_reason IS NULL) OR "
    "(status = 'settled' AND settled_at IS NOT NULL AND "
    "settlement_reason IN ('completed', 'cancelled', 'needs_attention'))"
)

RUN_PHASE_VALUES = ("baseline", "ready", "waiting_activation", "authorized", "completed")
RUN_PHASES = frozenset(RUN_PHASE_VALUES)
RUN_PHASE_CHECK = _column_check("phase", RUN_PHASE_VALUES)
WAIT_STATUS_VALUES = ("open", "cleared")
WAIT_STATUS_CHECK = _status_check(WAIT_STATUS_VALUES)


class OrchestrationGoal(Base):
    __tablename__ = "orchestration_goals"
    __table_args__ = (
        Index("idx_orch_goals_project_status", "project_id", "status"),
        Index("idx_orch_goals_project_created", "project_id", "created_at"),
        CheckConstraint(GOAL_STATUS_CHECK, name="ck_orchestration_goals_status"),
        CheckConstraint(GOAL_WEIGHT_CHECK, name="ck_orchestration_goals_weight"),
        CheckConstraint(AUTHORITY_MODEL_CHECK, name="ck_orchestration_goals_authority_model"),
        CheckConstraint(
            "manager_agent_id IS NULL OR manager_user_id IS NULL",
            name="ck_orchestration_goals_single_manager",
        ),
        CheckConstraint(
            "authority_model IS NULL OR authority_model != 'no_manager' OR (manager_agent_id IS NULL AND manager_user_id IS NULL)",
            name="ck_orchestration_goals_no_manager_constraint",
        ),
        CheckConstraint(GOAL_TYPE_CHECK, name="ck_orchestration_goals_goal_type"),
        UniqueConstraint("supersedes_goal_id", name="uq_orch_goals_supersedes_goal_id"),
        CheckConstraint(
            "(parent_goal_id IS NULL AND roadmap_version_id IS NULL AND roadmap_item_key IS NULL "
            "AND continuous_origin_key IS NULL AND parent_contract_snapshot IS NULL AND goal_delta IS NULL) OR "
            "(parent_goal_id IS NOT NULL AND roadmap_version_id IS NOT NULL AND roadmap_item_key IS NOT NULL "
            "AND continuous_origin_key IS NULL AND parent_contract_snapshot IS NOT NULL AND goal_delta IS NOT NULL) OR "
            "(parent_goal_id IS NOT NULL AND roadmap_version_id IS NULL AND roadmap_item_key IS NULL "
            "AND continuous_origin_key IS NOT NULL AND parent_contract_snapshot IS NOT NULL "
            "AND goal_delta IS NOT NULL)",
            name="ck_orch_goals_child_lineage_complete",
        ),
        CheckConstraint("parent_goal_id IS NULL OR parent_goal_id != id", name="ck_orch_goals_not_own_parent"),
        UniqueConstraint("parent_goal_id", "roadmap_item_key", name="uq_orch_goals_parent_item_key"),
        UniqueConstraint(
            "parent_goal_id", "continuous_origin_key", name="uq_orch_goals_parent_origin_key"
        ),
        Index("idx_orch_goals_parent_status", "parent_goal_id", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    original_request: Mapped[str] = mapped_column(Text, nullable=False)
    success_criteria: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    constraints: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    budget: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    orchestrator_context: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String, nullable=False, server_default="active")
    # Deterministic weight classification (Spec 6.2). Phase 5 sets it;
    # 'standard' until then. weight_overridden_by records who forced a
    # weight ("human:<user_id>" attribution style); forcing lighter than
    # classified creates a warning (Phase 5).
    weight: Mapped[str] = mapped_column(String(50), nullable=False, default="standard", server_default="standard")
    weight_overridden_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    explicit_multi_work_function: Mapped[bool] = mapped_column(
        nullable=False, default=False, server_default=false()
    )
    goal_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default="outcome", server_default="outcome"
    )
    supersedes_goal_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orchestration_goals.id", ondelete="SET NULL"), nullable=True
    )
    parent_goal_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orchestration_goals.id", ondelete="RESTRICT"), nullable=True
    )
    roadmap_version_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(
            "orchestration_roadmap_versions.id",
            name="fk_orch_goals_roadmap_version_id",
            ondelete="RESTRICT",
            use_alter=True,
        ),
        nullable=True,
    )
    roadmap_item_key: Mapped[str | None] = mapped_column(String(80), nullable=True)
    parent_contract_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    goal_delta: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    continuous_policy: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    continuous_state: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    continuous_origin_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Phase 6 (Spec 8): selected manager / main POC and authority model.
    # At most one of manager_agent_id / manager_user_id is set; authority_model
    # stays NULL until Baseline Process B first completes for this goal.
    manager_agent_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("agents.id", ondelete="SET NULL"), nullable=True
    )
    manager_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    authority_model: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    @validates("original_request")
    def validate_original_request(self, key: str, value: str) -> str:
        if inspect(self).persistent and value != self.original_request:
            raise ValueError("original_request is immutable")
        return value

    @validates("objective")
    def preserve_initial_objective(self, key: str, value: str) -> str:
        if getattr(self, "original_request", None) is None:
            self.original_request = value
        return value

    @validates("parent_contract_snapshot", "goal_delta", "continuous_origin_key")
    def preserve_child_lineage(self, key: str, value: dict | str | None) -> dict | str | None:
        if inspect(self).persistent:
            raise ValueError("child lineage is immutable")
        return value


class OrchestrationRun(Base):
    __tablename__ = "orchestration_runs"
    __table_args__ = (
        Index("idx_orch_runs_goal_status", "goal_id", "status"),
        Index(
            "uq_orch_runs_one_active_per_goal",
            "goal_id",
            unique=True,
            sqlite_where=text(ACTIVE_RUN_STATUS_CHECK),
            postgresql_where=text(ACTIVE_RUN_STATUS_CHECK),
        ),
        CheckConstraint(RUN_STATUS_CHECK, name="ck_orchestration_runs_status"),
        CheckConstraint(RUN_PHASE_CHECK, name="ck_orchestration_runs_phase"),
        UniqueConstraint("goal_id", "cycle_key", name="uq_orch_runs_goal_cycle_key"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    goal_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_goals.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, server_default="running")
    event_cursor: Mapped[int | None] = mapped_column(BigInteger(), nullable=True)
    plan_state: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    active_blockers: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    budget_state: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    retry_state: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    supervision_state: Mapped[dict] = mapped_column(
        JSON, nullable=False, default=dict, server_default=text("'{}'")
    )
    phase: Mapped[str] = mapped_column(
        String(50), nullable=False, default="baseline", server_default="baseline"
    )
    baseline_authorized: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=text("true")
    )
    cycle_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class OrchestrationRoadmapVersion(Base):
    __tablename__ = "orchestration_roadmap_versions"
    __table_args__ = (
        UniqueConstraint("goal_id", "version", name="uq_orch_roadmap_versions_goal_version"),
        UniqueConstraint("goal_id", "fingerprint", name="uq_orch_roadmap_versions_goal_fingerprint"),
        CheckConstraint("version > 0", name="ck_orch_roadmap_versions_positive_version"),
        Index("idx_orch_roadmap_versions_goal_created", "goal_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    goal_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_goals.id", ondelete="CASCADE"), nullable=False)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_runs.id", ondelete="CASCADE"), nullable=False)
    version: Mapped[int] = mapped_column(nullable=False)
    plan_artifact_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("artifacts.id", ondelete="RESTRICT"), nullable=False)
    snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    approval_reference: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    @validates("snapshot", "fingerprint", "approval_reference")
    def preserve_roadmap_lineage(self, key: str, value: dict | str) -> dict | str:
        if inspect(self).persistent:
            raise ValueError("roadmap lineage is immutable")
        return value


class OrchestrationRoadmapItem(Base):
    __tablename__ = "orchestration_roadmap_items"
    __table_args__ = (
        UniqueConstraint("goal_id", "item_key", name="uq_orch_roadmap_items_goal_key"),
        UniqueConstraint("task_id", name="uq_orch_roadmap_items_task"),
        UniqueConstraint("child_goal_id", name="uq_orch_roadmap_items_child_goal"),
        CheckConstraint(ROADMAP_UNIT_TYPE_CHECK, name="ck_orch_roadmap_items_unit_type"),
        CheckConstraint(
            "(unit_type = 'task' AND task_id IS NOT NULL AND child_goal_id IS NULL) OR "
            "(unit_type = 'goal' AND child_goal_id IS NOT NULL AND task_id IS NULL)",
            name="ck_orch_roadmap_items_target_matches_type",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    goal_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_goals.id", ondelete="CASCADE"), nullable=False)
    first_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_roadmap_versions.id", ondelete="RESTRICT"), nullable=False)
    item_key: Mapped[str] = mapped_column(String(80), nullable=False)
    unit_type: Mapped[str] = mapped_column(String(20), nullable=False)
    item_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    task_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tasks.id", ondelete="RESTRICT"), nullable=True)
    child_goal_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("orchestration_goals.id", ondelete="RESTRICT"), nullable=True)
    gate_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_gates.id", ondelete="RESTRICT"), nullable=False)
    released_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    @validates("item_snapshot")
    def preserve_roadmap_lineage(self, key: str, value: dict) -> dict:
        if inspect(self).persistent:
            raise ValueError("roadmap lineage is immutable")
        return value


class OrchestrationContinuousCandidate(Base):
    __tablename__ = "orchestration_continuous_candidates"
    __table_args__ = (
        UniqueConstraint("parent_goal_id", "origin_key", name="uq_orch_continuous_candidate_parent_origin"),
        UniqueConstraint("source_run_id", "position", name="uq_orch_continuous_candidate_run_position"),
        UniqueConstraint("child_goal_id", name="uq_orch_continuous_candidate_child"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    parent_goal_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("orchestration_goals.id", ondelete="RESTRICT"), nullable=False
    )
    source_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("orchestration_runs.id", ondelete="RESTRICT"), nullable=False
    )
    origin_key: Mapped[str] = mapped_column(String(255), nullable=False)
    position: Mapped[int] = mapped_column(nullable=False)
    snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    child_goal_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orchestration_goals.id", ondelete="RESTRICT"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    @validates("parent_goal_id", "source_run_id", "origin_key", "position", "snapshot")
    def preserve_candidate_identity(self, key: str, value: object) -> object:
        if inspect(self).persistent:
            raise ValueError("Continuous candidate identity is immutable")
        return value


class OrchestrationBudgetReservation(Base):
    __tablename__ = "orchestration_budget_reservations"
    __table_args__ = (
        UniqueConstraint("roadmap_item_id", name="uq_orch_budget_reservations_item"),
        UniqueConstraint("child_goal_id", name="uq_orch_budget_reservations_child"),
        UniqueConstraint("parent_goal_id", "continuous_origin_key", name="uq_orch_budget_parent_origin"),
        UniqueConstraint("discovery_run_id", name="uq_orch_budget_discovery_run"),
        CheckConstraint(BUDGET_RESERVATION_LINEAGE_CHECK, name="ck_orch_budget_reservation_lineage"),
        CheckConstraint(BUDGET_RESERVATION_STATUS_CHECK, name="ck_orch_budget_reservations_status"),
        CheckConstraint(BUDGET_RESERVATION_SETTLEMENT_CHECK, name="ck_orch_budget_reservations_settlement"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    parent_goal_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_goals.id", ondelete="RESTRICT"), nullable=False)
    roadmap_item_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orchestration_roadmap_items.id", ondelete="RESTRICT"), nullable=True
    )
    continuous_origin_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    discovery_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orchestration_runs.id", ondelete="RESTRICT"), nullable=True
    )
    child_goal_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orchestration_goals.id", ondelete="RESTRICT"), nullable=True
    )
    allocation: Mapped[dict] = mapped_column(JSON, nullable=False)
    settled_spend: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    measurement_complete: Mapped[bool] = mapped_column(nullable=False, default=True, server_default=true())
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active", server_default="active")
    settlement_reason: Mapped[str | None] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class OrchestrationDecision(Base):
    __tablename__ = "orchestration_decisions"
    __table_args__ = (
        Index("idx_orch_decisions_run_created", "run_id", "created_at"),
        Index("idx_orch_decisions_validator_status", "validator_status"),
        CheckConstraint(
            DECISION_VALIDATOR_STATUS_CHECK,
            name="ck_orchestration_decisions_validator_status",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_runs.id", ondelete="CASCADE"), nullable=False)
    decision_type: Mapped[str] = mapped_column(String(100), nullable=False)
    input_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    llm_output: Mapped[dict | list | str | None] = mapped_column(JSON, nullable=True)
    parsed_decision: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    validator_status: Mapped[str] = mapped_column(String(50), nullable=False, server_default="pending")
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class OrchestrationAction(Base):
    __tablename__ = "orchestration_actions"
    __table_args__ = (
        UniqueConstraint("run_id", "idempotency_key", name="uq_orch_actions_run_idempotency_key"),
        Index("idx_orch_actions_run_status", "run_id", "status"),
        CheckConstraint(ACTION_STATUS_CHECK, name="ck_orchestration_actions_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_runs.id", ondelete="CASCADE"), nullable=False)
    decision_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("orchestration_decisions.id", ondelete="SET NULL"), nullable=True
    )
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    action_type: Mapped[str] = mapped_column(String(100), nullable=False)
    request: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    dispatch_contract: Mapped[dict] = mapped_column(
        JSON, nullable=False, default=dict, server_default=text("'{}'")
    )
    budget_ledger: Mapped[dict] = mapped_column(
        JSON, nullable=False, default=dict, server_default=text("'{}'")
    )
    target_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    target_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, server_default="reserved")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class OrchestrationWait(Base):
    __tablename__ = "orchestration_waits"
    __table_args__ = (
        Index(
            "uq_orch_waits_run_key_open",
            "run_id",
            "wait_key",
            unique=True,
            sqlite_where=text("status = 'open'"),
            postgresql_where=text("status = 'open'"),
        ),
        Index("idx_orch_waits_due", "status", "due_recheck_at"),
        CheckConstraint(WAIT_STATUS_CHECK, name="ck_orchestration_waits_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("orchestration_runs.id", ondelete="CASCADE"), nullable=False
    )
    wait_key: Mapped[str] = mapped_column(String(255), nullable=False)
    owner: Mapped[dict] = mapped_column(JSON, nullable=False)
    awaited_event: Mapped[dict] = mapped_column(JSON, nullable=False)
    due_recheck_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    fallback: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="open")
    cleared_by_event_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("event_log.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    cleared_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class OrchestrationSchedulerState(Base):
    __tablename__ = "orchestration_scheduler_state"
    __table_args__ = (
        CheckConstraint("name = 'supervision'", name="ck_orch_scheduler_state_name"),
        UniqueConstraint("pass_key", name="uq_orch_scheduler_state_pass_key"),
    )

    name: Mapped[str] = mapped_column(
        String(50), primary_key=True, default="supervision", server_default="supervision"
    )
    cursor_goal_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    pass_key: Mapped[str | None] = mapped_column(String(100), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class OrchestrationGate(Base):
    __tablename__ = "orchestration_gates"
    __table_args__ = (
        Index("idx_orch_gates_run_status", "run_id", "status"),
        Index("idx_orch_gates_run_criterion", "run_id", "success_criterion_key"),
        CheckConstraint(GATE_STATUS_CHECK, name="ck_orchestration_gates_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_runs.id", ondelete="CASCADE"), nullable=False)
    success_criterion_key: Mapped[str] = mapped_column(String(100), nullable=False)
    gate_type: Mapped[str] = mapped_column(String(100), nullable=False)
    required_evidence: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(50), nullable=False, server_default="open")
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    # ponytail: multiple gates per (run_id, success_criterion_key) allowed — gates are not re-openable today but uniqueness is not enforced to preserve replay flexibility
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )
    accepted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    @validates("required_evidence")
    def validate_required_evidence(self, key: str, value: dict) -> dict:
        if not isinstance(value, dict):
            raise ValueError("required_evidence must be a dict")
        if "min_count" in value and not isinstance(value["min_count"], int):
            raise ValueError("required_evidence.min_count must be int")
        if "required_source_types" in value and not isinstance(value["required_source_types"], list):
            raise ValueError("required_evidence.required_source_types must be list")
        return value


class OrchestrationEvidence(Base):
    __tablename__ = "orchestration_evidence"
    __table_args__ = (
        Index("idx_orch_evidence_gate_created", "gate_id", "created_at"),
        Index("idx_orch_evidence_run_source", "run_id", "source_type", "source_id"),
        Index("idx_orch_evidence_observed_event", "observed_event_id"),
        Index("idx_orch_evidence_producer_agent", "producer_agent_id"),
        CheckConstraint(EVIDENCE_VERDICT_CHECK, name="ck_orchestration_evidence_verdict"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_runs.id", ondelete="CASCADE"), nullable=False)
    gate_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_gates.id", ondelete="CASCADE"), nullable=False)
    source_type: Mapped[str] = mapped_column(String(100), nullable=False)
    source_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    observed_event_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("event_log.id", ondelete="SET NULL"), nullable=True
    )
    producer_agent_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("agents.id", ondelete="SET NULL"), nullable=True
    )
    verdict: Mapped[str] = mapped_column(String(50), nullable=False, server_default="candidate")
    evidence_metadata: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class OrchestrationAgentSuggestion(Base):
    __tablename__ = "orchestration_agent_suggestions"
    __table_args__ = (
        Index("idx_orch_agent_suggestions_run_status", "run_id", "status"),
        Index("idx_orch_agent_suggestions_missing_work_function", "missing_work_function"),
        CheckConstraint(AGENT_SUGGESTION_STATUS_CHECK, name="ck_orchestration_agent_suggestions_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_runs.id", ondelete="CASCADE"), nullable=False)
    missing_work_function: Mapped[str] = mapped_column(String(100), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    suggested_role: Mapped[str | None] = mapped_column(String(100), nullable=True)
    suggested_capabilities: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    suggested_adapter_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    suggested_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    suggested_system_prompt_outline: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, server_default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )
