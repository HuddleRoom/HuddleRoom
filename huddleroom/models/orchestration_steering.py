import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from huddleroom.models.base import Base, _utcnow


STEERING_STATE_NAMESPACE = uuid.UUID("a0d57d54-1754-443b-91ca-393c41e371a8")
STEERING_PROPOSAL_NAMESPACE = uuid.UUID("49f2660d-a1c0-466e-8d02-82f92c335794")
STEERING_REQUEST_NAMESPACE = uuid.UUID("54c70b46-2f29-456f-8aad-bec4d4ac2642")
STEERING_TRANSITION_NAMESPACE = uuid.UUID("1f0952bb-38d1-45fc-826f-7292b9bf74d9")
STEERING_RESULT_NAMESPACE = uuid.UUID("0879bd62-b394-4d72-8bb4-0cff284a8c3e")


def steering_state_id(goal_id: uuid.UUID) -> uuid.UUID:
    return uuid.uuid5(STEERING_STATE_NAMESPACE, str(goal_id))


def steering_proposal_id(response_id: uuid.UUID) -> uuid.UUID:
    return uuid.uuid5(STEERING_PROPOSAL_NAMESPACE, str(response_id))


def steering_request_id(goal_id: uuid.UUID, actor_id: uuid.UUID, client_request_id: uuid.UUID) -> uuid.UUID:
    return uuid.uuid5(STEERING_REQUEST_NAMESPACE, f"{goal_id}:{actor_id}:{client_request_id}")


def steering_transition_id(request_id: uuid.UUID, sequence: int) -> uuid.UUID:
    return uuid.uuid5(STEERING_TRANSITION_NAMESPACE, f"{request_id}:{sequence}")


def steering_result_link_id(request_id: uuid.UUID, decision_id: uuid.UUID, action_id: uuid.UUID) -> uuid.UUID:
    return uuid.uuid5(STEERING_RESULT_NAMESPACE, f"{request_id}:{decision_id}:{action_id}")


class OrchestrationSteeringState(Base):
    __tablename__ = "orchestration_steering_state"
    __table_args__ = (
        UniqueConstraint("goal_id", name="uq_orch_steering_state_goal"),
        CheckConstraint("inbox_version >= 0 AND direction_version >= 0", name="ck_orch_steering_state_versions"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    goal_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_goals.id", name="fk_orch_steering_state_goal_id", ondelete="CASCADE"), nullable=False)
    inbox_version: Mapped[int] = mapped_column(nullable=False, default=0, server_default="0")
    direction_version: Mapped[int] = mapped_column(nullable=False, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)


class OrchestrationSteeringProposal(Base):
    __tablename__ = "orchestration_steering_proposals"
    __table_args__ = (
        UniqueConstraint("response_id", name="uq_orch_steering_proposals_response"),
        Index("idx_orch_steering_proposals_goal_status", "goal_id", "status"),
        CheckConstraint("status IN ('proposed', 'dismissed', 'promoted')", name="ck_orch_steering_proposals_status"),
        CheckConstraint("(status = 'proposed' AND dismissed_at IS NULL AND promoted_request_id IS NULL) OR (status = 'dismissed' AND dismissed_at IS NOT NULL AND promoted_request_id IS NULL) OR (status = 'promoted' AND dismissed_at IS NULL AND promoted_request_id IS NOT NULL)", name="ck_orch_steering_proposals_lifecycle"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    response_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_conversation_responses.id", name="fk_orch_steering_proposals_response_id", ondelete="CASCADE"), nullable=False)
    goal_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_goals.id", name="fk_orch_steering_proposals_goal_id", ondelete="CASCADE"), nullable=False)
    actor_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", name="fk_orch_steering_proposals_actor_id", ondelete="RESTRICT"), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="proposed", server_default="proposed")
    draft: Mapped[dict] = mapped_column(JSON, nullable=False)
    dismissed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    promoted_request_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)


class OrchestrationSteeringRequest(Base):
    __tablename__ = "orchestration_steering_requests"
    __table_args__ = (
        UniqueConstraint("goal_id", "actor_id", "client_request_id", name="uq_orch_steering_requests_goal_actor_client"),
        UniqueConstraint("goal_id", "sequence", name="uq_orch_steering_requests_goal_sequence"),
        Index("idx_orch_steering_requests_goal_status_sequence", "goal_id", "status", "sequence"),
        CheckConstraint("status IN ('pending', 'being_considered', 'applied', 'deferred', 'rejected', 'superseded', 'needs_clarification', 'withdrawn')", name="ck_orch_steering_requests_status"),
        CheckConstraint("target_type IN ('goal', 'plan_item', 'task')", name="ck_orch_steering_requests_target_type"),
        CheckConstraint("length(directive) BETWEEN 1 AND 4000", name="ck_orch_steering_requests_directive_length"),
        CheckConstraint("(scope = 'item' AND lifetime = 'selected_item') OR (scope = 'run' AND lifetime = 'remaining_current_run') OR (scope = 'goal' AND lifetime = 'future_runs')", name="ck_orch_steering_requests_scope_lifetime"),
        CheckConstraint("sequence >= 1", name="ck_orch_steering_requests_sequence"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    goal_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_goals.id", name="fk_orch_steering_requests_goal_id", ondelete="CASCADE"), nullable=False)
    actor_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", name="fk_orch_steering_requests_actor_id", ondelete="RESTRICT"), nullable=False)
    client_request_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    sequence: Mapped[int] = mapped_column(nullable=False)
    submitted_run_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    directive: Mapped[str] = mapped_column(String(4000), nullable=False)
    target_type: Mapped[str] = mapped_column(String(32), nullable=False)
    target_id: Mapped[str] = mapped_column(String(255), nullable=False)
    scope: Mapped[str] = mapped_column(String(32), nullable=False)
    lifetime: Mapped[str] = mapped_column(String(32), nullable=False)
    impact_summary: Mapped[str] = mapped_column(String(1000), nullable=False)
    source_proposal_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("orchestration_steering_proposals.id", name="fk_orch_steering_requests_source_proposal_id", ondelete="SET NULL"), nullable=True)
    supersedes_request_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("orchestration_steering_requests.id", name="fk_orch_steering_requests_supersedes_request_id", ondelete="SET NULL", use_alter=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", server_default="pending")
    reason_code: Mapped[str] = mapped_column(String(64), nullable=False)
    contract_version: Mapped[str] = mapped_column(String(96), nullable=False)
    plan_version: Mapped[str] = mapped_column(String(96), nullable=False)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    considered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)


class OrchestrationSteeringTransition(Base):
    __tablename__ = "orchestration_steering_transitions"
    __table_args__ = (
        UniqueConstraint("request_id", "sequence", name="uq_orch_steering_transitions_request_sequence"),
        Index("idx_orch_steering_transitions_request_sequence", "request_id", "sequence"),
        CheckConstraint("sequence >= 1", name="ck_orch_steering_transitions_sequence"),
        CheckConstraint("to_status IN ('pending', 'being_considered', 'applied', 'deferred', 'rejected', 'superseded', 'needs_clarification', 'withdrawn') AND (from_status IS NULL OR from_status IN ('pending', 'being_considered', 'applied', 'deferred', 'rejected', 'superseded', 'needs_clarification', 'withdrawn'))", name="ck_orch_steering_transitions_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_steering_requests.id", name="fk_orch_steering_transitions_request_id", ondelete="CASCADE"), nullable=False)
    sequence: Mapped[int] = mapped_column(nullable=False)
    from_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    to_status: Mapped[str] = mapped_column(String(32), nullable=False)
    reason_code: Mapped[str] = mapped_column(String(64), nullable=False)
    actor: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)


class OrchestrationSteeringResultLink(Base):
    __tablename__ = "orchestration_steering_result_links"
    __table_args__ = (UniqueConstraint("request_id", "decision_id", "action_id", name="uq_orch_steering_result_links_request_decision_action"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_steering_requests.id", name="fk_orch_steering_result_links_request_id", ondelete="CASCADE"), nullable=False)
    decision_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_decisions.id", name="fk_orch_steering_result_links_decision_id", ondelete="RESTRICT"), nullable=False)
    action_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_actions.id", name="fk_orch_steering_result_links_action_id", ondelete="RESTRICT"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
