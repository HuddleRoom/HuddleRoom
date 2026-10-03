import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from huddleroom.models.base import Base, _utcnow


MESSAGE_NAMESPACE = uuid.UUID("ec1c3f17-589d-4f4d-a6a3-1a810f0c739b")
RESPONSE_NAMESPACE = uuid.UUID("65f6b744-294a-454d-a55d-1a6b9046aa1f")
RESERVATION_NAMESPACE = uuid.UUID("cac7b606-6451-4b6f-a624-8470ee6795ca")
INVESTIGATION_NAMESPACE = uuid.UUID("e9590749-7660-47cd-87b6-1164f8bab519")
INVESTIGATION_RESERVATION_NAMESPACE = uuid.UUID("474d758a-437c-473e-a7fd-b7843755e1bb")
FEEDBACK_NAMESPACE = uuid.UUID("e45619f8-44b2-4cf8-b7a8-38e6e683239d")
FEEDBACK_RATING_CHECK = "rating IN ('helpful', 'not_helpful')"
FEEDBACK_REASON_CHECK = (
    "(rating = 'helpful' AND reason IS NULL) OR "
    "(rating = 'not_helpful' AND reason IS NOT NULL AND reason IN "
    "('unanswered', 'incorrect', 'missing_context', 'stale_context', 'unclear', 'too_limited', 'other'))"
)


def conversation_message_id(goal_id: uuid.UUID, actor_id: uuid.UUID, client_request_id: uuid.UUID) -> uuid.UUID:
    return uuid.uuid5(MESSAGE_NAMESPACE, f"{goal_id}:{actor_id}:{client_request_id}")


def conversation_response_id(message_id: uuid.UUID) -> uuid.UUID:
    return uuid.uuid5(RESPONSE_NAMESPACE, str(message_id))


def conversation_feedback_id(response_id: uuid.UUID, actor_id: uuid.UUID) -> uuid.UUID:
    return uuid.uuid5(FEEDBACK_NAMESPACE, f"{response_id}:{actor_id}")


def conversation_reservation_id(response_id: uuid.UUID) -> uuid.UUID:
    return uuid.uuid5(RESERVATION_NAMESPACE, str(response_id))


def conversation_provider_request_id(response_id: uuid.UUID) -> str:
    return f"rally-chat:{response_id}"


def conversation_investigation_id(response_id: uuid.UUID, context_version: str) -> uuid.UUID:
    return uuid.uuid5(INVESTIGATION_NAMESPACE, f"{response_id}:{context_version}")


def conversation_investigation_provider_identity(investigation_id: uuid.UUID) -> str:
    return f"rally-chat-investigation:{investigation_id}"


def conversation_investigation_provider_request_id(investigation_id: uuid.UUID, attempt: int) -> str:
    if attempt not in (1, 2):
        raise ValueError("investigation attempt must be 1 or 2")
    return f"{conversation_investigation_provider_identity(investigation_id)}:{attempt}"


def conversation_investigation_reservation_id(investigation_id: uuid.UUID) -> uuid.UUID:
    return uuid.uuid5(INVESTIGATION_RESERVATION_NAMESPACE, str(investigation_id))


class ConversationMessage(Base):
    __tablename__ = "orchestration_conversation_messages"
    __table_args__ = (
        UniqueConstraint("goal_id", "actor_id", "client_request_id", name="uq_orch_conversation_messages_goal_actor_request"),
        UniqueConstraint("goal_id", "sequence", name="uq_orch_conversation_messages_goal_sequence"),
        Index("idx_orch_conversation_messages_goal_sequence", "goal_id", "sequence"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    goal_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("orchestration_goals.id", name="fk_orch_conversation_messages_goal_id", ondelete="CASCADE"), nullable=False
    )
    actor_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", name="fk_orch_conversation_messages_actor_id", ondelete="RESTRICT"), nullable=False
    )
    client_request_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    sequence: Mapped[int] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)


class ConversationResponse(Base):
    __tablename__ = "orchestration_conversation_responses"
    __table_args__ = (
        UniqueConstraint("message_id", name="uq_orch_conversation_responses_message_id"),
        UniqueConstraint("provider_request_id", name="uq_orch_conversation_responses_provider_request_id"),
        Index("idx_orch_conversation_responses_status_deadline", "status", "deadline_at"),
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'interrupted_unknown')",
            name="ck_orch_conversation_responses_status",
        ),
        CheckConstraint(
            "(status = 'pending' AND started_at IS NULL AND deadline_at IS NULL AND finished_at IS NULL) OR "
            "(status = 'running' AND started_at IS NOT NULL AND deadline_at IS NOT NULL AND finished_at IS NULL) OR "
            "(status IN ('completed', 'interrupted_unknown') AND started_at IS NOT NULL AND deadline_at IS NOT NULL AND finished_at IS NOT NULL) OR "
            "(status = 'failed' AND finished_at IS NOT NULL AND ((started_at IS NULL AND deadline_at IS NULL) OR (started_at IS NOT NULL AND deadline_at IS NOT NULL)))",
            name="ck_orch_conversation_responses_lifecycle",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("orchestration_conversation_messages.id", name="fk_orch_conversation_responses_message_id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", server_default="pending")
    dossier: Mapped[dict] = mapped_column(JSON, nullable=False)
    context_manifest: Mapped[dict] = mapped_column(JSON, nullable=False)
    context_version: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_request_id: Mapped[str] = mapped_column(String(64), nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deadline_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)


class ConversationFeedback(Base):
    __tablename__ = "orchestration_conversation_feedback"
    __table_args__ = (
        UniqueConstraint("response_id", "actor_id", name="uq_orch_conversation_feedback_response_actor"),
        CheckConstraint(FEEDBACK_RATING_CHECK, name="ck_orch_conversation_feedback_rating"),
        CheckConstraint(FEEDBACK_REASON_CHECK, name="ck_orch_conversation_feedback_reason"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    response_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("orchestration_conversation_responses.id", name="fk_orch_conversation_feedback_response_id", ondelete="CASCADE"), nullable=False
    )
    actor_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", name="fk_orch_conversation_feedback_actor_id", ondelete="RESTRICT"), nullable=False
    )
    rating: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)


class ConversationReservation(Base):
    __tablename__ = "orchestration_conversation_reservations"
    __table_args__ = (
        UniqueConstraint("response_id", name="uq_orch_conversation_reservations_response_id"),
        Index("idx_orch_conversation_reservations_goal_actor_status", "goal_id", "actor_id", "status"),
        CheckConstraint(
            "ceiling_snapshot >= 0 AND reserved_tokens >= 0 AND settled_tokens >= 0 AND released_tokens >= 0 "
            "AND settled_tokens + released_tokens <= reserved_tokens",
            name="ck_orch_conversation_reservations_amounts",
        ),
        CheckConstraint(
            "status IN ('reserved', 'committed', 'settled', 'released', 'held_unknown')",
            name="ck_orch_conversation_reservations_status",
        ),
        CheckConstraint(
            "(status = 'reserved' AND committed_at IS NULL AND settled_at IS NULL AND released_at IS NULL) OR "
            "(status = 'committed' AND committed_at IS NOT NULL AND settled_at IS NULL AND released_at IS NULL) OR "
            "(status = 'settled' AND settled_tokens + released_tokens = reserved_tokens AND committed_at IS NOT NULL AND settled_at IS NOT NULL AND released_at IS NOT NULL) OR "
            "(status = 'released' AND settled_tokens = 0 AND released_tokens = reserved_tokens AND committed_at IS NULL AND settled_at IS NULL AND released_at IS NOT NULL) OR "
            "(status = 'held_unknown' AND committed_at IS NOT NULL AND settled_tokens = 0 AND released_tokens = 0 AND settled_at IS NULL AND released_at IS NULL)",
            name="ck_orch_conversation_reservations_lifecycle",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    response_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("orchestration_conversation_responses.id", name="fk_orch_conversation_reservations_response_id", ondelete="CASCADE"), nullable=False
    )
    goal_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("orchestration_goals.id", name="fk_orch_conversation_reservations_goal_id", ondelete="CASCADE"), nullable=False
    )
    actor_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", name="fk_orch_conversation_reservations_actor_id", ondelete="RESTRICT"), nullable=False
    )
    ceiling_snapshot: Mapped[int] = mapped_column(nullable=False)
    reserved_tokens: Mapped[int] = mapped_column(nullable=False)
    settled_tokens: Mapped[int] = mapped_column(nullable=False, default=0, server_default="0")
    released_tokens: Mapped[int] = mapped_column(nullable=False, default=0, server_default="0")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="reserved", server_default="reserved")
    committed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    released_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)


class ConversationInvestigation(Base):
    __tablename__ = "orchestration_conversation_investigations"
    __table_args__ = (
        UniqueConstraint("response_id", name="uq_orch_conversation_investigations_response"),
        UniqueConstraint("provider_identity", name="uq_orch_conversation_investigations_provider_identity"),
        UniqueConstraint("provider_request_id", name="uq_orch_conversation_investigations_provider_request"),
        Index("idx_orch_conversation_investigations_goal_status_deadline", "goal_id", "status", "deadline_at"),
        CheckConstraint("status IN ('pending', 'running', 'completed', 'limited', 'failed', 'cancelled', 'unavailable', 'interrupted_unknown')", name="ck_orch_conversation_investigations_status"),
        CheckConstraint("attempt_count BETWEEN 0 AND 2 AND repair_count BETWEEN 0 AND 1 AND retry_count BETWEEN 0 AND 1 AND repair_count + retry_count <= 1 AND attempt_count >= repair_count + retry_count AND accumulated_tokens >= 0", name="ck_orch_conversation_investigations_counters"),
        CheckConstraint(
            "(status = 'pending' AND attempt_count = 0 AND started_at IS NULL AND deadline_at IS NULL AND finished_at IS NULL) OR "
            "(status = 'running' AND attempt_count BETWEEN 1 AND 2 AND started_at IS NOT NULL AND deadline_at IS NOT NULL AND finished_at IS NULL) OR "
            "(status IN ('limited', 'unavailable') AND attempt_count = 0 AND finished_at IS NOT NULL) OR "
            "(status = 'failed' AND attempt_count = 0 AND repair_count = 0 AND retry_count = 0 AND accumulated_tokens = 0 AND provider_request_id IS NULL AND started_at IS NULL AND deadline_at IS NULL AND finished_at IS NOT NULL) OR "
            "(status IN ('completed', 'failed', 'interrupted_unknown') AND attempt_count BETWEEN 1 AND 2 AND finished_at IS NOT NULL) OR "
            "(status = 'cancelled' AND finished_at IS NOT NULL AND cancelled_at IS NOT NULL)",
            name="ck_orch_conversation_investigations_lifecycle",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    response_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_conversation_responses.id", name="fk_orch_conversation_investigations_response_id", ondelete="CASCADE"), nullable=False)
    goal_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_goals.id", name="fk_orch_conversation_investigations_goal_id", ondelete="CASCADE"), nullable=False)
    actor_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", name="fk_orch_conversation_investigations_actor_id", ondelete="RESTRICT"), nullable=False)
    context_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", server_default="pending")
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    scope: Mapped[list] = mapped_column(JSON, nullable=False)
    input_manifest: Mapped[dict] = mapped_column(JSON, nullable=False)
    provider_identity: Mapped[str] = mapped_column(String(96), nullable=False)
    provider_request_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    attempt_count: Mapped[int] = mapped_column(nullable=False, default=0, server_default="0")
    repair_count: Mapped[int] = mapped_column(nullable=False, default=0, server_default="0")
    retry_count: Mapped[int] = mapped_column(nullable=False, default=0, server_default="0")
    accumulated_tokens: Mapped[int] = mapped_column(nullable=False, default=0, server_default="0")
    report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deadline_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)


class ConversationInvestigationReservation(Base):
    __tablename__ = "orchestration_conversation_investigation_reservations"
    __table_args__ = (
        UniqueConstraint("investigation_id", name="uq_orch_conversation_investigation_reservations_investigation"),
        Index("idx_orch_conv_inv_res_goal_actor_status", "goal_id", "actor_id", "status"),
        CheckConstraint("ceiling_snapshot >= 0 AND reserved_tokens >= 0 AND settled_tokens >= 0 AND released_tokens >= 0 AND settled_tokens + released_tokens <= reserved_tokens", name="ck_orch_conversation_investigation_reservations_amounts"),
        CheckConstraint("status IN ('reserved', 'committed', 'settled', 'released', 'held_unknown')", name="ck_orch_conversation_investigation_reservations_status"),
        CheckConstraint(
            "(status = 'reserved' AND committed_at IS NULL AND settled_at IS NULL AND released_at IS NULL) OR "
            "(status = 'committed' AND committed_at IS NOT NULL AND settled_at IS NULL AND released_at IS NULL) OR "
            "(status = 'settled' AND settled_tokens + released_tokens = reserved_tokens AND committed_at IS NOT NULL AND settled_at IS NOT NULL AND released_at IS NOT NULL) OR "
            "(status = 'released' AND settled_tokens = 0 AND released_tokens = reserved_tokens AND committed_at IS NULL AND settled_at IS NULL AND released_at IS NOT NULL) OR "
            "(status = 'held_unknown' AND committed_at IS NOT NULL AND settled_tokens = 0 AND released_tokens = 0 AND settled_at IS NULL AND released_at IS NULL)",
            name="ck_orch_conversation_investigation_reservations_lifecycle",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    investigation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_conversation_investigations.id", name="fk_orch_conv_inv_res_investigation_id", ondelete="CASCADE"), nullable=False)
    goal_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orchestration_goals.id", name="fk_orch_conversation_investigation_reservations_goal_id", ondelete="CASCADE"), nullable=False)
    actor_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", name="fk_orch_conversation_investigation_reservations_actor_id", ondelete="RESTRICT"), nullable=False)
    ceiling_snapshot: Mapped[int] = mapped_column(nullable=False)
    reserved_tokens: Mapped[int] = mapped_column(nullable=False)
    settled_tokens: Mapped[int] = mapped_column(nullable=False, default=0, server_default="0")
    released_tokens: Mapped[int] = mapped_column(nullable=False, default=0, server_default="0")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="reserved", server_default="reserved")
    committed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    released_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)
