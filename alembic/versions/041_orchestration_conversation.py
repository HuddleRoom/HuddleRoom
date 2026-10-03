"""Persist goal-scoped orchestration conversation records.

Revision ID: 041
Revises: 040
Create Date: 2026-09-14
"""
# pylint: disable=invalid-name
import sqlalchemy as sa
from alembic import op


revision = "041"
down_revision = "040"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "orchestration_conversation_messages",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("goal_id", sa.Uuid(), nullable=False),
        sa.Column("actor_id", sa.Uuid(), nullable=False),
        sa.Column("client_request_id", sa.Uuid(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["goal_id"], ["orchestration_goals.id"], name="fk_orch_conversation_messages_goal_id", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["actor_id"], ["users.id"], name="fk_orch_conversation_messages_actor_id", ondelete="RESTRICT"),
        sa.UniqueConstraint("goal_id", "actor_id", "client_request_id", name="uq_orch_conversation_messages_goal_actor_request"),
        sa.UniqueConstraint("goal_id", "sequence", name="uq_orch_conversation_messages_goal_sequence"),
    )
    op.create_index("idx_orch_conversation_messages_goal_sequence", "orchestration_conversation_messages", ["goal_id", "sequence"])
    op.create_table(
        "orchestration_conversation_responses",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("message_id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("dossier", sa.JSON(), nullable=False),
        sa.Column("context_manifest", sa.JSON(), nullable=False),
        sa.Column("context_version", sa.String(64), nullable=False),
        sa.Column("provider_request_id", sa.String(64), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("error", sa.JSON(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["message_id"], ["orchestration_conversation_messages.id"], name="fk_orch_conversation_responses_message_id", ondelete="CASCADE"),
        sa.UniqueConstraint("message_id", name="uq_orch_conversation_responses_message_id"),
        sa.UniqueConstraint("provider_request_id", name="uq_orch_conversation_responses_provider_request_id"),
        sa.CheckConstraint("status IN ('pending', 'running', 'completed', 'failed', 'interrupted_unknown')", name="ck_orch_conversation_responses_status"),
        sa.CheckConstraint(
            "(status = 'pending' AND started_at IS NULL AND deadline_at IS NULL AND finished_at IS NULL) OR "
            "(status = 'running' AND started_at IS NOT NULL AND deadline_at IS NOT NULL AND finished_at IS NULL) OR "
            "(status IN ('completed', 'interrupted_unknown') AND started_at IS NOT NULL AND deadline_at IS NOT NULL AND finished_at IS NOT NULL) OR "
            "(status = 'failed' AND finished_at IS NOT NULL AND ((started_at IS NULL AND deadline_at IS NULL) OR (started_at IS NOT NULL AND deadline_at IS NOT NULL)))",
            name="ck_orch_conversation_responses_lifecycle",
        ),
    )
    op.create_index("idx_orch_conversation_responses_status_deadline", "orchestration_conversation_responses", ["status", "deadline_at"])
    op.create_table(
        "orchestration_conversation_reservations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("response_id", sa.Uuid(), nullable=False),
        sa.Column("goal_id", sa.Uuid(), nullable=False),
        sa.Column("actor_id", sa.Uuid(), nullable=False),
        sa.Column("ceiling_snapshot", sa.Integer(), nullable=False),
        sa.Column("reserved_tokens", sa.Integer(), nullable=False),
        sa.Column("settled_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("released_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(32), nullable=False, server_default="reserved"),
        sa.Column("committed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("settled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["response_id"], ["orchestration_conversation_responses.id"], name="fk_orch_conversation_reservations_response_id", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["goal_id"], ["orchestration_goals.id"], name="fk_orch_conversation_reservations_goal_id", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["actor_id"], ["users.id"], name="fk_orch_conversation_reservations_actor_id", ondelete="RESTRICT"),
        sa.UniqueConstraint("response_id", name="uq_orch_conversation_reservations_response_id"),
        sa.CheckConstraint("ceiling_snapshot >= 0 AND reserved_tokens >= 0 AND settled_tokens >= 0 AND released_tokens >= 0 AND settled_tokens + released_tokens <= reserved_tokens", name="ck_orch_conversation_reservations_amounts"),
        sa.CheckConstraint("status IN ('reserved', 'committed', 'settled', 'released', 'held_unknown')", name="ck_orch_conversation_reservations_status"),
        sa.CheckConstraint(
            "(status = 'reserved' AND committed_at IS NULL AND settled_at IS NULL AND released_at IS NULL) OR "
            "(status = 'committed' AND committed_at IS NOT NULL AND settled_at IS NULL AND released_at IS NULL) OR "
            "(status = 'settled' AND settled_tokens + released_tokens = reserved_tokens AND committed_at IS NOT NULL AND settled_at IS NOT NULL AND released_at IS NOT NULL) OR "
            "(status = 'released' AND settled_tokens = 0 AND released_tokens = reserved_tokens AND committed_at IS NULL AND settled_at IS NULL AND released_at IS NOT NULL) OR "
            "(status = 'held_unknown' AND committed_at IS NOT NULL AND settled_tokens = 0 AND released_tokens = 0 AND settled_at IS NULL AND released_at IS NULL)",
            name="ck_orch_conversation_reservations_lifecycle",
        ),
    )
    op.create_index("idx_orch_conversation_reservations_goal_actor_status", "orchestration_conversation_reservations", ["goal_id", "actor_id", "status"])


def downgrade() -> None:
    op.drop_index("idx_orch_conversation_reservations_goal_actor_status", table_name="orchestration_conversation_reservations")
    op.drop_table("orchestration_conversation_reservations")
    op.drop_index("idx_orch_conversation_responses_status_deadline", table_name="orchestration_conversation_responses")
    op.drop_table("orchestration_conversation_responses")
    op.drop_index("idx_orch_conversation_messages_goal_sequence", table_name="orchestration_conversation_messages")
    op.drop_table("orchestration_conversation_messages")
