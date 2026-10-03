"""Persist conversation investigation records.

Revision ID: 042
Revises: 041
Create Date: 2026-09-15
"""
# pylint: disable=invalid-name
import sqlalchemy as sa
from alembic import op


revision = "042"
down_revision = "041"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "orchestration_conversation_investigations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("response_id", sa.Uuid(), nullable=False),
        sa.Column("goal_id", sa.Uuid(), nullable=False),
        sa.Column("actor_id", sa.Uuid(), nullable=False),
        sa.Column("context_version", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("scope", sa.JSON(), nullable=False),
        sa.Column("input_manifest", sa.JSON(), nullable=False),
        sa.Column("provider_identity", sa.String(96), nullable=False),
        sa.Column("provider_request_id", sa.String(100), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("repair_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("accumulated_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("report", sa.JSON(), nullable=True),
        sa.Column("error", sa.JSON(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["response_id"], ["orchestration_conversation_responses.id"], name="fk_orch_conversation_investigations_response_id", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["goal_id"], ["orchestration_goals.id"], name="fk_orch_conversation_investigations_goal_id", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["actor_id"], ["users.id"], name="fk_orch_conversation_investigations_actor_id", ondelete="RESTRICT"),
        sa.UniqueConstraint("response_id", name="uq_orch_conversation_investigations_response"),
        sa.UniqueConstraint("provider_identity", name="uq_orch_conversation_investigations_provider_identity"),
        sa.UniqueConstraint("provider_request_id", name="uq_orch_conversation_investigations_provider_request"),
        sa.CheckConstraint("status IN ('pending', 'running', 'completed', 'limited', 'failed', 'cancelled', 'unavailable', 'interrupted_unknown')", name="ck_orch_conversation_investigations_status"),
        sa.CheckConstraint("attempt_count BETWEEN 0 AND 2 AND repair_count BETWEEN 0 AND 1 AND retry_count BETWEEN 0 AND 1 AND repair_count + retry_count <= 1 AND attempt_count >= repair_count + retry_count AND accumulated_tokens >= 0", name="ck_orch_conversation_investigations_counters"),
        sa.CheckConstraint(
            "(status = 'pending' AND attempt_count = 0 AND started_at IS NULL AND deadline_at IS NULL AND finished_at IS NULL) OR "
            "(status = 'running' AND attempt_count BETWEEN 1 AND 2 AND started_at IS NOT NULL AND deadline_at IS NOT NULL AND finished_at IS NULL) OR "
            "(status IN ('limited', 'unavailable') AND attempt_count = 0 AND finished_at IS NOT NULL) OR "
            "(status = 'failed' AND attempt_count = 0 AND repair_count = 0 AND retry_count = 0 AND accumulated_tokens = 0 AND provider_request_id IS NULL AND started_at IS NULL AND deadline_at IS NULL AND finished_at IS NOT NULL) OR "
            "(status IN ('completed', 'failed', 'interrupted_unknown') AND attempt_count BETWEEN 1 AND 2 AND finished_at IS NOT NULL) OR "
            "(status = 'cancelled' AND finished_at IS NOT NULL AND cancelled_at IS NOT NULL)",
            name="ck_orch_conversation_investigations_lifecycle",
        ),
    )
    op.create_index("idx_orch_conversation_investigations_goal_status_deadline", "orchestration_conversation_investigations", ["goal_id", "status", "deadline_at"])
    op.create_table(
        "orchestration_conversation_investigation_reservations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("investigation_id", sa.Uuid(), nullable=False),
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
        sa.ForeignKeyConstraint(["investigation_id"], ["orchestration_conversation_investigations.id"], name="fk_orch_conv_inv_res_investigation_id", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["goal_id"], ["orchestration_goals.id"], name="fk_orch_conversation_investigation_reservations_goal_id", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["actor_id"], ["users.id"], name="fk_orch_conversation_investigation_reservations_actor_id", ondelete="RESTRICT"),
        sa.UniqueConstraint("investigation_id", name="uq_orch_conversation_investigation_reservations_investigation"),
        sa.CheckConstraint("ceiling_snapshot >= 0 AND reserved_tokens >= 0 AND settled_tokens >= 0 AND released_tokens >= 0 AND settled_tokens + released_tokens <= reserved_tokens", name="ck_orch_conversation_investigation_reservations_amounts"),
        sa.CheckConstraint("status IN ('reserved', 'committed', 'settled', 'released', 'held_unknown')", name="ck_orch_conversation_investigation_reservations_status"),
        sa.CheckConstraint(
            "(status = 'reserved' AND committed_at IS NULL AND settled_at IS NULL AND released_at IS NULL) OR "
            "(status = 'committed' AND committed_at IS NOT NULL AND settled_at IS NULL AND released_at IS NULL) OR "
            "(status = 'settled' AND settled_tokens + released_tokens = reserved_tokens AND committed_at IS NOT NULL AND settled_at IS NOT NULL AND released_at IS NOT NULL) OR "
            "(status = 'released' AND settled_tokens = 0 AND released_tokens = reserved_tokens AND committed_at IS NULL AND settled_at IS NULL AND released_at IS NOT NULL) OR "
            "(status = 'held_unknown' AND committed_at IS NOT NULL AND settled_tokens = 0 AND released_tokens = 0 AND settled_at IS NULL AND released_at IS NULL)",
            name="ck_orch_conversation_investigation_reservations_lifecycle",
        ),
    )
    op.create_index("idx_orch_conv_inv_res_goal_actor_status", "orchestration_conversation_investigation_reservations", ["goal_id", "actor_id", "status"])


def downgrade() -> None:
    op.drop_index("idx_orch_conv_inv_res_goal_actor_status", table_name="orchestration_conversation_investigation_reservations")
    op.drop_table("orchestration_conversation_investigation_reservations")
    op.drop_index("idx_orch_conversation_investigations_goal_status_deadline", table_name="orchestration_conversation_investigations")
    op.drop_table("orchestration_conversation_investigations")
