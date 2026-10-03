"""m7_orchestration_decisions_actions

Revision ID: 017
Revises: 016
Create Date: 2026-07-01

"""
from alembic import op
import sqlalchemy as sa

revision = "017"
down_revision = "016"
branch_labels = None
depends_on = None


def _status_check(statuses: tuple[str, ...]) -> str:
    return f"status IN ({', '.join(repr(status) for status in statuses)})"


def _validator_status_check(statuses: tuple[str, ...]) -> str:
    return f"validator_status IN ({', '.join(repr(status) for status in statuses)})"


DECISION_VALIDATOR_STATUS_VALUES = ("pending", "accepted", "rejected")
ACTION_STATUS_VALUES = ("reserved", "completed", "failed")
DECISION_VALIDATOR_STATUS_CHECK = _validator_status_check(DECISION_VALIDATOR_STATUS_VALUES)
ACTION_STATUS_CHECK = _status_check(ACTION_STATUS_VALUES)


def upgrade() -> None:
    op.create_table(
        "orchestration_decisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey("orchestration_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("decision_type", sa.String(length=100), nullable=False),
        sa.Column("input_snapshot", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("llm_output", sa.JSON(), nullable=True),
        sa.Column("parsed_decision", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("validator_status", sa.String(length=50), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("rejection_reason", sa.Text(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            DECISION_VALIDATOR_STATUS_CHECK,
            name="ck_orchestration_decisions_validator_status",
        ),
    )
    op.create_index("idx_orch_decisions_run_created", "orchestration_decisions", ["run_id", "created_at"])
    op.create_index("idx_orch_decisions_validator_status", "orchestration_decisions", ["validator_status"])

    op.create_table(
        "orchestration_actions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey("orchestration_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column(
            "decision_id",
            sa.Uuid(),
            sa.ForeignKey("orchestration_decisions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("action_type", sa.String(length=100), nullable=False),
        sa.Column("request", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("target_type", sa.String(length=100), nullable=True),
        sa.Column("target_id", sa.Uuid(), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False, server_default=sa.text("'reserved'")),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(ACTION_STATUS_CHECK, name="ck_orchestration_actions_status"),
        sa.UniqueConstraint("run_id", "idempotency_key", name="uq_orch_actions_run_idempotency_key"),
    )
    op.create_index("idx_orch_actions_run_status", "orchestration_actions", ["run_id", "status"])


def downgrade() -> None:
    op.drop_index("idx_orch_actions_run_status", table_name="orchestration_actions")
    op.drop_table("orchestration_actions")
    op.drop_index("idx_orch_decisions_validator_status", table_name="orchestration_decisions")
    op.drop_index("idx_orch_decisions_run_created", table_name="orchestration_decisions")
    op.drop_table("orchestration_decisions")
