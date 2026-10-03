"""m7_5_orchestration_process_runs_warnings

Revision ID: 021
Revises: 020
Create Date: 2026-07-16

"""
from alembic import op
import sqlalchemy as sa

revision = "021"
down_revision = "020"
branch_labels = None
depends_on = None


def _column_check(column_name: str, values: tuple[str, ...]) -> str:
    return f"{column_name} IN ({', '.join(repr(value) for value in values)})"


PROCESS_RUN_STATUS_VALUES = ("running", "waiting_decision", "completed", "skipped")
WARNING_SEVERITY_VALUES = ("recommendation", "warning", "blocker", "hard_stop")
PROCESS_RUN_STATUS_CHECK = _column_check("status", PROCESS_RUN_STATUS_VALUES)
WARNING_SEVERITY_CHECK = _column_check("severity", WARNING_SEVERITY_VALUES)


def upgrade() -> None:
    op.create_table(
        "orchestration_process_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "goal_id", sa.Uuid(), sa.ForeignKey("orchestration_goals.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey("orchestration_runs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("process_type", sa.String(length=100), nullable=False),
        sa.Column("process_version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("status", sa.String(length=50), nullable=False, server_default=sa.text("'running'")),
        sa.Column("trigger_reason", sa.Text(), nullable=False),
        sa.Column("input_snapshot", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("outputs", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("skipped_by", sa.String(length=255), nullable=True),
        sa.Column("override_reason", sa.Text(), nullable=True),
        sa.Column(
            "superseded_by_id",
            sa.Uuid(),
            sa.ForeignKey("orchestration_process_runs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(PROCESS_RUN_STATUS_CHECK, name="ck_orchestration_process_runs_status"),
    )
    op.create_index(
        "idx_orch_process_runs_goal_type", "orchestration_process_runs", ["goal_id", "process_type"]
    )
    op.create_index("idx_orch_process_runs_run", "orchestration_process_runs", ["run_id"])

    op.create_table(
        "orchestration_warnings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "goal_id", sa.Uuid(), sa.ForeignKey("orchestration_goals.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey("orchestration_runs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("warning_type", sa.String(length=100), nullable=False),
        sa.Column("severity", sa.String(length=50), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column(
            "source_process_run_id",
            sa.Uuid(),
            sa.ForeignKey("orchestration_process_runs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "related_gate_id", sa.Uuid(), sa.ForeignKey("orchestration_gates.id", ondelete="SET NULL"), nullable=True
        ),
        sa.Column(
            "related_action_id",
            sa.Uuid(),
            sa.ForeignKey("orchestration_actions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("related_agent_id", sa.Uuid(), sa.ForeignKey("agents.id", ondelete="SET NULL"), nullable=True),
        sa.Column("acknowledged_by", sa.String(length=255), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column("resolved_by", sa.String(length=255), nullable=True),
        sa.Column("resolved_reason", sa.Text(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(WARNING_SEVERITY_CHECK, name="ck_orchestration_warnings_severity"),
    )
    op.create_index("idx_orch_warnings_goal_active", "orchestration_warnings", ["goal_id", "active"])
    op.create_index("idx_orch_warnings_run", "orchestration_warnings", ["run_id"])
    op.create_index(
        "idx_orch_warnings_source_process", "orchestration_warnings", ["source_process_run_id"]
    )


def downgrade() -> None:
    op.drop_table("orchestration_warnings")
    op.drop_table("orchestration_process_runs")
