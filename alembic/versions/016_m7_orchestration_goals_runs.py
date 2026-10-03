"""m7_orchestration_goals_runs

Revision ID: 016
Revises: 015
Create Date: 2026-06-29

"""
from alembic import op
import sqlalchemy as sa

revision = "016"
down_revision = "015"
branch_labels = None
depends_on = None


def _status_check(statuses: tuple[str, ...]) -> str:
    return f"status IN ({', '.join(repr(status) for status in statuses)})"


GOAL_STATUS_VALUES = ("active", "blocked", "paused", "completed", "cancelled")
RUN_STATUS_VALUES = ("running", "blocked", "paused", "completed", "cancelled")
ACTIVE_RUN_STATUS_VALUES = ("running", "blocked", "paused")
GOAL_STATUS_CHECK = _status_check(GOAL_STATUS_VALUES)
RUN_STATUS_CHECK = _status_check(RUN_STATUS_VALUES)
ACTIVE_RUN_STATUS_CHECK = _status_check(ACTIVE_RUN_STATUS_VALUES)


def upgrade() -> None:
    op.create_table(
        "orchestration_goals",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("success_criteria", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("constraints", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("budget", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("status", sa.String(), nullable=False, server_default=sa.text("'active'")),
        sa.Column("created_by_user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(GOAL_STATUS_CHECK, name="ck_orchestration_goals_status"),
    )
    op.create_index("idx_orch_goals_project_status", "orchestration_goals", ["project_id", "status"])
    op.create_index("idx_orch_goals_project_created", "orchestration_goals", ["project_id", "created_at"])

    op.create_table(
        "orchestration_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "goal_id",
            sa.Uuid(),
            sa.ForeignKey("orchestration_goals.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.String(), nullable=False, server_default=sa.text("'running'")),
        sa.Column("event_cursor", sa.Uuid(), nullable=True),
        sa.Column("plan_state", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("active_blockers", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("budget_state", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("retry_state", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(RUN_STATUS_CHECK, name="ck_orchestration_runs_status"),
    )
    op.create_index("idx_orch_runs_goal_status", "orchestration_runs", ["goal_id", "status"])
    op.create_index(
        "uq_orch_runs_one_active_per_goal",
        "orchestration_runs",
        ["goal_id"],
        unique=True,
        sqlite_where=sa.text(ACTIVE_RUN_STATUS_CHECK),
        postgresql_where=sa.text(ACTIVE_RUN_STATUS_CHECK),
    )


def downgrade() -> None:
    op.drop_index("uq_orch_runs_one_active_per_goal", table_name="orchestration_runs")
    op.drop_index("idx_orch_runs_goal_status", table_name="orchestration_runs")
    op.drop_table("orchestration_runs")
    op.drop_index("idx_orch_goals_project_created", table_name="orchestration_goals")
    op.drop_index("idx_orch_goals_project_status", table_name="orchestration_goals")
    op.drop_table("orchestration_goals")
