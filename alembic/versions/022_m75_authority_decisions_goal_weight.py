"""m7_5_authority_decisions_goal_weight

Revision ID: 022
Revises: 021
Create Date: 2026-07-16

"""
from alembic import op
import sqlalchemy as sa
from huddleroom.config import settings

revision = "022"
down_revision = "021"
branch_labels = None
depends_on = None


def _column_check(column_name: str, values: tuple[str, ...]) -> str:
    return f"{column_name} IN ({', '.join(repr(value) for value in values)})"


AUTHORITY_DECISION_STATUS_VALUES = ("pending", "answered", "cancelled", "expired")
AUTHORITY_VALUES = ("human", "manager", "team_lead", "agent")
GOAL_WEIGHT_VALUES = ("trivial", "standard", "substantial")
AUTHORITY_DECISION_STATUS_CHECK = _column_check("status", AUTHORITY_DECISION_STATUS_VALUES)
AUTHORITY_CHECK = _column_check("authority", AUTHORITY_VALUES)
GOAL_WEIGHT_CHECK = _column_check("weight", GOAL_WEIGHT_VALUES)


def upgrade() -> None:
    op.create_table(
        "orchestration_authority_decisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "goal_id", sa.Uuid(), sa.ForeignKey("orchestration_goals.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey("orchestration_runs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("decision_key", sa.String(length=255), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("authority", sa.String(length=50), nullable=False),
        sa.Column(
            "source_process_run_id",
            sa.Uuid(),
            sa.ForeignKey("orchestration_process_runs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("context", sa.Text(), nullable=True),
        sa.Column("options", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("recommendation", sa.Text(), nullable=True),
        sa.Column("selected_option", sa.String(length=255), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("decided_by_user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("decided_by_agent_id", sa.Uuid(), sa.ForeignKey("agents.id", ondelete="SET NULL"), nullable=True),
        sa.Column("authority_agent_id", sa.Uuid(), sa.ForeignKey("agents.id", ondelete="SET NULL"), nullable=True),
        sa.Column("consequences", sa.Text(), nullable=True),
        sa.Column("overrides_recommendation", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column(
            "created_warning_id",
            sa.Uuid(),
            sa.ForeignKey("orchestration_warnings.id", ondelete="SET NULL"),
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
        sa.Column("asked_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(AUTHORITY_DECISION_STATUS_CHECK, name="ck_orch_authority_decisions_status"),
        sa.CheckConstraint(AUTHORITY_CHECK, name="ck_orch_authority_decisions_authority"),
    )
    # Partial unique index (pending rows only) — see model docstring / Spec
    # Deviation 7: a rerun must be able to raise a fresh decision under the
    # same key once the prior one is terminal, so the key is not unique
    # across all time. Supported on both SQLite (3.8+) and Postgres.
    op.create_index(
        "uq_orch_authority_decisions_goal_key_pending",
        "orchestration_authority_decisions",
        ["goal_id", "decision_key"],
        unique=True,
        sqlite_where=sa.text("status = 'pending'"),
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.create_index(
        "idx_orch_authority_decisions_goal_status",
        "orchestration_authority_decisions",
        ["goal_id", "status"],
    )
    op.create_index("idx_orch_authority_decisions_run", "orchestration_authority_decisions", ["run_id"])

    op.add_column(
        "orchestration_goals",
        sa.Column("weight", sa.String(length=50), nullable=False, server_default=sa.text("'standard'")),
    )
    op.add_column(
        "orchestration_goals",
        sa.Column("weight_overridden_by", sa.String(length=255), nullable=True),
    )
    # SQLite cannot add a table CHECK constraint without rebuilding the
    # table; fresh SQLite DBs get this CHECK from the model via create_all,
    # migrated SQLite DBs rely on service-level validation (Phase 5).
    if not settings.is_sqlite:
        op.create_check_constraint("ck_orchestration_goals_weight", "orchestration_goals", GOAL_WEIGHT_CHECK)


def downgrade() -> None:
    if not settings.is_sqlite:
        op.drop_constraint("ck_orchestration_goals_weight", "orchestration_goals", type_="check")
    op.drop_column("orchestration_goals", "weight_overridden_by")
    op.drop_column("orchestration_goals", "weight")
    op.drop_table("orchestration_authority_decisions")
