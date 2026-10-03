"""Goal execution runtime: goal_type, run.phase, supersedes_goal_id.

Revision ID: 036
Revises: 035
Create Date: 2026-09-02

"""
# pylint: disable=invalid-name,no-member
from alembic import op
import sqlalchemy as sa

revision = "036"
down_revision = "035"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("orchestration_goals") as batch:
        batch.add_column(
            sa.Column("goal_type", sa.String(length=50), nullable=False, server_default="outcome")
        )
        batch.add_column(sa.Column("supersedes_goal_id", sa.Uuid(), nullable=True))
        batch.create_unique_constraint("uq_orch_goals_supersedes_goal_id", ["supersedes_goal_id"])
        batch.create_check_constraint(
            "ck_orchestration_goals_goal_type",
            "goal_type IN ('outcome', 'roadmap', 'continuous')",
        )
        batch.create_foreign_key(
            "fk_orch_goals_supersedes_goal_id",
            "orchestration_goals",
            ["supersedes_goal_id"],
            ["id"],
            ondelete="SET NULL",
        )
    with op.batch_alter_table("orchestration_runs") as batch:
        batch.add_column(
            sa.Column("phase", sa.String(length=50), nullable=False, server_default="baseline")
        )
        batch.create_check_constraint(
            "ck_orchestration_runs_phase",
            "phase IN ('baseline', 'ready', 'authorized', 'completed')",
        )


def downgrade() -> None:
    with op.batch_alter_table("orchestration_runs") as batch:
        batch.drop_constraint("ck_orchestration_runs_phase", type_="check")
        batch.drop_column("phase")
    with op.batch_alter_table("orchestration_goals") as batch:
        batch.drop_constraint("fk_orch_goals_supersedes_goal_id", type_="foreignkey")
        batch.drop_constraint("ck_orchestration_goals_goal_type", type_="check")
        batch.drop_constraint("uq_orch_goals_supersedes_goal_id", type_="unique")
        batch.drop_column("supersedes_goal_id")
        batch.drop_column("goal_type")
