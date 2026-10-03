"""Add manager selection columns to orchestration_goals

Revision ID: 028
Revises: 027
Create Date: 2026-07-19

"""
# pylint: disable=invalid-name,no-member
from alembic import op
import sqlalchemy as sa

revision = "028"
down_revision = "027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("orchestration_goals") as batch:
        batch.add_column(sa.Column("manager_agent_id", sa.Uuid(), nullable=True))
        batch.add_column(sa.Column("manager_user_id", sa.Uuid(), nullable=True))
        batch.add_column(sa.Column("authority_model", sa.String(length=50), nullable=True))
        batch.create_foreign_key(
            "fk_orch_goals_manager_agent_id",
            "agents",
            ["manager_agent_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_foreign_key(
            "fk_orch_goals_manager_user_id",
            "users",
            ["manager_user_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_check_constraint(
            "ck_orchestration_goals_authority_model",
            "authority_model IS NULL OR authority_model IN "
            "('agent_manager', 'human_manager', 'no_manager')",
        )
        batch.create_check_constraint(
            "ck_orchestration_goals_single_manager",
            "manager_agent_id IS NULL OR manager_user_id IS NULL",
        )


def downgrade() -> None:
    with op.batch_alter_table("orchestration_goals") as batch:
        batch.drop_constraint("ck_orchestration_goals_single_manager", type_="check")
        batch.drop_constraint("ck_orchestration_goals_authority_model", type_="check")
        batch.drop_constraint("fk_orch_goals_manager_user_id", type_="foreignkey")
        batch.drop_constraint("fk_orch_goals_manager_agent_id", type_="foreignkey")
        batch.drop_column("authority_model")
        batch.drop_column("manager_user_id")
        batch.drop_column("manager_agent_id")
