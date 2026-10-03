"""Add no_manager constraint to orchestration_goals

Revision ID: 029
Revises: 028
Create Date: 2026-07-22

"""
# pylint: disable=invalid-name,no-member
from alembic import op
import sqlalchemy as sa


revision = "029"
down_revision = "028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # MEDIUM fix: Reconcile data before adding constraint, mirroring migration 025's approach.
    # Any existing row with authority_model='no_manager' and a populated manager_agent_id
    # or manager_user_id would violate the constraint and fail migration on existing databases.
    bind = op.get_bind()
    bind.execute(sa.text("""
        UPDATE orchestration_goals
        SET manager_agent_id = NULL, manager_user_id = NULL
        WHERE authority_model = 'no_manager'
          AND (manager_agent_id IS NOT NULL OR manager_user_id IS NOT NULL)
    """))

    with op.batch_alter_table("orchestration_goals") as batch:
        batch.create_check_constraint(
            "ck_orchestration_goals_no_manager_constraint",
            "authority_model IS NULL OR authority_model != 'no_manager' OR (manager_agent_id IS NULL AND manager_user_id IS NULL)",
        )


def downgrade() -> None:
    with op.batch_alter_table("orchestration_goals") as batch:
        batch.drop_constraint("ck_orchestration_goals_no_manager_constraint", type_="check")
