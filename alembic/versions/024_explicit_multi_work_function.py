"""Add explicit_multi_work_function column to orchestration_goals

Revision ID: 024
Revises: 023
Create Date: 2026-07-19

"""
# pylint: disable=invalid-name,no-member
from alembic import op
import sqlalchemy as sa

revision = "024"
down_revision = "023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "orchestration_goals",
        sa.Column(
            "explicit_multi_work_function",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column("orchestration_goals", "explicit_multi_work_function")
