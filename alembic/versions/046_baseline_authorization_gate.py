"""Add baseline_authorized gate column to orchestration_runs.

Revision ID: 046
Revises: 045
Create Date: 2026-09-23
"""
# pylint: disable=invalid-name
import sqlalchemy as sa
from alembic import op


revision = "046"
down_revision = "045"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "orchestration_runs",
        sa.Column("baseline_authorized", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )
    op.execute("UPDATE orchestration_runs SET baseline_authorized = false WHERE phase = 'baseline'")


def downgrade() -> None:
    op.drop_column("orchestration_runs", "baseline_authorized")
