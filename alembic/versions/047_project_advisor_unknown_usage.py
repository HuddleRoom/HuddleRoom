"""Allow project advisor turns to retain unknown usage.

Revision ID: 047
Revises: 046
Create Date: 2026-10-06
"""
# pylint: disable=invalid-name
import sqlalchemy as sa
from alembic import op


revision = "047"
down_revision = "046"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("project_advisor_turns") as batch:
        batch.alter_column("tokens_used", existing_type=sa.Integer(), nullable=True)


def downgrade() -> None:
    unknown_count = op.get_bind().execute(
        sa.text("SELECT COUNT(*) FROM project_advisor_turns WHERE tokens_used IS NULL")
    ).scalar_one()
    if unknown_count:
        raise RuntimeError("Cannot downgrade while project advisor usage is unknown")
    with op.batch_alter_table("project_advisor_turns") as batch:
        batch.alter_column("tokens_used", existing_type=sa.Integer(), nullable=False)
