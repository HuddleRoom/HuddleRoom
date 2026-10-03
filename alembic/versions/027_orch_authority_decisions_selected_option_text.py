"""orchestration_authority_decisions: widen selected_option to Text

Revision ID: 027
Revises: 026
Create Date: 2026-07-19

"""
# pylint: disable=invalid-name,no-member
from alembic import op
import sqlalchemy as sa

revision = "027"
down_revision = "026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("orchestration_authority_decisions") as batch_op:
        batch_op.alter_column(
            "selected_option",
            existing_type=sa.String(255),
            type_=sa.Text(),
            existing_nullable=True,
        )


def downgrade() -> None:
    with op.batch_alter_table("orchestration_authority_decisions") as batch_op:
        batch_op.alter_column(
            "selected_option",
            existing_type=sa.Text(),
            type_=sa.String(255),
            existing_nullable=True,
        )
