"""Preserve the immutable original orchestration request."""

from alembic import op
import sqlalchemy as sa


revision = "034"
down_revision = "033"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("orchestration_goals", sa.Column("original_request", sa.Text(), nullable=True))
    op.execute(sa.text("UPDATE orchestration_goals SET original_request = objective"))
    with op.batch_alter_table("orchestration_goals") as batch:
        batch.alter_column("original_request", existing_type=sa.Text(), nullable=False)


def downgrade() -> None:
    op.drop_column("orchestration_goals", "original_request")
