"""Add orchestrator-owned context to orchestration goals."""

from alembic import op
import sqlalchemy as sa


revision = "033"
down_revision = "032"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "orchestration_goals",
        sa.Column("orchestrator_context", sa.JSON(), nullable=False, server_default="{}"),
    )


def downgrade() -> None:
    op.drop_column("orchestration_goals", "orchestrator_context")
