"""Add durable orchestration pause fence (SPR230/SPR231)."""
import sqlalchemy as sa
from alembic import op

revision = "051"
down_revision = "050"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("orchestration_runs", sa.Column(
        "pause_requested", sa.Boolean(), nullable=False, server_default=sa.false(),
    ))
    op.add_column("orchestration_runs", sa.Column(
        "control_version", sa.BigInteger(), nullable=False, server_default="0",
    ))
    op.execute("UPDATE orchestration_runs SET pause_requested = true WHERE status = 'paused'")


def downgrade() -> None:
    op.drop_column("orchestration_runs", "control_version")
    op.drop_column("orchestration_runs", "pause_requested")
