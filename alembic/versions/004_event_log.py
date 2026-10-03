"""add event_log table

Revision ID: 004
Revises: 003
Create Date: 2026-05-02

"""
from alembic import op
import sqlalchemy as sa

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "event_log",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("source", sa.String(50), nullable=False, server_default="system"),
        sa.Column(
            "emitted_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    op.create_index("idx_event_log_project_emitted", "event_log", ["project_id", "emitted_at"])
    op.create_index("idx_event_log_event_type", "event_log", ["event_type"])


def downgrade() -> None:
    op.drop_index("idx_event_log_event_type", table_name="event_log")
    op.drop_index("idx_event_log_project_emitted", table_name="event_log")
    op.drop_table("event_log")
