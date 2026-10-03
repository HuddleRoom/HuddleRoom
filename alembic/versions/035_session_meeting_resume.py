"""Add resume support to sessions and meetings.

Revision ID: 035
Revises: 034
Create Date: 2026-08-07

"""
from alembic import op
import sqlalchemy as sa


revision = "035"
down_revision = "034"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "sessions",
        sa.Column("resumable", sa.Boolean(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column(
        "sessions",
        sa.Column("provider_session_id", sa.String(), nullable=True),
    )
    op.add_column(
        "meetings",
        sa.Column("resume_state", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
    )


def downgrade() -> None:
    op.drop_column("meetings", "resume_state")
    op.drop_column("sessions", "provider_session_id")
    op.drop_column("sessions", "resumable")
