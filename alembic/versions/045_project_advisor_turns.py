"""Persist project advisor turns.

Revision ID: 045
Revises: 044
Create Date: 2026-09-18
"""
# pylint: disable=invalid-name
import sqlalchemy as sa
from alembic import op

from huddleroom.models.orchestration_advisor import ADVISOR_STATUS_CHECK


revision = "045"
down_revision = "044"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "project_advisor_turns",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("actor_id", sa.Uuid(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("citations", sa.JSON(), nullable=False),
        sa.Column("off_topic", sa.Boolean(), nullable=False),
        sa.Column("tokens_used", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], name="fk_project_advisor_turns_project_id", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["actor_id"], ["users.id"], name="fk_project_advisor_turns_actor_id", ondelete="RESTRICT"),
        sa.CheckConstraint(ADVISOR_STATUS_CHECK, name="ck_project_advisor_turns_status"),
    )


def downgrade() -> None:
    op.drop_table("project_advisor_turns")
