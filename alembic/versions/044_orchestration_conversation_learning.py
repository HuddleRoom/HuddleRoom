"""Persist immutable conversation answer feedback.

Revision ID: 044
Revises: 043
Create Date: 2026-09-16
"""
# pylint: disable=invalid-name
import sqlalchemy as sa
from alembic import op

from huddleroom.models.orchestration_conversation import FEEDBACK_RATING_CHECK, FEEDBACK_REASON_CHECK


revision = "044"
down_revision = "043"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "orchestration_conversation_feedback",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("response_id", sa.Uuid(), nullable=False),
        sa.Column("actor_id", sa.Uuid(), nullable=False),
        sa.Column("rating", sa.String(32), nullable=False),
        sa.Column("reason", sa.String(32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["response_id"], ["orchestration_conversation_responses.id"], name="fk_orch_conversation_feedback_response_id", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["actor_id"], ["users.id"], name="fk_orch_conversation_feedback_actor_id", ondelete="RESTRICT"),
        sa.UniqueConstraint("response_id", "actor_id", name="uq_orch_conversation_feedback_response_actor"),
        sa.CheckConstraint(FEEDBACK_RATING_CHECK, name="ck_orch_conversation_feedback_rating"),
        sa.CheckConstraint(FEEDBACK_REASON_CHECK, name="ck_orch_conversation_feedback_reason"),
    )


def downgrade() -> None:
    op.drop_table("orchestration_conversation_feedback")
