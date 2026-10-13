"""Add project chat index without rewriting legacy records.

Revision ID: 050
Revises: 049
"""
import sqlalchemy as sa
from alembic import op

revision = "050"
down_revision = "049"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "project_chat_turns",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("project_id", sa.Uuid(), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("actor_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("client_request_id", sa.Uuid(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("context_goal_id", sa.Uuid(), nullable=True),
        sa.Column("run_id", sa.Uuid(), nullable=True),
        sa.Column("frozen_context", sa.JSON(), nullable=False),
        sa.Column("source_kind", sa.String(32), nullable=False),
        sa.Column("source_record_id", sa.Uuid(), nullable=False),
        sa.Column("source_references", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("provider_request_id", sa.String(64), nullable=False),
        sa.Column("provider_status", sa.String(32), nullable=False, server_default="not_started"),
        sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("accounting_snapshot", sa.JSON(), nullable=False),
        sa.Column("interpretation", sa.JSON(), nullable=True),
        sa.Column("action_draft", sa.JSON(), nullable=True),
        sa.Column("action_receipt", sa.JSON(), nullable=True),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("error", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("project_id", "actor_id", "client_request_id", name="uq_project_chat_turns_request"),
        sa.UniqueConstraint("provider_request_id", name="uq_project_chat_turns_provider_request"),
        sa.CheckConstraint("status IN ('pending', 'running', 'completed', 'failed', 'interrupted_unknown')", name="ck_project_chat_turns_status"),
        sa.CheckConstraint("provider_status IN ('not_started', 'pending', 'running', 'completed', 'failed', 'interrupted_unknown')", name="ck_project_chat_turns_provider_status"),
        sa.CheckConstraint("source_kind IN ('project_advisor', 'goal_conversation', 'action', 'clarification')", name="ck_project_chat_turns_source"),
    )
    op.create_index("idx_project_chat_turns_thread", "project_chat_turns", ["project_id", "actor_id", "created_at"])
    op.create_index("idx_project_chat_turns_recovery", "project_chat_turns", ["provider_status", "deadline_at"])


def downgrade() -> None:
    op.drop_table("project_chat_turns")
