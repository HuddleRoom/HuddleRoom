"""m7_5_orchestration_memory_sections

Revision ID: 020
Revises: 019
Create Date: 2026-07-15

"""
from alembic import op
import sqlalchemy as sa

revision = "020"
down_revision = "019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "orchestration_memory_sections",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column(
            "goal_id", sa.Uuid(), sa.ForeignKey("orchestration_goals.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey("orchestration_runs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("section_key", sa.String(length=100), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("section_type", sa.String(length=50), nullable=False, server_default=sa.text("'text'")),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("always_load", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("toc_order", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_by", sa.String(length=255), nullable=False),
        sa.Column(
            "created_from_event_id", sa.Uuid(), sa.ForeignKey("event_log.id", ondelete="SET NULL"), nullable=True
        ),
        sa.Column(
            "updated_from_event_id", sa.Uuid(), sa.ForeignKey("event_log.id", ondelete="SET NULL"), nullable=True
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("goal_id", "section_key", name="uq_orch_memory_sections_goal_key"),
    )
    op.create_index(
        "idx_orch_memory_sections_goal_always", "orchestration_memory_sections", ["goal_id", "always_load"]
    )
    op.create_index("idx_orch_memory_sections_project", "orchestration_memory_sections", ["project_id"])


def downgrade() -> None:
    op.drop_table("orchestration_memory_sections")
