"""m7_5_orchestration_agent_reviews

Revision ID: 023
Revises: 022
Create Date: 2026-07-17

"""
from alembic import op
import sqlalchemy as sa

revision = "023"
down_revision = "022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "orchestration_agent_reviews",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "goal_id", sa.Uuid(), sa.ForeignKey("orchestration_goals.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey("orchestration_runs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("agent_id", sa.Uuid(), sa.ForeignKey("agents.id", ondelete="SET NULL"), nullable=True),
        sa.Column(
            "source_process_run_id",
            sa.Uuid(),
            sa.ForeignKey("orchestration_process_runs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("review_context", sa.Text(), nullable=True),
        sa.Column("proposed_work_functions", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("definition_snapshot", sa.JSON(), nullable=False),
        sa.Column("fit_summary", sa.Text(), nullable=False),
        sa.Column("strengths", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("risks", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("recommended_changes", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("approved_for_work_functions", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_orch_agent_reviews_goal_agent", "orchestration_agent_reviews", ["goal_id", "agent_id"]
    )
    op.create_index("idx_orch_agent_reviews_run", "orchestration_agent_reviews", ["run_id"])
    op.create_index(
        "idx_orch_agent_reviews_source_process", "orchestration_agent_reviews", ["source_process_run_id"]
    )


def downgrade() -> None:
    op.drop_table("orchestration_agent_reviews")
