"""Link orchestration warnings to agent reviews.

Revision ID: 030
Revises: 029
Create Date: 2026-07-22

"""
# pylint: disable=invalid-name,no-member
from alembic import op
import sqlalchemy as sa


revision = "030"
down_revision = "029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("orchestration_warnings") as batch:
        batch.add_column(sa.Column("source_agent_review_id", sa.Uuid(), nullable=True))
        batch.create_foreign_key(
            "fk_orch_warnings_source_agent_review_id",
            "orchestration_agent_reviews",
            ["source_agent_review_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_index(
            "idx_orch_warnings_source_agent_review", ["source_agent_review_id"]
        )


def downgrade() -> None:
    with op.batch_alter_table("orchestration_warnings") as batch:
        batch.drop_index("idx_orch_warnings_source_agent_review")
        batch.drop_constraint("fk_orch_warnings_source_agent_review_id", type_="foreignkey")
        batch.drop_column("source_agent_review_id")
