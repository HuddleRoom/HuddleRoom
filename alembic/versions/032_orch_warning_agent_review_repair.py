"""Repair the legacy orchestration warning agent-review column name."""

from alembic import op
import sqlalchemy as sa


revision = "032"
down_revision = "031"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("orchestration_warnings")}

    if "related_agent_review_id" in columns and "source_agent_review_id" not in columns:
        op.execute(
            sa.text(
                "ALTER TABLE orchestration_warnings "
                "RENAME COLUMN related_agent_review_id TO source_agent_review_id"
            )
        )

    if "idx_orch_warnings_source_agent_review" not in {
        index["name"] for index in sa.inspect(bind).get_indexes("orchestration_warnings")
    }:
        op.create_index(
            "idx_orch_warnings_source_agent_review",
            "orchestration_warnings",
            ["source_agent_review_id"],
        )


def downgrade() -> None:
    pass
