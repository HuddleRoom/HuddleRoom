"""orchestration_memory_sections: add project+goal+key unique constraint

Revision ID: 026
Revises: 025
Create Date: 2026-07-19

"""
# pylint: disable=invalid-name,no-member
from alembic import op

revision = "026"
down_revision = "025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # SQLite has no ALTER-based constraint support; batch mode recreates the
    # table under the hood. Postgres runs the plain ADD CONSTRAINT path.
    with op.batch_alter_table("orchestration_memory_sections") as batch_op:
        batch_op.create_unique_constraint(
            "uq_orch_memory_sections_project_goal_key",
            ["project_id", "goal_id", "section_key"],
        )


def downgrade() -> None:
    with op.batch_alter_table("orchestration_memory_sections") as batch_op:
        batch_op.drop_constraint(
            "uq_orch_memory_sections_project_goal_key",
            type_="unique",
        )
