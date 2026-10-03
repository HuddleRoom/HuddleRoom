"""meetings_cursor_index

Revision ID: 015
Revises: 014
Create Date: 2026-06-15

"""
from alembic import op

revision = '015'
down_revision = '014'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX idx_meetings_project_created_id ON meetings (project_id, created_at DESC, id DESC)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_meetings_project_created_id")
