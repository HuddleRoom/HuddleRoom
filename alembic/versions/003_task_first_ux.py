"""task-first UX — add origin tracking to sessions

Revision ID: 003
Revises: 002
Create Date: 2026-05-01

"""
from alembic import op
import sqlalchemy as sa
from huddleroom.config import settings

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if settings.is_postgres:
        op.add_column(
            "sessions",
            sa.Column(
                "origin",
                sa.String(),
                nullable=False,
                server_default="manual",
            ),
        )
        op.create_index("idx_sessions_origin", "sessions", ["origin"])
        # Unique partial index: only one active session per task at a time
        op.execute(sa.text(
            "CREATE UNIQUE INDEX uq_session_task_active ON sessions (task_id) "
            "WHERE status IN ('pending', 'running') AND task_id IS NOT NULL"
        ))
    else:
        conn = op.get_bind()
        conn.execute(
            sa.text(
                "ALTER TABLE sessions ADD COLUMN origin TEXT NOT NULL DEFAULT 'manual'"
            )
        )
        conn.execute(
            sa.text("CREATE INDEX idx_sessions_origin ON sessions (origin)")
        )
        # Unique partial index: only one active session per task at a time
        conn.execute(sa.text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_session_task_active ON sessions (task_id) "
            "WHERE status IN ('pending', 'running') AND task_id IS NOT NULL"
        ))


def downgrade() -> None:
    if settings.is_postgres:
        op.execute(sa.text("DROP INDEX IF EXISTS uq_session_task_active"))
        op.drop_index("idx_sessions_origin", table_name="sessions")
        op.drop_column("sessions", "origin")
    else:
        conn = op.get_bind()
        conn.execute(sa.text("DROP INDEX IF EXISTS uq_session_task_active"))
        conn.execute(sa.text("DROP INDEX IF EXISTS idx_sessions_origin"))
        conn.execute(sa.text("ALTER TABLE sessions DROP COLUMN origin"))
