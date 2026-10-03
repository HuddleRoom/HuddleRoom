"""sqlite compatibility — dialect-agnostic types

Revision ID: 002
Revises: 001
Create Date: 2026-04-25

"""
from alembic import op
import sqlalchemy as sa
from huddleroom.config import settings

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if settings.is_postgres:
        op.alter_column("sessions", "celery_task_id", new_column_name="runner_task_id")
        op.drop_index("idx_sessions_celery_task_id", table_name="sessions")
        op.create_index("idx_sessions_runner_task_id", "sessions", ["runner_task_id"])
    else:
        conn = op.get_bind()
        conn.execute(sa.text("DROP INDEX IF EXISTS idx_sessions_celery_task_id"))
        conn.execute(sa.text("ALTER TABLE sessions RENAME COLUMN celery_task_id TO runner_task_id"))
        conn.execute(sa.text("CREATE INDEX idx_sessions_runner_task_id ON sessions (runner_task_id)"))


def downgrade() -> None:
    if settings.is_postgres:
        op.alter_column("sessions", "runner_task_id", new_column_name="celery_task_id")
        op.drop_index("idx_sessions_runner_task_id", table_name="sessions")
        op.create_index("idx_sessions_celery_task_id", "sessions", ["celery_task_id"])
    else:
        conn = op.get_bind()
        conn.execute(sa.text("DROP INDEX IF EXISTS idx_sessions_runner_task_id"))
        conn.execute(sa.text("ALTER TABLE sessions RENAME COLUMN runner_task_id TO celery_task_id"))
        conn.execute(sa.text("CREATE INDEX idx_sessions_celery_task_id ON sessions (celery_task_id)"))
