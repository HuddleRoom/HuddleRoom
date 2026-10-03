"""add event_log dedup key

Revision ID: 012b
Revises: 012
Create Date: 2026-05-20

"""
from alembic import op
import sqlalchemy as sa

revision = "012b"
down_revision = "012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("event_log") as batch_op:
        batch_op.add_column(sa.Column("dedup_key", sa.String(length=255), nullable=True))
        batch_op.create_unique_constraint("uq_event_log_project_dedup_key", ["project_id", "dedup_key"])


def downgrade() -> None:
    with op.batch_alter_table("event_log") as batch_op:
        batch_op.drop_constraint("uq_event_log_project_dedup_key", type_="unique")
        batch_op.drop_column("dedup_key")
