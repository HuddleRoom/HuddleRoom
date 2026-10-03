"""meeting_outcome_fields_and_control_metadata

Revision ID: 008
Revises: 007
Create Date: 2026-05-16

"""
from alembic import op
import sqlalchemy as sa

revision = '008'
down_revision = '007'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('meeting_agenda_items') as batch_op:
        batch_op.add_column(sa.Column('resolution_kind', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('resolution_summary', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('required_followup', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('participants_heard', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('started_at', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('meeting_agenda_items') as batch_op:
        batch_op.drop_column('started_at')
        batch_op.drop_column('participants_heard')
        batch_op.drop_column('required_followup')
        batch_op.drop_column('resolution_summary')
        batch_op.drop_column('resolution_kind')
