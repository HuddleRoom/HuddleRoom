"""cli_meeting_turns

Revision ID: 012
Revises: 011
Create Date: 2026-05-20

"""
from alembic import op
import sqlalchemy as sa

revision = '012'
down_revision = '011'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('meeting_turns') as batch_op:
        batch_op.add_column(sa.Column('cli_session_id', sa.String(), nullable=True))
        batch_op.create_index('ix_meeting_turns_cli_session_id', ['cli_session_id'])


def downgrade() -> None:
    with op.batch_alter_table('meeting_turns') as batch_op:
        batch_op.drop_index('ix_meeting_turns_cli_session_id')
        batch_op.drop_column('cli_session_id')
