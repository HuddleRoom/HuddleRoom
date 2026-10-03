"""turn_debug_fields

Revision ID: 009
Revises: 008
Create Date: 2026-05-18

"""
from alembic import op
import sqlalchemy as sa

revision = '009'
down_revision = '008'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('meeting_turns') as batch_op:
        batch_op.add_column(sa.Column('prompt_messages', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('raw_response', sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('meeting_turns') as batch_op:
        batch_op.drop_column('raw_response')
        batch_op.drop_column('prompt_messages')
