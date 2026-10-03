"""turn_reasoning

Revision ID: 011
Revises: 010
Create Date: 2026-05-18

"""
from alembic import op
import sqlalchemy as sa

revision = '011'
down_revision = '010'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('meeting_turns') as batch_op:
        batch_op.add_column(sa.Column('reasoning_content', sa.String(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('meeting_turns') as batch_op:
        batch_op.drop_column('reasoning_content')
