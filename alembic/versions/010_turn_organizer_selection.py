"""turn_organizer_selection

Revision ID: 010
Revises: 009
Create Date: 2026-05-18

"""
from alembic import op
import sqlalchemy as sa

revision = '010'
down_revision = '009'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('meeting_turns') as batch_op:
        batch_op.add_column(sa.Column('organizer_selection', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('meeting_turns') as batch_op:
        batch_op.drop_column('organizer_selection')
