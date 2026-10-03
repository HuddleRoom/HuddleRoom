"""memory_items

Revision ID: 013
Revises: 012
Create Date: 2026-06-02

"""
from alembic import op
import sqlalchemy as sa

revision = '013'
down_revision = '012b'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'memory_items',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('agent_id', sa.Uuid(), sa.ForeignKey('agents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('project_id', sa.Uuid(), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=True),
        sa.Column('scope', sa.String(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('tags', sa.JSON(), nullable=False, server_default='[]'),
        sa.Column('embedding', sa.JSON(), nullable=True),
        sa.Column('shared', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_memory_agent_id', 'memory_items', ['agent_id'])
    op.create_index('idx_memory_project_id', 'memory_items', ['project_id'])
    op.create_index('idx_memory_scope', 'memory_items', ['scope'])
    op.create_index('idx_memory_shared', 'memory_items', ['shared'])


def downgrade() -> None:
    op.drop_index('idx_memory_shared', table_name='memory_items')
    op.drop_index('idx_memory_scope', table_name='memory_items')
    op.drop_index('idx_memory_project_id', table_name='memory_items')
    op.drop_index('idx_memory_agent_id', table_name='memory_items')
    op.drop_table('memory_items')
