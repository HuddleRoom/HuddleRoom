"""m4_extensions_organizer_signals_planner

Revision ID: 007
Revises: 006
Create Date: 2026-05-12

"""
from alembic import op
import sqlalchemy as sa

revision = '007'
down_revision = '006'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('meetings') as batch_op:
        batch_op.add_column(sa.Column('signal_check_enabled', sa.Boolean(), server_default=sa.text('false'), nullable=False))
        batch_op.add_column(sa.Column('organizer_agent_id', sa.Uuid(), nullable=True))
        batch_op.add_column(sa.Column('organizer_user_id', sa.Uuid(), nullable=True))
        batch_op.add_column(sa.Column('pending_grant_agent_id', sa.Uuid(), nullable=True))
        batch_op.add_column(sa.Column('planner_agent_id', sa.Uuid(), nullable=True))
        batch_op.add_column(sa.Column('planner_summary', sa.String(), nullable=True))

    op.create_table(
        'meeting_participant_signals',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('meeting_id', sa.Uuid(), nullable=False),
        sa.Column('agent_id', sa.Uuid(), nullable=False),
        sa.Column('signal_type', sa.String(), nullable=False),
        sa.Column('message', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('acknowledged_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['meeting_id'], ['meetings.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_mps_meeting_ack', 'meeting_participant_signals', ['meeting_id', 'acknowledged_at'])

    op.create_table(
        'meeting_requests',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('requesting_agent_id', sa.Uuid(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('reason', sa.String(), nullable=False),
        sa.Column('meeting_type', sa.String(), server_default='decision', nullable=False),
        sa.Column('suggested_participant_agent_ids', sa.JSON(), nullable=False),
        sa.Column('status', sa.String(), server_default='pending_approval', nullable=False),
        sa.Column('approved_by_user_id', sa.Uuid(), nullable=True),
        sa.Column('approved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_meeting_id', sa.Uuid(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['created_meeting_id'], ['meetings.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_mreq_project_status', 'meeting_requests', ['project_id', 'status'])


def downgrade() -> None:
    op.drop_index('idx_mreq_project_status', table_name='meeting_requests')
    op.drop_table('meeting_requests')
    op.drop_index('idx_mps_meeting_ack', table_name='meeting_participant_signals')
    op.drop_table('meeting_participant_signals')

    with op.batch_alter_table('meetings') as batch_op:
        batch_op.drop_column('planner_summary')
        batch_op.drop_column('planner_agent_id')
        batch_op.drop_column('pending_grant_agent_id')
        batch_op.drop_column('organizer_user_id')
        batch_op.drop_column('organizer_agent_id')
        batch_op.drop_column('signal_check_enabled')
