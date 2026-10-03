"""m4_meeting_engine

Revision ID: 006
Revises: 005
Create Date: 2026-05-08

"""
from alembic import op
import sqlalchemy as sa

revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'meetings',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('project_id', sa.Uuid(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('meeting_type', sa.String(), nullable=False),
        sa.Column('status', sa.String(), server_default='scheduled', nullable=False),
        sa.Column('turn_strategy', sa.String(), server_default='round_robin', nullable=False),
        sa.Column('deadlock_strategy', sa.String(), server_default='human_intervention', nullable=False),
        sa.Column('participant_agent_ids', sa.JSON(), nullable=False),
        sa.Column('participant_user_ids', sa.JSON(), nullable=False),
        sa.Column('max_duration_minutes', sa.Integer(), server_default=sa.text('30'), nullable=False),
        sa.Column('veto_window_hours', sa.Integer(), server_default=sa.text('24'), nullable=False),
        sa.Column('auto_start', sa.Boolean(), server_default=sa.text('true'), nullable=False),
        sa.Column('created_by_agent_id', sa.Uuid(), nullable=True),
        sa.Column('created_by_user_id', sa.Uuid(), nullable=True),
        sa.Column('created_by_trigger', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('trigger_reason', sa.String(), nullable=True),
        sa.Column('source_protocol_instance_id', sa.Uuid(), nullable=True),
        sa.Column('source_task_id', sa.Uuid(), nullable=True),
        sa.Column('timeout_task_id', sa.String(), nullable=True),
        sa.Column('participant_contexts', sa.JSON(), nullable=True),
        sa.Column('summary', sa.String(), nullable=True),
        sa.Column('is_partial', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('scheduled_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('preparing_started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('active_started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('concluding_started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('concluded_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('cancelled_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('cancelled_reason', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['source_task_id'], ['tasks.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('meetings', schema=None) as batch_op:
        batch_op.create_index('idx_meetings_project_status', ['project_id', 'status'])
        batch_op.create_index('idx_meetings_status_scheduled', ['status', 'scheduled_at'])
        batch_op.create_index('idx_meetings_source_task', ['source_task_id'])

    op.create_table(
        'meeting_agenda_items',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('meeting_id', sa.Uuid(), nullable=False),
        sa.Column('order', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('question', sa.String(), nullable=True),
        sa.Column('options', sa.JSON(), nullable=True),
        sa.Column('artifact_id', sa.Uuid(), nullable=True),
        sa.Column('artifact_url', sa.String(), nullable=True),
        sa.Column('turn_order', sa.JSON(), nullable=True),
        sa.Column('max_rounds', sa.Integer(), server_default=sa.text('3'), nullable=False),
        sa.Column('status', sa.String(), server_default='pending', nullable=False),
        sa.Column('is_deadlocked', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('current_round', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('consensus_check_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('requires_approval', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('creates_protocol', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['meeting_id'], ['meetings.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('meeting_agenda_items', schema=None) as batch_op:
        batch_op.create_index('idx_mai_meeting_order', ['meeting_id', 'order'])
        batch_op.create_index('idx_mai_meeting_status', ['meeting_id', 'status'])

    op.create_table(
        'meeting_turns',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('meeting_id', sa.Uuid(), nullable=False),
        sa.Column('agenda_item_id', sa.Uuid(), nullable=True),
        sa.Column('turn_number', sa.Integer(), nullable=False),
        sa.Column('round_number', sa.Integer(), server_default=sa.text('1'), nullable=False),
        sa.Column('speaker_agent_id', sa.Uuid(), nullable=True),
        sa.Column('speaker_user_id', sa.Uuid(), nullable=True),
        sa.Column('content', sa.String(), nullable=False),
        sa.Column('references', sa.JSON(), nullable=False),
        sa.Column('is_human_turn', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('is_override', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('moderator_note', sa.String(), nullable=True),
        sa.Column('token_count', sa.Integer(), nullable=True),
        sa.Column('model_used', sa.String(), nullable=True),
        sa.Column('provider_used', sa.String(), nullable=True),
        sa.Column('latency_ms', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['agenda_item_id'], ['meeting_agenda_items.id']),
        sa.ForeignKeyConstraint(['meeting_id'], ['meetings.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('meeting_turns', schema=None) as batch_op:
        batch_op.create_index('idx_mt_meeting_turn', ['meeting_id', 'turn_number'])
        batch_op.create_index('idx_mt_meeting_item_round', ['meeting_id', 'agenda_item_id', 'round_number'])
        batch_op.create_index('idx_mt_speaker_agent', ['speaker_agent_id'])

    op.create_table(
        'meeting_decisions',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('meeting_id', sa.Uuid(), nullable=False),
        sa.Column('agenda_item_id', sa.Uuid(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('question', sa.String(), nullable=True),
        sa.Column('chosen_option', sa.String(), nullable=False),
        sa.Column('rationale', sa.String(), nullable=False),
        sa.Column('alternatives_rejected', sa.JSON(), nullable=False),
        sa.Column('participants_agreed', sa.JSON(), nullable=False),
        sa.Column('dissent', sa.JSON(), nullable=False),
        sa.Column('decided_by', sa.String(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('is_partial', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('is_vetoed', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('veto_reason', sa.String(), nullable=True),
        sa.Column('vetoed_by_user_id', sa.Uuid(), nullable=True),
        sa.Column('vetoed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('knowledge_item_id', sa.Uuid(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['agenda_item_id'], ['meeting_agenda_items.id']),
        sa.ForeignKeyConstraint(['meeting_id'], ['meetings.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('meeting_decisions', schema=None) as batch_op:
        batch_op.create_index('idx_md_meeting', ['meeting_id'])
        batch_op.create_index('idx_md_agenda_item', ['agenda_item_id'])
        batch_op.create_index('idx_md_vetoed', ['is_vetoed'])

    op.create_table(
        'meeting_action_items',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('meeting_id', sa.Uuid(), nullable=False),
        sa.Column('depends_on_decision_id', sa.Uuid(), nullable=True),
        sa.Column('description', sa.String(), nullable=False),
        sa.Column('assignee_agent_id', sa.Uuid(), nullable=True),
        sa.Column('assignee_user_id', sa.Uuid(), nullable=True),
        sa.Column('priority', sa.Integer(), server_default=sa.text('70'), nullable=False),
        sa.Column('deadline_days', sa.Integer(), nullable=True),
        sa.Column('deadline_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(), server_default='open', nullable=False),
        sa.Column('task_id', sa.Uuid(), nullable=True),
        sa.Column('creates_protocol', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('protocol_instance_id', sa.Uuid(), nullable=True),
        sa.Column('is_partial', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['depends_on_decision_id'], ['meeting_decisions.id']),
        sa.ForeignKeyConstraint(['meeting_id'], ['meetings.id']),
        sa.ForeignKeyConstraint(['task_id'], ['tasks.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('meeting_action_items', schema=None) as batch_op:
        batch_op.create_index('idx_mact_meeting', ['meeting_id'])
        batch_op.create_index('idx_mact_assignee_status', ['assignee_agent_id', 'status'])
        batch_op.create_index('idx_mact_task', ['task_id'])

    op.create_table(
        'meeting_events',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('meeting_id', sa.Uuid(), nullable=False),
        sa.Column('event_type', sa.String(), nullable=False),
        sa.Column('payload', sa.JSON(), nullable=True),
        sa.Column('actor_agent_id', sa.Uuid(), nullable=True),
        sa.Column('actor_user_id', sa.Uuid(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['meeting_id'], ['meetings.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('meeting_events', schema=None) as batch_op:
        batch_op.create_index('idx_mev_meeting', ['meeting_id'])

    with op.batch_alter_table('tasks', schema=None) as batch_op:
        batch_op.add_column(sa.Column('created_by_meeting_id', sa.Uuid(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('tasks', schema=None) as batch_op:
        batch_op.drop_column('created_by_meeting_id')

    op.drop_table('meeting_events')
    op.drop_table('meeting_action_items')
    op.drop_table('meeting_decisions')
    op.drop_table('meeting_turns')
    op.drop_table('meeting_agenda_items')
    op.drop_table('meetings')
