"""m6_stub_backends

Revision ID: 014
Revises: 013
Create Date: 2026-06-08

"""
from alembic import op
import sqlalchemy as sa

revision = '014'
down_revision = '013'
branch_labels = None
depends_on = None

import uuid as _uuid_mod
_ANON_USER_ID = _uuid_mod.UUID('00000000-0000-0000-0000-000000000000')


def upgrade() -> None:
    dialect = op.get_bind().dialect.name
    bool_true = sa.text("true") if dialect == "postgresql" else sa.text("1")
    bool_false = sa.text("false") if dialect == "postgresql" else sa.text("0")

    # --- routing_rules ---
    op.create_table(
        'routing_rules',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('project_id', sa.Uuid(), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('priority', sa.Integer(), nullable=False, server_default=sa.text('0')),
        sa.Column('on_event', sa.String(), nullable=False),
        sa.Column('conditions', sa.JSON(), nullable=False, server_default='{}'),
        sa.Column('actions', sa.JSON(), nullable=False, server_default='{}'),
        sa.Column('enabled', sa.Boolean(), nullable=False, server_default=bool_true),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_rules_project_priority', 'routing_rules', ['project_id', 'priority'])
    op.create_index('idx_rules_project_enabled', 'routing_rules', ['project_id', 'enabled'])

    # --- hooks ---
    op.create_table(
        'hooks',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('project_id', sa.Uuid(), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('code', sa.Text(), nullable=False),
        sa.Column('status', sa.String(), nullable=False, server_default=sa.text("'proposed'")),
        sa.Column('trigger_event', sa.String(), nullable=False),
        sa.Column('execution_count', sa.Integer(), nullable=False, server_default=sa.text('0')),
        sa.Column('error_count', sa.Integer(), nullable=False, server_default=sa.text('0')),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint("status IN ('proposed', 'active', 'shadow', 'disabled')", name='ck_hooks_status'),
    )
    op.create_index('idx_hooks_project_status', 'hooks', ['project_id', 'status'])

    # --- patterns ---
    op.create_table(
        'patterns',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('project_id', sa.Uuid(), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('pattern_type', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('sample_size', sa.Integer(), nullable=False),
        sa.Column('context', sa.JSON(), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_patterns_project', 'patterns', ['project_id'])

    # --- optimizations ---
    op.create_table(
        'optimizations',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('project_id', sa.Uuid(), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('pattern_id', sa.Uuid(), sa.ForeignKey('patterns.id', ondelete='SET NULL'), nullable=True),
        sa.Column('type', sa.String(), nullable=False),
        sa.Column('generated_code', sa.Text(), nullable=False),
        sa.Column('status', sa.String(), nullable=False, server_default=sa.text("'proposed'")),
        sa.Column('error_rate', sa.Float(), nullable=False, server_default=sa.text('0.0')),
        sa.Column('fire_count', sa.Integer(), nullable=False, server_default=sa.text('0')),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint("type IN ('hook', 'rule', 'shortcut')", name='ck_optimizations_type'),
        sa.CheckConstraint(
            "status IN ('proposed', 'requires_approval', 'approved', 'active', 'shadow', 'disabled', 'rejected')",
            name='ck_optimizations_status',
        ),
    )
    op.create_index('idx_optimizations_project_status', 'optimizations', ['project_id', 'status'])

    # --- cost_metrics ---
    op.create_table(
        'cost_metrics',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('project_id', sa.Uuid(), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('optimization_id', sa.Uuid(), sa.ForeignKey('optimizations.id', ondelete='SET NULL'), nullable=True),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('llm_calls_saved', sa.Integer(), nullable=False, server_default=sa.text('0')),
        sa.Column('estimated_cost_saved_usd', sa.Float(), nullable=False, server_default=sa.text('0.0')),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_cost_metrics_project_date', 'cost_metrics', ['project_id', 'date'])

    # --- Seed anon user for auth-disabled mode ---
    if dialect == "postgresql":
        op.execute(sa.text(
            "INSERT INTO users (id, email, hashed_password, display_name, role, is_active, created_at, updated_at) "
            "VALUES (:id, 'anon@local', '$2b$12$roncFJiobp.SJUvZ5w/AI.d2sBUhQaQV/5C1Vjqsm7IqCPVIPQakS', 'Anonymous', 'admin', true, NOW(), NOW()) "
            "ON CONFLICT (id) DO NOTHING"
        ).bindparams(sa.bindparam("id", value=_ANON_USER_ID, type_=sa.Uuid())))
    else:
        op.execute(sa.text(
            "INSERT OR IGNORE INTO users (id, email, hashed_password, display_name, role, is_active, created_at, updated_at) "
            "VALUES (:id, 'anon@local', '$2b$12$roncFJiobp.SJUvZ5w/AI.d2sBUhQaQV/5C1Vjqsm7IqCPVIPQakS', 'Anonymous', 'admin', 1, datetime('now'), datetime('now'))"
        ).bindparams(sa.bindparam("id", value=_ANON_USER_ID, type_=sa.Uuid())))


def downgrade() -> None:
    op.drop_index('idx_cost_metrics_project_date', table_name='cost_metrics')
    op.drop_table('cost_metrics')
    op.drop_index('idx_optimizations_project_status', table_name='optimizations')
    op.drop_table('optimizations')
    op.drop_index('idx_patterns_project', table_name='patterns')
    op.drop_table('patterns')
    op.drop_index('idx_hooks_project_status', table_name='hooks')
    op.drop_table('hooks')
    op.drop_index('idx_rules_project_priority', table_name='routing_rules')
    op.drop_index('idx_rules_project_enabled', table_name='routing_rules')
    op.drop_table('routing_rules')
    # Do not remove anon user — may have FK references
