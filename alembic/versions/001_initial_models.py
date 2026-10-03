"""initial models

Revision ID: 001
Revises:
Create Date: 2026-04-24

"""
from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    dialect = op.get_bind().dialect.name

    # PostgreSQL-only: enable vector extension for pgvector support
    if dialect == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # helpers for dialect-specific defaults
    bool_true = sa.text("true") if dialect == "postgresql" else sa.text("1")
    bool_false = sa.text("false") if dialect == "postgresql" else sa.text("0")
    empty_json = sa.text("'{}'")
    empty_list = sa.text("'[]'")
    uuid_default = sa.text("gen_random_uuid()") if dialect == "postgresql" else None

    # users (no deps)
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), primary_key=True, server_default=uuid_default),
        sa.Column("email", sa.String(), unique=True, nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=True),
        sa.Column("role", sa.String(), nullable=False, server_default="member"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=bool_true),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )

    # projects (no deps)
    op.create_table(
        "projects",
        sa.Column("id", sa.Uuid(), primary_key=True, server_default=uuid_default),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("workspace_path", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="active"),
        sa.Column("config", sa.JSON(), nullable=False, server_default=empty_json),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )

    # agents (no deps)
    op.create_table(
        "agents",
        sa.Column("id", sa.Uuid(), primary_key=True, server_default=uuid_default),
        sa.Column("name", sa.String(), unique=True, nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("system_prompt", sa.String(), nullable=True),
        sa.Column("adapter_type", sa.String(), nullable=False, server_default="api"),
        sa.Column("cli_runtime", sa.String(), nullable=True),
        sa.Column("capabilities", sa.JSON(), nullable=False, server_default=empty_list),
        sa.Column("config", sa.JSON(), nullable=False, server_default=empty_json),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=bool_true),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("idx_agents_role", "agents", ["role"])
    op.create_index("idx_agents_is_active", "agents", ["is_active"])

    # api_keys (depends on users, agents, projects)
    op.create_table(
        "api_keys",
        sa.Column("id", sa.Uuid(), primary_key=True, server_default=uuid_default),
        sa.Column("agent_id", sa.Uuid(), sa.ForeignKey("agents.id", ondelete="CASCADE"), nullable=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=True),
        sa.Column("project_id", sa.Uuid(), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=True),
        sa.Column("key_prefix", sa.String(), nullable=False),
        sa.Column("hashed_key", sa.String(), unique=True, nullable=False),
        sa.Column("label", sa.String(), nullable=True),
        sa.Column("last_used_at", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint("agent_id IS NOT NULL OR user_id IS NOT NULL", name="ck_api_keys_owner"),
    )
    op.create_index("idx_api_keys_hashed_key", "api_keys", ["hashed_key"])
    op.create_index("idx_api_keys_agent_id", "api_keys", ["agent_id"])

    # tasks (depends on projects, agents, users)
    op.create_table(
        "tasks",
        sa.Column("id", sa.Uuid(), primary_key=True, server_default=uuid_default),
        sa.Column("project_id", sa.Uuid(), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("parent_id", sa.Uuid(), sa.ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True),
        sa.Column("protocol_instance_id", sa.Uuid(), nullable=True),   # FK deferred to M3
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="backlog"),
        sa.Column("priority", sa.Integer(), nullable=False, server_default=sa.text("50")),
        sa.Column("assigned_to", sa.Uuid(), sa.ForeignKey("agents.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_by_agent", sa.Uuid(), sa.ForeignKey("agents.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_by_user", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("adapter_type_override", sa.String(), nullable=True),
        sa.Column("depends_on", sa.JSON(), nullable=True),
        sa.Column("trigger", sa.JSON(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default=empty_json),
        sa.Column("due_at", sa.DateTime(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("idx_tasks_project_status", "tasks", ["project_id", "status"])
    op.create_index("idx_tasks_assigned_to", "tasks", ["assigned_to"])
    op.create_index("idx_tasks_parent_id", "tasks", ["parent_id"])
    op.create_index("idx_tasks_protocol_instance", "tasks", ["protocol_instance_id"])

    # sessions (depends on tasks, agents, projects)
    op.create_table(
        "sessions",
        sa.Column("id", sa.Uuid(), primary_key=True, server_default=uuid_default),
        sa.Column("task_id", sa.Uuid(), sa.ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True),
        sa.Column("agent_id", sa.Uuid(), sa.ForeignKey("agents.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("project_id", sa.Uuid(), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("protocol_instance_id", sa.Uuid(), nullable=True),   # FK deferred to M3
        sa.Column("meeting_id", sa.Uuid(), nullable=True),   # FK deferred to M4
        sa.Column("adapter_type", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("input_context", sa.JSON(), nullable=False, server_default=empty_json),
        sa.Column("output", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("celery_task_id", sa.String(), nullable=True),
        sa.Column("sandbox_path", sa.String(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default=empty_json),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("idx_sessions_task_id", "sessions", ["task_id"])
    op.create_index("idx_sessions_agent_id", "sessions", ["agent_id"])
    op.create_index("idx_sessions_project_status", "sessions", ["project_id", "status"])
    op.create_index("idx_sessions_celery_task_id", "sessions", ["celery_task_id"])
    op.create_index("idx_sessions_meeting_id", "sessions", ["meeting_id"])

    # knowledge_items (depends on projects, sessions, agents, users)
    op.create_table(
        "knowledge_items",
        sa.Column("id", sa.Uuid(), primary_key=True, server_default=uuid_default),
        sa.Column("project_id", sa.Uuid(), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=True),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_type", sa.String(), nullable=False),
        sa.Column("tags", sa.JSON(), nullable=True),
        # embedding: replaced Vector(1536) with JSON for SQLite compatibility
        sa.Column("embedding", sa.JSON(), nullable=True),
        sa.Column("provenance_type", sa.String(), nullable=False, server_default="human"),
        sa.Column("provenance_session_id", sa.Uuid(), sa.ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True),
        sa.Column("provenance_meeting_id", sa.Uuid(), nullable=True),   # FK deferred to M4
        sa.Column("provenance_protocol_instance_id", sa.Uuid(), nullable=True),   # FK deferred to M3
        sa.Column("created_by_agent", sa.Uuid(), sa.ForeignKey("agents.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_by_user", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("supersedes", sa.JSON(), nullable=True),
        sa.Column("is_superseded", sa.Boolean(), nullable=False, server_default=bool_false),
        sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("conflict_status", sa.String(), nullable=False, server_default="none"),
        sa.Column("conflicting_item_ids", sa.JSON(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default=empty_json),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("idx_knowledge_project_id", "knowledge_items", ["project_id"])
    op.create_index("idx_knowledge_content_type", "knowledge_items", ["content_type"])
    op.create_index("idx_knowledge_is_superseded", "knowledge_items", ["is_superseded"])
    op.create_index("idx_knowledge_conflict_status", "knowledge_items", ["conflict_status"])

    # channels (depends on projects, tasks)
    op.create_table(
        "channels",
        sa.Column("id", sa.Uuid(), primary_key=True, server_default=uuid_default),
        sa.Column("project_id", sa.Uuid(), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("channel_type", sa.String(), nullable=False),
        sa.Column("task_id", sa.Uuid(), sa.ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True),
        sa.Column("members", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("idx_channels_project_id", "channels", ["project_id"])
    op.create_index("idx_channels_task_id", "channels", ["task_id"])

    # messages (depends on channels, agents, users)
    op.create_table(
        "messages",
        sa.Column("id", sa.Uuid(), primary_key=True, server_default=uuid_default),
        sa.Column("channel_id", sa.Uuid(), sa.ForeignKey("channels.id", ondelete="CASCADE"), nullable=False),
        sa.Column("sender_agent_id", sa.Uuid(), sa.ForeignKey("agents.id", ondelete="SET NULL"), nullable=True),
        sa.Column("sender_user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("message_type", sa.String(), nullable=False, server_default="text"),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default=empty_json),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint(
            "sender_agent_id IS NOT NULL OR sender_user_id IS NOT NULL",
            name="ck_messages_sender",
        ),
    )
    op.create_index("idx_messages_channel_created", "messages", ["channel_id", "created_at"])
    op.create_index("idx_messages_sender_agent", "messages", ["sender_agent_id"])


def downgrade() -> None:
    op.drop_table("messages")
    op.drop_table("channels")
    op.drop_table("knowledge_items")
    op.drop_table("sessions")
    op.drop_table("tasks")
    op.drop_table("api_keys")
    op.drop_table("agents")
    op.drop_table("projects")
    op.drop_table("users")
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP EXTENSION IF EXISTS vector")
