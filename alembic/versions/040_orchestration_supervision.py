"""Persist durable orchestration supervision facts.

Revision ID: 040
Revises: 039
Create Date: 2026-09-09
"""
# pylint: disable=invalid-name,no-member
import sqlalchemy as sa
from alembic import op

revision = "040"
down_revision = "039"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "orchestration_waits",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey("orchestration_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("wait_key", sa.String(255), nullable=False),
        sa.Column("owner", sa.JSON(), nullable=False),
        sa.Column("awaited_event", sa.JSON(), nullable=False),
        sa.Column("due_recheck_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fallback", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="open"),
        sa.Column("cleared_by_event_id", sa.Uuid(), sa.ForeignKey("event_log.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("cleared_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("status IN ('open', 'cleared')", name="ck_orchestration_waits_status"),
    )
    op.create_index(
        "uq_orch_waits_run_key_open",
        "orchestration_waits",
        ["run_id", "wait_key"],
        unique=True,
        sqlite_where=sa.text("status = 'open'"),
        postgresql_where=sa.text("status = 'open'"),
    )
    op.create_index("idx_orch_waits_due", "orchestration_waits", ["status", "due_recheck_at"])
    op.create_table(
        "orchestration_scheduler_state",
        sa.Column("name", sa.String(50), primary_key=True, server_default="supervision"),
        sa.Column("cursor_goal_id", sa.Uuid(), nullable=True),
        sa.Column("pass_key", sa.String(100), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("name = 'supervision'", name="ck_orch_scheduler_state_name"),
        sa.UniqueConstraint("pass_key", name="uq_orch_scheduler_state_pass_key"),
    )

    with op.batch_alter_table("orchestration_runs") as batch:
        batch.add_column(sa.Column("supervision_state", sa.JSON(), nullable=False, server_default=sa.text("'{}'")))
    with op.batch_alter_table("orchestration_actions") as batch:
        batch.add_column(sa.Column("dispatch_contract", sa.JSON(), nullable=False, server_default=sa.text("'{}'")))
        batch.add_column(sa.Column("budget_ledger", sa.JSON(), nullable=False, server_default=sa.text("'{}'")))
    with op.batch_alter_table("orchestration_authority_decisions") as batch:
        batch.add_column(sa.Column("runtime_identity", sa.String(64), nullable=True))
        batch.add_column(sa.Column("contract_version", sa.String(255), nullable=True))
        batch.add_column(sa.Column("continuation", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("continuation_action_id", sa.Uuid(), nullable=True))
        batch.add_column(sa.Column("continuation_applied_at", sa.DateTime(timezone=True), nullable=True))
        batch.create_unique_constraint("uq_orch_authority_decisions_runtime_identity", ["runtime_identity"])
        batch.create_foreign_key(
            "fk_orch_authority_decisions_continuation_action_id",
            "orchestration_actions",
            ["continuation_action_id"],
            ["id"],
            ondelete="SET NULL",
        )
    with op.batch_alter_table("orchestration_memory_sections") as batch:
        batch.add_column(sa.Column("fact_status", sa.String(20), nullable=False, server_default="unverified"))
        batch.add_column(sa.Column("provenance", sa.JSON(), nullable=False, server_default=sa.text("'{}'")))
        batch.create_check_constraint(
            "ck_orch_memory_sections_fact_status",
            "fact_status IN ('unverified', 'accepted', 'superseded')",
        )


def downgrade() -> None:
    with op.batch_alter_table("orchestration_memory_sections") as batch:
        batch.drop_constraint("ck_orch_memory_sections_fact_status", type_="check")
        batch.drop_column("provenance")
        batch.drop_column("fact_status")
    with op.batch_alter_table("orchestration_authority_decisions") as batch:
        batch.drop_constraint("fk_orch_authority_decisions_continuation_action_id", type_="foreignkey")
        batch.drop_constraint("uq_orch_authority_decisions_runtime_identity", type_="unique")
        batch.drop_column("continuation_applied_at")
        batch.drop_column("continuation_action_id")
        batch.drop_column("continuation")
        batch.drop_column("contract_version")
        batch.drop_column("runtime_identity")
    with op.batch_alter_table("orchestration_actions") as batch:
        batch.drop_column("budget_ledger")
        batch.drop_column("dispatch_contract")
    with op.batch_alter_table("orchestration_runs") as batch:
        batch.drop_column("supervision_state")
    op.drop_table("orchestration_scheduler_state")
    op.drop_index("idx_orch_waits_due", table_name="orchestration_waits")
    op.drop_index("uq_orch_waits_run_key_open", table_name="orchestration_waits")
    op.drop_table("orchestration_waits")
