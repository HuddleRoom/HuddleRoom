"""Roadmap execution runtime.

Revision ID: 037
Revises: 036
Create Date: 2026-09-05
"""
# pylint: disable=invalid-name,no-member
from alembic import op
import sqlalchemy as sa

revision = "037"
down_revision = "036"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("orchestration_goals") as batch:
        batch.add_column(sa.Column("parent_goal_id", sa.Uuid(), nullable=True))
        batch.add_column(sa.Column("roadmap_version_id", sa.Uuid(), nullable=True))
        batch.add_column(sa.Column("roadmap_item_key", sa.String(80), nullable=True))
        batch.add_column(sa.Column("parent_contract_snapshot", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("goal_delta", sa.JSON(), nullable=True))
        batch.create_foreign_key("fk_orch_goals_parent_goal_id", "orchestration_goals", ["parent_goal_id"], ["id"], ondelete="RESTRICT")
        batch.create_check_constraint("ck_orch_goals_not_own_parent", "parent_goal_id IS NULL OR parent_goal_id != id")
        batch.create_check_constraint(
            "ck_orch_goals_child_lineage_complete",
            "(parent_goal_id IS NULL AND roadmap_version_id IS NULL AND roadmap_item_key IS NULL AND parent_contract_snapshot IS NULL AND goal_delta IS NULL) OR "
            "(parent_goal_id IS NOT NULL AND roadmap_version_id IS NOT NULL AND roadmap_item_key IS NOT NULL AND parent_contract_snapshot IS NOT NULL AND goal_delta IS NOT NULL)",
        )
        batch.create_unique_constraint("uq_orch_goals_parent_item_key", ["parent_goal_id", "roadmap_item_key"])
        batch.create_index("idx_orch_goals_parent_status", ["parent_goal_id", "status"])

    op.create_table(
        "orchestration_roadmap_versions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("goal_id", sa.Uuid(), sa.ForeignKey("orchestration_goals.id", ondelete="CASCADE"), nullable=False),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey("orchestration_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("plan_artifact_id", sa.Uuid(), sa.ForeignKey("artifacts.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("snapshot", sa.JSON(), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("approval_reference", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("goal_id", "version", name="uq_orch_roadmap_versions_goal_version"),
        sa.UniqueConstraint("goal_id", "fingerprint", name="uq_orch_roadmap_versions_goal_fingerprint"),
        sa.CheckConstraint("version > 0", name="ck_orch_roadmap_versions_positive_version"),
    )
    op.create_index("idx_orch_roadmap_versions_goal_created", "orchestration_roadmap_versions", ["goal_id", "created_at"])
    with op.batch_alter_table("orchestration_goals") as batch:
        batch.create_foreign_key("fk_orch_goals_roadmap_version_id", "orchestration_roadmap_versions", ["roadmap_version_id"], ["id"], ondelete="RESTRICT")

    op.create_table(
        "orchestration_roadmap_items",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("goal_id", sa.Uuid(), sa.ForeignKey("orchestration_goals.id", ondelete="CASCADE"), nullable=False),
        sa.Column("first_version_id", sa.Uuid(), sa.ForeignKey("orchestration_roadmap_versions.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("item_key", sa.String(80), nullable=False),
        sa.Column("unit_type", sa.String(20), nullable=False),
        sa.Column("item_snapshot", sa.JSON(), nullable=False),
        sa.Column("task_id", sa.Uuid(), sa.ForeignKey("tasks.id", ondelete="RESTRICT"), nullable=True),
        sa.Column("child_goal_id", sa.Uuid(), sa.ForeignKey("orchestration_goals.id", ondelete="RESTRICT"), nullable=True),
        sa.Column("gate_id", sa.Uuid(), sa.ForeignKey("orchestration_gates.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("goal_id", "item_key", name="uq_orch_roadmap_items_goal_key"),
        sa.UniqueConstraint("task_id", name="uq_orch_roadmap_items_task"),
        sa.UniqueConstraint("child_goal_id", name="uq_orch_roadmap_items_child_goal"),
        sa.CheckConstraint("unit_type IN ('task', 'goal')", name="ck_orch_roadmap_items_unit_type"),
        sa.CheckConstraint("(unit_type = 'task' AND task_id IS NOT NULL AND child_goal_id IS NULL) OR (unit_type = 'goal' AND child_goal_id IS NOT NULL AND task_id IS NULL)", name="ck_orch_roadmap_items_target_matches_type"),
    )
    op.create_table(
        "orchestration_budget_reservations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("parent_goal_id", sa.Uuid(), sa.ForeignKey("orchestration_goals.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("roadmap_item_id", sa.Uuid(), sa.ForeignKey("orchestration_roadmap_items.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("child_goal_id", sa.Uuid(), sa.ForeignKey("orchestration_goals.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("allocation", sa.JSON(), nullable=False),
        sa.Column("settled_spend", sa.JSON(), nullable=False),
        sa.Column("measurement_complete", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'active'")),
        sa.Column("settlement_reason", sa.String(20), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("settled_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("roadmap_item_id", name="uq_orch_budget_reservations_item"),
        sa.UniqueConstraint("child_goal_id", name="uq_orch_budget_reservations_child"),
        sa.CheckConstraint("status IN ('active', 'settled')", name="ck_orch_budget_reservations_status"),
        sa.CheckConstraint("(status = 'active' AND settled_at IS NULL AND settlement_reason IS NULL) OR (status = 'settled' AND settled_at IS NOT NULL AND settlement_reason IN ('completed', 'cancelled'))", name="ck_orch_budget_reservations_settlement"),
    )


def downgrade() -> None:
    op.drop_table("orchestration_budget_reservations")
    op.drop_table("orchestration_roadmap_items")
    with op.batch_alter_table("orchestration_goals") as batch:
        batch.drop_constraint("fk_orch_goals_roadmap_version_id", type_="foreignkey")
    op.drop_index("idx_orch_roadmap_versions_goal_created", table_name="orchestration_roadmap_versions")
    op.drop_table("orchestration_roadmap_versions")
    with op.batch_alter_table("orchestration_goals") as batch:
        batch.drop_index("idx_orch_goals_parent_status")
        batch.drop_constraint("uq_orch_goals_parent_item_key", type_="unique")
        batch.drop_constraint("ck_orch_goals_child_lineage_complete", type_="check")
        batch.drop_constraint("ck_orch_goals_not_own_parent", type_="check")
        batch.drop_constraint("fk_orch_goals_parent_goal_id", type_="foreignkey")
        batch.drop_column("goal_delta")
        batch.drop_column("parent_contract_snapshot")
        batch.drop_column("roadmap_item_key")
        batch.drop_column("roadmap_version_id")
        batch.drop_column("parent_goal_id")
