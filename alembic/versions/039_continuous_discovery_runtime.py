"""Continuous discovery runtime lineage.

Revision ID: 039
Revises: 038
Create Date: 2026-09-07
"""
# pylint: disable=invalid-name,no-member
import sqlalchemy as sa
from alembic import op

revision = "039"
down_revision = "038"
branch_labels = None
depends_on = None

RESERVATION_LINEAGE_V4 = (
    "(roadmap_item_id IS NOT NULL AND continuous_origin_key IS NULL AND discovery_run_id IS NULL "
    "AND child_goal_id IS NOT NULL) OR "
    "(roadmap_item_id IS NULL AND continuous_origin_key IS NOT NULL AND discovery_run_id IS NULL "
    "AND child_goal_id IS NOT NULL) OR "
    "(roadmap_item_id IS NULL AND continuous_origin_key IS NULL AND discovery_run_id IS NOT NULL "
    "AND child_goal_id IS NULL)"
)
RESERVATION_LINEAGE_V3 = (
    "(roadmap_item_id IS NOT NULL AND continuous_origin_key IS NULL) OR "
    "(roadmap_item_id IS NULL AND continuous_origin_key IS NOT NULL)"
)
RESERVATION_SETTLEMENT_V4 = (
    "(status = 'active' AND settled_at IS NULL AND settlement_reason IS NULL) OR "
    "(status = 'settled' AND settled_at IS NOT NULL AND "
    "settlement_reason IN ('completed', 'cancelled', 'needs_attention'))"
)
RESERVATION_SETTLEMENT_V3 = (
    "(status = 'active' AND settled_at IS NULL AND settlement_reason IS NULL) OR "
    "(status = 'settled' AND settled_at IS NOT NULL AND settlement_reason IN ('completed', 'cancelled'))"
)


def upgrade() -> None:
    op.create_table(
        "orchestration_continuous_candidates",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("parent_goal_id", sa.Uuid(), sa.ForeignKey("orchestration_goals.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("source_run_id", sa.Uuid(), sa.ForeignKey("orchestration_runs.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("origin_key", sa.String(255), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("snapshot", sa.JSON(), nullable=False),
        sa.Column("child_goal_id", sa.Uuid(), sa.ForeignKey("orchestration_goals.id", ondelete="RESTRICT"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("parent_goal_id", "origin_key", name="uq_orch_continuous_candidate_parent_origin"),
        sa.UniqueConstraint("source_run_id", "position", name="uq_orch_continuous_candidate_run_position"),
        sa.UniqueConstraint("child_goal_id", name="uq_orch_continuous_candidate_child"),
    )
    with op.batch_alter_table("orchestration_budget_reservations") as batch:
        batch.drop_constraint("ck_orch_budget_reservation_lineage", type_="check")
        batch.drop_constraint("ck_orch_budget_reservations_settlement", type_="check")
        batch.add_column(sa.Column("discovery_run_id", sa.Uuid(), nullable=True))
        batch.alter_column("child_goal_id", existing_type=sa.Uuid(), nullable=True)
        batch.create_foreign_key(
            "fk_orch_budget_reservations_discovery_run_id", "orchestration_runs",
            ["discovery_run_id"], ["id"], ondelete="RESTRICT",
        )
        batch.create_unique_constraint("uq_orch_budget_discovery_run", ["discovery_run_id"])
        batch.create_check_constraint("ck_orch_budget_reservation_lineage", RESERVATION_LINEAGE_V4)
        batch.create_check_constraint("ck_orch_budget_reservations_settlement", RESERVATION_SETTLEMENT_V4)


def downgrade() -> None:
    op.execute("DELETE FROM orchestration_budget_reservations WHERE discovery_run_id IS NOT NULL")
    op.drop_table("orchestration_continuous_candidates")
    with op.batch_alter_table("orchestration_budget_reservations") as batch:
        batch.drop_constraint("ck_orch_budget_reservation_lineage", type_="check")
        batch.drop_constraint("ck_orch_budget_reservations_settlement", type_="check")
        batch.drop_constraint("uq_orch_budget_discovery_run", type_="unique")
        batch.drop_constraint("fk_orch_budget_reservations_discovery_run_id", type_="foreignkey")
        batch.drop_column("discovery_run_id")
        batch.alter_column("child_goal_id", existing_type=sa.Uuid(), nullable=False)
        batch.create_check_constraint("ck_orch_budget_reservation_lineage", RESERVATION_LINEAGE_V3)
        batch.create_check_constraint("ck_orch_budget_reservations_settlement", RESERVATION_SETTLEMENT_V3)
