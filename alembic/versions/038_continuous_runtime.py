"""Continuous direct cycle runtime.

Revision ID: 038
Revises: 037
Create Date: 2026-09-06
"""
# pylint: disable=invalid-name,no-member
import sqlalchemy as sa
from alembic import op

revision = "038"
down_revision = "037"
branch_labels = None
depends_on = None

CHILD_LINEAGE_V3 = (
    "(parent_goal_id IS NULL AND roadmap_version_id IS NULL AND roadmap_item_key IS NULL "
    "AND continuous_origin_key IS NULL AND parent_contract_snapshot IS NULL AND goal_delta IS NULL) OR "
    "(parent_goal_id IS NOT NULL AND roadmap_version_id IS NOT NULL AND roadmap_item_key IS NOT NULL "
    "AND continuous_origin_key IS NULL AND parent_contract_snapshot IS NOT NULL AND goal_delta IS NOT NULL) OR "
    "(parent_goal_id IS NOT NULL AND roadmap_version_id IS NULL AND roadmap_item_key IS NULL "
    "AND continuous_origin_key IS NOT NULL AND parent_contract_snapshot IS NOT NULL AND goal_delta IS NOT NULL)"
)
CHILD_LINEAGE_V2 = (
    "(parent_goal_id IS NULL AND roadmap_version_id IS NULL AND roadmap_item_key IS NULL "
    "AND parent_contract_snapshot IS NULL AND goal_delta IS NULL) OR "
    "(parent_goal_id IS NOT NULL AND roadmap_version_id IS NOT NULL AND roadmap_item_key IS NOT NULL "
    "AND parent_contract_snapshot IS NOT NULL AND goal_delta IS NOT NULL)"
)


def upgrade() -> None:
    with op.batch_alter_table("orchestration_goals") as batch:
        batch.drop_constraint("ck_orch_goals_child_lineage_complete", type_="check")
        batch.add_column(sa.Column("continuous_policy", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("continuous_state", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("continuous_origin_key", sa.String(255), nullable=True))
        batch.create_unique_constraint(
            "uq_orch_goals_parent_origin_key", ["parent_goal_id", "continuous_origin_key"]
        )
        batch.create_check_constraint("ck_orch_goals_child_lineage_complete", CHILD_LINEAGE_V3)

    with op.batch_alter_table("orchestration_runs") as batch:
        batch.drop_constraint("ck_orchestration_runs_phase", type_="check")
        batch.add_column(sa.Column("cycle_key", sa.String(255), nullable=True))
        batch.create_unique_constraint("uq_orch_runs_goal_cycle_key", ["goal_id", "cycle_key"])
        batch.create_check_constraint(
            "ck_orchestration_runs_phase",
            "phase IN ('baseline', 'ready', 'waiting_activation', 'authorized', 'completed')",
        )

    with op.batch_alter_table("orchestration_budget_reservations") as batch:
        batch.alter_column("roadmap_item_id", existing_type=sa.Uuid(), nullable=True)
        batch.add_column(sa.Column("continuous_origin_key", sa.String(255), nullable=True))
        batch.create_unique_constraint(
            "uq_orch_budget_parent_origin", ["parent_goal_id", "continuous_origin_key"]
        )
        batch.create_check_constraint(
            "ck_orch_budget_reservation_lineage",
            "(roadmap_item_id IS NOT NULL AND continuous_origin_key IS NULL) OR "
            "(roadmap_item_id IS NULL AND continuous_origin_key IS NOT NULL)",
        )


def downgrade() -> None:
    op.execute("DELETE FROM orchestration_budget_reservations WHERE continuous_origin_key IS NOT NULL")
    op.execute("DELETE FROM orchestration_goals WHERE continuous_origin_key IS NOT NULL")
    op.execute("DELETE FROM orchestration_runs WHERE cycle_key IS NOT NULL OR phase = 'waiting_activation'")
    with op.batch_alter_table("orchestration_budget_reservations") as batch:
        batch.drop_constraint("ck_orch_budget_reservation_lineage", type_="check")
        batch.drop_constraint("uq_orch_budget_parent_origin", type_="unique")
        batch.drop_column("continuous_origin_key")
        batch.alter_column("roadmap_item_id", existing_type=sa.Uuid(), nullable=False)

    with op.batch_alter_table("orchestration_runs") as batch:
        batch.drop_constraint("ck_orchestration_runs_phase", type_="check")
        batch.drop_constraint("uq_orch_runs_goal_cycle_key", type_="unique")
        batch.drop_column("cycle_key")
        batch.create_check_constraint(
            "ck_orchestration_runs_phase",
            "phase IN ('baseline', 'ready', 'authorized', 'completed')",
        )

    with op.batch_alter_table("orchestration_goals") as batch:
        batch.drop_constraint("ck_orch_goals_child_lineage_complete", type_="check")
        batch.drop_constraint("uq_orch_goals_parent_origin_key", type_="unique")
        batch.drop_column("continuous_origin_key")
        batch.drop_column("continuous_state")
        batch.drop_column("continuous_policy")
        batch.create_check_constraint("ck_orch_goals_child_lineage_complete", CHILD_LINEAGE_V2)
