"""Goal-shape provenance on goals and successor-run lineage on runs.

Revision ID: 049
Revises: 048
Create Date: 2026-10-09

"""
# pylint: disable=invalid-name,no-member
from alembic import op
import sqlalchemy as sa

revision = "049"
down_revision = "048"
branch_labels = None
depends_on = None

GOAL_TYPE_RATIONALE = "Existing goal type preserved by the planning-hierarchy migration."
PLANNING_RATIONALE = "Planning mode mapped from the existing goal type by the planning-hierarchy migration."

GOAL_CHECKS = {
    "ck_orch_goals_goal_type_source": (
        "goal_type_source IN ('auto', 'user', 'parent_auto', 'parent_explicit', 'legacy')"
    ),
    "ck_orch_goals_planning_mode": (
        "planning_mode IN ('roadmap', 'delivery', 'implementation', 'continuous_policy')"
    ),
    "ck_orch_goals_type_rationale_nonblank": "length(trim(goal_type_rationale)) > 0",
    "ck_orch_goals_planning_rationale_nonblank": "length(trim(planning_rationale)) > 0",
    "ck_orch_goals_planning_mode_fits_type": (
        "(goal_type = 'roadmap' AND planning_mode = 'roadmap') OR "
        "(goal_type = 'continuous' AND planning_mode = 'continuous_policy') OR "
        "(goal_type = 'outcome' AND planning_mode IN ('delivery', 'implementation'))"
    ),
}
RUN_CHECKS = {
    "ck_orch_runs_not_own_successor": "supersedes_run_id IS NULL OR supersedes_run_id != id",
    "ck_orch_runs_terminal_reason": (
        "terminal_reason IS NULL OR (terminal_reason IN ('reclassified') "
        "AND status IN ('completed', 'cancelled'))"
    ),
    "ck_orch_runs_goal_type_snapshot": (
        "goal_type_snapshot IS NULL OR goal_type_snapshot IN ('outcome', 'roadmap', 'continuous')"
    ),
    "ck_orch_runs_planning_mode_snapshot": (
        "planning_mode_snapshot IS NULL OR "
        "planning_mode_snapshot IN ('roadmap', 'delivery', 'implementation', 'continuous_policy')"
    ),
}


def upgrade() -> None:
    with op.batch_alter_table("orchestration_goals") as batch:
        batch.add_column(sa.Column("goal_type_source", sa.String(length=32), nullable=True))
        batch.add_column(sa.Column("goal_type_rationale", sa.Text(), nullable=True))
        batch.add_column(sa.Column("planning_mode", sa.String(length=32), nullable=True))
        batch.add_column(sa.Column("planning_rationale", sa.Text(), nullable=True))
    op.get_bind().execute(
        sa.text(
            "UPDATE orchestration_goals SET goal_type_source = 'legacy', "
            "goal_type_rationale = :goal_type_rationale, "
            "planning_mode = CASE goal_type WHEN 'roadmap' THEN 'roadmap' "
            "WHEN 'continuous' THEN 'continuous_policy' ELSE 'implementation' END, "
            "planning_rationale = :planning_rationale"
        ),
        {"goal_type_rationale": GOAL_TYPE_RATIONALE, "planning_rationale": PLANNING_RATIONALE},
    )
    with op.batch_alter_table("orchestration_goals") as batch:
        batch.alter_column(
            "goal_type_source", existing_type=sa.String(length=32), nullable=False, server_default="legacy"
        )
        batch.alter_column(
            "goal_type_rationale", existing_type=sa.Text(), nullable=False, server_default=GOAL_TYPE_RATIONALE
        )
        batch.alter_column(
            "planning_mode", existing_type=sa.String(length=32), nullable=False, server_default="implementation"
        )
        batch.alter_column(
            "planning_rationale", existing_type=sa.Text(), nullable=False, server_default=PLANNING_RATIONALE
        )
        for name, check in GOAL_CHECKS.items():
            batch.create_check_constraint(name, check)

    with op.batch_alter_table("orchestration_runs") as batch:
        batch.add_column(sa.Column("supersedes_run_id", sa.Uuid(), nullable=True))
        batch.add_column(sa.Column("terminal_reason", sa.String(length=32), nullable=True))
        batch.add_column(sa.Column("goal_type_snapshot", sa.String(length=50), nullable=True))
        batch.add_column(sa.Column("planning_mode_snapshot", sa.String(length=32), nullable=True))
        batch.add_column(sa.Column("goal_contract_fingerprint", sa.String(length=64), nullable=True))
        batch.create_foreign_key(
            "fk_orch_runs_supersedes_run_id", "orchestration_runs", ["supersedes_run_id"], ["id"]
        )
        batch.create_unique_constraint("uq_orch_runs_supersedes_run_id", ["supersedes_run_id"])
        for name, check in RUN_CHECKS.items():
            batch.create_check_constraint(name, check)
    # Snapshots come from the goal; the contract fingerprint stays NULL until a run is frozen.
    op.execute(
        "UPDATE orchestration_runs SET "
        "goal_type_snapshot = (SELECT g.goal_type FROM orchestration_goals g WHERE g.id = orchestration_runs.goal_id), "
        "planning_mode_snapshot = (SELECT g.planning_mode FROM orchestration_goals g "
        "WHERE g.id = orchestration_runs.goal_id)"
    )


def downgrade() -> None:
    lineage = op.get_bind().execute(sa.text(
        "SELECT 1 FROM orchestration_runs WHERE supersedes_run_id IS NOT NULL OR terminal_reason IS NOT NULL "
        "UNION ALL "
        "SELECT 1 FROM orchestration_goals WHERE parent_goal_id IS NOT NULL AND goal_type != 'outcome'"
    )).first()
    if lineage is not None:
        raise RuntimeError(
            "049 downgrade refused: successor/recursive lineage exists; "
            "restore a verified pre-049 snapshot or forward-fix"
        )
    with op.batch_alter_table("orchestration_runs") as batch:
        for name in RUN_CHECKS:
            batch.drop_constraint(name, type_="check")
        batch.drop_constraint("uq_orch_runs_supersedes_run_id", type_="unique")
        batch.drop_constraint("fk_orch_runs_supersedes_run_id", type_="foreignkey")
        for column in (
            "goal_contract_fingerprint", "planning_mode_snapshot", "goal_type_snapshot",
            "terminal_reason", "supersedes_run_id",
        ):
            batch.drop_column(column)
    with op.batch_alter_table("orchestration_goals") as batch:
        for name in GOAL_CHECKS:
            batch.drop_constraint(name, type_="check")
        for column in ("planning_rationale", "planning_mode", "goal_type_rationale", "goal_type_source"):
            batch.drop_column(column)
