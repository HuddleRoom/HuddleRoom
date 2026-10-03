"""Add partial unique index on orchestration_process_runs (goal_id, process_type) WHERE superseded_by_id IS NULL

Revision ID: 025
Revises: 024
Create Date: 2026-07-19

NOTE: This index is also defined in the OrchestrationProcessRun model in
orchestration_process.py. When using create_all() (e.g., in tests), the
index is created directly from the model. This migration is for production
databases that use alembic migrations and may already have the index from
the model definition. The create_index call handles this gracefully.

"""
# pylint: disable=invalid-name,no-member,line-too-long,wrong-import-order
from alembic import op
import sqlalchemy as sa

revision = "025"
down_revision = "024"
branch_labels = None
depends_on = None


def reconcile_duplicate_current_runs(conn) -> None:
    """Reconcile pre-existing duplicates of 'current' process runs.

    A race condition before this migration could produce multiple rows with
    superseded_by_id IS NULL for the same (goal_id, process_type). This step
    finds and resolves such duplicates deterministically before creating the
    unique index.

    For each (goal_id, process_type) group with multiple current rows,
    keep the row with the highest created_at (using id as tiebreaker), and
    mark all others as superseded by that survivor.

    Raises RuntimeError if any group cannot be deterministically resolved
    (e.g., all duplicates have identical created_at and no id ordering).
    """
    # Find (goal_id, process_type) groups with multiple current runs
    find_dups_sql = """
        SELECT goal_id, process_type, COUNT(*) as dup_count
        FROM orchestration_process_runs
        WHERE superseded_by_id IS NULL
        GROUP BY goal_id, process_type
        HAVING COUNT(*) > 1
    """
    result = conn.execute(sa.text(find_dups_sql))
    duplicates = result.fetchall()

    if not duplicates:
        return  # No duplicates found

    # For each duplicate group, identify survivor and supersede others
    for goal_id, process_type, _ in duplicates:
        # Find the survivor: highest created_at, then highest id (string-sorted UUID)
        # SQLite and Postgres both support MAX() on timestamps and string comparison on UUIDs
        find_survivor_sql = """
            SELECT id FROM orchestration_process_runs
            WHERE goal_id = :goal_id AND process_type = :process_type AND superseded_by_id IS NULL
            ORDER BY created_at DESC, id DESC
            LIMIT 1
        """
        survivor_result = conn.execute(
            sa.text(find_survivor_sql),
            {"goal_id": str(goal_id), "process_type": process_type}
        )
        survivor_row = survivor_result.fetchone()
        if not survivor_row:
            raise RuntimeError(
                f"Failed to identify survivor for (goal_id={goal_id}, process_type={process_type}). "
                "Database may be corrupted."
            )

        survivor_id = survivor_row[0]

        # Supersede all other current runs in this group by setting superseded_by_id to survivor
        supersede_sql = """
            UPDATE orchestration_process_runs
            SET superseded_by_id = :survivor_id
            WHERE goal_id = :goal_id
              AND process_type = :process_type
              AND superseded_by_id IS NULL
              AND id != :survivor_id
        """
        conn.execute(
            sa.text(supersede_sql),
            {
                "survivor_id": str(survivor_id),
                "goal_id": str(goal_id),
                "process_type": process_type,
            }
        )


# Backward compatibility alias for existing internal calls
def _reconcile_duplicate_current_runs(conn) -> None:
    """Deprecated: use reconcile_duplicate_current_runs instead."""
    reconcile_duplicate_current_runs(conn)


def upgrade() -> None:
    # Add partial unique index to prevent race condition in start_process/skip_process.
    # The index is also defined in the model, so it may already exist (e.g. from
    # model create_all() in tests). Skip only on a verified already-exists condition
    # -- checked via the inspector -- so genuine create failures (duplicate current
    # rows, dialect/SQL errors) still fail the migration instead of being swallowed
    # and silently leaving concurrency correctness unprotected (review finding).
    index_name = "uq_orch_process_runs_goal_type_current"
    bind = op.get_bind()
    existing = {ix["name"] for ix in sa.inspect(bind).get_indexes("orchestration_process_runs")}
    if index_name in existing:
        return

    # Reconcile any pre-existing duplicate "current" rows before creating the unique index
    reconcile_duplicate_current_runs(bind)

    op.create_index(
        index_name,
        "orchestration_process_runs",
        ["goal_id", "process_type"],
        unique=True,
        sqlite_where=sa.text("superseded_by_id IS NULL"),
        postgresql_where=sa.text("superseded_by_id IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_orch_process_runs_goal_type_current", table_name="orchestration_process_runs")
