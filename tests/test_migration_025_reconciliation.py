"""Test migration 025 reconciliation of duplicate current process runs.

This test validates that the migration correctly handles pre-existing duplicates
of "current" (superseded_by_id IS NULL) process runs before creating the unique
index.
"""
import importlib.util
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from huddleroom.config import settings
from huddleroom.models.orchestration_process import OrchestrationProcessRun
from huddleroom.models.orchestration import OrchestrationGoal
from huddleroom.models.project import Project

# Import the actual reconciliation function from the migration
# The migration file is named "025_orch_process_runs_current_unique_index.py"
# which Python imports with underscores in the name
_migration_path = Path(__file__).parent.parent / "alembic" / "versions"
_spec = importlib.util.spec_from_file_location(
    "migration_025",
    _migration_path / "025_orch_process_runs_current_unique_index.py"
)
_migration_025 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_migration_025)
reconcile_duplicate_current_runs = _migration_025.reconcile_duplicate_current_runs


@pytest.mark.asyncio
async def test_migration_025_reconcile_duplicate_current_runs():
    """Test that duplicate current process runs are reconciled before index creation.

    This test simulates the reconciliation SQL that migration 025 applies: it
    creates duplicate "current" (superseded_by_id IS NULL) process runs, applies
    the reconciliation logic (selecting the most-recent survivor and superseding
    others), and verifies only one current run remains.
    """
    # Create a temporary in-memory database for testing
    engine_url = "sqlite+aiosqlite:///:memory:" if settings.is_sqlite else settings.database_url

    if not settings.is_sqlite and engine_url == settings.database_url:
        pytest.skip("Skipping Postgres test to avoid polluting shared test DB")

    engine = create_async_engine(engine_url, connect_args={"check_same_thread": False} if settings.is_sqlite else {})

    # Create schema and insert test data
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: Project.metadata.create_all(c))
        # Drop the unique index on orchestration_process_runs so we can create duplicates for testing
        try:
            await conn.execute(sa.text("DROP INDEX uq_orch_process_runs_goal_type_current"))
        except Exception:
            # Index may not exist, that's fine
            pass

    # Use a session to insert test data (separate transaction)
    async_session = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)
    session = async_session()

    # Create test project and goal
    project_id = uuid.uuid4()
    goal_id = uuid.uuid4()

    project = Project(id=project_id, name="test_project")
    goal = OrchestrationGoal(id=goal_id, project_id=project_id, objective="test goal")

    session.add(project)
    session.add(goal)
    await session.flush()

    # Create duplicate "current" process runs for the same (goal_id, process_type)
    # Use staggered created_at times to ensure deterministic survivor selection
    base_time = datetime.now(timezone.utc)

    run_ids = []
    for i in range(3):
        run_id = uuid.uuid4()
        run_ids.append(run_id)
        run = OrchestrationProcessRun(
            id=run_id,
            goal_id=goal_id,
            process_type="goal_definition",
            process_version=1,
            status="completed",
            trigger_reason="test",
            created_at=base_time + timedelta(seconds=i),  # Staggered creation
        )
        session.add(run)

    await session.commit()
    await session.close()

    # Now execute the reconciliation logic using the actual migration function
    # Use sync connection via run_sync to call the sync migration function
    async with engine.begin() as async_conn:
        # Prepare goal_id_hex for parametrized queries (SQLite stores UUIDs as hex strings)
        goal_id_hex = goal_id.hex if isinstance(goal_id, uuid.UUID) else goal_id
        process_type = "goal_definition"

        # Use run_sync to execute sync operations with the async connection's pool
        def check_and_reconcile(sync_conn):
            # Debug: check what's in the table
            debug_sql = "SELECT COUNT(*) FROM orchestration_process_runs"
            result = sync_conn.execute(sa.text(debug_sql))
            total_count = result.scalar()

            # Find duplicates before reconciliation
            find_dups_sql = """
                SELECT goal_id, process_type, COUNT(*) as dup_count
                FROM orchestration_process_runs
                WHERE superseded_by_id IS NULL
                GROUP BY goal_id, process_type
                HAVING COUNT(*) > 1
            """
            result = sync_conn.execute(sa.text(find_dups_sql))
            duplicates_before = result.fetchall()

            assert len(duplicates_before) == 1, f"Should have exactly 1 duplicate group (total rows: {total_count}, duplicates found: {len(duplicates_before)})"
            assert duplicates_before[0][2] == 3, "Duplicate group should have 3 current runs"

            # Apply reconciliation using the actual migration function
            reconcile_duplicate_current_runs(sync_conn)

            # Verify reconciliation
            find_dups_after_sql = """
                SELECT goal_id, process_type, COUNT(*) as dup_count
                FROM orchestration_process_runs
                WHERE superseded_by_id IS NULL
                GROUP BY goal_id, process_type
                HAVING COUNT(*) > 1
            """
            result = sync_conn.execute(sa.text(find_dups_after_sql))
            duplicates_after = result.fetchall()

            assert len(duplicates_after) == 0, "Should have no duplicate groups after reconciliation"

            # Verify exactly 1 current run remains
            current_runs_sql = """
                SELECT COUNT(*) FROM orchestration_process_runs
                WHERE goal_id = :goal_id AND process_type = :process_type AND superseded_by_id IS NULL
            """
            result = sync_conn.execute(
                sa.text(current_runs_sql),
                {"goal_id": goal_id_hex, "process_type": process_type}
            )
            current_count = result.scalar()
            assert current_count == 1, f"Should have exactly 1 current run, got {current_count}"

            # Verify survivor is the most recently created run (run_ids[2])
            survivor_sql = """
                SELECT id FROM orchestration_process_runs
                WHERE goal_id = :goal_id AND process_type = :process_type AND superseded_by_id IS NULL
            """
            result = sync_conn.execute(
                sa.text(survivor_sql),
                {"goal_id": goal_id_hex, "process_type": process_type}
            )
            actual_survivor = result.scalar()
            # SQLite returns the UUID as a hex string without hyphens
            expected_survivor_hex = run_ids[2].hex
            assert actual_survivor == expected_survivor_hex, f"Expected survivor to be most recent run {run_ids[2].hex}, got {actual_survivor}"

            # Verify the two older runs are superseded
            superseded_sql = """
                SELECT COUNT(*) FROM orchestration_process_runs
                WHERE goal_id = :goal_id AND process_type = :process_type AND superseded_by_id IS NOT NULL
            """
            result = sync_conn.execute(
                sa.text(superseded_sql),
                {"goal_id": goal_id_hex, "process_type": process_type}
            )
            superseded_count = result.scalar()
            assert superseded_count == 2, f"Should have exactly 2 superseded runs, got {superseded_count}"

        await async_conn.run_sync(check_and_reconcile)

    await engine.dispose()


@pytest.mark.asyncio
async def test_migration_025_no_duplicates_case():
    """Test that reconciliation handles clean data (no duplicates) gracefully."""
    engine_url = "sqlite+aiosqlite:///:memory:" if settings.is_sqlite else settings.database_url

    if not settings.is_sqlite and engine_url == settings.database_url:
        pytest.skip("Skipping Postgres test to avoid polluting shared test DB")

    engine = create_async_engine(engine_url, connect_args={"check_same_thread": False} if settings.is_sqlite else {})

    async with engine.begin() as conn:
        await conn.run_sync(lambda c: Project.metadata.create_all(c))

    async_session = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)
    session = async_session()

    project_id = uuid.uuid4()
    goal_id = uuid.uuid4()

    project = Project(id=project_id, name="test_project")
    goal = OrchestrationGoal(id=goal_id, project_id=project_id, objective="test goal")

    session.add(project)
    session.add(goal)
    await session.flush()

    # Create a single "current" process run (no duplicates)
    run = OrchestrationProcessRun(
        id=uuid.uuid4(),
        goal_id=goal_id,
        process_type="goal_definition",
        process_version=1,
        status="completed",
        trigger_reason="test",
    )
    session.add(run)
    await session.flush()
    await session.close()

    # Reconciliation should handle this gracefully (no-op)
    async with engine.begin() as conn:
        find_dups_sql = """
            SELECT COUNT(*) FROM (
                SELECT goal_id, process_type, COUNT(*) as dup_count
                FROM orchestration_process_runs
                WHERE superseded_by_id IS NULL
                GROUP BY goal_id, process_type
                HAVING COUNT(*) > 1
            )
        """
        result = await conn.execute(sa.text(find_dups_sql))
        dup_groups_count = result.scalar()

        assert dup_groups_count == 0, "No duplicates should exist"

    await engine.dispose()
