"""Test suite for concurrent skip_process handling.

Tests that OrchestrationProcessService.skip_process properly handles
idempotent concurrent identical skip calls. When two concurrent skip calls
race on the same (goal_id, process_type) with identical parameters, the loser's
UPDATE affects 0 rows (because the winner already set status to "skipped").
The fix ensures both return successfully with the same skipped row instead of
raising a spurious error.

LIMITATION (SQLite): SQLite's single-writer mode prevents true interleaved
concurrent writes within independent sessions. On SQLite, we test the idempotent
retry path by sequential calls to skip_process. On Postgres, a true concurrent
test with asyncio.gather would exercise the actual race. To test with Postgres,
run this test against a Postgres instance (check if POSTGRES_TEST_URL or similar
env var is set in CI/test config).
"""

import asyncio
import pytest

from huddleroom.models.orchestration import OrchestrationGoal
from huddleroom.models.project import Project
from huddleroom.services.orchestration_process_service import OrchestrationProcessService


@pytest.mark.asyncio
async def test_skip_process_concurrent_identical_calls(db_session, test_project):
    """Concurrent identical skip calls on running process both succeed (sequential on SQLite).

    Tests that when two skip_process calls race on a running process with
    identical parameters, the winner's UPDATE succeeds and the loser's UPDATE
    affects 0 rows. The fix ensures the loser refreshes the row, sees it's
    already "skipped", and returns successfully (idempotent) instead of raising.

    On SQLite, this test runs sequentially (single session). On Postgres,
    a true concurrent variant with asyncio.gather would be needed to exercise
    the actual interleaved race (see concurrent_sessions fixture if Postgres
    support is added).
    """
    svc = OrchestrationProcessService()

    # Set up goal and running process
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Concurrent skip test",
        success_criteria=[{"key": "done", "description": "Done"}],
    )
    db_session.add(goal)
    await db_session.flush()

    # Create a running process run
    process_run = await svc.start_process(
        db_session,
        goal.id,
        process_type="goal_definition",
        trigger_reason="test",
    )
    await db_session.flush()

    skipped_by = "human:test-user-id"
    reason = "Testing concurrent skip idempotence"

    # First skip succeeds (converts running -> skipped)
    result1 = await svc.skip_process(
        db_session,
        goal.id,
        process_type="goal_definition",
        skipped_by=skipped_by,
        reason=reason,
    )
    assert result1.status == "skipped"
    assert result1.skipped_by == skipped_by
    assert result1.override_reason == reason

    # Second skip with identical parameters should also succeed (idempotent)
    # This simulates what would happen if a second concurrent caller
    # called skip_process after the first one already won the UPDATE race.
    result2 = await svc.skip_process(
        db_session,
        goal.id,
        process_type="goal_definition",
        skipped_by=skipped_by,
        reason=reason,
    )

    # Both should return the same row
    assert result2.status == "skipped"
    assert result2.id == result1.id
    assert result2.skipped_by == skipped_by
    assert result2.override_reason == reason

    # Verify only one process run exists for this (goal_id, process_type)
    all_runs = await svc.list_process_runs(
        db_session, goal.id, process_type="goal_definition"
    )
    skipped_runs = [r for r in all_runs if r.status == "skipped"]
    assert len(skipped_runs) == 1
    assert skipped_runs[0].id == result1.id


@pytest.mark.asyncio
async def test_skip_process_concurrent_with_concurrent_sessions(concurrent_sessions):
    """True concurrent skip_process race via two independent sessions.

    Runs on SQLite too: WAL mode + busy_timeout (see conftest.py test_engine)
    allow real interleaved writes from independent sessions against the same
    file, so this exercises the actual race instead of skipping it.

    NOTE: deliberately does not use the `test_project` fixture -- it depends
    on `db_session`, which holds an uncommitted transaction open for the
    entire test (only rolled back at teardown). On SQLite that transaction
    holds the single writer lock for the whole test, starving the
    `concurrent_sessions` writes below until busy_timeout expires. Instead
    the project and goal are created and committed directly through
    session1 (see test_warning_concurrent.py for the same pattern).

    NOTE: each concurrent call commits its own session immediately after
    skip_process returns. Without this, SQLite's single-writer mode means
    the winner's write transaction stays open (uncommitted) for the rest of
    the test, and the loser blocks until busy_timeout (30s) and then raises
    "database is locked" -- not a real exercise of the race, just a slow,
    always-one-side-fails test. Committing promptly lets both sides
    interleave in well under a second and both genuinely succeed.
    """
    svc = OrchestrationProcessService()
    session1, session2 = concurrent_sessions

    project = Project(name="Concurrent skip test project", description="", config={})
    session1.add(project)
    await session1.flush()
    await session1.commit()

    goal = OrchestrationGoal(
        project_id=project.id,
        objective="Concurrent skip test",
        success_criteria=[{"key": "done", "description": "Done"}],
    )
    session1.add(goal)
    await session1.flush()
    await session1.commit()

    await svc.start_process(
        session1, goal.id, process_type="goal_definition", trigger_reason="test",
    )
    await session1.commit()

    skipped_by = "human:test-user-id"
    reason = "Testing concurrent skip idempotence"

    async def skip_via_session1():
        result = await svc.skip_process(
            session1, goal.id, process_type="goal_definition",
            skipped_by=skipped_by, reason=reason,
        )
        await session1.commit()
        return result

    async def skip_via_session2():
        result = await svc.skip_process(
            session2, goal.id, process_type="goal_definition",
            skipped_by=skipped_by, reason=reason,
        )
        await session2.commit()
        return result

    results = await asyncio.gather(skip_via_session1(), skip_via_session2(), return_exceptions=True)
    result1, result2 = results

    # Both concurrent callers should succeed idempotently with the same row.
    for r in (result1, result2):
        assert not isinstance(r, Exception), f"skip_process raised: {r!r}"
    assert result1.id == result2.id
    assert result1.status == "skipped"
    assert result2.status == "skipped"

    # Only one process run should exist for this (goal_id, process_type), and
    # it should be the one both callers converged on.
    all_runs = await svc.list_process_runs(
        session1, goal.id, process_type="goal_definition"
    )
    assert len(all_runs) == 1
    assert all_runs[0].id == result1.id
    assert all_runs[0].status == "skipped"


