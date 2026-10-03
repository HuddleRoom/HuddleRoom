"""Test suite for concurrent warning creation (fix #12).

Tests that OrchestrationWarningService.create_warning properly handles
idempotent concurrent warning creations. When two concurrent
create_warning calls race on the same (goal_id, warning_type, source_process_run_id,
run_id, related_*_id), one caller acquires the goal lock and the other returns
its warning unchanged.
"""

import asyncio
import pytest

from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import OrchestrationProcessRun
from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService
from huddleroom.services.orchestration_manager_selection import (
    NO_MANAGER_WARNING_TYPE,
    NO_MANAGER_WARNING_MESSAGE,
)


@pytest.mark.asyncio
async def test_create_warning_concurrent_idempotent(db_session, test_project):
    """Concurrent identical no_manager warnings on same goal don't duplicate (fix #12).

    Tests that when two create_warning calls race to create the same
    no_manager warning (identical goal_id, warning_type, source_process_run_id,
    run_id), only one active warning exists. The service's goal lock serializes
    the race, so check-then-insert is atomic w.r.t. duplicates.

    This single-session check complements the concurrent-session variant below,
    which exercises SQLite's writer lock.
    """
    warning_svc = OrchestrationWarningService()
    orchestration_svc = OrchestrationService()

    # Create a goal and run
    goal, run = await orchestration_svc.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Concurrent warning test",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )

    # Create a process run to link the warning to (simulates manager_selection skip)
    process_run = OrchestrationProcessRun(
        goal_id=goal.id,
        process_type="manager_selection",
        status="skipped",
        run_id=run.id,
        trigger_reason="test",
    )
    db_session.add(process_run)
    await db_session.flush()

    # First create_warning succeeds
    warning1 = await warning_svc.create_warning(
        db_session,
        goal.id,
        warning_type=NO_MANAGER_WARNING_TYPE,
        severity="warning",
        message=NO_MANAGER_WARNING_MESSAGE,
        run_id=run.id,
        source_process_run_id=process_run.id,
    )
    assert warning1 is not None
    assert warning1.active is True
    assert warning1.warning_type == NO_MANAGER_WARNING_TYPE
    goal_updated_at = goal.updated_at

    # Second create_warning with identical parameters should return the existing warning
    # (this simulates the concurrent caller that arrives after the first one already
    # won the INSERT race; under the lock, it hits the check and returns existing)
    warning2 = await warning_svc.create_warning(
        db_session,
        goal.id,
        warning_type=NO_MANAGER_WARNING_TYPE,
        severity="warning",
        message=NO_MANAGER_WARNING_MESSAGE,
        run_id=run.id,
        source_process_run_id=process_run.id,
    )

    # Both callers should see the same warning
    assert warning2.id == warning1.id
    assert warning2.active is True
    await db_session.refresh(goal)
    assert goal.updated_at == goal_updated_at.replace(tzinfo=None)


@pytest.mark.asyncio
async def test_create_warning_concurrent_with_concurrent_sessions(concurrent_sessions, tmp_path):
    """Test warning creation is idempotent with concurrent sessions.

    Runs on SQLite too: WAL mode + busy_timeout (see conftest.py test_engine)
    allow real interleaved writes from independent sessions against the same
    file, so this exercises the actual race instead of skipping it.

    NOTE: deliberately does not use the `test_project` fixture -- it depends
    on `db_session`, which holds an uncommitted transaction open for the
    entire test (only rolled back at teardown). On SQLite that transaction
    holds the single writer lock for the whole test, starving the
    `concurrent_sessions` writes below until busy_timeout expires. Instead
    the project is created and committed directly through session1.

    NOTE: each concurrent call commits its own session immediately after
    create_warning returns. Without this, SQLite's single-writer mode means
    the winner's write transaction stays open (uncommitted) for the rest of
    the test, and the loser blocks until busy_timeout (30s) and then raises
    "database is locked" -- not a real exercise of the race, just a slow,
    always-one-side-fails test. Committing promptly lets both sides
    interleave in well under a second and both genuinely succeed.
    """
    from huddleroom.models.project import Project

    warning_svc = OrchestrationWarningService()
    orchestration_svc = OrchestrationService()
    session1, session2 = concurrent_sessions

    # Release both calls immediately before the service acquires its goal
    # lock. The loser must then wait and return the winner's warning.
    lock_attempts = 0
    lock_ready = asyncio.Event()

    def synchronize_goal_lock(execute):
        async def synchronized(statement, *args, **kwargs):
            nonlocal lock_attempts
            if not lock_ready.is_set() and "UPDATE orchestration_goals" in str(statement):
                lock_attempts += 1
                if lock_attempts == 2:
                    lock_ready.set()
                await lock_ready.wait()
            return await execute(statement, *args, **kwargs)

        return synchronized

    session1.execute = synchronize_goal_lock(session1.execute)
    session2.execute = synchronize_goal_lock(session2.execute)

    project = Project(name="Concurrent warning test project", description="", workspace_path=str(tmp_path), config={})
    session1.add(project)
    await session1.flush()
    await session1.commit()

    # Create a goal and run in session1
    goal, run = await orchestration_svc.create_goal(
        session1,
        project_id=project.id,
        data=OrchestrationGoalCreate(
            objective="Concurrent warning test",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )

    # Create a process run in session1
    process_run = OrchestrationProcessRun(
        goal_id=goal.id,
        process_type="manager_selection",
        status="skipped",
        run_id=run.id,
        trigger_reason="test",
    )
    session1.add(process_run)
    await session1.flush()
    await session1.commit()

    # Try concurrent warning creation via asyncio.gather
    async def create_via_session1():
        result = await warning_svc.create_warning(
            session1, goal.id, warning_type=NO_MANAGER_WARNING_TYPE,
            severity="warning", message=NO_MANAGER_WARNING_MESSAGE,
            run_id=run.id, source_process_run_id=process_run.id,
        )
        await session1.commit()
        return result

    async def create_via_session2():
        result = await warning_svc.create_warning(
            session2, goal.id, warning_type=NO_MANAGER_WARNING_TYPE,
            severity="recommendation", message="A competing caller's message.",
            run_id=run.id, source_process_run_id=process_run.id,
        )
        await session2.commit()
        return result

    results = await asyncio.gather(create_via_session1(), create_via_session2(), return_exceptions=True)
    warning1, warning2 = results

    # Both concurrent callers should succeed idempotently with the same row.
    for w in (warning1, warning2):
        assert not isinstance(w, Exception), f"create_warning raised: {w!r}"
    assert warning1.id == warning2.id
    assert warning1.active is True
    assert warning2.active is True
    assert warning1.severity == warning2.severity
    assert warning1.message == warning2.message

    # Verify only one active warning of this type exists for this goal
    warnings = await warning_svc.list_warnings(
        session1, goal.id, active_only=True
    )
    matching_warnings = [
        w for w in warnings
        if w.warning_type == NO_MANAGER_WARNING_TYPE
        and w.source_process_run_id == process_run.id
        and w.run_id == run.id
    ]
    assert len(matching_warnings) == 1, (
        f"Expected 1 matching active warning, got {len(matching_warnings)}: "
        f"{[w.id for w in matching_warnings]}"
    )
    assert matching_warnings[0].id == warning1.id


@pytest.mark.asyncio
async def test_create_warning_rollback_does_not_persist_on_sqlite(concurrent_sessions, tmp_path):
    """The SQLite goal lock remains in the caller's transaction."""
    from huddleroom.models.project import Project

    session1, session2 = concurrent_sessions
    project = Project(name="Warning rollback project", description="", workspace_path=str(tmp_path), config={})
    session1.add(project)
    await session1.flush()
    await session1.commit()

    goal, run = await OrchestrationService().create_goal(
        session1,
        project_id=project.id,
        data=OrchestrationGoalCreate(
            objective="Warning rollback test",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )
    process_run = OrchestrationProcessRun(
        goal_id=goal.id,
        process_type="manager_selection",
        status="skipped",
        run_id=run.id,
        trigger_reason="test",
    )
    session1.add(process_run)
    await session1.commit()

    await OrchestrationWarningService().create_warning(
        session1,
        goal.id,
        warning_type=NO_MANAGER_WARNING_TYPE,
        severity="warning",
        message=NO_MANAGER_WARNING_MESSAGE,
        run_id=run.id,
        source_process_run_id=process_run.id,
    )
    goal_id = goal.id
    await session1.rollback()

    assert await OrchestrationWarningService().list_warnings(session2, goal_id) == []
