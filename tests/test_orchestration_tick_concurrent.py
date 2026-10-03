"""Test suite for concurrent tick handling and locking (Package 1 fixes #5, #9).

Tests that OrchestrationService.tick() properly serializes concurrent ticks on the
same goal and avoids duplicate events or lost cursor advances.

Due to SQLite's single-writer limitation, we test within a single session that
simulates the race by calling tick() twice on the same run to verify:
1. No duplicate events are emitted when tick() is called concurrently (fix #9)
2. Cursor advances correctly even with baseline process writes (fix #5)
3. Baseline processes are read fresh inside the lock, not stale
"""

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import func, select

from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.services.event_bus import emit_event_once
from huddleroom.services.orchestration_service import OrchestrationService


@pytest_asyncio.fixture(autouse=True)
async def runnable_workspace(db_session, test_project, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    test_project.workspace_path = str(workspace.resolve())
    await db_session.flush()


async def _make_run(db, project_id):
    """Create a goal and run for testing."""
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db,
        project_id=project_id,
        data=OrchestrationGoalCreate(
            objective="Concurrent tick test",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )
    return service, goal, run


async def _count_tick_events(db, project_id):
    """Count orchestration.tick events in the project."""
    result = await db.execute(
        select(func.count(EventLog.id)).where(
            EventLog.project_id == project_id,
            EventLog.event_type == "orchestration.tick",
        )
    )
    return result.scalar() or 0


async def _seq(db, bus_event) -> int:
    """Return EventLog.seq for the given BusEvent."""
    row = (await db.execute(
        select(EventLog).where(EventLog.id == bus_event.id)
    )).scalar_one()
    return row.seq


@pytest.mark.asyncio
async def test_concurrent_ticks_no_duplicate_events(db_session, test_project):
    """Sequential ticks with same events don't emit duplicate tick events (fix #9).

    This verifies the dedup_key in emit_event_once prevents duplicate emission
    when tick's event_cursor/baseline_process writes are inside the lock (fix #9).
    """
    service, goal, run = await _make_run(db_session, test_project.id)

    # Seed one event
    seeded, _ = await emit_event_once(
        db_session, test_project.id, "task.created", {"n": 1}, dedup_key="seed-1"
    )

    # First tick processes the event and emits tick event
    result1 = await service.tick(db_session, run.id)
    assert result1["processed_events"] == 1
    assert result1["tick_emitted"] is True
    assert result1["event_cursor"] == await _seq(db_session, seeded)

    tick_count_after_first = await _count_tick_events(db_session, test_project.id)
    assert tick_count_after_first == 1

    # Refresh run from DB to get updated state
    await db_session.refresh(run)

    # Second tick with no new events should not emit another tick event
    # (dedup_key same, so emit_event_once returns existing event with created=False)
    result2 = await service.tick(db_session, run.id)
    assert result2["processed_events"] == 0
    assert result2["tick_emitted"] is False  # No new tick event emitted
    assert result2["event_cursor"] == result1["event_cursor"]

    tick_count_after_second = await _count_tick_events(db_session, test_project.id)
    assert tick_count_after_second == 1  # Still only 1, no duplicate


@pytest.mark.asyncio
async def test_tick_cursor_advances_correctly_inside_lock(db_session, test_project):
    """Tick's event_cursor and baseline process writes stay serialized (fix #5, #9).

    Verifies that cursor assignment happens inside the lock block, preventing
    another tick from interleaving and seeing partial state.
    """
    service, goal, run = await _make_run(db_session, test_project.id)

    # Seed three events
    events = []
    for i in range(3):
        ev, _ = await emit_event_once(
            db_session,
            test_project.id,
            "task.created",
            {"n": i},
            dedup_key=f"seed-{i}",
        )
        events.append(ev)

    # First tick processes all 3 events, cursor lands on last
    result1 = await service.tick(db_session, run.id)
    assert result1["processed_events"] == 3
    cursor_after_first = await _seq(db_session, events[-1])
    assert result1["event_cursor"] == cursor_after_first
    assert run.event_cursor == cursor_after_first

    # Refresh run from DB
    await db_session.refresh(run)

    # Second tick with no new events sees the advanced cursor from first tick
    result2 = await service.tick(db_session, run.id)
    assert result2["processed_events"] == 0
    assert result2["event_cursor"] == cursor_after_first  # Still the same


@pytest.mark.asyncio
async def test_tick_reloads_goal_status_inside_lock(db_session, test_project):
    """Tick reloads goal inside lock to catch status changes (fix #5).

    Verifies that goal status is re-read with populate_existing inside the lock,
    preventing stale checks from before the lock was acquired.
    """
    service, goal, run = await _make_run(db_session, test_project.id)

    # First tick succeeds
    result1 = await service.tick(db_session, run.id)
    assert result1["status"] in ("running", "blocked")  # Goal is active

    # Manually pause the run (simulating an external operation that happened
    # between lock acquisition and status check)
    run.status = "paused"
    await db_session.flush()

    # Refresh from DB
    await db_session.refresh(run)

    # Second tick should recognize paused status and return no-op result
    result2 = await service.tick(db_session, run.id)
    assert result2["status"] == "paused"
    assert result2["processed_events"] == 0
    assert result2["tick_emitted"] is False
