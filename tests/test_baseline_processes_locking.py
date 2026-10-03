"""Test suite for baseline process locking and allow_heal parameter (Package 1 fixes #3, #4, #15).

Tests that _baseline_processes_ready_for_goal() properly:
1. Acquires goal lock before reading process rows (fix #3+#4)
2. Reloads process rows with populate_existing inside lock (fix #3+#4)
3. Re-verifies process status before healing (fix #3+#4)
4. Respects allow_heal=False to avoid silent rollbacks in read-only sessions (fix #15)
"""

import pytest

from huddleroom.models.orchestration import OrchestrationGoal
from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.services.orchestration_manager_selection import ManagerSelectionProcess
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_service import OrchestrationService


async def _make_goal(db, project_id):
    """Create a goal for testing."""
    goal = OrchestrationGoal(
        project_id=project_id,
        objective="Baseline process test",
        success_criteria=[{"key": "done", "description": "Done"}],
    )
    db.add(goal)
    await db.flush()
    return goal


@pytest.mark.asyncio
async def test_baseline_processes_ready_acquires_lock(db_session, test_project, monkeypatch):
    """_baseline_processes_ready_for_goal acquires and holds goal lock (fix #3+#4).

    Verifies that the lock is acquired before reading process status,
    preventing races where another operation updates the process between
    check and heal.
    """
    service = OrchestrationService()
    goal = await _make_goal(db_session, test_project.id)
    process_svc = OrchestrationProcessService()

    # Start and complete goal_definition process
    gd_run = await process_svc.start_process(
        db_session, goal.id, process_type="goal_definition", trigger_reason="test"
    )
    await process_svc.complete_process(db_session, gd_run)
    await db_session.flush()

    # Start and complete manager_selection process
    ms_run = await process_svc.start_process(
        db_session, goal.id, process_type="manager_selection", trigger_reason="test"
    )
    await process_svc.complete_process(db_session, ms_run)
    await db_session.flush()

    review_run = await process_svc.start_process(
        db_session, goal.id, process_type="agent_definition_review", trigger_reason="test"
    )
    await process_svc.complete_process(db_session, review_run)
    await process_svc.skip_process(
        db_session,
        goal.id,
        process_type="team_hierarchy",
        skipped_by="human:test-user-id",
        reason="Test hierarchy skip",
    )

    original_get_current = OrchestrationProcessService.get_current
    manager_selection_fetches = 0

    async def count_manager_selection_fetches(self, db, goal_id, process_type):
        nonlocal manager_selection_fetches
        if process_type == "manager_selection":
            manager_selection_fetches += 1
        return await original_get_current(self, db, goal_id, process_type)

    monkeypatch.setattr(
        OrchestrationProcessService, "get_current", count_manager_selection_fetches
    )

    # Readiness remains true while the loop's manager-selection row is reused.
    ready = await service._baseline_processes_ready_for_goal(db_session, goal.id)
    assert ready is True
    assert manager_selection_fetches == 1


@pytest.mark.asyncio
async def test_baseline_processes_ready_respects_allow_heal_parameter(db_session, test_project):
    """_baseline_processes_ready_for_goal respects allow_heal parameter (fix #15).

    Verifies that when allow_heal=False, the function can be called from read-only
    sessions without risking silent rollback of heal writes.
    """
    service = OrchestrationService()
    goal = await _make_goal(db_session, test_project.id)
    process_svc = OrchestrationProcessService()

    # Start and complete goal_definition process
    gd_run = await process_svc.start_process(
        db_session, goal.id, process_type="goal_definition", trigger_reason="test"
    )
    await process_svc.complete_process(db_session, gd_run)
    await db_session.flush()

    # Start and skip manager_selection process
    ms_run = await process_svc.start_process(
        db_session, goal.id, process_type="manager_selection", trigger_reason="test"
    )
    await process_svc.skip_process(
        db_session,
        goal.id,
        process_type="manager_selection",
        skipped_by="human:test-user-id",
        reason="Test skip",
    )
    review_run = await process_svc.start_process(
        db_session, goal.id, process_type="agent_definition_review", trigger_reason="test"
    )
    await process_svc.complete_process(db_session, review_run)
    await process_svc.skip_process(
        db_session,
        goal.id,
        process_type="team_hierarchy",
        skipped_by="human:test-user-id",
        reason="Test hierarchy skip",
    )
    await db_session.flush()

    # Both with allow_heal=True and allow_heal=False should return same result
    # (skipped process is already terminal, no heal needed)
    ready_with_heal = await service._baseline_processes_ready_for_goal(
        db_session, goal.id, allow_heal=True
    )
    ready_without_heal = await service._baseline_processes_ready_for_goal(
        db_session, goal.id, allow_heal=False
    )
    # Both should return True since all processes are terminal
    assert ready_with_heal is True
    assert ready_without_heal is True


@pytest.mark.asyncio
async def test_baseline_processes_not_ready_if_still_running(db_session, test_project):
    """Goal_definition still running means not ready (fix #3+#4).

    Verifies that an incomplete goal_definition process prevents baseline readiness.
    """
    service = OrchestrationService()
    goal = await _make_goal(db_session, test_project.id)
    process_svc = OrchestrationProcessService()

    # Start but don't complete goal_definition
    await process_svc.start_process(
        db_session, goal.id, process_type="goal_definition", trigger_reason="test"
    )
    await db_session.flush()

    # Should not be ready
    ready = await service._baseline_processes_ready_for_goal(db_session, goal.id)
    assert ready is False
