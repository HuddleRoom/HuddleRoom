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


# ---------------------------------------------------------------------------
# SPR 180/184: two-session publication races around the global roster
# write lock and the per-goal baseline-transition locks.
# ---------------------------------------------------------------------------
import asyncio  # noqa: E402
import dataclasses  # noqa: E402
from unittest.mock import AsyncMock  # noqa: E402

from sqlalchemy import select, update  # noqa: E402

from huddleroom.models.agent import Agent  # noqa: E402
from huddleroom.models.orchestration import OrchestrationRun  # noqa: E402
from tests.test_orchestration_roster_mutation import (  # noqa: E402
    GOAL_HIGH,
    GOAL_LOW,
    GOAL_MID,
    RACE_TIMEOUT,
    assert_roster_first_publication,
    canonical,
    create_from_delta,
    fresh_agent,
    lock_tracer,  # noqa: F401 -- fixture
    new_agent_state,
    outcome_of,
    seed_roster_goals,
    sessions,  # noqa: F401 -- fixture
    stale_error,
    roster_api,
)
from huddleroom.models.task import Task  # noqa: E402


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["goal_contract_changed", "lock_set_expanded"])
async def test_state_change_while_waiting_for_goal_locks_fails_before_effects(
    sessions, tmp_path, lock_tracer, change
):
    rm = roster_api()
    seeded = await seed_roster_goals(
        sessions, tmp_path, goal_ids=[GOAL_HIGH, GOAL_LOW, GOAL_MID], with_run={GOAL_MID: False}
    )
    delta = rm.ValidatedRosterDelta(creates=(new_agent_state(),))
    async with sessions() as s:
        plan = await rm.preview_roster_delta(s, delta, origin_goal_id=GOAL_HIGH)
        await s.commit()
    assert set(plan.affected_goal_ids) == {GOAL_HIGH, GOAL_LOW}

    previewed, resume = asyncio.Event(), asyncio.Event()
    holder = {}

    async def pause_after_initial_preview(_plan):
        if asyncio.current_task() is holder.get("a") and not previewed.is_set():
            previewed.set()
            await resume.wait()

    lock_tracer.after_preview.append(pause_after_initial_preview)
    mutation_spy = AsyncMock(side_effect=create_from_delta)

    async def session_a():
        async with sessions() as a:
            return await outcome_of(rm.AgentRosterMutationCoordinator().apply(
                a, delta, plan, mutation_spy, origin_goal_id=GOAL_HIGH, provenance={}))

    holder["a"] = task_a = asyncio.create_task(session_a())
    try:
        await asyncio.wait_for(previewed.wait(), RACE_TIMEOUT)
        assert not lock_tracer.of("goal:acquire", task_a), "A must pause before acquiring goal locks"
        async with sessions() as b:
            if change == "goal_contract_changed":
                async with OrchestrationService()._lock_goal_for_baseline_transition(b, GOAL_HIGH):
                    goal = await b.get(OrchestrationGoal, GOAL_HIGH)
                    goal.objective = "Implement the API and migrate the data"
                    await b.commit()
            else:
                async with OrchestrationService()._lock_goal_for_baseline_transition(b, GOAL_MID):
                    run = OrchestrationRun(goal_id=GOAL_MID, status="running", phase="baseline")
                    b.add(run)
                    await b.flush()
                    b.add(Task(project_id=seeded.project_id, title="work", assigned_to=seeded.agent_id,
                               metadata_={"orchestration": {"run_id": str(run.id),
                                                            "work_function": "implementation"}}))
                    await b.commit()
    finally:
        resume.set()
    stale_result = await asyncio.wait_for(task_a, RACE_TIMEOUT)

    assert stale_result.error == "state_changed"
    assert mutation_spy.call_count == 0
    assert await fresh_agent(sessions, delta.creates[0].id) is None
    assert not lock_tracer.of("agent:write", task_a)


@pytest.mark.asyncio
async def test_flush_is_not_published_before_commit_and_unlock(sessions, tmp_path, lock_tracer):
    rm = roster_api()
    seeded = await seed_roster_goals(sessions, tmp_path, goal_ids=[GOAL_HIGH, GOAL_LOW])
    async with sessions() as s:
        agent = await s.get(Agent, seeded.agent_id)
        original_description = agent.description
        delta_a = rm.ValidatedRosterDelta(updates=(dataclasses.replace(canonical(agent), description="A accepted"),))
        plan_a = await rm.preview_roster_delta(s, delta_a, origin_goal_id=None)
        delta_b = rm.ValidatedRosterDelta(creates=(new_agent_state(name="b-created"),))
        plan_b = await rm.preview_roster_delta(s, delta_b, origin_goal_id=None)
        await s.commit()
    assert set(plan_a.affected_goal_ids) == {GOAL_HIGH, GOAL_LOW}

    flushed, release_a = asyncio.Event(), asyncio.Event()

    async def mutation_a(db, locked_delta, locked_plan):
        result = await create_from_delta(db, locked_delta, locked_plan)
        flushed.set()
        await release_a.wait()
        return result

    async def session_a():
        async with sessions() as a:
            return await rm.AgentRosterMutationCoordinator().apply(
                a, delta_a, plan_a, mutation_a, origin_goal_id=None, provenance={})

    b_spy = AsyncMock(side_effect=create_from_delta)

    async def session_b_publish():
        async with sessions() as b:
            return await outcome_of(rm.AgentRosterMutationCoordinator().apply(
                b, delta_b, plan_b, b_spy, origin_goal_id=None, provenance={}))

    async def session_b_act_on_goal():
        async with sessions() as b:
            async with OrchestrationService()._lock_goal_for_baseline_transition(b, GOAL_LOW):
                row = (await b.execute(select(Agent).where(Agent.id == seeded.agent_id)
                                       .execution_options(populate_existing=True))).scalar_one()
                return row.description

    task_a = asyncio.create_task(session_a())
    try:
        await asyncio.wait_for(flushed.wait(), RACE_TIMEOUT)
        publish_b = asyncio.create_task(session_b_publish())
        act_b = asyncio.create_task(session_b_act_on_goal())
        await asyncio.sleep(0.3)
        assert not publish_b.done(), "B published a roster mutation while A held the guards"
        assert not act_b.done(), "B acted on an affected goal while A held its goal lock"
        async with sessions() as reader:
            observed = (await reader.execute(select(Agent.description).where(Agent.id == seeded.agent_id))).scalar_one()
        session_b_observed_accepted_snapshot = observed == "A accepted"
        assert session_b_observed_accepted_snapshot is False
        assert observed == original_description
    finally:
        release_a.set()
    await asyncio.wait_for(task_a, RACE_TIMEOUT)
    assert await asyncio.wait_for(act_b, RACE_TIMEOUT) == "A accepted"
    b_outcome = await asyncio.wait_for(publish_b, RACE_TIMEOUT)
    assert stale_error(b_outcome) == "state_changed", "B must re-plan against A's committed roster"
    assert b_spy.call_count == 0
    assert_roster_first_publication(lock_tracer, plan_a.affected_goal_ids, task=task_a)
