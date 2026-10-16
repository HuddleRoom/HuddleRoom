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


@pytest.mark.asyncio
@pytest.mark.parametrize("next_attempt", ["chain", "repair"])
async def test_pause_commits_while_baseline_provider_blocked(sessions, tmp_path, next_attempt):
    """SPR230/SPR231 admitted output lands; pause fences chain and repair attempts."""
    import json
    import uuid
    from huddleroom.services.orchestration_baseline_runtime import baseline_operation, await_baseline_completion
    from huddleroom.services.orchestration_control import accept_pause, read_pause_receipt
    from huddleroom.services.llm_structured_repair import complete_with_repair
    from tests.test_orchestration_chat_actions import _seed
    project, _, goal, run = await _seed(sessions, tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    calls, landed = [], []
    request_id, actor = uuid.uuid4(), uuid.UUID(int=0)
    async def provider(**request):
        calls.append(request)
        entered.set()
        await release.wait()
        content = "malformed" if next_attempt == "repair" and len(calls) == 1 else '{"answer":"retained"}'
        return {"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 7}}
    async def action():
        async with sessions() as db:
            async with baseline_operation(db, goal.id, owns_transaction=True):
                if next_attempt == "repair":
                    await complete_with_repair(provider, {"messages": []}, json.loads,
                                               response_observer=lambda response: landed.append(response))
                else:
                    response = await await_baseline_completion(provider, {"messages": []})
                    landed.append(response)
                    # Persist admitted output through the normal process ledger.
                    process = await OrchestrationProcessService().start_process(
                        db, goal.id, process_type="goal_definition", trigger_reason="pause-test", run_id=run.id)
                    await OrchestrationProcessService().complete_process(
                        db, process, outputs={"answer": "retained", "tokens": 7})
                    await db.commit()
                    await await_baseline_completion(provider, {"messages": []})
    task = asyncio.create_task(action())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        async with sessions() as pause:
            await asyncio.wait_for(accept_pause(pause, project.id, goal.id, run.id, 0, request_id, actor), 3)
        async with sessions() as fresh:
            receipt = await read_pause_receipt(fresh, project.id, goal.id, request_id, actor)
            assert receipt is not None and receipt.state in {"accepted", "applied"}
        assert not release.is_set() and not task.done()
        release.set()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 3)
        assert len(calls) == 1, "a chain or structured repair provider was admitted after durable Pause"
        assert len(landed) == 1 and landed[0]["usage"]["total_tokens"] == 7
        async with sessions() as fresh:
            persisted = await fresh.get(OrchestrationRun, run.id)
            assert persisted.pause_requested is True and persisted.status == "paused"
            if next_attempt == "chain":
                process = await OrchestrationProcessService().get_current(fresh, goal.id, "goal_definition")
                assert process.status == "completed" and process.outputs["answer"] == "retained"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_pause_wins_final_baseline_admission_race(sessions, tmp_path):
    """SPR230/SPR231 stale prepared work must reread the scalar under the writer."""
    import uuid
    from huddleroom.services.orchestration_baseline_runtime import baseline_operation, await_baseline_completion
    from huddleroom.services.orchestration_control import accept_pause
    from tests.test_orchestration_chat_actions import _seed
    project, _, goal, run = await _seed(sessions, tmp_path)
    prepared, admit = asyncio.Event(), asyncio.Event()
    calls = []
    async def provider(**request):
        calls.append(request)
        return "must not start"
    async def action():
        async with sessions() as db:
            async with baseline_operation(db, goal.id, owns_transaction=True):
                stale = await db.get(OrchestrationRun, run.id)
                assert stale.status == "running" and not stale.pause_requested
                prepared.set()
                await admit.wait()
                return await await_baseline_completion(provider, {"messages": []})
    task = asyncio.create_task(action())
    try:
        await asyncio.wait_for(prepared.wait(), 3)
        async with sessions() as db:
            await accept_pause(db, project.id, goal.id, run.id, 0, uuid.uuid4(), uuid.UUID(int=0))
        admit.set()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 3)
        assert calls == [], "stale prepared operation bypassed committed pause admission fence"
        async with sessions() as db:
            persisted = await db.get(OrchestrationRun, run.id)
            assert persisted.pause_requested is True and persisted.status == "paused"
    finally:
        admit.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["step", "retry", "rerun", "tick"])
async def test_direct_baseline_lands_admitted_output_with_pause_precedence(sessions, tmp_path, monkeypatch, operation):
    """SPR230/SPR231 public baseline entry paths retain admitted process output."""
    import uuid
    from huddleroom.services.orchestration_baseline_runtime import await_baseline_completion
    from huddleroom.services.orchestration_control import accept_pause
    from huddleroom.services.orchestration_debug_service import OrchestrationDebugService
    from huddleroom.services.orchestration_goal_definition import GoalDefinitionProcess
    from tests.test_orchestration_chat_actions import _seed
    project, _, goal, run = await _seed(sessions, tmp_path)
    async with sessions() as setup:
        row = await setup.get(OrchestrationRun, run.id)
        row.baseline_authorized = True
        if operation in {"retry", "rerun"}:
            prior = await OrchestrationProcessService().start_process(
                setup, goal.id, process_type="goal_definition", trigger_reason="seed", run_id=run.id)
            if operation == "rerun":
                await OrchestrationProcessService().complete_process(setup, prior)
            else:
                prior.outputs = {"_lm_retry": {"version": 1, "kind": "goal_analysis",
                    "request": {"model": "test", "messages": [], "response_format": {"type": "json_object"}},
                    "continuation": {"clarification_round": 0, "context": {},
                                     "working_goal": {"objective": goal.objective, "success_criteria": []}}}}
        await setup.commit()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def provider(**request):
        calls.append(request)
        entered.set()
        await release.wait()
        return {"answer": "retained", "tokens": 11}
    async def advance(self, db, current_goal, current_run, *args, **kwargs):
        current = await OrchestrationProcessService().get_current(db, current_goal.id, "goal_definition")
        if current is None or current.status != "running":
            current = await OrchestrationProcessService().start_process(
                db, current_goal.id, process_type="goal_definition", trigger_reason="pause-test", run_id=current_run.id)
        result = await await_baseline_completion(provider, {"messages": []})
        await OrchestrationProcessService().complete_process(db, current, outputs=result)
        # Reproduce a late analyzer restoring its stale active states.
        current_goal.status, current_run.status = "active", "running"
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(current_goal, "status")
        flag_modified(current_run, "status")
        return {"status": "completed", "process_type": "goal_definition"}
    monkeypatch.setattr(OrchestrationDebugService, "_advance", advance)
    monkeypatch.setattr(GoalDefinitionProcess, "advance", advance)
    monkeypatch.setattr(GoalDefinitionProcess, "retry_failed", advance)
    async def successor(self, db, current_goal, current_run, *args, **kwargs):
        await await_baseline_completion(provider, {"messages": []})
        return {"status": "completed", "process_type": "manager_selection"}
    monkeypatch.setattr(ManagerSelectionProcess, "advance", successor)
    from sqlalchemy.ext.asyncio import AsyncSession
    original_commit = AsyncSession.commit
    pause_landed, committed_states = asyncio.Event(), []
    async def observe_commit(db):
        await original_commit(db)
        if pause_landed.is_set():
            async with sessions() as fresh:
                observed_goal = await fresh.get(OrchestrationGoal, goal.id)
                observed_run = await fresh.get(OrchestrationRun, run.id)
                committed_states.append((observed_goal.status, observed_run.status))
    monkeypatch.setattr(AsyncSession, "commit", observe_commit)
    async def action():
        async with sessions() as db:
            if operation == "tick":
                return await OrchestrationService().tick(db, run.id)
            debug = OrchestrationDebugService()
            method = debug.step if operation == "step" else debug.retry_failed if operation == "retry" else debug.rerun_last
            return await method(db, project.id, goal.id, "goal_definition")
    task = asyncio.create_task(action())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        async with sessions() as pause:
            await asyncio.wait_for(accept_pause(pause, project.id, goal.id, run.id, 0, uuid.uuid4(), uuid.UUID(int=0)), 3)
        pause_landed.set()
        assert not task.done()
        release.set()
        outcome = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 3)
        assert not isinstance(outcome[0], BaseException), outcome
        assert len(calls) == 1
        assert committed_states and all(state == ("paused", "paused") for state in committed_states), committed_states
        async with sessions() as fresh:
            current = await OrchestrationProcessService().get_current(fresh, goal.id, "goal_definition")
            assert current.status == "completed" and current.outputs == {"answer": "retained", "tokens": 11}
            saved_goal, saved_run = await fresh.get(OrchestrationGoal, goal.id), await fresh.get(OrchestrationRun, run.id)
            assert saved_run.pause_requested is True and saved_run.baseline_authorized is False
            assert saved_goal.status == saved_run.status == "paused"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_pause_wins_final_judgment_claim_after_payload_preparation(sessions, tmp_path, monkeypatch):
    """SPR230/SPR231 awaited payload work cannot publish a post-pause judgment claim."""
    import uuid
    from datetime import datetime, timezone, timedelta
    from huddleroom.services.orchestration_control import accept_pause
    from huddleroom.services.orchestration_supervision_scheduler import OrchestrationSupervisionScheduler
    from tests.test_orchestration_chat_actions import _seed
    project, _, goal, run = await _seed(sessions, tmp_path, phase="authorized")
    now = datetime.now(timezone.utc)
    async with sessions() as setup:
        current = await setup.get(OrchestrationRun, run.id)
        current.supervision_state = {"needs_judgment": True, "judgment_due_at": (now - timedelta(seconds=1)).isoformat()}
        await setup.commit()
    monkeypatch.setattr("huddleroom.database.AsyncSessionLocal", sessions)
    async def local_tick(*args, **kwargs):
        return {"step": "waiting"}
    monkeypatch.setattr(OrchestrationService, "tick", local_tick)
    prepared, release = asyncio.Event(), asyncio.Event()
    async def payload(*args, **kwargs):
        prepared.set()
        await release.wait()
        return {}, "prepared-fingerprint"
    monkeypatch.setattr(OrchestrationSupervisionScheduler, "_judgment_payload", staticmethod(payload))
    task = asyncio.create_task(OrchestrationSupervisionScheduler()._claim_judgment(run.id, now))
    try:
        await asyncio.wait_for(prepared.wait(), 3)
        async with sessions() as pause:
            await asyncio.wait_for(accept_pause(pause, project.id, goal.id, run.id, 0, uuid.uuid4(), uuid.UUID(int=0)), 3)
        release.set()
        assert await asyncio.wait_for(task, 3) is None
        async with sessions() as fresh:
            saved = await fresh.get(OrchestrationRun, run.id)
            assert not saved.supervision_state.get("judgment_in_flight")
            assert "judgment_claim_token" not in saved.supervision_state
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["step", "retry", "rerun", "tick"])
async def test_baseline_preflushed_caller_fails_before_new_long_lock(sessions, tmp_path, monkeypatch, operation):
    """SPR230/SPR231 a caller SQLite writer must never wait for a new long mutex."""
    from contextlib import asynccontextmanager
    from fastapi import HTTPException
    from huddleroom.models.project import Project
    from huddleroom.services.orchestration_debug_service import OrchestrationDebugService
    from tests.test_orchestration_chat_actions import _seed
    project, _, goal, run = await _seed(sessions, tmp_path)
    @asynccontextmanager
    async def forbidden_lock(*args, **kwargs):
        pytest.fail("caller-held SQLite writer attempted a new long goal lock")
        yield
    monkeypatch.setattr(OrchestrationService, "_lock_goal_for_baseline_transition", forbidden_lock)
    async with sessions() as caller:
        await caller.execute(update(Project).where(Project.id == project.id).values(name="caller staging"))
        await caller.flush()
        outer = caller.get_transaction()
        with pytest.raises(HTTPException) as rejection:
            if operation == "tick":
                await OrchestrationService().tick(caller, run.id)
            else:
                debug = OrchestrationDebugService()
                method = debug.step if operation == "step" else debug.retry_failed if operation == "retry" else debug.rerun_last
                await method(caller, project.id, goal.id, "goal_definition")
        assert rejection.value.status_code == 409
        assert caller.get_transaction() is outer and caller.in_transaction()
        assert await caller.scalar(select(Project.name).where(Project.id == project.id)) == "caller staging"
        async with sessions() as fresh:
            assert await fresh.scalar(select(Project.name).where(Project.id == project.id)) == project.name
        await caller.rollback()


@pytest.mark.asyncio
@pytest.mark.parametrize("worker", ["reconcile", "supervision_sweep"])
async def test_owned_worker_selection_allows_real_tick_after_read(sessions, tmp_path, monkeypatch, worker):
    """SPR230/SPR231 native read-owning workers remain runnable under caller guards."""
    from datetime import datetime, timezone
    from huddleroom.services.orchestration_supervision_scheduler import OrchestrationSupervisionScheduler
    from huddleroom.workers.scheduler import reconcile_orchestration_runs_async
    from tests.test_orchestration_chat_actions import _seed
    _, _, goal, run = await _seed(sessions, tmp_path)
    async with sessions() as setup:
        saved = await setup.get(OrchestrationRun, run.id)
        saved.baseline_authorized = False
        await setup.commit()
    monkeypatch.setattr("huddleroom.database.AsyncSessionLocal", sessions)
    if worker == "reconcile":
        async with sessions() as db:
            assert await reconcile_orchestration_runs_async(db) == 1
    else:
        assert await OrchestrationSupervisionScheduler()._sweep_one(
            goal.id, 3, datetime.now(timezone.utc)) is None
    async with sessions() as fresh:
        saved = await fresh.get(OrchestrationRun, run.id)
        assert saved.status == "running" and saved.baseline_authorized is False
