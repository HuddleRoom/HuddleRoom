"""Composed stale admission and real SQLite worker replacement ordering."""
import asyncio
import copy
import uuid
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationRun, OrchestrationDecision
from huddleroom.models.orchestration import OrchestrationGoal
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.models.event_log import EventLog
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher
from fastapi import HTTPException
from huddleroom.workers import task_runner
from tests.test_cli_cancellation_ownership import owned_cli, cleanup_runner
from tests.test_orchestration_plan_delegation_gate import _plan_request


@pytest.mark.asyncio
async def test_out_of_order_real_controller_completions_fence_late_request_with_live_cli(
    test_engine, tmp_path, monkeypatch,
):
    from types import SimpleNamespace
    from huddleroom.models.agent import Agent
    from tests.test_cli_workspace_isolation import _create_cli_session, _wait_for_path
    from tests.test_orchestration_plan_delegation_gate import _make_run, _bind_team
    from huddleroom.workers import session_tasks
    import huddleroom.services.orchestration_service as module
    heartbeat = tmp_path / "controller-worker-started"
    factory, sid, project_id = await _create_cli_session(
        test_engine, tmp_path,
        f"import pathlib,time\npathlib.Path({str(heartbeat)!r}).write_text('started')\ntime.sleep(60)\n", timeout=60,
    )
    monkeypatch.setattr(module, "AsyncSessionLocal", factory)
    monkeypatch.setattr(session_tasks, "AsyncSessionLocal", factory)
    runner = str(uuid.uuid4())
    async with factory() as seed:
        row = await seed.get(Session, sid)
        planner = await seed.get(Agent, row.agent_id)
        planner.role, planner.capabilities = "planner", ["planning"]
        service, goal, run = await _make_run(seed, project_id)
        await _bind_team(seed, goal, planner)
        run_id, agent_id = run.id, planner.id
        await seed.commit()
    ready = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    snapshots = []
    class Adapter:
        def __init__(self, index):
            self.index = index
        async def decide(self, context, **kwargs):
            snapshots.append(copy.deepcopy(context))
            ready[self.index].set()
            await release[self.index].wait()
            parsed = {**_plan_request(agent_id), "scope": f"controller plan {self.index}"}
            return SimpleNamespace(input_snapshot=context, parsed_decision=parsed, llm_output=parsed)
    pending = []
    async def request(index):
        async with factory() as caller:
            return await service.request_llm_decision(caller, run_id, adapter=Adapter(index))
    try:
        pending = [asyncio.create_task(request(index)) for index in range(2)]
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in ready)), 5)
        assert snapshots[0]["run"]["plan_state"] == snapshots[1]["run"]["plan_state"]
        assert not snapshots[0]["run"]["plan_state"].get("planning_task_id")
        release[0].set()
        first = await asyncio.wait_for(pending[0], 5)
        async with factory() as caller:
            run = await caller.get(OrchestrationRun, run_id)
            action = await OrchestrationDecisionDispatcher(service).dispatch(caller, run, first)
            row = await caller.get(Session, sid)
            row.task_id, row.runner_task_id = action.target_id, runner
            await caller.commit()
            state = copy.deepcopy(run.plan_state)
        await task_runner.dispatch_session(str(sid), "cli", project_id, runner)
        await _wait_for_path(heartbeat)
        # A normal current-generation evaluation occurs while the older completion
        # is still pending, as in the asynchronous controller topology.
        class Noop:
            async def decide(self, context, **kwargs):
                parsed = {"action_type": "noop", "reason": "Live planner is active."}
                return SimpleNamespace(input_snapshot=context, parsed_decision=parsed, llm_output=parsed)
        async with factory() as caller:
            current = await service.request_llm_decision(caller, run_id, adapter=Noop())
            assert current.input_snapshot["run"]["plan_state"]["planning_task_id"] == str(action.target_id)
        release[1].set()
        late = await asyncio.wait_for(pending[1], 5)
        service.request_llm_decision = AsyncMock(return_value=current)
        async with factory() as caller:
            run = await caller.get(OrchestrationRun, run_id)
            result = await service._dispatch_execution_decision(caller, run, late)
            assert result.action_type == "noop"
            await caller.commit()
        async with factory() as check:
            assert (await check.get(OrchestrationRun, run_id)).plan_state == state
            late_row = await check.get(OrchestrationDecision, late.id)
            assert late_row.rejection_reason == "stale_plan_generation"
            assert len(list(await check.scalars(select(Task).where(Task.project_id == project_id)))) == 1
            assert len(list(await check.scalars(select(Session).where(Session.project_id == project_id)))) == 1
            assert not list(await check.scalars(select(EventLog).where(
                EventLog.project_id == project_id, EventLog.event_type == "session.cancelled",
            )))
        assert task_runner.is_task_live(runner)
        service.request_llm_decision.assert_awaited_once()
    finally:
        for event in release:
            event.set()
        await asyncio.gather(*pending, return_exceptions=True)
        if task_runner.is_task_live(runner):
            await task_runner.cancel_task(runner)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_new_creation", [False, True])
@pytest.mark.parametrize("operation", ["request", "revision", "dispatch_revision"])
async def test_live_planner_stale_fence_then_acknowledged_independent_replacement(
    test_engine, tmp_path, monkeypatch, fail_new_creation, operation,
):
    from huddleroom.config import settings
    from huddleroom.services.orchestration_capacity import available_slots
    monkeypatch.setattr(settings, "orchestration_max_concurrent_tasks", 1)
    factory, sid, project_id, runner, heartbeat, pids = await owned_cli(test_engine, tmp_path, monkeypatch)
    prepared, release = asyncio.Event(), asyncio.Event()
    replacement = None
    new_runner = None
    original_commit = AsyncSession.commit
    async def commit(db):
        row = db.identity_map.get((Session, (sid,), None))
        if row is not None and row.status == "cancelled":
            prepared.set()
            await release.wait()
        await original_commit(db)
    monkeypatch.setattr(AsyncSession, "commit", commit)
    try:
        async with factory() as caller:
            old = await caller.get(Session, sid)
            old_task_id = old.task_id
            task = await caller.get(Task, old_task_id)
            run_id = uuid.UUID(task.metadata_["orchestration"]["run_id"])
            run = await caller.get(OrchestrationRun, run_id)
            goal = await caller.get(OrchestrationGoal, run.goal_id)
            goal.status, run.status = "active", "running"
            run.phase = "ready"  # fixture already completed every baseline process
            await caller.commit()
            service = OrchestrationService()
            goal, run = await service.start_run(caller, project_id, goal.id, actor="human:offline-test")
            assert await available_slots(caller, goal.id) == 0
            state = copy.deepcopy(run.plan_state)
            stale = OrchestrationDecision(run_id=run_id, decision_type="execution", validator_status="accepted",
                input_snapshot={"run": {"id": str(run_id), "plan_state": {}}},
                parsed_decision={**_plan_request(old.agent_id), "scope": "late distinct snapshot"})
            caller.add(stale)
            await caller.commit()
            with pytest.raises(Exception) as error:
                await OrchestrationDecisionDispatcher(service).dispatch(caller, run, stale)
            assert type(error.value).__name__ == "PlanGenerationChanged"
            assert task_runner.is_task_live(runner)
            await caller.refresh(run)
            assert run.plan_state == state
            await caller.commit()
            seals = []
            assert_generation = service._assert_current_plan_generation
            async def observe_generation(db, rid, expected):
                result = await assert_generation(db, rid, expected)
                seals.append((copy.deepcopy(expected),
                              await db.scalar(select(Session.status).where(Session.id == sid)),
                              await db.scalar(select(Task.status).where(Task.id == old_task_id))))
                return result
            monkeypatch.setattr(service, "_assert_current_plan_generation", observe_generation)
            key = f"replacement:{uuid.uuid4()}"
            if fail_new_creation:
                async def reject_creation(*args, **kwargs):
                    raise RuntimeError("injected replacement creation failure")
                monkeypatch.setattr(service, "reserve_action", reject_creation)
            if operation in {"revision", "dispatch_revision"}:
                request = {"action_type": "request_plan_revision", "plan_task_id": str(old_task_id),
                           "revision_request": "Deliberately revise the currently running plan."}
                method = service.execute_request_plan_revision_action
                if operation == "dispatch_revision":
                    revision = OrchestrationDecision(run_id=run_id, decision_type="execution", validator_status="accepted",
                        input_snapshot={"run": {"id": str(run_id), "plan_state": copy.deepcopy(state)}},
                        parsed_decision=request)
                    caller.add(revision)
                    await caller.commit()
                    key = OrchestrationDecisionDispatcher.action_key(run_id, "request_plan_revision",
                        service.canonical_decision_request("request_plan_revision", request, run_id=run_id))
                    async def dispatch_revision(db, **kwargs):
                        return await OrchestrationDecisionDispatcher(service).dispatch(db, run, revision)
                    method = dispatch_revision
            else:
                request = {**_plan_request(old.agent_id), "scope": "intentional replacement"}
                method = service.execute_request_plan_action
            replacement = asyncio.create_task(method(caller, run_id=run_id, request=request, idempotency_key=key))
            await asyncio.wait_for(prepared.wait(), 5)
            assert not replacement.done()
            assert not caller.in_transaction(), "replacement caller retains a SQLite transaction during worker commit"
            async with factory() as check:
                assert await check.scalar(select(OrchestrationAction.id).where(
                    OrchestrationAction.run_id == run_id, OrchestrationAction.idempotency_key == key,
                )) is None
                assert len(list(await check.scalars(select(Task).where(Task.project_id == project_id)))) == 1
                assert len(list(await check.scalars(select(Session).where(Session.project_id == project_id)))) == 1
            stopped = heartbeat.read_text()
            await asyncio.sleep(.08)
            assert heartbeat.read_text() == stopped
            release.set()
            if fail_new_creation:
                with pytest.raises(RuntimeError, match="injected replacement creation failure"):
                    await asyncio.wait_for(replacement, 5)
                await caller.rollback()
                action = None
            else:
                action = await asyncio.wait_for(replacement, 5)
                await caller.commit()
                await caller.refresh(run)
                successor_id = uuid.UUID(run.plan_state["planning_task_id"])
                replay = await method(caller, run_id=run_id, request=request, idempotency_key=key)
                assert replay.id == action.id
                await caller.commit()
            expected = service._plan_generation_seal(run_id, state)
            assert [entry[0] for entry in seals[:3]] == [expected] * 3
            assert [entry[1] for entry in seals[:3]] == ["running", "cancelled", "cancelled"]
            assert seals[2][2] == "cancelled", "old task commit must precede final generation admission"
            async with factory() as check:
                old_session = await check.get(Session, sid)
                old_task = await check.get(Task, old_task_id)
                assert old_session.status == old_task.status == "cancelled"
                assert old_session.ended_at is not None
                assert old_session.metadata_["attempt"]["result_status"] == "cancelled"
                events = list(await check.scalars(select(EventLog).where(
                    EventLog.project_id == project_id, EventLog.event_type == "session.cancelled",
                )))
                assert len(events) == 1
                if action is not None:
                    assert await check.get(Task, successor_id) is not None
                else:
                    assert await check.scalar(select(OrchestrationAction.id).where(
                        OrchestrationAction.run_id == run_id, OrchestrationAction.idempotency_key == key,
                    )) is None
                assert len(list(await check.scalars(select(Task).where(Task.project_id == project_id)))) == (
                    1 if fail_new_creation else 2)
            assert heartbeat.read_text() == stopped
            assert not task_runner.is_task_live(runner)
            if action is not None:
                # Exercise ordinary activation with the single root slot now freed.
                await caller.refresh(run)
                await caller.refresh(goal)
                assert await available_slots(caller, goal.id) == 1
                started = await service._start_released_tasks(caller, goal, run)
                if not started:
                    task = await caller.get(Task, successor_id)
                    pytest.fail(str({"goal_status": goal.status, "run_status": run.status,
                                     "run_phase": run.phase, "task_status": task.status,
                                     "blocker": await service._activation_blocker(caller, run, task)}))
                await caller.commit()
                async with factory() as check:
                    sessions = list(await check.scalars(select(Session).where(
                        Session.task_id == successor_id,
                    )))
                    assert len(sessions) == 1
                    new_runner = sessions[0].runner_task_id
                async with asyncio.timeout(5):
                    while not task_runner.is_task_live(new_runner):
                        await asyncio.sleep(.01)
                assert not task_runner.is_task_live(runner)
                assert task_runner.is_task_live(new_runner)
    finally:
        release.set()
        if replacement is not None and not replacement.done():
            replacement.cancel()
            await asyncio.gather(replacement, return_exceptions=True)
        await cleanup_runner(runner, pids)
        if new_runner:
            await cleanup_runner(new_runner, pids)


@pytest.mark.asyncio
@pytest.mark.parametrize("constraint", ["explicit_transaction", "pending_write", "flushed_write"])
async def test_sqlite_cli_replacement_rejects_caller_transaction_before_destructive_effects(
    test_engine, tmp_path, monkeypatch, constraint,
):
    factory, sid, project_id, runner, heartbeat, pids = await owned_cli(test_engine, tmp_path, monkeypatch)
    original_cancel = task_runner.cancel_task
    cancel = AsyncMock(side_effect=AssertionError("caller transaction cancelled existing runner"))
    monkeypatch.setattr(task_runner, "cancel_task", cancel)
    try:
        async with factory() as caller:
            if constraint == "explicit_transaction":
                await caller.begin()
            old = await caller.get(Session, sid)
            old_task_id = old.task_id
            task = await caller.get(Task, old_task_id)
            original_description = task.description
            run_id = uuid.UUID(task.metadata_["orchestration"]["run_id"])
            if constraint in {"pending_write", "flushed_write"}:
                task.description = "Uncommitted caller work must survive until caller rollback."
                if constraint == "flushed_write":
                    await caller.flush()
            with pytest.raises(HTTPException) as caught:
                await OrchestrationService().execute_request_plan_action(
                    caller, run_id=run_id, request={**_plan_request(old.agent_id), "scope": "forbidden replacement"},
                    idempotency_key=f"forbidden:{uuid.uuid4()}",
                )
            assert caught.value.status_code == 409
            assert "transaction" in str(caught.value.detail)
            cancel.assert_not_awaited()
            assert task_runner.is_task_live(runner)
            await caller.rollback()
        async with factory() as check:
            assert (await check.get(Session, sid)).status == "running"
            assert (await check.get(Task, old_task_id)).description == original_description
            assert len(list(await check.scalars(select(Task).where(Task.project_id == project_id)))) == 1
    finally:
        monkeypatch.setattr(task_runner, "cancel_task", original_cancel)
        await cleanup_runner(runner, pids)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["author", "work_function", "readiness"])
async def test_live_cli_replacement_admission_failure_does_not_cancel_existing_worker(
    test_engine, tmp_path, monkeypatch, invalid,
):
    factory, sid, project_id, runner, heartbeat, pids = await owned_cli(test_engine, tmp_path, monkeypatch)
    original_cancel = task_runner.cancel_task
    cancel = AsyncMock(side_effect=AssertionError("invalid request cancelled live planner"))
    monkeypatch.setattr(task_runner, "cancel_task", cancel)
    try:
        async with factory() as caller:
            old = await caller.get(Session, sid)
            task = await caller.get(Task, old.task_id)
            run_id = uuid.UUID(task.metadata_["orchestration"]["run_id"])
            request = {**_plan_request(old.agent_id), "scope": "invalid replacement"}
            service = OrchestrationService()
            if invalid == "author":
                request["agent_id"] = str(uuid.uuid4())
            elif invalid == "work_function":
                request["work_function"] = "validation"
            else:
                service._ensure_baseline_processes_ready = AsyncMock(
                    side_effect=HTTPException(status_code=409, detail="baseline_not_ready"))
            with pytest.raises(HTTPException):
                await service.execute_request_plan_action(
                    caller, run_id=run_id, request=request, idempotency_key=f"invalid:{uuid.uuid4()}",
                )
            cancel.assert_not_awaited()
            await caller.rollback()
        assert task_runner.is_task_live(runner)
        async with factory() as check:
            assert (await check.get(Session, sid)).status == "running"
            assert len(list(await check.scalars(select(Task).where(Task.project_id == project_id)))) == 1
    finally:
        monkeypatch.setattr(task_runner, "cancel_task", original_cancel)
        await cleanup_runner(runner, pids)


@pytest.mark.asyncio
async def test_direct_raw_steering_plan_key_replay_does_not_cancel_live_successor(
    test_engine, tmp_path, monkeypatch,
):
    from huddleroom.services.orchestration_steering import OrchestrationSteeringService
    factory, sid, project_id, runner, heartbeat, pids = await owned_cli(test_engine, tmp_path, monkeypatch)
    new_runner = str(uuid.uuid4())
    original_cancel = task_runner.cancel_task
    try:
        async with factory() as caller:
            old = await caller.get(Session, sid)
            task = await caller.get(Task, old.task_id)
            run_id = uuid.UUID(task.metadata_["orchestration"]["run_id"])
            run = await caller.get(OrchestrationRun, run_id)
            goal = await caller.get(OrchestrationGoal, run.goal_id)
            service = OrchestrationService()
            versions = await OrchestrationSteeringService(service).current_versions(caller, goal, run)
            request = service._canonical_plan_request({**_plan_request(old.agent_id), "scope": "steered successor"})
            decision = OrchestrationDecision(run_id=run_id, decision_type="execution", validator_status="accepted",
                input_snapshot={"run": {"id": str(run_id), "plan_state": copy.deepcopy(run.plan_state)},
                                "steering": {"versions": vars(versions), "active_directions": []}},
                parsed_decision=request)
            caller.add(decision)
            await caller.commit()
            raw_key = f"steered:{uuid.uuid4()}"
            action = await service.execute_request_plan_action(
                caller, run_id=run_id, request=request, idempotency_key=raw_key, decision_id=decision.id,
            )
            assert action.idempotency_key.startswith(raw_key + ":steering:")
            successor = Session(agent_id=old.agent_id, project_id=project_id, task_id=action.target_id,
                                adapter_type="cli", status="pending", runner_task_id=new_runner,
                                metadata_={"_run_config": {"timeout": 60}})
            caller.add(successor)
            await caller.commit()
            successor_id = successor.id
            state = copy.deepcopy(run.plan_state)
        await task_runner.dispatch_session(str(successor_id), "cli", project_id, new_runner)
        async with asyncio.timeout(5):
            while True:
                async with factory() as check:
                    row = await check.get(Session, successor_id)
                    if row.status == "running":
                        break
                await asyncio.sleep(.01)
        cancel = AsyncMock(side_effect=AssertionError("steering-key exact replay cancelled successor"))
        monkeypatch.setattr(task_runner, "cancel_task", cancel)
        async with factory() as caller:
            replay = await service.execute_request_plan_action(
                caller, run_id=run_id, request=request, idempotency_key=raw_key, decision_id=decision.id,
            )
            assert replay.id == action.id
            assert (await caller.get(OrchestrationRun, run_id)).plan_state == state
            cancel.assert_not_awaited()
        assert task_runner.is_task_live(new_runner)
    finally:
        monkeypatch.setattr(task_runner, "cancel_task", original_cancel)
        await cleanup_runner(runner, pids)
        await cleanup_runner(new_runner, pids)
