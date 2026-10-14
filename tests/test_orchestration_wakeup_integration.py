"""SPR213 cross-path wakeup checks against committed SQLite state and the real bus."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun
from huddleroom.services.event_bus import EventBusService, emit_event_once
from huddleroom.services.orchestration_continuous_service import OrchestrationContinuousService
from huddleroom.services.orchestration_service import GOAL_RESUMED_EVENT_TYPE, OrchestrationService
from huddleroom.workers import orchestration_tasks, scheduler
from tests.test_orchestration_continuous_service import started_continuous
from tests.test_orchestration_runtime_e2e import _agent, _authorized_run, _seed_accepted_plan


def _sessions(test_engine):
    return async_sessionmaker(test_engine, expire_on_commit=False)


async def _until(predicate):
    async def check():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(check(), 5)


def _bus_and_sessions(monkeypatch, test_engine):
    bus = EventBusService()
    from huddleroom.services import event_bus
    from huddleroom import database

    monkeypatch.setattr(event_bus, "_event_bus", bus)
    monkeypatch.setattr(orchestration_tasks, "get_event_bus", lambda: bus)
    monkeypatch.setattr(orchestration_tasks, "AsyncSessionLocal", _sessions(test_engine))
    monkeypatch.setattr(database, "AsyncSessionLocal", _sessions(test_engine))
    monkeypatch.setattr(orchestration_tasks.settings, "orchestration_event_coalesce_seconds", 0)
    return bus


@pytest.mark.asyncio
async def test_spr213_bootstrap_selection_preserves_paused_and_continuous_fences(
    db_session, test_project, test_engine, monkeypatch,
):
    rows = []
    for goal_status, run_status in [
        ("active", "running"), ("blocked", "blocked"), ("paused", "paused"),
        ("completed", "completed"), ("cancelled", "cancelled"),
    ]:
        goal = OrchestrationGoal(
            project_id=test_project.id, objective=goal_status, original_request=goal_status,
            success_criteria=[], constraints={}, budget={}, status=goal_status,
        )
        db_session.add(goal)
        await db_session.flush()
        run = OrchestrationRun(goal_id=goal.id, budget_state={}, status=run_status)
        db_session.add(run)
        rows.append(run)
    missing = OrchestrationGoal(
        project_id=test_project.id, objective="no run", original_request="no run",
        success_criteria=[], constraints={}, budget={}, status="active",
    )
    db_session.add(missing)
    continuous, waiting = await started_continuous(db_session, test_project)
    due = datetime.now(timezone.utc) - timedelta(hours=2)
    continuous.continuous_state = {**continuous.continuous_state, "next_due_at": due.isoformat()}
    not_due, unclaimed = await started_continuous(db_session, test_project)
    paused_continuous, paused_waiting = await started_continuous(db_session, test_project)
    paused_continuous.status = "paused"
    paused_waiting.status = "paused"
    paused_continuous.continuous_state = {
        **paused_continuous.continuous_state, "next_due_at": due.isoformat(),
    }
    await db_session.commit()

    ticked, order = [], []
    real_claim = OrchestrationContinuousService.claim_due_cycle

    async def claim(self, db, goal_id, now=None):
        order.append(("claim", goal_id))
        return await real_claim(self, db, goal_id, now)

    async def tick(self, db, run_id, *, local_only=False):
        ticked.append(run_id)
        order.append(("tick", run_id))
        return {"run_id": run_id}

    monkeypatch.setattr(OrchestrationContinuousService, "claim_due_cycle", claim)
    monkeypatch.setattr(OrchestrationService, "tick", tick)
    _bus_and_sessions(monkeypatch, test_engine)
    await scheduler.reconcile_orchestration_runs_job()

    assert set(ticked) == {rows[0].id, rows[1].id, waiting.id}
    assert order.index(("claim", continuous.id)) < order.index(("tick", waiting.id))
    assert unclaimed.id not in ticked
    assert ("tick", paused_waiting.id) not in order
    async with _sessions(test_engine)() as reader:
        assert (await reader.get(OrchestrationRun, waiting.id)).phase == "authorized"
        assert (await reader.get(OrchestrationRun, unclaimed.id)).phase == "waiting_activation"
        assert (await reader.get(OrchestrationGoal, not_due.id)).status == "active"
        assert (await reader.get(OrchestrationRun, paused_waiting.id)).status == "paused"
        assert (await reader.get(OrchestrationGoal, paused_continuous.id)).status == "paused"


@pytest.mark.asyncio
async def test_spr213_resume_is_commit_bound(
    db_session, test_project, test_engine, monkeypatch, stub_decision, safe_effectiveness_review_continue,
):
    stub_decision(lambda _context: {"action_type": "noop", "reason": "resume integration"})
    goals = []
    for _ in range(2):
        service, goal, run = await _authorized_run(db_session, test_project)
        await service.pause_goal(db_session, test_project.id, goal.id)
        goals.append((goal.id, run.id))
    await db_session.commit()
    bus = _bus_and_sessions(monkeypatch, test_engine)
    execution_ready, subscribed, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls, results, publications, receipts = [], [], [], []
    real_tick = OrchestrationService.tick
    real_put, real_subscribe = bus.put, bus.subscribe

    def recorded_put(event):
        if event.event_type == GOAL_RESUMED_EVENT_TYPE:
            publications.append(event.payload["run_id"])
        real_put(event)

    async def recorded_subscribe(*args, **kwargs):
        async for event in real_subscribe(*args, **kwargs):
            if event.event_type == GOAL_RESUMED_EVENT_TYPE:
                receipts.append(event.payload["run_id"])
            yield event

    monkeypatch.setattr(bus, "put", recorded_put)
    monkeypatch.setattr(bus, "subscribe", recorded_subscribe)

    async def tick(self, db, run_id, *, local_only=False):
        calls.append((run_id, local_only))
        result = await real_tick(self, db, run_id, local_only=local_only)
        results.append(result)
        completed.set()
        return result

    monkeypatch.setattr(OrchestrationService, "tick", tick)
    listener = asyncio.create_task(orchestration_tasks.run_orchestration_event_supervisor(
        ready=subscribed, execution_ready=execution_ready,
    ))
    try:
        await asyncio.wait_for(subscribed.wait(), 5)
        async with _sessions(test_engine)() as writer:
            await OrchestrationService().resume_goal(writer, test_project.id, goals[0][0])
            assert publications == receipts == calls == []
            await writer.commit()
        await _until(lambda: receipts == [str(goals[0][1])])
        async with _sessions(test_engine)() as writer:
            await OrchestrationService().resume_goal(writer, test_project.id, goals[1][0])
            await writer.rollback()
        assert publications == receipts == [str(goals[0][1])]
        assert calls == []
        execution_ready.set()
        await asyncio.wait_for(completed.wait(), 5)
        assert calls == [(goals[0][1], False)]
        assert len(results) == 1 and results[0]["run_id"] == goals[0][1]
        assert publications == receipts == [str(goals[0][1])]
        async with _sessions(test_engine)() as writer:
            with pytest.raises(HTTPException) as error:
                await OrchestrationService().resume_goal(writer, test_project.id, goals[0][0])
            assert error.value.status_code == 409
            await writer.rollback()
        assert bus._subscribers[0][0].empty()
        assert publications == receipts == [str(goals[0][1])]
        assert calls == [(goals[0][1], False)] and len(results) == 1
        async with _sessions(test_engine)() as reader:
            events = list(await reader.scalars(select(EventLog).where(
                EventLog.event_type == GOAL_RESUMED_EVENT_TYPE,
                EventLog.project_id == test_project.id,
            )))
        assert len(events) == 1 and events[0].payload["run_id"] == str(goals[0][1])
    finally:
        listener.cancel()
        with pytest.raises(asyncio.CancelledError):
            await listener


@pytest.mark.asyncio
async def test_spr213_startup_resume_periodic_effects_are_serialized(
    db_session, test_project, test_engine, monkeypatch, stub_decision, safe_effectiveness_review_continue,
):
    from huddleroom.models.orchestration import OrchestrationGate
    from huddleroom.models.session import Session
    from huddleroom.models.task import Task
    from huddleroom.services.orchestration_effectiveness_analyzer import EffectivenessAnalyzer
    from huddleroom.services.session_service import SessionService
    from huddleroom.services.task_service import TaskService

    monkeypatch.setattr(SessionService, "_schedule_dispatch_after_commit", lambda *_args: None)
    planner = _agent("spr213-planner", ["planning"])
    auditor = _agent("spr213-auditor", ["investigation"])
    db_session.add_all([planner, auditor])
    await db_session.flush()
    service, goal, run = await _authorized_run(db_session, test_project)
    audit_action = await service.execute_create_delegation_task_action(
        db_session, run.id,
        {"action_type": "create_delegation_task", "agent_id": str(auditor.id),
         "work_function": "investigation", "scope": "Audit the existing work.",
         "deliverable": "Concrete audit findings."},
        f"run:{run.id}:kind:create_delegation_task:pre-plan-audit",
    )
    audit_task = await db_session.get(Task, audit_action.target_id)
    await TaskService().auto_start(db_session, audit_task)
    assert len(list(await db_session.scalars(select(Session).where(Session.task_id == audit_task.id)))) == 1
    await _seed_accepted_plan(db_session, test_project, service, run, planner, plan_items=[{
        "id": "audit-0", "work_function": "investigation", "scope": "Audit the existing work.",
        "deliverable": "Concrete audit findings.", "why_runnable": "Already running.",
        "adopt_task_id": str(audit_task.id),
    }])
    planning_task_id = uuid.UUID(run.plan_state["planning_task_id"])
    await service.pause_goal(db_session, test_project.id, goal.id)
    await db_session.commit()

    bus = _bus_and_sessions(monkeypatch, test_engine)
    subscribed, execution_ready = asyncio.Event(), asyncio.Event()
    entered, release, recovery_release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    attempts, active, peak, exits, outcomes, fake_review_calls = [], 0, 0, 0, [], []
    real_lock = OrchestrationService._lock_single_goal
    real_tick = OrchestrationService.tick
    fake_review = EffectivenessAnalyzer.review_request

    async def recorded_fake_review(self, request, *, project_id=None):
        fake_review_calls.append(True)
        return await fake_review(self, request, project_id=project_id)

    async def measured_tick(self, db, run_id, *, local_only=False):
        try:
            result = await real_tick(self, db, run_id, local_only=local_only)
            outcomes.append(result)
            return result
        except Exception as error:
            outcomes.append(error)
            raise

    @asynccontextmanager
    async def measured_lock(self, db, goal_id, *, tick_owns_transaction=False):
        nonlocal active, peak, exits
        if goal_id == goal.id and tick_owns_transaction:
            attempts.append(asyncio.current_task())
        async with real_lock(self, db, goal_id, tick_owns_transaction=tick_owns_transaction):
            if goal_id == goal.id and tick_owns_transaction:
                active += 1
                peak = max(peak, active)
                try:
                    if not entered.is_set():
                        entered.set()
                        await release.wait()
                    yield
                finally:
                    active -= 1
                    exits += 1
            else:
                yield

    async def fake_recovery(_ready_at):
        await recovery_release.wait()

    class Scheduler:
        def resume_job(self, _job_id):
            pass

    monkeypatch.setattr(OrchestrationService, "_lock_single_goal", measured_lock)
    monkeypatch.setattr(OrchestrationService, "tick", measured_tick)
    monkeypatch.setattr(EffectivenessAnalyzer, "review_request", recorded_fake_review)
    monkeypatch.setattr(scheduler, "recover_orchestration_job", fake_recovery)
    stub_decision(lambda _context: {"action_type": "noop", "reason": "three-path replay"})
    listener = asyncio.create_task(orchestration_tasks.run_orchestration_event_supervisor(
        ready=subscribed, execution_ready=execution_ready,
    ))
    initial = interval = None
    try:
        await asyncio.wait_for(subscribed.wait(), 5)
        async with _sessions(test_engine)() as writer:
            await OrchestrationService().resume_goal(writer, test_project.id, goal.id)
            await writer.commit()
        await _until(lambda: bus._subscribers[0][0].empty())
        runtime_ready = asyncio.Event()
        runtime_ready.set()
        initial = asyncio.create_task(scheduler.bootstrap_orchestration_async(
            Scheduler(), datetime.now(timezone.utc), runtime_ready, execution_ready=execution_ready,
        ))
        recovery_release.set()
        await asyncio.wait_for(entered.wait(), 5)
        interval = asyncio.create_task(scheduler.reconcile_orchestration_runs_job())
        await _until(lambda: len(attempts) >= 3)
        assert active == peak == 1
        release.set()
        await asyncio.wait_for(asyncio.gather(initial, interval), 15)
        await _until(lambda: exits >= 3 and active == 0)
        async with _sessions(test_engine)() as reader:
            actions = list(await reader.scalars(select(OrchestrationAction).where(
                OrchestrationAction.run_id == run.id,
            )))
            tasks = list(await reader.scalars(select(Task).where(
                Task.metadata_["orchestration"]["run_id"].as_string() == str(run.id),
            )))
            gates = list(await reader.scalars(select(OrchestrationGate).where(OrchestrationGate.run_id == run.id)))
            sessions = list(await reader.scalars(select(Session).where(Session.task_id == audit_task.id)))
            fresh_run = await reader.get(OrchestrationRun, run.id)
            wake_events = list(await reader.scalars(select(EventLog).where(
                EventLog.project_id == test_project.id, EventLog.event_type == GOAL_RESUMED_EVENT_TYPE,
            )))
            tick_events = list(await reader.scalars(select(EventLog).where(
                EventLog.project_id == test_project.id, EventLog.event_type == "orchestration.tick",
                EventLog.payload["run_id"].as_string() == str(run.id),
            )))
        assert [(item["plan_item_id"], item["task_id"], item.get("adopted"))
                for item in fresh_run.plan_state["expanded_items"]] == [
            ("audit-0", str(audit_task.id), True),
        ]
        assert [action.idempotency_key for action in actions if action.action_type == "adopt_task"] == [
            f"run:{run.id}:kind:adopt_task:task:{audit_task.id}",
        ]
        assert [(action.idempotency_key, action.request["plan_item_id"]) for action in actions
                if action.action_type == "release_item"] == [
            (f"run:{run.id}:kind:release_item:audit-0", "audit-0"),
        ]
        assert {task.id for task in tasks} == {audit_task.id, planning_task_id}
        assert len(tasks) == 2 and len(sessions) == 1 and sessions[0].task_id == audit_task.id
        assert [(gate.gate_type, gate.success_criterion_key) for gate in gates
                if gate.gate_type == "work_completed"] == [("work_completed", "plan_item:audit-0")]
        assert len(wake_events) == 1 and wake_events[0].payload["run_id"] == str(run.id)
        assert len(outcomes) == 3 and all(isinstance(result, dict) for result in outcomes), outcomes
        emitted = [result for result in outcomes if result["tick_emitted"]]
        assert len(emitted) == 1 and emitted[0]["processed_events"] > 0
        consumed_cursor = emitted[0]["event_cursor"]
        assert fresh_run.event_cursor == consumed_cursor
        consumed_tick_events = [event for event in tick_events if event.dedup_key == (
            f"orchestration.tick:run:{run.id}:cursor:{consumed_cursor}"
        )]
        assert len(consumed_tick_events) == 1
        assert consumed_tick_events[0].payload["event_cursor"] == str(consumed_cursor)
        assert fake_review_calls
        assert peak == 1 and len(attempts) >= 3
    finally:
        release.set()
        recovery_release.set()
        for task in (initial, interval, listener):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalidation", ["pause", "cancel", "reset", "supersede", "workspace"])
async def test_spr213_stale_wakeup_does_not_dispatch(
    db_session, test_project, test_engine, monkeypatch, invalidation, safe_effectiveness_review_continue,
):
    from huddleroom.services.session_service import SessionService

    service, goal, run = await _authorized_run(db_session, test_project)
    await db_session.commit()
    bus = _bus_and_sessions(monkeypatch, test_engine)
    subscribed, execution_ready, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    provider_calls, dispatch_calls, results = [], [], []
    real_tick = OrchestrationService.tick

    async def tick(self, db, run_id, *, local_only=False):
        try:
            results.append(await real_tick(self, db, run_id, local_only=local_only))
        except HTTPException as error:
            results.append(error)
            raise
        finally:
            finished.set()

    async def forbidden_decision(self, db, run_id, adapter=None):
        provider_calls.append(run_id)
        raise AssertionError("stale wakeup reached provider decision")

    def forbidden_dispatch(*_args):
        dispatch_calls.append(True)
        raise AssertionError("stale wakeup scheduled a session")

    monkeypatch.setattr(OrchestrationService, "tick", tick)
    monkeypatch.setattr(OrchestrationService, "request_llm_decision", forbidden_decision)
    monkeypatch.setattr(SessionService, "_schedule_dispatch_after_commit", forbidden_dispatch)
    listener = asyncio.create_task(orchestration_tasks.run_orchestration_event_supervisor(
        ready=subscribed, execution_ready=execution_ready,
    ))
    try:
        await asyncio.wait_for(subscribed.wait(), 5)
        async with _sessions(test_engine)() as writer:
            await emit_event_once(writer, test_project.id, GOAL_RESUMED_EVENT_TYPE, {"run_id": str(run.id)})
            await writer.commit()
        await _until(lambda: bus._subscribers[0][0].empty())
        assert not finished.is_set()
        async with _sessions(test_engine)() as writer:
            changed = OrchestrationService()
            if invalidation == "pause":
                await changed.pause_goal(writer, test_project.id, goal.id)
            elif invalidation == "cancel":
                await changed.cancel_goal(writer, test_project.id, goal.id)
            elif invalidation == "reset":
                await changed.reset_goal(writer, test_project.id, goal.id)
            elif invalidation == "supersede":
                old_run = await writer.get(OrchestrationRun, run.id)
                old_run.status = "completed"
                writer.add(OrchestrationRun(
                    goal_id=goal.id, status="running", phase="baseline", supersedes_run_id=run.id,
                ))
            else:
                project = await writer.get(type(test_project), test_project.id)
                project.workspace_path = "/this/workspace/does/not/exist"
            await writer.commit()
        execution_ready.set()
        await asyncio.wait_for(finished.wait(), 5)
        assert provider_calls == dispatch_calls == []
        async with _sessions(test_engine)() as reader:
            assert await reader.scalar(select(func.count()).select_from(OrchestrationAction).where(
                OrchestrationAction.run_id == run.id,
                OrchestrationAction.action_type == "create_delegation_task",
            )) == 0
        if invalidation == "reset":
            assert isinstance(results[0], HTTPException) and results[0].status_code == 404
        elif invalidation == "workspace":
            assert isinstance(results[0], HTTPException) and results[0].status_code in (400, 409)
        else:
            assert results[0]["status"] == {
                "pause": "paused", "cancel": "cancelled", "supersede": "completed",
            }[invalidation]
    finally:
        listener.cancel()
        await asyncio.gather(listener, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("permission", ["unstarted", "exhausted"])
async def test_spr213_wakeup_keeps_authorization_and_budget_fences(
    db_session, test_project, test_engine, monkeypatch, permission, safe_effectiveness_review_continue,
):
    from huddleroom.models.session import Session
    from huddleroom.models.task import Task
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.session_service import SessionService
    from tests.conftest import complete_baseline_processes

    worker = _agent("spr213-permission-worker", ["planning"])
    db_session.add(worker)
    await db_session.flush()
    if permission == "exhausted":
        service, goal, run = await _authorized_run(db_session, test_project)
        await _seed_accepted_plan(db_session, test_project, service, run, worker)
        run.budget_state = {"status": "exceeded", "overridden": False}
    else:
        service = OrchestrationService()
        goal, run = await service.create_goal(
            db_session, project_id=test_project.id,
            data=OrchestrationGoalCreate(
                objective="Unstarted wakeup", success_criteria=[{"key": "done", "description": "Done"}],
            ), created_by_user_id=None,
        )
        run.baseline_authorized = True
        await complete_baseline_processes(db_session, goal, run)
        await service.tick(db_session, run.id)
        assert run.phase == "ready"
    await db_session.commit()

    bus = _bus_and_sessions(monkeypatch, test_engine)
    subscribed, execution_ready, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    provider_calls, dispatch_calls, results = [], [], []
    real_tick = OrchestrationService.tick

    async def tick(self, db, run_id, *, local_only=False):
        try:
            results.append(await real_tick(self, db, run_id, local_only=local_only))
        finally:
            finished.set()

    async def forbidden_decision(self, db, run_id, adapter=None):
        provider_calls.append(run_id)
        raise AssertionError("permission fence reached provider")

    def forbidden_dispatch(*_args):
        dispatch_calls.append(True)
        raise AssertionError("permission fence scheduled a session")

    monkeypatch.setattr(OrchestrationService, "tick", tick)
    monkeypatch.setattr(OrchestrationService, "request_llm_decision", forbidden_decision)
    monkeypatch.setattr(SessionService, "_schedule_dispatch_after_commit", forbidden_dispatch)
    listener = asyncio.create_task(orchestration_tasks.run_orchestration_event_supervisor(
        ready=subscribed, execution_ready=execution_ready,
    ))
    try:
        await asyncio.wait_for(subscribed.wait(), 5)
        async with _sessions(test_engine)() as writer:
            await emit_event_once(writer, test_project.id, GOAL_RESUMED_EVENT_TYPE, {"run_id": str(run.id)})
            await writer.commit()
        await _until(lambda: bus._subscribers[0][0].empty())
        execution_ready.set()
        await asyncio.wait_for(finished.wait(), 5)
        assert provider_calls == dispatch_calls == []
        async with _sessions(test_engine)() as reader:
            fresh = await reader.get(OrchestrationRun, run.id)
            actions = list(await reader.scalars(select(OrchestrationAction).where(
                OrchestrationAction.run_id == run.id,
                OrchestrationAction.action_type.in_(["release_item", "create_delegation_task"]),
            )))
            tasks = list(await reader.scalars(select(Task).where(
                Task.metadata_["orchestration"]["run_id"].as_string() == str(run.id),
            )))
            sessions = list(await reader.scalars(select(Session).where(Session.task_id.in_(t.id for t in tasks))))
        assert actions == []
        if permission == "unstarted":
            assert fresh.phase == "ready" and tasks == sessions == []
        else:
            assert results[0]["authorized_execution"]["step"] == "budget_exhausted"
            assert fresh.budget_state["overridden"] is False
            assert len(tasks) == 1 and sessions == []  # existing planner only
    finally:
        listener.cancel()
        await asyncio.gather(listener, return_exceptions=True)
