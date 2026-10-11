"""Deterministic file-SQLite provider/reset concurrency regressions."""
import asyncio
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from huddleroom.models.base import Base
from huddleroom.models.project import Project
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import OrchestrationProcessRun
from huddleroom.services.orchestration_debug_service import OrchestrationDebugService
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_baseline_runtime import await_baseline_completion

pytestmark = pytest.mark.asyncio

@pytest.mark.parametrize("operation_kind", ["step", "rerun", "retry", "tick"])
@pytest.mark.parametrize("provider_error", [False, True])
async def test_running_visible_reset_finishes_before_provider_and_discards_result(tmp_path, monkeypatch, provider_error, operation_kind):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'baseline.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with sessions() as db:
        project = Project(name="baseline concurrency", workspace_path=str(tmp_path))
        db.add(project)
        await db.flush()
        goal = OrchestrationGoal(project_id=project.id, objective="Test reset", status="active")
        db.add(goal)
        await db.flush()
        run = OrchestrationRun(goal_id=goal.id, status="running", baseline_authorized=True)
        db.add(run)
        await db.flush()
        if operation_kind == "rerun":
            prior = await OrchestrationProcessService().start_process(
                db, goal.id, process_type="goal_definition", trigger_reason="seed", run_id=run.id)
            await OrchestrationProcessService().complete_process(db, prior)
        if operation_kind == "retry":
            prior = await OrchestrationProcessService().start_process(
                db, goal.id, process_type="goal_definition", trigger_reason="seed", run_id=run.id)
            prior.outputs = {"_lm_retry": {
                "version": 1, "kind": "goal_analysis",
                "request": {"model": "test", "messages": [], "response_format": {"type": "json_object"}},
                "continuation": {"clarification_round": 0, "context": {},
                                 "working_goal": {"objective": "Test reset", "success_criteria": []}},
            }}
        await db.commit()
        project_id, goal_id, old_run_id = project.id, goal.id, run.id
    entered, release, ended = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def provider(**request):
        entered.set()
        await release.wait()
        ended.set()
        if provider_error:
            raise RuntimeError("late provider failure")
        return {"answer": "stale"}
    async def advance(self, db, goal, run, process_type):
        process = await OrchestrationProcessService().start_process(
            db, goal.id, process_type=process_type, trigger_reason="test", run_id=run.id)
        await await_baseline_completion(provider, {"messages": []})
        await OrchestrationProcessService().complete_process(db, process)
        return {"status": "completed"}
    monkeypatch.setattr(OrchestrationDebugService, "_advance", advance)
    from huddleroom.services.orchestration_goal_definition import GoalDefinitionProcess
    async def tick_advance(self, db, goal, run):
        return await advance(self, db, goal, run, "goal_definition")
    monkeypatch.setattr(GoalDefinitionProcess, "advance", tick_advance)
    async def retry_advance(self, db, goal, run, current):
        return await advance(self, db, goal, run, "goal_definition")
    monkeypatch.setattr(GoalDefinitionProcess, "retry_failed", retry_advance)
    async def step():
        async with sessions() as db:
            if operation_kind == "rerun":
                return await OrchestrationDebugService().rerun_last(db, project_id, goal_id, "goal_definition")
            if operation_kind == "retry":
                return await OrchestrationDebugService().retry_failed(db, project_id, goal_id, "goal_definition")
            if operation_kind == "tick":
                return await OrchestrationService().tick(db, old_run_id)
            return await OrchestrationDebugService().step(db, project_id, goal_id, "goal_definition")
    action = asyncio.create_task(step())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        async with sessions() as db:
            current = await db.scalar(select(OrchestrationProcessRun).where(OrchestrationProcessRun.goal_id == goal_id, OrchestrationProcessRun.superseded_by_id.is_(None)))
            assert current is not None and current.status == "running"
        async with sessions() as db:
            _, fresh = await asyncio.wait_for(OrchestrationService().reset_goal(db, project_id, goal_id), 3)
            fresh_id = fresh.id
            assert fresh_id != old_run_id
        if operation_kind == "tick":
            assert (await action)["interrupted"]
        else:
            with pytest.raises(HTTPException, match="409"):
                await action
        assert not release.is_set()
        release.set()
        await asyncio.wait_for(ended.wait(), 3)
        async with sessions() as db:
            assert list((await db.scalars(select(OrchestrationProcessRun).where(OrchestrationProcessRun.goal_id == goal_id, OrchestrationProcessRun.superseded_by_id.is_(None)))).all()) == []
            assert list((await db.scalars(select(OrchestrationRun.id).where(OrchestrationRun.goal_id == goal_id))).all()) == [fresh_id]
        # A fresh generation can execute immediately after reset.
        provider_error = False
        operation_kind = "step"
        await step()
    finally:
        release.set()
        await asyncio.gather(action, return_exceptions=True)
        await engine.dispose()

async def test_reset_wins_simultaneous_completion_and_explicit_transaction_ownership(db_session):
    from huddleroom.services.orchestration_baseline_runtime import baseline_operation, interrupt_baseline
    goal_id = uuid.uuid4()
    entered, release = asyncio.Event(), asyncio.Event()
    async def provider(**request):
        entered.set()
        await release.wait()
        return "discard me"
    async def action():
        async with baseline_operation(db_session, goal_id, owns_transaction=False):
            return await await_baseline_completion(provider, {})
    task = asyncio.create_task(action())
    await entered.wait()
    # Both become ready before the parent is scheduled again.
    release.set()
    reset = asyncio.create_task(interrupt_baseline(goal_id))
    with pytest.raises(HTTPException) as exc:
        await task
    assert exc.value.status_code == 409
    await reset

async def test_explicit_transaction_provider_boundary_never_commits(db_session, monkeypatch):
    from huddleroom.services.orchestration_baseline_runtime import baseline_operation
    async def forbidden_commit():
        pytest.fail("caller transaction was committed")
    monkeypatch.setattr(db_session, "commit", forbidden_commit)
    async def provider(**request):
        return "ok"
    async with baseline_operation(db_session, uuid.uuid4(), owns_transaction=False):
        assert await await_baseline_completion(provider, {}) == "ok"


@pytest.mark.parametrize("with_invocation", [False, True])
@pytest.mark.parametrize("provider_error", [False, True])
async def test_real_repair_boundary_discards_late_provider_without_application_callbacks(
    db_session, monkeypatch, with_invocation, provider_error,
):
    """Cancellation must bypass parsing, metadata observation and repair retries."""
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from huddleroom.services import orchestration_baseline_runtime as runtime
    from huddleroom.services.agent_response_stream import AgentResponseInvocation, InvocationContext
    from huddleroom.services.llm_structured_repair import complete_with_repair

    entered, release = asyncio.Event(), asyncio.Event()
    events, calls = [], []
    async def publish(event):
        events.append(event)
    async def no_reset(_project_id):
        await asyncio.Event().wait()
    monkeypatch.setattr("huddleroom.services.agent_response_relay.register_project_reset_monitor",
                        AsyncMock(return_value=SimpleNamespace(generation=None)))
    monkeypatch.setattr("huddleroom.services.agent_response_relay.unregister_project_reset_monitor", AsyncMock())
    monkeypatch.setattr("huddleroom.services.agent_response_stream.wait_for_project_reset", no_reset)
    invocation = AgentResponseInvocation(
        InvocationContext(uuid.uuid4(), "system", "test", None, "api", "test", "model"), publish=publish,
    ) if with_invocation else None
    async def provider(**request):
        assert runtime._current.get() is None
        assert "stream" not in request
        calls.append(request)
        entered.set()
        await release.wait()
        if provider_error:
            raise RuntimeError("late provider failure")
        return {"choices": [{"message": {"content": '{"result": "stale"}'}}]}
    parse, observe = Mock(side_effect=json.loads), Mock()
    goal_id = uuid.uuid4()
    async def action():
        async with runtime.baseline_operation(db_session, goal_id, owns_transaction=False):
            return await complete_with_repair(provider, {"messages": []}, parse,
                                              invocation=invocation, response_observer=observe)
    task = asyncio.create_task(action())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        provider_task = next(item for item in runtime._detached if not item.done())
        await asyncio.wait_for(runtime.interrupt_baseline(goal_id), 3)
        with pytest.raises(HTTPException) as exc:
            await task
        assert exc.value.status_code == 409
        assert not provider_task.done()
        event_count = len(events)
        drained = asyncio.Event()
        provider_task.add_done_callback(lambda _: drained.set())
        release.set()
        await asyncio.wait_for(drained.wait(), 3)
        assert provider_task not in runtime._detached
        assert len(calls) == 1
        parse.assert_not_called()
        observe.assert_not_called()
        assert len(events) == event_count
        assert not provider_task._log_traceback
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
