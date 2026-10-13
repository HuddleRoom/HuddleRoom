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


# ---------------------------------------------------------------------------
# SPR 180/184: roster mutation transaction ownership. The coordinator
# owns its transaction in apply(); an explicit caller-owned root transaction must
# keep mutation_guard() across its real outer commit/rollback.
# ---------------------------------------------------------------------------
import dataclasses  # noqa: E402
from unittest.mock import AsyncMock  # noqa: E402

from huddleroom.models.agent import Agent  # noqa: E402
from tests.test_orchestration_roster_mutation import (  # noqa: E402
    GOAL_LOW,
    RACE_TIMEOUT,
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


async def _update_and_create_plans(sessions, tmp_path):
    rm = roster_api()
    seeded = await seed_roster_goals(sessions, tmp_path, goal_ids=[GOAL_LOW])
    async with sessions() as s:
        agent = await s.get(Agent, seeded.agent_id)
        update_delta = rm.ValidatedRosterDelta(
            updates=(dataclasses.replace(canonical(agent), description="caller edit"),))
        update_plan = await rm.preview_roster_delta(s, update_delta, origin_goal_id=None)
        create_delta = rm.ValidatedRosterDelta(creates=(new_agent_state(name="contender"),))
        create_plan = await rm.preview_roster_delta(s, create_delta, origin_goal_id=None)
        await s.commit()
        return rm, seeded, agent.description, (update_delta, update_plan), (create_delta, create_plan)


@pytest.mark.parametrize("outer", ["commit", "rollback"])
async def test_caller_owned_root_transaction_keeps_roster_guard_through_outer_commit(
    sessions, tmp_path, lock_tracer, outer
):
    rm, seeded, original, (delta, plan), (other_delta, other_plan) = await _update_and_create_plans(sessions, tmp_path)
    lock_tracer.start()
    contender_spy = AsyncMock(side_effect=create_from_delta)

    async def contender():
        async with sessions() as b:
            return await outcome_of(rm.AgentRosterMutationCoordinator().apply(
                b, other_delta, other_plan, contender_spy, origin_goal_id=None, provenance={}))

    async with sessions() as a:
        await a.begin()
        async with rm.AgentRosterMutationCoordinator().mutation_guard(
            a, delta, plan, origin_goal_id=None, transaction_owner="caller"
        ) as locked:
            assert locked.delta == delta and locked.plan == plan
            await create_from_delta(a, locked.delta, locked.plan)
            task_b = asyncio.create_task(contender())
            await asyncio.sleep(0.2)
            assert not task_b.done()
            await (a.commit() if outer == "commit" else a.rollback())
            await asyncio.sleep(0.2)
            assert not task_b.done(), "guard released before the caller left mutation_guard"
    contender_outcome = await asyncio.wait_for(task_b, RACE_TIMEOUT)

    expected = "caller edit" if outer == "commit" else original
    assert (await fresh_agent(sessions, seeded.agent_id)).description == expected
    if outer == "commit":
        assert stale_error(contender_outcome) == "state_changed" and contender_spy.call_count == 0
    else:
        assert stale_error(contender_outcome) is None and contender_spy.call_count == 1


async def test_caller_owned_savepoint_release_is_not_publication(sessions, tmp_path, lock_tracer):
    rm, seeded, original, (delta, plan), (other_delta, other_plan) = await _update_and_create_plans(sessions, tmp_path)
    lock_tracer.start()

    async with sessions() as a:
        await a.begin()
        with pytest.raises(Exception):
            async with rm.AgentRosterMutationCoordinator().mutation_guard(
                a, delta, plan, origin_goal_id=None, transaction_owner="caller"
            ) as locked:
                async with a.begin_nested():
                    await create_from_delta(a, locked.delta, locked.plan)
                # Savepoint released and flushed, root transaction still open: not publication.
    assert (await fresh_agent(sessions, seeded.agent_id)).description == original
    # The guard was still released: a later coordinated mutation proceeds.
    async with sessions() as b:
        result = await asyncio.wait_for(rm.AgentRosterMutationCoordinator().apply(
            b, other_delta, other_plan, create_from_delta, origin_goal_id=None, provenance={}), RACE_TIMEOUT)
    assert stale_error(result) is None
    assert await fresh_agent(sessions, other_delta.creates[0].id) is not None


async def test_apply_rejects_preexisting_explicit_root_transaction_before_effects(sessions, tmp_path, lock_tracer):
    rm, seeded, original, (delta, plan), _ = await _update_and_create_plans(sessions, tmp_path)
    lock_tracer.start()
    spy = AsyncMock(side_effect=create_from_delta)

    async with sessions() as a:
        await a.begin()
        outcome = await outcome_of(rm.AgentRosterMutationCoordinator().apply(
            a, delta, plan, spy, origin_goal_id=None, provenance={}))
        assert isinstance(outcome, Exception), "apply must reject a caller-owned root transaction"
        assert a.in_transaction(), "the coordinator must not commit/close the caller's transaction"
        await a.rollback()
    assert spy.call_count == 0
    assert not lock_tracer.of("agent:write")
    assert (await fresh_agent(sessions, seeded.agent_id)).description == original


@pytest.mark.parametrize(("owner", "shape"), [
    ("caller", "no_transaction"), ("caller", "autobegin"), ("coordinator", "explicit_root"),
])
async def test_mutation_guard_rejects_unsupported_transaction_shapes_before_effects(
    sessions, tmp_path, lock_tracer, owner, shape
):
    rm, seeded, original, (delta, plan), _ = await _update_and_create_plans(sessions, tmp_path)
    lock_tracer.start()
    entered = []

    async with sessions() as a:
        if shape == "explicit_root":
            await a.begin()
        elif shape == "autobegin":
            await a.get(Agent, seeded.agent_id)
        with pytest.raises(Exception):
            async with rm.AgentRosterMutationCoordinator().mutation_guard(
                a, delta, plan, origin_goal_id=None, transaction_owner=owner
            ) as locked:
                entered.append(locked)
                await create_from_delta(a, locked.delta, locked.plan)
        await a.rollback()
    assert entered == []
    assert not lock_tracer.of("agent:write")
    assert (await fresh_agent(sessions, seeded.agent_id)).description == original


async def test_observations_cross_empty_context_and_stop_immediately_on_reset(db_session):
    from contextvars import ContextVar
    from huddleroom.services import orchestration_baseline_runtime as runtime
    sentinel = ContextVar("spr189_sentinel", default=None)
    release, seen, ended = asyncio.Event(), asyncio.Event(), asyncio.Event()
    records = []
    async def provider(*, observer=None):
        assert runtime._current.get() is None and sentinel.get() is None
        await observer("output", "live\n")
        await release.wait()
        await observer("output", "discarded\n")
        ended.set()
        raise RuntimeError("late error must be drained")
    async def observer(stream, text):
        assert runtime._current.get() is not None and sentinel.get() == "parent"
        records.append((stream, text))
        seen.set()
    goal_id = uuid.uuid4()
    async def action():
        token = sentinel.set("parent")
        try:
            async with runtime.baseline_operation(db_session, goal_id, owns_transaction=False):
                await runtime.await_baseline_completion(provider, {}, observer=observer)
        finally:
            sentinel.reset(token)
    task = asyncio.create_task(action())
    try:
        await asyncio.wait_for(seen.wait(), 1)
        await asyncio.wait_for(runtime.interrupt_baseline(goal_id), 1)
        with pytest.raises(HTTPException):
            await task
        assert not release.is_set()
        release.set()
        await asyncio.wait_for(ended.wait(), 1)
        await asyncio.sleep(0)
        assert records == [("output", "live\n")]
        assert goal_id not in runtime._active
        assert not runtime._detached
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_observer_is_not_forwarded_to_unadvertised_injected_completion(db_session):
    from huddleroom.services import orchestration_baseline_runtime as runtime
    received = []
    async def provider(**request):
        received.append(request)
        return "final"
    async def observer(*_args):
        pytest.fail("nonobservable completion emitted an update")
    async with runtime.baseline_operation(db_session, uuid.uuid4(), owns_transaction=False):
        assert await runtime.await_baseline_completion(provider, {"messages": []}, observer=observer) == "final"
    assert received == [{"messages": []}]


async def test_observation_queue_overflow_drops_display_without_blocking_final(db_session, monkeypatch):
    from huddleroom.services import orchestration_baseline_runtime as runtime
    from huddleroom.services.agent_response_stream import AgentResponseInvocation, InvocationContext
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    async def no_reset(_project):
        await asyncio.Event().wait()
    monkeypatch.setattr("huddleroom.services.agent_response_relay.register_project_reset_monitor",
                        AsyncMock(return_value=SimpleNamespace(generation=None)))
    monkeypatch.setattr("huddleroom.services.agent_response_relay.unregister_project_reset_monitor", AsyncMock())
    monkeypatch.setattr("huddleroom.services.agent_response_stream.wait_for_project_reset", no_reset)
    records = []
    async def publish(event):
        records.append(event)
    final = {"choices": [{"message": {"content": "authoritative"}}]}
    async def provider(*, observer=None):
        # No await that yields between observations: maxsize32 must cut off rather than block.
        for index in range(100):
            await observer("output", str(index) + "\n")
        return final
    invocation = AgentResponseInvocation(
        InvocationContext(uuid.uuid4(), "system", "test", None, "api", "test", "model"), publish=publish)
    call = invocation.call(messages=[])
    async with call:
        async with runtime.baseline_operation(db_session, uuid.uuid4(), owns_transaction=False):
            assert await asyncio.wait_for(runtime.await_baseline_completion(provider, {}, observer=call.output), 1) is final
    outputs = [event for event in records if event.event_type == "agent_response.output"]
    assert 0 < len(outputs) <= 32
    assert all(len(event.payload["text"].encode()) <= 16384 for event in outputs)
    terminal = [event for event in records if event.event_type == "agent_response.terminal"]
    assert len(terminal) == 1
    assert sum(event.payload.get("observation_truncated") is True for event in records) == 1
    assert "99\n" not in "".join(event.payload["text"] for event in outputs)


async def test_default_api_iteration_and_reconstruction_are_provider_owned(db_session, monkeypatch):
    from contextvars import ContextVar
    from types import SimpleNamespace
    from huddleroom.config import settings
    from huddleroom.services import orchestration_baseline_runtime as runtime
    from huddleroom.services import orchestration_completion as completion
    sentinel = ContextVar("spr189_api_parent", default=None)
    first_seen, release = asyncio.Event(), asyncio.Event()
    final = {"choices": [{"message": {"content": "live"}}], "usage": {"total_tokens": 11}}
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="live\n", reasoning_content="think\n"))])
    async def chunks():
        assert runtime._current.get() is None and sentinel.get() is None
        yield chunk
        await release.wait()
    async def api(**request):
        assert request["stream"] is True
        assert "observer" not in request
        assert runtime._current.get() is None and sentinel.get() is None
        return chunks()
    def build(items, **_kwargs):
        assert runtime._current.get() is None and sentinel.get() is None
        assert items == [chunk]
        return final
    monkeypatch.setattr(settings, "orchestration_backend", "api")
    monkeypatch.setattr(completion.litellm, "acompletion", api)
    monkeypatch.setattr(completion.litellm, "stream_chunk_builder", build)
    updates = []
    async def observe(stream, text):
        assert sentinel.get() == "parent" and runtime._current.get() is not None
        updates.append((stream, text))
        first_seen.set()
    async def action():
        token = sentinel.set("parent")
        try:
            async with runtime.baseline_operation(db_session, uuid.uuid4(), owns_transaction=False):
                return await runtime.await_baseline_completion(completion.orchestration_completion,
                    {"messages": []}, observer=observe)
        finally:
            sentinel.reset(token)
    task = asyncio.create_task(action())
    try:
        await asyncio.wait_for(first_seen.wait(), 1)
        assert not task.done()
        release.set()
        assert await task is final
        assert updates == [("output", "live\n"), ("reasoning", "think\n")]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
