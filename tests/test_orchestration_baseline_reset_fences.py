"""Review regressions: eligibility, reset reservations and rerun recovery."""
import asyncio
import uuid
import pytest
from fastapi import HTTPException
from sqlalchemy import select, event
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from huddleroom.models.base import Base
from huddleroom.models.project import Project
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import OrchestrationProcessRun, OrchestrationAuthorityDecision
from huddleroom.services.orchestration_baseline_runtime import (
    baseline_operation, reset_baseline, require_baseline_available, await_baseline_completion,
)
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_debug_service import OrchestrationDebugService
from huddleroom.services.orchestration_process_service import OrchestrationProcessService

pytestmark = pytest.mark.asyncio

async def test_queued_step_is_discarded_before_reset_completes(tmp_path, monkeypatch):
    """A real second request queued on the goal lock cannot delay/reset fresh work."""
    from huddleroom.services import orchestration_service as service_module
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'queued.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with sessions() as db:
        project = Project(name="queued baseline", workspace_path=str(tmp_path))
        db.add(project)
        await db.flush()
        goal = OrchestrationGoal(project_id=project.id, objective="reset queued steps", status="active")
        db.add(goal)
        await db.flush()
        run = OrchestrationRun(goal_id=goal.id, status="running")
        db.add(run)
        await db.commit()
        project_id, goal_id, old_run_id = project.id, goal.id, run.id
    entered, release, queued = asyncio.Event(), asyncio.Event(), asyncio.Event()
    provider_calls = []
    async def provider(**request):
        provider_calls.append(request)
        entered.set()
        await release.wait()
        return "old result"
    async def advance(self, db, goal, run, process_type):
        process = await OrchestrationProcessService().start_process(
            db, goal.id, process_type=process_type, trigger_reason="test", run_id=run.id)
        await await_baseline_completion(provider, {})
        await OrchestrationProcessService().complete_process(db, process)
        return {"status": "completed"}
    monkeypatch.setattr(OrchestrationDebugService, "_advance", advance)
    async def step():
        async with sessions() as db:
            return await OrchestrationDebugService().step(db, project_id, goal_id, "goal_definition")
    first = asyncio.create_task(step())
    second = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        lock = service_module._SQLITE_GOAL_LOCKS[goal_id]._lock
        original_acquire = lock.acquire
        async def observe_waiter():
            assert lock.locked()
            queued.set()
            return await original_acquire()
        monkeypatch.setattr(lock, "acquire", observe_waiter)
        second = asyncio.create_task(step())
        await asyncio.wait_for(queued.wait(), 3)
        assert not second.done()
        monkeypatch.setattr(lock, "acquire", original_acquire)
        async with sessions() as db:
            _, fresh = await asyncio.wait_for(OrchestrationService().reset_goal(db, project_id, goal_id), 3)
            fresh_id = fresh.id
        for action in (first, second):
            with pytest.raises(HTTPException) as exc:
                await asyncio.wait_for(action, 3)
            assert exc.value.status_code == 409
        assert len(provider_calls) == 1 and not release.is_set()
        async with sessions() as db:
            assert list((await db.scalars(select(OrchestrationRun.id).where(
                OrchestrationRun.goal_id == goal_id))).all()) == [fresh_id]
            assert fresh_id != old_run_id
            assert list((await db.scalars(select(OrchestrationProcessRun).where(
                OrchestrationProcessRun.goal_id == goal_id))).all()) == []
    finally:
        release.set()
        await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
        await engine.dispose()

async def test_reset_reservation_rejects_registration_and_survives_other_reset_exit(db_session):
    goal_id = uuid.uuid4()
    async with reset_baseline(goal_id):
        async with reset_baseline(goal_id):
            with pytest.raises(HTTPException):
                require_baseline_available(goal_id)
        with pytest.raises(HTTPException):
            async with baseline_operation(db_session, goal_id, owns_transaction=False):
                pytest.fail("queued action entered reserved goal")
    require_baseline_available(goal_id)

async def test_unregistered_nested_transaction_is_never_published(db_session, monkeypatch):
    async def forbidden_commit():
        pytest.fail("unregistered savepoint was published")
    async def provider(**request):
        return "ok"
    async with baseline_operation(db_session, uuid.uuid4(), owns_transaction=True):
        nested = await db_session.begin_nested()
        monkeypatch.setattr(db_session, "commit", forbidden_commit)
        assert await await_baseline_completion(provider, {}) == "ok"
        assert nested.is_active
        await nested.rollback()

async def test_rejected_reset_never_signals_baseline(db_session, test_project, monkeypatch):
    goal = OrchestrationGoal(project_id=test_project.id, objective="terminal", status="completed")
    db_session.add(goal)
    await db_session.flush()
    async def forbidden_interrupt(goal_id):
        pytest.fail("ineligible reset interrupted baseline")
    monkeypatch.setattr("huddleroom.services.orchestration_baseline_runtime.interrupt_baseline", forbidden_interrupt)
    with pytest.raises(HTTPException) as exc:
        await OrchestrationService().reset_goal(db_session, test_project.id, goal.id)
    assert exc.value.status_code == 409
    with pytest.raises(HTTPException) as exc:
        await OrchestrationService().reset_goal(db_session, uuid.uuid4(), goal.id)
    assert exc.value.status_code == 404

async def test_rerun_postpublication_failure_restores_current_and_pending_question(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'rerun.db'}")
    @event.listens_for(engine.sync_engine, "connect")
    def enable_foreign_keys(connection, record):
        connection.execute("PRAGMA foreign_keys=ON")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        async with sessions() as db:
            project = Project(name="rerun", workspace_path=str(tmp_path))
            db.add(project)
            await db.flush()
            goal = OrchestrationGoal(project_id=project.id, objective="restore", status="active")
            db.add(goal)
            await db.flush()
            run = OrchestrationRun(goal_id=goal.id, status="running")
            db.add(run)
            await db.flush()
            prior = await OrchestrationProcessService().start_process(
                db, goal.id, process_type="goal_definition", trigger_reason="seed", run_id=run.id)
            prior.status = "waiting_decision"
            question = OrchestrationAuthorityDecision(goal_id=goal.id, run_id=run.id,
                source_process_run_id=prior.id, decision_key="test", title="question", question="What?", authority="human")
            db.add(question)
            await db.commit()
            project_id, goal_id, prior_id, question_id = project.id, goal.id, prior.id, question.id
        async def provider(**request):
            return "success"
        async def advance(self, db, goal, run, process_type):
            await await_baseline_completion(provider, {})
            raise RuntimeError("application failed after publication")
        monkeypatch.setattr(OrchestrationDebugService, "_advance", advance)
        async with sessions() as db:
            with pytest.raises(RuntimeError, match="application failed"):
                await OrchestrationDebugService().rerun_last(db, project_id, goal_id, "goal_definition")
        async with sessions() as db:
            rows = list((await db.scalars(select(OrchestrationProcessRun).where(OrchestrationProcessRun.goal_id == goal_id))).all())
            assert len(rows) == 1 and rows[0].id == prior_id
            assert rows[0].status == "waiting_decision" and rows[0].superseded_by_id is None
            question = await db.get(OrchestrationAuthorityDecision, question_id)
            assert question.status == "pending" and question.reason is None
    finally:
        await engine.dispose()
