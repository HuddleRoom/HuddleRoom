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
        pytest.fail("provider admitted through caller savepoint")
    async with db_session.begin_nested() as nested:
        monkeypatch.setattr(db_session, "commit", forbidden_commit)
        with pytest.raises(HTTPException) as conflict:
            async with baseline_operation(db_session, uuid.uuid4(), owns_transaction=True):
                await await_baseline_completion(provider, {})
        assert conflict.value.status_code == 409
        assert conflict.value.detail["conflict"] == "orchestration_transaction_shape"
        assert nested.is_active

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


# --------------------------------------------------------------------------- SPR 180/184: recovery reset fences
#
# Recovery honours the SPR 175/176 reset reservation instead of a parallel cancellation model: every
# recovery phase checks require_baseline_available (and the original run identity) before any effect
# and before its final commit. Harness: tests/test_orchestration_recovery_execution.py.

from huddleroom.models.orchestration import OrchestrationAction  # noqa: E402
from tests.test_orchestration_roster_mutation import seed_project_user, sessions  # noqa: E402,F401


async def _recovery_at(db, project, user, monkeypatch, phase):
    from huddleroom.services.orchestration_blocker_recovery import OrchestrationBlockerRecoveryService
    from tests.test_orchestration_recovery_execution import (
        Executors, ScriptedAnalyzer, approved, draft, propose, seed, trigger, verified, verify_call,
    )

    rows = await seed(db, project)
    executors = Executors(monkeypatch, rows)
    analyzer = ScriptedAnalyzer(draft(rows, verify_call(rows)))
    flow = type("Flow", (), {})()
    flow.rows, flow.executors, flow.trigger, flow.proposal = rows, executors, None, None
    if phase == "diagnose":
        flow.svc = OrchestrationBlockerRecoveryService(analyzer=analyzer)
        flow.trigger = await flow.svc.create_trigger(db, trigger(rows))
        return flow
    if phase == "answer":
        flow.svc, flow.proposal = await propose(db, rows, analyzer)
        return flow
    approved_flow = await approved(db, rows, user, verify_call(rows), analyzer=analyzer)
    flow.svc, flow.proposal = approved_flow.svc, approved_flow.proposal
    if phase == "verify":
        await flow.svc.execute_next_call(db, flow.proposal)
    elif phase == "retry":
        await verified(db, approved_flow)
    return flow


async def _recovery_rows(db, run_id):
    return sorted(((row.idempotency_key, row.status) for row in (await db.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run_id).execution_options(populate_existing=True))).all()))


@pytest.mark.parametrize("phase", ("diagnose", "answer", "apply", "verify", "retry"))
async def test_reset_reservation_fences_every_recovery_phase_before_any_effect(
    owned_recovery_setup, monkeypatch, phase,
):
    db_session, test_project, test_user = owned_recovery_setup
    flow = await _recovery_at(db_session, test_project, test_user, monkeypatch, phase)
    rows = flow.rows
    before = await _recovery_rows(db_session, rows.run.id)
    calls_before = len(flow.executors.calls)
    step = {
        "diagnose": lambda: flow.svc.diagnose(db_session, flow.trigger.id),
        "answer": lambda: flow.svc.reserve_answer(
            db_session, decision, selected_option="approve_fix", reason=None, user_id=test_user.id),
        "apply": lambda: flow.svc.execute_next_call(db_session, flow.proposal),
        "verify": lambda: flow.svc.verify_proposal(db_session, flow.proposal),
        "retry": lambda: flow.svc.retry_blocked_work(db_session, flow.proposal),
    }[phase]
    decision = None
    if phase == "answer":
        decision = (await db_session.scalars(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.related_action_id == flow.proposal.id))).one()

    async with reset_baseline(rows.goal.id):
        with pytest.raises(HTTPException) as exc:
            await step()

    assert exc.value.status_code == 409
    assert await _recovery_rows(db_session, rows.run.id) == before, "a reserved reset admits no recovery write"
    assert len(flow.executors.calls) == calls_before
    if decision is not None:
        await db_session.refresh(decision)
        assert decision.status == "pending"


async def test_reset_signalled_during_a_recovery_effect_discards_it_before_the_final_commit(sessions, tmp_path, monkeypatch):
    from huddleroom.services import orchestration_baseline_runtime as runtime
    from huddleroom.services.orchestration_blocker_recovery import OrchestrationBlockerRecoveryService
    from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher
    from tests.test_orchestration_recovery_execution import ScriptedAnalyzer, approved, draft, seed, verify_call

    async with sessions() as s:
        project, user = await seed_project_user(s, tmp_path)
        rows = await seed(s, project)
        flow = await approved(s, rows, user, verify_call(rows),
                              analyzer=ScriptedAnalyzer(draft(rows, verify_call(rows))), owns_setup=True)
        await s.commit()
        goal_id, run_id, proposal_id = rows.goal.id, rows.run.id, flow.proposal.id
        wrapper_key = f"recovery:{flow.pid}:apply:call-1"
    signalled = []

    async def effect_then_reset(self, db, run_id, request, idempotency_key, decision_id=None, **extra):
        action = await self.reserve_action(db, run_id, idempotency_key, "request_verification", request, decision_id)
        runtime._resetting[goal_id] = runtime._resetting.get(goal_id, 0) + 1  # reset reserved mid-effect
        signalled.append(True)
        return await self._mark_action_completed(db, action, "task", rows.task.id)

    monkeypatch.setattr(OrchestrationService, OrchestrationDecisionDispatcher._EXECUTORS["request_verification"],
                        effect_then_reset)
    try:
        async with sessions() as s:
            with pytest.raises(HTTPException) as exc:
                await OrchestrationBlockerRecoveryService().resume(s, proposal_id)
        assert exc.value.status_code == 409 and signalled
    finally:
        if runtime._resetting.get(goal_id):
            runtime._resetting[goal_id] -= 1
            if not runtime._resetting[goal_id]:
                runtime._resetting.pop(goal_id)

    async with sessions() as s:
        wrapper = (await s.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run_id, OrchestrationAction.idempotency_key == wrapper_key))).one()
        assert wrapper.status != "completed" and "tool_receipt_v1" not in (wrapper.dispatch_contract or {})
        effect = (await s.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run_id, OrchestrationAction.idempotency_key == f"{wrapper_key}:effect"))).first()
        assert effect is None or effect.status != "completed", "the discarded effect never reached its final commit"
        retries = (await s.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run_id, OrchestrationAction.action_type == "recovery_retry"))).all()
        assert retries == []


async def test_reset_during_recovery_diagnosis_discards_the_analyzer_result(sessions, tmp_path):
    from huddleroom.services.orchestration_blocker_recovery import OrchestrationBlockerRecoveryService
    from tests.test_orchestration_recovery_execution import draft, seed, trigger, verify_call

    async with sessions() as s:
        project, _user = await seed_project_user(s, tmp_path)
        rows = await seed(s, project)
        proposal_draft = draft(rows, verify_call(rows))
        await s.commit()
        project_id, goal_id, old_run_id = project.id, rows.goal.id, rows.run.id
        trigger_model = trigger(rows)
    entered, release = asyncio.Event(), asyncio.Event()

    class BlockingAnalyzer:
        async def analyze(self, snapshot, *, project_id, evidence_query, on_raw=None):
            entered.set()
            await release.wait()  # the provider answers only after the reset began
            return proposal_draft

    svc = OrchestrationBlockerRecoveryService(analyzer=BlockingAnalyzer())
    async with sessions() as s:
        trigger_id = (await svc.create_trigger(s, trigger_model)).id
        await s.commit()

    async def diagnose():
        async with sessions() as s:
            return await svc.diagnose(s, trigger_id)

    async def reset():
        async with sessions() as s:
            _goal, fresh = await OrchestrationService().reset_goal(s, project_id, goal_id)
            return fresh.id

    diagnosis = asyncio.create_task(diagnose())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        resetting = asyncio.create_task(reset())
        await asyncio.sleep(0.2)
        release.set()
        fresh_id = await asyncio.wait_for(resetting, 10)
        with pytest.raises(HTTPException) as exc:
            await asyncio.wait_for(diagnosis, 10)
        assert exc.value.status_code == 409
    finally:
        release.set()
        await asyncio.gather(diagnosis, return_exceptions=True)

    async with sessions() as s:
        assert list((await s.scalars(select(OrchestrationRun.id).where(OrchestrationRun.goal_id == goal_id))).all()) == [fresh_id]
        assert fresh_id != old_run_id
        assert (await s.scalars(select(OrchestrationAction).where(OrchestrationAction.run_id == fresh_id))).all() == []
        assert (await s.scalars(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.goal_id == goal_id))).all() == []


# SPR230/SPR231 only explicit recovery setup cases use this independently owned fixture.
from tests.test_orchestration_recovery_persistence import owned_recovery_setup  # noqa: E402,F401 -- fixture
