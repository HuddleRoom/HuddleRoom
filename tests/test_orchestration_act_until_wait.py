import json
import types
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from huddleroom.config import settings
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationDecision,
    OrchestrationGoal,
    OrchestrationRun,
)
from huddleroom.services.orchestration_service import OrchestrationService


async def _setup(db, project, *, accepted=True, goal_type="outcome"):
    goal = OrchestrationGoal(project_id=project.id, objective="Keep moving", status="active", goal_type=goal_type)
    db.add(goal)
    await db.flush()
    run = OrchestrationRun(
        goal_id=goal.id, status="running", phase="authorized",
        plan_state={"status": "accepted"} if accepted else {},
    )
    db.add(run)
    await db.flush()
    return goal, run


def _wire(monkeypatch, db, run, steps, *, cap=3, on_dispatch=None):
    """steps: list of (action_type, status) or an Exception, one per dispatch call."""
    service = OrchestrationService()
    monkeypatch.setattr(settings, "orchestration_max_actions_per_tick", cap)
    calls = {"n": 0, "contexts": []}

    async def decide(_db, _run_id):
        return types.SimpleNamespace(id=uuid.uuid4(), parsed_decision={"action_type": "noop"})

    async def dispatch(_db, _run, _decision):
        spec = steps[calls["n"]]
        calls["n"] += 1
        if isinstance(spec, Exception):
            raise spec
        action_type, status = spec
        action = OrchestrationAction(
            run_id=_run.id, idempotency_key=f"k:{uuid.uuid4()}", action_type=action_type, status=status,
        )
        _db.add(action)
        await _db.flush()
        if on_dispatch:
            await on_dispatch(_run)
        return action

    async def no_release(*_a, **_k):
        return 0

    monkeypatch.setattr(service, "request_llm_decision", decide)
    monkeypatch.setattr(service, "_dispatch_execution_decision", dispatch)
    monkeypatch.setattr(service, "_release_ready_work", no_release)
    return service, calls


async def test_loop_dispatches_up_to_cap_and_returns_all_action_ids(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project)
    service, calls = _wire(monkeypatch, db_session, run, [("request_verification", "completed")] * 5, cap=3)
    result = await service._advance_authorized_execution(db_session, goal, run)
    assert calls["n"] == 3
    assert len(result["action_ids"]) == 3 and len(set(result["action_ids"])) == 3
    assert result["action_id"] == result["action_ids"][0]
    assert result["step"] == "next_action"


async def test_loop_stops_on_noop_and_includes_it(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project)
    service, calls = _wire(monkeypatch, db_session, run, [("request_verification", "completed"), ("noop", "completed")])
    result = await service._advance_authorized_execution(db_session, goal, run)
    assert calls["n"] == 2
    assert len(result["action_ids"]) == 2


async def test_loop_stops_on_failed_action(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project)
    service, calls = _wire(monkeypatch, db_session, run, [("request_verification", "failed")] * 3)
    result = await service._advance_authorized_execution(db_session, goal, run)
    assert calls["n"] == 1
    assert "action_ids" not in result


async def test_loop_stops_on_reserved_action(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project)
    service, calls = _wire(monkeypatch, db_session, run, [("request_verification", "reserved")] * 3)
    await service._advance_authorized_execution(db_session, goal, run)
    assert calls["n"] == 1


async def test_loop_stops_when_same_action_repeats(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project)
    service = OrchestrationService()
    monkeypatch.setattr(settings, "orchestration_max_actions_per_tick", 3)
    calls = {"n": 0}

    async def once(_db, _goal, _run):
        calls["n"] += 1
        return {"step": "next_action", "action_id": str(same.id)}

    same = OrchestrationAction(run_id=run.id, idempotency_key="k", action_type="x", status="completed")
    db_session.add(same)
    await db_session.flush()
    monkeypatch.setattr(service, "_advance_authorized_execution_once", once)
    result = await service._advance_authorized_execution(db_session, goal, run)
    assert calls["n"] == 2
    assert result["action_ids"] == [str(same.id)]


async def test_loop_stops_on_non_loop_step(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project)
    service = OrchestrationService()
    calls = {"n": 0}

    async def once(_db, _goal, _run):
        calls["n"] += 1
        return {"step": "waiting", "reason": "budget_wait"}

    monkeypatch.setattr(service, "_advance_authorized_execution_once", once)
    result = await service._advance_authorized_execution(db_session, goal, run)
    assert calls["n"] == 1
    assert result == {"step": "waiting", "reason": "budget_wait"}


async def test_loop_stops_when_run_leaves_running(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project)

    async def pause(r):
        r.status = "paused"
        await db_session.flush()

    service, calls = _wire(monkeypatch, db_session, run, [("request_verification", "completed")] * 3, on_dispatch=pause)
    await service._advance_authorized_execution(db_session, goal, run)
    assert calls["n"] == 1


@pytest.mark.parametrize("field,value", [("goal_status", "paused"), ("run_phase", "completed")])
async def test_loop_stops_when_goal_paused_or_phase_changes(db_session, test_project, monkeypatch, field, value):
    goal, run = await _setup(db_session, test_project)

    async def change(r):
        if field == "goal_status":
            goal.status = value
        else:
            r.phase = value
        await db_session.flush()

    service, calls = _wire(monkeypatch, db_session, run, [("request_verification", "completed")] * 3, on_dispatch=change)
    await service._advance_authorized_execution(db_session, goal, run)
    assert calls["n"] == 1


@pytest.mark.parametrize("action_type", ["ask_human", "pause_run"])
async def test_human_handoff_actions_stop_loop(db_session, test_project, monkeypatch, action_type):
    goal, run = await _setup(db_session, test_project)
    service, calls = _wire(monkeypatch, db_session, run, [(action_type, "completed")] * 3)
    result = await service._advance_authorized_execution(db_session, goal, run)
    assert calls["n"] == 1
    assert run.status == "running" and run.phase == "authorized"
    assert "action_ids" not in result


async def test_roadmap_accept_plan_stops_loop_without_integrity_blocker(db_session, test_project, monkeypatch):
    # ponytail: scripted step; the realistic path is covered by tests/test_orchestration_roadmap_plans.py.
    goal, run = await _setup(db_session, test_project, goal_type="roadmap")
    service = OrchestrationService()
    monkeypatch.setattr(settings, "orchestration_max_actions_per_tick", 3)
    action = OrchestrationAction(run_id=run.id, idempotency_key="k", action_type="accept_plan", status="completed")
    db_session.add(action)
    await db_session.flush()
    calls = {"n": 0}

    async def once(_db, _goal, _run):
        calls["n"] += 1
        return {"step": "next_action", "action_id": str(action.id)}

    monkeypatch.setattr(service, "_advance_authorized_execution_once", once)
    result = await service._advance_authorized_execution(db_session, goal, run)
    assert calls["n"] == 1
    assert not service._has_active_blocker(run, "plan_criterion_integrity")
    assert result["step"] != "needs_attention"


async def test_non_http_exception_on_second_iteration_propagates(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project)
    service, calls = _wire(
        monkeypatch, db_session, run, [("request_verification", "completed"), RuntimeError("boom")]
    )
    with pytest.raises(RuntimeError):
        await service._advance_authorized_execution(db_session, goal, run)
    assert calls["n"] == 2


async def test_second_iteration_http_exception_propagates(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project)
    service, calls = _wire(
        monkeypatch, db_session, run, [("request_verification", "completed"), HTTPException(409, "boom")]
    )
    with pytest.raises(HTTPException) as exc:
        await service._advance_authorized_execution(db_session, goal, run)
    assert exc.value.status_code == 409
    assert calls["n"] == 2


async def test_reused_step_stops_loop_and_is_not_counted(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project)
    service = OrchestrationService()
    monkeypatch.setattr(settings, "orchestration_max_actions_per_tick", 3)
    calls = {"n": 0}

    async def once(_db, _goal, _run):
        calls["n"] += 1
        return {"step": "next_action", "action_id": str(uuid.uuid4()), "reused": True}

    monkeypatch.setattr(service, "_advance_authorized_execution_once", once)
    result = await service._advance_authorized_execution(db_session, goal, run)
    assert calls["n"] == 1
    assert "action_ids" not in result
    assert "reused" not in result


async def test_sqlite_autobegin_precommit_keeps_first_iteration_when_second_raises(
    test_engine, tmp_path, monkeypatch
):
    import huddleroom.services.orchestration_service as svc_module
    from huddleroom.models.project import Project
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_llm_decision_adapter import OrchestrationDecisionAdapterResult
    from tests.conftest import complete_baseline_processes

    if test_engine.dialect.name != "sqlite":
        pytest.skip("SQLite autobegin pre-commit path only")
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(svc_module, "AsyncSessionLocal", factory)
    monkeypatch.setattr(settings, "orchestration_max_actions_per_tick", 3)
    service = OrchestrationService()
    async with factory() as setup:
        project = Project(name="Autobegin", description="t", workspace_path=str(tmp_path), config={})
        setup.add(project)
        await setup.flush()
        goal, run = await service.create_goal(
            setup, project_id=project.id,
            data=OrchestrationGoalCreate(
                objective="Autobegin", success_criteria=[{"key": "done", "description": "Done"}],
                constraints={}, budget={"llm_calls": 5},
            ),
            created_by_user_id=None,
        )
        run.baseline_authorized = True
        await complete_baseline_processes(setup, goal, run)
        run.phase = "authorized"
        run.status = "running"
        goal.status = "active"
        await setup.commit()
        ids = (project.id, goal.id, run.id)

    decision = {"action_type": "ask_human", "question": "Which?", "reason": "Unsure"}

    class Stub:
        async def decide(self, context, project=None, goal=None):
            return OrchestrationDecisionAdapterResult(
                input_snapshot=context, llm_output={"raw_content": json.dumps({"decision": decision})},
                parsed_decision=decision,
            )

    calls = {"n": 0}

    async def once(db, goal_, run_):
        calls["n"] += 1
        if calls["n"] == 2:
            await service.request_llm_decision(db, run_.id, adapter=Stub())  # pre-commits iteration 1
            db.add(OrchestrationAction(
                run_id=run_.id, idempotency_key="iter2", action_type="request_verification", status="completed",
            ))
            await db.flush()
            raise HTTPException(409, "boom")
        action = OrchestrationAction(
            run_id=run_.id, idempotency_key="iter1", action_type="request_verification", status="completed",
        )
        db.add(action)
        await db.flush()
        return {"step": "next_action", "action_id": str(action.id)}

    async def no_release(*_a, **_k):
        return 0

    monkeypatch.setattr(service, "_advance_authorized_execution_once", once)
    try:
        async with factory() as tick_db:
            tick_db.info["orchestration_tick_owns_transaction"] = True
            goal_row = await tick_db.get(OrchestrationGoal, ids[1])
            run_row = await tick_db.get(OrchestrationRun, ids[2])
            with pytest.raises(HTTPException):
                await service._advance_authorized_execution(tick_db, goal_row, run_row)
            await tick_db.rollback()
        async with factory() as check:
            keys = set((await check.scalars(
                select(OrchestrationAction.idempotency_key).where(OrchestrationAction.run_id == ids[2])
            )).all())
        assert keys == {"iter1"}  # exactly one action; iter2 row rolled back
    finally:
        async with factory() as cleanup:
            await cleanup.execute(delete(OrchestrationAction).where(OrchestrationAction.run_id == ids[2]))
            await cleanup.execute(delete(OrchestrationDecision).where(OrchestrationDecision.run_id == ids[2]))
            await cleanup.execute(delete(OrchestrationRun).where(OrchestrationRun.id == ids[2]))
            await cleanup.execute(delete(OrchestrationGoal).where(OrchestrationGoal.id == ids[1]))
            await cleanup.execute(delete(Project).where(Project.id == ids[0]))
            await cleanup.commit()


async def test_first_iteration_http_exception_propagates(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project)
    service, _ = _wire(monkeypatch, db_session, run, [HTTPException(409, "boom")])
    with pytest.raises(HTTPException):
        await service._advance_authorized_execution(db_session, goal, run)


async def test_single_iteration_result_shape_unchanged(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project)
    service, _ = _wire(monkeypatch, db_session, run, [("noop", "completed")])
    result = await service._advance_authorized_execution(db_session, goal, run)
    assert set(result) == {"step", "action_id"}


async def test_cap_of_one_behaves_like_before(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project)
    service, calls = _wire(monkeypatch, db_session, run, [("request_verification", "completed")] * 3, cap=1)
    result = await service._advance_authorized_execution(db_session, goal, run)
    assert calls["n"] == 1
    assert "action_ids" not in result


async def test_plan_branch_plan_decision_steps_also_loop(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project, accepted=False)
    service, calls = _wire(monkeypatch, db_session, run, [("request_verification", "completed")] * 2, cap=2)
    result = await service._advance_authorized_execution(db_session, goal, run)
    assert calls["n"] == 2
    assert result["step"] == "plan_decision"
    assert len(result["action_ids"]) == 2


async def test_each_iteration_rereads_context(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project)
    service = OrchestrationService()
    monkeypatch.setattr(settings, "orchestration_max_actions_per_tick", 2)
    seen = []

    async def decide(_db, _run_id):
        context = await service._decision_context(_db, goal, run)
        seen.append(str(context))
        return types.SimpleNamespace(id=uuid.uuid4(), parsed_decision={"action_type": "noop"})

    async def dispatch(_db, _run, _decision):
        action = OrchestrationAction(
            run_id=_run.id, idempotency_key=f"k:{uuid.uuid4()}", action_type="request_verification", status="completed",
        )
        _db.add(action)
        await _db.flush()
        dispatch.first = dispatch.first or str(action.id)
        return action

    dispatch.first = None

    async def no_release(*_a, **_k):
        return 0

    monkeypatch.setattr(service, "request_llm_decision", decide)
    monkeypatch.setattr(service, "_dispatch_execution_decision", dispatch)
    monkeypatch.setattr(service, "_release_ready_work", no_release)
    await service._advance_authorized_execution(db_session, goal, run)
    assert len(seen) == 2
    assert dispatch.first not in seen[0]
    assert dispatch.first in seen[1]


@pytest.mark.parametrize("detail", ["budget_wait", "needs_attention"])
async def test_step_four_budget_or_attention_409_returns_waiting(db_session, test_project, monkeypatch, detail):
    goal, run = await _setup(db_session, test_project)
    service = OrchestrationService()

    async def decide(_db, _run_id):
        raise HTTPException(409, detail)

    async def no_release(*_a, **_k):
        return 0

    monkeypatch.setattr(service, "request_llm_decision", decide)
    monkeypatch.setattr(service, "_release_ready_work", no_release)
    result = await service._advance_authorized_execution(db_session, goal, run)
    assert result == {"step": "waiting", "reason": detail}
