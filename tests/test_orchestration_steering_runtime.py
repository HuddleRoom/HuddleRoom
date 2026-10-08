import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from huddleroom.config import settings
from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationDecision, OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_steering import OrchestrationSteeringResultLink
from huddleroom.models.project import Project
from huddleroom.models.task import Task
from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_steering import (
    OrchestrationSteeringService, SteeringDomainError, SteeringDraft, SteeringVersionsChanged,
)
from huddleroom.services.orchestration_supervision_context import OrchestrationSupervisionContextBuilder


pytestmark = pytest.mark.asyncio


async def test_sqlite_stale_rejection_is_durable_after_autobegin_caller_rollback(test_engine):
    if test_engine.dialect.name != "sqlite":
        pytest.skip("SQLite transaction ownership regression")
    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    async with sessions() as seed:
        project = Project(name=f"steering audit {uuid.uuid4()}", description="", config={})
        seed.add(project)
        await seed.flush()
        goal = OrchestrationGoal(project_id=project.id, objective="audit", success_criteria=[], constraints={}, budget={})
        seed.add(goal)
        await seed.flush()
        run = OrchestrationRun(goal_id=goal.id, event_cursor=None, plan_state={}, active_blockers=[], budget_state={}, retry_state={})
        seed.add(run)
        await seed.flush()
        decision = OrchestrationDecision(run_id=run.id, decision_type="m7", validator_status="accepted")
        seed.add(decision)
        await seed.commit()
        decision_id = decision.id

    service = OrchestrationService()
    async with sessions() as caller:
        await caller.get(OrchestrationDecision, decision_id)  # starts AUTOBEGIN
        await service._reject_stale_steering_decision(caller, decision_id)
        await caller.rollback()

    async with sessions() as reader:
        persisted = await reader.get(OrchestrationDecision, decision_id)
        assert persisted.validator_status == "rejected"
        assert persisted.rejection_reason == "stale_steering_versions"


async def test_stale_rejection_falls_back_when_independent_audit_cannot_see_decision(
    db_session, conversation_goal_run, monkeypatch,
):
    _goal, run = conversation_goal_run
    decision = OrchestrationDecision(run_id=run.id, decision_type="m7", validator_status="accepted")
    db_session.add(decision)
    await db_session.flush()

    class EmptyAuditSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def execute(self, _statement):
            return SimpleNamespace(rowcount=0)

        async def commit(self):
            return None

    # A bind-less wrapper selects the non-SQLite independent-audit path while
    # retaining the actual attached SQLite caller for the fallback update.
    caller = SimpleNamespace(bind=None, execute=db_session.execute, flush=db_session.flush)
    monkeypatch.setattr(
        "huddleroom.services.orchestration_service.AsyncSessionLocal", lambda: EmptyAuditSession(),
    )
    await OrchestrationService()._reject_stale_steering_decision(caller, decision.id)

    await db_session.refresh(decision)
    assert decision.validator_status == "rejected"
    assert decision.rejection_reason == "stale_steering_versions"


async def _applied_item(db, conversation_goal_run, test_project, test_user, monkeypatch):
    monkeypatch.setattr(settings, "orchestration_conversation_steering_enabled", True)
    goal, run = conversation_goal_run
    goal.status, goal.manager_user_id = "active", test_user.id
    run.status, run.phase = "running", "authorized"
    task = Task(
        project_id=test_project.id,
        title="Unstarted",
        status="ready",
        metadata_={"orchestration": {"goal_id": str(goal.id), "run_id": str(run.id)}},
    )
    db.add(task)
    await db.flush()
    steering = OrchestrationSteeringService()
    request = await steering.submit(
        db, test_project.id, goal.id, test_user.id, uuid.uuid4(),
        SteeringDraft("Validate first", "task", str(task.id), "item", "selected_item", "impact"),
    )
    await steering.process_pending(db, goal, run)
    return goal, run, task, request


async def test_supervision_context_projects_steering_and_omits_drifted_item(
    db_session, conversation_goal_run, test_project, test_user, monkeypatch,
):
    goal, run, task, request = await _applied_item(
        db_session, conversation_goal_run, test_project, test_user, monkeypatch,
    )
    context = await OrchestrationSupervisionContextBuilder().build(db_session, goal, run)
    assert context["steering"]["active_directions"] == [{
        "request_id": str(request.id), "directive": "Validate first", "target_type": "task",
        "target_id": str(task.id), "scope": "item", "lifetime": "selected_item",
        "impact_summary": "impact",
    }]

    task.status = "in_progress"
    rebuilt = await OrchestrationSupervisionContextBuilder().build(db_session, goal, run)
    assert rebuilt["steering"]["active_directions"] == []


async def test_steered_decision_uses_final_versioned_action_key_and_links_result(
    db_session, conversation_goal_run, test_project, test_user, monkeypatch,
):
    goal, run, _task, request = await _applied_item(
        db_session, conversation_goal_run, test_project, test_user, monkeypatch,
    )
    snapshot = await OrchestrationSteeringService().context_snapshot(db_session, goal, run)
    decision = OrchestrationDecision(
        run_id=run.id, decision_type="m7", validator_status="accepted",
        input_snapshot={"steering": snapshot}, parsed_decision={"action_type": "noop"},
    )
    db_session.add(decision)
    await db_session.flush()

    from huddleroom.services.orchestration_service import OrchestrationService
    action = await OrchestrationDecisionDispatcher(OrchestrationService()).dispatch(db_session, run, decision)

    assert ":steering:" in action.idempotency_key
    assert action.dispatch_contract["steering_versions"] == snapshot["versions"]
    links = await db_session.scalars(select(OrchestrationSteeringResultLink))
    assert [(link.request_id, link.decision_id, link.action_id) for link in links] == [
        (request.id, decision.id, action.id)
    ]


async def test_steered_decision_cannot_pause_a_run(
    db_session, conversation_goal_run, test_project, test_user, monkeypatch,
):
    goal, run, _task, _request = await _applied_item(
        db_session, conversation_goal_run, test_project, test_user, monkeypatch,
    )
    decision = OrchestrationDecision(
        run_id=run.id, decision_type="m7", validator_status="accepted",
        input_snapshot={"steering": await OrchestrationSteeringService().context_snapshot(db_session, goal, run)},
        parsed_decision={"action_type": "pause_run", "reason": "not allowed"},
    )
    db_session.add(decision)
    await db_session.flush()

    from huddleroom.services.orchestration_service import OrchestrationService
    service = OrchestrationService()
    assert await OrchestrationDecisionDispatcher(service).dispatch(db_session, run, decision) is None
    assert decision.validator_status == "rejected" and "dedicated control" in decision.rejection_reason
    # the tick path turns the rejection into a noop plus the backstop wait instead of raising
    from huddleroom.models.orchestration import OrchestrationWait
    noop = await service._dispatch_execution_decision(db_session, run, decision)
    assert noop.action_type == "noop"
    assert (await db_session.scalars(select(OrchestrationWait).where(
        OrchestrationWait.run_id == run.id, OrchestrationWait.status == "open"))).all()


async def test_stale_steering_versions_do_not_reserve_an_action(
    db_session, conversation_goal_run, test_project, test_user, monkeypatch,
):
    goal, run, _task, _request = await _applied_item(
        db_session, conversation_goal_run, test_project, test_user, monkeypatch,
    )
    steering = OrchestrationSteeringService()
    snapshot = await steering.context_snapshot(db_session, goal, run)
    decision = OrchestrationDecision(
        run_id=run.id, decision_type="m7", validator_status="accepted",
        input_snapshot={"steering": snapshot}, parsed_decision={"action_type": "noop"},
    )
    db_session.add(decision)
    await db_session.flush()
    await steering.submit(
        db_session, test_project.id, goal.id, test_user.id, uuid.uuid4(),
        SteeringDraft("Changed direction", "goal", str(goal.id), "run", "remaining_current_run", "impact"),
    )

    from huddleroom.services.orchestration_service import OrchestrationService
    with pytest.raises(SteeringVersionsChanged):
        await OrchestrationDecisionDispatcher(OrchestrationService()).dispatch(db_session, run, decision)


async def test_stale_steering_replays_a_completed_action_by_its_persisted_key(
    db_session, conversation_goal_run, test_project, test_user, monkeypatch,
):
    goal, run, _task, _request = await _applied_item(
        db_session, conversation_goal_run, test_project, test_user, monkeypatch,
    )
    steering = OrchestrationSteeringService()
    decision = OrchestrationDecision(
        run_id=run.id, decision_type="m7", validator_status="accepted",
        input_snapshot={"steering": await steering.context_snapshot(db_session, goal, run)},
        parsed_decision={"action_type": "noop"},
    )
    db_session.add(decision)
    await db_session.flush()
    service = OrchestrationService()
    first = await service.execute_noop_action(
        db_session, run.id, {}, "stale-steering-completed-action", decision.id,
    )
    assert first.status == "completed"
    await steering.submit(
        db_session, test_project.id, goal.id, test_user.id, uuid.uuid4(),
        SteeringDraft("Changed direction", "goal", str(goal.id), "run", "remaining_current_run", "impact"),
    )

    replay = await service.execute_noop_action(
        db_session, run.id, {}, first.idempotency_key, decision.id,
    )
    assert replay.id == first.id
    with pytest.raises(SteeringVersionsChanged):
        await service.reserve_action(
            db_session, run.id, "new-stale-steering-action", "noop", {}, decision.id,
        )
    assert await db_session.scalar(select(OrchestrationAction.id).where(
        OrchestrationAction.decision_id == decision.id,
        OrchestrationAction.id != first.id,
    )) is None


async def test_stale_steering_fails_a_reserved_action_replay(
    db_session, conversation_goal_run, test_project, test_user, monkeypatch,
):
    goal, run, _task, _request = await _applied_item(
        db_session, conversation_goal_run, test_project, test_user, monkeypatch,
    )
    steering = OrchestrationSteeringService()
    decision = OrchestrationDecision(
        run_id=run.id, decision_type="m7", validator_status="accepted",
        input_snapshot={"steering": await steering.context_snapshot(db_session, goal, run)},
        parsed_decision={"action_type": "noop"},
    )
    db_session.add(decision)
    await db_session.flush()
    service = OrchestrationService()
    reserved = await service.reserve_action(
        db_session, run.id, "stale-steering-reserved-action", "noop", {}, decision.id,
    )
    await steering.submit(
        db_session, test_project.id, goal.id, test_user.id, uuid.uuid4(),
        SteeringDraft("Changed direction", "goal", str(goal.id), "run", "remaining_current_run", "impact"),
    )

    with pytest.raises(SteeringVersionsChanged):
        await service.reserve_action(
            db_session, run.id, reserved.idempotency_key, "noop", {}, decision.id,
        )
    await db_session.refresh(reserved)
    assert (reserved.status, reserved.error) == ("failed", "stale_steering_versions")


@pytest.mark.parametrize("drift", ("started", "completed", "ownership"))
async def test_item_lifetime_drift_stales_a_frozen_steering_decision(
    db_session, conversation_goal_run, test_project, test_user, monkeypatch,
    drift,
):
    goal, run, task, _request = await _applied_item(
        db_session, conversation_goal_run, test_project, test_user, monkeypatch,
    )
    steering = OrchestrationSteeringService()
    decision = OrchestrationDecision(
        run_id=run.id, decision_type="m7", validator_status="accepted",
        input_snapshot={"steering": await steering.context_snapshot(db_session, goal, run)},
        parsed_decision={"action_type": "noop"},
    )
    db_session.add(decision)
    await db_session.flush()

    if drift == "ownership":
        task.metadata_ = {
            **task.metadata_,
            "orchestration": {**task.metadata_["orchestration"], "run_id": str(uuid.uuid4())},
        }
    else:
        task.status = "in_progress" if drift == "started" else "completed"
    with pytest.raises(SteeringVersionsChanged):
        await OrchestrationDecisionDispatcher(OrchestrationService()).dispatch(db_session, run, decision)
    assert await db_session.scalar(select(OrchestrationAction.id).where(
        OrchestrationAction.decision_id == decision.id,
    )) is None


async def test_stale_decision_is_rejected_then_reevaluated_once(
    db_session, conversation_goal_run, test_project, test_user, monkeypatch,
):
    goal, run, _task, _request = await _applied_item(
        db_session, conversation_goal_run, test_project, test_user, monkeypatch,
    )
    steering = OrchestrationSteeringService()
    stale = OrchestrationDecision(
        run_id=run.id, decision_type="m7", validator_status="accepted",
        input_snapshot={"steering": await steering.context_snapshot(db_session, goal, run)},
        parsed_decision={"action_type": "noop"},
    )
    db_session.add(stale)
    await db_session.flush()
    await steering.submit(
        db_session, test_project.id, goal.id, test_user.id, uuid.uuid4(),
        SteeringDraft("New direction", "goal", str(goal.id), "run", "remaining_current_run", "impact"),
    )
    service = OrchestrationService()
    calls = 0

    async def fresh_decision(_db, _run_id):
        nonlocal calls
        calls += 1
        fresh = OrchestrationDecision(
            run_id=run.id, decision_type="m7", validator_status="accepted",
            input_snapshot={"steering": await steering.context_snapshot(db_session, goal, run)},
            parsed_decision={"action_type": "noop"},
        )
        db_session.add(fresh)
        await db_session.flush()
        return fresh

    monkeypatch.setattr(service, "request_llm_decision", fresh_decision)
    action = await service._dispatch_execution_decision(db_session, run, stale)

    await db_session.refresh(stale)
    assert stale.validator_status == "rejected"
    assert stale.rejection_reason == "stale_steering_versions"
    assert calls == 1
    assert action is not None


async def test_late_item_lifetime_drift_fails_reservation_and_reevaluates_once(
    db_session, conversation_goal_run, concurrent_sessions, test_project, test_user, monkeypatch,
):
    goal, run, task, request = await _applied_item(
        db_session, conversation_goal_run, test_project, test_user, monkeypatch,
    )
    await db_session.commit()
    session_a, session_b = concurrent_sessions
    goal = await session_a.get(OrchestrationGoal, goal.id)
    run = await session_a.get(OrchestrationRun, run.id)
    task = await session_a.get(Task, task.id)
    steering = OrchestrationSteeringService()
    stale = OrchestrationDecision(
        run_id=run.id, decision_type="m7", validator_status="accepted",
        input_snapshot={"steering": await steering.context_snapshot(session_a, goal, run)},
        parsed_decision={"action_type": "noop"},
    )
    session_a.add(stale)
    await session_a.flush()

    service = OrchestrationService()
    original_existing = service._existing_action_for_key
    drifted = False
    existing_checks = 0
    reevaluations = 0
    refreshed = None

    async def drift_after_early_fence(*args, **kwargs):
        nonlocal drifted, existing_checks
        existing_checks += 1
        if existing_checks == 3:
            drifted = True
            await session_a.commit()
            current = await session_b.get(Task, task.id)
            current.status = "completed"
            await session_b.commit()
        return await original_existing(*args, **kwargs)

    async def fresh_decision(_db, _run_id):
        nonlocal reevaluations, refreshed
        reevaluations += 1
        fresh = OrchestrationDecision(
            run_id=run.id, decision_type="m7", validator_status="accepted",
            input_snapshot={"steering": await steering.context_snapshot(session_a, goal, run)},
            parsed_decision={"action_type": "noop"},
        )
        session_a.add(fresh)
        await session_a.flush()
        refreshed = fresh
        await steering.submit(
            session_a, test_project.id, goal.id, test_user.id, uuid.uuid4(),
            SteeringDraft("Second change", "goal", str(goal.id), "run", "remaining_current_run", "impact"),
        )
        return fresh

    monkeypatch.setattr(service, "_existing_action_for_key", drift_after_early_fence)
    monkeypatch.setattr(service, "request_llm_decision", fresh_decision)
    # The simulated concurrent writer commits session_a mid-executor, which a real savepoint cannot survive;
    # the failed-row-kept behaviour under a real savepoint is covered in test_orchestration_failed_action_outcome.
    from contextlib import nullcontext
    from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher
    monkeypatch.setattr(OrchestrationDecisionDispatcher, "_savepoint", staticmethod(lambda _db: nullcontext()))

    assert await service._dispatch_execution_decision(session_a, run, stale) is None
    action = await session_a.scalar(select(OrchestrationAction).where(
        OrchestrationAction.decision_id == stale.id,
    ))
    await session_a.refresh(stale)
    await session_a.refresh(refreshed)
    assert action is not None and action.status == "failed"
    assert reevaluations == 1
    assert (stale.rejection_reason, refreshed.rejection_reason) == (
        "stale_steering_versions", "stale_steering_versions",
    )
    assert await session_a.scalar(select(OrchestrationSteeringResultLink.id).where(
        OrchestrationSteeringResultLink.request_id == request.id,
    )) is None
    assert (action.target_type, action.target_id) == (None, None)
    assert await session_a.scalar(select(EventLog.seq).where(
        EventLog.project_id == test_project.id,
        EventLog.event_type == "orchestration.waiting",
    )) is None


async def test_second_stale_decision_waits_without_an_action(
    db_session, conversation_goal_run, test_project, test_user, monkeypatch,
):
    goal, run, _task, _request = await _applied_item(
        db_session, conversation_goal_run, test_project, test_user, monkeypatch,
    )
    steering = OrchestrationSteeringService()
    stale = OrchestrationDecision(
        run_id=run.id, decision_type="m7", validator_status="accepted",
        input_snapshot={"steering": await steering.context_snapshot(db_session, goal, run)},
        parsed_decision={"action_type": "noop"},
    )
    db_session.add(stale)
    await db_session.flush()
    await steering.submit(
        db_session, test_project.id, goal.id, test_user.id, uuid.uuid4(),
        SteeringDraft("First change", "goal", str(goal.id), "run", "remaining_current_run", "impact"),
    )
    service = OrchestrationService()
    refreshed = None

    async def fresh_decision(_db, _run_id):
        nonlocal refreshed
        refreshed = OrchestrationDecision(
            run_id=run.id, decision_type="m7", validator_status="accepted",
            input_snapshot={"steering": await steering.context_snapshot(db_session, goal, run)},
            parsed_decision={"action_type": "noop"},
        )
        db_session.add(refreshed)
        await db_session.flush()
        await steering.submit(
            db_session, test_project.id, goal.id, test_user.id, uuid.uuid4(),
            SteeringDraft("Second change", "goal", str(goal.id), "run", "remaining_current_run", "impact"),
        )
        return refreshed

    monkeypatch.setattr(service, "request_llm_decision", fresh_decision)
    assert await service._dispatch_execution_decision(db_session, run, stale) is None
    await db_session.refresh(stale)
    await db_session.refresh(refreshed)
    assert (stale.validator_status, refreshed.validator_status) == ("rejected", "rejected")
    assert await db_session.scalar(select(OrchestrationAction.id).where(OrchestrationAction.run_id == run.id)) is None
