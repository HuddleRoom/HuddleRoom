"""SPR #148 / #144: executor 4xx rejections recorded as failed actions are outcomes, not crashes."""
import types
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationDecision, OrchestrationWait
from huddleroom.models.session import Session
from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher
from huddleroom.services.orchestration_decision_validator import validate_orchestration_decision
from tests.conftest import adopt_session
from tests.test_orchestration_basic_recovery import _actions, _agent, _blocked_setup, _make_gate, _make_orchestrated_task
from tests.test_orchestration_act_until_wait import _setup
from tests.test_orchestration_roster_mutation import sessions  # noqa: F401 -- fixture

pytestmark = pytest.mark.usefixtures("safe_goal_analysis")


@pytest.fixture
async def db_session(test_engine):
    """These dispatcher outcome drivers own their persisted preparation."""
    async with AsyncSession(test_engine, expire_on_commit=False) as db:
        yield db
        await db.rollback()


@pytest.fixture(autouse=True)
def _owned_dispatch_preparation(db_session, monkeypatch):
    from huddleroom.services.orchestration_service import OrchestrationService
    original_dispatch, original_tick = OrchestrationDecisionDispatcher.dispatch, OrchestrationService.tick

    async def prepare(db):
        root = db.sync_session.get_transaction()
        if (db is db_session and not db.in_nested_transaction()
                and (root is None or root.origin.name == "AUTOBEGIN")):
            await db.flush()
            await db.commit()

    async def dispatch(self, db, *args, **kwargs):
        await prepare(db)
        return await original_dispatch(self, db, *args, **kwargs)

    async def tick(self, db, *args, **kwargs):
        await prepare(db)
        return await original_tick(self, db, *args, **kwargs)

    monkeypatch.setattr(OrchestrationDecisionDispatcher, "dispatch", dispatch)
    monkeypatch.setattr(OrchestrationService, "tick", tick)


@pytest.fixture
def dispatch_without_corrective_interception(monkeypatch):
    """retry_task/reassign_task are always corrective, so ordinary model dispatch now turns them into a
    recovery trigger (pinned below). These SPR #148/#144 tests use them only as a vehicle for the dispatcher's
    generic 4xx/attempt-key/cap/savepoint semantics -- the same executor path an approved recovery dispatch
    runs -- so they classify the dispatch as ordinary."""
    from huddleroom.schemas.orchestration_recovery import CorrectiveDisposition
    from huddleroom.services.orchestration_service import OrchestrationService

    async def ordinary(*_args, **_kwargs):
        return CorrectiveDisposition.ORDINARY

    monkeypatch.setattr(OrchestrationService, "classify_corrective_action", ordinary)


def test_record_warning_severity_validated():
    base = {"action_type": "record_warning", "warning_type": "x", "message": "m"}
    bad = validate_orchestration_decision({**base, "severity": "high"})
    assert not bad.accepted and "hard_stop" in bad.rejection_reason
    for sev in ("recommendation", "warning", "blocker", "hard_stop"):
        assert validate_orchestration_decision({**base, "severity": sev}).accepted


async def _decision(db, run, parsed):
    decision = OrchestrationDecision(
        run_id=run.id, decision_type="coordination", input_snapshot={},
        parsed_decision=parsed, validator_status="accepted",
    )
    db.add(decision)
    await db.flush()
    return decision


async def _task_in(db, project, run, status, agent=None):
    if agent is None:
        agent = _agent("dev", "developer", ["implementation"])
        db.add(agent)
        await db.flush()
    gate = await _make_gate(db, run.id)
    task = await _make_orchestrated_task(db, project.id, run.id, gate.id, agent.id, status=status)
    return task, agent


@pytest.mark.parametrize("action_type", ["retry_task", "reassign_task"])
@pytest.mark.usefixtures("dispatch_without_corrective_interception")
async def test_dispatch_4xx_failed_action_is_returned_and_rechecked(db_session, test_project, action_type):
    _goal, run = await _setup(db_session, test_project)
    task, agent = await _task_in(db_session, test_project, run, "backlog" if action_type == "retry_task" else "in_progress")
    parsed = {"action_type": action_type, "task_id": str(task.id)}
    if action_type == "reassign_task":
        other = _agent("dev2", "developer", ["implementation"])
        db_session.add(other)
        await db_session.flush()
        parsed["agent_id"] = str(other.id)
    decision = await _decision(db_session, run, parsed)
    from huddleroom.services.orchestration_service import OrchestrationService
    dispatcher = OrchestrationDecisionDispatcher(OrchestrationService())

    action = await dispatcher.dispatch(db_session, run, decision)
    assert action.status == "failed" and action.error == f"Task is {task.status}"
    assert "applies_decision_id" not in (action.dispatch_contract or {})
    # a failed rejection is not a permanent replay: the same decision is re-checked under a new attempt key
    replay = await dispatcher.dispatch(db_session, run, decision)
    assert replay.id != action.id and replay.status == "failed" and replay.error == action.error
    assert len(await _actions(db_session, run.id, action_type)) == 2


@pytest.mark.usefixtures("dispatch_without_corrective_interception")
async def test_tick_style_dispatch_creates_backstop_wait_and_one_iteration(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project)
    task, _ = await _task_in(db_session, test_project, run, "backlog")
    from huddleroom.services.orchestration_service import OrchestrationService
    service = OrchestrationService()
    calls = {"n": 0}

    async def decide(db, run_id):
        calls["n"] += 1
        return await _decision(db, run, {"action_type": "retry_task", "task_id": str(task.id)})

    async def no_release(*_a, **_k):
        return 0

    monkeypatch.setattr(service, "request_llm_decision", decide)
    monkeypatch.setattr(service, "_release_ready_work", no_release)

    result = await service._advance_authorized_execution(db_session, goal, run)

    assert calls["n"] == 1 and "action_ids" not in result
    failed = await _actions(db_session, run.id, "retry_task")
    assert len(failed) == 1 and failed[0].status == "failed"
    waits = (await db_session.scalars(select(OrchestrationWait).where(
        OrchestrationWait.run_id == run.id, OrchestrationWait.status == "open"))).all()
    assert len(waits) == 1 and (waits[0].owner or {}).get("type") == "orchestrator_decision"
    assert f"decision:{failed[0].decision_id}:recheck" in waits[0].wait_key


async def test_invalid_record_warning_becomes_noop_backstop(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project)
    from huddleroom.services.orchestration_service import OrchestrationService
    service = OrchestrationService()
    parsed = {"action_type": "record_warning", "warning_type": "x", "severity": "high", "message": "m"}
    decision = OrchestrationDecision(
        run_id=run.id, decision_type="coordination", input_snapshot={},
        parsed_decision=parsed, validator_status="rejected", rejection_reason="bad severity",
    )
    db_session.add(decision)
    await db_session.flush()

    action = await service._dispatch_execution_decision(db_session, run, decision)

    assert action.action_type == "noop"
    assert len((await db_session.scalars(select(OrchestrationWait).where(
        OrchestrationWait.run_id == run.id, OrchestrationWait.status == "open"))).all()) == 1


@pytest.mark.usefixtures("dispatch_without_corrective_interception")
async def test_5xx_and_non_http_still_propagate(db_session, test_project):
    _goal, run = await _setup(db_session, test_project)
    task, _ = await _task_in(db_session, test_project, run, "backlog")
    decision = await _decision(db_session, run, {"action_type": "retry_task", "task_id": str(task.id)})
    from huddleroom.services.orchestration_service import OrchestrationService
    service = OrchestrationService()
    dispatcher = OrchestrationDecisionDispatcher(service)

    async def conflict(*_a, **_k):
        raise HTTPException(status_code=409, detail="conflict")

    async def boom(*_a, **_k):
        raise RuntimeError("boom")

    async def server_error(*_a, **_k):
        key = OrchestrationDecisionDispatcher.action_key(run.id, "retry_task", decision.parsed_decision)
        action = await service.reserve_action(
            db_session, run_id=run.id, idempotency_key=key, action_type="retry_task", request={})
        await service._fail_reserved_action_for_current_flow(db_session, action, "x")
        raise HTTPException(status_code=500, detail="srv")

    async def server_error_bare(*_a, **_k):
        raise HTTPException(status_code=500, detail="srv")

    for fn, exc in ((server_error_bare, HTTPException), (boom, RuntimeError), (server_error, HTTPException)):
        service.execute_retry_task_action = fn
        with pytest.raises(exc):
            await dispatcher.dispatch(db_session, run, decision)
    # a bare 4xx is recorded as a failed action rather than crashing the tick
    service.execute_retry_task_action = conflict
    assert (await dispatcher.dispatch(db_session, run, decision)).status == "failed"


def test_unhashable_severity_rejected_without_typeerror():
    base = {"action_type": "record_warning", "warning_type": "x", "message": "m"}
    for sev in (["warning"], {"a": 1}):
        assert not validate_orchestration_decision({**base, "severity": sev}).accepted


class _Boom409:
    """Patch target for TaskService.run raising a 4xx after budget/metadata were written."""


async def _patch_run_4xx(monkeypatch, service, marker):
    from huddleroom.services.task_service import TaskService

    async def reserve(db, run_id, action, task):
        run = await db.get(__import__("huddleroom.models.orchestration", fromlist=["OrchestrationRun"]).OrchestrationRun, run_id)
        run.plan_state = {**(run.plan_state or {}), "leaked_budget": marker}
        await db.flush()

    async def run_409(*_a, **_k):
        raise HTTPException(status_code=409, detail="cannot run")

    monkeypatch.setattr(service, "_reserve_recovery_task_budget", reserve)
    monkeypatch.setattr(TaskService, "run", run_409)


@pytest.mark.parametrize("action_type", ["retry_task", "reassign_task"])
@pytest.mark.usefixtures("dispatch_without_corrective_interception")
async def test_run_4xx_discards_budget_and_task_metadata_but_keeps_failed_action(
    db_session, test_project, monkeypatch, action_type,
):
    from huddleroom.models.orchestration import OrchestrationRun
    from huddleroom.services.orchestration_service import OrchestrationService
    _goal, run = await _setup(db_session, test_project)
    task, agent = await _task_in(db_session, test_project, run, "failed")
    parsed = {"action_type": action_type, "task_id": str(task.id)}
    if action_type == "reassign_task":
        other = _agent("dev2", "developer", ["implementation"])
        db_session.add(other)
        await db_session.flush()
        parsed["agent_id"] = str(other.id)
    decision = await _decision(db_session, run, parsed)
    service = OrchestrationService()
    await _patch_run_4xx(monkeypatch, service, "x")
    meta_before = dict(task.metadata_ or {})
    assigned_before = task.assigned_to

    action = await OrchestrationDecisionDispatcher(service).dispatch(db_session, run, decision)

    assert action.status == "failed" and action.error == "cannot run"
    await db_session.refresh(task)
    await db_session.refresh(run)
    assert task.metadata_ == meta_before and task.assigned_to == assigned_before and task.status == "failed"
    assert "leaked_budget" not in (run.plan_state or {})
    assert len(await _actions(db_session, run.id, action_type)) == 1


@pytest.mark.usefixtures("dispatch_without_corrective_interception")
async def test_rejected_decision_is_retried_with_new_attempt_key(db_session, test_project, monkeypatch):
    from huddleroom.models.session import Session
    from huddleroom.services.orchestration_service import OrchestrationService
    from huddleroom.services.task_service import TaskService
    _goal, run = await _setup(db_session, test_project)
    task, agent = await _task_in(db_session, test_project, run, "backlog")
    decision = await _decision(db_session, run, {"action_type": "retry_task", "task_id": str(task.id)})
    dispatcher = OrchestrationDecisionDispatcher(OrchestrationService())

    first = await dispatcher.dispatch(db_session, run, decision)
    assert first.status == "failed"
    again = await dispatcher.dispatch(db_session, run, decision)  # still backlog: re-checked, new failed row
    assert again.status == "failed" and again.id != first.id
    assert again.idempotency_key.endswith(":rejected:1")

    task.status = "failed"
    await db_session.flush()

    async def run_ok(self, db, project_id, task_id, **_k):
        session = Session(
            agent_id=agent.id, task_id=task_id, project_id=project_id, adapter_type="api",
            status="pending", input_context={}, metadata_={}, origin="auto",
        )
        db.add(session)
        await db.flush()
        return task, session.id

    monkeypatch.setattr(TaskService, "run", run_ok)
    ok = await dispatcher.dispatch(db_session, run, decision)
    assert ok.status == "completed" and ok.idempotency_key.endswith(":rejected:2")
    replay = await dispatcher.dispatch(db_session, run, decision)
    assert replay.id == ok.id


@pytest.mark.parametrize("action_type", ["schedule_meeting", "start_graph"])
async def test_partial_work_before_4xx_is_discarded(db_session, test_project, monkeypatch, action_type):
    from huddleroom.models.graph import GraphRun
    from huddleroom.models.meeting import Meeting
    from huddleroom.services.orchestration_service import OrchestrationService
    _goal, run = await _setup(db_session, test_project)
    model = Meeting if action_type == "schedule_meeting" else GraphRun
    from huddleroom.models.graph import Graph
    graph = Graph(project_id=test_project.id, name=f"g-{uuid.uuid4()}", definition={})
    db_session.add(graph)
    await db_session.flush()
    service = OrchestrationService()

    async def partial(db, run_id, request, idempotency_key, decision_id=None, **_k):
        action = await service.reserve_action(
            db, run_id=run_id, idempotency_key=idempotency_key, action_type=action_type,
            request=request, decision_id=decision_id)
        db.add(Meeting(project_id=test_project.id, title="partial", meeting_type="x")
               if model is Meeting else GraphRun(
                   graph_id=graph.id, project_id=test_project.id, current_node="a", status="active"))
        await db.flush()
        await service._fail_reserved_action_for_current_flow(db, action, "late 4xx")
        raise HTTPException(status_code=422, detail="late 4xx")

    setattr(service, {"schedule_meeting": "execute_schedule_meeting_action",
                      "start_graph": "execute_start_graph_action"}[action_type], partial)
    decision = await _decision(db_session, run, {"action_type": action_type, "reason": "r"})
    monkeypatch.setattr(service, "canonical_decision_request", lambda *_a, **_k: {"action_type": action_type})

    action = await OrchestrationDecisionDispatcher(service).dispatch(db_session, run, decision)

    assert action.status == "failed" and action.error == "late 4xx"
    assert (await db_session.scalars(select(model).where(model.project_id == test_project.id))).all() == []


async def test_record_warning_high_real_validator_tick_chain(db_session, test_project, stub_decision, owned_session_factory):
    from huddleroom.models.orchestration_process import OrchestrationWarning
    from tests.test_orchestration_runtime_e2e import _authorized_run
    adopt_session(db_session)
    stub_decision(lambda _ctx: {
        "action_type": "record_warning", "warning_type": "x", "severity": "high", "message": "m"})
    service, _goal, run = await _authorized_run(db_session, test_project)

    await service.tick(db_session, run.id)

    decision = (await db_session.scalars(
        select(OrchestrationDecision).where(OrchestrationDecision.run_id == run.id)
        .order_by(OrchestrationDecision.created_at.desc()))).first()
    assert decision.validator_status == "rejected"
    assert (await db_session.scalars(select(OrchestrationWarning))).all() == []
    assert len(await _actions(db_session, run.id, "record_warning")) == 0
    assert len(await _actions(db_session, run.id, "noop")) >= 1


async def test_completed_then_failed_iteration_stops_cleanly(db_session, test_project, monkeypatch):
    from tests.test_orchestration_act_until_wait import _wire
    goal, run = await _setup(db_session, test_project)
    service, calls = _wire(monkeypatch, db_session, run, [
        ("request_verification", "completed"), ("retry_task", "failed"), ("retry_task", "completed")], cap=5)
    result = await service._advance_authorized_execution(db_session, goal, run)
    assert calls["n"] == 2 and len(result["action_ids"]) == 2


# ---- second review ----

async def test_budget_wait_keeps_state_written_before_it_was_raised(db_session, test_project):
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
    from huddleroom.services.orchestration_service import OrchestrationService
    from tests.test_orchestration_progress_view import _authority_decision
    goal, run = await _setup(db_session, test_project)
    service = OrchestrationService()

    async def pend_then_wait(db, run_id, request, idempotency_key, decision_id=None, **_k):
        await _authority_decision(db, goal, run, None, status="pending")
        raise HTTPException(status_code=409, detail="budget_wait")

    service.execute_create_delegation_task_action = pend_then_wait
    decision = await _decision(db_session, run, {"action_type": "create_delegation_task"})
    service.canonical_decision_request = lambda *_a, **_k: {"action_type": "create_delegation_task"}

    with pytest.raises(HTTPException) as info:
        await OrchestrationDecisionDispatcher(service).dispatch(db_session, run, decision)

    assert info.value.detail == "budget_wait"
    pending = (await db_session.scalars(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.run_id == run.id))).all()
    assert len(pending) == 1 and pending[0].status == "pending"


@pytest.mark.usefixtures("dispatch_without_corrective_interception")
async def test_4xx_rollback_drops_queued_bus_events_and_dispatches(sessions, tmp_path, monkeypatch):
    from huddleroom.models.event_log import EventLog
    from huddleroom.models.orchestration import OrchestrationRun
    from huddleroom.services.orchestration_service import OrchestrationService
    from huddleroom.services.task_service import TaskService
    from huddleroom.services.session_service import SessionService
    from huddleroom.services.event_bus import EventBusService, emit_event_once
    from tests.test_orchestration_roster_mutation import seed_project_user
    async with sessions() as setup:
        project, _ = await seed_project_user(setup, tmp_path)
        goal, run = await _setup(setup, project)
        task, agent = await _task_in(setup, project, run, "failed")
        decision = await _decision(setup, run, {"action_type": "retry_task", "task_id": str(task.id)})
        await setup.commit()
    bus, published, registered, abandoned = EventBusService(), [], [], {}
    monkeypatch.setattr(bus, "put", lambda event: published.append(event.id))
    monkeypatch.setattr("huddleroom.workers.task_runner.register_session", lambda *args, **kwargs: registered.append(args))
    async def emit_then_409(self, db, *_args, **_kwargs):
        event, _ = await emit_event_once(db, project.id, "test.rolled_back", {}, _bus=bus)
        pending = Session(project_id=project.id, task_id=task.id, agent_id=agent.id,
                          adapter_type="api", status="pending", runner_task_id=str(uuid.uuid4()))
        db.add(pending)
        await db.flush()
        SessionService()._schedule_dispatch_after_commit(db, pending.id, project.id, "api", pending.runner_task_id)
        abandoned.update(event_id=event.id, session_id=pending.id)
        raise HTTPException(status_code=409, detail="cannot run")
    monkeypatch.setattr(TaskService, "run", emit_then_409)
    service = OrchestrationService()
    async with sessions() as caller:
        async with service._lock_goal_for_baseline_transition(caller, goal.id):
            current = await caller.get(OrchestrationRun, run.id)
            accepted = await caller.get(OrchestrationDecision, decision.id)
            outer, _ = await emit_event_once(caller, project.id, "test.outer", {}, _bus=bus)
            action = await OrchestrationDecisionDispatcher(service).dispatch(caller, current, accepted)
            assert action.status == "failed"
            assert published == [] and registered == []
            await caller.commit()
    assert published == [outer.id] and registered == []
    async with sessions() as fresh:
        assert await fresh.scalar(select(EventLog).where(EventLog.id == abandoned["event_id"])) is None
        assert await fresh.get(Session, abandoned["session_id"]) is None


async def test_steering_versions_changed_keeps_failed_row(db_session, test_project):
    from huddleroom.services.orchestration_service import OrchestrationService
    from huddleroom.services.orchestration_steering import SteeringVersionsChanged
    _goal, run = await _setup(db_session, test_project)
    service = OrchestrationService()
    key = "k-steer"

    async def fail_then_stale(db, run_id, request, idempotency_key, decision_id=None, **_k):
        action = await service.reserve_action(
            db, run_id=run_id, idempotency_key=idempotency_key, action_type="noop", request=request,
            decision_id=decision_id)
        await service._fail_reserved_action_for_current_flow(db, action, "stale_steering_versions")
        raise SteeringVersionsChanged()

    service.execute_noop_action = fail_then_stale
    decision = await _decision(db_session, run, {"action_type": "noop"})
    with pytest.raises(SteeringVersionsChanged):
        await OrchestrationDecisionDispatcher(service).dispatch(db_session, run, decision)
    rows = await _actions(db_session, run.id, "noop")
    assert [(r.status, r.error) for r in rows] == [("failed", "stale_steering_versions")]


@pytest.mark.usefixtures("dispatch_without_corrective_interception")
async def test_rejection_attempts_are_capped(db_session, test_project):
    from huddleroom.services.orchestration_service import OrchestrationService
    _goal, run = await _setup(db_session, test_project)
    task, _ = await _task_in(db_session, test_project, run, "backlog")
    decision = await _decision(db_session, run, {"action_type": "retry_task", "task_id": str(task.id)})
    dispatcher = OrchestrationDecisionDispatcher(OrchestrationService())
    seen = [await dispatcher.dispatch(db_session, run, decision) for _ in range(5)]
    assert len(await _actions(db_session, run.id, "retry_task")) == 3
    assert seen[3].id == seen[4].id == seen[2].id and seen[4].status == "failed"


@pytest.mark.usefixtures("dispatch_without_corrective_interception")
async def test_rejection_cap_resets_after_other_progress(db_session, test_project, monkeypatch):
    from huddleroom.models.session import Session
    from huddleroom.services.orchestration_service import OrchestrationService
    from huddleroom.services.task_service import TaskService
    _goal, run = await _setup(db_session, test_project)
    task, agent = await _task_in(db_session, test_project, run, "backlog")
    decision = await _decision(db_session, run, {"action_type": "retry_task", "task_id": str(task.id)})
    dispatcher = OrchestrationDecisionDispatcher(OrchestrationService())
    for _ in range(4):
        await dispatcher.dispatch(db_session, run, decision)
    assert len(await _actions(db_session, run.id, "retry_task")) == 3

    db_session.add(OrchestrationAction(
        run_id=run.id, idempotency_key=f"progress-{uuid.uuid4()}", action_type="request_verification",
        status="completed"))
    await db_session.flush()
    task.status = "failed"

    async def run_ok(self, db, project_id, task_id, **_k):
        session = Session(agent_id=agent.id, task_id=task_id, project_id=project_id, adapter_type="api",
                          status="pending", input_context={}, metadata_={}, origin="auto")
        db.add(session)
        await db.flush()
        return task, session.id

    monkeypatch.setattr(TaskService, "run", run_ok)
    ok = await dispatcher.dispatch(db_session, run, decision)
    assert ok.status == "completed" and ok.idempotency_key.endswith(":rejected:3")


async def test_attempt_prefix_match_uses_delimiter(db_session, test_project):
    from huddleroom.services.orchestration_service import OrchestrationService
    _goal, run = await _setup(db_session, test_project)
    dispatcher = OrchestrationDecisionDispatcher(OrchestrationService())
    base = f"run:{run.id}:kind:x:request:abc"
    for k in (base, base + "def", base + "ghi"):  # the last two only share a raw prefix
        db_session.add(OrchestrationAction(run_id=run.id, idempotency_key=k, action_type="x", status="failed"))
    await db_session.flush()
    key, capped = await dispatcher._attempt_key(db_session, run.id, base, uuid.uuid4(), False)
    assert capped is None and key == base + ":rejected:1"


# ---------------------------------------------------------------------------
# SPR 180/184: ordinary model dispatch cannot apply a corrective fix after a
# durable failure; it creates one idempotent recovery trigger and mutates nothing.
# ---------------------------------------------------------------------------


async def _failed_task_with_session(db, project, run):
    task, agent = await _task_in(db, project, run, "failed")
    db.add(Session(agent_id=agent.id, task_id=task.id, project_id=project.id, adapter_type="api", status="failed",
                   input_context={}, metadata_={}, origin="auto"))
    await db.flush()
    return task, agent


async def _durable_state(db, task):
    from sqlalchemy import func
    from huddleroom.models.task import Task

    await db.refresh(task)
    sessions = await db.scalar(select(func.count(Session.id)).where(Session.task_id == task.id))
    tasks = await db.scalar(select(func.count(Task.id)))
    return task.status, task.assigned_to, sessions, tasks


@pytest.mark.parametrize("action_type", ["retry_task", "reassign_task", "replacement_delegation"])
async def test_model_corrective_action_after_durable_failure_creates_one_trigger(
    db_session, test_project, monkeypatch, action_type,
):
    from huddleroom.services.orchestration_service import OrchestrationService
    from tests.test_orchestration_basic_recovery import _fake_task_run

    calls: list = []
    monkeypatch.setattr("huddleroom.services.orchestration_service.TaskService.run", _fake_task_run(calls))
    _goal, run = await _setup(db_session, test_project)
    task, agent = await _failed_task_with_session(db_session, test_project, run)
    other = _agent("dev2", "developer", ["implementation"])
    db_session.add(other)
    await db_session.flush()
    parsed = {
        "retry_task": {"action_type": "retry_task", "task_id": str(task.id)},
        "reassign_task": {"action_type": "reassign_task", "task_id": str(task.id), "agent_id": str(other.id)},
        "replacement_delegation": {
            "action_type": "create_delegation_task", "agent_id": str(other.id), "work_function": "implementation",
            "scope": "Redo the failed implementation.", "deliverable": "A working patch.",
            "parent_task_id": str(task.id),
        },
    }[action_type]
    decision = await _decision(db_session, run, parsed)
    before = await _durable_state(db_session, task)
    dispatcher = OrchestrationDecisionDispatcher(OrchestrationService())

    await dispatcher.dispatch(db_session, run, decision)
    await dispatcher.dispatch(db_session, run, decision)  # replay

    effect_type = "create_delegation_task" if action_type == "replacement_delegation" else action_type
    assert [a for a in await _actions(db_session, run.id, effect_type) if a.status == "completed"] == []
    assert calls == []
    assert await _durable_state(db_session, task) == before
    recovery_triggers = await _actions(db_session, run.id, "recovery_trigger")
    assert len(recovery_triggers) == 1
    assert recovery_triggers[0].request["run_id"] == str(run.id)


def test_spr230_structured_transaction_conflict_is_not_misread_as_wait():
    from huddleroom.services.orchestration_service import OrchestrationService
    conflict = HTTPException(409, detail={"conflict": "orchestration_transaction_shape"})
    assert OrchestrationService._waiting_step_for_http_error(conflict) is None
