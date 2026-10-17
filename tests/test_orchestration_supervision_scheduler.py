import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select

from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.services.orchestration_supervision_scheduler import OrchestrationSupervisionScheduler


pytestmark = pytest.mark.asyncio


async def _run(db, project):
    goal = OrchestrationGoal(project_id=project.id, objective="Keep moving", status="active")
    db.add(goal)
    await db.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized")
    db.add(run)
    await db.flush()
    return goal, run


async def test_record_event_returns_matching_run_ids_and_preserves_its_first_deadline(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    scheduler = OrchestrationSupervisionScheduler()
    now = _utcnow()

    assert await scheduler.record_event(db_session, "task.status_changed", {"task_id": str(uuid.uuid4())}, now=now) == []
    assert await scheduler.record_event(db_session, "task.status_changed", {"run_id": str(run.id)}, now=now) == [run.id]
    first = dict(run.supervision_state)
    assert first["judgment_dirty"] is True
    assert _utcnow() <= datetime.fromisoformat(first["judgment_due_at"]) <= now + timedelta(seconds=1)

    assert await scheduler.record_event(db_session, "task.status_changed", {"run_id": str(run.id)}, now=now + timedelta(milliseconds=500)) == [run.id]
    assert run.supervision_state["judgment_due_at"] == first["judgment_due_at"]


async def test_record_event_routes_child_run_events_to_its_active_parent(db_session, test_project):
    parent, parent_run = await _run(db_session, test_project)
    child = OrchestrationGoal(
        project_id=test_project.id, parent_goal_id=parent.id, objective="Child work", status="active",
        continuous_origin_key="child-work", parent_contract_snapshot={}, goal_delta={},
    )
    db_session.add(child)
    await db_session.flush()
    child_run = OrchestrationRun(goal_id=child.id, status="running", phase="authorized")
    db_session.add(child_run)
    await db_session.flush()

    affected = await OrchestrationSupervisionScheduler().record_event(
        db_session, "orchestration.run_completed", {"run_id": str(child_run.id)}
    )

    assert set(affected) == {child_run.id, parent_run.id}


async def test_record_event_rechecks_authorized_control_inside_the_goal_lock(db_session, test_project):
    """A run no longer authorized at mutation time must not be dirtied."""
    _goal, run = await _run(db_session, test_project)
    run.phase = "completed"

    assert await OrchestrationSupervisionScheduler().record_event(
        db_session, "task.status_changed", {"run_id": str(run.id)}
    ) == []
    assert run.supervision_state in (None, {})


async def test_steering_change_wakes_the_matching_run(db_session, test_project):
    _goal, run = await _run(db_session, test_project)

    assert await OrchestrationSupervisionScheduler().record_event(
        db_session, "orchestration.steering_changed", {"run_id": str(run.id)}
    ) == [run.id]


async def test_judgment_fence_fingerprints_full_reports_while_payload_is_concise(db_session, test_project):
    from types import SimpleNamespace

    from huddleroom.services.orchestration_supervision_context import OrchestrationSupervisionContextBuilder
    from tests.test_orchestration_supervision_context import _report_run

    goal, run, _tasks, _markers = await _report_run(db_session, test_project.id)

    async def contract_version(db, goal, run):
        return "v1"

    service = SimpleNamespace(supervision=SimpleNamespace(_contract_version=contract_version))
    builder = OrchestrationSupervisionContextBuilder()
    payload, fingerprint = await OrchestrationSupervisionScheduler._judgment_payload(db_session, service, builder, goal, run)

    full = await builder.build(db_session, goal, run, body_limit=None, full_reports=True)
    assert fingerprint == builder.fingerprint_snapshot({**payload, "context": full})
    assert fingerprint != builder.fingerprint_snapshot(payload)  # the provider payload itself is concise
    (unrelated,) = [
        a for a in payload["context"]["actions"]
        if a["action_type"] == "report_consumed" and a["request"]["task_id"] == str(_tasks["unrelated"].id)
    ]
    assert unrelated["request"]["report"]["truncated"] == {"changes": 5, "decisions": 4}


async def test_waiver_event_wakes_only_authorized_run(db_session, test_project):
    _, run = await _run(db_session, test_project)
    scheduler = OrchestrationSupervisionScheduler()
    assert await scheduler.record_event(db_session, "meeting.action_item_waived", {"run_id": str(run.id)}) == [run.id]
    assert run.supervision_state["judgment_dirty"] is True
    run.phase = "completed"
    assert await scheduler.record_event(db_session, "meeting.action_item_waived", {"run_id": str(run.id)}) == []


async def test_goal_resumed_event_runs_one_full_tick_after_coalescing_without_local_sweep(monkeypatch):
    import asyncio

    from huddleroom.services import orchestration_service
    from huddleroom.services.event_bus import EventBusService
    from huddleroom.services.orchestration_service import OrchestrationService
    from huddleroom.workers import orchestration_tasks

    run_id = uuid.uuid4()
    ticks, local_sweeps = [], []

    async def fake_tick(self, db, rid, *, local_only=False):
        ticks.append((rid, local_only))
        return {}

    async def fake_evaluate_run(self, db, rid, *, now=None):
        local_sweeps.append(rid)
        return 0

    class FakeDb:
        from types import SimpleNamespace
        sync_session = SimpleNamespace(get_transaction=lambda: None)

        def in_nested_transaction(self):
            return False

        async def commit(self):
            pass

        async def rollback(self):
            pass

    class FakeSession:
        async def __aenter__(self):
            return FakeDb()

        async def __aexit__(self, *exc):
            return False

    bus = EventBusService()

    monkeypatch.setattr(OrchestrationService, "tick", fake_tick)
    monkeypatch.setattr(OrchestrationSupervisionScheduler, "evaluate_run", fake_evaluate_run)
    monkeypatch.setattr(orchestration_tasks, "AsyncSessionLocal", lambda: FakeSession())
    monkeypatch.setattr(orchestration_tasks, "get_event_bus", lambda: bus)
    monkeypatch.setattr(orchestration_tasks.settings, "orchestration_event_coalesce_seconds", 0)

    supervisor = asyncio.create_task(orchestration_tasks.run_orchestration_event_supervisor())
    try:
        await _spr213_wait_for_subscription(bus)
        for _ in range(3):  # repeated resumes coalesce to one tick
            bus.put(_spr213_event(run_id, orchestration_service.GOAL_RESUMED_EVENT_TYPE))
        await _spr213_wait_for_count(ticks, 1)
        assert ticks == [(run_id, False)]
        assert local_sweeps == []
    finally:
        supervisor.cancel()
        with pytest.raises(asyncio.CancelledError):
            await supervisor

    assert ticks == [(run_id, False)]
    assert local_sweeps == []


async def test_goal_resumed_event_runs_one_full_tick_and_no_local_evaluation(db_session, test_project, monkeypatch):
    from types import SimpleNamespace

    from huddleroom.models.event_log import EventLog
    from huddleroom.services import orchestration_service
    from huddleroom.workers import orchestration_tasks

    goal, run = await _run(db_session, test_project)
    event = EventLog(
        project_id=test_project.id, event_type=orchestration_service.GOAL_RESUMED_EVENT_TYPE,
        dedup_key=f"test:goal-resumed:{run.id}", payload={"run_id": str(run.id)}, source="test",
    )
    db_session.add(event)
    await db_session.flush()

    # SPR230/SPR231 a native worker owns a fresh transaction, not fixture's caller BEGIN.
    from sqlalchemy.ext.asyncio import async_sessionmaker
    await db_session.commit()
    sessions = async_sessionmaker(db_session.bind, expire_on_commit=False)

    ticks, evaluations = [], []

    async def fake_tick(self, db, run_id, local_only=True):
        ticks.append((run_id, local_only))

    async def fake_evaluate_run(self, db, run_id, now=None):
        evaluations.append(run_id)
        return 0

    monkeypatch.setattr(orchestration_tasks, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(orchestration_tasks, "settings", SimpleNamespace(orchestration_event_coalesce_seconds=0))
    monkeypatch.setattr(orchestration_service.OrchestrationService, "tick", fake_tick)
    monkeypatch.setattr(orchestration_tasks.OrchestrationSupervisionScheduler, "evaluate_run", fake_evaluate_run)

    assert await orchestration_tasks.evaluate_supervision_event_async(event.id) == 1
    assert ticks == [(run.id, False)]
    assert evaluations == []


class _Spr213Session:
    from types import SimpleNamespace
    sync_session = SimpleNamespace(get_transaction=lambda: None)

    def in_nested_transaction(self):
        return False

    def __init__(self, commits=None, rollbacks=None):
        self.commits = commits if commits is not None else []
        self.rollbacks = rollbacks if rollbacks is not None else []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def commit(self):
        self.commits.append(True)

    async def rollback(self):
        self.rollbacks.append(True)


def _spr213_event(run_id, event_type):
    from huddleroom.services.event_bus import BusEvent

    return BusEvent(uuid.uuid4(), uuid.uuid4(), event_type, {"run_id": str(run_id)}, "test", _utcnow())


async def _spr213_wait_for_count(items, count):
    import asyncio

    async def until_count():
        while len(items) < count:
            await asyncio.sleep(0)

    await asyncio.wait_for(until_count(), 1)


async def _spr213_wait_for_subscription(bus):
    import asyncio

    async def until_registered():
        while not bus._subscribers:
            await asyncio.sleep(0)

    await asyncio.wait_for(until_registered(), 1)


def _spr213_patch_supervisor(monkeypatch, bus, *, records=None, commits=None, rollbacks=None):
    from huddleroom.workers import orchestration_tasks

    monkeypatch.setattr(orchestration_tasks, "get_event_bus", lambda: bus)
    monkeypatch.setattr(orchestration_tasks, "AsyncSessionLocal", lambda: _Spr213Session(commits, rollbacks))
    monkeypatch.setattr(orchestration_tasks.settings, "orchestration_event_coalesce_seconds", 0)

    if records is not None:
        async def record(self, db, event_type, payload, **kwargs):
            records.append(uuid.UUID(payload["run_id"]))
            return [uuid.UUID(payload["run_id"])]

        monkeypatch.setattr(OrchestrationSupervisionScheduler, "record_event", record)


async def test_spr213_supervisor_collects_without_execution_permission(monkeypatch):
    import asyncio

    from huddleroom.services import orchestration_service
    from huddleroom.services.event_bus import EventBusService
    from huddleroom.workers import orchestration_tasks

    class TrackingEvent(asyncio.Event):
        def __init__(self):
            super().__init__()
            self.awaited = asyncio.Event()

        async def wait(self):
            self.awaited.set()
            await super().wait()

    bus = EventBusService()
    subscription_ready, execution_ready = asyncio.Event(), TrackingEvent()
    resumed_id, ordinary_id = uuid.uuid4(), uuid.uuid4()
    records, commits, ticks, evaluations = [], [], [], []
    _spr213_patch_supervisor(monkeypatch, bus, records=records, commits=commits)

    async def tick(self, db, run_id, *, local_only=False):
        ticks.append((run_id, local_only))

    async def evaluate(self, db, run_id, *, now=None):
        evaluations.append(run_id)

    monkeypatch.setattr(orchestration_service.OrchestrationService, "tick", tick)
    monkeypatch.setattr(OrchestrationSupervisionScheduler, "evaluate_run", evaluate)
    supervisor = asyncio.create_task(orchestration_tasks.run_orchestration_event_supervisor(
        ready=subscription_ready, execution_ready=execution_ready,
    ))
    try:
        await asyncio.wait_for(subscription_ready.wait(), 1)
        bus.put(_spr213_event(resumed_id, orchestration_service.GOAL_RESUMED_EVENT_TYPE))
        bus.put(_spr213_event(ordinary_id, "task.status_changed"))
        await _spr213_wait_for_count(records, 1)
        await asyncio.wait_for(execution_ready.awaited.wait(), 1)
        assert ticks == evaluations == []
        assert commits  # ordinary dirty state was committed while execution was gated
        execution_ready.set()
        await _spr213_wait_for_count(ticks, 1)
        await _spr213_wait_for_count(evaluations, 1)
        assert ticks == [(resumed_id, False)]
        assert evaluations == [ordinary_id]
    finally:
        supervisor.cancel()
        with pytest.raises(asyncio.CancelledError):
            await supervisor


async def test_spr213_drain_consumes_arrivals_during_tick(monkeypatch):
    import asyncio

    from huddleroom.services import orchestration_service
    from huddleroom.services.event_bus import EventBusService
    from huddleroom.workers import orchestration_tasks

    bus = EventBusService()
    entered, release = asyncio.Event(), asyncio.Event()
    run_a, run_b = uuid.uuid4(), uuid.uuid4()
    records, ticks, evaluations = [], [], []
    _spr213_patch_supervisor(monkeypatch, bus, records=records)

    async def tick(self, db, run_id, *, local_only=False):
        ticks.append((run_id, local_only))
        if run_id == run_a and ticks.count((run_a, False)) == 1:
            entered.set()
            await release.wait()

    async def evaluate(self, db, run_id, *, now=None):
        evaluations.append(run_id)

    monkeypatch.setattr(orchestration_service.OrchestrationService, "tick", tick)
    monkeypatch.setattr(OrchestrationSupervisionScheduler, "evaluate_run", evaluate)
    supervisor = asyncio.create_task(orchestration_tasks.run_orchestration_event_supervisor())
    try:
        await _spr213_wait_for_subscription(bus)
        for _ in range(2):
            bus.put(_spr213_event(run_a, orchestration_service.GOAL_RESUMED_EVENT_TYPE))
        await asyncio.wait_for(entered.wait(), 1)
        for _ in range(2):
            bus.put(_spr213_event(run_b, orchestration_service.GOAL_RESUMED_EVENT_TYPE))
        bus.put(_spr213_event(run_a, "task.status_changed"))
        bus.put(_spr213_event(run_a, orchestration_service.GOAL_RESUMED_EVENT_TYPE))
        await _spr213_wait_for_count(records, 1)
        release.set()
        await _spr213_wait_for_count(ticks, 3)
        assert ticks.count((run_a, False)) == 2
        assert ticks.count((run_b, False)) == 1
        assert evaluations == []
    finally:
        release.set()
        supervisor.cancel()
        with pytest.raises(asyncio.CancelledError):
            await supervisor


@pytest.mark.parametrize("failure_kind", ["full", "local"])
async def test_spr213_drain_failure_does_not_starve_runs(monkeypatch, failure_kind):
    import asyncio

    from huddleroom.services import orchestration_service
    from huddleroom.services.event_bus import EventBusService
    from huddleroom.workers import orchestration_tasks

    bus = EventBusService()
    ready, entered, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    first, neighbor, later = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    records, commits, rollbacks, ticks, evaluations = [], [], [], [], []
    _spr213_patch_supervisor(monkeypatch, bus, records=records, commits=commits, rollbacks=rollbacks)

    async def tick(self, db, run_id, *, local_only=False):
        ticks.append(run_id)
        if failure_kind == "full" and run_id == first:
            entered.set()
            await release.wait()
            raise RuntimeError("injected full tick failure")

    async def evaluate(self, db, run_id, *, now=None):
        evaluations.append(run_id)
        if failure_kind == "local" and run_id == first:
            entered.set()
            await release.wait()
            raise RuntimeError("injected local evaluation failure")

    monkeypatch.setattr(orchestration_service.OrchestrationService, "tick", tick)
    monkeypatch.setattr(OrchestrationSupervisionScheduler, "evaluate_run", evaluate)
    event_type = orchestration_service.GOAL_RESUMED_EVENT_TYPE if failure_kind == "full" else "task.status_changed"
    supervisor = asyncio.create_task(orchestration_tasks.run_orchestration_event_supervisor(ready=ready))
    try:
        await asyncio.wait_for(ready.wait(), 1)
        bus.put(_spr213_event(first, event_type))
        await asyncio.wait_for(entered.wait(), 1)
        bus.put(_spr213_event(neighbor, event_type))
        bus.put(_spr213_event(later, event_type))
        if failure_kind == "local":
            await _spr213_wait_for_count(records, 3)
        release.set()
        calls = ticks if failure_kind == "full" else evaluations
        await _spr213_wait_for_count(calls, 3)
        assert set(calls) == {first, neighbor, later}
        assert rollbacks
        assert commits
    finally:
        release.set()
        supervisor.cancel()
        with pytest.raises(asyncio.CancelledError):
            await supervisor


@pytest.mark.parametrize("active", [False, True], ids=["pending", "active"])
async def test_spr213_supervisor_awaits_drain_on_cancel(monkeypatch, active):
    import asyncio

    from huddleroom.services import orchestration_service
    from huddleroom.services.event_bus import EventBusService
    from huddleroom.workers import orchestration_tasks

    bus = EventBusService()
    ready, entered, finalized = asyncio.Event(), asyncio.Event(), asyncio.Event()
    execution_ready = asyncio.Event()
    ticks = []
    _spr213_patch_supervisor(monkeypatch, bus)

    async def tick(self, db, run_id, *, local_only=False):
        ticks.append(run_id)
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            finalized.set()

    monkeypatch.setattr(orchestration_service.OrchestrationService, "tick", tick)
    supervisor = asyncio.create_task(orchestration_tasks.run_orchestration_event_supervisor(
        ready=ready, execution_ready=execution_ready,
    ))
    await asyncio.wait_for(ready.wait(), 1)
    bus.put(_spr213_event(uuid.uuid4(), orchestration_service.GOAL_RESUMED_EVENT_TYPE))
    if active:
        execution_ready.set()
        await asyncio.wait_for(entered.wait(), 1)
    else:
        # A subscription queue acknowledgement confirms event delivery before cancellation.
        async def queue_drained():
            while bus._subscribers[0][0].qsize():
                await asyncio.sleep(0)
        await asyncio.wait_for(queue_drained(), 1)
    supervisor.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(supervisor, 1)
    assert len(ticks) == (1 if active else 0)
    assert finalized.is_set() is active
    assert bus._subscribers == []


@pytest.mark.asyncio
@pytest.mark.parametrize("write_kind", ["orm", "core"])
@pytest.mark.parametrize("outcome", ["success", "refusal", "exception"])
async def test_supplied_session_preserves_unrelated_writes_on_all_outcomes(
    test_engine, monkeypatch, tmp_path, write_kind, outcome,
):
    # This checks the reusable helper's publication/error boundary, not a real owning job.
    import asyncio
    from fastapi import HTTPException
    from sqlalchemy import delete, event, update
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from huddleroom.models.project import Project
    from huddleroom.services.orchestration_service import OrchestrationService

    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with factory() as setup:
        project = Project(name="SPR230/SPR231 caller preservation", description="committed", workspace_path=str(tmp_path), config={})
        setup.add(project)
        await setup.flush()
        goal = OrchestrationGoal(project_id=project.id, objective="Caller preservation", status="active")
        setup.add(goal)
        await setup.flush()
        run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized", budget_state={})
        setup.add(run)
        await setup.commit()
        project_id, goal_id, run_id = project.id, goal.id, run.id

    attempted = []

    async def boundary_tick(self, db, rid, **kwargs):
        attempted.append(rid)
        if outcome == "refusal":
            raise HTTPException(409, detail={"conflict": "orchestration_transaction_shape"})
        if outcome == "exception":
            raise RuntimeError("injected downstream failure")
        return {"run_id": rid, "status": "running"}

    monkeypatch.setattr(OrchestrationService, "tick", boundary_tick)
    try:
        async with factory() as caller:
            project = await caller.get(Project, project_id)
            if write_kind == "orm":
                project.description = "caller-only"
                await caller.flush()
            else:
                caller.expunge(project)
                await caller.execute(update(Project).where(Project.id == project_id).values(
                    description="caller-only",
                ).execution_options(synchronize_session=False))
                assert not caller.new and not caller.dirty and not caller.deleted
            root = caller.sync_session.get_transaction()
            calls = []
            for name in ("before_flush", "before_commit", "after_rollback"):
                event.listen(caller.sync_session, name, lambda *args, name=name: calls.append(name))
            try:
                await asyncio.wait_for(OrchestrationSupervisionScheduler()._tick(caller, run_id), 3)
            except (HTTPException, RuntimeError):
                assert outcome != "success"
            assert attempted == [run_id]
            assert calls == []
            assert caller.sync_session.get_transaction() is root and root.is_active
            with caller.no_autoflush:
                assert await caller.scalar(select(Project.description).where(Project.id == project_id)) == "caller-only"
            async with factory() as fresh:
                assert (await fresh.get(Project, project_id)).description == "committed"
            await caller.rollback()
        async with factory() as fresh:
            assert (await fresh.get(Project, project_id)).description == "committed"
    finally:
        async with factory() as cleanup:
            await cleanup.execute(delete(OrchestrationRun).where(OrchestrationRun.id == run_id))
            await cleanup.execute(delete(OrchestrationGoal).where(OrchestrationGoal.id == goal_id))
            await cleanup.execute(delete(Project).where(Project.id == project_id))
            await cleanup.commit()
