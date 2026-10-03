# pylint: disable=cyclic-import
import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from huddleroom.config import settings
from huddleroom.models.artifact import Artifact
from huddleroom.models.event_log import EventLog
from huddleroom.models.meeting import Meeting
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun, OrchestrationSchedulerState
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationWait
from huddleroom.models.orchestration_memory import OrchestrationMemorySection
from huddleroom.models.protocol import Protocol, ProtocolInstance
from huddleroom.models.task import Task
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_supervision_context import OrchestrationSupervisionContextBuilder
from huddleroom.services.orchestration_supervision_scheduler import OrchestrationSupervisionScheduler
from huddleroom.services.orchestration_supervision import SupervisionAssessment


pytestmark = pytest.mark.asyncio


async def _run(db, project):
    goal = OrchestrationGoal(project_id=project.id, objective="Supervise durable work", status="active")
    db.add(goal)
    await db.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized")
    db.add(run)
    await db.flush()
    return goal, run


async def _task_for_run(db, project, run):
    task = Task(
        project_id=project.id,
        title="Durable work",
        status="in_progress",
        metadata_={"orchestration": {"run_id": str(run.id)}},
    )
    db.add(task)
    await db.flush()
    return task


async def _fresh_sessions(db, monkeypatch):
    """Commit fixture data, then make runtime transaction boundaries observable."""
    from sqlalchemy.ext.asyncio import async_sessionmaker
    import huddleroom.database as database_module
    import huddleroom.services.orchestration_service as service_module

    sessions = async_sessionmaker(db.bind, expire_on_commit=False)
    await db.commit()
    monkeypatch.setattr(database_module, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(service_module, "AsyncSessionLocal", sessions)
    return sessions


async def test_evaluate_run_accepts_legacy_naive_deadline_as_utc(db_session, test_project, monkeypatch):
    """Dropping UTC normalization at the scheduler boundary must fail here."""
    _, run = await _run(db_session, test_project)
    called = []

    async def tick(_db, run_id, *, local_only=False):  # pylint: disable=unused-argument
        called.append((run_id, local_only))
        return {"outcome": "continue"}

    scheduler = OrchestrationSupervisionScheduler(tick=tick)
    run.supervision_state = {
        "judgment_dirty": True,
        "judgment_due_at": "2026-09-10T12:00:00",
    }

    assert await scheduler.evaluate_run(
        db_session, run.id, now=datetime(2026, 9, 10, 12, 0, 1, tzinfo=timezone.utc)
    ) == 1
    assert called == [(run.id, True)]


async def test_real_supervision_events_resolve_their_durable_owner_to_the_run(db_session, test_project):
    """Removing any real event name or its durable owner resolver must fail here."""
    _, run = await _run(db_session, test_project)
    task = await _task_for_run(db_session, test_project, run)
    artifact = Artifact(project_id=test_project.id, name="Result", artifact_type="report", linked_task_id=task.id)
    meeting = Meeting(project_id=test_project.id, title="Decision", meeting_type="decision", source_task_id=task.id)
    protocol = Protocol(project_id=test_project.id, name="Review", version="1", definition={}, triggers=[])
    db_session.add_all([artifact, meeting, protocol])
    await db_session.flush()
    instance = ProtocolInstance(
        protocol_id=protocol.id,
        project_id=test_project.id,
        linked_task_id=task.id,
        current_state="done",
    )
    db_session.add(instance)
    await db_session.flush()

    scheduler = OrchestrationSupervisionScheduler()
    cases = (
        ("artifact.breaking_change", {"artifact_id": str(artifact.id)}),
        ("meeting.concluded", {"meeting_id": str(meeting.id)}),
        ("protocol.failed", {"protocol_instance_id": str(instance.id)}),
        ("protocol.state_transitioned", {"protocol_instance_id": str(instance.id)}),
        ("orchestration.run_completed", {"run_id": str(run.id)}),
    )
    for event_type, payload in cases:
        run.supervision_state = {}
        assert await scheduler.record_event(
            db_session, event_type, payload, now=datetime.now(timezone.utc)
        ) == [run.id]


async def test_fingerprint_uses_full_memory_while_provider_context_is_bounded(db_session, test_project):
    """Fingerprinting the provider-truncated body would miss a durable change here."""
    goal, run = await _run(db_session, test_project)
    memory = OrchestrationMemorySection(
        project_id=test_project.id,
        goal_id=goal.id,
        section_key="long-fact",
        title="Long fact",
        body="a" * 4_000 + "first",
        created_by="orchestrator",
        fact_status="unverified",
        provenance={},
    )
    db_session.add(memory)
    await db_session.flush()
    builder = OrchestrationSupervisionContextBuilder()

    first = await builder.fingerprint(db_session, goal, run)
    provider = await builder.build_for_provider(db_session, goal, run, string_limit=32)
    memory.body = "a" * 4_000 + "second"
    await db_session.flush()

    assert len(provider["memory"][0]["body"]) == 32
    assert await builder.fingerprint(db_session, goal, run) != first


async def test_provider_context_is_a_bounded_copy_of_the_fingerprinted_snapshot(db_session, test_project):
    """Provider truncation must not change the canonical object that was fingerprinted."""
    goal, run = await _run(db_session, test_project)
    memory = OrchestrationMemorySection(
        project_id=test_project.id, goal_id=goal.id, section_key="canonical-copy",
        title="Canonical copy", body="x" * 2_000, created_by="orchestrator",
        fact_status="unverified", provenance={},
    )
    db_session.add(memory)
    await db_session.flush()
    builder = OrchestrationSupervisionContextBuilder()

    snapshot = await builder.build(db_session, goal, run, body_limit=None)
    fingerprint = builder.fingerprint_snapshot(snapshot)
    provider = builder.provider_snapshot(snapshot, string_limit=32)

    assert len(provider["memory"][0]["body"]) == 32
    assert len(snapshot["memory"][0]["body"]) == 2_000
    assert builder.fingerprint_snapshot(snapshot) == fingerprint


async def test_fingerprint_includes_goal_and_run_semantic_state(db_session, test_project):
    """A changed control field must fence an already claimed provider response."""
    goal, run = await _run(db_session, test_project)
    builder = OrchestrationSupervisionContextBuilder()
    first = await builder.fingerprint(db_session, goal, run)

    goal.constraints = {"budget": "reduced"}
    run.plan_state = {"status": "accepted", "revision": 2}
    await db_session.flush()

    assert await builder.fingerprint(db_session, goal, run) != first


async def test_wakeup_coalesces_bursty_sqlite_events_once(monkeypatch):
    """Scheduling an evaluator per committed event defeats the coalescing policy."""
    from huddleroom.workers import orchestration_tasks

    called = []

    async def evaluate(event_id):
        called.append(event_id)
        await asyncio.sleep(0)
        return 0

    monkeypatch.setattr(orchestration_tasks, "evaluate_supervision_event_async", evaluate)
    monkeypatch.setattr(
        orchestration_tasks,
        "settings",
        SimpleNamespace(is_sqlite=True, orchestration_event_coalesce_seconds=1),
    )

    orchestration_tasks.request_supervision_wakeup(uuid.uuid4())
    orchestration_tasks.request_supervision_wakeup(uuid.uuid4())
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert len(called) == 1


async def test_runtime_scheduler_uses_supervision_settings_for_intervals_and_bounds(monkeypatch):
    """Hard-coded runtime limits ignore the validated Settings contract."""
    from huddleroom.workers import orchestration_tasks
    from huddleroom.workers import scheduler as worker_scheduler

    runtime_settings = SimpleNamespace(
        orchestration_reconcile_interval_seconds=17,
        orchestration_event_coalesce_seconds=1,
        orchestration_semantic_progress_seconds=19,
        orchestration_sweep_goal_limit=23,
        orchestration_sweep_seconds_limit=29,
    )
    monkeypatch.setattr(worker_scheduler, "settings", runtime_settings, raising=False)
    sched = worker_scheduler.create_scheduler()
    try:
        assert sched.get_job("reconcile_orchestration_runs").trigger.interval.total_seconds() == 17
        assert sched.get_job("supervise_orchestration").trigger.interval.total_seconds() == 17
    finally:
        worker_scheduler.scheduler = None

    calls = []

    class FakeSupervisor:
        async def evaluate_due(self, _db, *, goal_limit=100, **_kwargs):
            calls.append(("due", goal_limit))
            return 0

        async def sweep(self, _db, *, goal_limit=100, timeout_seconds=5, collect_due=False, **_kwargs):
            calls.append(("sweep", goal_limit, timeout_seconds))
            assert collect_due is True
            return []

    @asynccontextmanager
    async def session_context():
        class FakeDb:
            async def commit(self):
                return None

            async def rollback(self):
                return None

        yield FakeDb()

    monkeypatch.setattr(orchestration_tasks, "settings", runtime_settings)
    monkeypatch.setattr(orchestration_tasks, "AsyncSessionLocal", session_context)
    monkeypatch.setattr(orchestration_tasks, "OrchestrationSupervisionScheduler", FakeSupervisor)

    await orchestration_tasks.supervise_orchestration_async()
    assert calls == [("sweep", 23, 29)]


async def test_durable_waits_use_the_configured_recheck_interval(
    db_session, test_project, test_agent, monkeypatch,
):
    """The wait timeout must follow Settings rather than a hidden literal."""
    from huddleroom.models.base import _utcnow
    from huddleroom.models.session import Session
    from huddleroom.services import orchestration_supervision as supervision_module

    goal, run = await _run(db_session, test_project)
    task = await _task_for_run(db_session, test_project, run)
    session = Session(
        project_id=test_project.id, task_id=task.id, agent_id=test_agent.id,
        adapter_type="api", status="running",
    )
    db_session.add(session)
    await db_session.flush()
    monkeypatch.setattr(
        supervision_module, "settings", SimpleNamespace(orchestration_reconcile_interval_seconds=17),
    )

    before = _utcnow()
    await OrchestrationService().supervision.reconcile_local(db_session, goal, run)
    wait = await db_session.scalar(__import__("sqlalchemy").select(OrchestrationWait))
    deadline = (
        wait.due_recheck_at.replace(tzinfo=timezone.utc)
        if wait.due_recheck_at.tzinfo is None else wait.due_recheck_at.astimezone(timezone.utc)
    )

    assert timedelta(seconds=16) <= deadline - before <= timedelta(seconds=18)


def _provider_assessment(reason="Provider reviewed the durable state"):
    return SupervisionAssessment(
        changes=({"kind": "review", "detail": "durable state inspected"},),
        risks=(),
        useful_learning=(),
        criterion_progress=(),
        disposition={
            "action_type": "continue",
            "origin": "provider",
            "reason": reason,
            "expected_result": "The current durable work can continue.",
            "contract_version": "start",
        },
    )


async def test_due_judgment_calls_provider_and_applies_its_valid_disposition(db_session, test_project, monkeypatch):
    """Removing the provider pass or bypassing its ledger disposition must fail here."""
    _goal, run = await _run(db_session, test_project)
    observed = []

    class Analyzer:
        async def assess(self, payload):
            observed.append(payload)
            return _provider_assessment()

    now = datetime(2026, 9, 10, 13, 0, tzinfo=timezone.utc)
    run.supervision_state = {"needs_judgment": True, "judgment_dirty": True, "judgment_due_at": now.isoformat()}
    scheduler = OrchestrationSupervisionScheduler(judge=Analyzer().assess)
    sessions = await _fresh_sessions(db_session, monkeypatch)

    async with sessions() as worker:
        assert await scheduler.evaluate_run(worker, run.id, now=now) == 1
    async with sessions() as observer:
        action = await observer.scalar(__import__("sqlalchemy").select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id
        ))
        run = await observer.get(OrchestrationRun, run.id)
    assert observed and "memory" in observed[0]["context"]
    assert action.action_type == "noop"
    assert action.dispatch_contract["origin"] == "provider"
    assert run.supervision_state.get("judgment_in_flight") is False


async def test_provider_payload_recursively_bounds_outer_and_context_strings(db_session, test_project, monkeypatch):
    """Transport bounds apply to every provider payload string, not only context fields."""
    goal, run = await _run(db_session, test_project)
    goal.objective = "x" * 4_001
    now = datetime(2026, 9, 10, 13, 0, 30, tzinfo=timezone.utc)
    run.supervision_state = {"needs_judgment": True, "judgment_dirty": True, "judgment_due_at": now.isoformat()}
    observed = []

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)

    class Analyzer:
        async def assess(self, payload):
            observed.extend(strings(payload))
            return _provider_assessment()

    sessions = await _fresh_sessions(db_session, monkeypatch)
    async with sessions() as worker:
        await OrchestrationSupervisionScheduler(judge=Analyzer().assess).evaluate_run(worker, run.id, now=now)

    assert observed
    assert max(map(len, observed)) <= 4_000


async def test_provider_claim_fingerprints_the_unbounded_outer_payload(db_session, test_project, monkeypatch):
    """The claim fence covers contract and wrapper fields before transport truncation."""
    goal, run = await _run(db_session, test_project)
    goal.objective = "x" * 4_001
    now = datetime(2026, 9, 10, 13, 0, 31, tzinfo=timezone.utc)
    run.supervision_state = {"needs_judgment": True, "judgment_due_at": now.isoformat()}
    sessions = await _fresh_sessions(db_session, monkeypatch)

    claim = await OrchestrationSupervisionScheduler()._claim_judgment(run.id, now)  # pylint: disable=protected-access
    assert claim is not None
    async with sessions() as observer:
        persisted = await observer.get(OrchestrationRun, run.id)
        canonical = await OrchestrationSupervisionContextBuilder().build(observer, goal, persisted, body_limit=None)
    outer = {
        "goal": canonical["goal"], "run_id": canonical["run"]["id"],
        "contract_version": claim["payload"]["contract_version"], "context": canonical,
    }
    assert claim["fingerprint"] == OrchestrationSupervisionContextBuilder.fingerprint_snapshot(outer)
    assert claim["fingerprint"] != OrchestrationSupervisionContextBuilder.fingerprint_snapshot(claim["payload"])


async def test_sweep_returns_due_judgment_even_after_dirty_bit_was_cleared(db_session, test_project, monkeypatch):
    """A persisted retry deadline, rather than dirty, is the sweep's work predicate."""
    _goal, run = await _run(db_session, test_project)
    now = datetime(2026, 9, 10, 13, 0, 32, tzinfo=timezone.utc)
    run.supervision_state = {
        "needs_judgment": True, "judgment_dirty": False, "judgment_due_at": now.isoformat(),
    }
    sessions = await _fresh_sessions(db_session, monkeypatch)

    async def local_tick(*_args, **_kwargs):
        return None

    async with sessions() as worker:
        assert await OrchestrationSupervisionScheduler(tick=local_tick).sweep(
            worker, timeout_seconds=2, goal_limit=1, now=now, collect_due=True,
        ) == [run.id]


async def test_sweep_singleton_creation_is_safe_when_two_workers_claim_first(db_session, monkeypatch):
    """The singleton bootstrap must not turn a duplicate beat into an IntegrityError."""
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from huddleroom import database as database_module

    sessions = async_sessionmaker(db_session.bind, expire_on_commit=False)
    await db_session.commit()
    monkeypatch.setattr(database_module, "AsyncSessionLocal", sessions)
    now = datetime(2026, 9, 10, 13, 0, 33, tzinfo=timezone.utc)
    first, second = await asyncio.gather(
        OrchestrationSupervisionScheduler()._claim_sweep_pass(now),  # pylint: disable=protected-access
        OrchestrationSupervisionScheduler()._claim_sweep_pass(now),  # pylint: disable=protected-access
    )
    assert sorted((first[0], second[0])) == [False, True]


@pytest.mark.parametrize("singleton_exists", (True, False))
async def test_sweep_claim_defers_for_a_held_sqlite_writer_then_claims_a_fresh_slot(
    test_engine, monkeypatch, singleton_exists,
):
    """A competing SQLite writer defers this pass without poisoning the next one."""
    from sqlalchemy.ext.asyncio import async_sessionmaker
    import huddleroom.database as database_module

    sessions = async_sessionmaker(test_engine, expire_on_commit=False)

    @asynccontextmanager
    async def short_timeout_session():
        async with test_engine.connect() as connection:
            original_timeout = await connection.scalar(text("PRAGMA busy_timeout"))
            await connection.commit()
            try:
                await connection.execute(text("PRAGMA busy_timeout = 50"))
                await connection.commit()
                async with async_sessionmaker(bind=connection, expire_on_commit=False)() as session:
                    yield session
            finally:
                await connection.execute(text(f"PRAGMA busy_timeout = {original_timeout}"))
                await connection.commit()

    monkeypatch.setattr(database_module, "AsyncSessionLocal", short_timeout_session)
    if singleton_exists:
        async with sessions() as setup:
            setup.add(OrchestrationSchedulerState(name="supervision"))
            await setup.commit()

    writer = sessions()
    try:
        await writer.execute(text("BEGIN IMMEDIATE"))
        now = datetime(2026, 9, 10, 13, 0, tzinfo=timezone.utc)
        scheduler = OrchestrationSupervisionScheduler()
        assert await scheduler.sweep(None, timeout_seconds=1, goal_limit=1, now=now) == 0
    finally:
        await writer.rollback()
        await writer.close()

    next_slot = now + timedelta(seconds=settings.orchestration_reconcile_interval_seconds)
    assert await scheduler._claim_sweep_pass(next_slot) == (True, None)  # pylint: disable=protected-access


@pytest.mark.parametrize(
    ("dialect_name", "message"),
    (("sqlite", "connection lost"), ("postgresql", "database is locked")),
)
async def test_sweep_claim_reraises_non_lock_operational_errors(monkeypatch, dialect_name, message):
    """Only SQLite lock contention is a deferred scheduler pass."""
    import huddleroom.database as database_module

    class BrokenSession:
        async def get(self, *_args):
            raise OperationalError("SELECT", {}, RuntimeError(message))

        def get_bind(self):
            return SimpleNamespace(dialect=SimpleNamespace(name=dialect_name))

    @asynccontextmanager
    async def broken_session():
        yield BrokenSession()

    monkeypatch.setattr(database_module, "AsyncSessionLocal", broken_session)
    with pytest.raises(OperationalError, match=message):
        await OrchestrationSupervisionScheduler()._claim_sweep_pass(  # pylint: disable=protected-access
            datetime(2026, 9, 10, 13, 0, tzinfo=timezone.utc)
        )


async def test_concurrent_due_evaluations_issue_one_provider_call_after_persisting_claim(  # pylint: disable=too-many-locals
    test_engine, tmp_path, monkeypatch,
):
    """Dropping the durable claim lets two schedulers send the same provider request."""
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from huddleroom.models.project import Project

    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    import huddleroom.database as database_module
    import huddleroom.services.orchestration_service as service_module
    monkeypatch.setattr(database_module, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(service_module, "AsyncSessionLocal", sessions)
    async with sessions() as setup:
        project = Project(name=f"Atomic judgment {uuid.uuid4()}", workspace_path=str(tmp_path), config={})
        setup.add(project)
        await setup.flush()
        _goal, run = await _run(setup, project)
        run_id = run.id
        now = datetime(2026, 9, 10, 13, 1, tzinfo=timezone.utc)
        run.supervision_state = {"needs_judgment": True, "judgment_dirty": True, "judgment_due_at": now.isoformat()}
        await setup.commit()

    started, release = asyncio.Event(), asyncio.Event()
    claimed_states, calls = [], 0

    class Analyzer:
        async def assess(self, _payload):
            nonlocal calls
            calls += 1
            async with sessions() as observer:
                claimed = await observer.get(OrchestrationRun, run_id)
                claimed_states.append(dict(claimed.supervision_state))
            started.set()
            await release.wait()
            return _provider_assessment()

    scheduler = OrchestrationSupervisionScheduler(judge=Analyzer().assess)
    async with sessions() as first_db, sessions() as second_db:
        first = asyncio.create_task(scheduler.evaluate_run(first_db, run_id, now=now))
        await asyncio.wait_for(started.wait(), timeout=1)
        second = await scheduler.evaluate_run(second_db, run_id, now=now)
        release.set()
        assert await first == 1

    assert second == 0
    assert calls == 1
    assert len(claimed_states) == 1
    assert claimed_states[0]["judgment_dirty"] is False
    assert claimed_states[0]["judgment_in_flight"] is True


async def test_event_during_provider_judgment_stays_dirty_for_one_fresh_follow_up(  # pylint: disable=too-many-locals
    db_session, test_project, monkeypatch,
):
    """Clearing dirty unconditionally loses an event received while the provider was running."""
    _goal, run = await _run(db_session, test_project)
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0

    class Analyzer:
        async def assess(self, _payload):
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                await release.wait()
            return _provider_assessment(reason=f"Provider pass {calls}")

    now = datetime(2026, 9, 10, 13, 2, tzinfo=timezone.utc)
    run.supervision_state = {"needs_judgment": True, "judgment_dirty": True, "judgment_due_at": now.isoformat()}
    scheduler = OrchestrationSupervisionScheduler(judge=Analyzer().assess)
    sessions = await _fresh_sessions(db_session, monkeypatch)
    async with sessions() as first_db:
        first = asyncio.create_task(scheduler.evaluate_run(first_db, run.id, now=now))
        await asyncio.wait_for(started.wait(), timeout=1)
        async with sessions() as event_db:
            await scheduler.record_event(event_db, "task.status_changed", {"run_id": str(run.id)}, now=now)
            await event_db.commit()
        release.set()
        assert await first == 0

    async with sessions() as observer:
        run = await observer.get(OrchestrationRun, run.id)
        assert run.supervision_state["judgment_dirty"] is True
    async with sessions() as follow_up:
        assert await scheduler.evaluate_run(
            follow_up, run.id, now=now + timedelta(seconds=settings.orchestration_event_coalesce_seconds)
        ) == 1
    assert calls == 2
    async with sessions() as observer:
        run = await observer.get(OrchestrationRun, run.id)
        assert run.supervision_state["judgment_dirty"] is False


async def test_provider_disposition_is_fenced_off_when_its_context_changes(
    db_session, test_project, monkeypatch,
):
    """Applying a result against an outdated durable context can create an invalid action."""
    goal, run = await _run(db_session, test_project)
    started, release = asyncio.Event(), asyncio.Event()

    class Analyzer:
        async def assess(self, _payload):
            started.set()
            await release.wait()
            return _provider_assessment()

    now = datetime(2026, 9, 10, 13, 3, tzinfo=timezone.utc)
    run.supervision_state = {"needs_judgment": True, "judgment_dirty": True, "judgment_due_at": now.isoformat()}
    scheduler = OrchestrationSupervisionScheduler(judge=Analyzer().assess)
    sessions = await _fresh_sessions(db_session, monkeypatch)
    async with sessions() as worker:
        pending = asyncio.create_task(scheduler.evaluate_run(worker, run.id, now=now))
        await asyncio.wait_for(started.wait(), timeout=1)
        async with sessions() as writer:
            writer.add(OrchestrationMemorySection(
                project_id=test_project.id, goal_id=goal.id, section_key="context-fence",
                title="Changed while provider ran", body="This must invalidate the answer.",
                created_by="orchestrator", fact_status="unverified", provenance={},
            ))
            await writer.commit()
        release.set()
        assert await pending == 0
    async with sessions() as observer:
        assert await observer.scalar(__import__("sqlalchemy").select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id
        )) is None


async def test_fair_sweep_uses_a_fresh_session_per_goal_after_an_error(  # pylint: disable=too-many-locals
    test_engine, tmp_path, monkeypatch,
):
    """Reusing the failed goal's session can poison the next goal and strand the cursor."""
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from huddleroom.models.project import Project
    import huddleroom.database as database_module
    from huddleroom.services.orchestration_service import OrchestrationService

    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    async with sessions() as setup:
        project = Project(name=f"Sweep isolation {uuid.uuid4()}", workspace_path=str(tmp_path), config={})
        setup.add(project)
        await setup.flush()
        first_goal = OrchestrationGoal(
            id=uuid.UUID(int=1), project_id=project.id, objective="First fair sweep goal", status="active"
        )
        second_goal = OrchestrationGoal(
            id=uuid.UUID(int=2), project_id=project.id, objective="Second fair sweep goal", status="active"
        )
        setup.add_all([first_goal, second_goal])
        await setup.flush()
        first_run = OrchestrationRun(goal_id=first_goal.id, status="running", phase="authorized")
        second_run = OrchestrationRun(goal_id=second_goal.id, status="running", phase="authorized")
        setup.add_all([first_run, second_run])
        await setup.flush()
        state = await setup.get(OrchestrationSchedulerState, "supervision")
        if state is not None:
            state.cursor_goal_id = state.pass_key = None
        await setup.commit()

    opened, attempted = [], []

    @asynccontextmanager
    async def fresh_session():
        async with sessions() as session:
            opened.append(session)
            yield session

    async def tick(_service, session, run_id, *, local_only=False):
        attempted.append((session, run_id, local_only))
        if run_id == first_run.id:
            raise RuntimeError("first goal failed after claiming its session")
        return {"outcome": "continue"}

    monkeypatch.setattr(database_module, "AsyncSessionLocal", fresh_session)
    monkeypatch.setattr(OrchestrationService, "tick", tick)
    scheduler = OrchestrationSupervisionScheduler()
    now = datetime(2026, 9, 10, 13, 4, tzinfo=timezone.utc)
    async with sessions() as control:
        assert await scheduler.sweep(control, timeout_seconds=2, goal_limit=2, now=now) == 2
        await control.commit()

    assert [run_id for _session, run_id, _local_only in attempted] == [first_run.id, second_run.id]
    assert all(local_only is True for _session, _run_id, local_only in attempted)
    assert len(opened) >= len(attempted) == 2
    assert len({id(session) for session, _run_id, _local_only in attempted}) == 2

    async with sessions() as observer:
        state = await observer.get(OrchestrationSchedulerState, "supervision")
        assert state.cursor_goal_id == max(first_goal.id, second_goal.id)
