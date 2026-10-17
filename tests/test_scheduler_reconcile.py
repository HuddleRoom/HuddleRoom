from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.services.orchestration_continuous_service import OrchestrationContinuousService
from huddleroom.services.orchestration_baseline_runtime import control_operation_owner
from huddleroom.workers.scheduler import (
    reconcile_continuous_goals_async,
    reconcile_orchestration_runs_async,
    reconcile_orchestration_runs_job,
)
from tests.test_orchestration_continuous_service import started_continuous


@pytest_asyncio.fixture
async def test_engine(tmp_path):
    """SPR230: per-test engine so the committed roadmap setup used by real native ticks is discarded with the file."""
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import create_async_engine
    from huddleroom.models.base import Base
    from tests.conftest import _seed_anon_actors

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'reconcile-owned-setup.db'}")

    @event.listens_for(engine.sync_engine, "connect")
    def pragmas(connection, _record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await _seed_anon_actors(connection)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(autouse=True)
async def clean_test_database(test_engine):
    # The per-test database is discarded; cyclic committed roadmap rows need no DELETE cleanup.
    yield


# committed-run operations cannot run under conftest's explicit begin() root (SPR230 caller contract)
@pytest_asyncio.fixture
async def db_session(test_engine, monkeypatch):
    # SPR230: no conftest begin() root, so setup can be committed; committed fixtures never launch agents.
    monkeypatch.setattr("huddleroom.workers.task_runner.register_session", lambda *_args, **_kwargs: None)
    async with AsyncSession(test_engine, expire_on_commit=False) as db:
        yield db
        await db.rollback()


def _commit_before_setup_ticks(monkeypatch):
    """Roadmap setup helpers tick a flushed session; publish their preparation first (setup only)."""
    from huddleroom.services.orchestration_service import OrchestrationService
    original_tick = OrchestrationService.tick

    async def setup_tick(service, db, *args, **kwargs):
        await db.flush()
        await db.commit()
        return await original_tick(service, db, *args, **kwargs)

    monkeypatch.setattr(OrchestrationService, "tick", setup_tick)
    return lambda: monkeypatch.setattr(OrchestrationService, "tick", original_tick)


async def make_run(db_session, project, *, goal_status, run_status):
    goal = OrchestrationGoal(
        project_id=project.id, objective="Ship X", original_request="Ship X",
        success_criteria=[], constraints={}, budget={}, status=goal_status,
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id, budget_state={}, status=run_status)
    db_session.add(run)
    await db_session.flush()
    return run


@pytest.mark.asyncio
async def test_reconcile_ticks_runnable_and_skips_paused(db_session, test_project, monkeypatch):
    from huddleroom.services.orchestration_service import OrchestrationService
    ticked: list = []

    async def _fake_tick(self, db, run_id):
        ticked.append(run_id)
        return {"run_id": run_id, "status": "running"}

    monkeypatch.setattr(OrchestrationService, "tick", _fake_tick)

    runnable = await make_run(db_session, test_project, goal_status="active", run_status="running")
    paused = await make_run(db_session, test_project, goal_status="paused", run_status="paused")

    await reconcile_orchestration_runs_async(db_session)

    assert runnable.id in ticked
    assert paused.id not in ticked


@pytest.mark.asyncio
async def test_scheduler_claims_due_continuous_before_ticking_and_skips_waiting_run(
    db_session, test_project, monkeypatch,
):
    from datetime import datetime
    from huddleroom.services.orchestration_service import OrchestrationService

    goal, run = await started_continuous(db_session, test_project)
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    claimed = []
    ticked = []
    real_claim = OrchestrationContinuousService.claim_due_cycle

    async def record_claim(self, db, goal_id, now=None):
        claimed.append(goal_id)
        return await real_claim(self, db, goal_id, now)

    async def record_tick(self, db, run_id):
        ticked.append(run_id)
        return {"run_id": run_id, "status": "running"}

    monkeypatch.setattr(OrchestrationContinuousService, "claim_due_cycle", record_claim)
    monkeypatch.setattr(OrchestrationService, "tick", record_tick)
    assert await reconcile_continuous_goals_async(db_session, due) == 1
    assert claimed == [goal.id]
    assert run.phase == "authorized"
    await reconcile_orchestration_runs_async(db_session)
    assert run.id in ticked


@pytest.mark.asyncio
async def test_reconcile_skips_unclaimed_continuous_waiting_activation(db_session, test_project, monkeypatch):
    from huddleroom.services.orchestration_service import OrchestrationService

    _, run = await started_continuous(db_session, test_project)
    ticked = []

    async def record_tick(_self, _db, run_id):
        ticked.append(run_id)

    monkeypatch.setattr(OrchestrationService, "tick", record_tick)
    await reconcile_orchestration_runs_async(db_session)
    assert run.id not in ticked


@pytest.mark.asyncio
async def test_reconcile_job_claims_continuous_before_ticking(monkeypatch):
    import huddleroom.database
    import huddleroom.workers.scheduler as scheduler

    order = []

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    async def claim(_db):
        order.append("claim")

    async def tick(_db):
        order.append("tick")

    monkeypatch.setattr(huddleroom.database, "AsyncSessionLocal", FakeSession)
    monkeypatch.setattr(scheduler, "reconcile_continuous_goals_async", claim)
    monkeypatch.setattr(scheduler, "reconcile_orchestration_runs_async", tick)
    await reconcile_orchestration_runs_job()
    assert order == ["claim", "tick"]


@pytest.mark.asyncio
async def test_reconcile_ticks_authorized_roadmap_parent_and_child_once(db_session, test_project, monkeypatch):
    from sqlalchemy import func
    from huddleroom.models.orchestration import OrchestrationAction, OrchestrationBudgetReservation, OrchestrationRoadmapItem
    from huddleroom.services.orchestration_service import OrchestrationService
    from tests.test_orchestration_roadmap_integration import release
    restore_tick = _commit_before_setup_ticks(monkeypatch)
    roadmap, goal, parent, _, row = await release(db_session, test_project)
    restore_tick()
    await db_session.commit()  # SPR230: publish committed roadmap setup before reconcile
    child_goal = await db_session.get(OrchestrationGoal, row.child_goal_id)
    child = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child_goal.id))
    # Scheduler owns selection only.  Keep the real tick/release path while
    # replacing the external decision adapter with a durable no-op decision.
    async def no_external_decision(self, db, run_id, adapter=None):
        run = await db.get(OrchestrationRun, run_id)
        active_goal = await db.get(OrchestrationGoal, run.goal_id)
        context = await self._decision_context(db, active_goal, run)
        return await self.record_validated_decision(
            db, run_id, context, {}, {"action_type": "noop", "reason": "scheduler replay"},
        )

    monkeypatch.setattr(OrchestrationService, "request_llm_decision", no_external_decision)
    real_tick = OrchestrationService.tick
    ticked: list = []

    async def recording_tick(self, db, run_id):
        ticked.append(run_id)
        return await real_tick(self, db, run_id)

    monkeypatch.setattr(OrchestrationService, "tick", recording_tick)
    before = {
        "items": await db_session.scalar(select(func.count()).select_from(OrchestrationRoadmapItem).where(
            OrchestrationRoadmapItem.goal_id == goal.id)),
        "reservations": await db_session.scalar(select(func.count()).select_from(OrchestrationBudgetReservation).where(
            OrchestrationBudgetReservation.parent_goal_id == goal.id)),
        "actions": await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
            OrchestrationAction.run_id.in_([parent.id, child.id]))),
    }
    await reconcile_orchestration_runs_async(db_session)
    assert {parent.id, child.id}.issubset(ticked)
    first = {
        "items": await db_session.scalar(select(func.count()).select_from(OrchestrationRoadmapItem).where(
            OrchestrationRoadmapItem.goal_id == goal.id)),
        "reservations": await db_session.scalar(select(func.count()).select_from(OrchestrationBudgetReservation).where(
            OrchestrationBudgetReservation.parent_goal_id == goal.id)),
        "actions": await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
            OrchestrationAction.run_id.in_([parent.id, child.id]))),
    }
    assert first["items"] == before["items"] and first["reservations"] == before["reservations"]
    ticked.clear()
    await reconcile_orchestration_runs_async(db_session)
    assert {parent.id, child.id}.issubset(ticked)
    second = {
        "items": await db_session.scalar(select(func.count()).select_from(OrchestrationRoadmapItem).where(
            OrchestrationRoadmapItem.goal_id == goal.id)),
        "reservations": await db_session.scalar(select(func.count()).select_from(OrchestrationBudgetReservation).where(
            OrchestrationBudgetReservation.parent_goal_id == goal.id)),
        "actions": await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
            OrchestrationAction.run_id.in_([parent.id, child.id]))),
    }
    assert second == first
    await roadmap.orchestration.pause_goal(db_session, test_project.id, goal.id)
    parent_actions = await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == parent.id))
    ticked.clear()
    await reconcile_orchestration_runs_async(db_session)
    assert parent.status == "paused" and child.id in ticked and parent.id not in ticked
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == parent.id)) == parent_actions


@pytest.mark.asyncio
async def test_reconcile_one_bad_run_does_not_abort_sweep(db_session, test_project, monkeypatch):
    from huddleroom.services.orchestration_service import OrchestrationService
    attempted: list = []
    ticked: list = []

    bad = await make_run(db_session, test_project, goal_status="active", run_status="running")
    good = await make_run(db_session, test_project, goal_status="active", run_status="running")

    async def _fake_tick(self, db, run_id):
        attempted.append(run_id)
        if run_id == bad.id:
            raise RuntimeError("boom")
        ticked.append(run_id)
        return {"run_id": run_id, "status": "running"}

    monkeypatch.setattr(OrchestrationService, "tick", _fake_tick)

    result = await reconcile_orchestration_runs_async(db_session)

    assert bad.id in attempted
    assert good.id in attempted
    assert good.id in ticked
    assert result == 1


@pytest.mark.asyncio
async def test_reconcile_nested_roadmap_one_lineage_per_key(db_session, test_project, monkeypatch, caplog):
    from collections import Counter

    from sqlalchemy import func

    from huddleroom.models.orchestration import (
        OrchestrationAction, OrchestrationBudgetReservation, OrchestrationEvidence, OrchestrationGate,
        OrchestrationRoadmapItem, OrchestrationRoadmapVersion,
    )
    from huddleroom.models.task import Task
    from huddleroom.services.orchestration_service import OrchestrationService
    from tests.test_orchestration_roadmap_e2e import _children_of, _snapshot, nested_roadmap

    async def noop_start(self, db, goal, run):
        return 0

    monkeypatch.setattr(OrchestrationService, "_start_released_tasks", noop_start)
    restore_tick = _commit_before_setup_ticks(monkeypatch)
    n = await nested_roadmap(db_session, test_project, monkeypatch)
    restore_tick()
    await db_session.commit()  # SPR230: publish committed roadmap setup before reconcile

    async def no_external_decision(self, db, run_id, adapter=None):
        run = await db.get(OrchestrationRun, run_id)
        goal = await db.get(OrchestrationGoal, run.goal_id)
        context = await self._decision_context(db, goal, run)
        return await self.record_validated_decision(
            db, run_id, context, {}, {"action_type": "noop", "reason": "scheduler replay"})

    monkeypatch.setattr(OrchestrationService, "request_llm_decision", no_external_decision)
    models = (OrchestrationAction, Task, OrchestrationGate, OrchestrationEvidence, OrchestrationRoadmapItem,
              OrchestrationRoadmapVersion, OrchestrationBudgetReservation, OrchestrationGoal, OrchestrationRun)

    async def assert_one_lineage_per_key():
        items = (await db_session.execute(select(OrchestrationRoadmapItem.goal_id, OrchestrationRoadmapItem.item_key))).all()
        assert set(Counter(items).values()) <= {1}
        reservations = (await db_session.execute(select(
            OrchestrationBudgetReservation.parent_goal_id, OrchestrationBudgetReservation.roadmap_item_id))).all()
        assert set(Counter(reservations).values()) <= {1} and len(reservations) == len(items) - 1  # build is a task
        for goal in (n.parent, n.child):
            runs = list(await db_session.scalars(select(OrchestrationRun).where(OrchestrationRun.goal_id == goal.id)))
            versions = list(await db_session.scalars(select(OrchestrationRoadmapVersion).where(
                OrchestrationRoadmapVersion.goal_id == goal.id)))
            assert len(runs) == len(versions) == 1 and versions[0].run_id == runs[0].id
        for goal_id, key in items:
            releases = await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
                OrchestrationAction.action_type == "release_roadmap_item",
                OrchestrationAction.idempotency_key.like(f"run:%:kind:release_roadmap_item:{key}"),
                OrchestrationAction.run_id == (await db_session.scalar(select(OrchestrationRun.id).where(
                    OrchestrationRun.goal_id == goal_id)))))
            assert releases == 1, key
        for child in [*await _children_of(db_session, n.parent), *await _children_of(db_session, n.child)]:
            child_runs = list(await db_session.scalars(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id)))
            assert len(child_runs) == 1
            assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
                OrchestrationAction.run_id == child_runs[0].id,
                OrchestrationAction.action_type == "authorize_execution")) == 1

    async def sweep():
        """One sweep must tick every runnable run: the scheduler swallows and logs tick failures."""
        runnable = await db_session.scalar(select(func.count()).select_from(OrchestrationRun).join(
            OrchestrationGoal, OrchestrationGoal.id == OrchestrationRun.goal_id).where(
            OrchestrationRun.status.in_(("running", "blocked")), OrchestrationGoal.status.in_(("active", "blocked"))))
        assert runnable >= 2
        # SPR230: the real reconcile job owns the session (reconcile_orchestration_runs_job); a bare supplied session is caller-composed.
        async with control_operation_owner(db_session):
            assert await reconcile_orchestration_runs_async(db_session) == runnable
        assert "reconcile failed" not in caplog.text and not [r for r in caplog.records if r.levelname == "ERROR"]

    assert await _children_of(db_session, n.child) == []
    await sweep()
    assert [g.roadmap_item_key for g in await _children_of(db_session, n.child)] == ["g1"]
    assert [g.roadmap_item_key for g in await _children_of(db_session, n.parent)] == ["area"]
    await assert_one_lineage_per_key()
    # The released child's first tick (sweep 2) is its own work; after that the sweeps are no-ops.
    await sweep()
    await assert_one_lineage_per_key()
    async def row_ids():
        return {model: [row["id"] for row in rows] for model, rows in (await _snapshot(db_session, models)).items()}

    settled = await row_ids()
    await sweep()
    assert await row_ids() == settled
    await assert_one_lineage_per_key()


# --- goal-type correction successors and Continuous rollover (no scheduler change) ---

@pytest.mark.asyncio
async def test_sweep_ticks_correction_successor_never_the_reclassified_run(db_session, test_project, monkeypatch):
    from huddleroom.services.orchestration_service import OrchestrationService
    from tests.test_orchestration_goal_shape import _goal_run
    from tests.test_orchestration_goal_type_correction import _apply, _request

    goal, old = await _goal_run(db_session, test_project, goal_type="outcome")
    request, _ = await _request(db_session, goal, old)
    _, successor = await _apply(db_session, goal, request)
    assert (old.status, old.terminal_reason) == ("cancelled", "reclassified")
    ticked: list = []

    async def record_tick(_self, _db, run_id):
        ticked.append(run_id)
        return {"run_id": run_id, "status": "running"}

    monkeypatch.setattr(OrchestrationService, "tick", record_tick)
    for _ in range(2):
        ticked.clear()
        assert await reconcile_orchestration_runs_async(db_session) == 1
        assert ticked == [successor.id]


@pytest.mark.asyncio
async def test_concurrent_type_corrections_on_committed_sessions_create_one_successor(
    db_session, test_project, test_engine,
):
    import asyncio

    from fastapi import HTTPException
    from sqlalchemy import func
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from huddleroom.services.orchestration_goal_shape import OrchestrationGoalShapeService
    from huddleroom.services.orchestration_service import OrchestrationService
    from tests.test_orchestration_goal_shape import _goal_run
    from tests.test_orchestration_goal_type_correction import _request

    # clean_test_database (autouse) wipes the committed rows after the test.
    goal, old = await _goal_run(db_session, test_project, goal_type="outcome")
    request, _ = await _request(db_session, goal, old)
    project_id, goal_id, old_id = test_project.id, goal.id, old.id
    await db_session.commit()
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    service = OrchestrationService()

    async def apply():
        # Same lock/commit/rollback shape as the apply_goal_type_correction router.
        async with factory() as db, service._lock_goal_for_baseline_transition(db, goal_id):
            try:
                live = await service.get_goal(db, project_id, goal_id)
                _, successor = await OrchestrationGoalShapeService().apply_correction(db, live, request, "human:u1")
                result = successor.id
                await db.commit()
                return result
            except Exception:
                await db.rollback()
                raise

    results = await asyncio.wait_for(asyncio.gather(apply(), apply(), return_exceptions=True), timeout=30)

    successes = [r for r in results if not isinstance(r, BaseException)]
    losers = [r for r in results if isinstance(r, BaseException)]
    assert successes and len(set(successes)) == 1
    for loser in losers:  # the loser either replays the same successor (above) or is stale
        assert isinstance(loser, HTTPException) and loser.status_code == 409
        assert loser.detail["conflict"] == "stale_run"
    async with factory() as reader:
        successor_ids = list(await reader.scalars(select(OrchestrationRun.id).where(
            OrchestrationRun.supersedes_run_id == old_id)))
        assert successor_ids == [successes[0]]
        assert await reader.scalar(select(func.count()).select_from(OrchestrationRun).where(
            OrchestrationRun.goal_id == goal_id)) == 2
        assert (await reader.get(OrchestrationRun, old_id)).terminal_reason == "reclassified"


@pytest.mark.asyncio
async def test_continuous_rollover_reconciles_cycle_one_artifacts_in_cycle_two(
    db_session, test_project, test_agent, monkeypatch,
):
    from datetime import datetime

    from fastapi import HTTPException
    from sqlalchemy import func

    from huddleroom.models.orchestration import OrchestrationBudgetReservation, OrchestrationContinuousCandidate
    from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService, run_lineage
    from huddleroom.services.orchestration_service import OrchestrationService
    from tests.test_orchestration_continuous_service import dispatched_discovery_source

    goal, cycle1, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch)
    task.status = "done"
    session.status = "completed"
    session.output = (
        '{"schema_version":1,"candidates":['
        '{"origin_key":"source:1","objective":"First","source_refs":["record:1"]},'
        '{"origin_key":"source:2","objective":"Second","source_refs":["record:2"]}]}'
    )
    session.metadata_ = {**session.metadata_, "token_count_in": 7, "token_count_out": 0,
                         "token_usage_complete": True, "_roadmap_elapsed_seconds": 0}
    real_reserve = OrchestrationBudgetService.reserve_continuous_child
    calls = []

    async def first_reserves_second_waits(self, *args, **kwargs):
        calls.append(1)
        if len(calls) > 1:
            raise HTTPException(status_code=409, detail="continuous_budget_wait")
        return await real_reserve(self, *args, **kwargs)

    monkeypatch.setattr(OrchestrationBudgetService, "reserve_continuous_child", first_reserves_second_waits)
    assert (await continuous.advance(db_session, goal, cycle1))["outcome"] == "children_created"
    monkeypatch.setattr(OrchestrationBudgetService, "reserve_continuous_child", real_reserve)

    # Cycle 1 leaves a child goal, a pending intake candidate and a settled discovery reservation behind.
    child1 = await db_session.scalar(select(OrchestrationGoal).where(OrchestrationGoal.parent_goal_id == goal.id))
    child1_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child1.id))
    assert len(goal.continuous_state["pending_candidates"]) == 1
    discovery = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.discovery_run_id == cycle1.id))
    child1_reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.child_goal_id == child1.id))
    assert (discovery.status, child1_reservation.status) == ("settled", "active")
    assert discovery.settled_spend["max_tokens"] == "7"
    child1.status = "completed"
    child1_run.status = child1_run.phase = "completed"

    cycle2 = await OrchestrationService().get_active_run_for_goal(db_session, test_project.id, goal.id)
    assert cycle2.id != cycle1.id
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    assert await continuous.claim_due_cycle(db_session, goal.id, due) is cycle2

    # Cycle 2 settled the finished cycle-1 child and drained the cycle-1 backlog.
    assert goal.continuous_state["pending_candidates"] == []
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationContinuousCandidate).where(
        OrchestrationContinuousCandidate.parent_goal_id == goal.id,
        OrchestrationContinuousCandidate.child_goal_id.is_not(None))) == 2
    await db_session.refresh(child1_reservation)
    assert child1_reservation.status == "settled"
    # Cycle-1 spend still counts toward the parent; cycle 2 has no lineage predecessor.
    remaining = await OrchestrationBudgetService().remaining(db_session, goal)
    assert remaining["direct_spend"]["max_tokens"] == "0"
    assert Decimal(remaining["settled_child_spend"]["max_tokens"]) >= 7
    assert [run.id for run in await run_lineage(db_session, cycle2)] == [cycle2.id]


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
            await asyncio.wait_for(reconcile_orchestration_runs_async(caller), 3)
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
