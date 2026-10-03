import pytest
from sqlalchemy import select

from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.services.orchestration_continuous_service import OrchestrationContinuousService
from huddleroom.workers.scheduler import (
    reconcile_continuous_goals_async,
    reconcile_orchestration_runs_async,
    reconcile_orchestration_runs_job,
)
from tests.test_orchestration_continuous_service import started_continuous


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
    roadmap, goal, parent, _, row = await release(db_session, test_project)
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
