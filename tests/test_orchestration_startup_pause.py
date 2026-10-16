"""Startup fences orchestration before recovery or scheduling can dispatch work."""
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.services.orchestration_service import OrchestrationService


@pytest.mark.asyncio
async def test_startup_pause_is_durable_preserves_work_and_allows_resume(db_session, test_engine, test_project):
    rows = []
    for goal_status in ("active", "blocked", "paused", "completed", "cancelled"):
        for run_status in ("running", "blocked", "paused", "completed", "cancelled"):
            goal = OrchestrationGoal(project_id=test_project.id, objective="Keep", original_request="Keep",
                                     status=goal_status, budget={"max_turns": 8})
            db_session.add(goal)
            await db_session.flush()
            phase = "baseline" if goal_status == "blocked" else "authorized"
            run = OrchestrationRun(goal_id=goal.id, status=run_status, phase=phase, event_cursor=42,
                                   plan_state={"draft": "keep"}, budget_state={"used": 3})
            db_session.add(run)
            await db_session.flush()
            rows.append((goal.id, run.id, goal_status, run_status))
    baseline = OrchestrationRun(goal_id=rows[0][0], status="completed", phase="baseline", baseline_authorized=True)
    db_session.add(baseline)
    await db_session.commit()
    service = OrchestrationService()
    async with AsyncSession(test_engine) as startup:
        async with startup.begin():
            await service.pause_existing_goals_for_startup(startup)
    async with AsyncSession(test_engine) as restarted:
        for goal_id, run_id, old_goal, old_run in rows:
            goal = await restarted.get(OrchestrationGoal, goal_id)
            run = await restarted.get(OrchestrationRun, run_id)
            assert goal.status == ("paused" if old_goal in ("active", "blocked") else old_goal)
            assert run.status == ("paused" if old_goal in ("active", "blocked", "paused") and old_run in ("running", "blocked") else old_run)
            # SPR230/SPR231: startup commits the scalar fence before recovery.
            fenced = old_goal in ("active", "blocked", "paused") and old_run in ("running", "blocked", "paused")
            assert run.pause_requested is fenced
            assert run.control_version == int(fenced)
            assert goal.budget == {"max_turns": 8}
            assert (run.phase, run.event_cursor, run.plan_state, run.budget_state) == (
                "baseline" if old_goal == "blocked" else "authorized", 42, {"draft": "keep"}, {"used": 3})
            assert run.baseline_authorized is not (old_goal == "blocked" and old_run in ("running", "blocked"))
        assert (await restarted.get(OrchestrationRun, baseline.id)).baseline_authorized is True
        await service.pause_existing_goals_for_startup(restarted)
        await restarted.commit()
        goal_id, run_id, _, _ = rows[0]
        goal, run = await service.resume_goal(restarted, test_project.id, goal_id)
        assert (goal.status, run.status, run.id) == ("active", "running", run_id)
        await restarted.commit()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "pause", "commit"])
async def test_lifespan_commits_pause_before_recovery_and_fails_closed(monkeypatch, failure):
    from huddleroom import main
    calls = []

    class StopAtRecovery(BaseException):
        pass

    class StartupDb:
        def __init__(self, *_args, **_kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, *_args):
            if exc_type is None:
                calls.append("commit")
                if failure == "commit":
                    raise RuntimeError("commit failed")
            return False
        def begin(self):
            return self

    async def init_db():
        calls.append("init")
    async def pause(_self, _db):
        calls.append("pause")
        if failure == "pause":
            raise RuntimeError("pause failed")
    async def recover():
        calls.append("recover")
        raise StopAtRecovery()

    monkeypatch.setattr(main, "init_db", init_db)
    monkeypatch.setattr(main, "register_litellm_debug_logger", lambda: None)
    monkeypatch.setattr("huddleroom.database.engine", object())
    monkeypatch.setattr("sqlalchemy.ext.asyncio.AsyncSession", StartupDb)
    monkeypatch.setattr(OrchestrationService, "pause_existing_goals_for_startup", pause, raising=False)
    monkeypatch.setattr(main.orchestration_goals.conversation_service, "recover_all", recover)
    with pytest.raises(RuntimeError if failure else StopAtRecovery):
        async with main.lifespan(None):
            pytest.fail("startup must not reach consumers")
    assert calls == (["init", "pause"] if failure == "pause" else
                     ["init", "pause", "commit"] + ([] if failure else ["commit", "recover"]))
