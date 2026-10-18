"""SPR230/SPR231: admitted planner output remains consumable across Pause/reload."""
import asyncio
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from huddleroom.config import settings
from huddleroom.models.artifact import Artifact
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services import orchestration_service
from huddleroom.services.event_bus import emit_event
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.session_service import SessionService
from huddleroom.services.session_sync import sync_task_from_session
from huddleroom.services.task_service import TaskService
from huddleroom.workers.session_tasks import _require_runnable_session, mark_attempt_effect_started
from tests.conftest import bind_plan_author_team
from tests.test_orchestration_roster_mutation import seed_project_user, sessions  # noqa: F401
from tests.test_orchestration_runtime_e2e import _accept_ingested, _agent, _authorized_run, _envelope_for


@pytest.mark.asyncio
async def test_planner_finishes_while_paused_survives_reload_and_consumes_once(
    sessions, test_engine, tmp_path, monkeypatch, stub_decision, safe_effectiveness_review_continue,
):
    """The original, actually claimed planner finishes before Resume; no replacement is needed."""
    monkeypatch.setattr(settings, "orchestration_max_actions_per_tick", 1)
    monkeypatch.setattr("huddleroom.workers.task_runner.register_session", lambda *_args, **_kwargs: None)
    # SPR230: committed decision sessions must come from the same engine as the test sessions.
    monkeypatch.setattr(orchestration_service, "AsyncSessionLocal", async_sessionmaker(test_engine, expire_on_commit=False))
    async with sessions() as setup:
        project, _ = await seed_project_user(setup, tmp_path)
        planner = _agent("paused-planner", ["planning"])
        producer = _agent("retained-plan-producer", ["implementation"])
        setup.add_all([planner, producer])
        await setup.flush()
        service, goal, run = await _authorized_run(setup, project, owns_setup=True)
        await bind_plan_author_team(setup, goal, planner)
        await setup.commit()
        action = await service.execute_request_plan_action(setup, run.id, {
            "action_type": "request_plan", "agent_id": str(planner.id), "work_function": "planning",
            "scope": "Plan one independently verifiable implementation item.",
        }, f"spr230:paused-planner:{run.id}")
        task = await setup.get(Task, action.target_id)
        output = _envelope_for(task)
        await setup.commit()
        _, session_id = await TaskService().run(setup, project.id, task.id)
        session = await setup.get(Session, session_id)
        await setup.commit()
        project_id, goal_id, run_id, task_id = project.id, goal.id, run.id, task.id
        session_id, runner_id = session.id, session.runner_task_id

    entered, release = asyncio.Event(), asyncio.Event()

    async def fake_transport():
        async with sessions() as worker:
            assert await _require_runnable_session(worker, session_id, runner_id)
            assert await mark_attempt_effect_started(worker, session_id, runner_id)
            entered.set()
            await release.wait()
            completed = await SessionService().update_status(worker, session_id, "completed", output=output)
            await emit_event(worker, project_id, "session.completed", {
                "session_id": str(session_id), "task_id": str(task_id),
            })
            await sync_task_from_session(worker, completed)
            await worker.commit()

    pending = asyncio.create_task(fake_transport())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        async with sessions() as pausing:
            await asyncio.wait_for(OrchestrationService().pause_goal(pausing, project_id, goal_id), 5)
        release.set()
        await asyncio.wait_for(pending, 5)
        # Both the submitting/worker sessions are closed before independent readback.
        async with sessions() as reload:
            stored = await reload.get(Session, session_id)
            saved_run = await reload.get(OrchestrationRun, run_id)
            assert stored.status == "completed" and stored.output == output
            assert (await reload.get(Task, task_id)).status == "done"
            assert stored.metadata_["attempt"]["effect_state"] == "started"
            assert saved_run.status == "paused" and saved_run.pause_requested
            assert (await reload.get(OrchestrationGoal, goal_id)).status == "paused"
            assert saved_run.baseline_authorized is False
            await reload.commit()
            assert (await OrchestrationService().tick(reload, run_id))["processed_events"] == 0
            assert saved_run.plan_state["status"] == "requested"
            await reload.commit()

        async with sessions() as resumed:
            await OrchestrationService().resume_goal(resumed, project_id, goal_id)
            await resumed.commit()
        stub_decision(_accept_ingested)
        async with sessions() as consumer:
            consumed = await OrchestrationService().tick(consumer, run_id)
            await consumer.commit()
        stub_decision(lambda _ctx: {"action_type": "noop"})
        async with sessions() as replay:
            await OrchestrationService().tick(replay, run_id)
            await replay.commit()
        async with sessions() as final:
            current = await final.get(OrchestrationRun, run_id)
            assert current.plan_state["status"] == "accepted", str((consumed, current.phase, current.plan_state))
            assert current.plan_state["planning_task_id"] == str(task_id)
            artifact_id = uuid.UUID(current.plan_state["plan_artifact_id"])
            assert (await final.get(Artifact, artifact_id)).linked_task_id == task_id
            actions = list(await final.scalars(select(OrchestrationAction).where(
                OrchestrationAction.run_id == run_id)))
            assert len([a for a in actions if a.action_type == "request_plan"]) == 1
            markers = [a for a in actions if a.action_type == "report_consumed"
                       and a.request.get("task_id") == str(task_id)]
            assert len(markers) == 1
            assert list(await final.scalars(select(Session.id).where(Session.task_id == task_id))) == [session_id]
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
