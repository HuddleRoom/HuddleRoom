"""Goal cancel with a live SQLite CLI session must not hit cli_cancel_transaction_conflict."""
import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from huddleroom.models.agent import Agent
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.models.project import Project
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.orchestration_baseline_runtime import control_operation_owner
from huddleroom.services.orchestration_service import OrchestrationService

RUNNER_ID = "runner-cli-1"


@pytest.fixture
def factory(test_engine, monkeypatch):
    f = async_sessionmaker(test_engine, expire_on_commit=False)
    monkeypatch.setattr("huddleroom.database.AsyncSessionLocal", f)
    monkeypatch.setattr("huddleroom.services.orchestration_service.AsyncSessionLocal", f)
    return f


@pytest.fixture
def cancel_calls(monkeypatch, test_engine):
    calls = []

    async def fake_cancel_task(runner_id):
        calls.append(runner_id)
        # Stand-in for the worker, the sole writer of cancellation facts.
        async with async_sessionmaker(test_engine, expire_on_commit=False)() as w:
            s = await w.scalar(select(Session).where(Session.runner_task_id == runner_id))
            s.status = "cancelled"
            from datetime import datetime, timezone
            s.ended_at = datetime.now(timezone.utc)
            await w.commit()
        return True

    monkeypatch.setattr("huddleroom.workers.task_runner.cancel_task", fake_cancel_task)
    return calls


async def _arrange(factory, tmp_path, *, child=False):
    async with factory() as db:
        project = Project(name=f"cli-cancel-{uuid.uuid4()}", workspace_path=str(tmp_path), config={})
        db.add(project)
        await db.flush()
        goal = OrchestrationGoal(
            project_id=project.id, objective="o", success_criteria=[], constraints={}, budget={},
            weight="standard",
        )
        db.add(goal)
        await db.flush()
        if child:
            # The session lives on a child goal; the parent has no work of its own.
            db.add(OrchestrationRun(goal_id=goal.id))
            parent_id = goal.id
            goal = OrchestrationGoal(
                project_id=project.id, objective="c", success_criteria=[], constraints={}, budget={},
                weight="standard", parent_goal_id=parent_id, continuous_origin_key="k",
                parent_contract_snapshot={}, goal_delta={},
            )
            db.add(goal)
            await db.flush()
        # Significant work makes close_cancelled write process/memory rows before the session cancel.
        run = OrchestrationRun(goal_id=goal.id, plan_state={"expanded_items": [{"key": "x"}]})
        db.add(run)
        await db.flush()
        task = Task(
            project_id=project.id, title="t", status="in_progress",
            metadata_={"orchestration": {"run_id": str(run.id)}},
        )
        db.add(task)
        await db.flush()
        agent = Agent(name=f"a-{uuid.uuid4()}", role="developer", provider="anthropic", model="m",
                      adapter_type="cli", capabilities=[], config={})
        db.add(agent)
        await db.flush()
        sess = Session(
            project_id=project.id, task_id=task.id, agent_id=agent.id, adapter_type="cli", status="running",
            runner_task_id=RUNNER_ID, metadata_={}, origin="auto",
        )
        db.add(sess)
        await db.commit()
        return project.id, (parent_id if child else goal.id), goal.id, sess.id


async def _statuses(factory, goal_id, sess_id):
    async with factory() as db:
        goal = await db.get(OrchestrationGoal, goal_id)
        sess = await db.get(Session, sess_id)
        return goal.status, sess.status


@pytest.mark.asyncio
async def test_service_cancel_goal_with_running_cli_session(factory, cancel_calls, tmp_path):
    project_id, goal_id, _, sess_id = await _arrange(factory, tmp_path)
    async with factory() as db, control_operation_owner(db):
        await OrchestrationService().cancel_goal(db, project_id, goal_id)
        await db.commit()
    assert await _statuses(factory, goal_id, sess_id) == ("cancelled", "cancelled")
    assert cancel_calls == [RUNNER_ID]


@pytest.mark.asyncio
async def test_http_cancel_goal_with_running_cli_session(factory, cancel_calls, tmp_path):
    from huddleroom.dependencies import _ANON_USER, get_current_user
    from huddleroom.main import create_app

    project_id, goal_id, _, sess_id = await _arrange(factory, tmp_path)
    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: _ANON_USER  # real get_db stays in place
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(f"/api/v1/projects/{project_id}/orchestration/goals/{goal_id}/cancel")
    assert r.status_code == 200, r.text
    assert await _statuses(factory, goal_id, sess_id) == ("cancelled", "cancelled")
    assert cancel_calls == [RUNNER_ID]


@pytest.mark.asyncio
async def test_cancel_parent_goal_cancels_child_cli_session(factory, cancel_calls, tmp_path):
    project_id, parent_id, child_id, sess_id = await _arrange(factory, tmp_path, child=True)
    async with factory() as db, control_operation_owner(db):
        await OrchestrationService().cancel_goal(db, project_id, parent_id)
        await db.commit()
    assert (await _statuses(factory, child_id, sess_id))[1] == "cancelled"
    assert cancel_calls == [RUNNER_ID]
    # The goals FK to the parent is restrictive; drop the child so the autouse cleanup can delete goals.
    async with factory() as db:
        await db.execute(delete(OrchestrationRun).where(OrchestrationRun.goal_id == child_id))
        await db.execute(delete(OrchestrationGoal).where(OrchestrationGoal.id == child_id))
        await db.commit()
