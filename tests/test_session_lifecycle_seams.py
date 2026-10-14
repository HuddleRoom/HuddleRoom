"""Session lifecycle seams: manual vs orchestrated dispatch, resume and cancel projection.

Sessions run through the real worker entry (execute_api_session / execute_cli_session);
only the provider (`litellm.acompletion`) and the CLI subprocess are faked.
"""
from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from huddleroom.models.agent import Agent
from huddleroom.models.event_log import EventLog
from huddleroom.models.project import Project
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.session_service import SessionService
from huddleroom.workers import session_tasks


@asynccontextmanager
async def _world(test_engine, tmp_path, monkeypatch, *, adapter="api", session_status="pending",
                 task_status="in_progress", resumable=False, runner="runner-1", link_task=True):
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(session_tasks, "AsyncSessionLocal", factory)
    async with factory() as db:
        project = Project(name=f"Project-{uuid.uuid4()}", workspace_path=str(tmp_path), config={})
        agent = Agent(
            name=f"agent-{uuid.uuid4()}", role="developer", provider="openai", model="gpt-4o-mini",
            adapter_type=adapter, capabilities=[], config={},
            **({"cli_runtime": "claude_code"} if adapter == "cli" else {}),
        )
        db.add_all([project, agent])
        await db.flush()
        task = Task(project_id=project.id, title="work", description="work", status=task_status,
                    assigned_to=agent.id) if link_task else None
        if task is not None:
            db.add(task)
            await db.flush()
        session = Session(
            project_id=project.id, agent_id=agent.id, task_id=task.id if task else None, adapter_type=adapter,
            status=session_status, runner_task_id=runner, resumable=resumable, metadata_={},
        )
        db.add(session)
        await db.commit()
        yield factory, project, agent, task, session


async def _reload(factory, model, pk):
    async with factory() as db:
        return await db.get(model, pk)


def _fake_completion():
    async def acompletion(**_kwargs):
        msg = SimpleNamespace(content="all done", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2))
    return acompletion


@pytest.mark.asyncio
async def test_manual_api_session_completes_without_orchestrator_claim(test_engine, tmp_path, monkeypatch):
    """A session no orchestrator owns must not be cancelled for a claim it never had."""
    async with _world(test_engine, tmp_path, monkeypatch) as (factory, _project, _agent, task, session):
        with patch("litellm.acompletion", new=_fake_completion()):
            await session_tasks.execute_api_session(str(session.id), "runner-1")
        stored = await _reload(factory, Session, session.id)
        assert stored.status == "completed", stored.error
        assert stored.ended_at is not None
        assert (await _reload(factory, Task, task.id)).status == "done"


@pytest.mark.asyncio
async def test_owned_api_session_still_loses_claim(test_engine, tmp_path, monkeypatch):
    from tests.test_orchestration_worker_ownership_recovery import _owned

    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(session_tasks, "AsyncSessionLocal", factory)
    async with factory() as db:
        project = Project(name=f"Project-{uuid.uuid4()}", workspace_path=str(tmp_path), config={})
        agent = Agent(name=f"agent-{uuid.uuid4()}", role="developer", provider="openai", model="gpt-4o-mini",
                      adapter_type="api", capabilities=[], config={})
        db.add_all([project, agent])
        await db.flush()
        _goal, _run, _action, _task, session = await _owned(
            db, project, agent, session_status="pending", task_status="in_progress", runner="runner-1")
        session.adapter_type = "api"
        await db.commit()
        session_id = session.id

    completion = AsyncMock()
    with patch("litellm.acompletion", new=completion), \
            patch.object(session_tasks, "mark_attempt_effect_started", new=AsyncMock(return_value=False)):
        with pytest.raises(asyncio.CancelledError, match="claim lost"):
            await session_tasks.execute_api_session(str(session_id), "runner-1")
    completion.assert_not_awaited()


class _OkProc:
    returncode = 0
    pid = 1

    async def communicate(self):
        return b'{"result": "done", "session_id": "sess-1"}', b""


@pytest.mark.asyncio
async def test_resume_of_failed_session_moves_task_back_to_in_progress(test_engine, tmp_path, monkeypatch):
    async with _world(test_engine, tmp_path, monkeypatch, adapter="cli", session_status="failed",
                      task_status="failed", resumable=True, runner=None) as (factory, project, _agent, task, session):
        async with factory() as db:
            with patch.object(SessionService, "_schedule_dispatch_after_commit"):
                await SessionService().resume(db, session.id)
            await db.commit()
        assert (await _reload(factory, Task, task.id)).status == "in_progress"
        async with factory() as db:
            events = (await db.execute(select(EventLog).where(
                EventLog.project_id == project.id, EventLog.event_type == "task.status_changed"))).scalars().all()
        assert [e.payload["status"] for e in events] == ["in_progress"]
        assert events[0].payload["previous_status"] == "failed"

        resumed = await _reload(factory, Session, session.id)
        runner = resumed.runner_task_id
        from huddleroom.adapters.cli_adapter import CliAdapter
        with patch.object(CliAdapter, "_setup_sandbox", return_value=None), \
                patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec",
                      new=AsyncMock(return_value=_OkProc())):
            await session_tasks.execute_cli_session(str(session.id), runner)
        assert (await _reload(factory, Session, session.id)).status == "completed"
        assert (await _reload(factory, Task, task.id)).status == "done"


@pytest.mark.asyncio
async def test_api_cancel_sets_ended_at_and_syncs_task(test_engine, tmp_path, monkeypatch):
    async with _world(test_engine, tmp_path, monkeypatch, session_status="running", runner=None) as (
            factory, _project, _agent, task, session):
        async with factory() as db:
            await SessionService().cancel(db, session.id)
            await db.commit()
        stored = await _reload(factory, Session, session.id)
        assert stored.status == "cancelled"
        assert stored.ended_at is not None
        assert (await _reload(factory, Task, task.id)).status == "failed"


@pytest.mark.asyncio
async def test_cli_cancel_syncs_linked_task(test_engine, tmp_path, monkeypatch):
    started = asyncio.Event()

    class HangingProc:
        returncode = None
        pid = 99999

        async def communicate(self):
            started.set()
            await asyncio.Event().wait()

        def kill(self):
            pass

        def terminate(self):
            pass

        async def wait(self):
            return -15

    async with _world(test_engine, tmp_path, monkeypatch, adapter="cli") as (factory, _project, _agent, task, session):
        from huddleroom.adapters.cli_adapter import CliAdapter
        with patch.object(CliAdapter, "_setup_sandbox", return_value=None), \
                patch.object(CliAdapter, "_terminate_process_group", new=AsyncMock()), \
                patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec",
                      new=AsyncMock(return_value=HangingProc())):
            worker = asyncio.create_task(session_tasks.execute_cli_session(str(session.id), "runner-1"))
            await asyncio.wait_for(started.wait(), timeout=10)
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker
        stored = await _reload(factory, Session, session.id)
        assert stored.status == "cancelled"
        assert stored.ended_at is not None
        assert (await _reload(factory, Task, task.id)).status == "failed"
