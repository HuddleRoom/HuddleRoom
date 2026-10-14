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


@pytest.mark.asyncio
async def test_cli_reset_watcher_cancel_syncs_linked_task(test_engine, tmp_path, monkeypatch):
    class HangingProc:
        returncode = None
        pid = 99999

        async def communicate(self):
            await asyncio.Event().wait()

    async def resetting(*_args, **_kwargs):
        return "resetting"

    async with _world(test_engine, tmp_path, monkeypatch, adapter="cli", session_status="running") as (
            factory, _project, _agent, task, session):
        from huddleroom.adapters.cli_adapter import CliAdapter
        with patch.object(CliAdapter, "_setup_sandbox", return_value=None), \
                patch.object(CliAdapter, "_watch_project_status", new=resetting), \
                patch.object(CliAdapter, "_terminate_process_group", new=AsyncMock()), \
                patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec",
                      new=AsyncMock(return_value=HangingProc())):
            async with factory() as db:
                await asyncio.wait_for(CliAdapter().run(session.id, db), timeout=10)
                await db.commit()
        stored = await _reload(factory, Session, session.id)
        assert stored.status == "cancelled"
        assert stored.ended_at is not None
        assert (await _reload(factory, Task, task.id)).status == "failed"


async def _set_run_config(factory, session_id, **config):
    async with factory() as db:
        row = await db.get(Session, session_id)
        row.metadata_ = {**(row.metadata_ or {}), "_run_config": config}
        await db.commit()


@pytest.mark.asyncio
async def test_api_deadline_expiry_after_retryable_error_terminalizes_session(test_engine, tmp_path, monkeypatch):
    import time

    import litellm

    from huddleroom.adapters import api_adapter

    async def rate_limited(**_kwargs):
        time.sleep(0.15)  # the claimed deadline passes while the provider call is failing
        raise litellm.RateLimitError("retry", "openai", "gpt-4o-mini")

    async def no_wait_for(awaitable, timeout=None):
        return await awaitable  # lets the retryable error reach the retry handler, not a cancel

    # Only wait_for is swapped (module-local); everything else stays asyncio.
    monkeypatch.setattr(api_adapter, "asyncio", SimpleNamespace(
        **{name: getattr(asyncio, name) for name in ("get_running_loop", "sleep", "CancelledError")},
        wait_for=no_wait_for))

    async with _world(test_engine, tmp_path, monkeypatch) as (factory, _project, _agent, task, session):
        await _set_run_config(factory, session.id, timeout=0.1)
        with patch("litellm.acompletion", new=rate_limited):
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(session_tasks.execute_api_session(str(session.id), "runner-1"), timeout=10)
        stored = await _reload(factory, Session, session.id)
        assert stored.status == "failed"
        assert stored.error == "TimeoutError: claimed session timeout exhausted"
        assert stored.ended_at is not None
        assert stored.metadata_["attempt"]["result_status"] == "failed"
        assert (await _reload(factory, Task, task.id)).status == "failed"


class _StreamProc:
    """CLI subprocess with real streams: emits chunks, then hangs (or fails the reader)."""
    returncode = None
    pid = 99999

    def __init__(self, chunks=()):
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        for chunk in chunks:
            self.stdout.feed_data(chunk)

    async def wait(self):
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_cli_collector_failure_kills_process_group_and_terminalizes(test_engine, tmp_path, monkeypatch):
    from huddleroom.adapters.cli_adapter import CliAdapter

    proc = _StreamProc()
    terminate = AsyncMock()

    async def broken_collector(*_args, **_kwargs):
        raise RuntimeError("collector blew up token=sk-abcdefghijklmnopqrstuvwxyz123456")

    async with _world(test_engine, tmp_path, monkeypatch, adapter="cli") as (factory, _project, _agent, task, session):
        with patch.object(CliAdapter, "_setup_sandbox", return_value=None), \
                patch.object(CliAdapter, "_terminate_process_group", new=terminate), \
                patch("huddleroom.adapters.cli_adapter.collect_cli_process", new=broken_collector), \
                patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec",
                      new=AsyncMock(return_value=proc)):
            with pytest.raises(RuntimeError):
                await asyncio.wait_for(session_tasks.execute_cli_session(str(session.id), "runner-1"), timeout=10)
        terminate.assert_awaited()
        stored = await _reload(factory, Session, session.id)
        assert stored.status == "failed"
        assert stored.error.startswith("RuntimeError: collector blew up")
        assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in stored.error
        assert stored.metadata_["attempt"]["result_status"] == "failed"
        assert (await _reload(factory, Task, task.id)).status == "failed"


@pytest.mark.asyncio
async def test_cli_timeout_keeps_early_provider_id_and_partial_output_for_resume(test_engine, tmp_path, monkeypatch):
    import json

    from huddleroom.adapters.cli_adapter import CliAdapter

    lines = [
        {"type": "system", "subtype": "init", "session_id": "prov-early-1"},
        {"type": "assistant", "session_id": "prov-early-1",
         "message": {"role": "assistant", "content": [{"type": "text", "text": "partial progress"}]}},
    ]
    proc = _StreamProc([(json.dumps(line) + "\n").encode() for line in lines])
    async with _world(test_engine, tmp_path, monkeypatch, adapter="cli") as (factory, _project, _agent, task, session):
        await _set_run_config(factory, session.id, timeout=0.5)
        with patch.object(CliAdapter, "_setup_sandbox", return_value=None), \
                patch.object(CliAdapter, "_terminate_process_group", new=AsyncMock()), \
                patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec",
                      new=AsyncMock(return_value=proc)):
            await asyncio.wait_for(session_tasks.execute_cli_session(str(session.id), "runner-1"), timeout=10)
        stored = await _reload(factory, Session, session.id)
        assert stored.status == "failed" and stored.error == "timeout"
        assert stored.provider_session_id == "prov-early-1"
        assert stored.output == "partial progress"
        assert stored.resumable is True

        async with factory() as db:
            with patch.object(SessionService, "_schedule_dispatch_after_commit"):
                await SessionService().resume(db, session.id)
            await db.commit()
        runner = (await _reload(factory, Session, session.id)).runner_task_id
        spawn = AsyncMock(return_value=_OkProc())
        with patch.object(CliAdapter, "_setup_sandbox", return_value=None), \
                patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", new=spawn):
            await session_tasks.execute_cli_session(str(session.id), runner)
        assert "prov-early-1" in spawn.await_args.args
        resumed = await _reload(factory, Session, session.id)
        assert resumed.status == "completed"
        assert resumed.metadata_["resume_outcome"] == "resumed"


@pytest.mark.asyncio
async def test_cli_timeout_without_provider_id_is_not_resumable(test_engine, tmp_path, monkeypatch):
    from huddleroom.adapters.cli_adapter import CliAdapter

    async with _world(test_engine, tmp_path, monkeypatch, adapter="cli") as (factory, _project, _agent, _task, session):
        await _set_run_config(factory, session.id, timeout=0.3)
        with patch.object(CliAdapter, "_setup_sandbox", return_value=None), \
                patch.object(CliAdapter, "_terminate_process_group", new=AsyncMock()), \
                patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec",
                      new=AsyncMock(return_value=_StreamProc())):
            await asyncio.wait_for(session_tasks.execute_cli_session(str(session.id), "runner-1"), timeout=10)
        stored = await _reload(factory, Session, session.id)
        assert stored.error == "timeout"
        assert stored.provider_session_id is None
        assert stored.resumable is False


_EARLY = [
    b'{"type": "system", "subtype": "init", "session_id": "prov-early-2"}\n',
    b'{"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "so far"}]}}\n',
]


@pytest.mark.asyncio
async def test_cli_provider_id_is_persisted_as_soon_as_seen(test_engine, tmp_path, monkeypatch):
    from huddleroom.adapters.cli_adapter import CliAdapter

    async with _world(test_engine, tmp_path, monkeypatch, adapter="cli") as (factory, _project, _agent, _task, session):
        with patch.object(CliAdapter, "_setup_sandbox", return_value=None), \
                patch.object(CliAdapter, "_terminate_process_group", new=AsyncMock()), \
                patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec",
                      new=AsyncMock(return_value=_StreamProc(_EARLY))):
            worker = asyncio.create_task(session_tasks.execute_cli_session(str(session.id), "runner-1"))
            try:
                for _ in range(100):  # the worker is still running: no timeout, no terminal write yet
                    stored = await _reload(factory, Session, session.id)
                    if stored.provider_session_id:
                        break
                    await asyncio.sleep(0.05)
                assert stored.status == "running"
                assert stored.provider_session_id == "prov-early-2"
            finally:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.asyncio
async def test_cli_collector_failure_keeps_partial_provider_id_and_output(test_engine, tmp_path, monkeypatch):
    from huddleroom.adapters.cli_adapter import CliAdapter

    async def broken_collector(proc, *, partial, **_kwargs):
        partial["stdout"] = b"".join(_EARLY)
        raise RuntimeError("collector blew up")

    async with _world(test_engine, tmp_path, monkeypatch, adapter="cli") as (factory, _project, _agent, _task, session):
        with patch.object(CliAdapter, "_setup_sandbox", return_value=None), \
                patch.object(CliAdapter, "_terminate_process_group", new=AsyncMock()), \
                patch("huddleroom.adapters.cli_adapter.collect_cli_process", new=broken_collector), \
                patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec",
                      new=AsyncMock(return_value=_StreamProc())):
            with pytest.raises(RuntimeError):
                await asyncio.wait_for(session_tasks.execute_cli_session(str(session.id), "runner-1"), timeout=10)
        stored = await _reload(factory, Session, session.id)
        assert stored.status == "failed"
        assert stored.provider_session_id == "prov-early-2"
        assert stored.output == "so far"
        assert stored.resumable is True


@pytest.mark.asyncio
async def test_cli_collector_cleanup_failure_does_not_mask_original_error(test_engine, tmp_path, monkeypatch):
    from huddleroom.adapters.cli_adapter import CliAdapter

    async def broken_collector(*_args, **_kwargs):
        raise RuntimeError("original collector error")

    async with _world(test_engine, tmp_path, monkeypatch, adapter="cli") as (factory, _project, _agent, _task, session):
        with patch.object(CliAdapter, "_setup_sandbox", return_value=None), \
                patch.object(CliAdapter, "_terminate_process_group", new=AsyncMock(side_effect=OSError("kill failed"))), \
                patch("huddleroom.adapters.cli_adapter.collect_cli_process", new=broken_collector), \
                patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec",
                      new=AsyncMock(return_value=_StreamProc())):
            with pytest.raises(RuntimeError, match="original collector error"):
                await asyncio.wait_for(session_tasks.execute_cli_session(str(session.id), "runner-1"), timeout=10)
        stored = await _reload(factory, Session, session.id)
        assert stored.status == "failed" and stored.error == "RuntimeError: original collector error"


@pytest.mark.asyncio
async def test_resumed_cli_timeout_is_labelled_resume_failed(test_engine, tmp_path, monkeypatch):
    from huddleroom.adapters.cli_adapter import CliAdapter

    async with _world(test_engine, tmp_path, monkeypatch, adapter="cli") as (factory, _project, _agent, _task, session):
        async with factory() as db:
            row = await db.get(Session, session.id)
            row.provider_session_id = "prov-old"
            row.metadata_ = {"_run_config": {"timeout": 0.3}}
            await db.commit()
        with patch.object(CliAdapter, "_setup_sandbox", return_value=None), \
                patch.object(CliAdapter, "_terminate_process_group", new=AsyncMock()), \
                patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec",
                      new=AsyncMock(return_value=_StreamProc())):
            await asyncio.wait_for(session_tasks.execute_cli_session(str(session.id), "runner-1"), timeout=10)
        stored = await _reload(factory, Session, session.id)
        assert stored.error == "timeout"
        assert stored.provider_session_id == "prov-old" and stored.resumable is True
        assert stored.metadata_["resume_outcome"] == "resume_failed"


@pytest.mark.asyncio
async def test_escaped_failure_reason_is_redacted_before_truncation(test_engine, tmp_path, monkeypatch):
    from huddleroom.adapters.cli_adapter import CliAdapter

    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"

    async def broken_collector(*_args, **_kwargs):
        raise RuntimeError("a" * 175 + f" token={secret}")  # the secret straddles the 200 char cut

    async with _world(test_engine, tmp_path, monkeypatch, adapter="cli") as (factory, _project, _agent, _task, session):
        with patch.object(CliAdapter, "_setup_sandbox", return_value=None), \
                patch.object(CliAdapter, "_terminate_process_group", new=AsyncMock()), \
                patch("huddleroom.adapters.cli_adapter.collect_cli_process", new=broken_collector), \
                patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec",
                      new=AsyncMock(return_value=_StreamProc())):
            with pytest.raises(RuntimeError):
                await asyncio.wait_for(session_tasks.execute_cli_session(str(session.id), "runner-1"), timeout=10)
        error = (await _reload(factory, Session, session.id)).error
        assert len(error) <= 200
        assert secret[:12] not in error and "sk-" not in error


@pytest.mark.asyncio
async def test_escaped_failure_after_effect_started_stays_unknown_effect_for_recovery(test_engine, tmp_path, monkeypatch):
    import time
    from datetime import datetime, timezone

    import litellm

    from huddleroom.adapters import api_adapter
    from huddleroom.services.orchestration_recovery_service import OrchestrationRecoveryService, RunnerObservation
    from tests.test_orchestration_worker_ownership_recovery import _owned

    async def rate_limited(**_kwargs):
        time.sleep(0.15)
        raise litellm.RateLimitError("retry", "openai", "gpt-4o-mini")

    async def no_wait_for(awaitable, timeout=None):
        return await awaitable

    monkeypatch.setattr(api_adapter, "asyncio", SimpleNamespace(
        **{name: getattr(asyncio, name) for name in ("get_running_loop", "sleep", "CancelledError")},
        wait_for=no_wait_for))
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(session_tasks, "AsyncSessionLocal", factory)
    async with factory() as db:
        project = Project(name=f"Project-{uuid.uuid4()}", workspace_path=str(tmp_path), config={})
        agent = Agent(name=f"agent-{uuid.uuid4()}", role="developer", provider="openai", model="gpt-4o-mini",
                      adapter_type="api", capabilities=[], config={})
        db.add_all([project, agent])
        await db.flush()
        goal, _run, _action, _task, session = await _owned(
            db, project, agent, session_status="pending", task_status="in_progress", runner="runner-1")
        session.adapter_type = "api"
        session.metadata_ = {**session.metadata_, "_run_config": {"timeout": 0.1}}
        await db.commit()
        session_id, goal_id = session.id, goal.id

    with patch("litellm.acompletion", new=rate_limited):
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(session_tasks.execute_api_session(str(session_id), "runner-1"), timeout=10)
    stored = await _reload(factory, Session, session_id)
    assert stored.status == "failed"
    assert stored.metadata_["attempt"]["effect_state"] == "started"
    async with factory() as db:
        service = OrchestrationRecoveryService()
        snapshot = await service.build_goal_snapshot(db, goal_id, datetime.now(timezone.utc))
        owned = snapshot.sessions[0]
        live = RunnerObservation("runner-1", "active", datetime.now(timezone.utc))
        gone = RunnerObservation("runner-1", "unknown", datetime.now(timezone.utc))
        assert service.classify(owned, live) == "unknown_external_effect"
        assert service.classify(owned, gone) not in {"interrupted_safe", "resume_exact"}


@pytest.mark.asyncio
async def test_early_provider_id_write_failure_does_not_kill_healthy_run(test_engine, tmp_path, monkeypatch):
    from huddleroom.adapters.cli_adapter import CliAdapter

    class FinishingProc(_StreamProc):
        returncode = 0

        async def wait(self):
            return 0

    proc = FinishingProc(_EARLY + [b'{"type": "result", "result": "done", "session_id": "prov-early-2"}\n'])
    proc.stdout.feed_eof()
    proc.stderr.feed_eof()
    persist = AsyncMock(side_effect=RuntimeError("database is locked"))
    async with _world(test_engine, tmp_path, monkeypatch, adapter="cli") as (factory, _project, _agent, task, session):
        with patch.object(CliAdapter, "_setup_sandbox", return_value=None), \
                patch.object(CliAdapter, "_persist_partial", new=persist), \
                patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec",
                      new=AsyncMock(return_value=proc)):
            await asyncio.wait_for(session_tasks.execute_cli_session(str(session.id), "runner-1"), timeout=10)
        persist.assert_awaited_once()
        stored = await _reload(factory, Session, session.id)
        assert stored.status == "completed", stored.error
        assert stored.provider_session_id == "prov-early-2"
        assert (await _reload(factory, Task, task.id)).status == "done"


@pytest.mark.asyncio
async def test_cli_spawn_failure_without_provider_id_is_not_resumable(test_engine, tmp_path, monkeypatch):
    from huddleroom.adapters.cli_adapter import CliAdapter

    async with _world(test_engine, tmp_path, monkeypatch, adapter="cli") as (factory, project, _agent, _task, session):
        for exc in (FileNotFoundError("nope"), OSError("spawn failed")):
            async with factory() as db:
                row = await db.get(Session, session.id)
                row.status, row.error, row.resumable = "pending", None, False
                row.started_at = None
                row.metadata_ = {}
                await db.commit()
            with patch.object(CliAdapter, "_setup_sandbox", return_value=None), \
                    patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec",
                          new=AsyncMock(side_effect=exc)):
                await session_tasks.execute_cli_session(str(session.id), "runner-1")
            stored = await _reload(factory, Session, session.id)
            assert stored.status == "failed" and stored.resumable is False
        async with factory() as db:
            events = (await db.execute(select(EventLog).where(
                EventLog.project_id == project.id, EventLog.event_type == "session.failed"))).scalars().all()
        assert events and all(e.payload["resumable"] is False for e in events)


@pytest.mark.asyncio
async def test_resumed_cli_collector_failure_is_labelled_resume_failed(test_engine, tmp_path, monkeypatch):
    from huddleroom.adapters.cli_adapter import CliAdapter

    async def broken_collector(*_args, **_kwargs):
        raise RuntimeError("collector blew up")

    async with _world(test_engine, tmp_path, monkeypatch, adapter="cli") as (factory, _project, _agent, _task, session):
        async with factory() as db:
            row = await db.get(Session, session.id)
            row.provider_session_id = "prov-old"
            await db.commit()
        with patch.object(CliAdapter, "_setup_sandbox", return_value=None), \
                patch.object(CliAdapter, "_terminate_process_group", new=AsyncMock()), \
                patch("huddleroom.adapters.cli_adapter.collect_cli_process", new=broken_collector), \
                patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec",
                      new=AsyncMock(return_value=_StreamProc())):
            with pytest.raises(RuntimeError):
                await asyncio.wait_for(session_tasks.execute_cli_session(str(session.id), "runner-1"), timeout=10)
        assert (await _reload(factory, Session, session.id)).metadata_["resume_outcome"] == "resume_failed"


@pytest.mark.asyncio
async def test_owner_cancel_emits_no_failed_task_event_but_manual_cancel_does(test_engine, tmp_path, monkeypatch):
    async def failed_events(factory, project_id):
        async with factory() as db:
            rows = (await db.execute(select(EventLog).where(
                EventLog.project_id == project_id, EventLog.event_type == "task.status_changed"))).scalars().all()
        return [e for e in rows if e.payload.get("status") == "failed"]

    async with _world(test_engine, tmp_path, monkeypatch, session_status="running", runner=None) as (
            factory, project, _agent, task, session):
        async with factory() as db:
            await SessionService().cancel(db, session.id, sync_task=False)
            from huddleroom.services.task_service import TaskService
            await TaskService().cancel(db, project.id, task.id)
            await db.commit()
        assert (await _reload(factory, Task, task.id)).status == "cancelled"
        assert (await _reload(factory, Session, session.id)).ended_at is not None
        assert await failed_events(factory, project.id) == []

    async with _world(test_engine, tmp_path, monkeypatch, session_status="running", runner=None) as (
            factory, project, _agent, task, session):
        async with factory() as db:
            await SessionService().cancel(db, session.id)
            await db.commit()
        assert len(await failed_events(factory, project.id)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_managed", [True, False])
async def test_sqlite_cli_cancel_task_sync_follows_canceller_ownership(test_engine, tmp_path, monkeypatch, owner_managed):
    from huddleroom.adapters.cli_adapter import CliAdapter
    from huddleroom.config import settings
    from huddleroom.workers import task_runner

    if not settings.is_sqlite:
        pytest.skip("worker-driven CLI cancel is the SQLite path")
    started = asyncio.Event()

    class HangingProc:
        returncode = None
        pid = 99999

        async def communicate(self):
            started.set()
            await asyncio.Event().wait()

    runner = f"runner-{uuid.uuid4()}"
    async with _world(test_engine, tmp_path, monkeypatch, adapter="cli", runner=runner) as (
            factory, project, _agent, task, session):
        with patch.object(CliAdapter, "_setup_sandbox", return_value=None), \
                patch.object(CliAdapter, "_terminate_process_group", new=AsyncMock()), \
                patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec",
                      new=AsyncMock(return_value=HangingProc())):
            await task_runner.dispatch_session(str(session.id), "cli", project.id, runner)
            await asyncio.wait_for(started.wait(), timeout=10)
            async with factory() as db:
                cancelled = await SessionService().cancel(db, session.id, sync_task=not owner_managed)
            assert cancelled.status == "cancelled"
        async with factory() as db:
            failed = [e for e in (await db.execute(select(EventLog).where(
                EventLog.project_id == project.id, EventLog.event_type == "task.status_changed"))).scalars()
                if e.payload.get("status") == "failed"]
        stored = await _reload(factory, Session, session.id)
        assert "cancel_owner_managed" not in (stored.metadata_ or {})
        assert (await _reload(factory, Task, task.id)).status == ("in_progress" if owner_managed else "failed")
        assert len(failed) == (0 if owner_managed else 1)
