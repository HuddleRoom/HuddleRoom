"""Provider-free cancellation with actual dispatch and independent transactions."""
import asyncio
import os
import signal
import sys
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from huddleroom.models.event_log import EventLog
from huddleroom.models.session import Session
from huddleroom.models.project import Project
from huddleroom.services.session_service import SessionService
from huddleroom.workers import session_tasks, task_runner
from tests.test_cli_workspace_isolation import _create_cli_session, _wait_for_path
from tests.test_orchestration_plan_delegation_gate import _make_run, _agent, _plan_request, _bind_team

async def owned_cli(test_engine, tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    heartbeat, pids = tmp_path / "heartbeat", tmp_path / "pids"
    child = ("import os,pathlib,time; " +
        f"pathlib.Path({str(pids)!r}).write_text(str(os.getpid())+' '+str(os.getpgrp())+' '+str(os.getppid())); " +
        f"p=pathlib.Path({str(heartbeat)!r}); " +
        "\nwhile True:\n p.write_text(str(time.monotonic()))\n time.sleep(.02)")
    script = tmp_path / "agent.py"
    script.write_text(f"#!{sys.executable}\nimport subprocess,sys,time\n" +
        f"subprocess.Popen([sys.executable,'-c',{child!r}])\n" +
        "print('started',flush=True)\ntime.sleep(60)\n")
    script.chmod(0o755)
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    runner = str(uuid.uuid4())
    async with factory() as db:
        project = Project(name=f"cancel-{uuid.uuid4()}", workspace_path=str(workspace), config={})
        planner = _agent("cli-cancel")
        planner.adapter_type = "cli"
        planner.config = {"cli_runtime": "custom", "script_path": str(script)}
        db.add_all([project, planner])
        await db.flush()
        service, goal, run = await _make_run(db, project.id)
        await _bind_team(db, goal, planner)
        action = await service.execute_request_plan_action(db, run_id=run.id,
            request=_plan_request(planner.id), idempotency_key=f"cancel:{runner}")
        row = Session(agent_id=planner.id, project_id=project.id, task_id=action.target_id,
            adapter_type="cli", status="pending", runner_task_id=runner,
            provider_session_id="known-native-session", metadata_={"_run_config": {"timeout": 60}})
        db.add(row)
        await db.commit()
        sid, pid = row.id, project.id
    monkeypatch.setattr(session_tasks, "AsyncSessionLocal", factory)
    await task_runner.dispatch_session(str(sid), "cli", pid, runner)
    try:
        await _wait_for_path(heartbeat)
    except BaseException:
        await cleanup_runner(runner, pids)
        raise
    return factory, sid, pid, runner, heartbeat, pids

async def cleanup_runner(runner, pids):
    if task_runner.is_task_live(runner):
        await task_runner.cancel_task(runner)
    if pids.exists():
        try:
            os.kill(int(pids.read_text().split()[0]), signal.SIGKILL)
        except ProcessLookupError:
            pass

@pytest.mark.asyncio
async def test_real_dispatch_cancel_waits_for_worker_commit_and_stops_output(test_engine, tmp_path, monkeypatch):
    factory, sid, project_id, runner, heartbeat, pids = await owned_cli(test_engine, tmp_path, monkeypatch)
    prepared, release = asyncio.Event(), asyncio.Event()
    original_commit, commits = AsyncSession.commit, []
    async def gated_commit(db):
        row = db.identity_map.get((Session, (sid,), None))
        if row is not None and row.status == "cancelled":
            commits.append(db)
            prepared.set()
            await release.wait()
        await original_commit(db)
    monkeypatch.setattr(AsyncSession, "commit", gated_commit)
    cancel = None
    try:
        assert task_runner.is_task_live(runner)
        child_pid, pgid, wrapper_pid = map(int, pids.read_text().split())
        assert pgid == wrapper_pid and os.getpgid(child_pid) == pgid
        async with factory() as caller:
            cached = await caller.get(Session, sid)
            version = cached.metadata_["attempt"]["attempt_version"]
            cancel = asyncio.create_task(SessionService().cancel(caller, sid))
            await asyncio.wait_for(prepared.wait(), 4)
            assert not cancel.done(), "caller returned before independent worker commit"
            stopped = heartbeat.read_text()
            await asyncio.sleep(.08)
            assert heartbeat.read_text() == stopped
            release.set()
            result = await asyncio.wait_for(cancel, 4)
            assert result.status == "cancelled" and result.ended_at is not None
            assert result.provider_session_id == "known-native-session"
            attempt = result.metadata_["attempt"]
            assert attempt["result_status"] == "cancelled"
            assert attempt["result_recorded_at"]
            assert attempt["provider_session_id"] == "known-native-session"
            assert attempt["usage_complete"] is False
            assert attempt["attempt_version"] == version + 1
            await caller.rollback()
        assert len(commits) == 1
        assert not task_runner.is_task_live(runner)
        async with factory() as verify:
            row = await verify.get(Session, sid)
            assert row.ended_at is not None and row.status == "cancelled"
            events = list((await verify.scalars(select(EventLog).where(
                EventLog.project_id == project_id, EventLog.event_type == "session.cancelled"))).all())
            assert len(events) == 1
        await asyncio.sleep(.08)
        assert heartbeat.read_text() == stopped
    finally:
        release.set()
        if cancel is not None and not cancel.done():
            await asyncio.gather(cancel, return_exceptions=True)
        await cleanup_runner(runner, pids)

@pytest.mark.asyncio
async def test_sqlite_cli_registry_miss_cannot_fabricate_cancellation(test_engine, tmp_path, monkeypatch):
    factory, sid, project_id = await _create_cli_session(test_engine, tmp_path, "print('unused')")
    async with factory.begin() as db:
        row = await db.get(Session, sid)
        row.status, row.runner_task_id = "running", "missing-runner"
    monkeypatch.setattr(task_runner, "cancel_task", AsyncMock(return_value=False))
    async with factory() as db:
        with pytest.raises(HTTPException) as error:
            await SessionService().cancel(db, sid)
        assert error.value.status_code == 409
        assert error.value.detail["code"] == "cli_runner_not_owned"
        assert (await db.get(Session, sid)).status == "running"
        assert not list((await db.scalars(select(EventLog).where(
            EventLog.project_id == project_id, EventLog.event_type == "session.cancelled"))).all())

@pytest.mark.asyncio
@pytest.mark.parametrize("adapter,sqlite", [("api", True), ("cli", False)])
async def test_generic_cancel_does_not_require_local_registry(test_engine, tmp_path, monkeypatch, adapter, sqlite):
    factory, sid, project_id = await _create_cli_session(test_engine, tmp_path, "print('unused')")
    import huddleroom.services.session_service as module
    monkeypatch.setattr("huddleroom.config.settings", SimpleNamespace(is_sqlite=sqlite))
    monkeypatch.setattr(task_runner, "cancel_task", AsyncMock(return_value=False))
    async with factory() as db:
        row = await db.get(Session, sid)
        row.adapter_type, row.runner_task_id = adapter, "celery-or-api-runner"
        await db.flush()
        result = await SessionService().cancel(db, sid)
        assert result.status == "cancelled"
        assert len(list((await db.scalars(select(EventLog).where(
            EventLog.project_id == project_id, EventLog.event_type == "session.cancelled"))).all())) == 1

@pytest.mark.asyncio
async def test_real_worker_commit_failure_cannot_report_success(test_engine, tmp_path, monkeypatch):
    factory, sid, project_id, runner, heartbeat, pids = await owned_cli(test_engine, tmp_path, monkeypatch)
    original_commit = AsyncSession.commit
    failed = []
    async def fail_terminal_commit(db):
        row = db.identity_map.get((Session, (sid,), None))
        if row is not None and row.status == "cancelled":
            failed.append(True)
            raise RuntimeError("injected cancellation commit failure")
        await original_commit(db)
    monkeypatch.setattr(AsyncSession, "commit", fail_terminal_commit)
    try:
        async with factory() as caller:
            with pytest.raises((HTTPException, RuntimeError)):
                await asyncio.wait_for(SessionService().cancel(caller, sid), 4)
        assert failed == [True]
        async with factory() as verify:
            row = await verify.get(Session, sid)
            assert row.status == "running" and row.ended_at is None
            assert not list((await verify.scalars(select(EventLog).where(
                EventLog.project_id == project_id, EventLog.event_type == "session.cancelled"))).all())
    finally:
        await cleanup_runner(runner, pids)

@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["cancelled", "completed", "failed"])
@pytest.mark.parametrize("ack", [True, False])
async def test_cancel_refresh_does_not_overwrite_independent_terminal_result(
    test_engine, tmp_path, monkeypatch, terminal, ack,
):
    from datetime import datetime, timezone
    factory, sid, project_id = await _create_cli_session(test_engine, tmp_path, "print('unused')")
    async with factory.begin() as db:
        row = await db.get(Session, sid)
        row.status, row.runner_task_id = "running", "known-runner"
    async def finish(_runner):
        async with factory.begin() as worker:
            row = await worker.get(Session, sid)
            row.status, row.ended_at = terminal, datetime.now(timezone.utc)
            worker.add(EventLog(project_id=project_id, event_type=f"session.{terminal}", payload={}))
        return ack
    monkeypatch.setattr(task_runner, "cancel_task", finish)
    async with factory() as caller:
        await caller.get(Session, sid)
        try:
            result = await SessionService().cancel(caller, sid)
        except HTTPException as exc:
            assert terminal != "cancelled" and exc.status_code == 409
        else:
            assert result.status == terminal and result.ended_at is not None
        await caller.rollback()
    async with factory() as verify:
        assert (await verify.get(Session, sid)).status == terminal
        events = list((await verify.scalars(select(EventLog).where(EventLog.project_id == project_id))).all())
        assert len(events) == 1 and events[0].event_type == f"session.{terminal}"

@pytest.mark.asyncio
@pytest.mark.parametrize("runner", [None, "known-runner"])
async def test_ack_without_terminal_commit_is_conflict(test_engine, tmp_path, monkeypatch, runner):
    factory, sid, project_id = await _create_cli_session(test_engine, tmp_path, "print('unused')")
    async with factory.begin() as db:
        row = await db.get(Session, sid)
        row.status, row.runner_task_id = "running", runner
    monkeypatch.setattr(task_runner, "cancel_task", AsyncMock(return_value=True))
    async with factory() as caller:
        with pytest.raises(HTTPException) as exc:
            await SessionService().cancel(caller, sid)
        assert exc.value.status_code == 409
        assert exc.value.detail["code"] == ("cli_cancel_not_terminalized" if runner else "cli_runner_not_owned")
        assert (await caller.get(Session, sid)).status == "running"
        assert not list((await caller.scalars(select(EventLog).where(
            EventLog.project_id == project_id, EventLog.event_type == "session.cancelled"))).all())

@pytest.mark.asyncio
async def test_concurrent_cancel_callers_do_not_interrupt_worker_terminal_commit(test_engine, tmp_path, monkeypatch):
    factory, sid, project_id, runner, heartbeat, pids = await owned_cli(test_engine, tmp_path, monkeypatch)
    prepared, release = asyncio.Event(), asyncio.Event()
    original_commit = AsyncSession.commit
    async def gated_commit(db):
        row = db.identity_map.get((Session, (sid,), None))
        if row is not None and row.status == "cancelled":
            prepared.set()
            await release.wait()
        await original_commit(db)
    monkeypatch.setattr(AsyncSession, "commit", gated_commit)
    callers = []
    try:
        async with factory() as first, factory() as second:
            callers.append(asyncio.create_task(SessionService().cancel(first, sid)))
            await asyncio.wait_for(prepared.wait(), 4)
            callers.append(asyncio.create_task(SessionService().cancel(second, sid)))
            await asyncio.sleep(.08)
            assert task_runner.is_task_live(runner), "second cancellation interrupted worker cleanup"
            assert all(not call.done() for call in callers)
            release.set()
            results = await asyncio.wait_for(asyncio.gather(*callers), 4)
            assert all(row.status == "cancelled" and row.ended_at is not None for row in results)
        async with factory() as verify:
            assert len(list((await verify.scalars(select(EventLog).where(
                EventLog.project_id == project_id, EventLog.event_type == "session.cancelled"))).all())) == 1
    finally:
        release.set()
        await asyncio.gather(*callers, return_exceptions=True)
        await cleanup_runner(runner, pids)

@pytest.mark.asyncio
async def test_detached_descendant_is_only_group_ownership_negative_control(tmp_path):
    """Deliberate setsid is diagnostic, not a reproduction of native runtime lineage."""
    from huddleroom.services.cli_streaming import terminate_process_group
    heartbeat, pid_path = tmp_path / "detached-heartbeat", tmp_path / "detached.pid"
    child = ("import os,pathlib,time; " +
        f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); " +
        f"p=pathlib.Path({str(heartbeat)!r}); " +
        "\nwhile True:\n p.write_text(str(time.monotonic()))\n time.sleep(.02)")
    # A deliberate new session lies outside the application's original process-group ownership.
    wrapper = ("import subprocess,sys,time; subprocess.Popen([sys.executable,'-c'," + repr(child) +
    "],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); time.sleep(60)")
    proc = await asyncio.create_subprocess_exec(sys.executable, "-c", wrapper, start_new_session=True)
    try:
        await _wait_for_path(heartbeat)
        pid = int(pid_path.read_text())
        assert os.getpgid(pid) != proc.pid
        await terminate_process_group(proc, grace_seconds=.1)
        before = heartbeat.read_text()
        await asyncio.sleep(.08)
        assert heartbeat.read_text() != before
    finally:
        await terminate_process_group(proc, grace_seconds=.1)
        if pid_path.exists():
            try:
                os.kill(int(pid_path.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass

@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["explicit", "pending", "flushed", "nested"])
async def test_standalone_cancel_rejects_caller_transaction_before_runner_effect(
    test_engine, tmp_path, monkeypatch, boundary,
):
    factory, sid, project_id, runner, heartbeat, pids = await owned_cli(test_engine, tmp_path, monkeypatch)
    signal_runner = AsyncMock(return_value=False)
    monkeypatch.setattr(task_runner, "cancel_task", signal_runner)
    try:
        async with factory() as caller:
            if boundary == "explicit":
                await caller.begin()
            elif boundary == "nested":
                await caller.begin_nested()
            project = await caller.get(Project, project_id)
            original_name = project.name
            if boundary in {"pending", "flushed"}:
                project.name = "uncommitted caller mutation"
                if boundary == "flushed":
                    await caller.flush()
                    assert not caller.new and not caller.dirty and not caller.deleted
            with pytest.raises(HTTPException) as error:
                await asyncio.wait_for(SessionService().cancel(caller, sid), 2)
            assert error.value.status_code == 409
            assert task_runner.is_task_live(runner)
            await caller.rollback()
        async with factory() as verify:
            assert (await verify.get(Project, project_id)).name == original_name
            row = await verify.get(Session, sid)
            assert row.status == "running" and row.ended_at is None
            assert not list((await verify.scalars(select(EventLog).where(
                EventLog.project_id == project_id, EventLog.event_type == "session.cancelled"))).all())
        signal_runner.assert_not_awaited()
    finally:
        # Restore actual cleanup; the spy represents the destructive boundary above.
        monkeypatch.undo()
        await cleanup_runner(runner, pids)
