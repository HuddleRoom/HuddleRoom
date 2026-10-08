import asyncio
import os
import signal
import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from huddleroom.models.meeting import Meeting, MeetingEvent
from huddleroom.models.agent import Agent
from huddleroom.models.project import Project
from huddleroom.adapters.cli_adapter import CliAdapter, CliTurnFailed
from huddleroom.schemas.project import ProjectUpdate
from huddleroom.services.meeting_service import MeetingService
from huddleroom.services.project_service import ProjectService


async def _make_runnable(project: Project, tmp_path: Path) -> None:
    workspace = tmp_path / str(uuid.uuid4())
    workspace.mkdir()
    project.workspace_path = str(workspace.resolve())


async def _make_turn_inputs(db, tmp_path: Path, timeout: float = 5, config: dict | None = None):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project = Project(name=f"Project-{uuid.uuid4()}", workspace_path=str(workspace), config={})
    agent = Agent(
        name=f"agent-{uuid.uuid4()}", role="reviewer", provider="openai", model="gpt-4o-mini",
        adapter_type="cli", capabilities=[],
        config=config or {"cli_runtime": "claude_code", "meeting_turn_timeout_seconds": timeout},
    )
    db.add_all([project, agent])
    await db.flush()
    meeting = Meeting(project_id=project.id, title="Turn", meeting_type="decision", status="active")
    db.add(meeting)
    await db.flush()
    await db.commit()
    return workspace, project, agent, meeting


async def _wait_for_path(path: Path) -> None:
    async with asyncio.timeout(2):
        while not path.exists():
            await asyncio.sleep(0.01)


async def _assert_process_exited(pid: int) -> None:
    async with asyncio.timeout(2):
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            await asyncio.sleep(0.01)


def _kill_process_if_running(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


@pytest.mark.asyncio
async def test_meeting_turn_uses_workspace_cwd_and_rally_context(db_session, tmp_path: Path):
    workspace, project, agent, meeting = await _make_turn_inputs(db_session, tmp_path)
    existing_session_id = "session_12345678"

    class CompletedProc:
        returncode = 0

        async def communicate(self):
            return b'{"result": "done"}', b""

    exchange = Mock()
    with patch.object(ProjectService, "require_runnable_project", new=AsyncMock(return_value=workspace)) as runnable:
        with patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", new=AsyncMock(return_value=CompletedProc())) as launch:
            with patch("huddleroom.adapters.cli_adapter.log_cli_exchange", exchange):
                content, _, _ = await CliAdapter().run_meeting_turn(
                    db_session, meeting, agent, project, "Discuss safely", existing_session_id
                )

    context_dir = workspace / ".huddleroom" / "agents" / str(agent.id) / "meetings" / str(meeting.id) / str(agent.id)
    command = launch.await_args.args
    assert content == "done"
    assert runnable.await_count == 2
    assert launch.await_args.kwargs["cwd"] == str(workspace)
    assert launch.await_args.kwargs["start_new_session"] is True
    assert command == (
        "claude", "--dangerously-skip-permissions", "--print", "--verbose", "--model", "gpt-4o-mini",
        "--output-format", "stream-json", "--resume", existing_session_id, "--", "Discuss safely",
    )
    assert (context_dir / "meeting_context.md").read_text() == "Discuss safely"
    assert (context_dir / "huddleroom_context.json").is_file()
    exchange.assert_called_once_with(
        prompt="Discuss safely",
        session_id=existing_session_id,
        response="done",
        runtime="claude_code",
        meeting_id=str(meeting.id),
        agent_id=str(agent.id),
    )


@pytest.mark.asyncio
async def test_meeting_turn_failure_logs_provider_session_and_preserves_exception(db_session, tmp_path: Path):
    workspace, project, agent, meeting = await _make_turn_inputs(db_session, tmp_path)

    class FailedProc:
        returncode = 2

        async def communicate(self):
            return b'{"session_id":"failed-12345678"}', b"api_key=private-value"

    exchange = Mock()
    with patch.object(ProjectService, "require_runnable_project", new=AsyncMock(return_value=workspace)):
        with patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", new=AsyncMock(return_value=FailedProc())):
            with patch("huddleroom.adapters.cli_adapter.log_cli_exchange", exchange):
                with pytest.raises(CliTurnFailed, match="exit_code=2") as error:
                    await CliAdapter().run_meeting_turn(
                        db_session, meeting, agent, project, "Discuss safely", "resumed-12345678"
                    )

    exchange.assert_called_once_with(
        prompt="Discuss safely",
        session_id="failed-12345678",
        error=error.value,
        runtime="claude_code",
        meeting_id=str(meeting.id),
        agent_id=str(agent.id),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_output", [b"[]", b"null"])
async def test_meeting_turn_failure_ignores_non_object_json(db_session, tmp_path: Path, failure_output: bytes):
    workspace, project, agent, meeting = await _make_turn_inputs(db_session, tmp_path)

    class FailedProc:
        returncode = 2

        async def communicate(self):
            return failure_output, b"failed"

    with patch.object(ProjectService, "require_runnable_project", new=AsyncMock(return_value=workspace)):
        with patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", new=AsyncMock(return_value=FailedProc())):
            with pytest.raises(CliTurnFailed, match="exit_code=2"):
                await CliAdapter().run_meeting_turn(db_session, meeting, agent, project, "Discuss safely", None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime", "stdout"),
    [
        ("copilot", b'{"type":"result","sessionId":"failed-runtime-session"}\n'),
        ("opencode", b'{"type":"step_start","sessionID":"failed-runtime-session"}\n'),
        ("pi", b'{"type":"session","id":"failed-runtime-session"}\n'),
    ],
)
async def test_step8_meeting_failure_captures_jsonl_session_id(
    db_session, tmp_path: Path, runtime: str, stdout: bytes,
):
    workspace, project, agent, meeting = await _make_turn_inputs(
        db_session, tmp_path, config={"cli_runtime": runtime},
    )

    class FailedProc:
        returncode = 2

        async def communicate(self):
            return stdout, b"failed"

    with patch.object(ProjectService, "require_runnable_project", new=AsyncMock(return_value=workspace)):
        with patch(
            "huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", new=AsyncMock(return_value=FailedProc())
        ):
            with pytest.raises(CliTurnFailed, match="exit_code=2") as error:
                await CliAdapter().run_meeting_turn(db_session, meeting, agent, project, "Discuss safely", None)

    assert error.value.session_id == "failed-runtime-session"


@pytest.mark.asyncio
async def test_meeting_turn_launch_failure_logs_and_preserves_exception(db_session, tmp_path: Path):
    workspace, project, agent, meeting = await _make_turn_inputs(db_session, tmp_path)
    launch_error = PermissionError("api_key=private-value")
    exchange = Mock()

    with patch.object(ProjectService, "require_runnable_project", new=AsyncMock(return_value=workspace)):
        with patch(
            "huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=launch_error),
        ):
            with patch("huddleroom.adapters.cli_adapter.log_cli_exchange", exchange):
                with pytest.raises(PermissionError) as error:
                    await CliAdapter().run_meeting_turn(
                        db_session, meeting, agent, project, "Discuss safely", "resumed-12345678"
                    )

    assert error.value is launch_error
    exchange.assert_called_once_with(
        prompt="Discuss safely",
        session_id="resumed-12345678",
        error=launch_error,
        runtime="claude_code",
        meeting_id=str(meeting.id),
        agent_id=str(agent.id),
    )


@pytest.mark.asyncio
async def test_meeting_turn_unprintable_launch_failure_preserves_exception(db_session, tmp_path: Path):
    workspace, project, agent, meeting = await _make_turn_inputs(db_session, tmp_path)

    class UnprintableError(Exception):
        def __str__(self):
            raise RuntimeError("cannot stringify")

    launch_error = UnprintableError()
    exchange = Mock()
    with patch.object(ProjectService, "require_runnable_project", new=AsyncMock(return_value=workspace)):
        with patch(
            "huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", new=AsyncMock(side_effect=launch_error)
        ):
            with patch("huddleroom.adapters.cli_adapter.log_cli_exchange", exchange):
                with pytest.raises(UnprintableError) as error:
                    await CliAdapter().run_meeting_turn(
                        db_session, meeting, agent, project, "Discuss safely", "resumed-12345678"
                    )

    assert error.value is launch_error
    assert exchange.call_args.kwargs["error"] is launch_error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime", "config", "expected_command", "stdout"),
    [
        (
            "codex",
            {},
            lambda workspace, context_path: (
                "codex", "exec", "--json", "--sandbox", "workspace-write", "--cd", str(workspace),
                "--skip-git-repo-check", "-c", "sandbox_workspace_write.network_access=true",
                "--model", "gpt-4o-mini", "Discuss safely",
            ),
            b'{"type":"thread.started","thread_id":"t1"}\n'
            b'{"type":"item.completed","item":{"type":"agent_message","text":"done"}}\n',
        ),
        (
            "aider",
            {},
            lambda workspace, context_path: (
                "aider", "--yes", "--no-pretty", "--model", "gpt-4o-mini", "--message", "Discuss safely",
            ),
            b'{"result": "done"}',
        ),
        (
            "copilot",
            {},
            lambda workspace, context_path: (
                "copilot", "-p", "Discuss safely", "-C", str(workspace), "--no-ask-user",
                "--allow-all-tools", "--output-format", "json", "--no-auto-update", "--model", "gpt-4o-mini",
            ),
            b'{"type":"assistant.message","data":{"content":"done"}}\n'
            b'{"type":"result","sessionId":"copilot-session"}\n',
        ),
        (
            "opencode",
            {},
            lambda workspace, context_path: (
                "opencode", "run", "--format", "json", "--auto", "--dir", str(workspace),
                "--model", "gpt-4o-mini", "Discuss safely",
            ),
            b'{"type":"text","sessionID":"opencode-session","part":{"type":"text","text":"done"}}\n',
        ),
        (
            "pi",
            {},
            lambda workspace, context_path: (
                "pi", "--print", "--mode", "json", "--no-approve", "--model", "gpt-4o-mini", "Discuss safely",
            ),
            b'{"type":"session","id":"pi-session"}\n'
            b'{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"done"}]}}\n',
        ),
        (
            "custom",
            {"script_path": "/tmp/meeting-agent"},
            lambda workspace, context_path: ("/tmp/meeting-agent", str(context_path)),
            b'{"result": "done"}',
        ),
    ],
)
async def test_meeting_turn_uses_the_session_runtime_policy(
    db_session, tmp_path: Path, runtime: str, config: dict, expected_command, stdout: bytes,
):
    workspace, project, agent, meeting = await _make_turn_inputs(
        db_session,
        tmp_path,
        config={"cli_runtime": runtime, "meeting_turn_timeout_seconds": 5, **config},
    )

    class CompletedProc:
        returncode = 0

        async def communicate(self):
            return stdout, b""

    with patch.object(ProjectService, "require_runnable_project", new=AsyncMock(return_value=workspace)):
        with patch(
            "huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=CompletedProc()),
        ) as launch:
            content, _, _ = await CliAdapter().run_meeting_turn(
                db_session, meeting, agent, project, "Discuss safely", None
            )

    context_path = (
        workspace / ".huddleroom" / "agents" / str(agent.id) / "meetings" / str(meeting.id) / str(agent.id)
        / "meeting_context.md"
    )
    assert content == "done"
    assert launch.await_args.args == expected_command(workspace, context_path)
    assert launch.await_args.kwargs["cwd"] == str(workspace)
    assert context_path.read_text() == "Discuss safely"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime", "config", "error"),
    [
        ("custom", {}, "custom_runtime_missing_script_path"),
        ("unknown-runtime", {}, "unsupported_cli_runtime"),
    ],
)
async def test_meeting_turn_rejects_invalid_runtime_before_launch(
    db_session, tmp_path: Path, runtime: str, config: dict, error: str,
):
    workspace, project, agent, meeting = await _make_turn_inputs(
        db_session, tmp_path, config={"cli_runtime": runtime, **config}
    )

    with patch.object(ProjectService, "require_runnable_project", new=AsyncMock(return_value=workspace)):
        with patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", new=AsyncMock()) as launch:
            with pytest.raises(RuntimeError, match=error):
                await CliAdapter().run_meeting_turn(db_session, meeting, agent, project, "Discuss safely", None)

    launch.assert_not_awaited()
    context_dir = workspace / ".huddleroom" / "agents" / str(agent.id) / "meetings" / str(meeting.id) / str(agent.id)
    assert not (context_dir / "meeting_context.md").exists()
    assert not (context_dir / "huddleroom_context.json").exists()


@pytest.mark.asyncio
async def test_meeting_turn_does_not_launch_after_workspace_disappears(db_session, tmp_path: Path):
    workspace, project, agent, meeting = await _make_turn_inputs(db_session, tmp_path)
    unavailable = HTTPException(
        status_code=409, detail={"code": "project_not_runnable", "reason": "workspace_unavailable"}
    )

    with patch.object(ProjectService, "require_runnable_project", new=AsyncMock(side_effect=[workspace, unavailable])):
        with patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", new=AsyncMock()) as launch:
            with pytest.raises(HTTPException) as error:
                await CliAdapter().run_meeting_turn(db_session, meeting, agent, project, "Discuss safely", None)

    assert error.value is unavailable
    launch.assert_not_awaited()
    context_dir = workspace / ".huddleroom" / "agents" / str(agent.id) / "meetings" / str(meeting.id) / str(agent.id)
    assert not (context_dir / "meeting_context.md").exists()
    assert not (context_dir / "huddleroom_context.json").exists()


@pytest.mark.asyncio
async def test_meeting_turn_timeout_terminates_process_group(db_session, tmp_path: Path):
    workspace, project, agent, meeting = await _make_turn_inputs(db_session, tmp_path, timeout=0.1)
    child_pid_path = tmp_path / "timeout-child.pid"
    original_launch = asyncio.create_subprocess_exec

    async def launch(*_args, **kwargs):
        return await original_launch(
            sys.executable, "-c",
            (
                "import pathlib, subprocess, sys\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid))\n"
                "import time; time.sleep(60)\n"
            ),
            **kwargs,
        )

    with patch.object(ProjectService, "require_runnable_project", new=AsyncMock(return_value=workspace)):
        with patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", new=launch):
            with pytest.raises(RuntimeError, match="timed out"):
                await CliAdapter().run_meeting_turn(db_session, meeting, agent, project, "Discuss safely", None)

    child_pid = int(child_pid_path.read_text())
    try:
        await _assert_process_exited(child_pid)
    finally:
        _kill_process_if_running(child_pid)


@pytest.mark.asyncio
async def test_meeting_turn_external_cancel_terminates_process_group(db_session, tmp_path: Path):
    workspace, project, agent, meeting = await _make_turn_inputs(db_session, tmp_path)
    child_pid_path = tmp_path / "cancel-child.pid"
    original_launch = asyncio.create_subprocess_exec

    async def launch(*_args, **kwargs):
        return await original_launch(
            sys.executable, "-c",
            (
                "import pathlib, subprocess, sys, time\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid))\n"
                "time.sleep(60)\n"
            ),
            **kwargs,
        )

    run_task = None
    child_pid = None
    try:
        with patch.object(ProjectService, "require_runnable_project", new=AsyncMock(return_value=workspace)):
            with patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", new=launch):
                run_task = asyncio.create_task(
                    CliAdapter().run_meeting_turn(db_session, meeting, agent, project, "Discuss safely", None)
                )
                await _wait_for_path(child_pid_path)
                child_pid = int(child_pid_path.read_text())
                run_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(run_task, timeout=1)

        await _assert_process_exited(child_pid)
    finally:
        if run_task is not None and not run_task.done():
            run_task.cancel()
            await asyncio.wait_for(asyncio.gather(run_task, return_exceptions=True), timeout=1)
        if child_pid is not None:
            _kill_process_if_running(child_pid)


@pytest.mark.asyncio
async def test_meeting_turn_resetting_watcher_terminates_process_group(db_session, tmp_path: Path):
    workspace, project, agent, meeting = await _make_turn_inputs(db_session, tmp_path)
    child_pid_path = tmp_path / "reset-child.pid"
    original_launch = asyncio.create_subprocess_exec

    async def launch(*_args, **kwargs):
        return await original_launch(
            sys.executable, "-c",
            (
                "import pathlib, subprocess, sys, time\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid))\n"
                "time.sleep(60)\n"
            ),
            **kwargs,
        )

    async def resetting_watcher(*_args, **_kwargs):
        await _wait_for_path(child_pid_path)
        return "resetting"

    with patch.object(ProjectService, "require_runnable_project", new=AsyncMock(return_value=workspace)):
        with patch.object(CliAdapter, "_watch_project_status", new=resetting_watcher):
            with patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", new=launch):
                with pytest.raises(HTTPException) as error:
                    await CliAdapter().run_meeting_turn(db_session, meeting, agent, project, "Discuss safely", None)

    assert error.value.detail == {"code": "project_not_runnable", "reason": "project_inactive"}

    child_pid = int(child_pid_path.read_text())
    try:
        await _assert_process_exited(child_pid)
    finally:
        _kill_process_if_running(child_pid)


@pytest.mark.asyncio
async def test_reset_during_in_flight_meeting_turn_persists_no_turn_or_redispatches(test_engine, tmp_path: Path):
    from huddleroom.models.meeting import MeetingAgendaItem, MeetingTurn
    from huddleroom.workers.meeting_tasks import run_meeting_turn_async

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    child_pid_path = tmp_path / "worker-reset-child.pid"
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory.begin() as setup_db:
        project = Project(name=f"Project-{uuid.uuid4()}", workspace_path=str(workspace), config={})
        agent = Agent(
            name=f"agent-{uuid.uuid4()}", role="reviewer", provider="openai", model="gpt-4o-mini",
            adapter_type="cli", capabilities=[], config={"cli_runtime": "claude_code"},
        )
        setup_db.add_all([project, agent])
        await setup_db.flush()
        meeting = Meeting(
            project_id=project.id, title="Reset race", meeting_type="decision", status="active",
            participant_agent_ids=[str(agent.id)],
        )
        setup_db.add(meeting)
        await setup_db.flush()
        setup_db.add(MeetingAgendaItem(
            meeting_id=meeting.id, order=1, title="Question", question="Continue?", status="active",
        ))
        project_id = project.id
        meeting_id = meeting.id

    original_launch = asyncio.create_subprocess_exec

    async def launch(*_args, **kwargs):
        return await original_launch(
            sys.executable, "-c",
            (
                "import pathlib, subprocess, sys, time\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid))\n"
                "time.sleep(60)\n"
            ),
            **kwargs,
        )

    with patch("huddleroom.workers.meeting_tasks.AsyncSessionLocal", session_factory):
        with patch(
            "huddleroom.services.meeting_context.MeetingContextService.build_cli_turn_prompt",
            new=AsyncMock(return_value=("Discuss safely", {})),
        ):
            with patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", new=launch):
                with patch("huddleroom.workers.meeting_tasks.dispatch_run_meeting_turn") as dispatch:
                    run_task = asyncio.create_task(run_meeting_turn_async(str(meeting_id)))
                    await _wait_for_path(child_pid_path)
                    async with session_factory.begin() as reset_db:
                        resetting_project = await reset_db.get(Project, project_id)
                        resetting_project.status = "resetting"
                    await asyncio.wait_for(run_task, timeout=3)

    child_pid = int(child_pid_path.read_text())
    try:
        await _assert_process_exited(child_pid)
        async with session_factory() as assert_db:
            turns = (await assert_db.execute(
                select(MeetingTurn).where(MeetingTurn.meeting_id == meeting_id)
            )).scalars().all()
        assert turns == []
        dispatch.assert_not_called()
    finally:
        _kill_process_if_running(child_pid)


@pytest.mark.asyncio
async def test_non_auto_start_meeting_can_be_configured_for_non_runnable_project(db_session, test_project):
    test_project.workspace_path = None
    meeting = await MeetingService().create_meeting(
        db_session, test_project.id, "Plan", "decision", [], [], auto_start=False
    )

    assert meeting.status == "scheduled"


@pytest.mark.asyncio
async def test_start_conflict_leaves_scheduled_meeting_and_does_not_launch(db_session, test_project):
    from huddleroom.workers.meeting_tasks import start_meeting_async

    test_project.workspace_path = None
    meeting = await MeetingService().create_meeting(
        db_session, test_project.id, "Blocked", "decision", [], []
    )
    db_session.commit = AsyncMock()
    db_session.rollback = AsyncMock()

    with patch("huddleroom.workers.meeting_tasks.AsyncSessionLocal") as sessions:
        sessions.return_value.__aenter__ = AsyncMock(return_value=db_session)
        sessions.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("huddleroom.workers.meeting_tasks.dispatch_run_meeting_turn") as dispatch:
            await start_meeting_async(str(meeting.id))

    assert meeting.status == "scheduled"
    dispatch.assert_not_called()


@pytest.mark.asyncio
async def test_turn_conflict_does_not_run_adapter_or_retry(db_session, test_project, test_agent, tmp_path: Path):
    from huddleroom.workers.meeting_tasks import run_meeting_turn_async

    await _make_runnable(test_project, tmp_path)
    meeting = Meeting(
        project_id=test_project.id,
        title="Active", meeting_type="decision", status="active",
        participant_agent_ids=[str(test_agent.id)],
    )
    db_session.add(meeting)
    await db_session.flush()
    test_project.status = "archived"
    db_session.commit = AsyncMock()
    db_session.rollback = AsyncMock()

    with patch("huddleroom.workers.meeting_tasks.AsyncSessionLocal") as sessions:
        sessions.return_value.__aenter__ = AsyncMock(return_value=db_session)
        sessions.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("huddleroom.services.meeting_runner.MeetingRunner.run_next_turn", new_callable=AsyncMock) as run_turn:
            await run_meeting_turn_async(str(meeting.id))

    assert meeting.status == "active"
    run_turn.assert_not_awaited()


@pytest.mark.asyncio
async def test_claim_serializes_workspace_change_without_sqlite_snapshot_error(concurrent_sessions, tmp_path: Path):
    claim_db, update_db = concurrent_sessions
    old_workspace = tmp_path / "old"
    new_workspace = tmp_path / "new"
    old_workspace.mkdir()
    new_workspace.mkdir()
    project = Project(name="Project", workspace_path=str(old_workspace.resolve()), config={})
    claim_db.add(project)
    await claim_db.flush()
    meeting = Meeting(project_id=project.id, title="Scheduled", meeting_type="decision", status="scheduled")
    claim_db.add(meeting)
    await claim_db.commit()

    claimed = await MeetingService().claim_scheduled_meeting(claim_db, meeting.id)
    assert claimed is not None and claimed.status == "preparing"

    boundary_attempted = asyncio.Event()

    def signal_workspace_boundary(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.startswith("UPDATE projects SET id=projects.id"):
            boundary_attempted.set()

    engine = update_db.bind
    event.listen(engine.sync_engine, "before_cursor_execute", signal_workspace_boundary)
    update = None
    update_error = None
    primary_error = None
    try:
        update = asyncio.create_task(
            ProjectService().update(update_db, project.id, ProjectUpdate(workspace_path=str(new_workspace)))
        )
        await asyncio.wait_for(boundary_attempted.wait(), timeout=1)
        assert not update.done()

        await claim_db.commit()
        await asyncio.wait({update})
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if update is not None:
            if not update.done():
                update.cancel()
            try:
                await update
            except asyncio.CancelledError:
                if primary_error is None:
                    raise
            except Exception as exc:
                if primary_error is None:
                    update_error = exc
        event.remove(engine.sync_engine, "before_cursor_execute", signal_workspace_boundary)

    if not isinstance(update_error, HTTPException):
        if update_error is not None:
            raise update_error
        raise AssertionError("Workspace update did not raise HTTPException")
    assert update_error.status_code == 409
    assert "database is locked" not in str(update_error)


@pytest.mark.asyncio
async def test_claim_rechecks_status_after_boundary_when_session_preloaded_meeting(
    concurrent_sessions, tmp_path: Path
):
    first_db, second_db = concurrent_sessions
    project = Project(name="Project", config={})
    await _make_runnable(project, tmp_path)
    first_db.add(project)
    await first_db.flush()
    meeting = Meeting(project_id=project.id, title="Scheduled", meeting_type="decision", status="scheduled")
    first_db.add(meeting)
    await first_db.commit()

    stale_meeting = await second_db.get(Meeting, meeting.id)
    assert stale_meeting is not None and stale_meeting.status == "scheduled"
    await second_db.commit()

    first_service = MeetingService()
    original_transition = first_service.transition_to_preparing
    first_ready_to_commit = asyncio.Event()
    release_first_claim = asyncio.Event()
    boundary_reached = asyncio.Event()

    async def hold_first_transition(db, claimed_meeting):
        first_ready_to_commit.set()
        await release_first_claim.wait()
        await original_transition(db, claimed_meeting)

    def signal_claim_boundary(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.startswith("UPDATE projects SET id=projects.id"):
            boundary_reached.set()

    engine = second_db.bind
    first_claim_task = None
    second_claim_task = None
    primary_error = None
    listener_installed = False
    try:
        first_service.transition_to_preparing = hold_first_transition
        first_claim_task = asyncio.create_task(first_service.claim_scheduled_meeting(first_db, meeting.id))
        await asyncio.wait_for(first_ready_to_commit.wait(), timeout=1)

        event.listen(engine.sync_engine, "before_cursor_execute", signal_claim_boundary)
        listener_installed = True
        second_claim_task = asyncio.create_task(MeetingService().claim_scheduled_meeting(second_db, meeting.id))
        await asyncio.wait_for(boundary_reached.wait(), timeout=1)
        assert not second_claim_task.done()

        release_first_claim.set()
        first_claim = await asyncio.wait_for(first_claim_task, timeout=1)
        assert first_claim is not None and first_claim.status == "preparing"
        await first_db.commit()
        second_claim = await asyncio.wait_for(second_claim_task, timeout=1)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        release_first_claim.set()
        tasks = [task for task in (first_claim_task, second_claim_task) if task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=1)
            except TimeoutError:
                if primary_error is None:
                    raise
        if listener_installed:
            event.remove(engine.sync_engine, "before_cursor_execute", signal_claim_boundary)

    assert second_claim is None
    assert stale_meeting.status == "preparing"
    await second_db.commit()
    transitions = list(
        (
            await first_db.execute(
                select(MeetingEvent).where(
                    MeetingEvent.meeting_id == meeting.id,
                    MeetingEvent.event_type == "state_transition",
                )
            )
        ).scalars()
    )
    assert [transition.payload for transition in transitions] == [{"from": "scheduled", "to": "preparing"}]
