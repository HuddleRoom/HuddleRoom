import asyncio
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from huddleroom.adapters.cli_adapter import CliAdapter
from huddleroom.models.agent import Agent
from huddleroom.models.event_log import EventLog
from huddleroom.models.project import Project
from huddleroom.models.session import Session
from huddleroom.services.project_service import ProjectService


async def _create_cli_session(test_engine, tmp_path, script_body: str, timeout: float = 5):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script = tmp_path / "agent.py"
    script.write_text(f"#!{sys.executable}\n{script_body}")
    script.chmod(0o755)
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory.begin() as db:
        project = Project(name=f"Project-{uuid.uuid4()}", workspace_path=str(workspace), config={})
        agent = Agent(
            name=f"agent-{uuid.uuid4()}", role="reviewer", provider="openai", model="gpt-4o-mini",
            adapter_type="cli", capabilities=[], config={"cli_runtime": "custom", "script_path": str(script)},
        )
        db.add_all([project, agent])
        await db.flush()
        session = Session(
            agent_id=agent.id,
            project_id=project.id,
            adapter_type="cli",
            status="pending",
            metadata_={"_run_config": {"timeout": timeout}},
        )
        db.add(session)
        await db.flush()
        return session_factory, session.id, project.id


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
async def test_repair_runtime_failure_terminalizes_running_cli_session(test_engine, tmp_path, monkeypatch):
    """A deadline/repair failure after launch is persisted instead of leaving the claim running."""
    session_factory, session_id, _ = await _create_cli_session(
        test_engine, tmp_path, "print('bad envelope')",
    )
    async with session_factory.begin() as db:
        session = await db.get(Session, session_id)
        session.metadata_ = {
            "_run_config": {
                "timeout": 5,
                "_roadmap_budget_enforced": True,
                "_roadmap_prior_usage": {"max_turns": "1", "max_hours": "0"},
            },
        }
        agent = await db.get(Agent, session.agent_id)
        agent.config = {"cli_runtime": "claude_code"}

    script = tmp_path / "agent.py"
    monkeypatch.setattr(CliAdapter, "_build_command", staticmethod(lambda *_args, **_kwargs: [str(script)]))

    async def repair_failure(**_kwargs):
        raise RuntimeError("resume deadline exhausted")

    monkeypatch.setattr("huddleroom.adapters.cli_adapter.cli_complete_with_repair", repair_failure)
    async with session_factory() as db:
        await CliAdapter().run(session_id, db)
        session = await db.get(Session, session_id)

    assert session.status == "failed"
    assert session.resumable is True
    assert session.ended_at is not None
    assert session.metadata_["_roadmap_turn_count"] == 2


@pytest.mark.asyncio
async def test_terminate_process_group_escalates_to_kill(monkeypatch):
    """A SIGTERM-resistant process group is escalated without an unbounded wait."""
    class StubbornProc:
        pid = 123
        returncode = None

        async def wait(self):
            await asyncio.Event().wait()

    signals = []
    monkeypatch.setattr(os, "killpg", lambda _pgid, sig: signals.append(sig))

    await CliAdapter._terminate_process_group(StubbornProc(), grace_seconds=0.01)

    assert [sig for sig in signals if sig] == [signal.SIGTERM, signal.SIGKILL]


@pytest.mark.asyncio
async def test_timeout_kills_pipe_owning_child_after_group_leader_exits(test_engine, tmp_path, monkeypatch):
    child_pid_path = tmp_path / "child.pid"
    child_command = (
        "import os, pathlib, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    session_factory, session_id, project_id = await _create_cli_session(
        test_engine,
        tmp_path,
        (
            "import subprocess, sys\n"
            f"subprocess.Popen([sys.executable, '-c', {child_command!r}])\n"
        ),
        timeout=0.5,
    )

    original_launch = asyncio.create_subprocess_exec
    launched_proc = None

    async def launch_after_child_is_ready(*args, **kwargs):
        nonlocal launched_proc
        launched_proc = await original_launch(*args, **kwargs)
        await _wait_for_path(child_pid_path)
        async with asyncio.timeout(2):
            while launched_proc.returncode is None:
                await asyncio.sleep(0.01)
        return launched_proc

    monkeypatch.setattr("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", launch_after_child_is_ready)
    try:
        async with session_factory() as db:
            await asyncio.wait_for(CliAdapter().run(session_id, db), timeout=5)
            session = await db.get(Session, session_id)
            events = (
                await db.execute(select(EventLog.event_type).where(EventLog.project_id == project_id))
            ).scalars().all()

        child_pid = int(child_pid_path.read_text())
        await _assert_process_exited(child_pid)
        assert session.status == "failed"
        assert session.error == "timeout"
        assert session.ended_at is not None
        assert session.resumable is True  # SPR #85: timeout branch is resumable
        assert events.count("session.failed") == 1
    finally:
        if launched_proc is not None:
            await CliAdapter._terminate_process_group(launched_proc, grace_seconds=0.1)
        if child_pid_path.exists():
            _kill_process_if_running(int(child_pid_path.read_text()))


@pytest.mark.asyncio
async def test_external_task_cancellation_kills_child_and_terminalizes_session(test_engine, tmp_path):
    child_pid_path = tmp_path / "child.pid"
    session_factory, session_id, project_id = await _create_cli_session(
        test_engine,
        tmp_path,
        (
            "import pathlib, subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid))\n"
            "time.sleep(60)\n"
        ),
    )

    async with session_factory() as db:
        run_task = asyncio.create_task(CliAdapter().run(session_id, db))
        await _wait_for_path(child_pid_path)
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run_task, timeout=3)
        session = await db.get(Session, session_id)
        events = (
            await db.execute(select(EventLog.event_type).where(EventLog.project_id == project_id))
        ).scalars().all()

    child_pid = int(child_pid_path.read_text())
    try:
        await _assert_process_exited(child_pid)
        assert session.status == "cancelled"
        assert session.error is None
        assert session.ended_at is not None
        assert events.count("session.cancelled") == 1
    finally:
        _kill_process_if_running(child_pid)


@pytest.mark.asyncio
async def test_reset_watcher_kills_child_and_terminalizes_session(test_engine, tmp_path):
    child_pid_path = tmp_path / "child.pid"
    session_factory, session_id, project_id = await _create_cli_session(
        test_engine,
        tmp_path,
        (
            "import pathlib, subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid))\n"
            "time.sleep(60)\n"
        ),
    )

    async with session_factory() as db:
        run_task = asyncio.create_task(CliAdapter().run(session_id, db))
        await _wait_for_path(child_pid_path)
        async with session_factory.begin() as reset_db:
            project = await reset_db.get(Project, project_id)
            project.status = "resetting"
        await asyncio.wait_for(run_task, timeout=3)
        session = await db.get(Session, session_id)
        events = (
            await db.execute(select(EventLog.event_type).where(EventLog.project_id == project_id))
        ).scalars().all()

    child_pid = int(child_pid_path.read_text())
    try:
        await _assert_process_exited(child_pid)
        assert session.status == "cancelled"
        assert session.error is None
        assert session.ended_at is not None
        assert events.count("session.cancelled") == 1
    finally:
        _kill_process_if_running(child_pid)


@pytest.mark.asyncio
async def test_watcher_failure_fails_session_instead_of_reporting_reset(test_engine, tmp_path, monkeypatch):
    session_factory, session_id, project_id = await _create_cli_session(
        test_engine,
        tmp_path,
        "import time\ntime.sleep(60)\n",
    )

    async def fail_watcher(*_args, **_kwargs):
        raise RuntimeError("watcher database failed")

    monkeypatch.setattr(CliAdapter, "_watch_project_status", fail_watcher)
    async with session_factory() as db:
        await asyncio.wait_for(CliAdapter().run(session_id, db), timeout=3)
        session = await db.get(Session, session_id)
        events = (
            await db.execute(select(EventLog.event_type).where(EventLog.project_id == project_id))
        ).scalars().all()

    assert session.status == "failed"
    assert session.error == "project_status_watcher_failed: watcher database failed"
    assert session.ended_at is not None
    assert session.resumable is True  # SPR #85: watcher-failure branch is resumable
    assert events.count("session.failed") == 1


@pytest.mark.asyncio
async def test_project_status_watcher_stops_on_resetting(test_engine):
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory.begin() as db:
        project = Project(name=f"Project-{uuid.uuid4()}", workspace_path="/tmp", config={})
        db.add(project)
        await db.flush()
        project_id = project.id

    async with session_factory() as db:
        watcher = asyncio.create_task(CliAdapter._watch_project_status(db, project_id, interval=0.01))
        async with session_factory.begin() as reset_db:
            project = await reset_db.get(Project, project_id)
            project.status = "resetting"
        assert await asyncio.wait_for(watcher, timeout=1) == "resetting"


@pytest.mark.asyncio
async def test_terminate_process_group_leaves_no_orphan_child():
    """The child launched by a CLI process is signalled with its parent."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        (
            "import subprocess, sys, time; "
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "print(child.pid, flush=True); time.sleep(60)"
        ),
        stdout=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    child_pid = int((await asyncio.wait_for(proc.stdout.readline(), timeout=1)).decode())
    try:
        await CliAdapter._terminate_process_group(proc, grace_seconds=1)
        for _ in range(20):
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("child process survived process-group termination")
    finally:
        await CliAdapter._terminate_process_group(proc, grace_seconds=0.1)


def test_codex_command_is_bound_to_workspace(tmp_path):
    workspace = tmp_path / "workspace"

    assert CliAdapter._build_command("codex", "review this", {}, workspace, workspace / "task.md", model="gpt-4o") == [
        "codex", "exec", "--sandbox", "workspace-write", "--cd", str(workspace),
        "--ask-for-approval", "never", "--model", "gpt-4o", "review this",
    ]


@pytest.mark.parametrize(
    ("runtime", "resume_args", "required_args"),
    [
        ("copilot", ("--resume=session-123",), ("-C", "--no-ask-user", "--allow-all-tools", "--output-format", "json")),
        ("opencode", ("--session", "session-123"), ("--auto", "--dir", "--format", "json")),
        ("pi", ("--session", "session-123"), ("--print", "--mode", "json", "--no-approve")),
    ],
)
def test_step8_cli_commands_are_noninteractive_and_bound_to_workspace(
    tmp_path, runtime, resume_args, required_args,
):
    """Each Step 8 runtime gets a noninteractive JSON command scoped to its workspace."""
    workspace = tmp_path / "workspace"
    command = CliAdapter._build_command(
        runtime,
        "inspect this workspace",
        {},
        workspace,
        workspace / "task.md",
        existing_session_id="session-123",
        model="openrouter/test-model",
    )

    assert command is not None
    assert command[0] == runtime
    for arg in required_args:
        assert arg in command
    assert command[command.index("--model") + 1] == "openrouter/test-model"
    assert tuple(command[command.index(resume_args[0]):command.index(resume_args[0]) + len(resume_args)]) == resume_args
    if runtime == "copilot":
        assert command[command.index("-C") + 1] == str(workspace)
    elif runtime == "opencode":
        assert command[command.index("--dir") + 1] == str(workspace)


@pytest.mark.parametrize(
    ("runtime", "included", "excluded"),
    [
        ("copilot", {"GITHUB_TOKEN", "ONECLI_GATEWAY"}, {"OPENROUTER_API_KEY"}),
        ("opencode", {"OPENROUTER_API_KEY", "ONECLI_GATEWAY"}, {"GITHUB_TOKEN"}),
        ("pi", {"OPENROUTER_API_KEY", "ONECLI_GATEWAY"}, {"GITHUB_TOKEN"}),
        ("codex", set(), {"GITHUB_TOKEN", "OPENROUTER_API_KEY", "ONECLI_GATEWAY"}),
    ],
)
def test_step8_runtime_env_scopes_credentials_and_onecli_config(monkeypatch, runtime, included, excluded):
    """Only named Step 8 runtimes receive their credentials and onecli transport settings."""
    for key, value in {
        "GITHUB_TOKEN": "github-secret",
        "OPENROUTER_API_KEY": "openrouter-secret",
        "ONECLI_GATEWAY": "http://127.0.0.1:8000",
        "ONECLI_GATEWAY_SKILL_PATH": "/tmp/skills",
        "HTTP_PROXY": "http://127.0.0.1:8080",
        "HTTPS_PROXY": "http://127.0.0.1:8080",
        "NO_PROXY": "localhost",
        "NODE_USE_ENV_PROXY": "1",
        "NODE_EXTRA_CA_CERTS": "/tmp/onecli-ca.pem",
    }.items():
        monkeypatch.setenv(key, value)
    agent = Agent(
        name=f"agent-{uuid.uuid4()}", role="reviewer", provider="openai", model="test",
        adapter_type="cli", capabilities=[], config={"cli_runtime": runtime},
    )
    project = Project(name=f"Project-{uuid.uuid4()}", workspace_path="/tmp", config={})

    env = CliAdapter()._build_env(uuid.uuid4(), None, agent, project)

    for key in included:
        assert key in env
    for key in excluded:
        assert key not in env
    if runtime in {"copilot", "opencode", "pi"}:
        assert {
            "ONECLI_GATEWAY_SKILL_PATH", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
            "NODE_USE_ENV_PROXY", "NODE_EXTRA_CA_CERTS",
        } <= env.keys()


@pytest.mark.parametrize("runtime, flag", [
    ("claude_code", "--model"),
    ("codex", "--model"),
    ("aider", "--model"),
    ("custom", None),
])
@pytest.mark.parametrize("agent_model, override, expected", [
    ("agent-model", None, "agent-model"),
    ("agent-model", "override-model", "override-model"),
])
def test_cli_command_uses_effective_model(runtime, flag, agent_model, override, expected, tmp_path):
    model = override or agent_model
    task_path = tmp_path / "task.md"
    config = {"script_path": "/tmp/custom-agent"} if runtime == "custom" else {}
    command = CliAdapter._build_command(runtime, "work", config, tmp_path, task_path, model=model)

    if flag:
        assert command[command.index(flag) + 1] == expected
    else:
        assert command == ["/tmp/custom-agent", str(task_path)]


@pytest.mark.asyncio
async def test_unknown_runtime_fails_session_without_launch(test_engine, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as db:
        project = Project(name=f"Project-{uuid.uuid4()}", workspace_path=str(workspace), config={})
        agent = Agent(
            name=f"agent-{uuid.uuid4()}", role="reviewer", provider="openai", model="gpt-4o-mini",
            adapter_type="cli", capabilities=[], config={"cli_runtime": "unknown-runtime"},
        )
        db.add_all([project, agent])
        await db.flush()
        session = Session(agent_id=agent.id, project_id=project.id, adapter_type="cli", status="pending", metadata_={})
        db.add(session)
        await db.commit()

        exchange = Mock()
        with patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", new=AsyncMock()) as launch:
            with patch("huddleroom.adapters.cli_adapter.log_cli_exchange", exchange):
                await CliAdapter().run(session.id, db)

        await db.refresh(session)
        failure = (
            await db.execute(
                select(EventLog).where(
                    EventLog.project_id == project.id,
                    EventLog.event_type == "session.failed",
                )
            )
        ).scalar_one()

    assert session.status == "failed"
    assert session.error == "unsupported_cli_runtime: unknown-runtime"
    assert session.ended_at is not None
    assert failure.payload["error"] == session.error
    launch.assert_not_awaited()
    exchange.assert_called_once_with(
        prompt="",
        session_id=None,
        error="unsupported_cli_runtime: unknown-runtime",
        runtime="unknown-runtime",
        rally_session_id=str(session.id),
    )


@pytest.mark.asyncio
async def test_custom_runtime_reads_task_from_rally_context(test_engine, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script = (Path(__file__).parent / "fixtures" / "echo_agent.sh").resolve()
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as db:
        project = Project(name=f"Project-{uuid.uuid4()}", workspace_path=str(workspace), config={})
        agent = Agent(
            name=f"agent-{uuid.uuid4()}", role="reviewer", provider="openai", model="gpt-4o-mini",
            adapter_type="cli", capabilities=[], config={"cli_runtime": "custom", "script_path": str(script)},
        )
        db.add_all([project, agent])
        await db.flush()
        session = Session(agent_id=agent.id, project_id=project.id, adapter_type="cli", status="pending", metadata_={})
        db.add(session)
        await db.commit()

        await CliAdapter().run(session.id, db)
        await db.refresh(session)

        context_dir = workspace / ".huddleroom" / "agents" / str(agent.id) / "sessions" / str(session.id)
        assert (context_dir / "task.md").is_file()
        assert (context_dir / "huddleroom_context.json").is_file()
        assert session.status == "completed"
        assert session.output == "# No task context\nTask completed successfully\n"


@pytest.mark.asyncio
async def test_launch_revalidation_failure_terminalizes_running_session(test_engine, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as db:
        project = Project(name=f"Project-{uuid.uuid4()}", workspace_path=str(workspace), config={})
        agent = Agent(
            name=f"agent-{uuid.uuid4()}", role="reviewer", provider="openai", model="gpt-4o-mini",
            adapter_type="cli", capabilities=[], config={"cli_runtime": "claude_code"},
        )
        db.add_all([project, agent])
        await db.flush()
        session = Session(agent_id=agent.id, project_id=project.id, adapter_type="cli", status="pending", metadata_={})
        db.add(session)
        await db.commit()

        validation = AsyncMock(side_effect=[
            workspace,
            HTTPException(
                status_code=409,
                detail={"code": "project_not_runnable", "reason": "workspace_unavailable"},
            ),
        ])
        with patch.object(ProjectService, "require_runnable_project", new=validation):
            with patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", new=AsyncMock()) as launch:
                await CliAdapter().run(session.id, db)

        await db.refresh(session)
        failure = (
            await db.execute(
                select(EventLog).where(
                    EventLog.project_id == project.id,
                    EventLog.event_type == "session.failed",
                )
            )
        ).scalar_one()
        assert session.status == "failed"
        assert session.error == "project_not_runnable: workspace_unavailable"
        assert session.ended_at is not None
        assert failure.payload["error"] == session.error
        launch.assert_not_awaited()


@pytest.mark.asyncio
async def test_initial_workspace_validation_terminalizes_pending_session(test_engine, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as db:
        project = Project(name=f"Project-{uuid.uuid4()}", workspace_path=str(workspace), config={})
        agent = Agent(
            name=f"agent-{uuid.uuid4()}", role="reviewer", provider="openai", model="gpt-4o-mini",
            adapter_type="cli", capabilities=[], config={"cli_runtime": "claude_code"},
        )
        db.add_all([project, agent])
        await db.flush()
        session = Session(agent_id=agent.id, project_id=project.id, adapter_type="cli", status="pending", metadata_={})
        db.add(session)
        await db.commit()

        validation = AsyncMock(side_effect=HTTPException(
            status_code=409, detail={"code": "project_not_runnable", "reason": "workspace_unavailable"},
        ))
        with patch.object(ProjectService, "require_runnable_project", new=validation):
            await CliAdapter().run(session.id, db)

        await db.refresh(session)
        assert session.status == "failed"
        assert session.error == "project_not_runnable: workspace_unavailable"
        assert session.ended_at is not None


@pytest.mark.asyncio
async def test_roadmap_workspace_requires_distinct_registered_git_worktree(test_engine, tmp_path):
    workspace = tmp_path / "repo"
    worktree = tmp_path / "worktree"
    workspace.mkdir()
    for command in (("git", "init"), ("git", "config", "user.email", "test@example.com"),
                    ("git", "config", "user.name", "Test")):
        subprocess.run(command, cwd=workspace, check=True, capture_output=True)
    (workspace / "README.md").write_text("test\n")
    subprocess.run(("git", "add", "README.md"), cwd=workspace, check=True, capture_output=True)
    subprocess.run(("git", "commit", "-m", "initial"), cwd=workspace, check=True, capture_output=True)
    subprocess.run(("git", "worktree", "add", "--detach", str(worktree)), cwd=workspace, check=True, capture_output=True)
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory.begin() as db:
        project = Project(
            name=f"Project-{uuid.uuid4()}", workspace_path=str(workspace),
            config={"roadmap_worktrees": {"isolated": str(worktree)}},
        )
        db.add(project)
        await db.flush()
        assert await ProjectService().require_roadmap_workspace(
            db, project.id, {"type": "git_worktree", "identifier": "isolated", "reversible": True},
        ) == worktree.resolve()
        project.config = {"roadmap_worktrees": {"shared": str(workspace)}}
        with pytest.raises(HTTPException, match="must differ"):
            await ProjectService().require_roadmap_workspace(
                db, project.id, {"type": "git_worktree", "identifier": "shared", "reversible": True},
            )
