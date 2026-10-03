import asyncio
import importlib
import sys
import types
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from huddleroom.models.agent import Agent
from huddleroom.models.project import Project
from huddleroom.models.session import Session
from huddleroom.schemas.session import SessionCreate
from huddleroom.services.project_reset_service import ProjectResetService
from huddleroom.services.project_service import ProjectService
from huddleroom.services.session_service import SessionService
from huddleroom.workers import session_tasks
from huddleroom.workers import task_runner


@pytest.fixture
def registered_session_tasks():
    """Register the optional Celery wrappers without requiring a broker or worker."""
    class FakeCeleryApp:
        def __init__(self):
            self.tasks = {}

        def task(self, **options):
            def register(func):
                self.tasks[options["name"]] = func
                return func

            return register

    app = FakeCeleryApp()
    celery_app = types.ModuleType("huddleroom.workers.celery_app")
    celery_app.app = app
    try:
        with patch.dict(sys.modules, {"huddleroom.workers.celery_app": celery_app}):
            importlib.reload(session_tasks)
            yield app.tasks
    finally:
        for name in ("run_api_session", "run_cli_session"):
            session_tasks.__dict__.pop(name, None)
        importlib.reload(session_tasks)


@pytest.mark.parametrize(
    ("registered_name", "execute_name"),
    [
        ("rally.workers.session_tasks.run_api_session", "execute_api_session"),
        ("rally.workers.session_tasks.run_cli_session", "execute_cli_session"),
    ],
)
@pytest.mark.unsupported_mode
def test_celery_session_task_propagates_project_conflict_without_retry(
    registered_session_tasks, registered_name, execute_name
):
    """Bypassing the shared runner or requesting retry would resurrect rejected queued work."""
    conflict = HTTPException(
        status_code=409,
        detail={"code": "project_not_runnable", "reason": "workspace_unset"},
    )
    execute = AsyncMock(side_effect=conflict)
    task = SimpleNamespace(retry=MagicMock(), request=SimpleNamespace(id="runner-id"))

    with patch.object(session_tasks, execute_name, new=execute):
        with pytest.raises(HTTPException) as exc_info:
            registered_session_tasks[registered_name](task, "session-id")

    assert exc_info.value is conflict
    execute.assert_awaited_once_with("session-id", "runner-id")
    task.retry.assert_not_called()


@pytest.mark.asyncio
async def test_create_rejects_unrunnable_project_before_persisting_session(db_session, test_project, test_agent):
    """Removing the launch guard would leave a pending session for an unusable workspace."""
    test_project.workspace_path = None
    with pytest.raises(HTTPException) as exc_info:
        await SessionService().create(
            db_session,
            SessionCreate(agent_id=test_agent.id, project_id=test_project.id),
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == {"code": "project_not_runnable", "reason": "workspace_unset"}
    assert not (await db_session.execute(select(Session))).scalars().all()


@pytest.mark.asyncio
async def test_session_start_before_reset_fence_is_registered_then_purged(
    test_engine, monkeypatch, tmp_path,
):
    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    async with sessions() as setup_db:
        project = Project(name="Race", workspace_path=str(workspace), config={})
        agent = Agent(
            name="race-agent",
            role="developer",
            provider="openai",
            model="test",
            adapter_type="api",
            capabilities=[],
            config={},
        )
        setup_db.add_all([project, agent])
        await setup_db.commit()
        project_id = project.id
        agent_id = agent.id

    create_db = sessions()
    reset_db = sessions()
    service = SessionService()
    guard_passed = asyncio.Event()
    release_create = asyncio.Event()
    release_delayed_dispatch = asyncio.Event()
    reset_lock_acquired = asyncio.Event()
    reset_completed = asyncio.Event()
    post_reset_dispatch = asyncio.Event()
    original_require = ProjectService.require_runnable_project
    original_lock = ProjectService.lock_workspace_boundary
    original_dispatch = task_runner.dispatch_session

    async def pause_after_guard(project_service, db, project_id):
        workspace_path = await original_require(project_service, db, project_id)
        if db is create_db:
            guard_passed.set()
            await release_create.wait()
        return workspace_path

    async def track_reset_lock(project_service, db, project_id):
        result = await original_lock(project_service, db, project_id)
        if db is reset_db:
            reset_lock_acquired.set()
        return result

    async def delay_old_after_commit_dispatch(*args):
        await release_delayed_dispatch.wait()
        await original_dispatch(*args)

    async def execute_session(_session_id, _runner_task_id):
        if reset_completed.is_set():
            post_reset_dispatch.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(ProjectService, "require_runnable_project", pause_after_guard)
    monkeypatch.setattr(ProjectService, "lock_workspace_boundary", track_reset_lock)
    monkeypatch.setattr(task_runner, "dispatch_session", delay_old_after_commit_dispatch)
    monkeypatch.setattr(task_runner, "execute_api_session", execute_session)

    create_task = None
    reset_task = None
    try:
        async def create_and_commit():
            await service.create(
                create_db,
                SessionCreate(agent_id=agent_id, project_id=project_id),
            )
            await create_db.commit()

        async def reset_project():
            try:
                await ProjectResetService().reset(reset_db, project_id, "Race")
            finally:
                reset_completed.set()

        create_task = asyncio.create_task(create_and_commit())
        await asyncio.wait_for(guard_passed.wait(), timeout=1)
        reset_task = asyncio.create_task(reset_project())

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(reset_lock_acquired.wait(), timeout=0.05)

        release_create.set()
        await asyncio.wait_for(asyncio.gather(create_task, reset_task), timeout=2)
        release_delayed_dispatch.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        async with sessions() as observer_db:
            assert not (await observer_db.execute(select(Session))).scalars().all()
        assert not post_reset_dispatch.is_set()
        assert not task_runner.get_running_tasks()
    finally:
        release_create.set()
        release_delayed_dispatch.set()
        for pending in (create_task, reset_task):
            if pending is not None and not pending.done():
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
        for task_id in list(task_runner.get_running_tasks()):
            await task_runner.cancel_task(task_id)
        await create_db.close()
        await reset_db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("execute", "adapter_path"),
    [
        (session_tasks.execute_api_session, "huddleroom.adapters.api_adapter.ApiAdapter.run"),
        (session_tasks.execute_cli_session, "huddleroom.adapters.cli_adapter.CliAdapter.run"),
    ],
)
async def test_worker_marks_unrunnable_project_failed_without_executing_adapter(
    test_engine, monkeypatch, tmp_path, execute, adapter_path
):
    """Changing a worker conflict to a normal return would hide a non-retryable rejection."""
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory.begin() as db:
        project = Project(name=f"Project-{uuid.uuid4()}", config={})
        agent = Agent(
            name=f"agent-{uuid.uuid4()}",
            role="developer",
            provider="openai",
            model="gpt-4o-mini",
            adapter_type="api",
            capabilities=[],
            config={},
        )
        db.add_all([project, agent])
        await db.flush()
        session = Session(
            agent_id=agent.id,
            project_id=project.id,
            adapter_type="api",
            status="pending",
            input_context={},
            runner_task_id="runner-id",
        )
        db.add(session)
        await db.flush()
        session_id = str(session.id)

    monkeypatch.setattr(session_tasks, "AsyncSessionLocal", session_factory)
    adapter_run = AsyncMock()
    with patch(adapter_path, new=adapter_run):
        with pytest.raises(HTTPException) as exc_info:
            await execute(session_id, "runner-id")

        assert exc_info.value.status_code == 409
        assert exc_info.value.detail == {"code": "project_not_runnable", "reason": "workspace_unset"}

        async with session_factory.begin() as db:
            project = await db.get(Project, project.id)
            project.workspace_path = str(tmp_path.resolve())
        await execute(session_id, "runner-id")

    async with session_factory() as db:
        stored_session = await db.get(Session, uuid.UUID(session_id))

    assert stored_session.status == "failed"
    assert stored_session.error == "project_not_runnable: workspace_unset"
    assert stored_session.ended_at is not None
    adapter_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_retry_runner_does_not_retry_project_not_runnable_conflict(monkeypatch):
    """Catching the conflict in the generic retry branch would requeue rejected work."""
    attempts = 0

    async def reject_once(_session_id: str, _runner_task_id: str) -> None:
        nonlocal attempts
        attempts += 1
        raise HTTPException(status_code=409, detail={"code": "project_not_runnable", "reason": "workspace_unset"})

    monkeypatch.setattr(task_runner, "execute_api_session", reject_once)

    task_id = await task_runner.retry_api_session("session-id", "project-id")
    await asyncio.wait_for(task_runner.get_running_tasks()[task_id], timeout=0.1)

    assert attempts == 1


@pytest.mark.asyncio
async def test_cancel_task_waits_for_adapter_cleanup(monkeypatch):
    """Returning before the adapter exits would leave cancelled CLI work running."""
    started = asyncio.Event()
    cleaned_up = asyncio.Event()

    async def run_until_cancelled(_session_id: str, _runner_task_id: str) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned_up.set()

    monkeypatch.setattr(task_runner, "execute_api_session", run_until_cancelled)

    task_id = await task_runner.dispatch_session("session-id", "api", "project-id")
    await started.wait()

    assert await task_runner.cancel_task(task_id)
    assert cleaned_up.is_set()
    assert task_id not in task_runner.get_running_tasks()


@pytest.mark.asyncio
async def test_worker_preserves_cancellation_that_races_with_project_validation(test_engine, monkeypatch):
    """Removing the terminal-status recheck would turn a cancellation into a workspace failure."""
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory.begin() as db:
        project = Project(name=f"Project-{uuid.uuid4()}", config={})
        agent = Agent(
            name=f"agent-{uuid.uuid4()}",
            role="developer",
            provider="openai",
            model="gpt-4o-mini",
            adapter_type="api",
            capabilities=[],
            config={},
        )
        db.add_all([project, agent])
        await db.flush()
        session = Session(
            agent_id=agent.id,
            project_id=project.id,
            adapter_type="api",
            status="pending",
            input_context={},
        )
        db.add(session)
        await db.flush()
        session_id = str(session.id)

    async def cancel_then_reject(_service, db, _project_id):
        async with session_factory.begin() as cancellation_db:
            session = await cancellation_db.get(Session, uuid.UUID(session_id))
            session.status = "cancelled"
        raise HTTPException(status_code=409, detail={"code": "project_not_runnable", "reason": "workspace_unset"})

    monkeypatch.setattr(session_tasks, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(session_tasks.ProjectService, "require_runnable_project", cancel_then_reject)
    adapter_run = AsyncMock()
    with patch("huddleroom.adapters.api_adapter.ApiAdapter.run", new=adapter_run):
        await session_tasks.execute_api_session(session_id, "runner-id")

    async with session_factory() as db:
        stored_session = await db.get(Session, uuid.UUID(session_id))

    assert stored_session.status == "cancelled"
    assert stored_session.error is None
    adapter_run.assert_not_awaited()
