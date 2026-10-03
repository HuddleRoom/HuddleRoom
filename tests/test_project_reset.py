import asyncio
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy import event, inspect, select

from huddleroom.models.agent import Agent
from huddleroom.models.artifact import Artifact, ArtifactWatcher
from huddleroom.models.api_key import ApiKey
from huddleroom.models.channel import Channel
from huddleroom.models.escalation import EscalationChain
from huddleroom.models.event_log import EventLog
from huddleroom.models.hook import Hook
from huddleroom.models.knowledge_item import KnowledgeItem
from huddleroom.models.meeting import (
    Meeting,
    MeetingActionItem,
    MeetingAgendaItem,
    MeetingDecision,
    MeetingEvent,
    MeetingParticipantSignal,
    MeetingRequest,
    MeetingTurn,
)
from huddleroom.models.memory_item import MemoryItem
from huddleroom.models.message import Message
from huddleroom.models.optimization import CostMetric, Optimization, Pattern
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationAgentSuggestion,
    OrchestrationDecision,
    OrchestrationEvidence,
    OrchestrationGate,
    OrchestrationGoal,
    OrchestrationRun,
)
from huddleroom.models.orchestration_memory import OrchestrationMemorySection
from huddleroom.models.orchestration_process import (
    OrchestrationAgentReview,
    OrchestrationAuthorityDecision,
    OrchestrationProcessRun,
    OrchestrationWarning,
)
from huddleroom.models.project import Project
from huddleroom.models.protocol import Protocol, ProtocolInstance, ProtocolTimeout, ProtocolTransition
from huddleroom.models.routing_rule import RoutingRule
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.project import ProjectUpdate
from huddleroom.schemas.session import SessionCreate
from huddleroom.services.project_service import ProjectService
from huddleroom.services.project_reset_service import ProjectResetService
from huddleroom.services.session_service import SessionService
from huddleroom.workers import task_runner
from huddleroom.workers import meeting_tasks


class FakeAdvisoryLockConnection:
    def __init__(self, unlock_result=True, unlock_error=None, unlock_started=None, unlock_blocker=None):
        self.unlock_result = unlock_result
        self.unlock_error = unlock_error
        self.unlock_started = unlock_started
        self.unlock_blocker = unlock_blocker
        self.invalidated = False
        self.returned_after_invalidation = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        self.returned_after_invalidation = self.invalidated

    async def execute(self, statement, _parameters):
        if "pg_advisory_unlock" in str(statement):
            if self.unlock_started:
                self.unlock_started.set()
            if self.unlock_blocker:
                await asyncio.wait_for(self.unlock_blocker.wait(), timeout=1)
            if self.unlock_error:
                raise self.unlock_error
        return SimpleNamespace(scalar_one=lambda: self.unlock_result)

    async def invalidate(self):
        self.invalidated = True


class FakePostgresBind:
    dialect = SimpleNamespace(name="postgresql")

    def __init__(self, connection):
        self.connection = connection

    def connect(self):
        return self.connection


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_postgres_reset_lock_invalidates_connection_when_unlock_returns_false():
    """Catches a failed advisory unlock returning a reusable pooled connection."""
    connection = FakeAdvisoryLockConnection(unlock_result=False)
    db = SimpleNamespace(bind=FakePostgresBind(connection))

    with pytest.raises(RuntimeError, match="Failed to release project reset lock"):
        async with ProjectResetService()._lock_project_reset(db, uuid.uuid4()):
            pass

    assert connection.invalidated
    assert connection.returned_after_invalidation


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_postgres_reset_lock_invalidates_connection_when_unlock_raises():
    """Catches an unlock error returning a possibly locked pooled connection."""
    connection = FakeAdvisoryLockConnection(unlock_error=RuntimeError("unlock failed"))
    db = SimpleNamespace(bind=FakePostgresBind(connection))

    with pytest.raises(RuntimeError, match="unlock failed"):
        async with ProjectResetService()._lock_project_reset(db, uuid.uuid4()):
            pass

    assert connection.invalidated
    assert connection.returned_after_invalidation


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_postgres_reset_lock_invalidates_connection_on_cancellation():
    """Catches cancellation while the advisory unlock is in flight."""
    unlock_started = asyncio.Event()
    connection = FakeAdvisoryLockConnection(
        unlock_started=unlock_started,
        unlock_blocker=asyncio.Event(),
    )
    db = SimpleNamespace(bind=FakePostgresBind(connection))

    async def lock_then_unlock():
        async with ProjectResetService()._lock_project_reset(db, uuid.uuid4()):
            pass

    lock_task = asyncio.create_task(lock_then_unlock())
    try:
        await asyncio.wait_for(unlock_started.wait(), timeout=1)
        lock_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(lock_task, timeout=1)
    finally:
        if not lock_task.done():
            lock_task.cancel()
            await asyncio.wait_for(asyncio.gather(lock_task, return_exceptions=True), timeout=1)

    assert connection.invalidated
    assert connection.returned_after_invalidation


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_postgres_reset_lock_keeps_connection_when_body_fails_after_unlocking():
    """Catches discarding a healthy connection after reset work fails normally."""
    connection = FakeAdvisoryLockConnection()
    db = SimpleNamespace(bind=FakePostgresBind(connection))

    with pytest.raises(RuntimeError, match="reset failed"):
        async with ProjectResetService()._lock_project_reset(db, uuid.uuid4()):
            raise RuntimeError("reset failed")

    assert not connection.invalidated
    assert not connection.returned_after_invalidation


@pytest.mark.asyncio
async def test_reset_requires_exact_project_name_confirmation(client: AsyncClient, test_project):
    response = await client.post(
        f"/api/v1/projects/{test_project.id}/reset",
        json={"confirm_name": f"{test_project.name} "},
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Project name confirmation does not match",
        "error": "http_error",
    }


@pytest.mark.asyncio
async def test_reset_rejects_archived_projects(client: AsyncClient, test_project, db_session):
    test_project.status = "archived"
    await db_session.flush()

    response = await client.post(
        f"/api/v1/projects/{test_project.id}/reset",
        json={"confirm_name": test_project.name},
    )

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_reset_commits_fence_before_attempting_shutdown(test_engine, monkeypatch, tmp_path):
    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    async with sessions() as setup_db:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        project = Project(name="Fence", workspace_path=str(workspace), config={})
        setup_db.add(project)
        await setup_db.commit()
        project_id = project.id

    async def failed_shutdown(_project_id):
        async with sessions() as observer_db:
            observed = await observer_db.get(Project, project_id)
            assert observed.status == "resetting"
            with pytest.raises(HTTPException, match="project_inactive"):
                await ProjectService().require_runnable_project(observer_db, project_id)
        raise RuntimeError("shutdown failed")

    monkeypatch.setattr(
        "huddleroom.services.project_reset_service.task_runner.cancel_project_sessions",
        failed_shutdown,
    )
    async with sessions() as reset_db:
        with pytest.raises(RuntimeError, match="shutdown failed"):
            await ProjectResetService().reset(reset_db, project_id, "Fence")

    async with sessions() as observer_db:
        observed = await observer_db.get(Project, project_id)
        assert observed.status == "resetting"


@pytest.mark.asyncio
async def test_reset_retries_from_resetting_and_repeats_with_zero_counts(
    test_engine,
):
    from huddleroom.database import get_db
    from huddleroom.main import create_app

    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    async with sessions() as setup_db:
        project = Project(name="Retry", status="resetting", config={})
        setup_db.add(project)
        await setup_db.flush()
        setup_db.add(EventLog(project_id=project.id, event_type="before-reset"))
        await setup_db.commit()
        project_id = project.id

    app = create_app()
    async with sessions() as request_db:
        async def override_get_db():
            yield request_db

        app.dependency_overrides[get_db] = override_get_db
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                f"/api/v1/projects/{project_id}/reset",
                json={"confirm_name": "Retry"},
            )

            assert response.status_code == 200
            assert response.json()["cancelled_sessions"] == 0
            assert response.json()["cancelled_meeting_tasks"] == 0
            assert response.json()["deletions"]["event_log"] == 1
            assert (await request_db.get(Project, project_id)).status == "active"
            assert (await request_db.execute(select(EventLog).where(EventLog.project_id == project_id))).scalars().all() == []

            repeat = await client.post(
                f"/api/v1/projects/{project_id}/reset",
                json={"confirm_name": "Retry"},
            )

    assert repeat.status_code == 200
    assert all(count == 0 for count in repeat.json()["deletions"].values())


@pytest.mark.asyncio
async def test_simultaneous_resets_wait_for_the_current_coordinator(test_engine, monkeypatch):
    """Catches a second reset crossing any late owner lifecycle boundary."""
    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    async with sessions() as setup_db:
        project = Project(name="Serialized", config={})
        setup_db.add(project)
        await setup_db.commit()
        project_id = project.id

    first_purge_started = asyncio.Event()
    allow_first_purge = asyncio.Event()
    first_purge_finished = asyncio.Event()
    first_reactivation_commit_started = asyncio.Event()
    allow_first_reactivation_commit = asyncio.Event()
    first_reactivation_commit_finished = asyncio.Event()
    second_cancellation_started = asyncio.Event()
    second_purge_started = asyncio.Event()
    second_reset_completed = asyncio.Event()
    shutdown_calls = 0
    purge_calls = 0

    async def observe_cancellation(_project_id):
        nonlocal shutdown_calls
        shutdown_calls += 1
        if shutdown_calls == 2:
            second_cancellation_started.set()
        return 0

    original_purge = ProjectService.reset

    async def block_first_purge(service, db, purge_project_id):
        nonlocal purge_calls
        purge_calls += 1
        if purge_calls == 1:
            first_purge_started.set()
            await asyncio.wait_for(allow_first_purge.wait(), timeout=1)
        else:
            second_purge_started.set()
        result = await original_purge(service, db, purge_project_id)
        if purge_calls == 1:
            first_purge_finished.set()
        return result

    monkeypatch.setattr(task_runner, "cancel_project_sessions", observe_cancellation)
    monkeypatch.setattr(ProjectService, "reset", block_first_purge)

    first_reset = None
    second_reset = None
    reset_results = []
    async with sessions() as first_db, sessions() as second_db:
        original_commit = AsyncSession.commit

        async def block_first_reactivation_commit(db):
            if db is first_db and first_purge_finished.is_set():
                first_reactivation_commit_started.set()
                await asyncio.wait_for(allow_first_reactivation_commit.wait(), timeout=1)
                result = await original_commit(db)
                first_reactivation_commit_finished.set()
                return result
            return await original_commit(db)

        monkeypatch.setattr(AsyncSession, "commit", block_first_reactivation_commit)
        try:
            first_reset = asyncio.create_task(ProjectResetService().reset(first_db, project_id, "Serialized"))
            await asyncio.wait_for(first_purge_started.wait(), timeout=1)

            async def run_second_reset():
                try:
                    return await ProjectResetService().reset(second_db, project_id, "Serialized")
                finally:
                    second_reset_completed.set()

            second_reset = asyncio.create_task(run_second_reset())
            allow_first_purge.set()
            await asyncio.wait_for(first_reactivation_commit_started.wait(), timeout=1)
            try:
                await asyncio.wait_for(second_cancellation_started.wait(), timeout=0.1)
                second_cancelled_early = True
            except TimeoutError:
                second_cancelled_early = False
            try:
                await asyncio.wait_for(second_purge_started.wait(), timeout=0.1)
                second_purged_early = True
            except TimeoutError:
                second_purged_early = False
            second_reactivated_early = second_reset_completed.is_set()
        finally:
            allow_first_purge.set()
            allow_first_reactivation_commit.set()
            for reset_task in (first_reset, second_reset):
                if reset_task is None:
                    continue
                try:
                    reset_results.append(await asyncio.wait_for(asyncio.shield(reset_task), timeout=1))
                except TimeoutError:
                    reset_task.cancel()
                    reset_results.extend(
                        await asyncio.wait_for(asyncio.gather(reset_task, return_exceptions=True), timeout=1)
                    )

    assert not any(isinstance(result, BaseException) for result in reset_results)
    assert not second_cancelled_early
    assert not second_purged_early
    assert not second_reactivated_early
    assert first_reactivation_commit_finished.is_set()
    assert shutdown_calls == 2
    assert second_purge_started.is_set()
    assert second_reset_completed.is_set()


@pytest.mark.asyncio
async def test_waiting_reset_refreshes_stale_state_after_owner_reactivates(test_engine, monkeypatch):
    """Catches a queued reset retaining `resetting` after the owner commits `active`."""
    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    async with sessions() as setup_db:
        project = Project(name="Refresh", config={})
        setup_db.add(project)
        await setup_db.commit()
        project_id = project.id

    first_shutdown_started = asyncio.Event()
    allow_first_shutdown = asyncio.Event()
    shutdown_calls = 0
    stale_project = None
    second_previous_status = None
    second_lock_attempted = asyncio.Event()

    async def observe_second_shutdown(_project_id):
        nonlocal shutdown_calls
        shutdown_calls += 1
        if shutdown_calls == 1:
            first_shutdown_started.set()
            await asyncio.wait_for(allow_first_shutdown.wait(), timeout=1)
        return 0

    monkeypatch.setattr(task_runner, "cancel_project_sessions", observe_second_shutdown)
    original_lock_boundary = ProjectService.lock_workspace_boundary

    async def preserve_waiting_session_cache(project_service, db, locked_project_id):
        if db is second_db:
            return None
        return await original_lock_boundary(project_service, db, locked_project_id)

    monkeypatch.setattr(ProjectService, "lock_workspace_boundary", preserve_waiting_session_cache)
    original_reset_lock = ProjectResetService._lock_project_reset

    @asynccontextmanager
    async def record_second_lock_attempt(service, db, locked_project_id):
        if db is second_db:
            second_lock_attempted.set()
        async with original_reset_lock(service, db, locked_project_id):
            yield

    monkeypatch.setattr(ProjectResetService, "_lock_project_reset", record_second_lock_attempt)

    def record_second_status(target, _value, oldvalue, _initiator):
        nonlocal second_previous_status
        if target is stale_project and _value == "resetting":
            second_previous_status = oldvalue

    event.listen(Project.status, "set", record_second_status)
    first_reset = None
    second_reset = None
    reset_results = []
    async with sessions() as first_db, sessions() as second_db:
        try:
            first_reset = asyncio.create_task(ProjectResetService().reset(first_db, project_id, "Refresh"))
            await asyncio.wait_for(first_shutdown_started.wait(), timeout=1)
            stale_project = await second_db.get(Project, project_id)
            assert stale_project.status == "resetting"
            await second_db.commit()
            assert (await second_db.get(Project, project_id)).status == "resetting"
            second_reset = asyncio.create_task(ProjectResetService().reset(second_db, project_id, "Refresh"))
            await asyncio.wait_for(second_lock_attempted.wait(), timeout=1)
        finally:
            allow_first_shutdown.set()
            for reset_task in (first_reset, second_reset):
                if reset_task is None:
                    continue
                try:
                    reset_results.append(await asyncio.wait_for(asyncio.shield(reset_task), timeout=1))
                except TimeoutError:
                    reset_task.cancel()
                    reset_results.extend(
                        await asyncio.wait_for(asyncio.gather(reset_task, return_exceptions=True), timeout=1)
                    )
            event.remove(Project.status, "set", record_second_status)

    assert not any(isinstance(result, BaseException) for result in reset_results)
    assert second_previous_status == "active"


@pytest.mark.asyncio
async def test_workspace_update_is_rejected_after_the_reset_fence(test_engine, monkeypatch, tmp_path):
    """Catches a workspace change that slips through while reset shutdown is running."""
    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    workspace = tmp_path / "workspace"
    replacement_workspace = tmp_path / "replacement-workspace"
    workspace.mkdir()
    replacement_workspace.mkdir()
    async with sessions() as setup_db:
        project = Project(name="Workspace fence", workspace_path=str(workspace), config={})
        setup_db.add(project)
        await setup_db.commit()
        project_id = project.id

    shutdown_started = asyncio.Event()
    allow_shutdown = asyncio.Event()

    async def block_shutdown(_project_id):
        shutdown_started.set()
        await asyncio.wait_for(allow_shutdown.wait(), timeout=1)
        return 0

    monkeypatch.setattr(task_runner, "cancel_project_sessions", block_shutdown)

    reset_task = None
    reset_results = []
    async with sessions() as reset_db, sessions() as update_db:
        try:
            reset_task = asyncio.create_task(ProjectResetService().reset(reset_db, project_id, "Workspace fence"))
            await asyncio.wait_for(shutdown_started.wait(), timeout=1)
            try:
                await asyncio.wait_for(
                    ProjectService().update(
                        update_db,
                        project_id,
                        ProjectUpdate(workspace_path=str(replacement_workspace)),
                    ),
                    timeout=1,
                )
                workspace_update_rejected = False
            except HTTPException as exc:
                workspace_update_rejected = exc.status_code == 409
        finally:
            await asyncio.wait_for(update_db.rollback(), timeout=1)
            allow_shutdown.set()
            if reset_task is not None:
                try:
                    reset_results.append(await asyncio.wait_for(asyncio.shield(reset_task), timeout=1))
                except TimeoutError:
                    reset_task.cancel()
                    reset_results.extend(
                        await asyncio.wait_for(asyncio.gather(reset_task, return_exceptions=True), timeout=1)
                    )

    assert not any(isinstance(result, BaseException) for result in reset_results)
    assert workspace_update_rejected
    async with sessions() as observer_db:
        assert (await observer_db.get(Project, project_id)).workspace_path == str(workspace)


@pytest.mark.asyncio
async def test_concurrent_session_start_after_reset_fence_is_rejected_without_dispatch(
    test_engine, monkeypatch, tmp_path,
):
    """Catches a start creating or dispatching work after reset has fenced the project."""
    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    async with sessions() as setup_db:
        project = Project(name="Start fence", workspace_path=str(workspace), config={})
        agent = Agent(
            name="fence-agent",
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

    shutdown_started = asyncio.Event()
    allow_shutdown = asyncio.Event()

    async def block_shutdown(_project_id):
        shutdown_started.set()
        await asyncio.wait_for(allow_shutdown.wait(), timeout=1)
        return 0

    monkeypatch.setattr(task_runner, "cancel_project_sessions", block_shutdown)
    original_lock_boundary = ProjectService.lock_workspace_boundary

    async def preserve_start_session_cache(project_service, db, locked_project_id):
        if db is start_db:
            return None
        return await original_lock_boundary(project_service, db, locked_project_id)

    monkeypatch.setattr(ProjectService, "lock_workspace_boundary", preserve_start_session_cache)
    reset_task = None
    reset_results = []
    async with sessions() as reset_db, sessions() as start_db:
        try:
            stale_project = await start_db.get(Project, project_id)
            assert stale_project.status == "active"
            await start_db.commit()
            reset_task = asyncio.create_task(ProjectResetService().reset(reset_db, project_id, "Start fence"))
            await asyncio.wait_for(shutdown_started.wait(), timeout=1)
            assert stale_project.status == "active"
            with pytest.raises(HTTPException) as exc_info:
                await asyncio.wait_for(
                    SessionService().create(start_db, SessionCreate(agent_id=agent_id, project_id=project_id)),
                    timeout=1,
                )
            start_error = exc_info.value
        finally:
            await asyncio.wait_for(start_db.rollback(), timeout=1)
            allow_shutdown.set()
            if reset_task is not None:
                try:
                    reset_results.append(await asyncio.wait_for(asyncio.shield(reset_task), timeout=1))
                except TimeoutError:
                    reset_task.cancel()
                    reset_results.extend(
                        await asyncio.wait_for(asyncio.gather(reset_task, return_exceptions=True), timeout=1)
                    )

    assert not any(isinstance(result, BaseException) for result in reset_results)
    assert start_error.status_code == 409
    assert start_error.detail == {"code": "project_not_runnable", "reason": "project_inactive"}
    async with sessions() as observer_db:
        assert not (await observer_db.execute(select(Session).where(Session.project_id == project_id))).scalars().all()
    assert not task_runner.get_running_tasks()


@pytest.mark.asyncio
async def test_reset_awaits_registered_work_before_purging(test_engine, monkeypatch):
    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    async with sessions() as setup_db:
        project = Project(name="Shutdown", config={})
        setup_db.add(project)
        await setup_db.flush()
        task = Task(project_id=project.id, title="purge after shutdown")
        meeting = Meeting(
            project_id=project.id,
            title="still active until purge",
            meeting_type="general",
            status="active",
        )
        setup_db.add(task)
        setup_db.add(meeting)
        await setup_db.commit()
        project_id = project.id
        task_row_id = task.id
        meeting_id = meeting.id

    session_started = asyncio.Event()
    meeting_started = asyncio.Event()
    session_stopped = asyncio.Event()
    meeting_stopped = asyncio.Event()
    celery_stopped = asyncio.Event()

    async def running_session(_session_id, _runner_task_id):
        session_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            session_stopped.set()

    async def running_meeting():
        meeting_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            meeting_stopped.set()

    monkeypatch.setattr(task_runner, "execute_api_session", running_session)
    original_reset = ProjectService.reset

    async def established_celery_shutdown(_project_id):
        celery_stopped.set()

    async def purge_after_shutdown(service, db, purge_project_id):
        assert session_stopped.is_set()
        assert meeting_stopped.is_set()
        assert celery_stopped.is_set()
        assert (await db.get(Meeting, meeting_id)).status == "active"
        return await original_reset(service, db, purge_project_id)

    monkeypatch.setattr(
        meeting_tasks,
        "revoke_and_await_project_meeting_tasks",
        established_celery_shutdown,
    )
    monkeypatch.setattr(ProjectService, "reset", purge_after_shutdown)
    task_id = await task_runner.dispatch_session("reset-session", "api", project_id)
    meeting_tasks._track_in_process("reset-meeting", project_id, running_meeting())
    await asyncio.wait_for(asyncio.gather(session_started.wait(), meeting_started.wait()), timeout=1)

    async with sessions() as reset_db:
        result = await ProjectResetService().reset(reset_db, project_id, "Shutdown")

    assert result["cancelled_sessions"] == 1
    assert result["cancelled_meeting_tasks"] == 1
    assert session_stopped.is_set()
    assert meeting_stopped.is_set()
    assert task_id not in task_runner.get_running_tasks()
    async with sessions() as observer_db:
        assert await observer_db.get(Task, task_row_id) is None


@pytest.mark.asyncio
async def test_reset_rolls_back_partial_purge_and_keeps_fence(test_engine, monkeypatch):
    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    async with sessions() as setup_db:
        project = Project(name="Rollback", config={})
        setup_db.add(project)
        await setup_db.flush()
        task = Task(project_id=project.id, title="must survive")
        setup_db.add(task)
        await setup_db.commit()
        project_id = project.id
        task_id = task.id

    async def partial_purge(_service, db, project_id):
        await db.execute(Task.__table__.delete().where(Task.project_id == project_id))
        raise RuntimeError("purge failed")

    monkeypatch.setattr(ProjectService, "reset", partial_purge)
    async with sessions() as reset_db:
        with pytest.raises(RuntimeError, match="purge failed"):
            await ProjectResetService().reset(reset_db, project_id, "Rollback")

    async with sessions() as observer_db:
        assert await observer_db.get(Task, task_id) is not None
        assert (await observer_db.get(Project, project_id)).status == "resetting"


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_reset_preserves_operational_state_when_celery_shutdown_is_unconfirmed(
    test_engine, monkeypatch,
):
    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    async with sessions() as setup_db:
        project = Project(name="Celery failure", config={})
        setup_db.add(project)
        await setup_db.flush()
        task = Task(project_id=project.id, title="must survive")
        setup_db.add(task)
        await setup_db.commit()
        project_id = project.id
        task_id = task.id

    async def unconfirmed_shutdown(_project_id):
        raise TimeoutError("Celery acknowledgement timed out")

    monkeypatch.setattr(
        meeting_tasks,
        "revoke_and_await_project_meeting_tasks",
        unconfirmed_shutdown,
    )
    async with sessions() as reset_db:
        with pytest.raises(TimeoutError, match="acknowledgement"):
            await ProjectResetService().reset(reset_db, project_id, "Celery failure")

    async with sessions() as observer_db:
        assert await observer_db.get(Task, task_id) is not None
        assert (await observer_db.get(Project, project_id)).status == "resetting"


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_reset_preserves_operational_state_without_celery_app(
    test_engine, monkeypatch,
):
    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    async with sessions() as setup_db:
        project = Project(name="Missing Celery", config={})
        setup_db.add(project)
        await setup_db.flush()
        task = Task(project_id=project.id, title="must survive")
        setup_db.add(task)
        await setup_db.commit()
        project_id = project.id
        task_id = task.id

    monkeypatch.setattr(meeting_tasks, "settings", SimpleNamespace(is_sqlite=False))
    monkeypatch.setattr(meeting_tasks, "_meeting_task_app", None)
    async with sessions() as reset_db:
        with pytest.raises(RuntimeError, match="Celery meeting task app is unavailable"):
            await ProjectResetService().reset(reset_db, project_id, "Missing Celery")

    async with sessions() as observer_db:
        assert await observer_db.get(Task, task_id) is not None
        assert (await observer_db.get(Project, project_id)).status == "resetting"


@pytest.mark.asyncio
async def test_cancel_project_sessions_only_stops_and_awaits_selected_project(monkeypatch):
    project_a = "project-a"
    project_b = "project-b"
    started = {session_id: asyncio.Event() for session_id in ("a-1", "a-2", "b-1")}
    cleaned_up = {session_id: asyncio.Event() for session_id in started}

    async def run_until_cancelled(session_id: str, _runner_task_id: str) -> None:
        started[session_id].set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned_up[session_id].set()

    monkeypatch.setattr(task_runner, "execute_api_session", run_until_cancelled)

    task_ids = {
        session_id: await task_runner.dispatch_session(session_id, "api", project_id)
        for session_id, project_id in (("a-1", project_a), ("a-2", project_a), ("b-1", project_b))
    }
    await asyncio.wait_for(
        asyncio.gather(*(event.wait() for event in started.values())),
        timeout=1,
    )

    try:
        assert await asyncio.wait_for(task_runner.cancel_project_sessions(project_a), timeout=1) == 2
        assert cleaned_up["a-1"].is_set()
        assert cleaned_up["a-2"].is_set()
        assert not cleaned_up["b-1"].is_set()
        assert task_ids["a-1"] not in task_runner.get_running_tasks()
        assert task_ids["a-2"] not in task_runner.get_running_tasks()
        assert task_ids["b-1"] in task_runner.get_running_tasks()
        assert await task_runner.cancel_project_sessions(project_a) == 0
    finally:
        for task_id in task_ids.values():
            await task_runner.cancel_task(task_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_project_session_registry_cleans_up_completed_tasks(monkeypatch, fails):
    release = asyncio.Event()

    async def finish(_session_id: str, _runner_task_id: str) -> None:
        await release.wait()
        if fails:
            raise RuntimeError("boom")

    monkeypatch.setattr(task_runner, "execute_api_session", finish)

    task_id = await task_runner.dispatch_session("session-id", "api", "project-id")
    task = task_runner.get_running_tasks()[task_id]
    release.set()
    await asyncio.wait_for(task, timeout=1)

    assert task_id not in task_runner.get_running_tasks()
    assert await task_runner.cancel_project_sessions("project-id") == 0


@pytest.mark.asyncio
async def test_cancel_project_sessions_times_out_without_dropping_stuck_task(monkeypatch):
    first_cancelled = asyncio.Event()

    async def swallow_first_cancellation(_session_id: str, _runner_task_id: str) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            first_cancelled.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(task_runner, "execute_api_session", swallow_first_cancellation)
    monkeypatch.setattr(task_runner, "PROJECT_CANCELLATION_TIMEOUT_SECONDS", 0.01)
    task_id = await task_runner.dispatch_session("session-id", "api", "project-id")

    try:
        cancel_operation = asyncio.create_task(task_runner.cancel_project_sessions("project-id"))
        await asyncio.wait_for(first_cancelled.wait(), timeout=1)
        with pytest.raises(TimeoutError, match="1"):
            await cancel_operation
        assert task_id in task_runner.get_running_tasks()
        assert await task_runner.cancel_project_sessions("project-id") == 1
        assert task_id not in task_runner.get_running_tasks()
    finally:
        await task_runner.cancel_task(task_id)


@pytest.mark.asyncio
async def test_cancel_project_meeting_tasks_only_stops_selected_project(monkeypatch):
    started = {name: asyncio.Event() for name in ("a-start", "a-turn", "b-finalize")}

    async def wait_for_cancel(meeting_id: str) -> None:
        started[meeting_id].set()
        await asyncio.Event().wait()

    monkeypatch.setattr(meeting_tasks, "settings", SimpleNamespace(is_sqlite=True))
    monkeypatch.setattr(meeting_tasks, "start_meeting_async", wait_for_cancel)
    monkeypatch.setattr(meeting_tasks, "run_meeting_turn_async", wait_for_cancel)
    monkeypatch.setattr(meeting_tasks, "_run_finalize_in_process", wait_for_cancel)

    meeting_tasks.dispatch_start_meeting("a-start", "project-a")
    meeting_tasks.dispatch_run_meeting_turn("a-turn", "project-a")
    meeting_tasks.dispatch_finalize_meeting("b-finalize", "project-b")
    await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started.values())), timeout=1)

    assert await meeting_tasks.cancel_project_meeting_tasks("project-a") == 2
    assert len(meeting_tasks.get_running_meeting_tasks()) == 1
    await meeting_tasks.cancel_project_meeting_tasks("project-b")


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_meeting_registry_cleans_completed_and_failed_tasks(monkeypatch, fails):
    release = asyncio.Event()

    async def finish(_meeting_id: str) -> None:
        await release.wait()
        if fails:
            raise RuntimeError("boom")

    monkeypatch.setattr(meeting_tasks, "settings", SimpleNamespace(is_sqlite=True))
    monkeypatch.setattr(meeting_tasks, "start_meeting_async", finish)
    meeting_tasks.dispatch_start_meeting("meeting", "project")
    task = next(iter(meeting_tasks.get_running_meeting_tasks().values()))
    release.set()
    if fails:
        with pytest.raises(RuntimeError, match="boom"):
            await asyncio.wait_for(task, timeout=1)
    else:
        await asyncio.wait_for(task, timeout=1)
    await asyncio.sleep(0)

    assert not meeting_tasks.get_running_meeting_tasks()


@pytest.mark.asyncio
async def test_cancel_project_meeting_tasks_cancels_delayed_timeout_and_retry(monkeypatch):
    dispatched = []
    entered = 0
    both_entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed_sleep(_seconds: float) -> None:
        nonlocal entered
        entered += 1
        if entered == 2:
            both_entered.set()
        await release.wait()

    async def timeout(_meeting_id: str) -> None:
        dispatched.append("timeout")

    monkeypatch.setattr(meeting_tasks, "settings", SimpleNamespace(is_sqlite=True))
    monkeypatch.setattr(meeting_tasks.asyncio, "sleep", delayed_sleep)
    monkeypatch.setattr(meeting_tasks, "meeting_timeout_async", timeout)
    monkeypatch.setattr(meeting_tasks, "dispatch_run_meeting_turn", lambda *_: dispatched.append("retry"))
    meeting_tasks._schedule_in_process_timeout("meeting", "project", 60)
    meeting_tasks._schedule_in_process_turn_retry("meeting", "project", 60)
    await asyncio.wait_for(both_entered.wait(), timeout=1)

    assert await asyncio.wait_for(meeting_tasks.cancel_project_meeting_tasks("project"), timeout=1) == 2
    assert dispatched == []
    assert not meeting_tasks.get_running_meeting_tasks()


@pytest.mark.asyncio
async def test_meeting_cancellation_timeout_keeps_ownership(monkeypatch):
    first_cancelled = asyncio.Event()

    async def swallow_first_cancellation() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            first_cancelled.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(meeting_tasks, "PROJECT_CANCELLATION_TIMEOUT_SECONDS", 0.01)
    meeting_tasks._track_in_process("test", "project", swallow_first_cancellation())
    task_id = next(iter(meeting_tasks.get_running_meeting_tasks()))
    cancelling = asyncio.create_task(meeting_tasks.cancel_project_meeting_tasks("project"))
    await asyncio.wait_for(first_cancelled.wait(), timeout=1)
    with pytest.raises(TimeoutError, match="1"):
        await cancelling
    assert task_id in meeting_tasks.get_running_meeting_tasks()
    assert await meeting_tasks.cancel_project_meeting_tasks("project") == 1


@pytest.mark.unsupported_mode
def test_meeting_celery_jobs_are_stamped_and_cooperatively_revocable(monkeypatch):
    calls = []

    class Task:
        def s(self, *args):
            calls.append(("signature", args))
            return self

        def stamp(self, **kwargs):
            calls.append(("stamp", kwargs))
            return self

        def apply_async(self, **kwargs):
            calls.append(("apply_async", kwargs))

    class Control:
        def revoke_by_stamped_headers(self, headers, terminate):
            calls.append(("revoke", headers, terminate))

    class App:
        control = Control()

    monkeypatch.setattr(meeting_tasks, "settings", SimpleNamespace(is_sqlite=False))
    monkeypatch.setattr(meeting_tasks, "start_meeting", Task())
    monkeypatch.setattr(meeting_tasks, "run_meeting_turn", Task())
    monkeypatch.setattr(meeting_tasks, "finalize_meeting", Task())
    monkeypatch.setattr(meeting_tasks, "meeting_timeout", Task())
    monkeypatch.setattr(meeting_tasks, "_meeting_task_app", App())

    meeting_tasks.dispatch_start_meeting("meeting", "project")
    meeting_tasks.dispatch_run_meeting_turn("meeting", "project")
    meeting_tasks.dispatch_finalize_meeting("meeting", "project")
    meeting_tasks.dispatch_meeting_timeout("meeting", "project")
    meeting_tasks.schedule_meeting_timeout("meeting", "project", 7)
    meeting_tasks.revoke_project_meeting_tasks("project")

    assert calls == [
        ("signature", ("meeting",)),
        ("stamp", {"project_id": "project"}),
        ("apply_async", {}),
        ("signature", ("meeting",)),
        ("stamp", {"project_id": "project"}),
        ("apply_async", {}),
        ("signature", ("meeting",)),
        ("stamp", {"project_id": "project"}),
        ("apply_async", {}),
        ("signature", ("meeting",)),
        ("stamp", {"project_id": "project"}),
        ("apply_async", {}),
        ("signature", ("meeting",)),
        ("stamp", {"project_id": "project"}),
        ("apply_async", {"countdown": 7}),
        ("revoke", {"project_id": "project"}, False),
    ]


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_meeting_celery_shutdown_waits_for_stamped_jobs_to_clear(monkeypatch):
    project_job = {"id": "meeting-job", "stamps": {"project_id": "project"}}
    inspections = [
        {"worker": [project_job]},
        {"worker": []},
    ]
    calls = []

    class Inspector:
        def active(self):
            calls.append("active")
            return inspections.pop(0)

        def reserved(self):
            calls.append("reserved")
            return {"worker": []}

        def scheduled(self):
            calls.append("scheduled")
            return {"worker": []}

    class Control:
        def ping(self, timeout):
            calls.append(("ping", timeout))
            return [{"worker": {"ok": "pong"}}]

        def revoke_by_stamped_headers(self, headers, terminate, reply, timeout):
            calls.append(("revoke", headers, terminate, reply, timeout))
            return [{"worker": {"ok": ["meeting-job"]}}]

        def inspect(self, destination, timeout):
            calls.append(("inspect", destination, timeout))
            return Inspector()

    monkeypatch.setattr(meeting_tasks, "settings", SimpleNamespace(is_sqlite=False))
    monkeypatch.setattr(meeting_tasks, "_meeting_task_app", SimpleNamespace(control=Control()))
    monkeypatch.setattr(meeting_tasks, "PROJECT_CANCELLATION_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(meeting_tasks, "CELERY_SHUTDOWN_POLL_SECONDS", 0, raising=False)

    await meeting_tasks.revoke_and_await_project_meeting_tasks("project")

    revoke_call = next(call for call in calls if isinstance(call, tuple) and call[0] == "revoke")
    assert revoke_call[2] is False
    assert calls.count("active") == 2
    assert calls.count("reserved") == 2
    assert calls.count("scheduled") == 2


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_meeting_celery_shutdown_rejects_missing_worker_acknowledgement(monkeypatch):
    class Control:
        def ping(self, timeout):
            return [
                {"worker-a": {"ok": "pong"}},
                {"worker-b": {"ok": "pong"}},
            ]

        def revoke_by_stamped_headers(self, headers, terminate, reply, timeout):
            return [{"worker-a": {"ok": []}}]

    monkeypatch.setattr(meeting_tasks, "settings", SimpleNamespace(is_sqlite=False))
    monkeypatch.setattr(meeting_tasks, "_meeting_task_app", SimpleNamespace(control=Control()))

    with pytest.raises(TimeoutError, match="Not all Celery workers"):
        await meeting_tasks.revoke_and_await_project_meeting_tasks("project")


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_meeting_celery_shutdown_rejects_incomplete_inspection(monkeypatch):
    class Inspector:
        def active(self):
            return {"worker-a": []}

        def reserved(self):
            return None

    class Control:
        def ping(self, timeout):
            return [{"worker-a": {"ok": "pong"}}]

        def revoke_by_stamped_headers(self, headers, terminate, reply, timeout):
            return [{"worker-a": {"ok": []}}]

        def inspect(self, destination, timeout):
            return Inspector()

    monkeypatch.setattr(meeting_tasks, "settings", SimpleNamespace(is_sqlite=False))
    monkeypatch.setattr(meeting_tasks, "_meeting_task_app", SimpleNamespace(control=Control()))

    with pytest.raises(RuntimeError, match="reserved inspection"):
        await meeting_tasks.revoke_and_await_project_meeting_tasks("project")


@pytest.mark.asyncio
async def test_reset_purges_all_operational_state_without_touching_project_setup(
    db_session, test_project, test_agent, test_user, tmp_path,
):
    """Catches a forgotten/unscoped operational delete while retaining project setup."""
    workspace = tmp_path / "workspace"
    rally_dir = workspace / ".rally"
    rally_dir.mkdir(parents=True)
    marker = rally_dir / "keep.txt"
    marker.write_text("keep")
    test_project.workspace_path = str(workspace)
    test_project.config = {"keep": True}
    hook = Hook(
        project_id=test_project.id,
        name="retained-hook",
        code="pass",
        trigger_event="task.created",
        status="active",
        execution_count=3,
        error_count=2,
    )
    protocol = Protocol(project_id=test_project.id, name="retained-protocol", definition={})
    api_key = ApiKey(
        project_id=test_project.id,
        user_id=test_user.id,
        key_prefix="rly_",
        hashed_key="retained-api-key",
        label="retained",
    )
    routing_rule = RoutingRule(
        project_id=test_project.id,
        name="retained-rule",
        on_event="task.created",
        conditions={"kind": "task"},
        actions={"route": "agent"},
    )
    escalation_chain = EscalationChain(
        project_id=test_project.id,
        name="retained-escalation",
        definition={"level": 1},
        steps=[{"action": "notify"}],
    )
    db_session.add_all([hook, protocol, api_key, routing_rule, escalation_chain])
    await db_session.flush()

    task = Task(project_id=test_project.id, title="task")
    meeting = Meeting(project_id=test_project.id, title="meeting", meeting_type="decision")
    protocol_instance = ProtocolInstance(
        project_id=test_project.id,
        protocol_id=protocol.id,
        current_state="start",
    )
    channel = Channel(project_id=test_project.id, name="channel", channel_type="task")
    goal = OrchestrationGoal(project_id=test_project.id, objective="goal")
    artifact = Artifact(project_id=test_project.id, name="artifact", artifact_type="file")
    pattern = Pattern(
        project_id=test_project.id,
        pattern_type="test",
        description="test",
        confidence=1.0,
        sample_size=1,
    )
    db_session.add_all([task, meeting, protocol_instance, channel, goal, artifact, pattern])
    await db_session.flush()

    agenda_item = MeetingAgendaItem(meeting_id=meeting.id, order=1, title="agenda")
    run = OrchestrationRun(goal_id=goal.id)
    optimization = Optimization(
        project_id=test_project.id,
        pattern_id=pattern.id,
        type="hook",
        generated_code="pass",
    )
    db_session.add_all([agenda_item, run, optimization])
    await db_session.flush()

    decision = MeetingDecision(
        meeting_id=meeting.id,
        agenda_item_id=agenda_item.id,
        title="decision",
        chosen_option="yes",
        rationale="because",
        decided_by="agent",
    )
    gate = OrchestrationGate(
        run_id=run.id,
        success_criterion_key="done",
        gate_type="test",
    )
    db_session.add_all([decision, gate])
    await db_session.flush()

    target_children = [
        Session(task_id=task.id, agent_id=test_agent.id, project_id=test_project.id, adapter_type="api"),
        MeetingTurn(meeting_id=meeting.id, agenda_item_id=agenda_item.id, turn_number=1, content="turn"),
        MeetingActionItem(meeting_id=meeting.id, depends_on_decision_id=decision.id, description="action"),
        MeetingEvent(meeting_id=meeting.id, event_type="started"),
        MeetingParticipantSignal(meeting_id=meeting.id, agent_id=test_agent.id, signal_type="ready"),
        MeetingRequest(
            project_id=test_project.id,
            requesting_agent_id=test_agent.id,
            title="request",
            reason="reason",
        ),
        ProtocolTransition(protocol_instance_id=protocol_instance.id, to_state="next"),
        ProtocolTimeout(protocol_instance_id=protocol_instance.id, state_name="next", timeout_action="stop", expires_at=task.created_at),
        Message(channel_id=channel.id, sender_agent_id=test_agent.id, content="message"),
        OrchestrationDecision(run_id=run.id, decision_type="test"),
        OrchestrationAction(run_id=run.id, idempotency_key="action", action_type="test"),
        OrchestrationEvidence(run_id=run.id, gate_id=gate.id, source_type="test"),
        OrchestrationAgentSuggestion(run_id=run.id, missing_work_function="test", reason="reason"),
        OrchestrationProcessRun(goal_id=goal.id, run_id=run.id, process_type="goal_definition", trigger_reason="test"),
        OrchestrationWarning(goal_id=goal.id, run_id=run.id, warning_type="test", severity="warning", message="warning"),
        OrchestrationAuthorityDecision(
            goal_id=goal.id,
            run_id=run.id,
            decision_key="authority",
            title="authority",
            authority="human",
            question="question",
        ),
        OrchestrationAgentReview(
            goal_id=goal.id,
            run_id=run.id,
            agent_id=test_agent.id,
            definition_snapshot={},
            fit_summary="fit",
        ),
        OrchestrationMemorySection(
            project_id=test_project.id,
            goal_id=goal.id,
            run_id=run.id,
            section_key="test",
            title="memory",
            body="body",
            created_by="test",
        ),
        MemoryItem(agent_id=test_agent.id, project_id=test_project.id, scope="project", content="memory"),
        KnowledgeItem(project_id=test_project.id, content="knowledge", content_type="note"),
        ArtifactWatcher(artifact_id=artifact.id, watcher_kind="agent", watcher_id=test_agent.id),
        EventLog(project_id=test_project.id, event_type="test"),
        CostMetric(project_id=test_project.id, optimization_id=optimization.id, date=task.created_at.date()),
    ]
    db_session.add_all(target_children)
    retained_project = Project(name="retained", config={})
    db_session.add(retained_project)
    await db_session.flush()
    retained_task = Task(project_id=retained_project.id, title="retained")
    retained_meeting = Meeting(project_id=retained_project.id, title="retained", meeting_type="decision")
    retained_protocol_instance = ProtocolInstance(
        project_id=retained_project.id,
        protocol_id=protocol.id,
        current_state="start",
    )
    retained_channel = Channel(project_id=retained_project.id, name="retained", channel_type="task")
    retained_goal = OrchestrationGoal(project_id=retained_project.id, objective="retained")
    retained_artifact = Artifact(project_id=retained_project.id, name="retained", artifact_type="file")
    retained_pattern = Pattern(
        project_id=retained_project.id,
        pattern_type="test",
        description="retained",
        confidence=1.0,
        sample_size=1,
    )
    db_session.add_all([
        retained_task,
        retained_meeting,
        retained_protocol_instance,
        retained_channel,
        retained_goal,
        retained_artifact,
        retained_pattern,
    ])
    await db_session.flush()

    retained_agenda_item = MeetingAgendaItem(meeting_id=retained_meeting.id, order=1, title="retained")
    retained_run = OrchestrationRun(goal_id=retained_goal.id)
    retained_optimization = Optimization(
        project_id=retained_project.id,
        pattern_id=retained_pattern.id,
        type="hook",
        generated_code="pass",
    )
    db_session.add_all([retained_agenda_item, retained_run, retained_optimization])
    await db_session.flush()

    retained_decision = MeetingDecision(
        meeting_id=retained_meeting.id,
        agenda_item_id=retained_agenda_item.id,
        title="retained",
        chosen_option="yes",
        rationale="because",
        decided_by="agent",
    )
    retained_gate = OrchestrationGate(
        run_id=retained_run.id,
        success_criterion_key="retained",
        gate_type="test",
    )
    db_session.add_all([retained_decision, retained_gate])
    await db_session.flush()

    retained_children = [
        Session(
            task_id=retained_task.id,
            agent_id=test_agent.id,
            project_id=retained_project.id,
            adapter_type="api",
        ),
        MeetingTurn(
            meeting_id=retained_meeting.id,
            agenda_item_id=retained_agenda_item.id,
            turn_number=1,
            content="retained",
        ),
        MeetingActionItem(
            meeting_id=retained_meeting.id,
            depends_on_decision_id=retained_decision.id,
            description="retained",
        ),
        MeetingEvent(meeting_id=retained_meeting.id, event_type="started"),
        MeetingParticipantSignal(
            meeting_id=retained_meeting.id,
            agent_id=test_agent.id,
            signal_type="ready",
        ),
        MeetingRequest(
            project_id=retained_project.id,
            requesting_agent_id=test_agent.id,
            title="retained",
            reason="reason",
        ),
        ProtocolTransition(protocol_instance_id=retained_protocol_instance.id, to_state="next"),
        ProtocolTimeout(
            protocol_instance_id=retained_protocol_instance.id,
            state_name="next",
            timeout_action="stop",
            expires_at=task.created_at,
        ),
        Message(channel_id=retained_channel.id, sender_agent_id=test_agent.id, content="retained"),
        OrchestrationDecision(run_id=retained_run.id, decision_type="test"),
        OrchestrationAction(run_id=retained_run.id, idempotency_key="retained", action_type="test"),
        OrchestrationEvidence(run_id=retained_run.id, gate_id=retained_gate.id, source_type="test"),
        OrchestrationAgentSuggestion(
            run_id=retained_run.id,
            missing_work_function="test",
            reason="reason",
        ),
        OrchestrationProcessRun(
            goal_id=retained_goal.id,
            run_id=retained_run.id,
            process_type="goal_definition",
            trigger_reason="test",
        ),
        OrchestrationWarning(
            goal_id=retained_goal.id,
            run_id=retained_run.id,
            warning_type="test",
            severity="warning",
            message="warning",
        ),
        OrchestrationAuthorityDecision(
            goal_id=retained_goal.id,
            run_id=retained_run.id,
            decision_key="retained",
            title="retained",
            authority="human",
            question="question",
        ),
        OrchestrationAgentReview(
            goal_id=retained_goal.id,
            run_id=retained_run.id,
            agent_id=test_agent.id,
            definition_snapshot={},
            fit_summary="fit",
        ),
        OrchestrationMemorySection(
            project_id=retained_project.id,
            goal_id=retained_goal.id,
            run_id=retained_run.id,
            section_key="retained",
            title="memory",
            body="body",
            created_by="test",
        ),
        MemoryItem(
            agent_id=test_agent.id,
            project_id=retained_project.id,
            scope="project",
            content="retained",
        ),
        KnowledgeItem(project_id=retained_project.id, content="retained", content_type="note"),
        ArtifactWatcher(
            artifact_id=retained_artifact.id,
            watcher_kind="agent",
            watcher_id=test_agent.id,
        ),
        EventLog(project_id=retained_project.id, event_type="test"),
        CostMetric(
            project_id=retained_project.id,
            optimization_id=retained_optimization.id,
            date=task.created_at.date(),
        ),
    ]
    db_session.add_all(retained_children)
    await db_session.flush()

    target_rows = [
        task,
        meeting,
        protocol_instance,
        channel,
        goal,
        artifact,
        pattern,
        agenda_item,
        run,
        optimization,
        decision,
        gate,
        *target_children,
    ]
    retained_rows = [
        retained_task,
        retained_meeting,
        retained_protocol_instance,
        retained_channel,
        retained_goal,
        retained_artifact,
        retained_pattern,
        retained_agenda_item,
        retained_run,
        retained_optimization,
        retained_decision,
        retained_gate,
        *retained_children,
    ]
    preserved_rows = [
        (test_project, ("name", "description", "workspace_path", "config", "status")),
        (protocol, ("name", "description", "definition", "triggers", "escalation_chain", "is_active")),
        (hook, ("name", "description", "code", "status", "trigger_event")),
        (api_key, ("project_id", "user_id", "key_prefix", "hashed_key", "label")),
        (test_user, ("email", "display_name", "role", "is_active")),
        (test_agent, ("name", "role", "provider", "model", "adapter_type", "capabilities", "config", "is_active")),
        (routing_rule, ("project_id", "name", "description", "priority", "on_event", "conditions", "actions", "enabled")),
        (escalation_chain, ("project_id", "name", "description", "definition", "steps", "is_active")),
    ]
    preserved = [
        (type(row), row.id, fields, tuple(getattr(row, field) for field in fields))
        for row, fields in preserved_rows
    ]
    hook_id = hook.id
    target_identities = [
        (type(row), inspect(row).identity[0])
        for row in target_rows
    ]
    retained_identities = [
        (type(row), inspect(row).identity[0])
        for row in retained_rows
    ]

    counts = await ProjectService().reset(db_session, test_project.id)

    assert counts == {
        "sessions": 1,
        "tasks": 1,
        "meeting_agenda_items": 1,
        "meeting_turns": 1,
        "meeting_decisions": 1,
        "meeting_action_items": 1,
        "meeting_events": 1,
        "meeting_participant_signals": 1,
        "meeting_requests": 1,
        "meetings": 1,
        "protocol_transitions": 1,
        "protocol_timeouts": 1,
        "protocol_instances": 1,
        "messages": 1,
        "channels": 1,
        "orchestration_decisions": 1,
        "orchestration_actions": 1,
        "orchestration_gates": 1,
        "orchestration_evidence": 1,
        "orchestration_agent_suggestions": 1,
        "orchestration_process_runs": 1,
        "orchestration_warnings": 1,
        "orchestration_authority_decisions": 1,
        "orchestration_agent_reviews": 1,
        "orchestration_memory_sections": 1,
        "orchestration_runs": 1,
        "orchestration_goals": 1,
        "memory_items": 1,
        "knowledge_items": 1,
        "artifact_watchers": 1,
        "artifacts": 1,
        "event_log": 1,
        "cost_metrics": 1,
        "optimizations": 1,
        "patterns": 1,
    }
    db_session.expire_all()
    for model, row_id in target_identities:
        assert await db_session.get(model, row_id) is None
    for model, row_id in retained_identities:
        assert await db_session.get(model, row_id) is not None
    for model, row_id, fields, expected in preserved:
        actual = await db_session.get(model, row_id)
        assert actual is not None
        assert tuple(getattr(actual, field) for field in fields) == expected
    retained_hook = await db_session.get(Hook, hook_id)
    assert retained_hook is not None
    assert retained_hook.execution_count == retained_hook.error_count == 0
    assert marker.read_text() == "keep"
