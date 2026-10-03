import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from huddleroom.models.project import Project
from huddleroom.models.protocol import Protocol, ProtocolInstance, ProtocolTimeout, ProtocolTransition
from huddleroom.routers.protocols import resume_protocol_instance
from huddleroom.services.agent_service import GLOBAL_PROJECT_ID
from huddleroom.services.event_bus import BusEvent
from huddleroom.services.protocol_engine import ProtocolEngineService
from huddleroom.services.project_reset_service import ProjectResetService
from huddleroom.services.project_service import ProjectService


def _protocol(project_id: uuid.UUID, *, triggers: list[dict] | None = None) -> Protocol:
    return Protocol(
        project_id=project_id,
        name=f"workspace-guard-{uuid.uuid4()}",
        version="1.0",
        definition={
            "initial_state": "waiting",
            "states": {
                "waiting": {"transitions": [{"trigger_event": "go", "to": "done"}]},
                "done": {},
            },
            "terminal_states": {"success": ["done"]},
        },
        triggers=triggers or [],
    )


def _assert_not_runnable(exc_info: pytest.ExceptionInfo[HTTPException]) -> None:
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == {"code": "project_not_runnable", "reason": "workspace_unset"}


@pytest.mark.asyncio
async def test_global_event_skips_workspace_guard_and_protocol_triggers(db_session, monkeypatch):
    async def unexpected(*args, **kwargs):
        raise AssertionError("global events must not enter project protocol processing")

    monkeypatch.setattr(ProjectService, "lock_workspace_boundary", unexpected)
    monkeypatch.setattr(ProjectService, "require_runnable_project", unexpected)
    monkeypatch.setattr(ProtocolEngineService, "_check_triggers", unexpected)

    await ProtocolEngineService().process_event(
        db_session,
        BusEvent(uuid.uuid4(), GLOBAL_PROJECT_ID, "agent.created", {}, "test"),
    )


@pytest.mark.asyncio
async def test_start_rejects_unrunnable_project_before_creating_instance(db_session, test_project):
    """Removing the start guard would persist an execution instance without a runnable workspace."""
    test_project.workspace_path = None
    protocol = _protocol(test_project.id)
    db_session.add(protocol)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await ProtocolEngineService().start_protocol(
            db_session,
            protocol,
            BusEvent(uuid.uuid4(), test_project.id, "trigger", {}, "test"),
        )

    _assert_not_runnable(exc_info)
    assert not (await db_session.execute(select(ProtocolInstance))).scalars().all()


@pytest.mark.asyncio
async def test_protocol_event_started_before_reset_is_serialized_then_purged(
    test_engine, monkeypatch, tmp_path,
):
    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    async with sessions() as setup_db:
        project = Project(name="Protocol race", workspace_path=str(workspace), config={})
        setup_db.add(project)
        await setup_db.flush()
        protocol = _protocol(project.id, triggers=[{"event_type": "go"}])
        setup_db.add(protocol)
        await setup_db.commit()
        project_id = project.id

    event_db = sessions()
    reset_db = sessions()
    guard_passed = asyncio.Event()
    release_event = asyncio.Event()
    reset_lock_acquired = asyncio.Event()
    original_require = ProjectService.require_runnable_project
    original_lock = ProjectService.lock_workspace_boundary

    async def pause_after_guard(project_service, db, project_id):
        workspace_path = await original_require(project_service, db, project_id)
        if db is event_db:
            guard_passed.set()
            await release_event.wait()
        return workspace_path

    async def track_reset_lock(project_service, db, project_id):
        result = await original_lock(project_service, db, project_id)
        if db is reset_db:
            reset_lock_acquired.set()
        return result

    monkeypatch.setattr(ProjectService, "require_runnable_project", pause_after_guard)
    monkeypatch.setattr(ProjectService, "lock_workspace_boundary", track_reset_lock)
    event_task = None
    reset_task = None
    try:
        async def process_event():
            async with event_db.begin():
                await ProtocolEngineService().process_event(
                    event_db,
                    BusEvent(uuid.uuid4(), project_id, "go", {}, "test"),
                )

        event_task = asyncio.create_task(process_event())
        await asyncio.wait_for(guard_passed.wait(), timeout=1)
        reset_task = asyncio.create_task(
            ProjectResetService().reset(reset_db, project_id, "Protocol race")
        )

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(reset_lock_acquired.wait(), timeout=0.05)

        release_event.set()
        await asyncio.wait_for(asyncio.gather(event_task, reset_task), timeout=2)
        async with sessions() as observer_db:
            assert not (await observer_db.execute(select(ProtocolInstance))).scalars().all()
    finally:
        release_event.set()
        for pending in (event_task, reset_task):
            if pending is not None and not pending.done():
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
        await event_db.close()
        await reset_db.close()


@pytest.mark.asyncio
async def test_event_transition_rejects_unrunnable_project_without_mutating_instance(db_session, test_project):
    """Removing the event recheck would transition a previously-started protocol after workspace loss."""
    test_project.workspace_path = None
    protocol = _protocol(test_project.id)
    db_session.add(protocol)
    await db_session.flush()
    instance = ProtocolInstance(protocol_id=protocol.id, project_id=test_project.id, current_state="waiting", context={})
    db_session.add(instance)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await ProtocolEngineService().process_event(
            db_session,
            BusEvent(uuid.uuid4(), test_project.id, "go", {}, "test"),
        )

    _assert_not_runnable(exc_info)
    await db_session.refresh(instance)
    assert instance.current_state == "waiting"
    assert not (await db_session.execute(select(ProtocolTransition))).scalars().all()


@pytest.mark.asyncio
async def test_manual_advance_and_resume_reject_unrunnable_project_without_mutation(db_session, test_project):
    """Removing either API guard would restart or advance protocol execution for an unusable workspace."""
    test_project.workspace_path = None
    protocol = _protocol(test_project.id)
    db_session.add(protocol)
    await db_session.flush()
    active = ProtocolInstance(protocol_id=protocol.id, project_id=test_project.id, current_state="waiting", context={})
    paused = ProtocolInstance(
        protocol_id=protocol.id, project_id=test_project.id, current_state="waiting", status="paused", context={}
    )
    db_session.add_all([active, paused])
    await db_session.flush()

    with pytest.raises(HTTPException) as advance_error:
        await ProtocolEngineService().advance_manually(db_session, active, "done")
    _assert_not_runnable(advance_error)

    with pytest.raises(HTTPException) as resume_error:
        await resume_protocol_instance(test_project.id, paused.id, db=db_session)
    _assert_not_runnable(resume_error)

    await db_session.refresh(active)
    await db_session.refresh(paused)
    assert active.current_state == "waiting"
    assert paused.status == "paused"


@pytest.mark.asyncio
async def test_resume_http_returns_structured_conflict_without_mutating_instance(
    client, auth_headers, db_session, test_project
):
    """Removing the HTTP guard would resume a paused instance or lose the stable conflict payload."""
    test_project.workspace_path = None
    protocol = _protocol(test_project.id)
    db_session.add(protocol)
    await db_session.flush()
    paused = ProtocolInstance(
        protocol_id=protocol.id,
        project_id=test_project.id,
        current_state="waiting",
        status="paused",
        context={},
    )
    db_session.add(paused)
    await db_session.flush()

    response = await client.post(
        f"/api/v1/projects/{test_project.id}/protocol-instances/{paused.id}/resume",
        headers=auth_headers,
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": {"code": "project_not_runnable", "reason": "workspace_unset"}
    }
    await db_session.refresh(paused)
    assert paused.status == "paused"
    assert not (await db_session.execute(select(ProtocolTransition))).scalars().all()


@pytest.mark.asyncio
async def test_timeout_worker_rejects_unrunnable_project_without_resolving_timeout(db_session, test_project):
    """Removing the worker recheck would emit escalation work after the workspace becomes invalid."""
    test_project.workspace_path = None
    protocol = _protocol(test_project.id)
    db_session.add(protocol)
    await db_session.flush()
    instance = ProtocolInstance(protocol_id=protocol.id, project_id=test_project.id, current_state="waiting", context={})
    db_session.add(instance)
    await db_session.flush()
    timeout = ProtocolTimeout(
        protocol_instance_id=instance.id,
        state_name="waiting",
        timeout_action="escalate",
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    db_session.add(timeout)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await ProtocolEngineService().process_timeouts(db_session)

    _assert_not_runnable(exc_info)
    await db_session.refresh(timeout)
    assert not timeout.resolved
