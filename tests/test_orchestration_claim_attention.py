"""Regression coverage for the durable claim-attention boundary."""
import uuid
from datetime import timedelta
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from huddleroom.models.event_log import EventLog
from huddleroom.models.base import Base, _utcnow
from huddleroom.models.agent import Agent
from huddleroom.models.knowledge_item import KnowledgeItem
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationBudgetReservation, OrchestrationGoal, OrchestrationRun
from huddleroom.models.project import Project
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.models.user import User
from huddleroom.dependencies import _ANON_AGENT, _ANON_USER
from huddleroom.routers import agent_self_service, sessions as sessions_router, tasks as tasks_router
from huddleroom.schemas.session import SessionCreate
from huddleroom.schemas.task import StatusPatch, TaskAssign
from huddleroom.services.action_executor import ActionExecutor
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.session_service import SessionClaimAttention
from huddleroom.services.task_service import TaskService
from tests.test_orchestration_roadmap_integration import child_item, release
from tests.test_orchestration_runtime_e2e import _agent


@pytest.fixture(autouse=True)
def _manual_task_start(monkeypatch):
    # These tests drive the release-then-manual-run flow; auto-start is covered in test_orchestration_task_autostart.py.
    async def _noop(self, db, goal, run):
        return 0
    monkeypatch.setattr(OrchestrationService, "_start_released_tasks", _noop)


pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def committed_db(tmp_path):
    """Use an isolated database so route commits can replay across connections."""
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'claim_attention.db'}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    db = AsyncSession(bind=engine, expire_on_commit=False)
    db.add_all([
        User(id=_ANON_USER.id, email=_ANON_USER.email, hashed_password="", is_active=True, role="admin"),
        Agent(
            id=_ANON_AGENT.id, name=_ANON_AGENT.name, role="agent", provider="local", model="local",
            adapter_type="api", capabilities=[], config={}, is_active=False,
        ),
    ])
    project = Project(name="claim attention", workspace_path=str(tmp_path), config={})
    db.add(project)
    await db.commit()
    project_id = project.id
    try:
        yield db, project, engine
    finally:
        await db.rollback()
        await db.execute(delete(EventLog).where(EventLog.project_id == project_id))
        await db.execute(delete(Project).where(Project.id == project_id))
        await db.commit()
        await db.close()
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)
        await engine.dispose()


async def _exhausted_parent(db, project):
    agent = _agent("claim-parent", ["implementation"])
    db.add(agent)
    await db.flush()
    _roadmap, parent, run, _result, _row = await release(db, project, child_item())
    parent.budget = {"caps": {"max_tokens": 0}}
    await db.flush()
    return parent, run, agent


async def _exhausted_child(db, project):
    agent = _agent("claim-child", ["implementation"])
    db.add(agent)
    await db.flush()
    _roadmap, parent, _parent_run, _result, row = await release(db, project, child_item(
        allocation={"max_tokens": 0, "max_turns": 0, "max_hours": 0},
    ))
    child = await db.get(OrchestrationGoal, row.child_goal_id)
    run = await db.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    await db.flush()
    return parent, child, run, agent


async def _claim_task(db, project, run, agent, *, status="backlog", assigned=True):
    task = Task(
        project_id=project.id,
        title="claim boundary task",
        status=status,
        assigned_to=agent.id if assigned else None,
        metadata_={"orchestration": {"run_id": str(run.id)}},
    )
    db.add(task)
    await db.flush()
    db.add(OrchestrationAction(
        run_id=run.id, idempotency_key=f"test:claim-lineage:{run.id}:{task.id}",
        action_type="create_delegation_task", request={}, target_type="task", target_id=task.id, status="completed",
    ))
    await db.flush()
    return task


async def _assert_one_claim(db, run_id, goal_id):
    run = await db.get(OrchestrationRun, run_id, populate_existing=True)
    assert run is not None
    blockers = [item for item in run.active_blockers if item.get("scope") == f"claim:{goal_id}"]
    assert len(blockers) == 1
    return blockers[0]


@pytest.mark.parametrize("scope", ["parent", "child"])
async def test_public_claim_refusal_commits_one_typed_blocker_without_session_or_dispatch(
    committed_db, scope
):
    db_session, test_project, claim_engine = committed_db
    if scope == "parent":
        goal, run, agent = await _exhausted_parent(db_session, test_project)
    else:
        _parent, goal, run, agent = await _exhausted_child(db_session, test_project)
    goal_id, run_id, agent_id, task_id, project_id = goal.id, run.id, agent.id, None, test_project.id
    task = await _claim_task(db_session, test_project, run, agent, status="in_progress")
    task_id = task.id
    await db_session.commit()

    request = SessionCreate(agent_id=agent_id, task_id=task_id, project_id=project_id)
    with pytest.raises(SessionClaimAttention) as refused:
        await sessions_router.create_session(request, None, db_session)
    assert refused.value.run_id == run_id
    assert refused.value.blocker["scope"] == f"claim:{goal_id}"
    assert not db_session.sync_session.info.get("pending_session_dispatches")
    await db_session.close()
    replay_db = AsyncSession(bind=claim_engine, expire_on_commit=False)
    try:
        with pytest.raises(SessionClaimAttention):
            await sessions_router.create_session(request, None, replay_db)
        assert await replay_db.scalar(select(func.count()).select_from(Session).where(Session.task_id == task_id)) == 0
        await _assert_one_claim(replay_db, run_id, goal_id)
    finally:
        await replay_db.close()


async def test_settled_child_refuses_new_and_resumed_claims(committed_db):
    db_session, test_project, _claim_engine = committed_db
    _roadmap, parent, _parent_run, _result, row = await release(db_session, test_project, child_item())
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.child_goal_id == child.id,
    ))
    agent = await db_session.get(Agent, uuid.UUID(child.orchestrator_context["team"]["agent_ids"][0]))
    task = await _claim_task(db_session, test_project, run, agent, status="failed")
    existing = Session(
        project_id=test_project.id, task_id=task.id, agent_id=agent.id, adapter_type="api",
        status="failed", resumable=True, metadata_={"_run_config": {"max_tokens": 1, "timeout": 1}},
    )
    db_session.add(existing)
    await db_session.flush()
    existing_id = existing.id
    reservation.status = "settled"
    reservation.settled_at = _utcnow()
    reservation.settlement_reason = "cancelled"
    await db_session.commit()

    with pytest.raises(SessionClaimAttention, match="reservation"):
        await sessions_router.create_session(
            SessionCreate(agent_id=agent.id, task_id=task.id, project_id=test_project.id), None, db_session,
        )
    with pytest.raises(SessionClaimAttention, match="reservation"):
        await sessions_router.resume_session(existing_id, None, db_session)
    existing = await db_session.get(Session, existing_id)
    assert existing.status == "failed" and existing.resumable is True


async def test_terminal_roadmap_run_refuses_claim(committed_db):
    db_session, test_project, _claim_engine = committed_db
    _roadmap, parent, run, _result, _row = await release(db_session, test_project, child_item())
    agent = _agent("terminal-claim", ["implementation"])
    db_session.add(agent); await db_session.flush()
    task = await _claim_task(db_session, test_project, run, agent)
    parent.status = run.status = "completed"
    await db_session.flush()

    with pytest.raises(SessionClaimAttention, match="no longer active"):
        await sessions_router.create_session(
            SessionCreate(agent_id=agent.id, task_id=task.id, project_id=test_project.id), None, db_session,
        )


async def test_automatic_ready_and_assign_refusals_keep_outer_task_events(committed_db):
    db_session, test_project, _claim_engine = committed_db
    parent, parent_run, parent_agent = await _exhausted_parent(db_session, test_project)
    parent_task = await _claim_task(db_session, test_project, parent_run, parent_agent)
    with pytest.raises(SessionClaimAttention):
        await tasks_router.patch_status(
            test_project.id, parent_task.id, StatusPatch(status="ready"), None, db_session
        )

    _parent, child, child_run, child_agent = await _exhausted_child(db_session, test_project)
    child_task = await _claim_task(
        db_session, test_project, child_run, child_agent, status="ready", assigned=False
    )
    with pytest.raises(SessionClaimAttention):
        await tasks_router.assign_task(
            test_project.id, child_task.id, TaskAssign(agent_id=child_agent.id), None, db_session
        )

    await db_session.refresh(parent_task)
    await db_session.refresh(child_task)
    assert parent_task.status == "ready"
    assert child_task.assigned_to == child_agent.id
    events = list((await db_session.scalars(select(EventLog).where(EventLog.project_id == test_project.id))).all())
    assert len([event for event in events if event.event_type == "task.status_changed"
                and event.payload.get("task_id") == str(parent_task.id)]) == 1
    assert len([event for event in events if event.event_type == "task.assigned"
                and event.payload.get("task_id") == str(child_task.id)]) == 1
    assert await db_session.scalar(select(func.count()).select_from(Session).where(Session.task_id.in_([parent_task.id, child_task.id]))) == 0
    await _assert_one_claim(db_session, parent_run.id, parent.id)
    await _assert_one_claim(db_session, child_run.id, child.id)


async def test_claim_refused_completion_and_self_service_preserve_outer_writes(
    committed_db, monkeypatch
):
    db_session, test_project, _claim_engine = committed_db
    parent, run, agent = await _exhausted_parent(db_session, test_project)
    task = await _claim_task(db_session, test_project, run, agent)
    task.title = "outer write survives"
    attention = SessionClaimAttention(run.id, {"kind": "budget_integrity", "scope": f"claim:{parent.id}"}, "full")

    async def refuse(*_args, **_kwargs):
        raise attention

    monkeypatch.setattr(TaskService, "transition_status", refuse)
    result = await ActionExecutor()._complete_task(
        db_session, {"task_id": str(task.id)}, SimpleNamespace(linked_task_id=task.id)
    )
    assert result == {"action_type": "complete_task", "status": "claim_refused", "task_id": str(task.id)}
    await db_session.refresh(task)
    assert task.title == "outer write survives"
    await _assert_one_claim(db_session, run.id, parent.id)

    monkeypatch.undo()
    report_session = Session(
        project_id=test_project.id, agent_id=agent.id, adapter_type="api", status="completed"
    )
    db_session.add(report_session)
    await db_session.flush()
    with pytest.raises(SessionClaimAttention):
        await agent_self_service.post_agent_report(
            agent_self_service.AgentReport(
                session_id=report_session.id,
                knowledge_items=[agent_self_service.KnowledgeItemReport(content="kept", content_type="note")],
                status_update=agent_self_service.StatusUpdateReport(task_id=task.id, new_status="ready"),
            ),
            agent,
            db_session,
        )
    assert await db_session.scalar(select(func.count()).select_from(KnowledgeItem).where(KnowledgeItem.content == "kept")) == 1
    await db_session.refresh(task)
    assert task.status == "ready"
    await _assert_one_claim(db_session, run.id, parent.id)


async def test_retry_and_reassign_refusals_are_replay_safe_and_do_not_dispatch(committed_db):
    db_session, test_project, claim_engine = committed_db
    parent, parent_run, primary = await _exhausted_parent(db_session, test_project)
    parent_id, parent_run_id = parent.id, parent_run.id
    retry_task = await _claim_task(db_session, test_project, parent_run, primary, status="failed")
    retry_task_id = retry_task.id
    service = OrchestrationService()
    retry_request = {"task_id": str(retry_task.id)}
    first_retry = await service.execute_retry_task_action(db_session, parent_run_id, retry_request, "retry-claim")
    first_retry_id = first_retry.id

    _parent, child, child_run, child_primary = await _exhausted_child(db_session, test_project)
    child_id, child_run_id = child.id, child_run.id
    reassign_task = await _claim_task(db_session, test_project, child_run, child_primary, status="failed")
    reassign_task_id = reassign_task.id
    # The fixture's accepted immutable team contains child_primary.  Exercise
    # the budget refusal path, rather than an unrelated team-membership denial.
    reassign_request = {"task_id": str(reassign_task.id), "agent_id": str(child_primary.id)}
    first_reassign = await service.execute_reassign_task_action(
        db_session, child_run.id, reassign_request, "reassign-claim"
    )
    first_reassign_id = first_reassign.id
    await db_session.commit()
    assert not db_session.sync_session.info.get("pending_session_dispatches")
    await db_session.close()
    replay_db = AsyncSession(bind=claim_engine, expire_on_commit=False)
    try:
        replay_retry = await service.execute_retry_task_action(replay_db, parent_run_id, retry_request, "retry-claim")
        replay_reassign = await service.execute_reassign_task_action(
            replay_db, child_run_id, reassign_request, "reassign-claim"
        )
        assert (replay_retry.id, replay_retry.status) == (first_retry_id, "failed")
        assert (replay_reassign.id, replay_reassign.status) == (first_reassign_id, "failed")
        reassign_task = await replay_db.get(Task, reassign_task_id)
        assert reassign_task.assigned_to == child_primary.id
        assert await replay_db.scalar(select(func.count()).select_from(Session).where(
            Session.task_id.in_([retry_task_id, reassign_task_id])
        )) == 0
        events = list((await replay_db.scalars(select(EventLog.event_type).where(
            EventLog.project_id == test_project.id
        ))).all())
        assert "orchestration.task_retried" not in events and "orchestration.task_reassigned" not in events
        await _assert_one_claim(replay_db, parent_run_id, parent_id)
        await _assert_one_claim(replay_db, child_run_id, child_id)
    finally:
        await replay_db.close()


async def test_paused_parent_tick_recovers_only_its_claim_after_headroom_returns(committed_db):
    db_session, test_project, _claim_engine = committed_db
    _roadmap, parent, parent_run, _result, _row = await release(
        db_session, test_project, child_item(), cap=400
    )
    parent.budget = {"caps": {"max_tokens": 401, "max_turns": 10, "max_hours": 10}}
    parent.status = parent_run.status = "paused"
    parent_run.active_blockers = [
        {"kind": "budget_integrity", "scope": f"claim:{parent.id}"},
        {"kind": "budget_measurement", "scope": "claim:unrelated"},
    ]
    await db_session.flush()

    await OrchestrationService().tick(db_session, parent_run.id)

    await db_session.refresh(parent_run)
    assert not any(item.get("scope") == f"claim:{parent.id}" for item in parent_run.active_blockers)
    assert parent_run.active_blockers == [{"kind": "budget_measurement", "scope": "claim:unrelated"}]


async def test_child_tick_recovers_only_its_claim_when_parent_is_paused_and_keeps_measurement_attention(
    committed_db
):
    db_session, test_project, _claim_engine = committed_db
    _roadmap, parent, parent_run, _result, row = await release(
        db_session, test_project, child_item(), cap=400
    )
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.child_goal_id == child.id
    ))
    parent.status = parent_run.status = "paused"
    child.status = child_run.status = "paused"
    child_run.active_blockers = [
        {"kind": "budget_integrity", "scope": f"claim:{child.id}"},
        {"kind": "budget_measurement", "scope": "claim:unrelated"},
    ]
    parent_run.active_blockers = [{"kind": "budget_integrity", "scope": f"claim:{parent.id}"}]
    measurement_agent = _agent("measurement-session", ["implementation"])
    db_session.add(measurement_agent)
    await db_session.flush()
    child_task = await _claim_task(db_session, test_project, child_run, measurement_agent, status="done")
    # The completed run session deliberately lacks token telemetry, so local recovery must retain its claim blocker.
    db_session.add(Session(
        project_id=test_project.id, task_id=child_task.id, agent_id=measurement_agent.id,
        adapter_type="api", status="completed", metadata_={},
        started_at=child_run.started_at, ended_at=child_run.started_at + timedelta(seconds=1),
    ))
    await db_session.flush()

    await OrchestrationService().tick(db_session, child_run.id)
    await db_session.refresh(child_run)
    assert any(item.get("scope") == f"claim:{child.id}" for item in child_run.active_blockers)
    assert any(item.get("scope") == "claim:unrelated" for item in child_run.active_blockers)

    measured = await db_session.scalar(select(Session).where(Session.task_id == child_task.id))
    measured.metadata_ = {"token_count_in": 1, "token_count_out": 1}
    await db_session.flush()
    await OrchestrationService().tick(db_session, child_run.id)
    await db_session.refresh(child_run)
    assert not any(item.get("scope") == f"claim:{child.id}" for item in child_run.active_blockers)
    assert any(item.get("scope") == "claim:unrelated" for item in child_run.active_blockers)
    await db_session.refresh(parent_run)
    assert parent_run.active_blockers == [{"kind": "budget_integrity", "scope": f"claim:{parent.id}"}]
    assert reservation.allocation == {"max_tokens": "400", "max_turns": "5", "max_hours": "1"}
    assert parent.budget["caps"]["max_tokens"] == 400


async def test_router_resume_refusal_commits_once_then_child_tick_recovers_exact_cap_claim(committed_db):
    db_session, test_project, claim_engine = committed_db
    _roadmap, parent, parent_run, _result, row = await release(db_session, test_project, child_item())
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    agent = _agent("router-resume-claim", ["implementation"])
    db_session.add(agent); await db_session.flush()
    tasks = [Task(project_id=test_project.id, title=title, status="failed", assigned_to=agent.id,
        metadata_={"orchestration": {"run_id": str(child_run.id)}}) for title in ("first", "second")]
    db_session.add_all(tasks); await db_session.flush()
    for task in tasks:
        db_session.add(OrchestrationAction(
            run_id=child_run.id, idempotency_key=f"test:claim-lineage:{child_run.id}:{task.id}",
            action_type="create_delegation_task", request={}, target_type="task", target_id=task.id, status="completed",
        ))
    await db_session.flush()
    sessions = [Session(task_id=task.id, agent_id=agent.id, project_id=test_project.id,
        adapter_type="api", status="failed", resumable=True,
        started_at=child_run.started_at, ended_at=child_run.started_at,
        metadata_={"token_count_in": 0, "token_count_out": 0,
                   "_run_config": {"max_tokens": 400, "timeout": 3600}}) for task in tasks]
    db_session.add_all(sessions); await db_session.commit()
    parent_id, parent_run_id = parent.id, parent_run.id
    child_id, child_run_id = child.id, child_run.id
    first_id, second_id = sessions[0].id, sessions[1].id
    await db_session.close()

    first_db = AsyncSession(bind=claim_engine, expire_on_commit=False)
    try:
        resumed = await sessions_router.resume_session(first_id, None, first_db)
        assert resumed.status == "pending"
        await first_db.commit()
    finally:
        await first_db.close()

    for _ in range(2):
        refused_db = AsyncSession(bind=claim_engine, expire_on_commit=False)
        try:
            with pytest.raises(SessionClaimAttention):
                await sessions_router.resume_session(second_id, None, refused_db)
            second = await refused_db.get(Session, second_id)
            assert second.status == "failed" and second.resumable is True and second.runner_task_id is None
            assert not refused_db.sync_session.info.get("pending_session_dispatches")
            assert await refused_db.scalar(select(func.count()).select_from(Session).where(
                Session.task_id.in_([tasks[0].id, tasks[1].id])
            )) == 2
            await _assert_one_claim(refused_db, child_run_id, child_id)
        finally:
            await refused_db.close()

    recovery_db = AsyncSession(bind=claim_engine, expire_on_commit=False)
    try:
        first = await recovery_db.get(Session, first_id)
        first.status = "completed"
        first.started_at = child_run.started_at
        first.ended_at = child_run.started_at + timedelta(minutes=1)
        first.metadata_ = {"token_count_in": 50, "token_count_out": 50}
        recovered_child_run = await recovery_db.get(OrchestrationRun, child_run_id)
        recovered_parent_run = await recovery_db.get(OrchestrationRun, parent_run_id)
        recovered_child = await recovery_db.get(OrchestrationGoal, child_id)
        recovered_parent = await recovery_db.get(OrchestrationGoal, parent_id)
        recovered_child.status = recovered_child_run.status = "paused"
        recovered_parent.status = recovered_parent_run.status = "paused"
        await recovery_db.commit()

        await OrchestrationService().tick(recovery_db, child_run_id)
        await recovery_db.refresh(recovered_child_run)
        assert not any(item.get("scope") == f"claim:{child_id}" for item in recovered_child_run.active_blockers)
        assert await recovery_db.scalar(select(func.count()).select_from(Session)) == 2

        resumed = await sessions_router.resume_session(second_id, None, recovery_db)
        assert resumed.status == "pending"
    finally:
        await recovery_db.close()
