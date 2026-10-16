import uuid

import pytest

from fastapi import HTTPException

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun
from huddleroom.models.task import Task
from huddleroom.schemas.session import SessionCreate
from huddleroom.services.session_service import SessionService
from tests.test_orchestration_roster_mutation import sessions  # noqa: F401 -- fixture


async def _root_outcome(db, project, agent, caps, *, lineage=True, parent_goal_id=None):
    from huddleroom.services.orchestration_service import OrchestrationService

    child = {"parent_goal_id": parent_goal_id, "continuous_origin_key": f"child-{uuid.uuid4()}",
             "parent_contract_snapshot": {}, "goal_delta": {}} if parent_goal_id else {}
    goal = OrchestrationGoal(project_id=project.id, objective="Deliver", budget={"caps": caps}, **child)
    db.add(goal)
    await db.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized")
    db.add(run)
    await db.flush()
    task = Task(project_id=project.id, title="Deliver", assigned_to=agent.id,
                metadata_={"orchestration": {"run_id": str(run.id)}})
    db.add(task)
    await db.flush()
    db.add(OrchestrationAction(
        run_id=run.id, idempotency_key=OrchestrationService()._authorize_execution_key(run.id),
        action_type="authorize_execution", status="completed",
    ))
    action = OrchestrationAction(
        run_id=run.id, idempotency_key="deliver", action_type="create_delegation_task", status="completed",
        target_type="task", target_id=task.id,
    ) if lineage else None
    if action is not None:
        db.add(action)
    await db.flush()
    if action is not None:
        task.metadata_ = {"orchestration": {"run_id": str(run.id), "action_id": str(action.id)}}
        await db.flush()
    return goal, run, task, action


@pytest.mark.asyncio
async def test_root_outcome_claim_clamps_remaining_and_action_allocation(
    db_session, test_project, test_agent, monkeypatch,
):
    from huddleroom.models.orchestration import OrchestrationBudgetReservation
    from huddleroom.models.session import Session
    from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService

    goal, run, task, action = await _root_outcome(
        db_session, test_project, test_agent, {"max_tokens": "100", "max_hours": "1"},
    )
    action.budget_ledger = {
        "allocation": {"max_tokens": "40", "max_hours": "0.75"}, "reserved": {},
        "committed": {"max_tokens": "40", "max_hours": "0.75"}, "consumed": {},
    }
    for title, status, metadata in [
        ("spent", "completed", {"token_count_in": 10, "token_count_out": 10, "_roadmap_elapsed_seconds": 900}),
        ("live", "running", {"_run_config": {"max_tokens": 10, "timeout": 450}}),
    ]:
        other = Task(project_id=test_project.id, title=title)
        db_session.add(other)
        await db_session.flush()
        db_session.add(OrchestrationAction(
            run_id=run.id, idempotency_key=title, action_type="create_delegation_task", status="completed",
            target_type="task", target_id=other.id,
        ))
        db_session.add(Session(project_id=test_project.id, task_id=other.id, agent_id=test_agent.id,
                               adapter_type="api", status=status, metadata_=metadata))
    child = OrchestrationGoal(project_id=test_project.id, objective="Retained child")
    db_session.add(child)
    await db_session.flush()
    db_session.add(OrchestrationBudgetReservation(
        parent_goal_id=goal.id, child_goal_id=child.id, continuous_origin_key="retained",
        allocation={"max_tokens": "20", "max_hours": "0.125"},
    ))
    await db_session.flush()
    assert (await OrchestrationBudgetService().remaining(db_session, goal))["remaining"] == {
        "max_tokens": "50", "max_hours": "0.500",
    }
    # Editable metadata cannot remove durable ownership.
    task.metadata_ = {"orchestration": {"action_id": str(action.id)}}
    run.active_blockers = [{"scope": f"claim:{goal.id}"}, {"scope": "claim:unrelated"}]
    scheduled = []
    monkeypatch.setattr(SessionService, "_schedule_dispatch_after_commit", lambda *args: scheduled.append(args[2]))
    session = await SessionService().create(db_session, SessionCreate(
        agent_id=test_agent.id, project_id=test_project.id, task_id=task.id,
    ))
    assert session.metadata_["_run_config"] == {
        "max_tokens": 40, "timeout": 1800, "_roadmap_budget_enforced": True,
    }
    assert scheduled == [session.id]
    assert run.active_blockers == [{"scope": "claim:unrelated"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter,caps,allowed", [
    ("api", {"max_tokens": "100"}, True),
    ("cli", {"max_hours": "0.5"}, True),
    ("cli", {"max_tokens": "100"}, False),
    ("webhook", {"max_hours": "0.5"}, False),
])
async def test_root_outcome_adapter_capability_is_fail_closed(
    db_session, test_project, test_agent, monkeypatch, adapter, caps, allowed,
):
    from sqlalchemy import func, select
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
    from huddleroom.models.session import Session
    from huddleroom.services.session_service import SessionClaimAttention

    _goal, run, task, _action = await _root_outcome(db_session, test_project, test_agent, caps)
    scheduled = []
    monkeypatch.setattr(SessionService, "_schedule_dispatch_after_commit", lambda *args: scheduled.append(args[2]))
    request = SessionCreate(agent_id=test_agent.id, project_id=test_project.id, task_id=task.id,
                            adapter_type_override=adapter)
    if allowed:
        run.active_blockers = [{"scope": f"adapter-capability:{run.id}"}, {"scope": "adapter-capability:other"}]
        session = await SessionService().create(db_session, request)
        assert session.metadata_["_run_config"]["_roadmap_budget_enforced"] is True
        assert session.metadata_["_run_config"].get("timeout") == (1800 if adapter == "cli" else None)
        assert run.active_blockers == [{"scope": "adapter-capability:other"}]
    else:
        for _ in range(2):
            with pytest.raises(SessionClaimAttention) as refused:
                await SessionService().create(db_session, request)
            assert refused.value.blocker["scope"] == f"adapter-capability:{run.id}"
            await SessionService.persist_claim_attention(db_session, refused.value)
        assert len(run.active_blockers) == 1
        assert await db_session.scalar(select(func.count()).select_from(Session).where(Session.task_id == task.id)) == 0
        assert not scheduled
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAuthorityDecision)) == 0


async def _planning_task(db_session, test_project, test_agent, *, action_id, metadata_action_id=None):
    task = Task(
        id=uuid.uuid4(), project_id=test_project.id, title="Plan", assigned_to=test_agent.id,
        metadata_={"orchestration": {"action_id": str(metadata_action_id or action_id)}},
    )
    db_session.add(task)
    await db_session.flush()
    return task


@pytest.mark.asyncio
async def test_capped_standalone_plan_claim_stores_remaining_limits_before_dispatch(
    db_session, test_project, test_agent, monkeypatch,
):
    """Removing the metadata-bound plan action would let this capped claim dispatch unbounded."""
    goal = OrchestrationGoal(
        id=uuid.uuid4(), project_id=test_project.id, objective="Plan",
        budget={"caps": {"max_tokens": "100", "max_hours": "1"}},
    )
    run = OrchestrationRun(id=uuid.uuid4(), goal_id=goal.id, status="running", phase="authorized")
    db_session.add(goal)
    await db_session.flush()
    db_session.add(run)
    await db_session.flush()
    action = OrchestrationAction(
        run_id=run.id, idempotency_key="plan", action_type="request_plan", status="completed",
    )
    db_session.add(action)
    await db_session.flush()
    task = await _planning_task(db_session, test_project, test_agent, action_id=action.id)
    action.target_type, action.target_id = "task", task.id
    scheduled = []
    monkeypatch.setattr(SessionService, "_schedule_dispatch_after_commit", lambda *args: scheduled.append(args[2]))

    session = await SessionService().create(
        db_session, SessionCreate(agent_id=test_agent.id, project_id=test_project.id, task_id=task.id),
    )

    assert session.metadata_["_run_config"] == {
        "max_tokens": 100, "timeout": 3600, "_roadmap_budget_enforced": True,
    }
    assert scheduled == [session.id]


@pytest.mark.asyncio
async def test_funded_plan_claim_clamps_session_limits_to_its_allocation(
    db_session, test_project, test_agent, monkeypatch,
):
    goal = OrchestrationGoal(
        id=uuid.uuid4(), project_id=test_project.id, objective="Plan", budget={"caps": {"max_tokens": "9"}},
    )
    run = OrchestrationRun(id=uuid.uuid4(), goal_id=goal.id, status="running", phase="authorized")
    db_session.add_all([goal, run])
    await db_session.flush()
    action = OrchestrationAction(
        run_id=run.id, idempotency_key="plan-funded", action_type="request_plan", status="completed",
        budget_ledger={
            "allocation": {"max_tokens": "4"}, "reserved": {}, "committed": {"max_tokens": "4"},
            "consumed": {}, "usage_state": "known", "enforceability": "enforceable",
        },
    )
    db_session.add(action)
    await db_session.flush()
    task = await _planning_task(db_session, test_project, test_agent, action_id=action.id)
    action.target_type, action.target_id = "task", task.id
    monkeypatch.setattr(SessionService, "_schedule_dispatch_after_commit", lambda *_args: None)

    session = await SessionService().create(
        db_session, SessionCreate(agent_id=test_agent.id, project_id=test_project.id, task_id=task.id),
    )

    assert session.metadata_["_run_config"]["max_tokens"] == 4


@pytest.mark.asyncio
async def test_standalone_plan_claim_rejects_metadata_action_with_wrong_task_lineage(
    db_session, test_project, test_agent,
):
    """Changing the action target must reject the claim instead of selecting another action."""
    goal = OrchestrationGoal(
        id=uuid.uuid4(), project_id=test_project.id, objective="Plan", budget={"caps": {"max_tokens": "100"}},
    )
    run = OrchestrationRun(id=uuid.uuid4(), goal_id=goal.id, status="running", phase="authorized")
    db_session.add(goal)
    await db_session.flush()
    db_session.add(run)
    await db_session.flush()
    action = OrchestrationAction(
        run_id=run.id, idempotency_key="plan", action_type="request_plan", status="completed",
        target_type="task", target_id=uuid.uuid4(),
    )
    db_session.add(action)
    await db_session.flush()
    task = await _planning_task(db_session, test_project, test_agent, action_id=action.id)
    db_session.add(OrchestrationAction(
        run_id=run.id, idempotency_key="other-plan", action_type="request_plan", status="completed",
        target_type="task", target_id=task.id,
    ))
    await db_session.flush()

    with pytest.raises(HTTPException, match="Planning action task lineage is invalid") as exc_info:
        await SessionService().create(
            db_session, SessionCreate(agent_id=test_agent.id, project_id=test_project.id, task_id=task.id),
        )

    assert exc_info.value.status_code == 409


async def _publication_seed(sessions, tmp_path):
    from tests.test_orchestration_roster_mutation import seed_project_user, make_agent
    async with sessions() as setup:
        project, _ = await seed_project_user(setup, tmp_path)
        agent = make_agent(name=f"publication-{uuid.uuid4()}", capabilities=["implementation"])
        setup.add(agent)
        await setup.commit()
        return project, agent


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["nested_commit_outer_rollback", "nested_rollback", "rollback_reuse", "close_reuse", "commit_once"])
async def test_session_dispatch_waits_for_outer_transaction_outcome(sessions, tmp_path, monkeypatch, outcome):
    """SPR230/SPR231 actual SessionService scheduling follows the outer DB outcome."""
    import sqlite3
    from sqlalchemy import update
    from huddleroom.models.project import Project
    from huddleroom.models.session import Session
    project, agent = await _publication_seed(sessions, tmp_path)
    registered, visible = [], []
    def register(session_id, *args, **kwargs):
        registered.append(uuid.UUID(session_id))
        with sqlite3.connect(sessions.kw["bind"].url.database) as reader:
            visible.append(reader.execute("SELECT count(*) FROM sessions WHERE id=?", (uuid.UUID(session_id).hex,)).fetchone()[0])
    monkeypatch.setattr("huddleroom.workers.task_runner.register_session", register)
    caller = sessions()
    try:
        await caller.execute(update(Project).where(Project.id == project.id).values(name="outer writer"))
        if outcome == "nested_commit_outer_rollback":
            nested = await caller.begin_nested()
            dropped = await SessionService().create(caller, SessionCreate(agent_id=agent.id, project_id=project.id))
            await nested.commit()
            assert registered == [], "SAVEPOINT commit published a session before outer commit"
            await caller.rollback()
            expected, missing = [], [dropped.id]
        else:
            outer = await SessionService().create(caller, SessionCreate(agent_id=agent.id, project_id=project.id))
            if outcome == "nested_rollback":
                nested = await caller.begin_nested()
                SessionService()._schedule_dispatch_after_commit(
                    caller, outer.id, outer.project_id, outer.adapter_type, outer.runner_task_id)
                dropped = await SessionService().create(caller, SessionCreate(agent_id=agent.id, project_id=project.id))
                await nested.rollback()
                assert registered == []
                await caller.commit()
                expected, missing = [outer.id], [dropped.id]
            elif outcome in {"rollback_reuse", "close_reuse"}:
                await (caller.rollback() if outcome == "rollback_reuse" else caller.close())
                await caller.commit()
                assert registered == []
                await caller.execute(update(Project).where(Project.id == project.id).values(name="reused writer"))
                reused = await SessionService().create(caller, SessionCreate(agent_id=agent.id, project_id=project.id))
                await caller.commit()
                expected, missing = [reused.id], [outer.id]
            else:
                nested = await caller.begin_nested()
                SessionService()._schedule_dispatch_after_commit(
                    caller, outer.id, outer.project_id, outer.adapter_type, outer.runner_task_id)
                await nested.commit()
                assert registered == []
                await caller.commit()
                await caller.commit()
                expected, missing = [outer.id], []
        assert registered == expected and visible == [1] * len(expected)
        assert not caller.sync_session.info.get("pending_session_dispatches")
        async with sessions() as fresh:
            for session_id in expected:
                assert await fresh.get(Session, session_id) is not None
            for session_id in missing:
                assert await fresh.get(Session, session_id) is None
    finally:
        await caller.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "resume"])
@pytest.mark.parametrize("caller_owned", [False, True])
async def test_workspace_writer_and_pause_settle_in_commit_order(sessions, tmp_path, monkeypatch, operation, caller_owned):
    """SPR230/SPR231 SQLite writer order settles workspace/session admission before Pause."""
    import asyncio
    from sqlalchemy import select, update
    from huddleroom.models.project import Project
    from huddleroom.models.session import Session
    from huddleroom.services.project_service import ProjectService
    from huddleroom.services.orchestration_service import OrchestrationService
    from huddleroom.services import orchestration_control as controls
    from huddleroom.workers.session_tasks import _require_runnable_session
    project, agent = await _publication_seed(sessions, tmp_path)
    registered = []
    monkeypatch.setattr("huddleroom.workers.task_runner.register_session", lambda *args, **kwargs: registered.append(args[0]))
    async with sessions() as setup:
        goal, run, task, _ = await _root_outcome(setup, project, agent, {})
        await setup.commit()
    previous_id = None
    if operation == "resume":
        async with sessions() as setup:
            previous = await SessionService().create(setup, SessionCreate(
                agent_id=agent.id, project_id=project.id, task_id=task.id))
            previous.status = "failed"
            previous.resumable = True
            await setup.commit()
            previous_id = previous.id
    registered.clear()
    workspace_written, pause_attempted, release, returned, end_caller = [asyncio.Event() for _ in range(5)]
    holder, observed = {}, {}
    original_workspace, original_writer = ProjectService.lock_workspace_boundary, controls.acquire_control_writer
    async def workspace(service, db, project_id):
        await original_workspace(service, db, project_id)
        if db is holder.get("a") and not workspace_written.is_set():
            workspace_written.set()
            await release.wait()
    async def writer(db, run_id):
        if db is holder.get("b"):
            pause_attempted.set()
        await original_writer(db, run_id)
    monkeypatch.setattr(ProjectService, "lock_workspace_boundary", workspace)
    monkeypatch.setattr(controls, "acquire_control_writer", writer)
    async def admitting():
        async with sessions() as db:
            holder["a"] = db
            if caller_owned:
                await db.begin()
                await db.execute(update(Project).where(Project.id == project.id).values(description="caller staged"))
                await db.flush()
            result = await (SessionService().create(db, SessionCreate(
                agent_id=agent.id, project_id=project.id, task_id=task.id)) if operation == "create"
                else SessionService().resume(db, previous_id))
            observed["session_id"], observed["runner_id"] = result.id, result.runner_task_id
            returned.set()
            if caller_owned:
                assert db.in_transaction() and registered == []
                await end_caller.wait()
            await db.commit()
    async def pausing():
        async with sessions() as db:
            holder["b"] = db
            return await OrchestrationService().pause_goal(db, project.id, goal.id)
    a = asyncio.create_task(admitting())
    a.add_done_callback(lambda _task: returned.set())
    b = None
    try:
        await asyncio.wait_for(workspace_written.wait(), 3)
        b = asyncio.create_task(pausing())
        await asyncio.wait_for(pause_attempted.wait(), 3)
        release.set()
        await asyncio.wait_for(returned.wait(), 3)
        if a.done():
            await a
        if caller_owned:
            async with sessions() as fresh:
                assert await fresh.scalar(select(Project.description).where(Project.id == project.id)) != "caller staged"
                assert registered == []
            assert not b.done()
            end_caller.set()
        await asyncio.wait_for(asyncio.gather(a, b), 3)
        assert registered == [str(observed["session_id"])]
        async with sessions() as fresh:
            saved = await fresh.get(OrchestrationRun, run.id)
            assert saved.pause_requested is True and saved.status == "paused"
            session = await fresh.get(Session, observed["session_id"])
            assert session.status == "pending"
            if caller_owned:
                assert await fresh.scalar(select(Project.description).where(Project.id == project.id)) == "caller staged"
        async with sessions() as worker:
            assert await _require_runnable_session(worker, observed["session_id"], observed["runner_id"]) is False
        async with sessions() as fresh:
            assert (await fresh.get(Session, observed["session_id"])).status == "pending"
    finally:
        release.set()
        end_caller.set()
        await asyncio.gather(a, *([b] if b else []), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("paused_target", ["self", "ancestor"])
async def test_final_worker_claim_rereads_pause_after_awaited_preflight(sessions, tmp_path, monkeypatch, paused_target):
    """SPR230/SPR231 no pending-to-running claim is admitted from stale preflight."""
    import asyncio
    from huddleroom.models.session import Session
    from huddleroom.services.project_service import ProjectService
    from huddleroom.services.orchestration_service import OrchestrationService
    from huddleroom.workers.session_tasks import _require_runnable_session
    project, agent = await _publication_seed(sessions, tmp_path)
    monkeypatch.setattr("huddleroom.workers.task_runner.register_session", lambda *args, **kwargs: None)
    async with sessions() as setup:
        parent = None
        if paused_target == "ancestor":
            parent, _, _, _ = await _root_outcome(setup, project, agent, {})
        goal, run, task, _ = await _root_outcome(
            setup, project, agent, {}, parent_goal_id=parent.id if parent else None)
        target = parent or goal
        if parent:
            from huddleroom.models.orchestration import OrchestrationBudgetReservation
            setup.add(OrchestrationBudgetReservation(parent_goal_id=parent.id, child_goal_id=goal.id,
                continuous_origin_key=goal.continuous_origin_key, allocation={}))
        await setup.commit()
    async with sessions() as setup:
        session = await SessionService().create(setup, SessionCreate(
            agent_id=agent.id, project_id=project.id, task_id=task.id))
        await setup.commit()
        session_id, runner_id = session.id, session.runner_task_id
    entered, release = asyncio.Event(), asyncio.Event()
    holder = {}
    original = ProjectService.require_runnable_project
    async def prepared(service, db, project_id):
        result = await original(service, db, project_id)
        if db is holder.get("worker"):
            entered.set()
            await release.wait()
        return result
    monkeypatch.setattr(ProjectService, "require_runnable_project", prepared)
    async def claim():
        async with sessions() as worker:
            holder["worker"] = worker
            return await _require_runnable_session(worker, session_id, runner_id)
    pending = asyncio.create_task(claim())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        async with sessions() as pause:
            await asyncio.wait_for(OrchestrationService().pause_goal(pause, project.id, target.id), 3)
        release.set()
        assert await asyncio.wait_for(pending, 3) is False
        async with sessions() as fresh:
            saved = await fresh.get(Session, session_id)
            assert saved.status == "pending" and saved.started_at is None
            assert saved.runner_task_id == runner_id and not (saved.metadata_ or {}).get("attempt")
            own = await fresh.get(OrchestrationGoal, goal.id)
            assert own.status == ("active" if paused_target == "ancestor" else "paused")
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        if paused_target == "ancestor":
            from sqlalchemy import null, update
            async with sessions() as cleanup:
                await cleanup.execute(update(OrchestrationGoal).where(OrchestrationGoal.id == goal.id).values(
                    parent_goal_id=None, continuous_origin_key=None, parent_contract_snapshot=null(), goal_delta=null()))
                await cleanup.commit()


@pytest.mark.asyncio
async def test_task_activation_pause_wins_after_preflight(sessions, tmp_path, monkeypatch):
    """SPR230/SPR231 awaited task preflight cannot admit activation after accepted Pause."""
    import asyncio
    from sqlalchemy import select
    from huddleroom.models.session import Session
    from huddleroom.services.task_service import TaskService
    from huddleroom.services.orchestration_service import OrchestrationService
    project, agent = await _publication_seed(sessions, tmp_path)
    async with sessions() as setup:
        goal, _, task, _ = await _root_outcome(setup, project, agent, {})
        await setup.commit()
    entered, release = asyncio.Event(), asyncio.Event()
    original = SessionService.preflight_claim
    async def preflight(service, db, task, agent, *args, **kwargs):
        await original(service, db, task, agent, *args, **kwargs)
        entered.set()
        await release.wait()
    monkeypatch.setattr(SessionService, "preflight_claim", preflight)
    registered = []
    monkeypatch.setattr("huddleroom.workers.task_runner.register_session", lambda *args, **kwargs: registered.append((args, kwargs)))
    async def activate():
        async with sessions() as db:
            try:
                await TaskService().run(db, project.id, task.id)
            except HTTPException as exc:
                assert exc.status_code == 409
            await db.commit()
    pending = asyncio.create_task(activate())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        async with sessions() as pause:
            await OrchestrationService().pause_goal(pause, project.id, goal.id)
        release.set()
        await asyncio.wait_for(pending, 3)
        async with sessions() as fresh:
            assert (await fresh.get(Task, task.id)).status == "backlog"
            assert await fresh.scalar(select(Session).where(Session.task_id == task.id)) is None
        assert registered == []
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_native_dispatch_preflushed_caller_conflicts_before_long_lock(sessions, tmp_path, monkeypatch):
    """SPR230/SPR231 preserve caller writes instead of awaiting a native effect mutex."""
    from contextlib import asynccontextmanager
    from sqlalchemy import update, select
    from huddleroom.models.project import Project
    from huddleroom.models.orchestration import OrchestrationDecision
    from huddleroom.services.orchestration_service import OrchestrationService
    from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher
    project, agent = await _publication_seed(sessions, tmp_path)
    async with sessions() as setup:
        goal, run, _, _ = await _root_outcome(setup, project, agent, {})
        decision = OrchestrationDecision(run_id=run.id, decision_type="execution", validator_status="accepted",
            input_snapshot={}, parsed_decision={"action_type": "noop", "reason": "wait"})
        setup.add(decision)
        await setup.commit()
    @asynccontextmanager
    async def forbidden(*args, **kwargs):
        pytest.fail("native dispatch waited for a new goal lock while caller held SQLite writer")
        yield
    monkeypatch.setattr(OrchestrationService, "_lock_goal_for_baseline_transition", forbidden)
    async with sessions() as caller:
        current = await caller.get(OrchestrationRun, run.id)
        accepted = await caller.get(OrchestrationDecision, decision.id)
        await caller.execute(update(Project).where(Project.id == project.id).values(description="caller staging"))
        outer = caller.get_transaction()
        with pytest.raises(HTTPException) as conflict:
            await OrchestrationDecisionDispatcher(OrchestrationService()).dispatch(caller, current, accepted)
        assert conflict.value.status_code == 409 and caller.get_transaction() is outer
        assert await caller.scalar(select(Project.description).where(Project.id == project.id)) == "caller staging"
        async with sessions() as fresh:
            assert await fresh.scalar(select(Project.description).where(Project.id == project.id)) != "caller staging"
        await caller.rollback()


@pytest.mark.asyncio
async def test_paused_pending_reservation_redispatches_same_runner_on_resume(sessions, tmp_path, monkeypatch):
    """SPR230/SPR231 denied claim stays recoverable; Resume reuses the native reservation."""
    from unittest.mock import AsyncMock
    from huddleroom.models.session import Session
    from huddleroom.services.orchestration_service import OrchestrationService
    from huddleroom.workers.session_tasks import _require_runnable_session

    project, agent = await _publication_seed(sessions, tmp_path)
    registered = []
    monkeypatch.setattr("huddleroom.workers.task_runner.register_session", lambda *args, **kwargs: registered.append((args, kwargs)))
    async with sessions() as setup:
        goal, run, task, _ = await _root_outcome(setup, project, agent, {})
        await setup.commit()
        reserved = await SessionService().create(setup, SessionCreate(
            agent_id=agent.id, project_id=project.id, task_id=task.id))
        await setup.commit()
        session_id, runner_id = reserved.id, reserved.runner_task_id
    assert len(registered) == 1
    registered.clear()
    async with sessions() as pause:
        await OrchestrationService().pause_goal(pause, project.id, goal.id)
    async with sessions() as worker:
        assert await _require_runnable_session(worker, session_id, runner_id) is False
    async with sessions() as resume:
        await OrchestrationService().resume_goal(resume, project.id, goal.id)
        await resume.commit()
    # Stop unrelated authorized planning only after the actual pending-session
    # redispatch checkpoint. No admission, scheduling or worker helper is stubbed.
    service = OrchestrationService()
    monkeypatch.setattr(service, "_hard_stop_active", AsyncMock(return_value=True))
    async with sessions() as tick:
        await service.tick(tick, run.id)
        await tick.commit()
    assert len(registered) == 1
    assert registered[0][0][0] == str(session_id)
    assert registered[0][1]["task_id"] == runner_id
    async with sessions() as worker:
        assert await _require_runnable_session(worker, session_id, runner_id) is True
        await worker.commit()
    async with sessions() as tick:
        await service.tick(tick, run.id)
        await tick.commit()
    assert len(registered) == 1
    async with sessions() as fresh:
        from sqlalchemy import select
        all_sessions = list((await fresh.scalars(select(Session).where(Session.task_id == task.id))).all())
        assert len(all_sessions) == 1
        assert all_sessions[0].id == session_id and all_sessions[0].runner_task_id == runner_id
        assert all_sessions[0].status == "running"
