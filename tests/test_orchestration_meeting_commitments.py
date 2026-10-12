import uuid

import pytest

from huddleroom.models.graph import Graph, GraphRun
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGate, OrchestrationGoal
from huddleroom.models.task import Task
from huddleroom.services.orchestration_meeting_commitments import meeting_commitments
from tests.test_orchestration_delegation_contracts import _agent, _delegation_request, _make_run
from tests.test_orchestration_progress_view import (
    _action_item, _delegation, _meeting, _run_with_criteria, _task,
)

pytestmark = pytest.mark.asyncio


async def _setup(db, project):
    _goal, run = await _run_with_criteria(db, project, [])
    source = await _task(db, project, run)
    return run, await _meeting(db, project, source)


async def _one(db, run):
    (entry,) = await meeting_commitments(db, run)
    return entry


async def test_assigned_without_completion_and_open(db_session, test_project):
    run, meeting = await _setup(db_session, test_project)
    await _action_item(db_session, meeting, assignee_user_id=uuid.uuid4())
    entry = await _one(db_session, run)
    assert (entry["state"], entry["fulfilled"], entry["task_id"]) == ("assigned", False, None)
    assert entry["reason"]
    await _action_item(db_session, meeting, description="Other")
    states = {e["description"]: e["state"] for e in await meeting_commitments(db_session, run)}
    assert states["Other"] == "open"


@pytest.mark.parametrize("status", ["failed", "blocked", "cancelled"])
async def test_failed_linked_task(db_session, test_project, status):
    run, meeting = await _setup(db_session, test_project)
    task = await _task(db_session, test_project, run, status=status)
    await _action_item(db_session, meeting, task_id=task.id, status="task_created")
    entry = await _one(db_session, run)
    assert (entry["state"], entry["fulfilled"], entry["task_id"]) == ("task_failed", False, str(task.id))
    assert entry["reason"]


async def test_in_flight_task(db_session, test_project):
    run, meeting = await _setup(db_session, test_project)
    task = await _task(db_session, test_project, run, status="in_progress")
    await _action_item(db_session, meeting, task_id=task.id, status="task_created")
    assert (await _one(db_session, run))["state"] == "task_in_flight"


async def _done_task_with_gate(db, project, run, gate_status):
    gate = OrchestrationGate(
        run_id=run.id, success_criterion_key="k", gate_type="work_completed", required_evidence={}, status=gate_status,
    )
    db.add(gate)
    await db.flush()
    task = await _task(db, project, run, status="done")
    task.metadata_ = {"orchestration": {"run_id": str(run.id), "plan_item_gate_id": str(gate.id)}}
    await db.flush()
    return task


async def test_done_task_gate_not_accepted_is_unverified(db_session, test_project):
    run, meeting = await _setup(db_session, test_project)
    task = await _done_task_with_gate(db_session, test_project, run, "open")
    await _action_item(db_session, meeting, task_id=task.id, status="task_created")
    entry = await _one(db_session, run)
    assert (entry["state"], entry["fulfilled"]) == ("done_unverified", False)


async def test_done_task_with_accepted_gate_fulfilled(db_session, test_project):
    run, meeting = await _setup(db_session, test_project)
    task = await _done_task_with_gate(db_session, test_project, run, "accepted")
    await _action_item(db_session, meeting, task_id=task.id, status="task_created")
    entry = await _one(db_session, run)
    assert (entry["state"], entry["fulfilled"]) == ("task_done", True)


async def test_gateless_done_task_fulfilled(db_session, test_project):
    run, meeting = await _setup(db_session, test_project)
    task = await _task(db_session, test_project, run, status="done")
    await _action_item(db_session, meeting, task_id=task.id, status="task_created")
    entry = await _one(db_session, run)
    assert (entry["state"], entry["fulfilled"]) == ("task_done", True)


async def test_completed_graph_fulfilled(db_session, test_project):
    run, meeting = await _setup(db_session, test_project)
    graph = Graph(name=f"g-{uuid.uuid4()}", version=1, definition={})
    db_session.add(graph)
    await db_session.flush()
    graph_run = GraphRun(graph_id=graph.id, project_id=test_project.id, current_node="end", status="completed")
    db_session.add(graph_run)
    await db_session.flush()
    await _action_item(db_session, meeting, creates_graph=True, graph_run_id=graph_run.id)
    entry = await _one(db_session, run)
    assert (entry["state"], entry["fulfilled"]) == ("graph_completed", True)


@pytest.mark.parametrize("status", ["done", "resolved", "waived", "cancelled"])
async def test_explicit_resolution(db_session, test_project, status):
    run, meeting = await _setup(db_session, test_project)
    await _action_item(db_session, meeting, status=status)
    entry = await _one(db_session, run)
    assert (entry["state"], entry["fulfilled"]) == ("resolved", True)


async def test_other_run_commitment_ignored(db_session, test_project):
    run, _meeting_row = await _setup(db_session, test_project)
    other_run, other_meeting = await _setup(db_session, test_project)
    await _action_item(db_session, other_meeting)
    assert await meeting_commitments(db_session, run) == []
    assert len(await meeting_commitments(db_session, other_run)) == 1


async def test_historical_tag_only_delegation_linked_via_fallback(db_session, test_project):
    run, meeting = await _setup(db_session, test_project)
    item = await _action_item(db_session, meeting)
    failed = await _task(db_session, test_project, run, status="failed")
    await _delegation(db_session, run, failed, [f"meeting_action_item:{str(item.id).upper()}"])
    entry = await _one(db_session, run)
    assert (entry["state"], entry["task_id"]) == ("task_failed", str(failed.id))
    assert item.task_id is None  # derived only, no backfill write


async def test_delegation_executor_links_item(db_session, test_project):
    agent = _agent("delegate", capabilities=["implementation"])
    db_session.add(agent)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    source = await _task(db_session, test_project, run)
    item = await _action_item(db_session, await _meeting(db_session, test_project, source))
    request = _delegation_request(agent.id)
    request["inputs"] = [f" meeting_action_item:{str(item.id).upper()} "]

    action = await service.execute_create_delegation_task_action(
        db_session, run_id=run.id, request=request, idempotency_key="run:mc:kind:create_delegation_task:a",
    )

    await db_session.refresh(item)
    assert (item.task_id, item.status) == (action.target_id, "task_created")
    assert (await _one(db_session, run))["state"] == "task_in_flight"


async def test_delegation_executor_ignores_foreign_run_item(db_session, test_project):
    agent = _agent("delegate", capabilities=["implementation"])
    db_session.add(agent)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    _other_run, other_meeting = await _setup(db_session, test_project)
    item = await _action_item(db_session, other_meeting)
    request = _delegation_request(agent.id)
    request["inputs"] = [f"meeting_action_item:{item.id}"]

    await service.execute_create_delegation_task_action(
        db_session, run_id=run.id, request=request, idempotency_key="run:mc:kind:create_delegation_task:b",
    )

    await db_session.refresh(item)
    assert (item.task_id, item.status) == (None, "open")


async def _delegate(service, db, run, agent, item, key):
    request = _delegation_request(agent.id)
    request["inputs"] = [f"meeting_action_item:{item.id}"]
    request["budget"] = {"max_tokens": 4000}
    return await service.execute_create_delegation_task_action(
        db, run_id=run.id, request=request, idempotency_key=f"run:mc:kind:create_delegation_task:{key}",
    )


async def test_redelegation_repoints_dead_task_not_live_one(db_session, test_project):
    agent = _agent("delegate", capabilities=["implementation"])
    db_session.add(agent)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    source = await _task(db_session, test_project, run)
    item = await _action_item(db_session, await _meeting(db_session, test_project, source))
    first = await _delegate(service, db_session, run, agent, item, "r1")
    assert item.task_id == first.target_id
    # live task: a second delegation must not repoint
    await _delegate(service, db_session, run, agent, item, "r2")
    assert item.task_id == first.target_id
    first_task = await db_session.get(Task, first.target_id)
    first_task.status = "failed"
    await db_session.flush()
    second = await _delegate(service, db_session, run, agent, item, "r3")
    assert item.task_id == second.target_id != first.target_id
    assert item.status == "task_created"


async def test_cancelled_task_commitment_and_progress_follow_up(db_session, test_project):
    from huddleroom.services.orchestration_progress_view import OrchestrationProgressView

    goal, run = await _run_with_criteria(db_session, test_project, [])
    source = await _task(db_session, test_project, run)
    task = await _task(db_session, test_project, run, status="cancelled")
    item = await _action_item(
        db_session, await _meeting(db_session, test_project, source), task_id=task.id, status="task_created",
    )
    await _delegation(db_session, run, task, [f"meeting_action_item:{item.id}"])
    entry = await _one(db_session, run)
    assert entry["state"] == "task_failed" and entry["suggested_actions"] == ["create_delegation_task", "resolve_meeting_action_item"]
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    follow_up = next(f for f in situation.untracked_follow_ups if f["id"] == str(item.id))
    assert follow_up["kind"] == "meeting_action_item"
    assert follow_up["suggested_actions"] == ["create_delegation_task", "resolve_meeting_action_item"]
    assert "cancelled" in follow_up["summary"]


async def test_graph_in_flight_vs_completed(db_session, test_project):
    run, meeting = await _setup(db_session, test_project)
    graph = Graph(name=f"g-{uuid.uuid4()}", version=1, definition={})
    db_session.add(graph)
    await db_session.flush()
    graph_run = GraphRun(graph_id=graph.id, project_id=test_project.id, current_node="n", status="active")
    db_session.add(graph_run)
    await db_session.flush()
    await _action_item(db_session, meeting, creates_graph=True, graph_run_id=graph_run.id)
    entry = await _one(db_session, run)
    assert (entry["state"], entry["fulfilled"], entry["suggested_actions"]) == ("graph_in_flight", False, ["resolve_meeting_action_item"])
    graph_run.status = "completed"
    await db_session.flush()
    assert (await _one(db_session, run))["state"] == "graph_completed"


async def _graph_item(db, project, meeting, status):
    graph = Graph(name=f"g-{uuid.uuid4()}", version=1, definition={})
    db.add(graph)
    await db.flush()
    graph_run = GraphRun(graph_id=graph.id, project_id=project.id, current_node="n", status=status)
    db.add(graph_run)
    await db.flush()
    return await _action_item(db, meeting, description="Ship it", creates_graph=True, graph_run_id=graph_run.id)


@pytest.mark.parametrize("status", ["failed", "cancelled"])
async def test_failed_graph_commitment_and_follow_up(db_session, test_project, status):
    from huddleroom.services.orchestration_progress_view import OrchestrationProgressView
    run, meeting = await _setup(db_session, test_project)
    item = await _graph_item(db_session, test_project, meeting, status)
    entry = await _one(db_session, run)
    assert (entry["state"], entry["fulfilled"], entry["reason"]) == ("graph_failed", False, f"graph run {status}")
    assert entry["suggested_actions"] == ["create_delegation_task", "ask_human", "resolve_meeting_action_item"]
    goal = await db_session.get(OrchestrationGoal, run.goal_id)
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert situation.untracked_follow_ups == [{
        "kind": "meeting_action_item", "id": str(item.id),
        "summary": f"Ship it (graph {status}; delegate or ask owner)",
        "suggested_actions": ["create_delegation_task", "ask_human", "resolve_meeting_action_item"],
    }]


async def test_active_graph_in_flight_no_follow_up(db_session, test_project):
    from huddleroom.services.orchestration_progress_view import OrchestrationProgressView
    run, meeting = await _setup(db_session, test_project)
    await _graph_item(db_session, test_project, meeting, "active")
    assert (await _one(db_session, run))["state"] == "graph_in_flight"
    goal = await db_session.get(OrchestrationGoal, run.goal_id)
    assert (await OrchestrationProgressView().build(db_session, goal, run)).untracked_follow_ups == []


async def test_tag_only_delegation_failed_then_redelegated(db_session, test_project):
    from huddleroom.services.orchestration_progress_view import OrchestrationProgressView
    run, meeting = await _setup(db_session, test_project)
    item = await _action_item(db_session, meeting, description="Do it")
    dead = await _task(db_session, test_project, run, status="failed")
    await _delegation(db_session, run, dead, [f"meeting_action_item:{item.id}"])
    entry = await _one(db_session, run)
    assert (entry["state"], entry["task_id"]) == ("task_failed", str(dead.id))
    goal = await db_session.get(OrchestrationGoal, run.goal_id)
    ups = (await OrchestrationProgressView().build(db_session, goal, run)).untracked_follow_ups
    (follow_up,) = [u for u in ups if u["kind"] == "meeting_action_item"]
    assert follow_up["id"] == str(item.id) and follow_up["suggested_actions"] == ["create_delegation_task", "resolve_meeting_action_item"]
    live = await _task(db_session, test_project, run, status="in_progress")
    await _delegation(db_session, run, live, [f"meeting_action_item:{item.id}"])
    assert (await _one(db_session, run))["state"] == "task_in_flight"
    ups = (await OrchestrationProgressView().build(db_session, goal, run)).untracked_follow_ups
    assert not [u for u in ups if u["kind"] == "meeting_action_item"]


@pytest.mark.parametrize("new_status,state,fulfilled", [("active", "graph_in_flight", False), ("completed", "graph_completed", True)])
async def test_relinked_graph_run_clears_failure(db_session, test_project, new_status, state, fulfilled):
    from huddleroom.services.orchestration_progress_view import OrchestrationProgressView
    run, meeting = await _setup(db_session, test_project)
    item = await _graph_item(db_session, test_project, meeting, "failed")
    assert (await _one(db_session, run))["state"] == "graph_failed"
    graph = Graph(name=f"g-{uuid.uuid4()}", version=1, definition={})
    db_session.add(graph)
    await db_session.flush()
    new = GraphRun(graph_id=graph.id, project_id=test_project.id, current_node="n", status=new_status)
    db_session.add(new)
    await db_session.flush()
    item.graph_run_id = new.id
    await db_session.flush()
    entry = await _one(db_session, run)
    assert (entry["state"], entry["fulfilled"]) == (state, fulfilled)
    goal = await db_session.get(OrchestrationGoal, run.goal_id)
    ups = (await OrchestrationProgressView().build(db_session, goal, run)).untracked_follow_ups
    assert not [u for u in ups if u["kind"] == "meeting_action_item"]


@pytest.mark.parametrize("task_status", ["in_progress", "failed"])
async def test_waiver_preserves_linked_work_and_first_audit(db_session, test_project, task_status):
    from sqlalchemy import select
    from huddleroom.models.meeting import MeetingEvent
    from huddleroom.services.orchestration_meeting_commitments import waive_meeting_action_item
    run, meeting = await _setup(db_session, test_project)
    task = await _task(db_session, test_project, run, status=task_status)
    item = await _action_item(db_session, meeting, task_id=task.id, status="task_created")
    actor = uuid.uuid4()
    row, changed = await waive_meeting_action_item(db_session, item_id=item.id, reason="  No longer relevant  ", actor_user_id=actor)
    assert changed and row.status == "dismissed"
    _, changed_again = await waive_meeting_action_item(db_session, item_id=item.id, reason="Second reason")
    assert not changed_again
    await db_session.refresh(task)
    assert task.status == task_status and row.task_id == task.id
    events = list((await db_session.scalars(select(MeetingEvent).where(MeetingEvent.meeting_id == meeting.id))).all())
    assert len(events) == 1
    assert events[0].actor_user_id == actor
    assert events[0].payload["reason"] == "No longer relevant"
    assert (await _one(db_session, run))["fulfilled"]


@pytest.mark.parametrize("status", ["done", "dismissed"])
async def test_waiver_terminal_is_inert(db_session, test_project, status):
    from huddleroom.services.orchestration_meeting_commitments import waive_meeting_action_item
    _, meeting = await _setup(db_session, test_project)
    item = await _action_item(db_session, meeting, status=status)
    row, changed = await waive_meeting_action_item(db_session, item_id=item.id, reason="Irrelevant")
    assert not changed and row.status == status


async def test_waiver_completed_linked_work_is_inert(db_session, test_project):
    from huddleroom.services.orchestration_meeting_commitments import waive_meeting_action_item
    run, meeting = await _setup(db_session, test_project)
    task = await _task(db_session, test_project, run, status="done")
    item = await _action_item(db_session, meeting, task_id=task.id, status="task_created")
    row, changed = await waive_meeting_action_item(db_session, item_id=item.id, reason="Irrelevant")
    assert not changed and row.status == "task_created"


@pytest.mark.parametrize("reason", ["", "   ", "x" * 501, None, 12])
async def test_waiver_requires_short_nonblank_reason(db_session, test_project, reason):
    from huddleroom.services.orchestration_meeting_commitments import waive_meeting_action_item
    _, meeting = await _setup(db_session, test_project)
    item = await _action_item(db_session, meeting)
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as error:
        await waive_meeting_action_item(db_session, item_id=item.id, reason=reason)
    assert error.value.status_code == 422
    assert item.status == "open"


async def test_dismissed_item_is_not_relinked(db_session, test_project):
    from huddleroom.services.orchestration_meeting_commitments import link_meeting_action_items
    run, meeting = await _setup(db_session, test_project)
    old = await _task(db_session, test_project, run, status="failed")
    item = await _action_item(db_session, meeting, task_id=old.id, status="dismissed")
    new = await _task(db_session, test_project, run)
    action = OrchestrationAction(run_id=run.id, action_type="create_delegation_task", idempotency_key="test:relink", request={"inputs": [f"meeting_action_item:{item.id}"]})
    await link_meeting_action_items(db_session, run.id, action, new)
    assert item.status == "dismissed" and item.task_id == old.id


async def test_waiver_action_replay_and_foreign_ownership(db_session, test_project):
    from fastapi import HTTPException
    from sqlalchemy import select
    from huddleroom.models.meeting import MeetingEvent
    service, _, run = await _make_run(db_session, test_project.id)
    source = await _task(db_session, test_project, run)
    meeting = await _meeting(db_session, test_project, source)
    item = await _action_item(db_session, meeting)
    request = {"item_id": str(item.id), "reason": "  Irrelevant  "}
    action = await service.execute_resolve_meeting_action_item_action(db_session, run_id=run.id, request=request, idempotency_key="waiver:first")
    replay = await service.execute_resolve_meeting_action_item_action(db_session, run_id=run.id, request=request, idempotency_key="waiver:first")
    assert action.id == replay.id and action.status == "completed"
    events = list((await db_session.scalars(select(MeetingEvent).where(MeetingEvent.meeting_id == meeting.id))).all())
    assert len(events) == 1 and events[0].payload["action_id"] == str(action.id)
    _, foreign_meeting = await _setup(db_session, test_project)
    foreign = await _action_item(db_session, foreign_meeting)
    with pytest.raises(HTTPException):
        await service.execute_resolve_meeting_action_item_action(db_session, run_id=run.id, request={"item_id": str(foreign.id), "reason": "Irrelevant"}, idempotency_key="waiver:foreign")
    assert foreign.status == "open"


@pytest.mark.parametrize("fence", ["paused", "cancelled_goal", "unauthorized"])
async def test_waiver_action_rechecks_lifecycle(db_session, test_project, fence):
    from fastapi import HTTPException
    service, goal, run = await _make_run(db_session, test_project.id)
    source = await _task(db_session, test_project, run)
    item = await _action_item(db_session, await _meeting(db_session, test_project, source))
    if fence == "paused":
        run.status = "paused"
    elif fence == "cancelled_goal":
        goal.status = "cancelled"
    else:
        run.phase = "baseline"
    await db_session.flush()
    with pytest.raises(HTTPException):
        await service.execute_resolve_meeting_action_item_action(db_session, run_id=run.id, request={"item_id": str(item.id), "reason": "Irrelevant"}, idempotency_key=f"waiver:fence:{fence}")
    assert item.status == "open"


async def test_concurrent_waivers_write_one_audit_and_preserve_stale_completion(tmp_path):
    import asyncio
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from huddleroom.models.base import Base
    from huddleroom.models.project import Project
    from huddleroom.models.meeting import MeetingActionItem, MeetingEvent
    from huddleroom.services.orchestration_meeting_commitments import waive_meeting_action_item
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'waiver.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with factory() as seed:
            project = Project(name="waiver race")
            seed.add(project)
            await seed.flush()
            run, meeting = await _setup(seed, project)
            item = await _action_item(seed, meeting)
            task = await _task(seed, project, run, status="in_progress")
            stale = await _action_item(seed, meeting, task_id=task.id, status="task_created")
            await seed.commit()
            item_id, stale_id, task_id, meeting_id = item.id, stale.id, task.id, meeting.id
        async def submit(reason):
            async with factory() as session:
                _, changed = await waive_meeting_action_item(session, item_id=item_id, reason=reason)
                await session.commit()
                return changed
        assert sorted(await asyncio.gather(submit("First"), submit("Second"))) == [False, True]
        async with factory() as reader:
            stale_row = await reader.get(MeetingActionItem, stale_id)
            await reader.get(Task, task_id)  # load pre-completion classification inputs
            await reader.commit()
            async with factory() as finisher:
                finished = await finisher.get(Task, task_id)
                finished.status = "done"
                await finisher.commit()
            _, changed = await waive_meeting_action_item(reader, item_id=stale_id, reason="No longer relevant")
            assert not changed and stale_row.status == "task_created"
            events = list((await reader.scalars(select(MeetingEvent).where(MeetingEvent.meeting_id == meeting_id))).all())
            assert len(events) == 1 and events[0].payload["reason"] in {"First", "Second"}
    finally:
        await engine.dispose()


async def test_waiver_audit_and_event_roll_back_together(db_session, test_project):
    from sqlalchemy import select
    from huddleroom.models.meeting import MeetingEvent
    from huddleroom.models.event_log import EventLog
    from huddleroom.services.orchestration_meeting_commitments import waive_meeting_action_item
    _, meeting = await _setup(db_session, test_project)
    item = await _action_item(db_session, meeting)
    item_id = item.id
    nested = await db_session.begin_nested()
    await waive_meeting_action_item(db_session, item_id=item_id, reason="Irrelevant")
    assert len(list((await db_session.scalars(select(EventLog).where(EventLog.event_type == "meeting.action_item_waived"))).all())) == 1
    await nested.rollback()
    await db_session.refresh(item)
    assert item.status == "open"
    assert not list((await db_session.scalars(select(MeetingEvent).where(MeetingEvent.meeting_id == meeting.id))).all())
    assert not list((await db_session.scalars(select(EventLog).where(EventLog.event_type == "meeting.action_item_waived"))).all())


async def test_blocked_goal_can_waive_failed_work_without_unblocking_it(db_session, test_project):
    from huddleroom.schemas.orchestration_recovery import CorrectiveDisposition
    service, goal, run = await _make_run(db_session, test_project.id)
    goal.status = "blocked"
    run.status = "blocked"
    run.active_blockers = [{"code": "failed_work", "message": "The linked work failed"}]
    source = await _task(db_session, test_project, run)
    failed = await _task(db_session, test_project, run, status="failed")
    gate = OrchestrationGate(run_id=run.id, success_criterion_key="delegated", gate_type="work_completed", required_evidence={}, status="failed")
    db_session.add(gate)
    await db_session.flush()
    item = await _action_item(db_session, await _meeting(db_session, test_project, source), task_id=failed.id, status="task_created")
    action = await service.execute_resolve_meeting_action_item_action(
        db_session, run_id=run.id, request={"item_id": str(item.id), "reason": "Obligation is irrelevant"}, idempotency_key="waiver:blocked",
    )
    assert action.status == "completed" and item.status == "dismissed"
    for row in (goal, run, failed, gate):
        await db_session.refresh(row)
    assert goal.status == "blocked" and run.status == "blocked" and run.phase == "authorized"
    assert run.active_blockers == [{"code": "failed_work", "message": "The linked work failed"}]
    assert failed.status == "failed" and gate.status == "failed" and item.task_id == failed.id
    assert await service.classify_corrective_action(db_session, run, "retry_task", {"task_id": str(failed.id)}, source="model") == CorrectiveDisposition.REQUIRES_RECOVERY
