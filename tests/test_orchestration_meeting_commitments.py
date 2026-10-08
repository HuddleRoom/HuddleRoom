import uuid

import pytest

from huddleroom.models.graph import Graph, GraphRun
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGate
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
    assert entry["state"] == "task_failed" and entry["suggested_actions"] == ["create_delegation_task"]
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    follow_up = next(f for f in situation.untracked_follow_ups if f["id"] == str(item.id))
    assert follow_up["kind"] == "meeting_action_item"
    assert follow_up["suggested_actions"] == ["create_delegation_task"]
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
    assert (entry["state"], entry["fulfilled"], entry["suggested_actions"]) == ("graph_in_flight", False, [])
    graph_run.status = "completed"
    await db_session.flush()
    assert (await _one(db_session, run))["state"] == "graph_completed"
