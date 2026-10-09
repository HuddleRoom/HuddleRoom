"""Root-wide capacity: slot counting across a finite goal tree and ranked activation."""
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.services.orchestration_capacity import (
    activate_ready, available_slots, queued_tasks, rank_queued, root_goal_id, subtree_run_ids,
)
from huddleroom.services.orchestration_roadmap_service import descendant_count, expansion_scope
from huddleroom.services.orchestration_service import OrchestrationService
from tests.test_orchestration_runtime_e2e import _agent

pytestmark = pytest.mark.asyncio


def item(id_, priority=50, depends_on=()):
    return SimpleNamespace(id=id_, priority=priority, depends_on=list(depends_on))


async def _root(db, project, goal_type="outcome"):
    goal, run = await OrchestrationService().create_goal(
        db, project_id=project.id, data=OrchestrationGoalCreate(objective="root", success_criteria=[]),
        created_by_user_id=None,
    )
    goal.goal_type, goal.status, run.status, run.phase = goal_type, "active", "running", "authorized"
    if goal_type == "continuous":
        goal.planning_mode = "continuous_policy"
    await db.flush()
    return goal, run


async def _child(db, project, parent, goal_type="outcome", phase="authorized"):
    goal = OrchestrationGoal(
        project_id=project.id, parent_goal_id=parent.id, objective="child", original_request="child",
        success_criteria=[], constraints={}, budget={}, goal_type=goal_type, status="active",
        parent_contract_snapshot={}, goal_delta={}, continuous_origin_key=str(uuid.uuid4()),
    )
    db.add(goal)
    await db.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase=phase)
    db.add(run)
    await db.flush()
    return goal, run


async def _task(db, project, run, status="backlog", agent=None, **kwargs):
    task = Task(project_id=project.id, title="t", status=status, assigned_to=agent.id if agent else None,
                metadata_={"orchestration": {"run_id": str(run.id)}}, **kwargs)
    db.add(task)
    await db.flush()
    return task


async def _agent_row(db):
    agent = _agent("capacity", ["implementation"])
    db.add(agent)
    await db.flush()
    return agent


async def _unblocked(db, run, task):
    return None


async def test_rank_order_priority_then_unlocks_then_plan_order_then_id():
    a, b, c, d, e = (item(x) for x in "abcde")
    a.priority = 90
    items = [e, d, c, b, a]
    # b unlocks one dependent; c and d tie on everything but plan order; e has no plan order
    items.append(item("dep", priority=1, depends_on=["plan_item:b"]))
    ranked = rank_queued(items, {"c": 1, "d": 0, "dep": 9})
    assert [i.id for i in ranked] == ["a", "b", "d", "c", "e", "dep"]
    assert [i.id for i in rank_queued([item("y"), item("x")], {})] == ["x", "y"]


async def test_slots_count_ready_in_progress_and_live_sessions_once(db_session, test_project):
    agent = await _agent_row(db_session)
    _, run = await _root(db_session, test_project)
    goal = await db_session.get(OrchestrationGoal, run.goal_id)
    assert await available_slots(db_session, goal.id) == 2
    await _task(db_session, test_project, run, "backlog")
    await _task(db_session, test_project, run, "blocked")
    await _task(db_session, test_project, run, "done")
    assert await available_slots(db_session, goal.id) == 2
    ready = await _task(db_session, test_project, run, "ready")
    assert await available_slots(db_session, goal.id) == 1
    # a live session on an already counted task does not count twice
    db_session.add(Session(project_id=test_project.id, task_id=ready.id, agent_id=agent.id, adapter_type="api", status="running"))
    await db_session.flush()
    assert await available_slots(db_session, goal.id) == 1
    # a live session on a blocked task counts once
    blocked = await _task(db_session, test_project, run, "blocked")
    db_session.add(Session(project_id=test_project.id, task_id=blocked.id, agent_id=agent.id, adapter_type="api", status="pending"))
    await db_session.flush()
    assert await available_slots(db_session, goal.id) == 0
    await _task(db_session, test_project, run, "in_progress")
    assert await available_slots(db_session, goal.id) == 0


async def test_child_and_grandchild_runs_share_the_root_count(db_session, test_project):
    root, root_run = await _root(db_session, test_project)
    child, child_run = await _child(db_session, test_project, root, "roadmap")
    grand, grand_run = await _child(db_session, test_project, child)
    assert await root_goal_id(db_session, grand.id) == root.id
    assert set(await subtree_run_ids(db_session, root.id)) == {root_run.id, child_run.id, grand_run.id}
    assert await descendant_count(db_session, root.id) == 2
    await _task(db_session, test_project, root_run, "in_progress")
    await _task(db_session, test_project, grand_run, "ready")
    assert await available_slots(db_session, root.id) == 0


async def test_continuous_case_is_its_own_root(db_session, test_project):
    cont, cont_run = await _root(db_session, test_project, "continuous")
    case, case_run = await _child(db_session, test_project, cont)
    grand, grand_run = await _child(db_session, test_project, case)
    assert await root_goal_id(db_session, case.id) == case.id
    assert await root_goal_id(db_session, grand.id) == case.id
    assert await root_goal_id(db_session, cont.id) == cont.id
    assert await subtree_run_ids(db_session, cont.id) == [cont_run.id]
    await _task(db_session, test_project, case_run, "in_progress")
    await _task(db_session, test_project, grand_run, "in_progress")
    assert await available_slots(db_session, cont.id) == 2
    assert await available_slots(db_session, case.id) == 0


async def test_limit_comes_from_the_root_policy_not_the_childs(db_session, test_project):
    root, root_run = await _root(db_session, test_project)
    child, child_run = await _child(db_session, test_project, root)
    root.orchestrator_context = {"expansion_policy": {"max_concurrent_tasks": 3}}
    child.orchestrator_context = {"expansion_policy": {"max_concurrent_tasks": 1}}
    await db_session.flush()
    assert (await expansion_scope(db_session, child))[2]["max_concurrent_tasks"] == 3
    assert await available_slots(db_session, root.id) == 3
    root.orchestrator_context = {}
    await db_session.flush()
    assert await available_slots(db_session, root.id) == 2  # settings fallback


async def test_queued_tasks_only_unstarted_assigned_backlog_of_authorized_runs(db_session, test_project):
    agent = await _agent_row(db_session)
    root, run = await _root(db_session, test_project)
    _, baseline_run = await _child(db_session, test_project, root, phase="baseline")
    keep = await _task(db_session, test_project, run, agent=agent)
    await _task(db_session, test_project, run)  # unassigned
    await _task(db_session, test_project, run, "ready", agent=agent)
    await _task(db_session, test_project, baseline_run, agent=agent)
    started = await _task(db_session, test_project, run, agent=agent)
    db_session.add(Session(project_id=test_project.id, task_id=started.id, agent_id=agent.id, adapter_type="api", status="failed"))
    await db_session.flush()
    assert [t.id for t in await queued_tasks(db_session, root.id)] == [keep.id]


async def test_activate_ready_starts_best_ranked_into_free_slots_only(db_session, test_project):
    agent = await _agent_row(db_session)
    root, run = await _root(db_session, test_project)
    low = await _task(db_session, test_project, run, agent=agent, priority=10)
    high = await _task(db_session, test_project, run, agent=agent, priority=90)
    mid = await _task(db_session, test_project, run, agent=agent, priority=50)
    result = await activate_ready(db_session, root.id, is_blocked=_unblocked)
    assert [t.id for t in result["started"]] == [high.id, mid.id]
    assert [t.id for t in result["queued"]] == [low.id]
    assert result["available_slots"] == 0
    assert (await db_session.get(Task, low.id)).status == "backlog"
    again = await activate_ready(db_session, root.id, is_blocked=_unblocked)
    assert again["started"] == [] and again["available_slots"] == 0


async def test_blocked_and_refused_tasks_do_not_consume_slots(db_session, test_project):
    agent = await _agent_row(db_session)
    inactive = _agent("inactive", ["implementation"])
    inactive.is_active = False
    db_session.add(inactive)
    await db_session.flush()
    root, run = await _root(db_session, test_project)
    blocked = await _task(db_session, test_project, run, agent=agent, priority=99)
    refused = await _task(db_session, test_project, run, agent=inactive, priority=80)
    ok1 = await _task(db_session, test_project, run, agent=agent, priority=70)
    ok2 = await _task(db_session, test_project, run, agent=agent, priority=60)

    async def is_blocked(db, run_, task):
        return ("dependency", task.id) if task.id == blocked.id else None

    result = await activate_ready(db_session, root.id, is_blocked=is_blocked)
    assert [t.id for t in result["started"]] == [ok1.id, ok2.id]
    assert {t.id for t in result["queued"]} == {refused.id}


async def test_attach_plan_item_metadata_records_index_and_unlocks():
    from huddleroom.models.artifact import Artifact
    from huddleroom.schemas.orchestration import OrchestrationPlanItem

    raw = [
        {"id": "a", "title": "A"}, {"id": "b", "depends_on": ["a"]}, {"id": "c", "depends_on": ["plan_item:a", "b"]},
    ]
    artifact = Artifact(id=uuid.uuid4(), metadata_={"plan_items": raw})
    task = Task(metadata_={"orchestration": {"run_id": "r"}})
    plan_item = SimpleNamespace(id="a", success_criterion_keys=[], model_dump=lambda mode: {"id": "a"})
    action = SimpleNamespace(id=uuid.uuid4())
    gate = SimpleNamespace(id=uuid.uuid4())
    OrchestrationService._attach_plan_item_metadata(task, artifact, plan_item, action, gate)
    assert task.metadata_["orchestration"]["plan_index"] == 0
    assert task.metadata_["orchestration"]["plan_unlocks"] == 2
    plan_item.id = "c"
    OrchestrationService._attach_plan_item_metadata(task, artifact, plan_item, action, gate)
    assert task.metadata_["orchestration"]["plan_index"] == 2
    assert task.metadata_["orchestration"]["plan_unlocks"] == 0


async def test_release_headroom_subtracts_startable_queue_but_not_dependency_blocked(db_session, test_project):
    agent = await _agent_row(db_session)
    root, run = await _root(db_session, test_project)
    service = OrchestrationService()
    assert await service._release_headroom(db_session, root.id) == 2
    dep = await _task(db_session, test_project, run, "in_progress", agent=agent)
    blocked = await _task(db_session, test_project, run, agent=agent)
    blocked.depends_on = [str(dep.id)]
    await db_session.flush()
    assert await service._release_headroom(db_session, root.id) == 1  # blocked backlog holds no slot
    await _task(db_session, test_project, run, agent=agent)  # startable backlog holds one
    assert await service._release_headroom(db_session, root.id) == 0


async def test_tick_fills_root_slots_by_rank_queues_the_rest_and_starts_it_when_a_slot_frees(db_session, test_project):
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
    from huddleroom.services.orchestration_progress_view import OrchestrationProgressView

    agent = await _agent_row(db_session)
    root, run = await _root(db_session, test_project)
    low = await _task(db_session, test_project, run, agent=agent, priority=10)  # earliest, lowest priority
    mid = await _task(db_session, test_project, run, agent=agent, priority=50)
    high = await _task(db_session, test_project, run, agent=agent, priority=90)
    service = OrchestrationService()
    await service.tick(db_session, run.id)
    statuses = {name: (await db_session.get(Task, task.id)).status for name, task in (("low", low), ("mid", mid), ("high", high))}
    assert statuses == {"low": "backlog", "mid": "in_progress", "high": "in_progress"}
    await db_session.refresh(run)
    await db_session.refresh(root)
    situation = await OrchestrationProgressView().build(db_session, root, run)
    assert situation.capacity_wait["available_slots"] == 0
    assert situation.capacity_wait["queued_task_ids"] == [str(low.id)]
    assert not [b for b in run.active_blockers if isinstance(b, dict)]
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.run_id == run.id)) == 0
    (await db_session.get(Task, high.id)).status = "done"
    for session in await db_session.scalars(select(Session).where(Session.task_id == high.id)):
        session.status = "completed"
    await db_session.flush()
    await service.tick(db_session, run.id)
    assert (await db_session.get(Task, low.id)).status == "in_progress"


async def test_root_child_and_grandchild_ready_work_starts_at_most_two_in_total(db_session, test_project):
    agent = await _agent_row(db_session)
    root, root_run = await _root(db_session, test_project)
    child, child_run = await _child(db_session, test_project, root, "roadmap")
    grand, grand_run = await _child(db_session, test_project, child)
    tasks = [await _task(db_session, test_project, run, agent=agent) for run in (root_run, child_run, grand_run)]
    await OrchestrationService().tick(db_session, root_run.id)
    statuses = [(await db_session.get(Task, t.id)).status for t in tasks]
    assert statuses.count("in_progress") == 2 and statuses.count("backlog") == 1
    assert await available_slots(db_session, root.id) == 0
