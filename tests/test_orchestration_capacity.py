"""Root-wide capacity: slot counting across a finite goal tree and ranked activation."""
import asyncio
import uuid
from datetime import timedelta
from types import SimpleNamespace

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from huddleroom.models.orchestration import (
    OrchestrationAction, OrchestrationBudgetReservation, OrchestrationDecision, OrchestrationGoal, OrchestrationRoadmapItem,
    OrchestrationRun,
)
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.services.orchestration_capacity import (
    activate_ready, available_slots, queued_tasks, rank_queued, root_goal_id, subtree_run_ids,
)
from huddleroom.services.orchestration_roadmap_service import descendant_count, expansion_scope
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.task_service import TaskService
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
    await _authorize(db, run)
    return goal, run


async def _authorize(db, run):
    # Effective authorization gates activation and claims; a NULL fingerprint is a grandfathered legacy Start.
    db.add(OrchestrationAction(run_id=run.id, idempotency_key=f"run:{run.id}:kind:authorize_execution",
                               action_type="authorize_execution", status="completed"))
    await db.flush()


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
    db.add(OrchestrationBudgetReservation(  # the parent release this child runs under
        parent_goal_id=parent.id, continuous_origin_key=goal.continuous_origin_key, child_goal_id=goal.id,
        allocation={},
    ))
    await db.flush()
    await _authorize(db, run)
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


async def test_rank_order_priority_then_unlocks_then_plan_order():
    a, b, c, d, e = (item(x) for x in "abcde")
    a.priority = 90
    items = [e, d, c, b, a]
    # b unlocks one dependent; c and d tie on everything but plan order; e has no plan order
    items.append(item("dep", priority=1, depends_on=["plan_item:b"]))
    ranked = rank_queued(items, {"c": 1, "d": 0, "dep": 9})
    assert [i.id for i in ranked] == ["a", "b", "d", "c", "e", "dep"]


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
    assert result["queued"] == []  # the inactive assignee is not queued at all
    assert (await db_session.get(Task, refused.id)).status == "backlog"


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


async def _stopped_sibling_setup(db, project, stop):
    agent = await _agent_row(db)
    root, _ = await _root(db, project)
    c1, run1 = await _child(db, project, root)
    c2, run2 = await _child(db, project, root)
    stop(run2)
    await db.flush()
    stopped_task = await _task(db, project, run2, agent=agent)
    return agent, root, (c1, run1), (c2, run2), stopped_task


def _exhaust_budget(run):
    run.budget_state = {"status": "exceeded"}


async def test_activate_ready_skips_budget_exhausted_sibling_run(db_session, test_project):
    _, root, (c1, run1), _, stopped = await _stopped_sibling_setup(db_session, test_project, _exhaust_budget)
    ok = await _task(db_session, test_project, run1, agent=await _agent_row(db_session))
    assert await OrchestrationService()._start_released_tasks(db_session, c1, run1) == 1
    assert (await db_session.get(Task, ok.id)).status == "in_progress"
    assert (await db_session.get(Task, stopped.id)).status == "backlog"
    assert await db_session.scalar(select(func.count()).select_from(Session).where(Session.task_id == stopped.id)) == 0


@pytest.mark.parametrize("kind", ["repeated_failure", "plan_criterion_integrity", "goal_definition_analyzer_error"])
async def test_activate_ready_skips_sibling_with_stop_blocker(db_session, test_project, kind):
    def block(run):
        run.active_blockers = [{"kind": kind, "reason": "stopped"}]

    _, root, (c1, run1), _, stopped = await _stopped_sibling_setup(db_session, test_project, block)
    await _task(db_session, test_project, run1, agent=await _agent_row(db_session))
    await OrchestrationService()._start_released_tasks(db_session, c1, run1)
    assert (await db_session.get(Task, stopped.id)).status == "backlog"
    assert await db_session.scalar(select(func.count()).select_from(Session).where(Session.task_id == stopped.id)) == 0


async def test_release_headroom_ignores_stopped_sibling_queued_tasks(db_session, test_project):
    _, root, _, _, _ = await _stopped_sibling_setup(db_session, test_project, _exhaust_budget)
    assert await OrchestrationService()._release_headroom(db_session, root.id) == await available_slots(db_session, root.id)


async def test_inactive_assignee_does_not_pin_headroom(db_session, test_project):
    inactive = _agent("inactive", ["implementation"])
    inactive.is_active = False
    db_session.add(inactive)
    await db_session.flush()
    root, run = await _root(db_session, test_project)
    await _task(db_session, test_project, run, agent=inactive)
    assert await queued_tasks(db_session, root.id) == []
    assert await OrchestrationService()._release_headroom(db_session, root.id) == 2


async def test_rank_queued_ties_keep_creation_order(db_session, test_project):
    agent = await _agent_row(db_session)
    root, run = await _root(db_session, test_project)
    older = await _task(db_session, test_project, run, agent=agent, id=uuid.UUID(int=2 ** 127 + 1))
    newer = await _task(db_session, test_project, run, agent=agent, id=uuid.UUID(int=1))
    older.created_at = newer.created_at - timedelta(seconds=5)
    await db_session.flush()
    assert str(older.id) > str(newer.id)
    ranked = rank_queued(await queued_tasks(db_session, root.id), {})
    assert [t.id for t in ranked] == [older.id, newer.id]


async def test_activate_ready_refusal_then_success_starts_next(db_session, test_project, monkeypatch):
    agent = await _agent_row(db_session)
    root, run = await _root(db_session, test_project)
    first = await _task(db_session, test_project, run, agent=agent, priority=90)
    second = await _task(db_session, test_project, run, agent=agent, priority=10)
    first_id, second_id = first.id, second.id
    real = TaskService.transition_status

    async def refusing(self, db, project_id, task_id, *args, **kwargs):
        if task_id == first_id:
            # SQLAlchemy expires only objects dirtied inside the rolled-back savepoint: dirty the next task too.
            for written in (task_id, second_id):
                (await db.get(Task, written)).title = "written then rolled back"
            await db.flush()
            raise HTTPException(status_code=409, detail="refused")
        return await real(self, db, project_id, task_id, *args, **kwargs)

    monkeypatch.setattr(TaskService, "transition_status", refusing)
    result = await activate_ready(db_session, root.id, is_blocked=_unblocked)
    assert [t.id for t in result["started"]] == [second_id]
    assert (await db_session.get(Task, first_id)).status == "backlog"
    assert (await db_session.get(Task, second_id)).status == "in_progress"


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
    assert situation.authorization_wait is None and "authorization_wait" not in situation.as_context()
    assert "authorization_wait" not in (await db_session.get(Task, low.id)).metadata_["orchestration"]
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


async def test_queued_planning_task_is_not_starved_by_later_lower_ranked_work(db_session, test_project):
    agent = await _agent_row(db_session)
    root, run = await _root(db_session, test_project)
    audits = [await _task(db_session, test_project, run, "in_progress", agent=agent, priority=10) for _ in range(2)]
    planner = await _task(db_session, test_project, run, agent=agent, priority=70)
    later = [await _task(db_session, test_project, run, agent=agent, priority=50) for _ in range(2)]
    service = OrchestrationService()
    await service.tick(db_session, run.id)
    assert (await db_session.get(Task, planner.id)).status == "backlog"  # root full: queued, not dropped
    order = [t.id for t in rank_queued(await queued_tasks(db_session, root.id), {})]
    assert order[0] == planner.id and order == [t.id for t in rank_queued(await queued_tasks(db_session, root.id), {})]
    for audit in audits:  # both audits finish
        audit.status = "done"
    await db_session.flush()
    await service.tick(db_session, run.id)
    assert (await db_session.get(Task, planner.id)).status == "in_progress"
    started = [(await db_session.get(Task, t.id)).status for t in later]
    assert sorted(started) == ["backlog", "in_progress"]  # planner + one best-ranked later task


# --- 13e: every goal lock takes the finite-tree root lock first (committed sessions, real concurrency) ---

@pytest_asyncio.fixture
async def committed_tree_cleanup(test_engine, db_session, monkeypatch):
    # Committed ticks dispatch sessions after commit; never start a real runner.
    monkeypatch.setattr("huddleroom.workers.task_runner.register_session", lambda *args, **kwargs: None)
    yield
    await db_session.rollback()
    async with test_engine.begin() as connection:
        await connection.execute(delete(OrchestrationBudgetReservation))
        await connection.execute(delete(OrchestrationRoadmapItem))
        parents = select(OrchestrationGoal.parent_goal_id).where(OrchestrationGoal.parent_goal_id.is_not(None))
        while (await connection.execute(delete(OrchestrationGoal).where(  # leaves first: nested trees
            OrchestrationGoal.parent_goal_id.is_not(None), OrchestrationGoal.id.not_in(parents),
        ))).rowcount:
            pass


@pytest.mark.parametrize("holder,waiter,blocked", [
    ("root", "grand", True), ("child", "sibling", True), ("continuous", "case", False),
])
async def test_goal_lock_takes_the_finite_root_lock_first(
    db_session, test_project, test_engine, committed_tree_cleanup, holder, waiter, blocked,
):
    root, _ = await _root(db_session, test_project)
    child, _ = await _child(db_session, test_project, root, "roadmap")
    sibling, _ = await _child(db_session, test_project, root)
    grand, _ = await _child(db_session, test_project, child)
    continuous, _ = await _root(db_session, test_project, "continuous")
    case, _ = await _child(db_session, test_project, continuous)
    goals = {"root": root.id, "child": child.id, "sibling": sibling.id, "grand": grand.id,
             "continuous": continuous.id, "case": case.id}
    await db_session.commit()
    service = OrchestrationService()
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    held, release = asyncio.Event(), asyncio.Event()

    async def hold():
        async with factory() as db, service._lock_goal_for_baseline_transition(db, goals[holder]):
            held.set()
            await release.wait()

    async def lock_waiter():
        async with factory() as db, service._lock_goal_for_baseline_transition(db, goals[waiter]):
            return True

    holding = asyncio.create_task(hold())
    await held.wait()
    if blocked:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(lock_waiter(), 0.2)
    else:
        assert await asyncio.wait_for(lock_waiter(), 2)  # a Continuous case is its own root
    release.set()
    await holding
    assert await asyncio.wait_for(lock_waiter(), 5)


async def test_racing_sibling_ticks_cannot_overfill_root_slots(
    db_session, test_project, test_engine, committed_tree_cleanup,
):
    agent = await _agent_row(db_session)
    root, root_run = await _root(db_session, test_project)
    runs = []
    for _ in range(2):
        _, run = await _child(db_session, test_project, root)
        run.plan_state = {"child_delta_baseline": {"status": "accepted"}}
        runs.append(run)
    await _task(db_session, test_project, root_run, "in_progress", agent=agent)
    tasks = [await _task(db_session, test_project, run, agent=agent) for run in runs]
    run_ids, task_ids = [run.id for run in runs], [task.id for task in tasks]
    await db_session.commit()
    factory = async_sessionmaker(test_engine, expire_on_commit=False)

    async def tick(run_id):
        async with factory() as db:
            return await OrchestrationService().tick(db, run_id)

    await asyncio.wait_for(asyncio.gather(*(tick(run_id) for run_id in run_ids)), timeout=30)
    async with factory() as reader:
        statuses = [(await reader.get(Task, task_id)).status for task_id in task_ids]
        assert sorted(statuses) == ["backlog", "in_progress"]
        assert await reader.scalar(select(func.count()).select_from(Session).where(Session.task_id.in_(task_ids))) == 1
        assert await available_slots(reader, root.id) == 0


async def _accept_child_roadmap(db, project, child, items):
    from huddleroom.models.artifact import Artifact
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
    from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService, parse_roadmap_items

    service = OrchestrationService()
    run = await db.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    artifact = Artifact(project_id=project.id, name="c-plan", artifact_type="plan", status="draft",
                        metadata_={"plan_items": items})
    normalized = [x.model_dump(mode="json") for x in parse_roadmap_items(items, set(child.budget["caps"]))]
    decision = OrchestrationAuthorityDecision(
        goal_id=child.id, run_id=run.id, authority="human", title="approve", question="approve",
        decision_key=f"roadmap_plan:{service._accepted_plan_fingerprint(normalized)}", status="answered",
        selected_option="approve", decided_by_user_id=child.manager_user_id,
    )
    db.add_all([artifact, decision]); await db.flush()
    version = await OrchestrationRoadmapService(service).accept_version(
        db, child, run, artifact, {"kind": "authority_decision", "id": str(decision.id)},
    )
    run.plan_state = {**run.plan_state, "status": "accepted", "accepted_artifact_id": str(artifact.id),
                      "roadmap_version_id": str(version.id), "roadmap_version": version.version,
                      "accepted_plan_fingerprint": version.fingerprint}
    await db.flush()
    return run


async def _roadmap_children(db, project, keys, child_items):
    """Committed root Roadmap with released Roadmap children, each with its own accepted Roadmap of child_items."""
    from tests.test_orchestration_roadmap_children import accepted_goal_roadmap, goal_item

    allocation = {"max_tokens": 400, "max_turns": 2, "max_hours": 2}
    roadmap, root, root_run, _ = await accepted_goal_roadmap(db, project, [
        {**goal_item(key, allocation), "goal_type": "roadmap"} for key in keys])
    for _ in keys:
        await roadmap.advance(db, root, root_run)
    children = list(await db.scalars(select(OrchestrationGoal).where(OrchestrationGoal.parent_goal_id == root.id)))
    assert len(children) == len(keys)
    run_ids = [(await _accept_child_roadmap(db, project, child, [goal_item(key, {
        "max_tokens": tokens, "max_turns": 1, "max_hours": 1}) for key, tokens in child_items])).id for child in children]
    return root, [child.id for child in children], run_ids


async def _race_ticks(test_engine, run_ids):
    factory = async_sessionmaker(test_engine, expire_on_commit=False)

    async def tick(run_id):
        async with factory() as db:
            return await OrchestrationService().tick(db, run_id)

    results = await asyncio.wait_for(asyncio.gather(*(tick(run_id) for run_id in run_ids)), timeout=30)
    return results, factory()


async def test_racing_cousin_releases_cannot_exceed_the_root_descendant_limit(
    db_session, test_project, test_engine, committed_tree_cleanup,
):
    root, child_ids, run_ids = await _roadmap_children(db_session, test_project, ("c1", "c2"), [("g", 100)])
    root.orchestrator_context = {**root.orchestrator_context, "expansion_policy": {"max_descendant_goals": 3}}
    await db_session.commit()
    _, reader = await _race_ticks(test_engine, run_ids)
    async with reader:
        assert await reader.scalar(select(func.count()).select_from(OrchestrationGoal).where(
            OrchestrationGoal.parent_goal_id.in_(child_ids))) == 1
        assert await reader.scalar(select(func.count()).select_from(OrchestrationBudgetReservation).where(
            OrchestrationBudgetReservation.parent_goal_id.in_(child_ids))) == 1
        blockers = [b for run_id in run_ids for b in (await reader.get(OrchestrationRun, run_id)).active_blockers
                    if isinstance(b, dict) and b.get("kind") == "expansion_limit"]
        assert len(blockers) == 1 and "max_descendant_goals 3" in blockers[0]["reason"]


async def test_racing_ticks_of_a_child_roadmap_reserve_its_budget_once(
    db_session, test_project, test_engine, committed_tree_cleanup,
):
    # Both ticks take the root lock, then the child lock; the lowered child cap fits only one grandchild.
    _, (child_id,), (run_id,) = await _roadmap_children(db_session, test_project, ("c1",), [("g1", 200), ("g2", 200)])
    child = await db_session.get(OrchestrationGoal, child_id)
    child.budget = {"caps": {**child.budget["caps"], "max_tokens": "300"}}
    await db_session.commit()
    results, reader = await _race_ticks(test_engine, [run_id, run_id])
    assert sorted(result["authorized_execution"].get("reason", "") for result in results) == ["", "budget_wait"]
    async with reader:
        reservations = list(await reader.scalars(select(OrchestrationBudgetReservation).where(
            OrchestrationBudgetReservation.parent_goal_id == child_id)))
        assert len(reservations) == 1
        assert await reader.scalar(select(func.count()).select_from(OrchestrationGoal).where(
            OrchestrationGoal.parent_goal_id == child_id)) == 1


async def test_pending_renewal_child_tick_is_noop(db_session, test_project):
    agent = await _agent_row(db_session)
    root, _ = await _root(db_session, test_project)
    _, run = await _child(db_session, test_project, root)
    run.plan_state = {"child_delta_baseline": {"status": "pending_renewal"}}
    task = await _task(db_session, test_project, run, agent=agent)
    await db_session.flush()
    await OrchestrationService().tick(db_session, run.id)
    await db_session.refresh(task)
    assert task.status == "backlog"
    assert await db_session.scalar(select(func.count()).select_from(Session).where(Session.task_id == task.id)) == 0
    assert await db_session.scalar(
        select(func.count()).select_from(OrchestrationDecision).where(OrchestrationDecision.run_id == run.id)
    ) == 0
