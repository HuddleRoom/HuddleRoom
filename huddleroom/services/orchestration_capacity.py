"""Root-wide task capacity: one slot count and one ranked activation order per finite goal tree."""
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping

from sqlalchemy import exists, select

from huddleroom.models.orchestration import ACTIVE_RUN_STATUSES, OrchestrationGoal, OrchestrationRun
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.task_service import TaskService

CAPACITY_WAIT = "capacity_wait"
_MAX_WALK = 64


async def root_and_depth(db, goal_id) -> tuple[uuid.UUID, int]:
    """Nearest ancestor-or-self whose parent is None or Continuous, and goal's depth below it."""
    current, depth = goal_id, 0
    parent_id = await db.scalar(select(OrchestrationGoal.parent_goal_id).where(OrchestrationGoal.id == current))
    while parent_id is not None and depth < _MAX_WALK:
        parent = (await db.execute(select(OrchestrationGoal.goal_type, OrchestrationGoal.parent_goal_id).where(
            OrchestrationGoal.id == parent_id
        ))).first()
        if parent is None or parent.goal_type == "continuous":
            break
        current, parent_id, depth = parent_id, parent.parent_goal_id, depth + 1
    return current, depth


async def root_goal_id(db, goal_id) -> uuid.UUID:
    return (await root_and_depth(db, goal_id))[0]


async def _subtree_goal_ids(db, root_id) -> list[uuid.UUID]:
    # ponytail: a Continuous root's cases are their own roots, so the Continuous goal counts alone.
    from huddleroom.services.orchestration_roadmap_service import subtree_goal_ids
    goal_type = await db.scalar(select(OrchestrationGoal.goal_type).where(OrchestrationGoal.id == root_id))
    return [root_id] if goal_type == "continuous" else await subtree_goal_ids(db, root_id)


async def subtree_run_ids(db, root_id) -> list[uuid.UUID]:
    goal_ids = await _subtree_goal_ids(db, root_id)
    return list(await db.scalars(select(OrchestrationRun.id).where(
        OrchestrationRun.goal_id.in_(goal_ids), OrchestrationRun.status.in_(ACTIVE_RUN_STATUSES)
    )))


def _in_runs(run_ids):
    return Task.metadata_["orchestration"]["run_id"].as_string().in_([str(run_id) for run_id in run_ids])


async def available_slots(db, root_id) -> int:
    """Root limit minus ready/in-progress tasks and tasks with a live session, across every active run of the tree."""
    from huddleroom.services.orchestration_roadmap_service import expansion_scope
    root = await db.get(OrchestrationGoal, root_id)
    limit = (await expansion_scope(db, root))[2]["max_concurrent_tasks"]
    run_ids = await subtree_run_ids(db, root_id)
    if not run_ids:
        return max(0, limit)
    run_tasks = select(Task.id).where(_in_runs(run_ids))
    used = set(await db.scalars(select(Task.id).where(_in_runs(run_ids), Task.status.in_(["ready", "in_progress"]))))
    used.update(await db.scalars(select(Session.task_id).where(
        Session.status.in_(["pending", "running"]), Session.task_id.in_(run_tasks)
    )))
    return max(0, limit - len(used))


async def queued_tasks(db, root_id) -> list[Task]:
    """Never-started assigned backlog tasks of authorized, running runs of active goals, oldest first."""
    goal_ids = await _subtree_goal_ids(db, root_id)
    run_ids = list(await db.scalars(
        select(OrchestrationRun.id).join(OrchestrationGoal, OrchestrationGoal.id == OrchestrationRun.goal_id).where(
            OrchestrationRun.goal_id.in_(goal_ids), OrchestrationRun.status == "running",
            OrchestrationRun.phase == "authorized", OrchestrationGoal.status == "active",
        )
    ))
    if not run_ids:
        return []
    return list(await db.scalars(
        select(Task).where(
            _in_runs(run_ids), Task.status == "backlog", Task.assigned_to.is_not(None),
            ~exists().where(Session.task_id == Task.id),
        ).order_by(Task.created_at.asc(), Task.id.asc())
    ))


def _dependency_counts(items) -> Counter:
    return Counter(
        str(dep).removeprefix("plan_item:") for item in items for dep in (getattr(item, "depends_on", None) or [])
    )


def rank_queued(items, plan_order: Mapping[str, int], unlocks: Mapping[str, int] | None = None) -> list:
    """Priority, then dependents unlocked, then plan order, then stable id. Works for tasks and plan items."""
    if unlocks is None:
        unlocks = _dependency_counts(items)
    return sorted(items, key=lambda item: (
        -item.priority, -unlocks.get(str(item.id), 0), plan_order.get(str(item.id), float("inf")), str(item.id)
    ))


async def activate_ready(
    db, root_id, *, is_blocked: Callable[[object, OrchestrationRun, Task], Awaitable[object]]
) -> dict:
    """Start the best-ranked startable queued tasks into the free root slots; the rest wait (capacity_wait)."""
    slots = await available_slots(db, root_id)
    queued = await queued_tasks(db, root_id)
    runs: dict[str, OrchestrationRun] = {}
    startable = []
    for task in queued:
        run_id = task.metadata_["orchestration"]["run_id"]
        if run_id not in runs:
            runs[run_id] = await db.get(OrchestrationRun, uuid.UUID(run_id))
        if await is_blocked(db, runs[run_id], task) is None:
            startable.append(task)
    counts = _dependency_counts(queued)
    plan_order, unlocks = {}, {}
    for task in queued:
        meta = task.metadata_.get("orchestration") or {}
        key = str(task.id)
        if meta.get("plan_index") is not None:
            plan_order[key] = meta["plan_index"]
        unlocks[key] = max(meta.get("plan_unlocks") or 0, counts.get(key, 0))
    started = []
    for task in rank_queued(startable, plan_order, unlocks):
        if slots <= 0:
            break
        await TaskService().auto_start(db, task)
        if task.status == "in_progress":  # ponytail: local decrement is valid under the root lock
            started.append(task)
            slots -= 1
    return {"started": started, "queued": [t for t in startable if t not in started], "available_slots": slots}
