"""Meeting action items as run commitments: derived state, plus delegation linking.

`meeting_commitments` is read-only. Task link is item.task_id, else (historical
rows) the target of a completed create_delegation_task action tagged
`meeting_action_item:<id>`. `link_meeting_action_items` is the write side used
by the delegation executor so new delegations persist the link.
"""
import uuid
from typing import Any

from fastapi import HTTPException
from sqlalchemy import case, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.graph import GraphRun
from huddleroom.models.meeting import Meeting, MeetingActionItem, MeetingEvent
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGate, OrchestrationRun
from huddleroom.models.task import Task
from huddleroom.services.orchestration_progress_view import _orchestration_meta, _string_list, _tag

OPEN_ITEM_STATUSES = ("open", "task_created")
IN_FLIGHT_TASK_STATUSES = ("backlog", "ready", "in_progress")
FAILED_TASK_STATUSES = ("failed", "blocked", "cancelled")
SUGGESTED_ACTIONS = {
    "open": ["create_delegation_task", "ask_human", "resolve_meeting_action_item"],
    "assigned": ["create_delegation_task", "ask_human", "resolve_meeting_action_item"],
    "task_failed": ["create_delegation_task", "resolve_meeting_action_item"],
    "graph_failed": ["create_delegation_task", "ask_human", "resolve_meeting_action_item"],
    "task_in_flight": ["resolve_meeting_action_item"],
    "graph_in_flight": ["resolve_meeting_action_item"],
    "done_unverified": ["request_verification", "resolve_meeting_action_item"],
}
FAILED_GRAPH_STATUSES = ("failed", "cancelled")  # engine only sets "failed"; "cancelled" is defensive
TAG_PREFIX = "meeting_action_item:"


def _uuid(value: Any) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(value))
    except ValueError:
        return None


async def _run_meeting_ids(db: AsyncSession, run_id: uuid.UUID) -> set[uuid.UUID]:
    rows = await db.execute(
        select(Meeting.id).join(Task, Task.id == Meeting.source_task_id)
        .where(Task.metadata_["orchestration"]["run_id"].as_string() == str(run_id))
    )
    return set(rows.scalars().all())


def _item_ids_from_tags(inputs: Any) -> list[uuid.UUID]:
    ids = []
    for raw in _string_list(inputs):
        tag = _tag(raw)
        if tag.startswith(TAG_PREFIX) and (item_id := _uuid(tag[len(TAG_PREFIX):])):
            ids.append(item_id)
    return ids


def _classify(item, task, gate, graph_run) -> tuple[str, bool, str | None]:
    """(state, fulfilled, reason) following the documented rule order."""
    if item.status not in OPEN_ITEM_STATUSES:
        return "resolved", True, None
    if item.creates_graph and graph_run is not None and graph_run.status == "completed":
        return "graph_completed", True, None
    if task is None:
        if item.creates_graph and graph_run is not None and graph_run.status in FAILED_GRAPH_STATUSES:
            return "graph_failed", False, f"graph run {graph_run.status}"
        if item.creates_graph and graph_run is not None:
            return "graph_in_flight", False, "graph run not completed"
        has_assignee = item.assignee_agent_id or item.assignee_user_id
        return ("assigned", False, "assigned but no task completed it") if has_assignee else (
            "open", False, "no task or resolution")
    if task.status in IN_FLIGHT_TASK_STATUSES:
        return "task_in_flight", False, "linked task still in flight"
    if task.status != "done":
        return "task_failed", False, f"linked task {task.status}"
    if gate is not None and gate.status != "accepted":
        return "done_unverified", False, "task done but plan-item gate not accepted"
    # ponytail: gate-less done tasks count as fulfilled; nothing could resolve them otherwise.
    return "task_done", True, None


async def delegation_task_fallback(db: AsyncSession, run_id: uuid.UUID) -> dict[uuid.UUID, uuid.UUID]:
    """item id -> task id from completed tagged delegations (historical rows lack item.task_id)."""
    fallback: dict[uuid.UUID, uuid.UUID] = {}
    actions = (await db.execute(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run_id,
        OrchestrationAction.action_type == "create_delegation_task",
        OrchestrationAction.status == "completed",
        OrchestrationAction.target_type == "task",
    ).order_by(OrchestrationAction.created_at.asc(), OrchestrationAction.id.asc()))).scalars().all()
    for action in actions:
        for item_id in _item_ids_from_tags((action.request or {}).get("inputs")):
            if action.target_id is not None:
                fallback[item_id] = action.target_id  # latest delegation wins
    return fallback


async def meeting_commitments(db: AsyncSession, run: OrchestrationRun) -> list[dict]:
    meeting_ids = await _run_meeting_ids(db, run.id)
    if not meeting_ids:
        return []
    items = list((await db.execute(
        select(MeetingActionItem).where(MeetingActionItem.meeting_id.in_(meeting_ids))
        .order_by(MeetingActionItem.created_at.asc(), MeetingActionItem.id.asc())
    )).scalars().all())
    fallback = await delegation_task_fallback(db, run.id)
    task_ids = {i.task_id or fallback.get(i.id) for i in items} - {None}
    tasks = {t.id: t for t in (await db.execute(select(Task).where(Task.id.in_(task_ids)))).scalars().all()} if task_ids else {}
    graph_ids = {i.graph_run_id for i in items if i.creates_graph and i.graph_run_id}
    graph_runs = {g.id: g for g in (await db.execute(select(GraphRun).where(GraphRun.id.in_(graph_ids)))).scalars().all()} if graph_ids else {}
    gate_ids = {_uuid(_orchestration_meta(t).get("plan_item_gate_id")) for t in tasks.values()} - {None}
    gates = {g.id: g for g in (await db.execute(select(OrchestrationGate).where(OrchestrationGate.id.in_(gate_ids)))).scalars().all()} if gate_ids else {}

    meetings = {m.id: m for m in (await db.scalars(select(Meeting).where(Meeting.id.in_(meeting_ids)))).all()}

    # ponytail: ~6 reads per call (supervision context + closeout); cache per tick if the fingerprint gets hot.
    result = []
    for item in items:
        task_id = item.task_id or fallback.get(item.id)
        task = tasks.get(task_id) if task_id else None
        gate = gates.get(_uuid(_orchestration_meta(task).get("plan_item_gate_id"))) if task else None
        state, fulfilled, reason = _classify(item, task, gate, graph_runs.get(item.graph_run_id))
        result.append({
            "id": str(item.id),
            "meeting_id": str(item.meeting_id),
            "meeting_title": meetings[item.meeting_id].title,
            "created_at": item.created_at.isoformat() if item.created_at is not None else None,
            "graph_run_id": str(item.graph_run_id) if item.graph_run_id else None,
            "description": item.description,
            "state": state,
            "task_id": str(task.id) if task else None,
            "reason": reason,
            "fulfilled": fulfilled,
            "suggested_actions": SUGGESTED_ACTIONS.get(state, []),
        })
    return result


async def link_meeting_action_items(
    db: AsyncSession, run_id: uuid.UUID, action: OrchestrationAction, task: Task,
) -> None:
    """Persist task_id/status on tagged items of this run's meetings; skip foreign items.

    An item linked to a different task is repointed only when that old task is dead (failed/cancelled).
    """
    item_ids = _item_ids_from_tags((action.request or {}).get("inputs"))
    if not item_ids:
        return
    meeting_ids = await _run_meeting_ids(db, run_id)
    for item_id in dict.fromkeys(item_ids):
        item = await db.get(MeetingActionItem, item_id)
        if item is None or item.meeting_id not in meeting_ids or item.status == "dismissed":
            continue
        if item.task_id is not None and item.task_id != task.id:
            old = await db.get(Task, item.task_id)
            if old is not None and old.status not in ("failed", "cancelled"):
                continue
        await db.execute(update(MeetingActionItem).where(
            MeetingActionItem.id == item.id, MeetingActionItem.status != "dismissed",
            MeetingActionItem.task_id == item.task_id,
        ).values(task_id=task.id, status=case(
            (MeetingActionItem.status == "open", "task_created"), else_=MeetingActionItem.status,
        )).execution_options(synchronize_session=False))
        await db.refresh(item)
    await db.flush()


def waiver_reason(value: Any) -> str:
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 500:
        raise HTTPException(status_code=422, detail="Waiver reason must be a nonblank string of 1–500 characters")
    return value.strip()


async def waive_meeting_action_item(
    db: AsyncSession, *, item_id: uuid.UUID, reason: str,
    actor_user_id: uuid.UUID | None = None, run_id: uuid.UUID | None = None,
    action_id: uuid.UUID | None = None,
) -> tuple[MeetingActionItem, bool]:
    """Waive only the obligation; serialize classification and transition in the caller's transaction."""
    reason = waiver_reason(reason)
    # A write before reads acquires SQLite's transaction write lock and a row lock on Postgres.
    locked = await db.execute(update(MeetingActionItem).where(MeetingActionItem.id == item_id)
                              .values(status=MeetingActionItem.status).execution_options(synchronize_session=False))
    if not locked.rowcount:
        raise HTTPException(status_code=404, detail="Action item not found")
    item = await db.get(MeetingActionItem, item_id, populate_existing=True)
    meeting = await db.get(Meeting, item.meeting_id)
    source = await db.get(Task, meeting.source_task_id, populate_existing=True) if meeting.source_task_id else None
    owning_run_id = _uuid(_orchestration_meta(source).get("run_id")) if source else None
    if run_id is not None and (owning_run_id != run_id or item.meeting_id not in await _run_meeting_ids(db, run_id)):
        raise HTTPException(status_code=404, detail="Action item not found in orchestration run")
    fallback = await delegation_task_fallback(db, owning_run_id) if owning_run_id else {}
    task_id = item.task_id or fallback.get(item.id)
    task = await db.get(Task, task_id, populate_existing=True) if task_id else None
    gate_id = _uuid(_orchestration_meta(task).get("plan_item_gate_id")) if task else None
    gate = await db.get(OrchestrationGate, gate_id, populate_existing=True) if gate_id else None
    graph = await db.get(GraphRun, item.graph_run_id, populate_existing=True) if item.graph_run_id else None
    if _classify(item, task, gate, graph)[1]:
        return item, False
    prior_status = item.status
    changed = await db.execute(update(MeetingActionItem).where(
        MeetingActionItem.id == item.id, MeetingActionItem.status.in_(OPEN_ITEM_STATUSES),
    ).values(status="dismissed").execution_options(synchronize_session=False))
    await db.refresh(item)
    if not changed.rowcount:
        return item, False
    payload = {
        "item_id": str(item.id), "prior_status": prior_status, "new_status": "dismissed",
        "reason": reason, "actor": "orchestrator" if action_id is not None else "human",
        "run_id": str(owning_run_id) if owning_run_id else None,
        "action_id": str(action_id) if action_id else None,
    }
    db.add(MeetingEvent(meeting_id=item.meeting_id, event_type="action_item_waived",
                        payload=payload, actor_user_id=actor_user_id))
    from huddleroom.services.event_bus import emit_event
    await emit_event(db, meeting.project_id, "meeting.action_item_waived",
                     {**payload, "meeting_id": str(meeting.id)}, source=payload["actor"])
    return item, True
