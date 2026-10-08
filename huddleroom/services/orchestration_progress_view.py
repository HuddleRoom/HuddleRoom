"""Progress view for an orchestration run: per-success-criterion state.

Pure read model. Loads the run's tasks, gates and delegation actions once,
links them to goal criteria, then derives one state per criterion.
Later progress slices extend `build` with more sections over the same rows.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.meeting import Meeting, MeetingActionItem
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationDecision,
    OrchestrationGate,
    OrchestrationGoal,
    OrchestrationRun,
    OrchestrationWait,
)
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.task import Task
from huddleroom.services.orchestration_wake_when import ORCHESTRATOR_WAIT_OWNER_TYPE

logger = logging.getLogger(__name__)
ACTIVE_TASK_STATUSES = ("backlog", "ready", "in_progress", "blocked")
FOLLOW_UP_SUMMARY_MAX = 200
FOLLOW_UP_CAP = 10
# Source ranks for untracked follow-ups; the next slice sorts by (rank, created_at, id).
RANK_MEETING_ACTION_ITEM = 0
RANK_ANSWERED_DECISION = 1
RANK_STALLED_TASK = 2
RANK_UNVERIFIED_GATE = 3
STALLED_TASK_STATUSES = ("failed", "blocked")
TASK_RECOVERY_ACTION_TYPES = ("retry_task", "reassign_task", "request_split")


@dataclass(frozen=True)
class ProgressSituation:
    progress_view: list[dict] = field(default_factory=list)
    untracked_follow_ups: list[dict] = field(default_factory=list)
    untracked_follow_ups_total: int = 0
    error: bool = False

    def as_context(self) -> dict[str, Any]:
        context: dict[str, Any] = {
            "progress_view": self.progress_view,
            "untracked_follow_ups": self.untracked_follow_ups,
            "untracked_follow_ups_total": self.untracked_follow_ups_total,
        }
        if self.error:
            context["progress_view_error"] = True
        return context


def _criteria(goal: OrchestrationGoal) -> list[tuple[str, str]]:
    """(key, description) per goal criterion, malformed entries and duplicate keys skipped."""
    seen: dict[str, str] = {}
    for item in goal.success_criteria or []:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        if not isinstance(key, str) or not key:
            continue
        if key not in seen:
            description = item.get("description")
            seen[key] = description if isinstance(description, str) else ""
    return list(seen.items())


def _orchestration_meta(task: Task) -> dict:
    meta = task.metadata_ or {}
    orchestration = meta.get("orchestration") if isinstance(meta, dict) else None
    return orchestration if isinstance(orchestration, dict) else {}


def _string_list(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _tag(value: str) -> str:
    """Normalize a `prefix:value` tag: strip whitespace; lowercase only the UUID of meeting_action_item tags."""
    prefix, _, rest = value.strip().partition(":")
    if prefix == "meeting_action_item":
        return f"{prefix}:{rest.strip().lower()}"
    return f"{prefix}:{rest.strip()}"


def _naive_utc(value: datetime) -> datetime:
    """SQLite returns naive datetimes; normalize aware values to naive UTC so both compare."""
    return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value


async def _follow_up_rows(
    db: AsyncSession, run: OrchestrationRun, tasks: list[Task], gates: list[OrchestrationGate],
    actions: list[OrchestrationAction],
) -> list[tuple[int, datetime, str, dict]]:
    """(rank, created_at, id, item) per untracked follow-up, in source order."""
    rows: list[tuple[int, datetime, str, dict]] = []

    meeting_ids = list((await db.execute(
        select(Meeting.id).where(Meeting.source_task_id.in_([task.id for task in tasks]))
    )).scalars().all())
    items = list((await db.execute(
        select(MeetingActionItem)
        .where(
            MeetingActionItem.meeting_id.in_(meeting_ids),
            MeetingActionItem.status == "open",
            MeetingActionItem.task_id.is_(None),
            or_(MeetingActionItem.creates_graph.is_(False), MeetingActionItem.graph_run_id.is_(None)),
        )
        .order_by(MeetingActionItem.created_at.asc(), MeetingActionItem.id.asc())
    )).scalars().all()) if meeting_ids else []
    delegated_inputs = {_tag(item) for action in actions for item in _string_list((action.request or {}).get("inputs"))}
    for item in items:
        if f"meeting_action_item:{item.id}" in delegated_inputs:
            continue
        rows.append((RANK_MEETING_ACTION_ITEM, _naive_utc(item.created_at), str(item.id), {
            "kind": "meeting_action_item",
            "id": str(item.id),
            "summary": item.description[:FOLLOW_UP_SUMMARY_MAX],
            "suggested_actions": ["create_delegation_task"],
        }))

    latest = (await db.execute(
        select(func.max(OrchestrationDecision.created_at)).where(OrchestrationDecision.run_id == run.id)
    )).scalar()
    latest = _naive_utc(latest) if latest is not None else None
    decisions = list((await db.execute(
        select(OrchestrationAuthorityDecision)
        .where(
            OrchestrationAuthorityDecision.run_id == run.id,
            OrchestrationAuthorityDecision.related_action_id.is_not(None),
            OrchestrationAuthorityDecision.status == "answered",
            OrchestrationAuthorityDecision.decided_at.is_not(None),
        )
        .order_by(OrchestrationAuthorityDecision.created_at.asc(), OrchestrationAuthorityDecision.id.asc())
    )).scalars().all())
    for decision in decisions:
        if latest is not None and _naive_utc(decision.decided_at) <= latest:
            continue
        # ponytail: no suggested actions; any action that applies the answer is valid. Add a mapping when the UI needs one.
        rows.append((RANK_ANSWERED_DECISION, _naive_utc(decision.created_at), str(decision.id), {
            "kind": "answered_decision",
            "id": str(decision.id),
            "summary": f"{decision.title}: {decision.selected_option or ''}"[:FOLLOW_UP_SUMMARY_MAX],
            "suggested_actions": [],
        }))

    # Recovery and verification requests for this run, fetched once and filtered in Python.
    tracking = list((await db.execute(
        select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id,
            OrchestrationAction.action_type.in_((*TASK_RECOVERY_ACTION_TYPES, "request_verification")),
        )
    )).scalars().all())
    latest_recovery: dict[str, datetime] = {}
    verified_gate_ids: set[str] = set()
    for action in tracking:
        request = action.request or {}
        if action.action_type == "request_verification":
            gate_id = request.get("gate_id")
            if isinstance(gate_id, str):
                verified_gate_ids.add(gate_id)
        elif isinstance(request.get("task_id"), str):
            task_id = request["task_id"]
            created = _naive_utc(action.created_at)
            latest_recovery[task_id] = max(latest_recovery.get(task_id, created), created)
    # Gates whose producer task is in flight or failed; a failed producer is already listed as a stalled task.
    producer_gate_ids: set[str] = set()
    for task in tasks:
        gate_id = _orchestration_meta(task).get("plan_item_gate_id")
        if task.status in (*ACTIVE_TASK_STATUSES, "failed") and isinstance(gate_id, str):
            producer_gate_ids.add(gate_id)

    for task in tasks:
        if task.status not in STALLED_TASK_STATUSES:
            continue
        task_id = str(task.id)
        recovered_at = latest_recovery.get(task_id)
        if recovered_at is not None and recovered_at >= _naive_utc(task.updated_at):
            continue
        # ponytail: blocked gets no actions; retry/reassign 409 unless failed, no unblock executor exists. Add one when it does.
        suggested = ["retry_task", "reassign_task"] if task.status == "failed" else []
        rows.append((RANK_STALLED_TASK, _naive_utc(task.created_at), task_id, {
            "kind": "stalled_task",
            "id": task_id,
            "summary": f"{task.title} is {task.status}"[:FOLLOW_UP_SUMMARY_MAX],
            "suggested_actions": suggested,
        }))

    for gate in gates:
        gate_id = str(gate.id)
        if gate.status != "open" or gate_id in verified_gate_ids or gate_id in producer_gate_ids:
            continue
        summary = f"{gate.gate_type} gate for {gate.success_criterion_key} has no verification request"
        rows.append((RANK_UNVERIFIED_GATE, _naive_utc(gate.created_at), gate_id, {
            "kind": "unverified_gate",
            "id": gate_id,
            "summary": summary[:FOLLOW_UP_SUMMARY_MAX],
            "suggested_actions": ["request_verification"],
        }))
    return rows


class OrchestrationProgressView:
    async def build(self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun) -> ProgressSituation:
        tasks = list((await db.execute(
            select(Task)
            .where(
                Task.project_id == goal.project_id,
                Task.metadata_["orchestration"]["run_id"].as_string() == str(run.id),
            )
            .order_by(Task.created_at.asc(), Task.id.asc())
        )).scalars().all())
        gates = list((await db.execute(
            select(OrchestrationGate)
            .where(OrchestrationGate.run_id == run.id)
            .order_by(OrchestrationGate.created_at.asc(), OrchestrationGate.id.asc())
        )).scalars().all())
        actions = list((await db.execute(
            select(OrchestrationAction)
            .where(
                OrchestrationAction.run_id == run.id,
                OrchestrationAction.action_type == "create_delegation_task",
                OrchestrationAction.status == "completed",
                OrchestrationAction.target_type == "task",
            )
            .order_by(OrchestrationAction.created_at.asc(), OrchestrationAction.id.asc())
        )).scalars().all())

        criteria = _criteria(goal)
        task_links: dict[str, set[str]] = {key: set() for key, _ in criteria}
        for task in tasks:
            for key in _string_list(_orchestration_meta(task).get("success_criterion_keys")):
                if key in task_links:
                    task_links[key].add(str(task.id))
        for action in actions:
            inputs = (action.request or {}).get("inputs")
            for item in _string_list(inputs):
                tag = _tag(item)
                if tag.startswith("criterion:") and tag[len("criterion:"):] in task_links:
                    task_links[tag[len("criterion:"):]].add(str(action.target_id))

        task_by_id = {str(task.id): task for task in tasks}
        entries = []
        for key, description in criteria:
            linked_task_ids = [tid for tid in (str(t.id) for t in tasks) if tid in task_links[key]]
            linked_gates = [
                gate for gate in gates
                if gate.success_criterion_key == key
                or key in _string_list((gate.required_evidence or {}).get("success_criterion_keys"))
            ]
            in_flight = any(task_by_id[tid].status in ACTIVE_TASK_STATUSES for tid in linked_task_ids)
            if linked_gates and all(gate.status == "accepted" for gate in linked_gates):
                state = "accepted"
            elif not in_flight and any(gate.status in ("open", "failed") for gate in linked_gates):
                state = "evidence_pending"
            elif in_flight:
                state = "in_flight"
            else:
                state = "no_work"
            entries.append({
                "criterion_key": key,
                "description": description,
                "state": state,
                "task_ids": linked_task_ids,
                "gate_ids": [str(gate.id) for gate in linked_gates],
            })
        follow_up_rows = await _follow_up_rows(db, run, tasks, gates, actions)
        waits = list((await db.execute(
            select(OrchestrationWait).where(OrchestrationWait.run_id == run.id, OrchestrationWait.status == "open")
        )).scalars().all())
        # Items an open orchestrator wait is already watching are tracked, not untracked. Owner is JSON: filter in Python.
        tracked_ids = {
            str(value)
            for wait in waits
            if isinstance(wait.owner, dict) and wait.owner.get("type") == ORCHESTRATOR_WAIT_OWNER_TYPE
            for value in ((wait.awaited_event or {}).get("matcher") or {}).values()
        }
        follow_up_rows = sorted(
            (row for row in follow_up_rows if row[2] not in tracked_ids),
            key=lambda row: (row[0], row[1], row[2]),
        )
        return ProgressSituation(
            progress_view=entries,
            untracked_follow_ups=[item for *_, item in follow_up_rows[:FOLLOW_UP_CAP]],
            untracked_follow_ups_total=len(follow_up_rows),
        )

    async def build_safe(self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun) -> ProgressSituation:
        goal_id, run_id = goal.id, run.id
        try:
            return await self.build(db, goal, run)
        except Exception:
            # ponytail: no savepoint; a DB-level failure would poison the transaction anyway. Savepoint if reads ever fail alone.
            logger.warning("progress view failed goal_id=%s run_id=%s", goal_id, run_id, exc_info=True)
            return ProgressSituation(error=True)
