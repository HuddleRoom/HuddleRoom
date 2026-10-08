"""Progress view for an orchestration run: per-success-criterion state.

Pure read model. Loads the run's tasks, gates and delegation actions once,
links them to goal criteria, then derives one state per criterion.
Later progress slices extend `build` with more sections over the same rows.
"""
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.graph import GraphRun
from huddleroom.models.meeting import Meeting, MeetingActionItem
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationGate,
    OrchestrationGoal,
    OrchestrationRun,
    OrchestrationWait,
)
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.orchestration_wake_when import ORCHESTRATOR_WAIT_OWNER_TYPE

logger = logging.getLogger(__name__)
ACTIVE_TASK_STATUSES = ("backlog", "ready", "in_progress", "blocked")
# Same status sets as OrchestrationService._blocked_task_awaiting.
ACTIVE_MEETING_STATUSES = ("scheduled", "preparing", "active", "concluding")
ACTIVE_GRAPH_RUN_STATUS = "active"
FOLLOW_UP_SUMMARY_MAX = 200
FOLLOW_UP_CAP = 10
# Source ranks for untracked follow-ups; the next slice sorts by (rank, created_at, id).
RANK_MEETING_ACTION_ITEM = 0
RANK_ANSWERED_DECISION = 1
RANK_STALLED_TASK = 2
RANK_UNVERIFIED_GATE = 3
STALLED_TASK_STATUSES = ("failed", "blocked")
TASK_RECOVERY_ACTION_TYPES = ("retry_task", "reassign_task")


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


def _depends_on_ids(task: Task) -> list[uuid.UUID]:
    ids: list[uuid.UUID] = []
    for raw in task.depends_on or []:
        try:
            ids.append(uuid.UUID(str(raw)))
        except ValueError:
            continue
    return ids


def _block_reason(task: Task) -> str | None:
    meta = task.metadata_ if isinstance(task.metadata_, dict) else {}
    blocked = meta.get("blocked")
    reason = blocked.get("reason") if isinstance(blocked, dict) else None
    if not isinstance(reason, str) or not reason.strip():
        return None
    return reason.strip()


def _blocked_wait(
    task: Task, live_sessions: dict[str, str], dependencies: dict[str, tuple[str, str]],
    meetings: dict[str, str], graph_runs: dict[str, str],
) -> str | None:
    """What a blocked task waits on, in service order: unmet dependency, live session, active meeting, active graph run."""
    for dep_id in _depends_on_ids(task):
        dep = dependencies.get(str(dep_id))
        if dep is not None and dep[1] in {"backlog", "ready", "in_progress", "blocked"}:
            return f"waiting on task {dep[0]} ({dep[1]})"
    session_status = live_sessions.get(str(task.id))
    if session_status:
        return f"session {session_status}"
    meeting_title = meetings.get(str(task.id))
    if meeting_title is not None:
        return f"waiting on meeting {meeting_title}"
    graph_run_id = graph_runs.get(str(task.id))
    return f"waiting on graph run {graph_run_id}" if graph_run_id else None


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


def _select_follow_ups(rows: list[tuple[int, datetime, str, dict]]) -> list[tuple[int, datetime, str, dict]]:
    """Cap to FOLLOW_UP_CAP: one representative per kind first, then fill in global order; keep global order."""
    ordered = sorted(rows, key=lambda row: (row[0], row[1], row[2]))
    chosen: list[int] = []
    for kind in dict.fromkeys(row[3]["kind"] for row in ordered):
        group = [i for i, row in enumerate(ordered) if row[3]["kind"] == kind]
        chosen.append(next((i for i in group if ordered[i][3]["suggested_actions"]), group[0]))
    chosen = chosen[:FOLLOW_UP_CAP]
    picked = set(chosen)
    chosen += [i for i in range(len(ordered)) if i not in picked][:FOLLOW_UP_CAP - len(chosen)]
    return [ordered[i] for i in sorted(chosen)]


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
            "suggested_actions": ["create_delegation_task", "ask_human"],
        }))
    # Delegated items whose linked task died need re-delegation; the delegated_inputs filter must not hide them.
    dead = list((await db.execute(
        select(MeetingActionItem, Task)
        .join(Task, Task.id == MeetingActionItem.task_id)
        .where(
            MeetingActionItem.meeting_id.in_(meeting_ids),
            MeetingActionItem.status == "task_created",
            Task.status.in_(("failed", "cancelled")),
        )
        .order_by(MeetingActionItem.created_at.asc(), MeetingActionItem.id.asc())
    )).all()) if meeting_ids else []
    for item, linked in dead:
        suffix = f" (task {linked.status}; re-delegate)"
        rows.append((RANK_MEETING_ACTION_ITEM, _naive_utc(item.created_at), str(item.id), {
            "kind": "meeting_action_item",
            "id": str(item.id),
            "summary": (item.description[:FOLLOW_UP_SUMMARY_MAX - len(suffix)] + suffix)[:FOLLOW_UP_SUMMARY_MAX],
            "suggested_actions": ["create_delegation_task"],
        }))

    # Completed actions name the decision they applied in dispatch_contract; decision_continuation is bookkeeping, not consumption.
    applied_contracts = (await db.execute(
        select(OrchestrationAction.dispatch_contract).where(
            OrchestrationAction.run_id == run.id,
            OrchestrationAction.status == "completed",
            OrchestrationAction.action_type != "decision_continuation",
        )
    )).scalars().all()
    applied_decision_ids = {
        str(contract["applies_decision_id"])
        for contract in applied_contracts
        if isinstance(contract, dict) and contract.get("applies_decision_id")
    }
    # Process-owned decisions (source_process_run_id set) are consumed by their owning process's advance(), not as follow-ups here.
    decisions = list((await db.execute(
        select(OrchestrationAuthorityDecision)
        .where(
            OrchestrationAuthorityDecision.run_id == run.id,
            OrchestrationAuthorityDecision.source_process_run_id.is_(None),
            OrchestrationAuthorityDecision.related_action_id.is_not(None),
            OrchestrationAuthorityDecision.status == "answered",
            OrchestrationAuthorityDecision.decided_at.is_not(None),
        )
        .order_by(OrchestrationAuthorityDecision.created_at.asc(), OrchestrationAuthorityDecision.id.asc())
    )).scalars().all())
    for decision in decisions:
        if str(decision.id) in applied_decision_ids:
            continue
        # ponytail: no suggested actions; any action that applies the answer is valid. Add a mapping when the UI needs one.
        rows.append((RANK_ANSWERED_DECISION, _naive_utc(decision.created_at), str(decision.id), {
            "kind": "answered_decision",
            "id": str(decision.id),
            "summary": (
                f"{decision.title}: {decision.selected_option}" if decision.selected_option else decision.title
            )[:FOLLOW_UP_SUMMARY_MAX],
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
    latest_verification: dict[str, OrchestrationAction] = {}
    for action in tracking:
        request = action.request or {}
        if action.action_type == "request_verification":
            gate_id = request.get("gate_id")
            if isinstance(gate_id, str):
                current = latest_verification.get(gate_id)
                if current is None or (_naive_utc(action.created_at), str(action.id)) > (
                    _naive_utc(current.created_at), str(current.id)
                ):
                    latest_verification[gate_id] = action
        elif isinstance(request.get("task_id"), str):
            task_id = request["task_id"]
            created = _naive_utc(action.created_at)
            latest_recovery[task_id] = max(latest_recovery.get(task_id, created), created)
    # Gates whose producer task is in flight or failed; a failed producer is already listed as a stalled task.
    producer_gate_ids: set[str] = set()
    plan_titles: dict[str, str] = {}
    for task in tasks:
        gate_id = _orchestration_meta(task).get("plan_item_gate_id")
        if not isinstance(gate_id, str):
            continue
        plan_titles.setdefault(gate_id, task.title)
        if task.status in (*ACTIVE_TASK_STATUSES, "failed"):
            producer_gate_ids.add(gate_id)

    stalled: list[Task] = []
    for task in tasks:
        if task.status not in STALLED_TASK_STATUSES:
            continue
        recovered_at = latest_recovery.get(str(task.id))
        if recovered_at is not None and recovered_at >= _naive_utc(task.updated_at):
            continue
        stalled.append(task)
    blocked = [task for task in stalled if task.status == "blocked"]
    # Blocked tasks: batched queries for live sessions, dependency tasks, active meetings and active graph runs.
    live_sessions: dict[str, str] = {}
    dependencies: dict[str, tuple[str, str]] = {}
    meetings: dict[str, str] = {}
    graph_runs: dict[str, str] = {}
    if blocked:
        blocked_ids = [task.id for task in blocked]
        for session_task_id, session_status in (await db.execute(
            select(Session.task_id, Session.status).where(
                Session.task_id.in_(blocked_ids),
                Session.status.in_(("pending", "running")),
            ).order_by(Session.created_at.asc(), Session.id.asc())
        )).all():
            live_sessions.setdefault(str(session_task_id), session_status)
        dependency_ids = {dep for task in blocked for dep in _depends_on_ids(task)}
        if dependency_ids:
            dependencies = {
                str(dep_id): (dep_title, dep_status)
                for dep_id, dep_title, dep_status in (await db.execute(
                    select(Task.id, Task.title, Task.status).where(Task.id.in_(list(dependency_ids)))
                )).all()
            }
        for meeting_task_id, meeting_title in (await db.execute(
            select(Meeting.source_task_id, Meeting.title).where(
                Meeting.source_task_id.in_(blocked_ids),
                Meeting.status.in_(ACTIVE_MEETING_STATUSES),
            ).order_by(Meeting.created_at.asc(), Meeting.id.asc())
        )).all():
            meetings.setdefault(str(meeting_task_id), meeting_title)
        for run_task_id, graph_run_id in (await db.execute(
            select(GraphRun.linked_task_id, GraphRun.id).where(
                GraphRun.linked_task_id.in_(blocked_ids),
                GraphRun.status == ACTIVE_GRAPH_RUN_STATUS,
            ).order_by(GraphRun.created_at.asc(), GraphRun.id.asc())
        )).all():
            graph_runs.setdefault(str(run_task_id), str(graph_run_id))
    for task in stalled:
        task_id = str(task.id)
        if task.status == "failed":
            summary = f"{task.title} is failed"
            suggested = ["retry_task", "reassign_task"]
        else:
            waiting = _blocked_wait(task, live_sessions, dependencies, meetings, graph_runs)
            summary = f"{task.title} is blocked" + (f" ({waiting})" if waiting else "")
            suggested = [] if waiting else ["reassign_task", "ask_human"]
            reason = _block_reason(task)
            if reason:
                summary += f": {reason}"
        rows.append((RANK_STALLED_TASK, _naive_utc(task.created_at), task_id, {
            "kind": "stalled_task",
            "id": task_id,
            "summary": summary[:FOLLOW_UP_SUMMARY_MAX],
            "suggested_actions": suggested,
        }))

    for gate in gates:
        gate_id = str(gate.id)
        if gate.status != "open" or gate_id in producer_gate_ids:
            continue
        verification = latest_verification.get(gate_id)
        if verification is not None and verification.status != "failed":
            continue
        title = plan_titles.get(gate_id)
        if verification is None:
            summary = (
                f"{title} ({gate.gate_type} gate) awaiting verification" if title
                else f"{gate.gate_type} gate for {gate.success_criterion_key} has no verification request"
            )
        else:
            subject = f"{title} ({gate.gate_type} gate)" if title else f"{gate.gate_type} gate for {gate.success_criterion_key}"
            reason = f": {verification.error}" if verification.error else ""
            summary = f"{subject} awaiting verification; last verification failed{reason}"
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
        untracked = [row for row in follow_up_rows if row[2] not in tracked_ids]
        return ProgressSituation(
            progress_view=entries,
            untracked_follow_ups=[item for *_, item in _select_follow_ups(untracked)],
            untracked_follow_ups_total=len(untracked),
        )

    async def build_safe(self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun) -> ProgressSituation:
        goal_id, run_id = goal.id, run.id
        try:
            return await self.build(db, goal, run)
        except Exception:
            # ponytail: no savepoint; a DB-level failure would poison the transaction anyway. Savepoint if reads ever fail alone.
            logger.warning("progress view failed goal_id=%s run_id=%s", goal_id, run_id, exc_info=True)
            return ProgressSituation(error=True)
