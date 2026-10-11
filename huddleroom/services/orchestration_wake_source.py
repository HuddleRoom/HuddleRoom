"""Read-only proof that a noop wake condition has a relevant live producer."""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.graph import GraphRun
from huddleroom.models.meeting import Meeting
from huddleroom.models.orchestration import OrchestrationGoal
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.orchestration_steering import OrchestrationSteeringRequest
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.orchestration_dependencies import first_unmet_dependency
from huddleroom.services.orchestration_wake_when import normalize_wake_when


@dataclass(frozen=True)
class SourceRef:
    kind: str
    id: uuid.UUID


@dataclass(frozen=True)
class LiveWorkRefV1:
    kind: Literal["task", "session", "meeting", "graph_run", "steering_request", "authority_decision"]
    id: uuid.UUID
    blocked_source_kind: str
    blocked_source_id: uuid.UUID


@dataclass(frozen=True)
class WakeSourceVerdict:
    verdict: Literal["live", "eventless"]
    reason_code: str
    producer: LiveWorkRefV1 | None = None
    matched_event: dict | None = None


_TASK_LIVE_STATUSES = frozenset({"backlog", "ready", "in_progress"})
_SESSION_LIVE_STATUSES = frozenset({"pending", "running"})
_MEETING_LIVE_STATUSES = frozenset({"scheduled", "preparing", "active", "concluding"})
_STEERING_LIVE_STATUSES = frozenset({"pending", "being_considered"})


class OrchestrationWakeSourceService:
    async def evaluate(
        self,
        db: AsyncSession,
        run,
        *,
        blocked_source: SourceRef,
        wake_when: dict,
        live_work_refs: Sequence[LiveWorkRefV1],
    ) -> WakeSourceVerdict:
        """Return live only for a current, linked producer that can progress."""
        wake = normalize_wake_when(wake_when)
        relevant = [
            ref for ref in live_work_refs
            if ref.blocked_source_kind == blocked_source.kind and ref.blocked_source_id == blocked_source.id
        ]
        if not relevant:
            return self._eventless("timer_without_live_producer" if not wake["events"] else "no_relevant_producer")

        last_reason = "producer_not_live"
        for ref in relevant:
            live, reason = await self._is_live(db, run, blocked_source, ref)
            if not live:
                last_reason = reason
                continue
            if not wake["events"]:
                return WakeSourceVerdict("live", "matched_live_producer", ref)
            for event in wake["events"]:
                if self._matches(ref, event, await self._row(db, ref)):
                    return WakeSourceVerdict("live", "matched_live_producer", ref, event)
            last_reason = "unmatched_event"
        return self._eventless(last_reason)

    @staticmethod
    def _eventless(reason_code: str) -> WakeSourceVerdict:
        return WakeSourceVerdict("eventless", reason_code)

    async def _row(self, db: AsyncSession, ref: LiveWorkRefV1):
        models = {
            "task": Task,
            "session": Session,
            "meeting": Meeting,
            "graph_run": GraphRun,
            "steering_request": OrchestrationSteeringRequest,
            "authority_decision": OrchestrationAuthorityDecision,
        }
        return await db.get(models[ref.kind], ref.id)

    async def _source_task(self, db: AsyncSession, run, source: SourceRef) -> Task | None:
        if source.kind != "task":
            return None
        task = await db.get(Task, source.id)
        if task is None or not self._belongs_to_run(task, run):
            return None
        return task

    @staticmethod
    def _belongs_to_run(task: Task, run) -> bool:
        metadata = task.metadata_ if isinstance(task.metadata_, dict) else {}
        orchestration = metadata.get("orchestration") if isinstance(metadata.get("orchestration"), dict) else {}
        return orchestration.get("run_id") == str(run.id)

    async def _is_live(self, db: AsyncSession, run, source: SourceRef, ref: LiveWorkRefV1) -> tuple[bool, str]:
        row = await self._row(db, ref)
        if row is None:
            return False, "producer_missing"
        source_task = await self._source_task(db, run, source)
        if ref.kind == "task":
            if source_task is None or row.id == source_task.id or not self._belongs_to_run(row, run):
                return False, "producer_not_independently_runnable"
            unmet = await first_unmet_dependency(db, run, source_task)
            if unmet is None or unmet[0] != row.id:
                return False, "producer_unrelated"
            if row.status not in _TASK_LIVE_STATUSES or await first_unmet_dependency(db, run, row) is not None:
                return False, "producer_not_independently_runnable"
            return True, "matched_live_producer"

        if source_task is None:
            return False, "producer_unrelated"
        if ref.kind == "session":
            return (
                (row.task_id == source_task.id and row.project_id == source_task.project_id and row.status in _SESSION_LIVE_STATUSES),
                "producer_terminal" if row.status not in _SESSION_LIVE_STATUSES else "producer_unrelated",
            )
        if ref.kind == "meeting":
            return (
                (row.source_task_id == source_task.id and row.project_id == source_task.project_id and row.status in _MEETING_LIVE_STATUSES),
                "producer_terminal" if row.status not in _MEETING_LIVE_STATUSES else "producer_unrelated",
            )
        if ref.kind == "graph_run":
            return (
                (row.linked_task_id == source_task.id and row.project_id == source_task.project_id and row.status == "active"),
                "producer_terminal" if row.status != "active" else "producer_unrelated",
            )
        goal = await db.get(OrchestrationGoal, run.goal_id)
        if ref.kind == "steering_request":
            matches_source = row.target_id in {str(source_task.id), str(goal.id) if goal else ""}
            return (
                (row.goal_id == run.goal_id and row.submitted_run_id == run.id and matches_source
                 and row.status in _STEERING_LIVE_STATUSES),
                "producer_terminal" if row.status not in _STEERING_LIVE_STATUSES else "producer_unrelated",
            )
        if ref.kind == "authority_decision":
            return (
                (row.goal_id == run.goal_id and row.run_id == run.id and row.status == "pending"),
                "producer_terminal" if row.status != "pending" else "producer_unrelated",
            )
        return False, "producer_unrelated"

    @staticmethod
    def _matches(ref: LiveWorkRefV1, event: dict, row) -> bool:
        matcher = event["matcher"]
        event_type = event["event_type"]
        if ref.kind == "task":
            return event_type in {"task.status_changed", "task.assigned"} and matcher == {"task_id": str(row.id)}
        if ref.kind == "session":
            if event_type == "task.status_changed":
                return matcher == {"task_id": str(row.task_id)}
            values = {"session_id": str(row.id), "task_id": str(row.task_id)}
            if row.graph_run_id is not None:
                values["graph_run_id"] = str(row.graph_run_id)
            return event_type.startswith("session.") and bool(matcher) and all(values.get(k) == v for k, v in matcher.items())
        if ref.kind == "meeting":
            return event_type in {"meeting.scheduled", "meeting.concluded"} and matcher == {"meeting_id": str(row.id)}
        if ref.kind == "graph_run":
            return event_type in {"graph.run_completed", "graph.run_failed", "graph.run_advanced"} and matcher == {"graph_run_id": str(row.id)}
        if ref.kind == "authority_decision":
            return event_type == "authority.decision_resolved" and matcher == {"decision_id": str(row.id)}
        return False
