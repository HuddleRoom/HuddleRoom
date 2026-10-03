"""Event-first, provider-free scheduling for authorized orchestration runs."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from time import monotonic
from typing import Any
from uuid import UUID

from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError, OperationalError

from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationGoal,
    OrchestrationRun,
    OrchestrationSchedulerState,
)
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.models.artifact import Artifact
from huddleroom.models.meeting import Meeting
from huddleroom.models.protocol import ProtocolInstance
from huddleroom.config import settings


SUPPORTED_EVENTS = frozenset({
    "task.status_changed", "task.assigned", "session.created", "session.started",
    "session.completed", "session.failed", "session.cancelled", "session.resumed",
    "authority.decision_resolved", "artifact.created", "artifact.content_changed",
    "artifact.breaking_change", "meeting.scheduled",
    "meeting.concluded", "protocol.completed", "protocol.failed",
    "protocol.state_transitioned", "orchestration.run_completed",
    "orchestration.steering_changed",
})


class OrchestrationSupervisionScheduler:
    """Coalesce committed events; execution itself stays in OrchestrationService.tick."""

    def __init__(self, tick=None, judge=None) -> None:
        self._tick_impl = tick
        self._judge_impl = judge

    @staticmethod
    def _parse_time(value: str) -> datetime:
        parsed = datetime.fromisoformat(value)
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)

    async def _tick(self, db, run_id, *, local_only=False):
        if self._tick_impl is not None:
            return await self._tick_impl(db, run_id, local_only=local_only)
        from huddleroom.services.orchestration_service import OrchestrationService
        return await OrchestrationService().tick(db, run_id, local_only=local_only)

    async def _judge(self, payload):
        if self._judge_impl is not None:
            return await self._judge_impl(payload)
        from huddleroom.services.orchestration_supervision_analyzer import OrchestrationSupervisionAnalyzer
        return await OrchestrationSupervisionAnalyzer().assess(payload)

    @staticmethod
    async def _judgment_payload(db, service, context, goal, run):
        """Build the complete decision input before making its bounded transport copy."""
        snapshot = await context.build(db, goal, run, body_limit=None)
        payload = {
            "goal": snapshot["goal"],
            "run_id": snapshot["run"]["id"],
            "contract_version": await service.supervision._contract_version(db, goal, run),
            "context": snapshot,
        }
        return payload, context.fingerprint_snapshot(payload)

    # pylint: disable=too-many-boolean-expressions
    async def _claim_judgment(self, run_id, now):
        """Commit a complete local snapshot before the untrusted provider call."""
        from huddleroom.database import AsyncSessionLocal
        from huddleroom.services.orchestration_service import OrchestrationService
        from huddleroom.services.orchestration_supervision_context import OrchestrationSupervisionContextBuilder

        async with AsyncSessionLocal() as db:
            service, context = OrchestrationService(), OrchestrationSupervisionContextBuilder()
            await service.tick(db, run_id, local_only=True)
            run = await db.get(OrchestrationRun, run_id, populate_existing=True)
            if run is None:
                await db.rollback()
                return None
            # pylint: disable=protected-access
            async with service._lock_goal_for_baseline_transition(db, run.goal_id):
                goal = await db.get(OrchestrationGoal, run.goal_id, populate_existing=True)
                run = await db.get(OrchestrationRun, run_id, populate_existing=True)
                state = dict(run.supervision_state or {})
                if (
                    goal is None
                    or goal.status not in {"active", "blocked"}
                    or run.status not in {"running", "blocked"}
                    or run.phase != "authorized"
                    or state.get("judgment_in_flight")
                    or not state.get("needs_judgment")
                    or not state.get("judgment_due_at")
                    or self._parse_time(state["judgment_due_at"]) > now
                ):
                    await db.commit()
                    return None
                canonical_payload, fingerprint = await self._judgment_payload(
                    db, service, context, goal, run,
                )
                state.update({
                    "judgment_in_flight": True,
                    "judgment_dirty": False,
                    "context_fingerprint": fingerprint,
                })
                run.supervision_state = state
                payload = context.provider_snapshot(canonical_payload)
                await db.commit()
                return {
                    "goal_id": goal.id,
                    "run_id": run.id,
                    "fingerprint": fingerprint,
                        "payload": payload,
                }

    # pylint: disable=too-many-locals,too-many-branches
    async def _finish_judgment(self, claim, assessment) -> bool:
        """Fence provider output against the durable snapshot, then apply once."""
        from huddleroom.database import AsyncSessionLocal
        from huddleroom.services.orchestration_service import OrchestrationService
        from huddleroom.services.orchestration_supervision_context import OrchestrationSupervisionContextBuilder

        async with AsyncSessionLocal() as db:
            service, context = OrchestrationService(), OrchestrationSupervisionContextBuilder()
            try:
                # pylint: disable=protected-access
                async with service._lock_goal_for_baseline_transition(db, claim["goal_id"]):
                    goal = await db.get(OrchestrationGoal, claim["goal_id"], populate_existing=True)
                    run = await db.get(OrchestrationRun, claim["run_id"], populate_existing=True)
                    state = dict(run.supervision_state or {}) if run is not None else {}
                    current = None
                    if goal is not None and run is not None:
                        _payload, current = await self._judgment_payload(db, service, context, goal, run)
                    allowed = (
                        goal is not None and run is not None and goal.status in {"active", "blocked"}
                        and run.status in {"running", "blocked"} and run.phase == "authorized"
                        and state.get("judgment_in_flight") and current == claim["fingerprint"]
                        and not state.get("judgment_dirty")
                        and not service._budget_is_exhausted(run)  # pylint: disable=protected-access
                    )
                    if allowed:
                        # apply_disposition rechecks current control and contract
                        # while this reentrant goal lock is held.
                        await service.supervision.apply_disposition(db, goal, run, assessment)
                    if run is not None:
                        state = dict(run.supervision_state or {})
                        state["judgment_in_flight"] = False
                        if allowed:
                            state["judgment_dirty"] = False
                            state["judgment_failures"] = 0
                            state["judgment_due_at"] = (
                                _utcnow() + timedelta(seconds=settings.orchestration_semantic_progress_seconds)
                            ).isoformat()
                        run.supervision_state = state
                await db.commit()
                return allowed
            except Exception:
                await db.rollback()
                # A failed provider disposition must never strand the flight bit.
                await self._update_judgment_failure(claim["run_id"], _utcnow(), failure=False)
                raise

    async def _update_judgment_failure(self, run_id, now, *, failure: bool) -> None:
        """Release a failed claim without overwriting a concurrent event's deadline."""
        from huddleroom.database import AsyncSessionLocal
        from huddleroom.services.orchestration_service import OrchestrationService

        async with AsyncSessionLocal() as db:
            candidate = await db.get(OrchestrationRun, run_id)
            if candidate is None:
                await db.rollback()
                return
            service = OrchestrationService()
            async with service._lock_goal_for_baseline_transition(db, candidate.goal_id):
                goal = await db.get(OrchestrationGoal, candidate.goal_id, populate_existing=True)
                run = await db.get(OrchestrationRun, run_id, populate_existing=True)
                if (
                    goal is None or run is None or goal.status not in {"active", "blocked"}
                    or run.status not in {"running", "blocked"} or run.phase != "authorized"
                ):
                    await db.commit()
                    return
                state = dict(run.supervision_state or {})
                state["judgment_in_flight"] = False
                if failure:
                    state["judgment_dirty"] = True
                    state["judgment_failures"] = int(state.get("judgment_failures", 0)) + 1
                    retry_due = now + timedelta(seconds=settings.orchestration_reconcile_interval_seconds)
                    due = state.get("judgment_due_at")
                    if not due or self._parse_time(due) > retry_due:
                        state["judgment_due_at"] = retry_due.isoformat()
                run.supervision_state = state
                await db.commit()

    async def _evaluate_claimed(self, run_id, now) -> int:
        for attempt in range(2):
            claim = await self._claim_judgment(run_id, now)
            if claim is None:
                return 0
            try:
                assessment = await self._judge(claim["payload"])
                return int(await self._finish_judgment(claim, assessment))
            except Exception:
                # Each retry obtains a new local snapshot in a new transaction.
                await self._update_judgment_failure(run_id, now, failure=True)
        return 0

    async def record_event(
        self, db, event_type: str, payload: dict[str, Any], *, event_id=None, now=None, project_id=None,
    ) -> list[UUID]:
        """Persist a non-sliding dirty deadline for directly identified active runs."""
        if event_type not in SUPPORTED_EVENTS or not isinstance(payload, dict):
            return []
        now = now or _utcnow()
        now = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
        ids: set[UUID] = set()
        for key in ("run_id", "orchestration_run_id"):
            try:
                if payload.get(key):
                    ids.add(UUID(str(payload[key])))
            except (TypeError, ValueError):
                pass
        goal_id = payload.get("goal_id")
        if goal_id:
            try:
                ids.update((await db.scalars(select(OrchestrationRun.id).where(
                    OrchestrationRun.goal_id == UUID(str(goal_id)),
                    OrchestrationRun.status.in_(("running", "blocked")),
                ))).all())
            except (TypeError, ValueError):
                pass
        task_id = payload.get("task_id")
        if task_id:
            try:
                task = await db.get(Task, UUID(str(task_id)))
                if project_id is not None and (task is None or task.project_id != project_id):
                    task = None
                metadata = dict(task.metadata_ or {}) if task is not None else {}
                run_id = metadata.get("orchestration", {}).get("run_id")
                if run_id:
                    ids.add(UUID(str(run_id)))
            except (TypeError, ValueError):
                pass
        session_id = payload.get("session_id")
        if session_id:
            try:
                session = await db.get(Session, UUID(str(session_id)))
                if session is not None and project_id is not None and session.project_id != project_id:
                    session = None
                if session is not None and session.task_id:
                    task = await db.get(Task, session.task_id)
                    if project_id is not None and (task is None or task.project_id != project_id):
                        task = None
                    run_id = (task.metadata_ or {}).get("orchestration", {}).get("run_id") if task else None
                    if run_id:
                        ids.add(UUID(str(run_id)))
            except (TypeError, ValueError):
                pass
        decision_id = payload.get("decision_id")
        if decision_id:
            try:
                decision = await db.get(OrchestrationAuthorityDecision, UUID(str(decision_id)))
                if decision is not None and decision.run_id:
                    ids.add(decision.run_id)
            except (TypeError, ValueError):
                pass
        action_id = payload.get("action_id")
        if action_id:
            try:
                action = await db.get(OrchestrationAction, UUID(str(action_id)))
                if action is not None:
                    ids.add(action.run_id)
            except (TypeError, ValueError):
                pass
        # Events commonly name an object one hop away from the orchestrated
        # task. Resolve that durable chain before declaring the event unrelated.
        for key, model, task_attr in (
            ("artifact_id", Artifact, "linked_task_id"),
            ("meeting_id", Meeting, "source_task_id"),
            ("protocol_instance_id", ProtocolInstance, "linked_task_id"),
        ):
            try:
                value = payload.get(key)
                owner = await db.get(model, UUID(str(value))) if value else None
                if project_id is not None and (owner is None or owner.project_id != project_id):
                    owner = None
                task_id = getattr(owner, task_attr, None) if owner is not None else None
                task = await db.get(Task, task_id) if task_id else None
                if project_id is not None and (task is None or task.project_id != project_id):
                    task = None
                run_id = (task.metadata_ or {}).get("orchestration", {}).get("run_id") if task else None
                if run_id:
                    ids.add(UUID(str(run_id)))
            except (TypeError, ValueError):
                pass
        # A child run's completion is also a durable state change for every
        # active ancestor that owns it.
        for run_id in tuple(ids):
            goal_id = await db.scalar(select(OrchestrationRun.goal_id).where(OrchestrationRun.id == run_id))
            while goal_id is not None:
                parent_id = await db.scalar(select(OrchestrationGoal.parent_goal_id).where(
                    OrchestrationGoal.id == goal_id
                ))
                if parent_id is None:
                    break
                parent_run_id = await db.scalar(select(OrchestrationRun.id).where(
                    OrchestrationRun.goal_id == parent_id,
                    OrchestrationRun.status.in_(("running", "blocked")),
                ).order_by(OrchestrationRun.created_at.desc()).limit(1))
                if parent_run_id is not None:
                    ids.add(parent_run_id)
                goal_id = parent_id
        if not ids:
            return []
        runs = list((await db.scalars(select(OrchestrationRun).join(OrchestrationGoal).where(
            OrchestrationRun.id.in_(ids), OrchestrationRun.status.in_(("running", "blocked")),
            OrchestrationGoal.status.in_(("active", "blocked")),
            *((OrchestrationGoal.project_id == project_id,) if project_id is not None else ()),
        ))).all())
        from huddleroom.services.orchestration_service import OrchestrationService
        service = OrchestrationService()
        affected = []
        for candidate in runs:
            async with service._lock_goal_for_baseline_transition(db, candidate.goal_id):
                goal = await db.get(OrchestrationGoal, candidate.goal_id, populate_existing=True)
                run = await db.get(OrchestrationRun, candidate.id, populate_existing=True)
                if (
                    goal is None
                    or run is None
                    or (project_id is not None and goal.project_id != project_id)
                    or goal.status not in {"active", "blocked"}
                    or run.status not in {"running", "blocked"}
                    or run.phase != "authorized"
                ):
                    continue
                state = dict(run.supervision_state or {})
                state["judgment_dirty"] = True
                state["last_event_id"] = str(event_id) if event_id else state.get("last_event_id")
                due = state.get("judgment_due_at")
                candidate_due = now + timedelta(seconds=settings.orchestration_event_coalesce_seconds)
                if not due or self._parse_time(due) > candidate_due:
                    state["judgment_due_at"] = candidate_due.isoformat()
                run.supervision_state = state
                affected.append(run.id)
        await db.flush()
        return affected

    async def evaluate_run(self, db, run_id, *, now=None) -> int:
        now = now or _utcnow()
        now = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
        run = await db.get(OrchestrationRun, run_id)
        if run is None:
            return 0
        state = dict(run.supervision_state or {})
        due = state.get("judgment_due_at")
        if state.get("judgment_in_flight") or not due or self._parse_time(due) > now:
            return 0
        if self._tick_impl is not None:
            state["judgment_dirty"], state["judgment_in_flight"] = False, True
            run.supervision_state = state
            try:
                await self._tick(db, run.id, local_only=True)
                return 1
            finally:
                await db.refresh(run)
                state = dict(run.supervision_state or {})
                state["judgment_in_flight"] = False
                run.supervision_state = state
        return await self._evaluate_claimed(run.id, now)

    async def _claim_sweep_pass(self, now):
        from huddleroom.database import AsyncSessionLocal
        slot = str(int(now.timestamp()) // settings.orchestration_reconcile_interval_seconds)
        async with AsyncSessionLocal() as db:
            try:
                state = await db.get(OrchestrationSchedulerState, "supervision")
                if state is None:
                    try:
                        async with db.begin_nested():
                            db.add(OrchestrationSchedulerState(name="supervision"))
                            await db.flush()
                    except IntegrityError:
                        # Another worker initialized the singleton; CAS below decides the winner.
                        pass
                result = await db.execute(
                    update(OrchestrationSchedulerState)
                    .where(
                        OrchestrationSchedulerState.name == "supervision",
                        or_(
                            OrchestrationSchedulerState.pass_key.is_(None),
                            OrchestrationSchedulerState.pass_key != slot,
                        ),
                    )
                    .values(pass_key=slot)
                    .returning(OrchestrationSchedulerState.name, OrchestrationSchedulerState.cursor_goal_id)
                )
                claimed = result.one_or_none()
                if claimed is None:
                    await db.rollback()
                    return False, None
                cursor = claimed.cursor_goal_id
                await db.commit()
                return True, cursor
            except OperationalError as exc:
                if db.get_bind().dialect.name != "sqlite" or "database is locked" not in str(exc).lower():
                    raise
                await db.rollback()
                return False, None

    async def _next_sweep_goal(self, cursor):
        from huddleroom.database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            query = select(OrchestrationGoal.id).where(
                OrchestrationGoal.status.in_(("active", "blocked")),
            ).order_by(OrchestrationGoal.id)
            if cursor is not None:
                query = query.where(OrchestrationGoal.id > cursor)
            goal_id = (await db.scalars(query.limit(1))).first()
            await db.rollback()
            return goal_id

    async def _advance_sweep_cursor(self, goal_id):
        from huddleroom.database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            state = await db.get(OrchestrationSchedulerState, "supervision")
            if state is not None:
                state.cursor_goal_id = goal_id
                await db.commit()

    async def _sweep_one(self, goal_id, timeout_seconds, now):
        from huddleroom.database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            run_id = await db.scalar(select(OrchestrationRun.id).where(
                OrchestrationRun.goal_id == goal_id,
                OrchestrationRun.status.in_(("running", "blocked")),
            ).order_by(OrchestrationRun.created_at.desc()).limit(1))
            if run_id is not None:
                await asyncio.wait_for(
                    self._tick(db, run_id, local_only=True),
                    timeout=max(0.01, timeout_seconds),
                )
                run = await db.get(OrchestrationRun, run_id, populate_existing=True)
                await db.commit()
                state = dict(run.supervision_state or {}) if run is not None else {}
                due = state.get("judgment_due_at")
                if (
                    state.get("needs_judgment")
                    and not state.get("judgment_in_flight")
                    and due
                    and self._parse_time(due) <= now
                ):
                    return run_id
            return None

    async def sweep(self, db, *, timeout_seconds=5, goal_limit=100, now=None, collect_due=False):
        """Run a bounded, UUID-cursor fair pass. Advance after every attempt."""
        now = now or _utcnow()
        claimed, cursor = await self._claim_sweep_pass(now)
        if not claimed:
            return [] if collect_due else 0
        start, processed, wrapped, due_run_ids = monotonic(), 0, False, []
        while processed < goal_limit and monotonic() - start < timeout_seconds:
            goal_id = await self._next_sweep_goal(cursor)
            if goal_id is None:
                if cursor is None or wrapped:
                    break
                cursor, wrapped = None, True
                continue
            try:
                due_run_id = await self._sweep_one(goal_id, min(1, timeout_seconds), now)
                if due_run_id is not None:
                    due_run_ids.append(due_run_id)
            except Exception:
                pass
            finally:
                # Cursor advancement is independent of an attempted goal's transaction.
                await self._advance_sweep_cursor(goal_id)
                cursor, processed = goal_id, processed + 1
        return due_run_ids if collect_due else processed
