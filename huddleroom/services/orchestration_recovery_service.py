"""Goal-scoped, durable recovery of owned orchestration attempts."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified
from fastapi import HTTPException

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun, OrchestrationWait
from huddleroom.models.session import Session
from huddleroom.models.task import Task

LIVE = frozenset(("active", "reserved", "started", "retrying"))


@dataclass(frozen=True)
class OwnedSessionSnapshot:
    session_id: uuid.UUID; task_id: uuid.UUID; action_id: uuid.UUID; runner_task_id: str | None
    session_status: str; agent_id: uuid.UUID; project_id: uuid.UUID; resumable: bool
    provider_session_id: str | None; created_at: datetime; attempt: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class GoalRecoverySnapshot:
    goal_id: uuid.UUID; run_id: uuid.UUID; project_id: uuid.UUID; goal_status: str; run_status: str; phase: str
    authority: tuple[tuple[str, object], ...]; retry_state: tuple[tuple[str, object], ...]
    budget_state: tuple[tuple[str, object], ...]; sessions: tuple[OwnedSessionSnapshot, ...]
    scheduler_ready_at: datetime


@dataclass(frozen=True)
class RunnerObservation:
    runner_task_id: str | None; state: str; observed_at: datetime


@dataclass(frozen=True)
class RecoveryResult:
    classification: str; disposition: str; goal_id: uuid.UUID
    action_id: uuid.UUID | None = None; wait_id: uuid.UUID | None = None


def _freeze(value: object) -> tuple[tuple[str, object], ...]:
    """Return an immutable, recursively scalarized record for the apply fence."""
    def scalar(item: object) -> object:
        if isinstance(item, dict):
            return tuple(sorted((str(key), scalar(nested)) for key, nested in item.items()))
        if isinstance(item, (list, tuple)):
            return tuple(scalar(nested) for nested in item)
        if isinstance(item, set):
            return tuple(sorted((scalar(nested) for nested in item), key=repr))
        return item
    return scalar(value if isinstance(value, dict) else {})


class OrchestrationRecoveryService:
    """No session-at-a-time API: snapshots, observations, then locked apply."""

    async def build_goal_snapshot(self, db: AsyncSession, goal_id: uuid.UUID,
                                  scheduler_ready_at: datetime) -> GoalRecoverySnapshot | None:
        goal = await db.scalar(select(OrchestrationGoal).where(OrchestrationGoal.id == goal_id)
                               .execution_options(populate_existing=True))
        if goal is None or goal.status not in {"active", "blocked", "paused"}:
            return None
        run = await db.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == goal.id,
            OrchestrationRun.status.in_(("running", "blocked", "paused")),
            OrchestrationRun.phase != "waiting_activation").execution_options(populate_existing=True))
        if run is None:
            return None
        rows = list((await db.execute(select(Session, Task).join(Task, Task.id == Session.task_id).where(
            Session.project_id == goal.project_id).execution_options(populate_existing=True))).all())
        owned = []
        for session, task in rows:
            from huddleroom.workers.session_tasks import orchestration_lineage_state
            lineage_owned = await orchestration_lineage_state(db, session)
            if lineage_owned is not True:
                continue
            metadata = task.metadata_ if isinstance(task.metadata_, dict) else {}
            orch = metadata["orchestration"]
            try:
                task_action_id = uuid.UUID(str(orch["action_id"]))
            except (KeyError, TypeError, ValueError):
                continue
            action = await db.scalar(select(OrchestrationAction).where(OrchestrationAction.id == task_action_id)
                                     .execution_options(populate_existing=True))
            if (str(orch.get("run_id")) != str(run.id)
                    or action is None or action.run_id != run.id or task.project_id != session.project_id
                    or task.assigned_to != session.agent_id):
                continue
            # A recovery apply may create a retry or retire this attempt.  Fence every
            # durable scalar that can alter that decision, not just its ownership ids.
            attempt = (session.metadata_ or {}).get("attempt") if isinstance(session.metadata_, dict) else {}
            attempt = attempt if isinstance(attempt, dict) else {}
            owned.append(OwnedSessionSnapshot(session.id, task.id, action.id, session.runner_task_id, session.status,
                session.agent_id, session.project_id, session.resumable, session.provider_session_id, session.created_at, _freeze({
                    "session": {
                        "task_id": str(session.task_id), "agent_id": str(session.agent_id),
                        "project_id": str(session.project_id), "adapter_type": session.adapter_type,
                        "status": session.status, "runner_task_id": session.runner_task_id,
                        "provider_session_id": session.provider_session_id, "resumable": session.resumable,
                        "created_at": session.created_at, "started_at": session.started_at,
                        "ended_at": session.ended_at, "metadata": session.metadata_ or {},
                    },
                    "task": {
                        "id": str(task.id), "project_id": str(task.project_id), "status": task.status,
                        "assigned_to": str(task.assigned_to), "adapter_type_override": task.adapter_type_override,
                        "metadata": task.metadata_ or {},
                    },
                    "action": {
                        "id": str(action.id), "run_id": str(action.run_id), "idempotency_key": action.idempotency_key,
                        "action_type": action.action_type, "status": action.status, "request": action.request or {},
                        "target_type": action.target_type, "target_id": str(action.target_id),
                        "dispatch_contract": action.dispatch_contract or {}, "budget_ledger": action.budget_ledger or {},
                    },
                    # Keep the fields classifier reads at the top level without
                    # letting untrusted attempt metadata overwrite the fence.
                    "attempt": attempt, "effect_state": attempt.get("effect_state"),
                    "usage_complete": attempt.get("usage_complete"),
                    "output_present": session.output is not None,
                })) )
        owned.sort(key=lambda item: str(item.session_id))
        return GoalRecoverySnapshot(goal.id, run.id, goal.project_id, goal.status, run.status, run.phase,
            _freeze({
                "status": goal.status, "authority_model": goal.authority_model,
                "manager_agent_id": str(goal.manager_agent_id), "manager_user_id": str(goal.manager_user_id),
                "constraints": goal.constraints or {}, "budget": goal.budget or {},
            }),
            _freeze({
                "status": run.status, "phase": run.phase, "event_cursor": run.event_cursor,
                "plan_state": run.plan_state or {}, "active_blockers": run.active_blockers or [],
                "retry_state": run.retry_state or {},
            }),
            _freeze(run.budget_state or {}), tuple(owned), scheduler_ready_at)

    async def resolve_owned_session(self, db: AsyncSession, session_id: uuid.UUID) -> OwnedSessionSnapshot | None:
        """Public resolver for generic watchdogs; never exposes ORM ownership rows."""
        session = await db.get(Session, session_id)
        if session is None or session.task_id is None:
            return None
        from huddleroom.workers.session_tasks import orchestration_lineage_state
        if await orchestration_lineage_state(db, session) is not True:
            return None
        task = await db.get(Task, session.task_id)
        metadata = task.metadata_ if task and isinstance(task.metadata_, dict) else {}
        orchestration = metadata.get("orchestration") if isinstance(metadata.get("orchestration"), dict) else {}
        try:
            action_id = uuid.UUID(str(orchestration["action_id"]))
        except (KeyError, TypeError, ValueError):
            return None
        action = await db.get(OrchestrationAction, action_id)
        if action is None:
            return None
        attempt = (session.metadata_ or {}).get("attempt") if isinstance(session.metadata_, dict) else {}
        attempt = attempt if isinstance(attempt, dict) else {}
        return OwnedSessionSnapshot(
            session.id, task.id, action.id, session.runner_task_id, session.status, session.agent_id,
            session.project_id, session.resumable, session.provider_session_id, session.created_at,
            _freeze({"attempt": attempt, "effect_state": attempt.get("effect_state"),
                     "usage_complete": attempt.get("usage_complete"),
                     "output_present": session.output is not None}),
        )

    @staticmethod
    def classify(session: OwnedSessionSnapshot, observation: RunnerObservation) -> str:
        attempt = dict(session.attempt)
        attempt_record = dict(attempt.get("attempt", ())) if isinstance(attempt.get("attempt"), tuple) else attempt

        def definitively_pre_effect() -> bool:
            """A lost registry is safe only when durable state proves no effect began."""
            version = attempt_record.get("attempt_version")
            usage_counters = ("token_count_in", "token_count_out", "input_tokens", "output_tokens", "total_tokens")
            external_markers = (
                "provider_session_id", "provider_session", "provider_id", "idempotency_key",
                "provider_idempotency_key", "result", "result_status", "result_recorded_at", "output",
            )
            return (
                isinstance(session.runner_task_id, str)
                and bool(session.runner_task_id)
                and attempt_record.get("claimed_runner_task_id") == session.runner_task_id
                and isinstance(version, int) and not isinstance(version, bool) and version > 0
                and attempt_record.get("effect_state") == "not_started"
                and attempt_record.get("usage_complete") is False
                and attempt.get("output_present") is False
                and session.provider_session_id is None
                and all(attempt_record.get(key) is None for key in external_markers)
                and all(
                    counter not in attempt_record
                    or (
                        isinstance(attempt_record[counter], (int, float))
                        and not isinstance(attempt_record[counter], bool)
                        and attempt_record[counter] == 0
                    )
                    for counter in usage_counters
                )
            )

        if session.session_status in {"completed", "cancelled"}:
            return "terminal"
        if session.session_status == "failed":
            if observation.state in LIVE and observation.runner_task_id == session.runner_task_id:
                return "unknown_external_effect"
            if (session.resumable and session.provider_session_id and attempt.get("effect_state") == "started"
                    and attempt.get("usage_complete") is True and attempt_record.get("result_status") == "failed"
                    and attempt_record.get("provider_session_id") in (None, session.provider_session_id)):
                return "resume_exact"
            return "terminal"
        if observation.state in LIVE:
            return "live"
        if observation.state in {"unknown", "terminal_failure", "terminal_revoked"} and definitively_pre_effect():
            return "interrupted_safe"
        return "unknown_external_effect"

    async def _wait(self, db: AsyncSession, run: OrchestrationRun, session: OwnedSessionSnapshot | None, reason: str):
        source = str(session.session_id) if session else "memory"
        runner = session.runner_task_id if session else "none"
        key = f"recovery:{run.id}:{source}:{runner}:{reason}"
        wait = await db.scalar(select(OrchestrationWait).where(OrchestrationWait.run_id == run.id,
            OrchestrationWait.wait_key == key, OrchestrationWait.status == "open"))
        if wait is None:
            owner = {"type": "session", "id": source} if session else {"type": "run", "id": str(run.id)}
            event_type = "session.completed"
            matcher = {"session_id": source, "runner_task_id": runner}
            expected_result = "session.completed"
            if session is None and reason == "upgrade_reconciliation":
                event_type = "memory.upgrade_resolved"
                matcher = {"goal_id": str(run.goal_id), "run_id": str(run.id),
                           "source": "accepted_source_reconciliation"}
                expected_result = "memory.upgrade_resolved"
            wait = OrchestrationWait(run_id=run.id, wait_key=key, owner=owner,
                awaited_event={"event_type": event_type, "matcher": matcher},
                due_recheck_at=datetime.now(timezone.utc) + timedelta(seconds=30),
                fallback={"action_type": "attention", "reason": reason, "expected_result": expected_result})
            db.add(wait); await db.flush()
        return wait

    async def _clear_recovery_wait(self, db: AsyncSession, run: OrchestrationRun,
                                   session: OwnedSessionSnapshot) -> None:
        wait = await db.scalar(select(OrchestrationWait).where(
            OrchestrationWait.run_id == run.id,
            OrchestrationWait.wait_key.like(f"recovery:{run.id}:{session.session_id}:%"),
            OrchestrationWait.status == "open",
        ))
        if wait is not None:
            wait.status = "cleared"
            wait.cleared_at = datetime.now(timezone.utc)

    @staticmethod
    def _current_session(sessions: tuple[OwnedSessionSnapshot, ...]) -> OwnedSessionSnapshot | None:
        """Prefer a current attempt; historical terminal attempts are fallback only."""
        if not sessions:
            return None
        rank = {"pending": 0, "running": 0, "failed": 1, "completed": 2, "cancelled": 2}
        return min(sessions, key=lambda item: (rank.get(item.session_status, 0), -item.created_at.timestamp(), str(item.session_id)))

    async def _completed_noop(self, db: AsyncSession, run: OrchestrationRun,
                              session: OwnedSessionSnapshot | None) -> OrchestrationAction:
        from huddleroom.services.orchestration_service import OrchestrationService
        service = OrchestrationService()
        source = str(session.session_id) if session else "memory"
        action = await service.reserve_action(
            db, run.id, f"recovery:{run.id}:session:{source}:terminal_noop",
            "noop", {"action_type": "noop", "session_id": source},
        )
        if action.status == "reserved":
            await service._mark_action_completed(db, action, "session" if session else "run",
                                                 session.session_id if session else run.id)
        return action

    async def apply_goal_recovery(self, db: AsyncSession, snapshot: GoalRecoverySnapshot,
                                  observations: dict[uuid.UUID, RunnerObservation],
                                  recovery_timing: dict[str, object] | None = None) -> RecoveryResult | None:
        from huddleroom.services.orchestration_service import OrchestrationService
        async with OrchestrationService()._lock_goal_for_baseline_transition(db, snapshot.goal_id):
            current = await self.build_goal_snapshot(db, snapshot.goal_id, snapshot.scheduler_ready_at)
            if current is None or current != snapshot:
                return None
            run = await db.scalar(select(OrchestrationRun).where(OrchestrationRun.id == snapshot.run_id)
                                  .execution_options(populate_existing=True))
            assert run is not None
            if recovery_timing is not None:
                state = dict(run.supervision_state or {})
                recovery = dict(state.get("recovery") or {})
                recovery["last_pass"] = dict(recovery_timing)
                recovery.update(recovery_timing)
                state["recovery"] = recovery
                run.supervision_state = state
                flag_modified(run, "supervision_state")
            from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService
            memory_marker = (run.supervision_state or {}).get("memory_upgrade_reconciled")
            session = self._current_session(snapshot.sessions)
            unresolved = int(memory_marker.get("unresolved", 0)) if isinstance(memory_marker, dict) else await OrchestrationMemoryService().reconcile_accepted_sources(
                db, snapshot.goal_id, snapshot.run_id
            )
            if session is None:
                wait = await self._wait(db, run, None, "upgrade_reconciliation") if unresolved else None
                if wait is not None:
                    state = dict(run.supervision_state or {})
                    state["memory_upgrade_reconciled"] = {
                        "run_id": str(run.id), "outcome": "unresolved", "unresolved": unresolved,
                        "wait_id": str(wait.id),
                    }
                    run.supervision_state = state
                    flag_modified(run, "supervision_state")
                    return RecoveryResult("memory_only", "memory_only", snapshot.goal_id, wait_id=wait.id)
                if not unresolved:
                    state = dict(run.supervision_state or {})
                    state["memory_upgrade_reconciled"] = {
                        "run_id": str(run.id), "outcome": "promoted", "unresolved": 0,
                    }
                    run.supervision_state = state
                    flag_modified(run, "supervision_state")
                action = await self._completed_noop(db, run, None)
                return RecoveryResult("memory_only", "memory_only", snapshot.goal_id,
                                      action_id=action.id if action else None, wait_id=wait.id if wait else None)
            if not isinstance(memory_marker, dict) and not unresolved:
                state = dict(run.supervision_state or {})
                state["memory_upgrade_reconciled"] = {
                    "run_id": str(run.id), "outcome": "promoted", "unresolved": 0,
                }
                run.supervision_state = state
                flag_modified(run, "supervision_state")
            elif unresolved and not isinstance(memory_marker, dict):
                recovery = (run.supervision_state or {}).get("recovery") or {}
                prior_sessions = recovery.get("sessions") if isinstance(recovery, dict) else {}
                # A first owned-session pass belongs to the worker.  On the next
                # pass its durable assessment leaves room for the aggregate
                # reconciliation wait, which becomes the replay marker.
                if isinstance(prior_sessions, dict) and str(session.session_id) in prior_sessions:
                    memory_wait = await self._wait(db, run, None, "upgrade_reconciliation")
                    state = dict(run.supervision_state or {})
                    state["memory_upgrade_reconciled"] = {
                        "run_id": str(run.id), "outcome": "unresolved", "unresolved": unresolved,
                        "wait_id": str(memory_wait.id),
                    }
                    run.supervision_state = state
                    flag_modified(run, "supervision_state")
            observed = observations.get(session.session_id, RunnerObservation(session.runner_task_id, "unknown", datetime.now(timezone.utc)))
            classification = self.classify(session, observed)
            if snapshot.goal_status == "paused" or snapshot.run_status == "paused":
                disposition = "control_retained"
            else:
                disposition = classification
            state = dict(run.supervision_state or {}); recovery = dict(state.get("recovery") or {})
            sessions = dict(recovery.get("sessions") or {})
            for sibling in snapshot.sessions:
                sibling_observed = observations.get(sibling.session_id, RunnerObservation(
                    sibling.runner_task_id, "unknown", datetime.now(timezone.utc),
                ))
                sibling_classification = self.classify(sibling, sibling_observed)
                sessions[str(sibling.session_id)] = {
                    "source_runner_task_id": sibling.runner_task_id,
                    "current_runner_task_id": sibling.runner_task_id,
                    "backend_observation": sibling_observed.state,
                    "classification": sibling_classification,
                    "disposition": "observed" if sibling.session_id != session.session_id else disposition,
                    "scheduler_ready_at": snapshot.scheduler_ready_at.isoformat(),
                    "assessed_at": (recovery_timing or {}).get("assessment_at", datetime.now(timezone.utc).isoformat()),
                    "action_id": None, "wait_id": None,
                }
            entry = {"source_runner_task_id": session.runner_task_id,
                "current_runner_task_id": session.runner_task_id, "backend_observation": observed.state,
                "classification": classification, "disposition": disposition,
                "scheduler_ready_at": snapshot.scheduler_ready_at.isoformat(),
                "assessed_at": (recovery_timing or {}).get("assessment_at", datetime.now(timezone.utc).isoformat())}
            if recovery_timing is not None:
                entry.update(recovery_timing)
            sessions[str(session.session_id)] = entry
            recovery["sessions"] = sessions; state["recovery"] = recovery; run.supervision_state = state
            action = None
            wait = None
            if disposition == "resume_exact":
                try:
                    action = await OrchestrationService().execute_retry_task_action(db, run.id,
                        {"action_type": "retry_task", "task_id": str(session.task_id)},
                        f"run:{run.id}:kind:retry_task:task:{session.task_id}", exact_source_session_id=session.session_id)
                except HTTPException:
                    action = None
                if action is not None and action.status == "completed":
                    await self._clear_recovery_wait(db, run, session)
                else:
                    wait = await self._wait(db, run, session, "exact_continuation_refused")
            elif disposition == "interrupted_safe":
                # The authoritative backend says this pending attempt died before
                # effects.  Retire it before TaskService's active-session guard
                # creates the replacement attempt.
                source = await db.get(Session, session.session_id)
                if source is None or source.status not in {"pending", "running"}:
                    return None
                source_action = await db.get(OrchestrationAction, session.action_id)
                settled_budget = False
                if source_action is not None and source_action.budget_ledger:
                    from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService

                    budget = OrchestrationBudgetService()
                    allocation = source_action.budget_ledger.get("allocation", {})
                    amounts = {}
                    for dimension in allocation:
                        if dimension == "max_tokens" or (dimension == "max_hours" and source.started_at is None):
                            amounts[dimension] = "0"
                        else:
                            amounts[dimension] = format(
                                budget._session_spend(source, dimension, allow_incomplete=True), "f"
                            )
                    goal = await db.get(OrchestrationGoal, snapshot.goal_id)
                    if goal is None:
                        return None
                    await budget.settle_action_budget(
                        db, goal, run, source_action, amounts, measurement_complete=True,
                        observation_id=f"action:{source_action.id}:session:{source.id}:interrupted_safe",
                    )
                    settled_budget = True
                source.status = "failed"
                source.ended_at = datetime.now(timezone.utc)
                from huddleroom.services.session_sync import sync_task_from_session
                await sync_task_from_session(db, source, settle_budget=not settled_budget)
                await self._clear_recovery_wait(db, run, session)
                action = await OrchestrationService().execute_retry_task_action(db, run.id,
                    {"action_type": "retry_task", "task_id": str(session.task_id)},
                    f"run:{run.id}:kind:retry_task:task:{session.task_id}",
                    recovery_disposition="interrupted_safe", source_session_id=session.session_id)
            elif disposition == "terminal":
                await self._clear_recovery_wait(db, run, session)
                action = await self._completed_noop(db, run, session)
            else:
                wait = await self._wait(db, run, session, disposition)
            entry["action_id"] = str(action.id) if action else None
            entry["wait_id"] = str(wait.id) if wait else None
            flag_modified(run, "supervision_state")
            return RecoveryResult(classification, disposition, snapshot.goal_id,
                                  action_id=action.id if action else None, wait_id=wait.id if wait else None)
