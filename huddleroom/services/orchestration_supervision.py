"""Execute current, authorized supervision dispositions through the action ledger."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from collections.abc import Mapping
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGate, OrchestrationGoal, OrchestrationRun, OrchestrationWait
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.session import Session
from huddleroom.models.meeting import Meeting
from huddleroom.models.graph import GraphRun
from huddleroom.config import settings


DISPOSITIONS = frozenset({"continue", "pause", "follow_up", "verify", "reassign", "meeting", "graph", "replan", "attention"})


@dataclass(frozen=True)
class SupervisionAssessment:
    changes: tuple[dict, ...] = ()
    risks: tuple[dict, ...] = ()
    useful_learning: tuple[dict, ...] = ()
    criterion_progress: tuple[dict, ...] = ()
    disposition: dict | None = None


class OrchestrationSupervisionService:
    def __init__(self, orchestration) -> None:
        self.orchestration = orchestration

    async def _contract_version(self, db, goal, run) -> str:
        if goal.goal_type != "roadmap":
            return "start"
        from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService
        version = await OrchestrationRoadmapService(self.orchestration).current_version(db, goal.id)
        if version is None:
            raise HTTPException(status_code=409, detail="Roadmap supervision has no current contract")
        return f"plan:{version.version}"

    @staticmethod
    def _wait_contract(origin, owner, awaited_event, recheck_seconds, fallback, expected_result):
        if not isinstance(origin, str) or not origin.strip():
            raise HTTPException(status_code=409, detail="wait requires non-empty origin")
        if not isinstance(owner, Mapping) or not isinstance(owner.get("type"), str) or not owner["type"].strip() or not owner.get("id"):
            raise HTTPException(status_code=409, detail="wait requires owner identity")
        if not isinstance(awaited_event, Mapping) or not isinstance(awaited_event.get("event_type"), str) or not awaited_event["event_type"].strip():
            raise HTTPException(status_code=409, detail="wait requires awaited event_type")
        matcher = awaited_event.get("matcher", {})
        if not isinstance(matcher, Mapping):
            raise HTTPException(status_code=409, detail="wait matcher must be an object")
        if not isinstance(recheck_seconds, int) or isinstance(recheck_seconds, bool) or recheck_seconds <= 0:
            raise HTTPException(status_code=409, detail="wait requires positive recheck_seconds")
        if not isinstance(fallback, Mapping) or fallback.get("action_type") not in {"continue", "attention"}:
            raise HTTPException(status_code=409, detail="wait fallback must be continue or attention")
        if not isinstance(expected_result, str) or not expected_result.strip():
            raise HTTPException(status_code=409, detail="wait requires expected_result")
        return {
            "origin": origin.strip(), "owner": dict(owner),
            "awaited_event": {"event_type": awaited_event["event_type"].strip(), "matcher": dict(matcher)},
            "recheck_seconds": recheck_seconds, "fallback": dict(fallback),
            "expected_result": expected_result.strip(),
        }

    async def create_wait(
        self, db, run, wait_key=None, owner=None, awaited_event=None, due_in_seconds=None, fallback=None,
        *, origin=None, recheck_seconds=None, expected_result=None,
    ):
        """Create one locked, replay-safe wait.  Origin is its durable identity."""
        origin = origin if origin is not None else wait_key
        recheck_seconds = recheck_seconds if recheck_seconds is not None else due_in_seconds
        # Older callers used the event's top-level subject as a matcher.
        if isinstance(awaited_event, Mapping):
            event_type = awaited_event.get("event_type", awaited_event.get("type"))
            matcher = awaited_event.get("matcher")
            if matcher is None:
                matcher = {key: value for key, value in awaited_event.items() if key not in {"type", "event_type"}}
            awaited_event = {"event_type": event_type, "matcher": matcher}
        if expected_result is None and isinstance(fallback, Mapping):
            expected_result = fallback.get("expected_result")
        contract = self._wait_contract(origin, owner, awaited_event, recheck_seconds, fallback, expected_result)
        key = f"run:{run.id}:wait:{contract['origin']}"
        stored_fallback = {**contract["fallback"], "expected_result": contract["expected_result"], "recheck_seconds": contract["recheck_seconds"]}
        async with self.orchestration._lock_goal_for_baseline_transition(db, run.goal_id):
            current_goal = await db.get(OrchestrationGoal, run.goal_id, populate_existing=True)
            current_run = await db.get(OrchestrationRun, run.id, populate_existing=True)
            if current_goal is None or current_run is None or current_goal.status != "active" or current_run.status != "running" or current_run.phase != "authorized":
                raise HTTPException(status_code=409, detail="wait is not authorized")
            existing = (await db.scalars(select(OrchestrationWait).where(
                OrchestrationWait.run_id == current_run.id, OrchestrationWait.wait_key == key
            ).order_by(OrchestrationWait.created_at.desc()))).first()
            if existing is not None:
                if (existing.owner, existing.awaited_event, existing.fallback) != (contract["owner"], contract["awaited_event"], stored_fallback):
                    raise HTTPException(status_code=409, detail="wait replay conflicts with existing contract")
                return existing
            wait = OrchestrationWait(
                run_id=current_run.id, wait_key=key, owner=contract["owner"], awaited_event=contract["awaited_event"],
                due_recheck_at=_utcnow() + timedelta(seconds=contract["recheck_seconds"]), fallback=stored_fallback,
            )
            try:
                async with db.begin_nested():
                    db.add(wait)
                    await db.flush()
            except IntegrityError:
                existing = (await db.scalars(select(OrchestrationWait).where(
                    OrchestrationWait.run_id == current_run.id, OrchestrationWait.wait_key == key
                ).order_by(OrchestrationWait.created_at.desc()))).first()
                if existing is not None and (existing.owner, existing.awaited_event, existing.fallback) == (contract["owner"], contract["awaited_event"], stored_fallback):
                    return existing
                raise HTTPException(status_code=409, detail="wait replay conflicts with existing contract")
            return wait

    async def clear_matching_waits(self, db, run, *, event_type, subject_id=None, event_id=None, matcher=None):
        """Clear only waits whose event type and complete matcher match exactly."""
        if not isinstance(event_type, str) or not event_type:
            return 0
        values = dict(matcher or {})
        if subject_id is not None:
            values["subject_id"] = str(subject_id)
        waits = list((await db.scalars(select(OrchestrationWait).where(
            OrchestrationWait.run_id == run.id, OrchestrationWait.status == "open"
        ))).all())
        cleared = 0
        for wait in waits:
            expected = wait.awaited_event or {}
            if expected.get("event_type") != event_type:
                continue
            if any(values.get(key) != value for key, value in dict(expected.get("matcher") or {}).items()):
                continue
            wait.status, wait.cleared_by_event_id, wait.cleared_at = "cleared", event_id, _utcnow()
            cleared += 1
        if cleared:
            await db.flush()
        return cleared

    async def reconcile_local(self, db, goal, run, now=None, events=(), *, allow_release=True):
        """Deterministic local liveness pass; it never asks a provider or dispatches recovery."""
        now = now or _utcnow()
        state = dict(run.supervision_state or {})
        if goal.status != "active" or run.status != "running" or run.phase != "authorized":
            state["last_assessment"] = {"at": now.isoformat(), "outcome": "controlled"}
            run.supervision_state = state
            return {"outcome": "controlled"}
        if goal.goal_type == "roadmap" and self.orchestration._json_object_or_empty(run.plan_state).get("status") == "accepted":
            return {"outcome": "continue"}
        # Any subsequent liveness pass supersedes the prior idle snapshot; a
        # fresh idle state below recreates the same fingerprinted blocker.
        run.active_blockers = [blocker for blocker in run.active_blockers if not (
            isinstance(blocker, Mapping) and blocker.get("kind") == "everyone_idle"
        )]
        for event in events:
            payload = dict(getattr(event, "payload", {}) or {})
            await self.clear_matching_waits(
                db, run, event_type=event.event_type, subject_id=payload.get("subject_id"),
                event_id=event.id, matcher=payload,
            )
        all_tasks = await self.orchestration._orchestrated_tasks_for_run(
            db, run.id,
            statuses=["backlog", "ready", "in_progress", "blocked", "done", "failed", "cancelled"],
        )
        active_tasks = [task for task in all_tasks if task.status in {"backlog", "ready", "in_progress", "blocked"}]
        plan_accepted = self.orchestration._json_object_or_empty(run.plan_state).get("status") == "accepted"
        task_ids = {task.id for task in all_tasks}
        active_meetings = list((await db.scalars(select(Meeting).where(
            Meeting.source_task_id.in_(task_ids) if task_ids else False,
            Meeting.status.in_(("scheduled", "preparing", "active", "concluding")),
        ).order_by(Meeting.created_at, Meeting.id))).all())
        active_graph_runs = list((await db.scalars(select(GraphRun).where(
            GraphRun.linked_task_id.in_(task_ids) if task_ids else False,
            GraphRun.status == "active",
        ).order_by(GraphRun.created_at, GraphRun.id))).all())
        active_meeting = active_meetings[0] if active_meetings else None
        active_graph_run = active_graph_runs[0] if active_graph_runs else None
        provider_task_ids = {
            task_id for task_id in (
                *(meeting.source_task_id for meeting in active_meetings),
                *(graph_run.linked_task_id for graph_run in active_graph_runs),
            ) if task_id is not None
        }

        # Materialize every concrete occurrence before looking at time.  This
        # makes a recovery tick replay-safe even when several sources coexist.
        decisions = list((await db.scalars(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.run_id == run.id,
            OrchestrationAuthorityDecision.status == "pending",
        ).order_by(OrchestrationAuthorityDecision.asked_at, OrchestrationAuthorityDecision.id))).all())
        for decision in decisions:
            await self.create_wait(
                db, run, origin=f"decision:{decision.id}:resolved",
                owner={"type": "authority_decision", "id": str(decision.id), "decision_id": str(decision.id)},
                awaited_event={"event_type": "authority.decision_resolved", "matcher": {"decision_id": str(decision.id)}},
                recheck_seconds=settings.orchestration_reconcile_interval_seconds, fallback={"action_type": "attention", "reason": "Authority decision was not resolved."},
                expected_result="The named authority decision reaches a durable terminal state.",
            )
        sessions = list((await db.scalars(select(Session).where(
            Session.task_id.in_(task_ids) if task_ids else False,
            Session.status.in_(("pending", "running")),
        ).order_by(Session.created_at, Session.id))).all())
        sessions_by_task = {session.task_id: session for session in sessions}
        for session in sessions:
            if session.task_id in provider_task_ids:
                continue
            await self.create_wait(
                db, run, origin=f"session:{session.id}:task:{session.task_id}:status",
                owner={"type": "session", "id": str(session.id), "task_id": str(session.task_id)},
                awaited_event={"event_type": "task.status_changed", "matcher": {"task_id": str(session.task_id)}},
                recheck_seconds=settings.orchestration_reconcile_interval_seconds, fallback={"action_type": "attention", "reason": "Task result did not arrive."},
                expected_result="The owned task reaches a durable terminal or blocked state.",
            )
        for task in active_tasks:
            if (
                task.status == "in_progress"
                and task.assigned_to is not None
                and task.id not in sessions_by_task
                and task.id not in provider_task_ids
            ):
                await self.create_wait(
                    db, run, origin=f"task:{task.id}:session_created",
                    owner={"type": "task", "id": str(task.id), "task_id": str(task.id)},
                    awaited_event={"event_type": "session.created", "matcher": {"task_id": str(task.id)}},
                    recheck_seconds=settings.orchestration_reconcile_interval_seconds, fallback={"action_type": "attention", "reason": "Assigned task has no session."},
                    expected_result="The assigned task creates a session.",
                )
        child_runs = list((await db.scalars(select(OrchestrationRun).join(
            OrchestrationGoal, OrchestrationRun.goal_id == OrchestrationGoal.id
        ).where(
            OrchestrationGoal.parent_goal_id == goal.id,
            OrchestrationRun.status.in_(("running", "blocked", "paused")),
        ).order_by(OrchestrationRun.created_at, OrchestrationRun.id))).all())
        for child in child_runs:
            await self.create_wait(
                db, run, origin=f"child_run:{child.id}:completed",
                owner={"type": "child_run", "id": str(child.id), "run_id": str(child.id)},
                awaited_event={"event_type": "orchestration.run_completed", "matcher": {"run_id": str(child.id)}},
                recheck_seconds=settings.orchestration_reconcile_interval_seconds, fallback={"action_type": "attention", "reason": "Child run did not complete."},
                expected_result="The child orchestration run reaches a terminal state.",
            )
        waits = list((await db.scalars(select(OrchestrationWait).where(
            OrchestrationWait.run_id == run.id, OrchestrationWait.status == "open"
        ))).all())

        # A persisted wait is only authoritative while its concrete owner is
        # still live.  Do this before selecting a future wait left by an older
        # occurrence of the same task/session/child.
        pending_decision_ids = {str(decision.id) for decision in decisions}
        live_session_ids = {str(session.id) for session in sessions}
        live_child_ids = {str(child.id) for child in child_runs}
        active_task_ids = {str(task.id) for task in active_tasks}
        for wait in waits:
            owner = dict(wait.owner or {})
            owner_type, owner_id = owner.get("type"), str(owner.get("id"))
            stale = (
                owner_type == "authority_decision" and owner_id not in pending_decision_ids
                or owner_type == "session" and owner_id not in live_session_ids
                or owner_type == "task" and (
                    str(owner.get("task_id", owner_id)) not in active_task_ids
                    or wait.wait_key.endswith(":session_created") and not any(
                        str(task.id) == str(owner.get("task_id", owner_id))
                        and task.status == "in_progress"
                        and task.assigned_to is not None
                        and task.id not in sessions_by_task
                        and task.id not in provider_task_ids
                        for task in active_tasks
                    )
                )
                or owner_type == "child_run" and owner_id not in live_child_ids
            )
            if stale:
                wait.status, wait.cleared_at = "cleared", now
        waits = [wait for wait in waits if wait.status == "open"]
        def is_due(wait):
            reference = now.replace(tzinfo=None) if wait.due_recheck_at.tzinfo is None else now
            return wait.due_recheck_at <= reference
        task_id_strings = {str(task.id) for task in active_tasks}
        graph = {str(task.id): {str(dep) for dep in (task.depends_on or []) if str(dep) in task_id_strings} for task in active_tasks}
        seen, visiting = set(), set()
        def cyclic(task_id):
            if task_id in visiting:
                return True
            if task_id in seen:
                return False
            visiting.add(task_id)
            result = any(cyclic(dep) for dep in graph[task_id])
            visiting.remove(task_id)
            seen.add(task_id)
            return result
        if any(cyclic(task_id) for task_id in graph):
            self.orchestration._upsert_active_blocker(run, {"kind": "dependency_cycle", "scope": "supervision"})
            outcome = "dependency_cycle"
        elif any(
            task.assigned_to is None
            and task.id not in sessions_by_task
            and task.id not in provider_task_ids
            for task in active_tasks
        ):
            self.orchestration._upsert_active_blocker(run, {"kind": "orphaned_ownership", "scope": "supervision"})
            outcome = "orphaned_ownership"
        else:
            outcome = None
        if outcome is not None:
            action = await self.orchestration.execute_record_warning_action(
                db, run.id,
                {"action_type": "record_warning", "warning_type": "supervision_attention", "severity": "warning", "message": outcome},
                f"run:{run.id}:supervision:{outcome}",
            )
            state["last_assessment"] = {"at": now.isoformat(), "outcome": outcome}
            run.supervision_state = state
            return {"outcome": outcome, "action_id": str(action.id)}
        released = 0
        if allow_release and goal.goal_type == "outcome" and plan_accepted:
            try:
                released = await self.orchestration._release_ready_work(db, goal, run)
            except HTTPException as exc:
                if exc.status_code != 409 or exc.detail not in {"budget_wait", "needs_attention"}:
                    raise
                reason = str(exc.detail)
                # A bare budget shortage has no trustworthy owner/event.  Do
                # not invent a generic budget-transition wait; surface it.
                self.orchestration._upsert_active_blocker(run, {"kind": "supervision_needs_attention", "scope": "liveness", "reason": reason})
                action = await self.orchestration.execute_record_warning_action(
                    db, run.id, {"action_type": "record_warning", "warning_type": "supervision_attention", "severity": "warning", "message": reason},
                    f"run:{run.id}:supervision:needs_attention",
                )
                state["last_assessment"] = {"at": now.isoformat(), "outcome": reason}
                run.supervision_state = state
                return {"outcome": reason, "action_id": str(action.id)}
        if released:
            state["last_assessment"] = {"at": now.isoformat(), "outcome": "released", "count": released}
            run.supervision_state = state
            return {"outcome": "released", "count": released}
        # A gate by itself is not an attempted verification.  Let the executor
        # choose/dispatch one; only a failed concrete verifier is attention.
        verification_actions = list((await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id,
            OrchestrationAction.action_type == "request_verification",
        ).order_by(OrchestrationAction.created_at, OrchestrationAction.id))).all())
        failed_verification = next((action for action in verification_actions if action.status == "failed"), None)
        if failed_verification is not None:
            action = await self.orchestration.execute_record_warning_action(
                db, run.id,
                {"action_type": "record_warning", "warning_type": "supervision_attention", "severity": "warning",
                 "message": str(failed_verification.error or "Verification action failed.")},
                f"run:{run.id}:supervision:verification:{failed_verification.id}:failed",
            )
            state["last_assessment"] = {"at": now.isoformat(), "outcome": "needs_attention"}
            run.supervision_state = state
            return {"outcome": "needs_attention", "action_id": str(action.id)}
        if active_meeting is not None or active_graph_run is not None:
            source_type, source_id = ("meeting", active_meeting.id) if active_meeting is not None else ("graph", active_graph_run.id)
            state["last_assessment"] = {"at": now.isoformat(), "outcome": "durable_source_active", "source_type": source_type, "source_id": str(source_id)}
            run.supervision_state = state
            return {"outcome": "durable_source_active", "source_type": source_type, "source_id": str(source_id)}
        due = [wait for wait in waits if is_due(wait)]
        if due:
            wait = due[0]
            wait.status, wait.cleared_at = "cleared", now
            fallback = dict(wait.fallback or {})
            action = await (
                self.orchestration.execute_noop_action(
                    db, run.id,
                    {"action_type": "noop", "reason": str(fallback.get("reason") or "Wait recheck is due")},
                    f"run:{run.id}:wait:{wait.id}:fallback",
                )
                if fallback.get("action_type") == "continue"
                else self.orchestration.execute_record_warning_action(
                    db, run.id,
                    {"action_type": "record_warning", "warning_type": "supervision_attention", "severity": "warning",
                     "message": str(fallback.get("reason") or "Wait recheck is due")},
                    f"run:{run.id}:wait:{wait.id}:fallback",
                )
            )
            state["last_assessment"] = {"at": now.isoformat(), "outcome": "due_fallback", "wait_id": str(wait.id)}
            run.supervision_state = state
            return {"outcome": "due_fallback", "fallback": fallback, "action_id": str(action.id)}
        future = [wait for wait in waits if not is_due(wait)]
        if future:
            wait = future[0]
            state["last_assessment"] = {"at": now.isoformat(), "outcome": "waiting", "wait_id": str(wait.id)}
            run.supervision_state = state
            return {"outcome": "waiting", "wait_id": str(wait.id)}
        if not plan_accepted:
            state["last_assessment"] = {"at": now.isoformat(), "outcome": "continue"}
            run.supervision_state = state
            return {"outcome": "continue"}
        has_open_gate = await db.scalar(select(OrchestrationGate.id).where(
            OrchestrationGate.run_id == run.id, OrchestrationGate.status == "open"
        ).limit(1)) is not None
        if allow_release and not active_tasks and plan_accepted and not has_open_gate:
            fingerprint = self.orchestration._stable_hash({"plan": run.plan_state, "run_id": str(run.id)})
            run.active_blockers = [blocker for blocker in run.active_blockers if not (
                isinstance(blocker, Mapping) and blocker.get("kind") == "everyone_idle"
            )] + [{"kind": "everyone_idle", "scope": "supervision", "fingerprint": fingerprint}]
            action = await self.orchestration.execute_noop_action(
                db, run.id, {"action_type": "noop", "reason": "All authorized work is idle; closeout is ready."},
                f"run:{run.id}:supervision:everyone_idle:{fingerprint}",
            )
            state["last_assessment"] = {"at": now.isoformat(), "outcome": "closeout_ready", "fingerprint": fingerprint}
            run.supervision_state = state
            return {"outcome": "closeout_ready", "action_id": str(action.id), "fingerprint": fingerprint}
        state["last_assessment"] = {"at": now.isoformat(), "outcome": "continue"}
        run.supervision_state = state
        return {"outcome": "continue"}

    async def apply_disposition(self, db, goal, run, assessment: SupervisionAssessment):
        disposition = dict(assessment.disposition or {})
        kind = disposition.get("action_type")
        if kind not in DISPOSITIONS:
            raise HTTPException(status_code=409, detail="Unknown supervision disposition")
        request = disposition.get("request")
        if isinstance(request, dict) and "invalidate_assumptions" in request:
            raise HTTPException(status_code=409, detail="Replan assumption invalidation requires Roadmap acceptance")
        expected_result = disposition.get("expected_result")
        if not isinstance(expected_result, str) or not expected_result.strip():
            raise HTTPException(status_code=409, detail="Supervision disposition requires expected_result")
        async with self.orchestration._lock_goal_for_baseline_transition(db, goal.id):
            current_goal = await db.scalar(select(OrchestrationGoal).where(
                OrchestrationGoal.id == goal.id).execution_options(populate_existing=True))
            current_run = await db.scalar(select(OrchestrationRun).where(
                OrchestrationRun.id == run.id).execution_options(populate_existing=True))
            if current_goal is None or current_run is None or current_run.goal_id != current_goal.id:
                raise HTTPException(status_code=409, detail="Supervision lineage is stale")
            if (current_goal.status not in {"active", "blocked"} or current_run.status not in {"running", "blocked"}
                    or current_run.phase != "authorized"):
                raise HTTPException(status_code=409, detail="Supervision disposition is not authorized")
            contract = disposition.get("contract_version")
            if not isinstance(contract, str) or contract != await self._contract_version(db, current_goal, current_run):
                raise HTTPException(status_code=409, detail="Supervision disposition has a stale contract")
            return await self._apply_current(db, current_goal, current_run, disposition)

    async def _apply_current(self, db, goal, run, disposition):
        kind, origin = disposition["action_type"], str(disposition.get("origin") or disposition["action_type"])
        reason, request = str(disposition.get("reason") or "Supervision disposition"), dict(disposition.get("request") or {})
        executor, action_type, raw = self._mapping(kind, request, reason)
        canonical = self.orchestration.canonical_decision_request(action_type, raw, run_id=run.id)
        if kind == "replan":
            key = await self.orchestration.roadmap_replan_action_key(db, run)
        elif kind == "follow_up":
            source = await self.orchestration._follow_up_source_session_id(db, run.id, canonical)
            canonical["source_session_id"] = str(source) if source else None
            key = self.orchestration._follow_up_delegation_key(run.id, canonical)
        else:
            digest = self.orchestration._stable_hash({"origin": origin, "request": canonical, "contract": disposition["contract_version"]})
            key = f"run:{run.id}:kind:supervision:{kind}:origin:{digest}"
        action = await self.orchestration.reserve_action(db, run_id=run.id, idempotency_key=key, action_type=action_type, request=canonical)
        self._set_contract(action, goal, run, origin, key, disposition, reason)
        await db.flush()
        result = await getattr(self.orchestration, executor)(db, run.id, canonical, key)
        self._set_contract(result, goal, run, origin, key, disposition, reason)
        await db.flush()
        return result

    @staticmethod
    def _mapping(kind, request, reason):
        mapping = {
            "continue": ("execute_noop_action", "noop", {"action_type": "noop", "reason": reason}),
            "pause": ("execute_pause_run_action", "pause_run", {"action_type": "pause_run", "reason": reason}),
            "follow_up": ("execute_create_delegation_task_action", "create_delegation_task", {"action_type": "create_delegation_task", **request, "work_function": "follow_up"}),
            "verify": ("execute_request_verification_action", "request_verification", {"action_type": "request_verification", **request}),
            "reassign": ("execute_reassign_task_action", "reassign_task", {"action_type": "reassign_task", **request}),
            "meeting": ("execute_schedule_meeting_action", "schedule_meeting", {"action_type": "schedule_meeting", **request}),
            "graph": ("execute_start_graph_action", "start_graph", {"action_type": "start_graph", **request}),
            "replan": ("execute_request_roadmap_replan_action", "request_roadmap_replan", {"action_type": "request_roadmap_replan", **request, "reason": reason}),
            "attention": ("execute_record_warning_action", "record_warning", {"action_type": "record_warning", "warning_type": "supervision_attention", "severity": "warning", "message": reason}),
        }
        return mapping[kind]

    @staticmethod
    def _set_contract(action, goal, run, origin, key, disposition, _reason):
        action.dispatch_contract = {
            "owner": {"goal_id": str(goal.id), "run_id": str(run.id), "type": "orchestrator"},
            "origin": origin, "idempotency_key": key, "authority_basis": {"contract_version": disposition["contract_version"]},
            "budget_basis": {"action_id": str(action.id), "ledger": action.budget_ledger or {}, "snapshot": run.budget_state or {}},
            "expected_result": disposition["expected_result"], "disposition": disposition["action_type"], "contract_version": disposition["contract_version"],
        }
