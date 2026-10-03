import uuid
import hashlib
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

from croniter import croniter
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import SessionTransactionOrigin

from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration import (
    OrchestrationAction, OrchestrationBudgetReservation, OrchestrationContinuousCandidate,
    OrchestrationGoal, OrchestrationRun,
)
from huddleroom.models.session import Session
from huddleroom.models.agent import Agent
from huddleroom.schemas.orchestration import OrchestrationContinuousPolicy, parse_discovery_batch
from huddleroom.services.event_bus import emit_event_once
from huddleroom.services.orchestration_budget_service import BudgetMeasurementError, OrchestrationBudgetService, parent_caps
from huddleroom.services.project_service import ProjectService
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.session_service import SessionClaimAttention, SessionService
from huddleroom.services.task_service import TaskService

# The existing orchestration service deliberately owns these transition primitives.
# pylint: disable=protected-access


def _utc(value: datetime) -> datetime:
    value = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def next_slot(policy: dict, after: datetime) -> datetime:
    activation = policy["activation"]
    local_after = _utc(after).astimezone(ZoneInfo(activation["timezone"]))
    return _utc(croniter(activation["cron"], local_after).get_next(datetime))


class OrchestrationContinuousService:
    def __init__(self, orchestration):
        self.orchestration = orchestration

    _MAX_REJECTED_DISCOVERY_OUTPUT = 10_000
    _REJECTED_DISCOVERY_SNIPPET = 1_000
    _REJECTED_DISCOVERY_ERROR = 1_000

    @staticmethod
    def _state(goal: OrchestrationGoal) -> dict:
        return deepcopy(goal.continuous_state or {})

    @staticmethod
    def _caller_owns_transaction(db):
        transaction = db.sync_session.get_transaction()
        return transaction is not None and transaction.origin != SessionTransactionOrigin.AUTOBEGIN

    @staticmethod
    async def _finish_locked(db, caller_owns_transaction, result):
        if not caller_owns_transaction:
            await db.commit()
        return result

    async def update_policy(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
        policy: OrchestrationContinuousPolicy,
        *,
        actor: str,
    ) -> tuple[OrchestrationGoal, OrchestrationRun]:
        caller_owns_transaction = self._caller_owns_transaction(db)
        async with self.orchestration._lock_goal_for_baseline_transition(db, goal_id):
            goal = await db.scalar(
                select(OrchestrationGoal)
                .where(OrchestrationGoal.id == goal_id, OrchestrationGoal.project_id == project_id)
                .execution_options(populate_existing=True)
            )
            run = await self.orchestration.get_active_run_for_goal(db, project_id, goal_id)
            if goal is None or run is None:
                raise HTTPException(status_code=404, detail="Continuous goal not found")
            if goal.goal_type != "continuous" or goal.status not in {"active", "blocked", "paused"}:
                raise HTTPException(status_code=409, detail="Goal is not configurable Continuous work")
            if (goal.continuous_state or {}).get("stopped_at"):
                raise HTTPException(status_code=409, detail="Stopped Continuous policy is immutable")
            if set(policy.per_case_budget) != set(parent_caps(goal)):
                raise HTTPException(status_code=409, detail="Continuous policy must budget every parent dimension")
            if policy.discovery_source is not None:
                if set(policy.discovery_source.budget.per_cycle) != set(parent_caps(goal)):
                    raise HTTPException(status_code=409, detail="Continuous policy must budget every parent dimension")
                await self.validate_discovery_assignment(db, goal, policy.discovery_source.model_dump(mode="json"))
            for key, value in policy.child_template.constraints.items():
                if key in goal.constraints and goal.constraints[key] != value:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Continuous child constraint conflicts with parent: {key}",
                    )
            await ProjectService().lock_workspace_boundary(db, project_id)
            await ProjectService().require_runnable_project(db, project_id)
            version = int((goal.continuous_policy or {}).get("version", 0)) + 1
            goal.continuous_policy = {"version": version, **policy.model_dump(mode="json")}
            state = self._state(goal)
            state.update({
                "policy_version": version,
                "next_due_at": (
                    _iso(next_slot(goal.continuous_policy, _utcnow()))
                    if state.get("started_at") else state.get("next_due_at")
                ),
                "last_claimed_slot": state.get("last_claimed_slot"),
                "pending_slots": state.get("pending_slots", []),
                "pending_candidates": state.get("pending_candidates", []),
                "health": state.get("health", "healthy"),
                "health_reason": state.get("health_reason"),
                "cycles_completed": int(state.get("cycles_completed", 0)),
                "started_at": state.get("started_at"),
                "stopped_at": state.get("stopped_at"),
            })
            goal.continuous_state = state
            action_request = {"actor": actor, "policy_version": version}
            if policy.discovery_source is not None:
                action_request.update({
                    "agent_id": str(policy.discovery_source.agent_id),
                    "work_function": policy.discovery_source.work_function,
                    "access_requirements": policy.discovery_source.access_requirements,
                })
            action = await self.orchestration.reserve_action(
                db,
                run.id,
                f"run:{run.id}:kind:continuous_policy:{version}",
                "update_continuous_policy",
                action_request,
            )
            if action.status == "reserved":
                await self.orchestration._mark_action_completed(db, action, target_type="goal", target_id=goal.id)
            if policy.discovery_source is None:
                run.active_blockers = [item for item in run.active_blockers if not (
                    isinstance(item, dict) and item.get("kind") == "continuous_discovery_source_unrunnable"
                )]
            await db.flush()
            return await self._finish_locked(db, caller_owns_transaction, (goal, run))

    async def initialize_after_start(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        now: datetime,
    ) -> None:
        policy = deepcopy(goal.continuous_policy or {})
        if not policy:
            raise self.orchestration._start_conflict("goal_not_runnable", "Continuous policy is required")
        OrchestrationContinuousPolicy.model_validate({key: value for key, value in policy.items() if key != "version"})
        if policy.get("discovery_source") is not None:
            try:
                await self.validate_discovery_assignment(db, goal, policy["discovery_source"])
            except HTTPException as exc:
                raise self.orchestration._start_conflict("goal_not_runnable", str(exc.detail)) from exc
        state = self._state(goal)
        state.update({
            "policy_version": policy["version"], "next_due_at": _iso(next_slot(policy, now)),
            "last_claimed_slot": None, "pending_slots": [], "pending_candidates": [],
            "health": "healthy", "health_reason": None, "cycles_completed": 0,
            "started_at": _iso(now), "stopped_at": None,
        })
        goal.continuous_state = state
        run.phase = "waiting_activation"
        await db.flush()

    async def validate_discovery_assignment(self, db, goal, source) -> None:
        agent = await db.get(Agent, uuid.UUID(str(source["agent_id"])))
        hierarchy = await OrchestrationProcessService().get_current(db, goal.id, "team_hierarchy")
        outputs = hierarchy.outputs if hierarchy is not None and hierarchy.status == "completed" else {}
        role_to_agent = self.orchestration._json_object_or_empty(outputs).get("role_to_agent", {})
        if agent is None or not agent.is_active or role_to_agent.get(source["work_function"]) != str(agent.id):
            raise HTTPException(status_code=409, detail="continuous_discovery_source_unrunnable")

    async def dispatch_discovery_source(self, db, goal, run, now):
        plan_state = self.orchestration._json_object_or_empty(run.plan_state)
        policy = self.orchestration._json_object_or_empty(plan_state.get("continuous_policy"))
        source = policy["discovery_source"]
        discovery = self.orchestration._json_object_or_empty(plan_state.get("discovery"))
        task_service = TaskService()
        if discovery.get("active_session_id"):
            return await task_service.get(db, goal.project_id, uuid.UUID(discovery["active_task_id"]))
        await self.validate_discovery_assignment(db, goal, source)
        reservation = await OrchestrationBudgetService().reserve_discovery_source(db, goal, run, source, now)
        candidate_limit = discovery["candidate_limit"]
        request = {
            "action_type": "create_delegation_task", "agent_id": source["agent_id"],
            "work_function": source["work_function"], "scope": source["instructions"],
            "inputs": [f"Filter: {source['source_filter']}", f"Origin key rule: {source['origin_key_rule']}",
                       f"Access prerequisites: {source['access_requirements']}", f"Cycle key: {run.cycle_key}",
                       f"Candidate limit: {candidate_limit}"],
            "deliverable": "One unfenced JSON object matching the discovery batch schema.",
            "forbidden_work": ["Do not mutate the source.", "Do not return credentials, commands, prose, or fenced JSON."],
            "success_evidence": ["A schema_version=1 candidate batch within candidate_limit."],
            "budget": source["budget"]["per_cycle"],
            "report_schema": {"schema_version": 1, "candidates": [{"origin_key": "str", "objective": "str", "source_refs": ["str"]}]},
            "orchestrator_context": {"continuous": {"discovery_run_id": str(run.id), "cycle_key": run.cycle_key,
                "policy_version": policy["version"], "candidate_limit": candidate_limit}},
        }
        action = await self.orchestration.execute_create_delegation_task_action(
            db, run.id, request, f"run:{run.id}:kind:create_delegation_task:discovery:{run.cycle_key}",
        )
        task = await task_service.get(db, goal.project_id, action.target_id)
        discovery = {**discovery, "source_task_id": str(task.id), "active_task_id": str(task.id)}
        run.plan_state = {**plan_state, "discovery": discovery}
        await db.flush()
        run_kwargs = {"timeout": source["timeout_seconds"]}
        if "max_tokens" in source["budget"]["per_cycle"]:
            run_kwargs["max_tokens"] = int(Decimal(source["budget"]["per_cycle"]["max_tokens"]))
        try:
            _, session_id = await task_service.run(db, goal.project_id, task.id, **run_kwargs)
        except SessionClaimAttention as exc:
            await SessionService.persist_claim_attention(db, exc)
            await OrchestrationBudgetService().settle_discovery_source(db, reservation, run, reason="needs_attention")
            self.orchestration._upsert_active_blocker(run, {
                "kind": "continuous_discovery_source_unrunnable", "reason": str(exc.detail),
            })
            return None
        discovery = {**discovery, "active_session_id": str(session_id)}
        run.plan_state = {**run.plan_state, "discovery": discovery}
        await db.flush()
        return task

    @staticmethod
    def _stop_condition_reached(policy: dict, state: dict, now: datetime) -> bool:
        condition = policy["stop_condition"]
        if condition["mode"] == "max_cycles":
            return int(state.get("cycles_completed", 0)) >= condition["max_cycles"]
        return condition["mode"] == "deadline" and now >= _utc(datetime.fromisoformat(condition["deadline"]))

    async def _active_case_count(self, db, goal_id: uuid.UUID) -> int:
        return await db.scalar(select(func.count()).select_from(  # pylint: disable=not-callable
            OrchestrationGoal).where(
            OrchestrationGoal.parent_goal_id == goal_id,
            OrchestrationGoal.continuous_origin_key.is_not(None),
            OrchestrationGoal.status.in_(("active", "blocked", "paused")),
        ))

    async def derive_health(self, db, goal: OrchestrationGoal, now: datetime | None = None) -> dict:
        now = _utc(now or _utcnow())
        state = self._state(goal)
        children = list((await db.scalars(select(OrchestrationGoal).where(
            OrchestrationGoal.parent_goal_id == goal.id,
            OrchestrationGoal.continuous_origin_key.is_not(None),
        ))).all())
        parent_run = await self.orchestration.get_active_run_for_goal(db, goal.project_id, goal.id)
        active = [child for child in children if child.status in {"active", "blocked", "paused"}]
        active_child_runs = list((await db.scalars(
            select(OrchestrationRun).where(
                OrchestrationRun.goal_id.in_([child.id for child in active]),
                OrchestrationRun.status.in_(("running", "blocked", "paused")),
            ).order_by(OrchestrationRun.goal_id, OrchestrationRun.created_at, OrchestrationRun.id)
        )).all()) if active else []
        child_blockers = [
            blocker
            for child_run in active_child_runs
            for blocker in child_run.active_blockers
        ]
        cycle_runs = list((await db.scalars(select(OrchestrationRun).where(
            OrchestrationRun.goal_id == goal.id,
            OrchestrationRun.cycle_key.in_([child.continuous_origin_key for child in active]),
        ))).all()) if active else []
        cycle_policies = {
            cycle_run.cycle_key: self.orchestration._json_object_or_empty(cycle_run.plan_state).get("continuous_policy")
            for cycle_run in cycle_runs
        }
        candidates = list((await db.scalars(select(OrchestrationContinuousCandidate).where(
            OrchestrationContinuousCandidate.parent_goal_id == goal.id,
            OrchestrationContinuousCandidate.origin_key.in_([child.continuous_origin_key for child in active]),
        ))).all()) if active else []
        candidate_policies = {
            candidate.origin_key: self.orchestration._json_object_or_empty(candidate.snapshot).get("policy")
            for candidate in candidates
        }
        response_targets = {**{
            origin: policy["response_target_seconds"]
            for origin, policy in cycle_policies.items() if policy
        }, **{
            origin: policy["response_target_seconds"]
            for origin, policy in candidate_policies.items() if policy
        }}
        slo_missed = any(
            (
                _utc(child_run.completed_at or now)
                - _utc(child_run.started_at)
            ).total_seconds() > response_targets[child.continuous_origin_key]
            for child in active
            for child_run in active_child_runs
            if child_run.goal_id == child.id
            and child_run.started_at is not None
            and child.continuous_origin_key in response_targets
        )
        backlog = len(state.get("pending_candidates", []))
        try:
            budget = await OrchestrationBudgetService().continuous_available(db, goal, goal.continuous_policy, now)
        except BudgetMeasurementError as exc:
            budget = {}
            if parent_run is not None:
                self.orchestration._upsert_active_blocker(parent_run, {
                    "kind": "budget_measurement", "dimension": exc.dimension,
                    "session_id": str(exc.session_id), "scope": f"continuous:{goal.id}",
                })
        else:
            if parent_run is not None:
                parent_run.active_blockers = [blocker for blocker in parent_run.active_blockers if not (
                    isinstance(blocker, dict)
                    and blocker.get("kind") == "budget_measurement"
                    and blocker.get("scope") == f"continuous:{goal.id}"
                )]
        if state.get("stopped_at"):
            health, reason = "stopped", None
        elif parent_run is not None and parent_run.active_blockers:
            health, reason = "needs_attention", parent_run.active_blockers[0].get("kind", "runtime_blocker")
        elif child_blockers:
            health, reason = "needs_attention", child_blockers[0].get("kind", "child_runtime_blocker")
        elif any(child.status == "blocked" for child in children):
            health, reason = "needs_attention", "child_failure"
        elif slo_missed:
            health, reason = "degraded", "response_target_missed"
        elif backlog:
            health, reason = "degraded", (
                "budget_wait" if any(item.get("reason") == "budget_wait" for item in state.get("pending_candidates", [])
                                 if isinstance(item, dict)) else "backlog"
            )
        else:
            health, reason = "healthy", None
        state.update({
            "health": health,
            "health_reason": reason,
            "active_cases": len(active),
            "backlog": backlog,
            "budget": budget,
        })
        goal.continuous_state = state
        await db.flush()
        return {"health": health, "reason": reason, "active_cases": len(active), "backlog": backlog, "budget": budget}

    async def _terminalize_waiting_run(self, db, goal, run, now) -> None:
        state = self._state(goal)
        state.update({"stopped_at": _iso(now), "health": "stopped", "health_reason": None})
        goal.continuous_state = state
        if run.phase == "waiting_activation" and run.status in {"running", "blocked", "paused"}:
            run.status = run.phase = "completed"
            run.completed_at = now
        await db.flush()

    async def _complete_stopped_claimed_cycle(self, db, goal, run, now):
        state = self._state(goal)
        state.update({"stopped_at": _iso(now), "health": "stopped", "health_reason": None})
        goal.continuous_state = state
        return await self._complete_cycle(db, goal, run, "stopped", now)

    async def stop(self, db, project_id, goal_id, *, actor: str, reason: str, now: datetime | None = None):
        now = _utc(now or _utcnow())
        caller_owns_transaction = self._caller_owns_transaction(db)
        async with self.orchestration._lock_goal_for_baseline_transition(db, goal_id):
            goal = await db.scalar(select(OrchestrationGoal).where(
                OrchestrationGoal.id == goal_id,
                OrchestrationGoal.project_id == project_id,
            ).execution_options(populate_existing=True))
            if goal is None:
                raise HTTPException(status_code=404, detail="Continuous goal not found")
            run = await self.orchestration.get_active_run_for_goal(db, project_id, goal_id)
            if run is None:
                run = await self.orchestration.get_run_for_goal(db, project_id, goal_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Continuous goal not found")
            if goal.goal_type != "continuous" or not (goal.continuous_state or {}).get("started_at"):
                raise HTTPException(status_code=409, detail="Continuous work has not started")
            if goal.status in {"completed", "cancelled"}:
                raise HTTPException(status_code=409, detail=f"Cannot stop goal in status '{goal.status}'")
            state = self._state(goal)
            if state.get("stopped_at"):
                return await self._finish_locked(db, caller_owns_transaction, (goal, run))
            action = await self.orchestration.reserve_action(
                db, run.id, f"goal:{goal.id}:kind:stop_continuous", "stop_continuous",
                {"actor": actor, "reason": reason, "stopped_at": _iso(now),
                 "pending_candidates": deepcopy(state.get("pending_candidates", []))},
            )
            await self._cancel_discovery_work(db, goal, reason="cancelled")
            state["pending_candidates"] = []
            goal.continuous_state = state
            if run.phase == "authorized" and run.cycle_key:
                await self._complete_stopped_claimed_cycle(db, goal, run, now)
            else:
                await self._terminalize_waiting_run(db, goal, run, now)
            goal.status = "active"
            if action.status == "reserved":
                await self.orchestration._mark_action_completed(db, action, target_type="goal", target_id=goal.id)
            await db.flush()
            return await self._finish_locked(db, caller_owns_transaction, (goal, run))

    async def cancel_children(self, db, goal, *, cancelled_by: str) -> None:
        run = await self.orchestration.get_active_run_for_goal(db, goal.project_id, goal.id)
        if run is not None:
            state = self._state(goal)
            action = await self.orchestration.reserve_action(
                db, run.id, f"goal:{goal.id}:kind:cancel_continuous", "cancel_continuous",
                {"actor": cancelled_by, "pending_candidates": deepcopy(state.get("pending_candidates", []))},
            )
            if action.status == "reserved":
                await self.orchestration._mark_action_completed(db, action, target_type="goal", target_id=goal.id)
            await self._cancel_discovery_work(db, goal, reason="cancelled")
            state["pending_candidates"] = []
            goal.continuous_state = state
        children = list((await db.scalars(select(OrchestrationGoal).where(
            OrchestrationGoal.parent_goal_id == goal.id,
            OrchestrationGoal.continuous_origin_key.is_not(None),
            OrchestrationGoal.status.not_in(("completed", "cancelled")),
        ))).all())
        for child in children:
            child_run = await self.orchestration.get_run_for_goal(db, child.project_id, child.id)
            await self.orchestration._cancel_goal_no_commit(db, child, child_run, cancelled_by=cancelled_by)
        await self.settle_terminal_children(db, goal)

    async def _cancel_discovery_work(self, db, goal, *, reason: str) -> None:
        """Cancel owned source/repair work without touching durable candidate rows."""
        runs = list((await db.scalars(select(OrchestrationRun).where(
            OrchestrationRun.goal_id == goal.id,
            OrchestrationRun.status.in_(("running", "blocked", "paused")),
        ))).all())
        tasks = TaskService()
        sessions = SessionService()
        for source_run in runs:
            discovery = self.orchestration._json_object_or_empty(
                self.orchestration._json_object_or_empty(source_run.plan_state).get("discovery")
            )
            if not discovery:
                continue
            for task_value in {discovery.get("source_task_id"), discovery.get("repair_task_id")} - {None}:
                task_id = self.orchestration._event_uuid(task_value)
                task = await tasks.get(db, goal.project_id, task_id) if task_id else None
                if task is None:
                    continue
                active_sessions = list((await db.scalars(select(Session).where(
                    Session.task_id == task.id, Session.status.in_(("pending", "running")),
                ))).all())
                for session in active_sessions:
                    await sessions.cancel(db, session.id)
                if task.status not in {"completed", "cancelled", "failed", "done"}:
                    await tasks.cancel(db, goal.project_id, task.id)
            reservation = await db.scalar(select(OrchestrationBudgetReservation).where(
                OrchestrationBudgetReservation.discovery_run_id == source_run.id,
            ))
            if reservation is None:
                attention = SessionClaimAttention(source_run.id, {
                    "kind": "budget_integrity", "scope": f"discovery_claim:{source_run.id}",
                    "reason": "Discovery reservation is not active.",
                }, "Discovery reservation is not active.")
                await SessionService.persist_claim_attention(db, attention)
                self.orchestration._upsert_active_blocker(source_run, {
                    "kind": "budget_integrity", "reason": str(attention.detail),
                })
            elif reservation.status == "active":
                await OrchestrationBudgetService().settle_discovery_source(db, reservation, source_run, reason=reason)

    async def release_pending_candidates(self, db, goal, run, now):
        """Release durable FIFO backlog from the active successor before claiming work."""
        state = self._state(goal)
        pending = list(state.get("pending_candidates", []))
        if not pending:
            return None
        remaining = []
        wait_reason = None
        for index, entry in enumerate(pending):
            if not isinstance(entry, dict):
                remaining.extend(pending[index:])
                break
            candidate_id = self.orchestration._event_uuid(entry.get("candidate_id"))
            candidate = await db.get(OrchestrationContinuousCandidate, candidate_id) if candidate_id else None
            if candidate is not None:
                snapshot = self.orchestration._json_object_or_empty(candidate.snapshot)
                policy = self.orchestration._json_object_or_empty(snapshot.get("policy"))
                if not policy:
                    remaining.extend(pending[index:])
                    self.orchestration._upsert_active_blocker(run, {
                        "kind": "continuous_integrity", "reason": "Pending discovery candidate has no frozen policy",
                    })
                    wait_reason = "needs_attention"
                    break
                payload = candidate
                origin_key = candidate.origin_key
            else:
                payload = entry.get("candidate")
                origin_key = entry.get("origin_key") or entry.get("cycle_key")
                policy = self.orchestration._json_object_or_empty(entry.get("policy"))
                if not policy and isinstance(entry.get("cycle_key"), str):
                    source_run = await db.scalar(select(OrchestrationRun).where(
                        OrchestrationRun.goal_id == goal.id,
                        OrchestrationRun.cycle_key == entry["cycle_key"],
                    ))
                    policy = self.orchestration._json_object_or_empty(
                        self.orchestration._json_object_or_empty(source_run.plan_state).get("continuous_policy")
                    ) if source_run is not None else {}
                if not isinstance(payload, dict) or not isinstance(origin_key, str):
                    remaining.extend(pending[index:])
                    self.orchestration._upsert_active_blocker(run, {
                        "kind": "continuous_integrity", "reason": "Pending Continuous candidate is invalid",
                    })
                    wait_reason = "needs_attention"
                    break
                if not policy:
                    remaining.extend(pending[index:])
                    self.orchestration._upsert_active_blocker(run, {
                        "kind": "continuous_integrity", "reason": "Pending Continuous candidate has no frozen policy",
                    })
                    wait_reason = "needs_attention"
                    break
            if await self._active_case_count(db, goal.id) >= policy["max_active_cases"]:
                wait_reason = "capacity"
                remaining.extend([{**item, "reason": wait_reason} for item in pending[index:]])
                break
            try:
                if candidate is not None:
                    child = await self._release_discovery_child(db, goal, run, policy, payload, now)
                    candidate.child_goal_id = child.id
                else:
                    await self._release_direct_child(db, goal, run, policy, payload, now, origin_key=origin_key)
            except HTTPException as exc:
                if exc.status_code != 409 or exc.detail != "continuous_budget_wait":
                    raise
                wait_reason = "budget_wait"
                remaining.extend([{**item, "reason": wait_reason} for item in pending[index:]])
                break
        state["pending_candidates"] = remaining
        if not remaining:
            run.active_blockers = [item for item in run.active_blockers if not (
                isinstance(item, dict) and item.get("kind") == "continuous_backlog_overflow"
            )]
        state["health_reason"] = wait_reason or ("backlog" if remaining else None)
        goal.continuous_state = state
        await db.flush()
        return wait_reason

    async def claim_due_cycle(self, db, goal_id: uuid.UUID, now: datetime | None = None):  # pylint: disable=too-many-return-statements
        now = _utc(now or _utcnow())
        caller_owns_transaction = self._caller_owns_transaction(db)
        async with self.orchestration._lock_goal_for_baseline_transition(db, goal_id):
            goal = await db.scalar(select(OrchestrationGoal).where(
                OrchestrationGoal.id == goal_id,
            ).execution_options(populate_existing=True))
            if goal is None or goal.goal_type != "continuous":
                return await self._finish_locked(db, caller_owns_transaction, None)
            await self.settle_terminal_children(db, goal, now)
            run = await db.scalar(select(OrchestrationRun).where(
                OrchestrationRun.goal_id == goal_id,
                OrchestrationRun.status.in_(("running", "blocked", "paused")),
            ).order_by(OrchestrationRun.created_at.desc()).limit(1).execution_options(populate_existing=True))
            if run is None:
                return await self._finish_locked(db, caller_owns_transaction, None)
            state = self._state(goal)
            claimed_policy = self.orchestration._json_object_or_empty(run.plan_state).get("continuous_policy")
            if run.phase == "authorized" and not claimed_policy:
                self.orchestration._upsert_active_blocker(run, {
                    "kind": "continuous_integrity", "reason": "Claimed cycle has no policy snapshot",
                })
                await self.derive_health(db, goal, now)
                return await self._finish_locked(db, caller_owns_transaction, None)
            stop_policy = claimed_policy if run.phase == "authorized" else goal.continuous_policy
            if self._stop_condition_reached(stop_policy, state, now):
                if run.phase == "authorized" and run.cycle_key:
                    await self._complete_stopped_claimed_cycle(db, goal, run, now)
                else:
                    await self._terminalize_waiting_run(db, goal, run, now)
                return await self._finish_locked(db, caller_owns_transaction, None)
            if (
                goal.status != "active"
                or state.get("stopped_at")
                or run.phase != "waiting_activation"
                or run.status != "running"
            ):
                return await self._finish_locked(db, caller_owns_transaction, None)
            authorization = await db.scalar(select(OrchestrationAction.id).where(
                OrchestrationAction.run_id == run.id,
                OrchestrationAction.action_type == "authorize_execution",
                OrchestrationAction.status == "completed",
            ).limit(1))
            if authorization is None:
                self.orchestration._upsert_active_blocker(run, {
                    "kind": "continuous_authorization", "reason": "Waiting cycle has no authorization action",
                })
                await self.derive_health(db, goal, now)
                return await self._finish_locked(db, caller_owns_transaction, None)
            run.active_blockers = [item for item in run.active_blockers if not (
                isinstance(item, dict) and item.get("kind") == "continuous_authorization"
            )]
            due = _utc(datetime.fromisoformat(state["next_due_at"]))
            if due > now:
                return await self._finish_locked(db, caller_owns_transaction, None)
            discovery = goal.continuous_policy.get("discovery_source")
            discovery_state = None
            if await self.release_pending_candidates(db, goal, run, now):
                await self.derive_health(db, goal, now)
                return await self._finish_locked(db, caller_owns_transaction, None)
            if discovery is not None:
                try:
                    await self.validate_discovery_assignment(db, goal, discovery)
                except HTTPException as exc:
                    self.orchestration._upsert_active_blocker(run, {
                        "kind": "continuous_discovery_source_unrunnable", "reason": str(exc.detail),
                    })
                    await self.derive_health(db, goal, now)
                    return await self._finish_locked(db, caller_owns_transaction, None)
                run.active_blockers = [item for item in run.active_blockers if not (
                    isinstance(item, dict) and item.get("kind") == "continuous_discovery_source_unrunnable"
                )]
            await self.derive_health(db, goal, now)
            if self._state(goal).get("health") == "needs_attention":
                return await self._finish_locked(db, caller_owns_transaction, None)
            state = self._state(goal)
            if await self._active_case_count(db, goal.id) >= goal.continuous_policy["max_active_cases"]:
                return await self._finish_locked(db, caller_owns_transaction, None)
            if discovery is not None:
                backlog_free = goal.continuous_policy["max_backlog"] - len(state.get("pending_candidates", []))
                active_free = goal.continuous_policy["max_active_cases"] - await self._active_case_count(db, goal.id)
                candidate_limit = min(discovery["max_candidates"], max(0, active_free) + max(0, backlog_free))
                if candidate_limit < 1:
                    return await self._finish_locked(db, caller_owns_transaction, None)
                try:
                    await OrchestrationBudgetService().reserve_discovery_source(db, goal, run, discovery, now)
                except HTTPException as exc:
                    if exc.status_code == 409 and exc.detail == "continuous_discovery_budget_wait":
                        return await self._finish_locked(db, caller_owns_transaction, None)
                    raise
                discovery_state = {"source_task_id": None, "active_task_id": None, "active_session_id": None,
                                   "repair_task_id": None, "candidate_limit": candidate_limit,
                                   "recovery": None, "accepted_batch": None}
            cycle_key = _iso(due)
            action = await self.orchestration.reserve_action(
                db, run.id, f"run:{run.id}:kind:claim_continuous_cycle:{cycle_key}",
                "claim_continuous_cycle", {"cycle_key": cycle_key, "policy_version": goal.continuous_policy["version"]},
            )
            run.cycle_key = cycle_key
            run.plan_state = {**self.orchestration._json_object_or_empty(run.plan_state),
                              "continuous_policy": deepcopy(goal.continuous_policy),
                              **({"discovery": discovery_state} if discovery_state is not None else {})}
            run.phase = "authorized"
            state.update({"last_claimed_slot": cycle_key, "next_due_at": _iso(next_slot(goal.continuous_policy, now))})
            goal.continuous_state = state
            if action.status == "reserved":
                await self.orchestration._mark_action_completed(db, action, target_type="run", target_id=run.id)
            await emit_event_once(db, goal.project_id, "orchestration.continuous_cycle_claimed",
                {"goal_id": str(goal.id), "run_id": str(run.id), "cycle_key": cycle_key}, source="orchestrator",
                dedup_key=f"orchestration.continuous_cycle_claimed:{goal.id}:{cycle_key}")
            await db.flush()
            return await self._finish_locked(db, caller_owns_transaction, run)

    @staticmethod
    def _direct_candidate(policy: dict) -> dict | None:
        return deepcopy(policy["child_template"]) if policy["adapter_filter"]["enabled"] else None

    def _parent_snapshot(self, goal, policy, origin_key):
        context = self.orchestration._json_object_or_empty(goal.orchestrator_context)
        return {"snapshot_version": 1, "parent_goal_id": str(goal.id),
                "continuous_policy_version": policy["version"], "origin_key": origin_key,
                "objective": goal.objective, "constraints": deepcopy(goal.constraints),
                "budget_policy": deepcopy(goal.budget), "authority": {"authority_model": goal.authority_model,
                "manager_agent_id": str(goal.manager_agent_id) if goal.manager_agent_id else None,
                "manager_user_id": str(goal.manager_user_id) if goal.manager_user_id else None},
                "team": deepcopy(context.get("team", {})), "success_criteria": deepcopy(goal.success_criteria),
                "workspace_policy": deepcopy(context.get("workspace_policy", {}))}

    async def _release_direct_child(self, db, goal, run, policy, candidate, now, *, origin_key=None):
        origin_key = origin_key or run.cycle_key
        existing = await db.scalar(select(OrchestrationGoal).where(
            OrchestrationGoal.parent_goal_id == goal.id, OrchestrationGoal.continuous_origin_key == origin_key,
        ))
        if existing is not None:
            return existing
        child_id = uuid.uuid5(uuid.NAMESPACE_URL, f"rally:continuous-child:{goal.id}:{origin_key}")
        child_run_id = uuid.uuid5(uuid.NAMESPACE_URL, f"rally:continuous-child-run:{goal.id}:{origin_key}")
        snapshot = self._parent_snapshot(goal, policy, origin_key)
        child = OrchestrationGoal(id=child_id, project_id=goal.project_id, objective=candidate["objective"],
            original_request=candidate["objective"], success_criteria=deepcopy(candidate["success_criteria"]),
            constraints={**deepcopy(goal.constraints), **deepcopy(candidate["constraints"])},
            budget={"caps": deepcopy(policy["per_case_budget"])}, goal_type="outcome", parent_goal_id=goal.id,
            continuous_origin_key=origin_key, parent_contract_snapshot=snapshot,
            goal_delta={"objective": candidate["objective"], "constraints": deepcopy(candidate["constraints"]),
                        "success_criteria": deepcopy(candidate["success_criteria"])},
            authority_model=goal.authority_model,
            manager_agent_id=goal.manager_agent_id, manager_user_id=goal.manager_user_id,
            orchestrator_context={"continuous": {"policy_version": policy["version"], "origin_key": origin_key},
                                  "team": deepcopy(snapshot["team"]),
                                  "workspace_policy": deepcopy(snapshot["workspace_policy"])})
        child_run = OrchestrationRun(id=child_run_id, goal_id=child.id, phase="authorized",
            budget_state={"caps": deepcopy(policy["per_case_budget"])}, plan_state={"child_delta_baseline": {
                "status": "accepted", "snapshot_version": 1, "parent_goal_id": str(goal.id),
                "continuous_policy_version": policy["version"], "origin_key": origin_key}})
        authorization = OrchestrationAction(id=uuid.uuid5(uuid.NAMESPACE_URL, f"rally:authorize-child:{child_run.id}"),
            run_id=child_run.id, idempotency_key=f"run:{child_run.id}:kind:authorize_execution",
            action_type="authorize_execution", status="completed",
            request={"action_type": "authorize_execution", "actor": f"continuous_parent:{goal.id}"},
            target_type="run", target_id=child_run.id)
        async with db.begin_nested():
            release = await self.orchestration.reserve_action(
                db, run.id, f"run:{run.id}:kind:release_continuous_child:{origin_key}",
                "release_continuous_child", {"origin_key": origin_key},
            )
            db.add(child)
            await db.flush()
            db.add(child_run)
            await db.flush()
            db.add(authorization)
            await db.flush()
            await OrchestrationBudgetService().reserve_continuous_child(
                db, goal, child, origin_key, policy["per_case_budget"], policy, now)
            await self.orchestration._mark_action_completed(db, release, target_type="goal", target_id=child.id)
            await emit_event_once(db, goal.project_id, "orchestration.continuous_child_released",
                {"goal_id": str(goal.id), "run_id": str(run.id), "cycle_key": origin_key,
                 "child_goal_id": str(child.id), "child_run_id": str(child_run.id)},
                source="orchestrator", dedup_key=f"orchestration.continuous_child_released:{goal.id}:{origin_key}")
        return child

    def _candidate_row(self, goal, run, position, item, discovery, policy):
        origin_key = item["origin_key"]
        return OrchestrationContinuousCandidate(
            id=uuid.uuid5(uuid.NAMESPACE_URL, f"rally:continuous-candidate:{goal.id}:{origin_key}"),
            parent_goal_id=goal.id,
            source_run_id=run.id,
            origin_key=origin_key,
            position=position,
            snapshot={
                "snapshot_version": 1,
                "candidate": deepcopy(item),
                "source": {
                    "run_id": str(run.id),
                    "task_id": discovery["accepted_task_id"],
                    "session_id": discovery.get("accepted_session_id"),
                    "cycle_key": run.cycle_key,
                },
                "policy": deepcopy(policy),
                "parent_contract": self._parent_snapshot(goal, policy, origin_key),
            },
        )

    async def _release_discovery_child(self, db, goal, run, policy, candidate, now):
        existing = await db.scalar(select(OrchestrationGoal).where(
            OrchestrationGoal.parent_goal_id == goal.id,
            OrchestrationGoal.continuous_origin_key == candidate.origin_key,
        ))
        if existing is not None:
            return existing
        template = policy["child_template"]
        origin_key = candidate.origin_key
        candidate_key = str(candidate.id)
        child_id = uuid.uuid5(uuid.NAMESPACE_URL, f"rally:continuous-child:{goal.id}:{origin_key}")
        child_run_id = uuid.uuid5(uuid.NAMESPACE_URL, f"rally:continuous-child-run:{goal.id}:{origin_key}")
        snapshot = deepcopy(candidate.snapshot)
        contract = deepcopy(snapshot["parent_contract"])
        provenance = {"candidate": deepcopy(snapshot["candidate"]), "source": deepcopy(snapshot["source"]),
                      "policy_version": policy["version"]}
        child = OrchestrationGoal(
            id=child_id, project_id=goal.project_id, objective=template["objective"],
            original_request=template["objective"], success_criteria=deepcopy(template["success_criteria"]),
            constraints={**deepcopy(goal.constraints), **deepcopy(template["constraints"])},
            budget={"caps": deepcopy(policy["per_case_budget"])}, goal_type="outcome", parent_goal_id=goal.id,
            continuous_origin_key=origin_key, parent_contract_snapshot=contract,
            goal_delta={"objective": template["objective"], "constraints": deepcopy(template["constraints"]),
                        "success_criteria": deepcopy(template["success_criteria"])},
            authority_model=goal.authority_model, manager_agent_id=goal.manager_agent_id,
            manager_user_id=goal.manager_user_id,
            orchestrator_context={"continuous": {"policy_version": policy["version"], "origin_key": origin_key,
                                                   "discovery_provenance": provenance},
                                  "team": deepcopy(contract["team"]),
                                  "workspace_policy": deepcopy(contract["workspace_policy"])},
        )
        child_run = OrchestrationRun(
            id=child_run_id, goal_id=child.id, phase="authorized",
            budget_state={"caps": deepcopy(policy["per_case_budget"])}, plan_state={"child_delta_baseline": {
                "status": "accepted", "snapshot_version": 1, "parent_goal_id": str(goal.id),
                "continuous_policy_version": policy["version"], "origin_key": origin_key,
            }},
        )
        authorization = OrchestrationAction(
            id=uuid.uuid5(uuid.NAMESPACE_URL, f"rally:authorize-child:{child_run.id}"), run_id=child_run.id,
            idempotency_key=f"run:{child_run.id}:kind:authorize_execution", action_type="authorize_execution",
            status="completed", request={"action_type": "authorize_execution", "actor": f"continuous_parent:{goal.id}"},
            target_type="run", target_id=child_run.id,
        )
        async with db.begin_nested():
            release = await self.orchestration.reserve_action(
                db, run.id, f"run:{run.id}:kind:release_continuous_child:{candidate_key}",
                "release_continuous_child", {"origin_key": origin_key, "candidate_id": str(candidate.id)},
            )
            db.add(child)
            await db.flush()
            db.add(child_run)
            await db.flush()
            db.add(authorization)
            await db.flush()
            await OrchestrationBudgetService().reserve_continuous_child(
                db, goal, child, origin_key, policy["per_case_budget"], policy, now,
            )
            await self.orchestration._mark_action_completed(db, release, target_type="goal", target_id=child.id)
            await emit_event_once(db, goal.project_id, "orchestration.continuous_child_released",
                {"goal_id": str(goal.id), "run_id": str(run.id), "cycle_key": origin_key,
                 "child_goal_id": str(child.id), "child_run_id": str(child_run.id)}, source="orchestrator",
                dedup_key=f"orchestration.continuous_child_released:{goal.id}:{candidate_key}")
        return child

    async def _represented_discovery_origins(self, db, goal, state):
        candidate_origins = set((await db.scalars(select(OrchestrationContinuousCandidate.origin_key).where(
            OrchestrationContinuousCandidate.parent_goal_id == goal.id,
        ))).all())
        child_origins = set((await db.scalars(select(OrchestrationGoal.continuous_origin_key).where(
            OrchestrationGoal.parent_goal_id == goal.id,
            OrchestrationGoal.continuous_origin_key.is_not(None),
        ))).all())
        pending_origins = {
            item["origin_key"] for item in state.get("pending_candidates", [])
            if isinstance(item, dict) and isinstance(item.get("origin_key"), str)
        }
        return candidate_origins | child_origins | pending_origins

    async def fan_out_discovery_batch(self, db, goal, run, now):
        plan_state = self.orchestration._json_object_or_empty(run.plan_state)
        discovery = self.orchestration._json_object_or_empty(plan_state.get("discovery"))
        policy = self.orchestration._json_object_or_empty(plan_state.get("continuous_policy"))
        batch = discovery["accepted_batch"]["candidates"]
        async def placement():
            current_state = self._state(goal)
            represented = await self._represented_discovery_origins(db, goal, current_state)
            current_fresh = [item for item in batch if item["origin_key"] not in represented]
            active_free = max(0, policy["max_active_cases"] - await self._active_case_count(db, goal.id))
            backlog_free = max(0, policy["max_backlog"] - len(current_state.get("pending_candidates", [])))
            available = await OrchestrationBudgetService().continuous_available(db, goal, policy, now)
            slots = [max(Decimal("0"), Decimal(available["available"][key]) // Decimal(policy["per_case_budget"][key]))
                     for key in policy["per_case_budget"] if Decimal(policy["per_case_budget"][key]) > 0]
            return current_state, current_fresh, max(0, min(active_free, int(min(slots)) if slots else active_free)), backlog_free

        state, fresh, release_slots, backlog_free = await placement()
        if len(fresh) > release_slots + backlog_free:
            self.orchestration._upsert_active_blocker(run, {
                "kind": "continuous_backlog_overflow", "observed": len(fresh),
                "available": release_slots + backlog_free,
            })
            reservation = await db.scalar(select(OrchestrationBudgetReservation).where(
                OrchestrationBudgetReservation.discovery_run_id == run.id,
            ))
            await OrchestrationBudgetService().settle_discovery_source(db, reservation, run, reason="needs_attention")
            await self.derive_health(db, goal, now)
            return {"step": "needs_attention"}

        rows = [self._candidate_row(goal, run, position, item, discovery, policy)
                for position, item in enumerate(batch) if item in fresh]
        collision_stalled = False
        if rows:
            for _ in range(len(batch) + 1):
                try:
                    async with db.begin_nested():
                        db.add_all(rows)
                        await db.flush()
                    break
                except IntegrityError:
                    # Parent locking is primary; a collision replays from durable represented origins.
                    prior_origins = {item["origin_key"] for item in fresh}
                    state, fresh, release_slots, backlog_free = await placement()
                    if not {item["origin_key"] for item in fresh} < prior_origins:
                        collision_stalled = True
                        break
                    rows = [self._candidate_row(goal, run, position, item, discovery, policy)
                            for position, item in enumerate(batch) if item in fresh]
                    if not rows:
                        break
            else:
                collision_stalled = True
        if collision_stalled:
            self.orchestration._upsert_active_blocker(run, {
                "kind": "continuous_candidate_collision", "reason": "Candidate uniqueness collision made no progress",
            })
            reservation = await db.scalar(select(OrchestrationBudgetReservation).where(
                OrchestrationBudgetReservation.discovery_run_id == run.id,
            ))
            await OrchestrationBudgetService().settle_discovery_source(db, reservation, run, reason="needs_attention")
            await self.derive_health(db, goal, now)
            return {"step": "needs_attention"}

        if rows:
            run.active_blockers = [item for item in run.active_blockers if not (
                isinstance(item, dict) and item.get("kind") == "continuous_backlog_overflow"
            )]

        pending = list(state.get("pending_candidates", []))
        created = 0
        budget_wait = False
        for row in rows:
            if not budget_wait and created < release_slots:
                try:
                    child = await self._release_discovery_child(db, goal, run, policy, row, now)
                except HTTPException as exc:
                    if exc.status_code != 409 or exc.detail != "continuous_budget_wait":
                        raise
                    budget_wait = True
                else:
                    row.child_goal_id = child.id
                    created += 1
                    continue
            pending.append({"candidate_id": str(row.id), "reason": "budget_wait" if budget_wait else "capacity"})
        state["pending_candidates"] = pending
        goal.continuous_state = state
        reservation = await db.scalar(select(OrchestrationBudgetReservation).where(
            OrchestrationBudgetReservation.discovery_run_id == run.id,
        ))
        await OrchestrationBudgetService().settle_discovery_source(db, reservation, run, reason="completed")
        outcome = "no_action" if not fresh else "children_created" if created else "backlogged"
        return await self._complete_cycle(db, goal, run, outcome, now)

    async def _complete_cycle(self, db, goal, run, outcome, now):
        key = f"run:{run.id}:kind:complete_cycle:{run.cycle_key}"
        existing = await self.orchestration._existing_action_for_key(db, run.id, key)
        if existing is not None and run.status == "completed":
            return {"step": "complete_cycle", "outcome": existing.request["outcome"]}
        action = await self.orchestration.reserve_action(
            db, run.id, key, "complete_cycle", {"cycle_key": run.cycle_key, "outcome": outcome},
        )
        state = self._state(goal)
        state["cycles_completed"] = int(state.get("cycles_completed", 0)) + 1
        goal.continuous_state = state
        await self.orchestration._mark_action_completed(db, action, target_type="run", target_id=run.id)
        run.status = run.phase = "completed"
        run.completed_at = now
        await emit_event_once(db, goal.project_id, "orchestration.continuous_cycle_completed",
            {"goal_id": str(goal.id), "run_id": str(run.id), "cycle_key": run.cycle_key, "outcome": outcome},
            source="orchestrator", dedup_key=f"orchestration.continuous_cycle_completed:{goal.id}:{run.cycle_key}")
        if not state.get("stopped_at"):
            successor_id = uuid.uuid5(uuid.NAMESPACE_URL, f"rally:continuous-successor:{run.id}")
            successor = await db.get(OrchestrationRun, successor_id)
            if successor is None:
                successor = OrchestrationRun(id=successor_id, goal_id=goal.id, phase="waiting_activation")
                db.add(successor)
                await db.flush()
                db.add(OrchestrationAction(
                    id=uuid.uuid5(uuid.NAMESPACE_URL, f"rally:authorize-successor:{successor.id}"),
                    run_id=successor.id, idempotency_key=f"run:{successor.id}:kind:authorize_execution",
                    action_type="authorize_execution", status="completed",
                    request={"action_type": "authorize_execution", "actor": f"continuous_parent:{goal.id}"},
                    target_type="run", target_id=successor.id,
                ))
        await db.flush()
        await self.derive_health(db, goal, now)
        return {"step": "complete_cycle", "outcome": outcome}

    @classmethod
    def _rejected_discovery_output(cls, output: str) -> dict[str, str | int]:
        return {
            "snippet": output[:cls._REJECTED_DISCOVERY_SNIPPET],
            "digest": hashlib.sha256(output.encode()).hexdigest(),
            "length": len(output),
        }

    @classmethod
    def _rejected_discovery_error(cls, error: object) -> str:
        return str(error)[:cls._REJECTED_DISCOVERY_ERROR]

    async def _discovery_needs_attention(self, db, goal, run, reason, state, error, output, now):
        reservation = await db.scalar(select(OrchestrationBudgetReservation).where(
            OrchestrationBudgetReservation.discovery_run_id == run.id,
        ))
        try:
            await OrchestrationBudgetService().settle_discovery_source(
                db, reservation, run, reason="needs_attention",
            )
        except SessionClaimAttention as exc:
            await SessionService.persist_claim_attention(db, exc)
            self.orchestration._upsert_active_blocker(run, {
                "kind": "budget_integrity", "reason": str(exc.detail),
            })
            await self.derive_health(db, goal, now)
            return {"step": "needs_attention"}
        self.orchestration._upsert_active_blocker(run, {
            "kind": reason,
            "task_id": state.get("active_task_id"),
            "session_id": state.get("active_session_id"),
            "error": self._rejected_discovery_error(error),
            "rejected_output": self._rejected_discovery_output(output),
        })
        await self.derive_health(db, goal, now)
        return {"step": "needs_attention"}

    async def _dispatch_discovery_repair(self, db, goal, run, policy, state, task, session_id, error, output):
        safe_output = self._rejected_discovery_output(output)
        try:
            action = await self.orchestration.execute_create_delegation_task_action(
                db,
                run.id,
                {
                "action_type": "create_delegation_task",
                "agent_id": policy["discovery_source"]["agent_id"],
                "work_function": "follow_up",
                "parent_task_id": str(task.id),
                "source_session_id": str(session_id) if session_id else None,
                "scope": "Return a corrected discovery JSON batch for the validation errors.",
                "inputs": [f"Validation error: {self._rejected_discovery_error(error)}", f"Rejected output: {safe_output['snippet']}",
                           f"Rejected output digest: {safe_output['digest']}"],
                "deliverable": "One corrected unfenced schema_version=1 JSON object.",
                "forbidden_work": ["Do not change source scope, filter, identity rule, or candidate limit."],
                "success_evidence": ["The corrected object passes the frozen discovery schema."],
                "budget": policy["discovery_source"]["budget"]["per_cycle"],
                "report_schema": {"schema_version": 1, "candidates": []},
                "orchestrator_context": {"continuous": {"discovery_run_id": str(run.id), "cycle_key": run.cycle_key,
                    "candidate_limit": state["candidate_limit"]}},
                },
                f"run:{run.id}:kind:create_delegation_task:discovery_repair:{task.id}",
            )
        except SessionClaimAttention as exc:
            await SessionService.persist_claim_attention(db, exc)
            return None
        except HTTPException:
            return None
        if action.target_id is None:
            return None
        repair_task = await TaskService().get(db, goal.project_id, action.target_id)
        if repair_task is None:
            return None
        state.update({"recovery": "repair", "repair_task_id": str(repair_task.id),
                      "active_task_id": str(repair_task.id), "active_session_id": None})
        run.plan_state = {**self.orchestration._json_object_or_empty(run.plan_state), "discovery": state}
        await db.flush()
        run_kwargs = {"timeout": policy["discovery_source"]["timeout_seconds"]}
        if "max_tokens" in policy["discovery_source"]["budget"]["per_cycle"]:
            run_kwargs["max_tokens"] = int(Decimal(policy["discovery_source"]["budget"]["per_cycle"]["max_tokens"]))
        try:
            _, repair_session_id = await TaskService().run(db, goal.project_id, repair_task.id, **run_kwargs)
        except SessionClaimAttention as exc:
            await SessionService.persist_claim_attention(db, exc)
            return None
        except HTTPException:
            return None
        state["active_session_id"] = str(repair_session_id)
        run.plan_state = {**self.orchestration._json_object_or_empty(run.plan_state), "discovery": state}
        await db.flush()
        return repair_task

    async def _advance_discovery(self, db, goal, run, policy, now):
        plan_state = self.orchestration._json_object_or_empty(run.plan_state)
        state = self.orchestration._json_object_or_empty(plan_state.get("discovery"))
        if state.get("accepted_batch") is not None:
            return await self.fan_out_discovery_batch(db, goal, run, now)
        task_id = self.orchestration._event_uuid(state.get("active_task_id"))
        task = await TaskService().get(db, goal.project_id, task_id) if task_id else None
        if task is None:
            return await self._discovery_needs_attention(
                db, goal, run, "continuous_discovery_source_execution_failed", state, "Active source task is missing", "", now,
            )
        if task.status == "failed":
            if state.get("recovery") is not None:
                return await self._discovery_needs_attention(
                    db, goal, run, "continuous_discovery_source_execution_failed", state, "Source execution failed twice", "", now,
                )
            source = policy["discovery_source"]
            request = {"action_type": "retry_task", "task_id": str(task.id), "timeout": source["timeout_seconds"]}
            if "max_tokens" in source["budget"]["per_cycle"]:
                request["max_tokens"] = int(Decimal(source["budget"]["per_cycle"]["max_tokens"]))
            try:
                action = await self.orchestration.execute_retry_task_action(
                    db, run.id, request, f"run:{run.id}:kind:retry_task:discovery:{task.id}",
                )
            except SessionClaimAttention as exc:
                await SessionService.persist_claim_attention(db, exc)
                return await self._discovery_needs_attention(
                    db, goal, run, "continuous_discovery_source_execution_failed", state,
                    "Source retry could not start", "", now,
                )
            except HTTPException:
                action = None
            if action is None or action.target_id is None:
                return await self._discovery_needs_attention(
                    db, goal, run, "continuous_discovery_source_execution_failed", state, "Source retry could not start", "", now,
                )
            state.update({"recovery": "retry", "active_session_id": str(action.target_id)})
            run.plan_state = {**plan_state, "discovery": state}
            await db.flush()
            return {"step": "waiting_discovery_retry"}
        if task.status != "done":
            return {"step": "discovery_running"}
        session_id, output = await self.orchestration.consume_terminal_task_output(
            db, run, task, self.orchestration._event_uuid(state.get("active_session_id")),
        )
        output = output or ""
        try:
            if len(output) > self._MAX_REJECTED_DISCOVERY_OUTPUT:
                raise ValueError("discovery output exceeds safe size")
            batch = parse_discovery_batch(output, state["candidate_limit"])
        except (ValueError, ValidationError) as exc:
            if state.get("recovery") is not None:
                return await self._discovery_needs_attention(
                    db, goal, run, "continuous_discovery_source_output_invalid", state,
                    self._rejected_discovery_error(exc), output, now,
                )
            repaired = await self._dispatch_discovery_repair(
                db, goal, run, policy, state, task, session_id, self._rejected_discovery_error(exc), output,
            )
            if repaired is None:
                return await self._discovery_needs_attention(
                    db, goal, run, "continuous_discovery_source_output_invalid", state,
                    self._rejected_discovery_error(exc), output, now,
                )
            return {"step": "waiting_discovery_repair"}
        state.update({"accepted_batch": batch.model_dump(mode="json"), "accepted_task_id": str(task.id),
                      "accepted_session_id": str(session_id) if session_id else None})
        run.plan_state = {**plan_state, "discovery": state}
        await db.flush()
        return await self.fan_out_discovery_batch(db, goal, run, now)

    async def advance(self, db, goal, run, now: datetime | None = None):  # pylint: disable=too-many-return-statements
        now = _utc(now or _utcnow())
        if goal.status in {"paused", "cancelled"}:
            return {"step": "control_held"}
        if run.status == "completed" and run.cycle_key:
            existing = await self.orchestration._existing_action_for_key(
                db, run.id, f"run:{run.id}:kind:complete_cycle:{run.cycle_key}")
            if existing is not None:
                return {"step": "complete_cycle", "outcome": existing.request["outcome"]}
        if run.phase != "authorized" or not run.cycle_key:
            return {"step": "waiting_activation"}
        policy = deepcopy(self.orchestration._json_object_or_empty(run.plan_state).get("continuous_policy"))
        if not policy:
            self.orchestration._upsert_active_blocker(run, {
                "kind": "continuous_integrity", "reason": "Claimed cycle has no policy snapshot",
            })
            await self.derive_health(db, goal, now)
            return {"step": "needs_attention"}
        if self._stop_condition_reached(policy, self._state(goal), now):
            return await self._complete_stopped_claimed_cycle(db, goal, run, now)
        if self._state(goal).get("stopped_at"):
            return await self._complete_cycle(db, goal, run, "stopped", now)
        if policy.get("cycle_mode") == "discovery":
            if any(item.get("kind") == "continuous_discovery_source_unrunnable" for item in run.active_blockers):
                try:
                    await self.validate_discovery_assignment(db, goal, policy["discovery_source"])
                except HTTPException:
                    return {"step": "needs_attention"}
                run.active_blockers = [item for item in run.active_blockers if not (
                    isinstance(item, dict) and item.get("kind") == "continuous_discovery_source_unrunnable"
                )]
            if any(item.get("kind") == "budget_integrity" for item in run.active_blockers):
                return {"step": "needs_attention"}
            discovery = self.orchestration._json_object_or_empty(
                self.orchestration._json_object_or_empty(run.plan_state).get("discovery")
            )
            if discovery.get("active_task_id") or discovery.get("accepted_batch") is not None:
                return await self._advance_discovery(db, goal, run, policy, now)
            try:
                task = await self.dispatch_discovery_source(db, goal, run, now)
            except HTTPException as exc:
                if exc.status_code != 409 or exc.detail != "continuous_discovery_source_unrunnable":
                    raise
                reservation = await db.scalar(select(OrchestrationBudgetReservation).where(
                    OrchestrationBudgetReservation.discovery_run_id == run.id,
                ))
                await OrchestrationBudgetService().settle_discovery_source(
                    db, reservation, run, reason="needs_attention",
                )
                self.orchestration._upsert_active_blocker(run, {
                    "kind": "continuous_discovery_source_unrunnable", "reason": str(exc.detail),
                })
                task = None
            if task is None:
                await self.derive_health(db, goal, now)
                return {"step": "needs_attention"}
            return {"step": "discovery_running", "task_id": str(task.id)}
        candidate = self._direct_candidate(policy)
        if candidate is None:
            return await self._complete_cycle(db, goal, run, "no_action", now)
        try:
            await self._release_direct_child(db, goal, run, policy, candidate, now)
        except HTTPException as exc:
            if exc.status_code != 409 or exc.detail != "continuous_budget_wait":
                raise
            state = self._state(goal)
            pending = list(state.get("pending_candidates", []))
            if not any(item.get("origin_key") == run.cycle_key for item in pending):
                if len(pending) >= policy["max_backlog"]:
                    self.orchestration._upsert_active_blocker(run, {
                        "kind": "continuous_backlog_overflow", "limit": policy["max_backlog"],
                    })
                    await self.derive_health(db, goal, now)
                    return {"step": "needs_attention"}
                pending.append({"origin_key": run.cycle_key, "cycle_key": run.cycle_key,
                                "policy_version": policy["version"], "policy": deepcopy(policy),
                                "candidate": deepcopy(candidate)})
            state.update({"pending_candidates": pending, "health": "degraded", "health_reason": "budget_wait"})
            goal.continuous_state = state
            return await self._complete_cycle(db, goal, run, "budget_wait", now)
        return await self._complete_cycle(db, goal, run, "child_created", now)

    async def settle_terminal_children(self, db, goal, now: datetime | None = None) -> int:
        parent_run = await self.orchestration.get_active_run_for_goal(db, goal.project_id, goal.id)
        reservations = list((await db.scalars(select(OrchestrationBudgetReservation).where(
            OrchestrationBudgetReservation.parent_goal_id == goal.id,
            OrchestrationBudgetReservation.continuous_origin_key.is_not(None),
            OrchestrationBudgetReservation.status == "active",
        ))).all())
        settled = 0
        budget = OrchestrationBudgetService()
        for reservation in reservations:
            child = await db.get(OrchestrationGoal, reservation.child_goal_id)
            if child is None or child.status not in {"completed", "cancelled"}:
                continue
            child_run = await self.orchestration.get_run_for_goal(db, child.project_id, child.id)
            if child_run is None:
                continue
            try:
                await budget.settle_child(db, reservation, child_run,
                    reason="cancelled" if child.status == "cancelled" else "completed", conservative_cancel=False)
            except BudgetMeasurementError as exc:
                if parent_run is not None:
                    self.orchestration._upsert_active_blocker(parent_run, {"kind": "budget_measurement",
                        "origin_key": reservation.continuous_origin_key, "dimension": exc.dimension,
                        "session_id": str(exc.session_id)})
                continue
            except HTTPException as exc:
                if exc.status_code != 409 or not str(exc.detail).startswith("Child exceeded reserved"):
                    raise
                if parent_run is not None:
                    self.orchestration._upsert_active_blocker(parent_run, {"kind": "budget_integrity",
                        "origin_key": reservation.continuous_origin_key, "reason": str(exc.detail)})
                continue
            if parent_run is not None:
                origin_key = reservation.continuous_origin_key
                if not reservation.measurement_complete:
                    self.orchestration._upsert_active_blocker(parent_run, {
                        "kind": "budget_measurement", "origin_key": origin_key,
                        "reason": "Continuous child budget measurement is incomplete",
                    })
                else:
                    parent_run.active_blockers = [blocker for blocker in parent_run.active_blockers if not (
                        isinstance(blocker, dict)
                        and blocker.get("kind") in {"budget_measurement", "budget_integrity"}
                        and blocker.get("origin_key") == origin_key
                    )]
            settled += 1
        await self.derive_health(db, goal, now)
        await db.flush()
        return settled
