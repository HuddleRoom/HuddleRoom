# pylint: disable=too-many-public-methods,too-many-locals,protected-access

import uuid
from datetime import timedelta
from decimal import Decimal, InvalidOperation

from fastapi import HTTPException

from sqlalchemy import select, union

from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationBudgetReservation,
    OrchestrationGoal,
    OrchestrationRoadmapItem,
    OrchestrationRoadmapVersion,
    OrchestrationRun,
)
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import ROADMAP_BUDGET_DIMENSIONS


def canonical_amounts(raw: object, *, allowed: set[str] | frozenset[str]) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise HTTPException(status_code=422, detail="Budget amounts must be an object")
    result = {}
    for key, value in raw.items():
        if key not in ROADMAP_BUDGET_DIMENSIONS or key not in allowed:
            raise HTTPException(status_code=409, detail=f"Unsupported measured budget dimension '{key}'")
        if isinstance(value, bool):
            raise HTTPException(status_code=422, detail=f"Invalid amount for '{key}'")
        try:
            amount = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Invalid amount for '{key}'") from exc
        if not amount.is_finite() or amount < 0:
            raise HTTPException(status_code=422, detail=f"Invalid amount for '{key}'")
        result[key] = format(amount.normalize(), "f")
    return result


def parent_caps(goal: OrchestrationGoal) -> dict[str, str]:
    raw = goal.budget.get("caps", {}) if isinstance(goal.budget, dict) else {}
    return canonical_amounts(raw, allowed=ROADMAP_BUDGET_DIMENSIONS)


def protected_quantum(dimension: str, cap: Decimal) -> Decimal:
    """Return the smallest protected execution allowance for one dimension."""
    quantum = {
        "max_tokens": Decimal("1"),
        "max_turns": Decimal("1"),
        "max_hours": Decimal("1") / Decimal("3600"),
    }[dimension]
    return max(Decimal("0"), min(quantum, cap))


def validate_declared_allocations(goal, items) -> None:
    caps = parent_caps(goal)
    totals = {key: Decimal("0") for key in caps}
    for item in items:
        for key, amount in item.allocation.items():
            totals[key] += Decimal(str(amount))
    for key, total in totals.items():
        if total > Decimal(caps[key]):
            raise HTTPException(status_code=409, detail=f"Roadmap allocations exceed parent cap '{key}'")


class BudgetMeasurementError(Exception):
    def __init__(self, dimension: str, session_id):
        self.dimension = dimension
        self.session_id = session_id
        super().__init__(f"Missing {dimension} measurement for session {session_id}")


def empty_action_budget(amounts: dict[str, str], enforceable: bool) -> dict:
    """The one durable action budget shape; amount maps never overlap."""
    return {
        "allocation": amounts,
        "reserved": amounts,
        "committed": {},
        "consumed": {},
        "usage_state": "known",
        "enforceability": "enforceable" if enforceable else "non_enforceable",
    }


class OrchestrationBudgetService:
    TASK_LINEAGE_ACTIONS = (
        "create_delegation_task",
        "request_plan",
        "request_plan_revision",
        "request_roadmap_replan",
        "request_verification",
    )

    def snapshot(self, goal: OrchestrationGoal) -> dict:
        """Return the immutable, canonical budget shape used by supervision."""
        caps = parent_caps(goal)
        zero = {key: "0" for key in caps}
        return {
            "caps": caps,
            "dimensions": tuple(caps),
            "consumed": dict(zero),
            "reserved": dict(zero),
            "committed": dict(zero),
            "remaining": dict(caps),
        }

    @staticmethod
    def _snapshot_amounts(snapshot: dict, key: str) -> dict[str, Decimal]:
        return {
            dimension: Decimal(str((snapshot.get(key) or {}).get(dimension, "0")))
            for dimension in snapshot["caps"]
        }

    @staticmethod
    def _format_amounts(amounts: dict[str, Decimal]) -> dict[str, str]:
        return {key: format(value.normalize(), "f") for key, value in amounts.items()}

    def _refresh_snapshot(self, snapshot: dict) -> dict:
        caps = {key: Decimal(value) for key, value in snapshot["caps"].items()}
        consumed = self._snapshot_amounts(snapshot, "consumed")
        reserved = self._snapshot_amounts(snapshot, "reserved")
        committed = self._snapshot_amounts(snapshot, "committed")
        snapshot["remaining"] = self._format_amounts({
            key: caps[key] - consumed[key] - reserved[key] - committed[key]
            for key in caps
        })
        return snapshot

    def _action_amounts(self, snapshot: dict, raw: object) -> dict[str, Decimal]:
        canonical = canonical_amounts(raw, allowed=set(snapshot["caps"]))
        return {
            key: Decimal(canonical.get(key, "0"))
            for key in snapshot["caps"]
        }

    def _ledger(self, action) -> dict:
        ledger = getattr(action, "budget_ledger", None)
        if not isinstance(ledger, dict):
            raise HTTPException(status_code=409, detail="Action budget ledger is invalid")
        return ledger

    @staticmethod
    def _nonzero(amounts: dict[str, Decimal]) -> dict[str, str]:
        return {key: format(value.normalize(), "f") for key, value in amounts.items() if value}

    def _ledger_amounts(self, snapshot: dict, ledger: dict, key: str) -> dict[str, Decimal]:
        return self._action_amounts(snapshot, ledger.get(key, {}))

    def reserve(
        self, snapshot: dict, action, amounts: object, *, enforceable: bool = True, closeout: bool = False,
        protected_allowance: bool = False,
    ) -> dict:
        """Idempotently reserve action capacity from a canonical snapshot."""
        if snapshot.get("measurement_complete") is False and not closeout:
            raise HTTPException(status_code=409, detail="Budget measurement is incomplete")
        requested = self._action_amounts(snapshot, amounts)
        ledger = self._ledger(action)
        request = self._nonzero(requested)
        if ledger:
            if ledger.get("allocation", ledger.get("reserved")) != request:
                raise HTTPException(status_code=409, detail="Action budget replay conflicts with allocation")
            return self._refresh_snapshot(snapshot)
        remaining = self._snapshot_amounts(snapshot, "remaining")
        # Preserve one bounded verification/closeout attempt before discretionary work.
        # Its own reservation is still checked against the same cap below.
        protected = {
            key: protected_quantum(key, Decimal(value))
            for key, value in snapshot["caps"].items()
        }
        available = remaining if protected_allowance or closeout or snapshot.get("protected_allowance_reserved") else {
            key: max(Decimal("0"), remaining[key] - protected[key]) for key in remaining
        }
        if any(requested[key] > available[key] for key in requested):
            raise HTTPException(status_code=409, detail="Action budget exceeds remaining capacity")
        reserved = self._snapshot_amounts(snapshot, "reserved")
        snapshot["reserved"] = self._format_amounts({
            key: reserved[key] + requested[key] for key in requested
        })
        action.budget_ledger = empty_action_budget(request, enforceable)
        if closeout:
            action.budget_ledger["protected"] = "closeout"
            snapshot["protected_allowance_reserved"] = True
        return self._refresh_snapshot(snapshot)

    def commit(self, snapshot: dict, action, amounts: object | None = None) -> dict:
        """Move an existing reservation into committed capacity exactly once."""
        ledger = self._ledger(action)
        if not ledger or not {"reserved", "committed", "consumed"}.issubset(ledger):
            raise HTTPException(status_code=409, detail="Action budget is not reserved")
        reserved = self._ledger_amounts(snapshot, ledger, "reserved")
        committed = self._ledger_amounts(snapshot, ledger, "committed")
        allocation = self._ledger_amounts(snapshot, ledger, "allocation")
        if amounts is not None and self._action_amounts(snapshot, amounts) != allocation:
            raise HTTPException(status_code=409, detail="Action budget replay conflicts with allocation")
        # A completed settlement is terminal too: retries must be a no-op, not
        # recreate a committed hold after the reservation was released.
        if ledger.get("final_observation") or any(committed.values()) or any(self._ledger_amounts(snapshot, ledger, "consumed").values()):
            return self._refresh_snapshot(snapshot)
        reserved = self._snapshot_amounts(snapshot, "reserved")
        committed = self._snapshot_amounts(snapshot, "committed")
        snapshot["reserved"] = self._format_amounts({key: reserved[key] - allocation[key] for key in allocation})
        snapshot["committed"] = self._format_amounts({key: committed[key] + allocation[key] for key in allocation})
        action.budget_ledger = {**ledger, "reserved": {}, "committed": self._nonzero(allocation)}
        return self._refresh_snapshot(snapshot)

    def settle(
        self, snapshot: dict, action, amounts: object, *, measurement_complete: bool = True,
        observation_id: str | None = None,
    ) -> dict:
        """Settle a committed action, retaining actual overage in the snapshot."""
        ledger = self._ledger(action)
        if not ledger or not {"reserved", "committed", "consumed"}.issubset(ledger):
            raise HTTPException(status_code=409, detail="Action budget is not committed")
        actual = self._action_amounts(snapshot, amounts)
        measurement_key = f"{observation_id or 'anonymous'}:{'final' if measurement_complete else 'provisional'}"
        observed = list(ledger.get("observed_measurements", []))
        observations = dict(ledger.get("measurement_amounts", {}))
        if ledger.get("final_observation"):
            if (ledger["final_observation"] != measurement_key
                    or observations.get(measurement_key) != self._nonzero(actual)):
                raise HTTPException(status_code=409, detail="Action budget replay conflicts with settlement")
            return self._refresh_snapshot(snapshot)
        if measurement_key in observed:
            if observations.get(measurement_key) != self._nonzero(actual):
                raise HTTPException(status_code=409, detail="Action budget replay conflicts with settlement")
            return self._refresh_snapshot(snapshot)
        if measurement_complete and ledger.get("consumed"):
            raise HTTPException(status_code=409, detail="Action budget replay conflicts with settlement")
        observed.append(measurement_key)
        observations[measurement_key] = self._nonzero(actual)
        if not measurement_complete:
            action.budget_ledger = {
                **ledger, "usage_state": "unknown", "provisional_usage": self._nonzero(actual),
                "observed_measurements": observed, "measurement_amounts": observations,
            }
            return self._refresh_snapshot(snapshot)
        committed_allocation = self._ledger_amounts(snapshot, ledger, "committed")
        allocation = committed_allocation
        if not any(allocation.values()):
            allocation = self._ledger_amounts(snapshot, ledger, "reserved")
        committed = self._snapshot_amounts(snapshot, "committed")
        reserved = self._snapshot_amounts(snapshot, "reserved")
        consumed = self._snapshot_amounts(snapshot, "consumed")
        if any(committed_allocation.values()):
            snapshot["committed"] = self._format_amounts({
                key: committed[key] - allocation[key] for key in allocation
            })
        else:
            snapshot["reserved"] = self._format_amounts({
                key: reserved[key] - allocation[key] for key in allocation
            })
        snapshot["consumed"] = self._format_amounts({key: consumed[key] + actual[key] for key in actual})
        action.budget_ledger = {
            **ledger, "reserved": {}, "committed": {}, "consumed": self._nonzero(actual),
            "usage_state": "known", "final_observation": measurement_key,
            "observed_measurements": observed, "measurement_amounts": observations,
        }
        return self._refresh_snapshot(snapshot)

    def apply_overage(self, run: OrchestrationRun, snapshot: dict) -> bool:
        """Persist the one run-level stop marker used by all execution paths."""
        exceeded = any(Decimal(value) < 0 for value in snapshot["remaining"].values())
        state = dict(getattr(run, "budget_state", {}) or {})
        run.budget_state = {
            **state, "caps": dict(snapshot["caps"]),
            "status": "exceeded" if exceeded else "active",
        }
        return exceeded

    async def snapshot_for_run(self, db, goal: OrchestrationGoal, run: OrchestrationRun) -> dict:
        """Reconcile durable session, reservation and action facts into one run snapshot."""
        parent = await self.roadmap_parent(db, goal) or await self.continuous_parent(db, goal)
        state_caps = (run.budget_state or {}).get("caps") if isinstance(run.budget_state, dict) else None
        caps = canonical_amounts(state_caps, allowed=ROADMAP_BUDGET_DIMENSIONS) if state_caps else (
            await self.claim_caps(db, run, goal, parent) if parent is not None else parent_caps(goal)
        )
        snapshot = self.snapshot(type("BudgetGoal", (), {"budget": {"caps": caps}})())
        dimensions = set(caps)
        parent_run = parent is not None and goal.id == parent.id
        reservations = []
        if parent_run:
            reservations = list((await db.scalars(select(OrchestrationBudgetReservation).where(
                OrchestrationBudgetReservation.parent_goal_id == goal.id,
            ))).all())
        actions = list((await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id,
        ))).all())
        ledger_actions = [action for action in actions if isinstance(action.budget_ledger, dict)
                          and {"reserved", "committed", "consumed"}.issubset(action.budget_ledger)]
        snapshot["protected_allowance_reserved"] = any(
            action.budget_ledger.get("protected") == "closeout"
            and (action.budget_ledger.get("reserved") or action.budget_ledger.get("committed"))
            for action in ledger_actions
        )
        ledger_action_ids = {str(action.id) for action in ledger_actions}
        ledger_task_ids = {
            action.target_id for action in ledger_actions
            if action.target_type == "task" and action.target_id is not None
        }
        excluded_task_ids = set()
        if parent_run and any(reservation.discovery_run_id == run.id for reservation in reservations):
            excluded_task_ids.update(self._discovery_task_ids(run))
        consumed = {key: Decimal(value) for key, value in (
            await self.measured_run_spend(
                db, run, dimensions, excluded_task_ids=excluded_task_ids, excluded_action_ids=ledger_action_ids,
                ledger_task_ids=ledger_task_ids,
            )
        ).items()}
        committed = {key: Decimal(value) for key, value in (
            await self.active_run_commitments(
                db, run, dimensions, caps, excluded_task_ids=excluded_task_ids, excluded_action_ids=ledger_action_ids,
                ledger_task_ids=ledger_task_ids,
            )
        ).items()}
        reserved = {key: Decimal("0") for key in caps}
        if parent_run:
            for reservation in reservations:
                source = reservation.settled_spend if reservation.status == "settled" else reservation.allocation
                target = consumed if reservation.status == "settled" else reserved
                for key, value in self._action_amounts(snapshot, source).items():
                    target[key] += value
        usage_state = "known"
        for action in ledger_actions:
            ledger = action.budget_ledger
            if ledger.get("usage_state") == "unknown":
                usage_state = "unknown"
            for target, source in ((reserved, ledger.get("reserved", {})),
                                   (committed, ledger.get("committed", {})),
                                   (consumed, ledger.get("consumed", {}))):
                for key, value in self._action_amounts(snapshot, source).items():
                    target[key] += value
            if ledger.get("usage_state") == "unknown":
                provisional = self._ledger_amounts(snapshot, ledger, "provisional_usage")
                allocation = self._ledger_amounts(snapshot, ledger, "committed")
                if not any(allocation.values()):
                    allocation = self._ledger_amounts(snapshot, ledger, "reserved")
                for key in committed:
                    committed[key] += max(Decimal("0"), provisional[key] - allocation[key])
        snapshot["consumed"] = self._format_amounts(consumed)
        snapshot["reserved"] = self._format_amounts(reserved)
        snapshot["committed"] = self._format_amounts(committed)
        self._refresh_snapshot(snapshot)
        state = dict(run.budget_state or {})
        run.budget_state = {**state, **snapshot, "usage_state": usage_state}
        self.apply_overage(run, snapshot)
        await db.flush()
        return snapshot

    @staticmethod
    def _require_budget_action_control(goal: OrchestrationGoal, run: OrchestrationRun) -> None:
        if goal.status not in {"active", "blocked"} or run.status not in {"running", "blocked"}:
            raise HTTPException(status_code=409, detail="Orchestration run is not budget-runnable")
        if run.phase != "authorized":
            raise HTTPException(status_code=409, detail="Orchestration run is not authorized")

    async def supervision_summary(self, db, goal, run) -> dict:
        snapshot = await self.snapshot_for_run(db, goal, run)
        return {
            "cap": dict(snapshot["caps"]),
            "consumed": {key: value for key, value in snapshot["consumed"].items() if Decimal(value)},
            "reserved": {key: value for key, value in snapshot["reserved"].items() if Decimal(value)},
            "committed": {key: value for key, value in snapshot["committed"].items() if Decimal(value)},
            "remaining": dict(snapshot["remaining"]),
            "usage_state": (run.budget_state or {}).get("usage_state", "known"),
        }

    async def protected_action_allocation(self, db, goal, run) -> dict[str, str]:
        """Give one verification/closeout attempt a bounded, explicit allowance."""
        snapshot = await self.snapshot_for_run(db, goal, run)
        return {
            dimension: format(protected_quantum(dimension, Decimal(snapshot["remaining"][dimension])), "f")
            for dimension in snapshot["caps"]
        }

    async def can_dispatch(self, db, goal, run, amounts) -> bool:
        from huddleroom.services.orchestration_service import OrchestrationService

        async with OrchestrationService()._lock_goal_for_baseline_transition(db, goal.id):
            goal = await db.get(OrchestrationGoal, goal.id)
            run = await db.get(OrchestrationRun, run.id)
            return await self._can_dispatch(db, goal, run, amounts)

    async def _can_dispatch(self, db, goal, run, amounts) -> bool:
        try:
            summary = await self.supervision_summary(db, goal, run)
        except BudgetMeasurementError as exc:
            self._budget_blocker(run, "budget_measurement", dimension=exc.dimension,
                                 session_id=str(exc.session_id))
            await db.flush()
            return False
        if summary["usage_state"] != "known":
            if not any(
                isinstance(item, dict) and item.get("kind") == "budget_measurement" and item.get("scope")
                for item in (run.active_blockers or [])
            ):
                self._budget_blocker(run, "budget_measurement")
            await db.flush()
            return False
        self.apply_overage(run, {"caps": summary["cap"], "remaining": summary["remaining"]})
        if (run.budget_state or {}).get("status") == "exceeded":
            self._budget_blocker(run, "budget_exhausted")
            await db.flush()
            return False
        requested = canonical_amounts(amounts, allowed=set(summary["cap"]))
        return all(Decimal(value) <= Decimal(summary["remaining"].get(key, "0")) for key, value in requested.items())

    @staticmethod
    def _budget_blocker(run, kind: str, **details) -> None:
        scope = details.get("scope")
        current = [item for item in (run.active_blockers or []) if not (
            isinstance(item, dict) and item.get("kind") == kind and item.get("scope") == scope
        )]
        run.active_blockers = [*current, {"kind": kind, **details}]

    async def reserve_action_budget(self, db, goal, run, action, amounts, *, enforceable=True, closeout=False) -> dict:
        from huddleroom.services.orchestration_service import OrchestrationService

        async with OrchestrationService()._lock_goal_for_baseline_transition(db, goal.id):
            goal, run, action = await self._locked_budget_entities(db, goal, run, action)
            self._require_budget_action_control(goal, run)
            if not closeout and not await self._can_dispatch(db, goal, run, amounts):
                raise HTTPException(status_code=409, detail="Action budget cannot dispatch")
            snapshot = await self.snapshot_for_run(db, goal, run)
            result = self.reserve(
                snapshot, action, amounts, enforceable=enforceable, closeout=closeout,
                protected_allowance=closeout,
            )
            run.budget_state = {**dict(run.budget_state or {}), **result}
            await db.flush()
            return result

    async def commit_action_budget(self, db, goal, run, action, amounts=None) -> dict:
        from huddleroom.services.orchestration_service import OrchestrationService

        async with OrchestrationService()._lock_goal_for_baseline_transition(db, goal.id):
            goal, run, action = await self._locked_budget_entities(db, goal, run, action)
            self._require_budget_action_control(goal, run)
            snapshot = await self.snapshot_for_run(db, goal, run)
            result = self.commit(snapshot, action, amounts)
            run.budget_state = {**dict(run.budget_state or {}), **result}
            await db.flush()
            return result

    async def settle_action_budget(
        self, db, goal, run, action, amounts, *, measurement_complete=True, observation_id=None,
    ) -> dict:
        from huddleroom.services.orchestration_service import OrchestrationService

        async with OrchestrationService()._lock_goal_for_baseline_transition(db, goal.id):
            goal, run, action = await self._locked_budget_entities(db, goal, run, action)
            snapshot = await self.snapshot_for_run(db, goal, run)
            result = self.settle(snapshot, action, amounts, measurement_complete=measurement_complete,
                                 observation_id=observation_id)
            self.apply_overage(run, result)
            run.budget_state = {**dict(run.budget_state or {}), **result}
            await db.flush()
            return result

    @staticmethod
    async def _locked_budget_entities(db, goal, run, action):
        """Reload the canonical budget lineage after taking its goal lock."""
        goal = await db.get(OrchestrationGoal, goal.id, populate_existing=True)
        run = await db.get(OrchestrationRun, run.id, populate_existing=True)
        action = await db.get(OrchestrationAction, action.id, populate_existing=True)
        if goal is None or run is None or action is None or run.goal_id != goal.id or action.run_id != run.id:
            raise HTTPException(status_code=409, detail="Action budget lineage is invalid")
        return goal, run, action

    async def owning_roadmap_run(self, db, task: Task | None) -> OrchestrationRun | None:
        """Resolve a task's Roadmap run from durable action/item lineage only."""
        if task is None:
            return None
        actions = list((await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.target_type == "task",
            OrchestrationAction.target_id == task.id,
            OrchestrationAction.action_type.in_(self.TASK_LINEAGE_ACTIONS),
            OrchestrationAction.status == "completed",
        ).order_by(OrchestrationAction.created_at.asc(), OrchestrationAction.id.asc()))).all())
        runs = []
        for action in actions:
            run = await db.get(OrchestrationRun, action.run_id)
            if run is not None:
                goal = await db.get(OrchestrationGoal, run.goal_id)
                if goal is not None and await self.roadmap_parent(db, goal) is not None:
                    runs.append(run)
        run_ids = {run.id for run in runs}
        if len(run_ids) == 1:
            return runs[0]

        item = await db.scalar(select(OrchestrationRoadmapItem).where(
            OrchestrationRoadmapItem.task_id == task.id,
        ).order_by(OrchestrationRoadmapItem.item_key.asc(), OrchestrationRoadmapItem.id.asc()))
        if item is None:
            return None
        version = await db.get(OrchestrationRoadmapVersion, item.first_version_id)
        run = await db.get(OrchestrationRun, version.run_id) if version is not None else None
        if run is None:
            return None
        goal = await db.get(OrchestrationGoal, run.goal_id)
        if goal is None or await self.roadmap_parent(db, goal) is None:
            return None
        return run if not run_ids or run.id in run_ids else None

    async def _sessions_for_run(
        self, db, run: OrchestrationRun, statuses: tuple[str, ...], *, excluded_task_ids: set[uuid.UUID] | None = None,
        excluded_action_ids: set[str] | None = None, ledger_task_ids: set[uuid.UUID] | None = None,
    ) -> list[Session]:
        action_tasks = select(OrchestrationAction.target_id.label("task_id")).where(
            OrchestrationAction.run_id == run.id,
            OrchestrationAction.target_type == "task",
            OrchestrationAction.action_type.in_(self.TASK_LINEAGE_ACTIONS),
            OrchestrationAction.status == "completed",
        )
        item_tasks = select(OrchestrationRoadmapItem.task_id.label("task_id")).join(
            OrchestrationRoadmapVersion,
            OrchestrationRoadmapVersion.id == OrchestrationRoadmapItem.first_version_id,
        ).where(
            OrchestrationRoadmapVersion.run_id == run.id,
            OrchestrationRoadmapItem.task_id.is_not(None),
        )
        task_ids = union(action_tasks, item_tasks).subquery()
        query = select(Session).where(
            Session.status.in_(statuses),
            Session.task_id.in_(select(task_ids.c.task_id)),
        )
        if excluded_task_ids:
            query = query.where(Session.task_id.not_in(excluded_task_ids))
        sessions = list((await db.scalars(query)).all())
        if not excluded_action_ids:
            return sessions
        filtered = []
        for session in sessions:
            action_id = str(((session.metadata_ or {}).get("orchestration") or {}).get("action_id"))
            if ledger_task_ids and session.task_id in ledger_task_ids and action_id not in (excluded_action_ids or set()):
                raise BudgetMeasurementError("max_tokens", session.id)
            if excluded_action_ids and action_id in excluded_action_ids:
                continue
            filtered.append(session)
        return filtered

    async def _sessions_for_task_ids(self, db, task_ids: set[uuid.UUID], statuses: tuple[str, ...]) -> list[Session]:
        if not task_ids:
            return []
        return list((await db.scalars(
            select(Session).where(Session.task_id.in_(task_ids), Session.status.in_(statuses))
        )).all())

    @staticmethod
    def _discovery_task_ids(run: OrchestrationRun) -> set[uuid.UUID]:
        from huddleroom.services.orchestration_service import OrchestrationService

        discovery = OrchestrationService._json_object_or_empty(run.plan_state).get("discovery", {})
        state = OrchestrationService._json_object_or_empty(discovery)
        return {
            uuid.UUID(value)
            for value in (state.get("source_task_id"), state.get("repair_task_id"))
            if isinstance(value, str) and value
        }

    async def claim_caps(self, db, run: OrchestrationRun, goal: OrchestrationGoal, parent: OrchestrationGoal) -> dict[str, str]:
        if goal.id == parent.id:
            return parent_caps(parent)
        reservation = await db.scalar(select(OrchestrationBudgetReservation).where(
            OrchestrationBudgetReservation.parent_goal_id == parent.id,
            OrchestrationBudgetReservation.child_goal_id == goal.id,
        ))
        if reservation is None:
            return {}
        return canonical_amounts(reservation.allocation, allowed=set(parent_caps(parent)))

    async def roadmap_parent(self, db, goal: OrchestrationGoal) -> OrchestrationGoal | None:
        if goal.goal_type == "roadmap":
            return goal
        if goal.parent_goal_id is None:
            return None
        parent = await db.get(OrchestrationGoal, goal.parent_goal_id)
        if parent is None or parent.goal_type != "roadmap":
            return None
        reservation = await db.scalar(select(OrchestrationBudgetReservation.id).where(
            OrchestrationBudgetReservation.parent_goal_id == parent.id,
            OrchestrationBudgetReservation.child_goal_id == goal.id,
        ))
        return parent if reservation is not None else None

    async def continuous_parent(self, db, goal: OrchestrationGoal) -> OrchestrationGoal | None:
        if goal.goal_type == "continuous":
            return goal
        if goal.parent_goal_id is None:
            return None
        parent = await db.get(OrchestrationGoal, goal.parent_goal_id)
        if parent is None or parent.goal_type != "continuous":
            return None
        reservation = await db.scalar(select(OrchestrationBudgetReservation.id).where(
            OrchestrationBudgetReservation.parent_goal_id == parent.id,
            OrchestrationBudgetReservation.child_goal_id == goal.id,
            OrchestrationBudgetReservation.continuous_origin_key.is_not(None),
        ))
        return parent if reservation is not None else None

    async def owning_continuous_run(self, db, task: Task | None) -> OrchestrationRun | None:
        if task is None:
            return None
        actions = list((await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.target_type == "task",
            OrchestrationAction.target_id == task.id,
            OrchestrationAction.action_type.in_(self.TASK_LINEAGE_ACTIONS),
            OrchestrationAction.status == "completed",
        ).order_by(OrchestrationAction.created_at.asc(), OrchestrationAction.id.asc()))).all())
        runs = []
        for action in actions:
            run = await db.get(OrchestrationRun, action.run_id)
            goal = await db.get(OrchestrationGoal, run.goal_id) if run is not None else None
            if goal is not None and await self.continuous_parent(db, goal) is not None:
                runs.append(run)
        return runs[0] if len({run.id for run in runs}) == 1 else None

    @staticmethod
    def _sum(values, dimensions: set[str]) -> dict[str, str]:
        totals = {key: Decimal("0") for key in dimensions}
        for value in values:
            for key in dimensions:
                totals[key] += Decimal(str((value or {}).get(key, "0")))
        return {key: format(value.normalize(), "f") for key, value in totals.items()}

    @staticmethod
    def _session_spend(session: Session, dimension: str, *, allow_incomplete: bool = False) -> Decimal:
        metadata = session.metadata_ or {}
        if dimension == "max_turns":
            value = metadata.get("_roadmap_turn_count", 1)
        elif dimension == "max_hours":
            value = metadata.get("_roadmap_elapsed_seconds")
            if value is None:
                if session.started_at is None or session.ended_at is None:
                    raise BudgetMeasurementError(dimension, session.id)
                value = Decimal(str((session.ended_at - session.started_at).total_seconds())) / Decimal("3600")
            else:
                value = OrchestrationBudgetService._commitment_amount(value, dimension, session.id) / Decimal("3600")
        else:
            grants = Decimal("0")
            if isinstance(metadata.get("_roadmap_cli_budget_approval"), dict):
                for grant in metadata.get("_roadmap_cli_token_grants", []):
                    grants += OrchestrationBudgetService._commitment_amount(
                        grant, dimension, session.id, whole=True
                    )
            known = None
            if "token_count_in" in metadata and "token_count_out" in metadata:
                # Parse observed counters first.  The audit marker records whether
                # telemetry is complete; it must not discard valid observed spend.
                try:
                    token_in = Decimal(str(metadata["token_count_in"]))
                    token_out = Decimal(str(metadata["token_count_out"]))
                except (InvalidOperation, ValueError) as exc:
                    if not grants:
                        raise BudgetMeasurementError(dimension, session.id) from exc
                else:
                    if (isinstance(metadata["token_count_in"], bool) or isinstance(metadata["token_count_out"], bool)
                            or not token_in.is_finite() or not token_out.is_finite() or token_in < 0 or token_out < 0):
                        if not grants:
                            raise BudgetMeasurementError(dimension, session.id)
                    else:
                        known = token_in + token_out
            if known is None and not grants:
                raise BudgetMeasurementError(dimension, session.id)
            if metadata.get("token_usage_complete") is False and not allow_incomplete and not grants:
                raise BudgetMeasurementError(dimension, session.id)
            value = max(known or Decimal("0"), grants)
        try:
            value = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise BudgetMeasurementError(dimension, session.id) from exc
        if (isinstance(value, bool) or not value.is_finite() or value < 0
                or (dimension == "max_turns" and value != value.to_integral_value())):
            raise BudgetMeasurementError(dimension, session.id)
        return value

    @staticmethod
    def _commitment_amount(value: object, dimension: str, session_id, *, whole: bool = False) -> Decimal:
        if isinstance(value, bool):
            raise BudgetMeasurementError(dimension, session_id)
        try:
            amount = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise BudgetMeasurementError(dimension, session_id) from exc
        if not amount.is_finite() or amount < 0 or (whole and amount != amount.to_integral_value()):
            raise BudgetMeasurementError(dimension, session_id)
        return amount

    async def _measured_sessions_spend(self, sessions: list[Session], dimensions: set[str]) -> dict[str, str]:
        totals = {key: Decimal("0") for key in dimensions}
        for session in sessions:
            for dimension in dimensions:
                totals[dimension] += self._session_spend(session, dimension)
        return {key: format(value.normalize(), "f") for key, value in totals.items()}

    async def measured_task_spend(
        self, db, task_ids: set[uuid.UUID], dimensions: set[str],
    ) -> dict[str, str]:
        return await self._measured_sessions_spend(
            await self._sessions_for_task_ids(db, task_ids, ("completed", "failed", "cancelled")), dimensions,
        )

    async def known_task_spend(
        self, db, task_ids: set[uuid.UUID], dimensions: set[str],
    ) -> tuple[dict[str, str], bool]:
        totals = {key: Decimal("0") for key in dimensions}
        complete = True
        sessions = await self._sessions_for_task_ids(db, task_ids, ("completed", "failed", "cancelled"))
        for session in sessions:
            for dimension in dimensions:
                try:
                    totals[dimension] += self._session_spend(session, dimension, allow_incomplete=True)
                except BudgetMeasurementError:
                    complete = False
                if dimension == "max_tokens" and (session.metadata_ or {}).get("token_usage_complete") is False:
                    complete = False
        return {key: format(value.normalize(), "f") for key, value in totals.items()}, complete

    async def measured_run_spend(
        self, db, run: OrchestrationRun | None, dimensions: set[str], *, excluded_task_ids: set[uuid.UUID] | None = None,
        excluded_action_ids: set[str] | None = None, ledger_task_ids: set[uuid.UUID] | None = None,
    ) -> dict[str, str]:
        if run is None:
            return {key: "0" for key in dimensions}
        return await self._measured_sessions_spend(
            await self._sessions_for_run(
                db, run, ("completed", "failed", "cancelled"), excluded_task_ids=excluded_task_ids,
                excluded_action_ids=excluded_action_ids, ledger_task_ids=ledger_task_ids,
            ), dimensions,
        )

    async def known_run_spend(self, db, run: OrchestrationRun | None, dimensions: set[str]) -> tuple[dict[str, str], bool]:
        """Return valid terminal spend without letting one bad row hide another."""
        if run is None:
            return {key: "0" for key in dimensions}, True
        sessions = await self._sessions_for_run(db, run, ("completed", "failed", "cancelled"))
        totals = {key: Decimal("0") for key in dimensions}
        complete = True
        for session in sessions:
            for dimension in dimensions:
                try:
                    totals[dimension] += self._session_spend(session, dimension, allow_incomplete=True)
                except BudgetMeasurementError:
                    complete = False
                if dimension == "max_tokens" and (session.metadata_ or {}).get("token_usage_complete") is False:
                    complete = False
        return {key: format(value.normalize(), "f") for key, value in totals.items()}, complete

    async def _active_sessions_commitments(self, sessions: list[Session], dimensions: set[str], caps: dict[str, str]):
        totals = {key: Decimal("0") for key in dimensions}
        for session in sessions:
            config = (session.metadata_ or {}).get("_run_config", {})
            if not isinstance(config, dict):
                raise BudgetMeasurementError("max_tokens", session.id)
            prior = config.get("_roadmap_prior_usage", {})
            if not isinstance(prior, dict):
                raise BudgetMeasurementError("max_tokens", session.id)
            if "max_turns" in dimensions:
                totals["max_turns"] += self._commitment_amount(
                    prior.get("max_turns", 0), "max_turns", session.id, whole=True
                ) + 1
            if "max_tokens" in dimensions:
                if "max_tokens" not in config:
                    raise BudgetMeasurementError("max_tokens", session.id)
                grants = (session.metadata_ or {}).get("_roadmap_cli_token_grants", [])
                if isinstance((session.metadata_ or {}).get("_roadmap_cli_budget_approval"), dict):
                    if not isinstance(grants, list):
                        raise BudgetMeasurementError("max_tokens", session.id)
                    totals["max_tokens"] += sum(
                        (self._commitment_amount(grant, "max_tokens", session.id, whole=True) for grant in grants),
                        Decimal("0"),
                    )
                else:
                    totals["max_tokens"] += self._commitment_amount(
                        prior.get("max_tokens", 0), "max_tokens", session.id, whole=True
                    ) + self._commitment_amount(config["max_tokens"], "max_tokens", session.id, whole=True)
            if "max_hours" in dimensions:
                if "timeout" not in config:
                    raise BudgetMeasurementError("max_hours", session.id)
                totals["max_hours"] += self._commitment_amount(
                    prior.get("max_hours", 0), "max_hours", session.id
                ) + self._commitment_amount(config["timeout"], "max_hours", session.id, whole=True) / Decimal("3600")
        return {key: format(value.normalize(), "f") for key, value in totals.items()}

    async def active_task_commitments(self, db, task_ids: set[uuid.UUID], dimensions: set[str], caps: dict[str, str]):
        return await self._active_sessions_commitments(
            await self._sessions_for_task_ids(db, task_ids, ("pending", "running")), dimensions, caps,
        )

    async def active_run_commitments(
        self, db, run: OrchestrationRun | None, dimensions: set[str], caps: dict[str, str], *,
        excluded_task_ids: set[uuid.UUID] | None = None, excluded_action_ids: set[str] | None = None,
        ledger_task_ids: set[uuid.UUID] | None = None,
    ):
        if run is None:
            return {key: "0" for key in dimensions}
        return await self._active_sessions_commitments(
            await self._sessions_for_run(
                db, run, ("pending", "running"), excluded_task_ids=excluded_task_ids,
                excluded_action_ids=excluded_action_ids, ledger_task_ids=ledger_task_ids,
            ), dimensions, caps,
        )

    async def run_remaining(self, db, run: OrchestrationRun, caps: dict[str, str]) -> dict[str, str]:
        dimensions = set(caps)
        direct = await self.measured_run_spend(db, run, dimensions)
        active = await self.active_run_commitments(db, run, dimensions, caps)
        return {
            key: format(Decimal(caps[key]) - Decimal(direct[key]) - Decimal(active[key]), "f")
            for key in caps
        }

    async def remaining(self, db, parent):
        caps = canonical_amounts(parent.budget.get("caps", {}), allowed=ROADMAP_BUDGET_DIMENSIONS)
        parent_run = await db.scalar(select(OrchestrationRun).where(
            OrchestrationRun.goal_id == parent.id
        ).order_by(OrchestrationRun.started_at.desc()).limit(1))
        excluded = self._discovery_task_ids(parent_run) if parent_run is not None else set()
        direct = await self.measured_run_spend(db, parent_run, set(caps), excluded_task_ids=excluded)
        commitments = await self.active_run_commitments(
            db, parent_run, set(caps), caps, excluded_task_ids=excluded,
        )
        reservations = list((await db.scalars(select(OrchestrationBudgetReservation).where(
            OrchestrationBudgetReservation.parent_goal_id == parent.id
        ))).all())
        settled = self._sum((row.settled_spend for row in reservations if row.status == "settled"), set(caps))
        active = self._sum((row.allocation for row in reservations if row.status == "active"), set(caps))
        remaining = {key: format(Decimal(caps[key]) - Decimal(direct[key]) - Decimal(settled[key]) - Decimal(active[key]) - Decimal(commitments[key]), "f") for key in caps}
        return {"caps": caps, "direct_spend": direct, "settled_child_spend": settled,
                "active_reservations": active, "active_commitments": commitments, "remaining": remaining}

    async def continuous_available(self, db, parent, policy, now):
        dimensions = set(parent_caps(parent))
        limits = canonical_amounts(policy["rolling_budget"]["limits"], allowed=dimensions)
        cutoff = (now or _utcnow()) - timedelta(seconds=policy["rolling_budget"]["window_seconds"])
        recent_rows = list((await db.scalars(select(OrchestrationBudgetReservation).where(
            OrchestrationBudgetReservation.parent_goal_id == parent.id,
            OrchestrationBudgetReservation.continuous_origin_key.is_not(None),
            OrchestrationBudgetReservation.status == "settled",
            OrchestrationBudgetReservation.settled_at >= cutoff,
        ))).all())
        active_rows = list((await db.scalars(select(OrchestrationBudgetReservation).where(
            OrchestrationBudgetReservation.parent_goal_id == parent.id,
            OrchestrationBudgetReservation.continuous_origin_key.is_not(None),
            OrchestrationBudgetReservation.status == "active",
        ))).all())
        recent = self._sum((row.settled_spend for row in recent_rows), dimensions)
        active = self._sum((row.allocation for row in active_rows), dimensions)
        rolling = {
            key: format(Decimal(limits[key]) - Decimal(recent[key]) - Decimal(active[key]), "f")
            for key in dimensions
        }
        total = (await self.remaining(db, parent))["remaining"]
        available = {key: format(min(Decimal(total[key]), Decimal(rolling[key])), "f") for key in dimensions}
        return {"total": total, "rolling": rolling, "available": available}

    async def discovery_available(self, db, parent, source, now):
        dimensions = set(parent_caps(parent))
        rolling = source["budget"]["rolling"]
        limits = canonical_amounts(rolling["limits"], allowed=dimensions)
        cutoff = now - timedelta(seconds=rolling["window_seconds"])
        recent = list((await db.scalars(select(OrchestrationBudgetReservation).where(
            OrchestrationBudgetReservation.parent_goal_id == parent.id,
            OrchestrationBudgetReservation.discovery_run_id.is_not(None),
            OrchestrationBudgetReservation.status == "settled",
            OrchestrationBudgetReservation.settled_at >= cutoff,
        ))).all())
        active = list((await db.scalars(select(OrchestrationBudgetReservation).where(
            OrchestrationBudgetReservation.parent_goal_id == parent.id,
            OrchestrationBudgetReservation.discovery_run_id.is_not(None),
            OrchestrationBudgetReservation.status == "active",
        ))).all())
        spent = self._sum((row.settled_spend for row in recent), dimensions)
        allocated = self._sum((row.allocation for row in active), dimensions)
        rolling_remaining = {
            key: format(Decimal(limits[key]) - Decimal(spent[key]) - Decimal(allocated[key]), "f")
            for key in dimensions
        }
        total = (await self.remaining(db, parent))["remaining"]
        return {
            "total": total,
            "rolling": rolling_remaining,
            "available": {
                key: format(min(Decimal(total[key]), Decimal(rolling_remaining[key])), "f")
                for key in dimensions
            },
        }

    async def reserve_discovery_source(self, db, parent, run, source, now):
        existing = await db.scalar(select(OrchestrationBudgetReservation).where(
            OrchestrationBudgetReservation.discovery_run_id == run.id,
        ))
        if existing is not None:
            return existing
        allocation = canonical_amounts(
            source["budget"]["per_cycle"], allowed=set(parent_caps(parent)),
        )
        available = await self.discovery_available(db, parent, source, now)
        if any(
            Decimal(amount) > Decimal(available["available"][key])
            for key, amount in allocation.items()
        ):
            raise HTTPException(status_code=409, detail="continuous_discovery_budget_wait")
        reservation = OrchestrationBudgetReservation(
            id=uuid.uuid5(uuid.NAMESPACE_URL, f"rally:discovery-reservation:{run.id}"),
            parent_goal_id=parent.id, roadmap_item_id=None, continuous_origin_key=None,
            discovery_run_id=run.id, child_goal_id=None, allocation=allocation,
        )
        db.add(reservation)
        await db.flush()
        return reservation

    async def discovery_remaining(self, db, run, caps):
        task_ids = self._discovery_task_ids(run)
        dimensions = set(caps)
        spent = await self.measured_task_spend(db, task_ids, dimensions)
        commitments = await self.active_task_commitments(db, task_ids, dimensions, caps)
        return {
            key: format(Decimal(caps[key]) - Decimal(spent[key]) - Decimal(commitments[key]), "f")
            for key in caps
        }

    async def settle_discovery_source(self, db, reservation, run, *, reason):
        if reservation is None:
            from huddleroom.services.session_service import SessionClaimAttention

            raise SessionClaimAttention(run.id, {
                "kind": "budget_integrity", "scope": f"discovery_claim:{run.id}",
                "reason": "Discovery reservation is not active.",
            }, "Discovery reservation is not active.")
        if reservation.status == "settled":
            return reservation
        dimensions = set(reservation.allocation)
        task_ids = self._discovery_task_ids(run)
        if reason == "completed":
            actual = await self.measured_task_spend(db, task_ids, dimensions)
            measurement_complete = True
        else:
            actual, measurement_complete = await self.known_task_spend(db, task_ids, dimensions)
            if not measurement_complete:
                actual = {
                    key: format(max(Decimal(actual[key]), Decimal(reservation.allocation[key])), "f")
                    for key in dimensions
                }
        if reason == "completed" and any(
            Decimal(actual[key]) > Decimal(reservation.allocation[key]) for key in actual
        ):
            raise HTTPException(status_code=409, detail="Discovery source exceeded reserved budget")
        reservation.settled_spend = actual
        reservation.measurement_complete = measurement_complete
        reservation.status = "settled"
        reservation.settlement_reason = reason
        reservation.settled_at = _utcnow()
        await db.flush()
        return reservation

    async def reserve_continuous_child(self, db, parent, child, origin_key, allocation, policy, now):
        existing = await db.scalar(select(OrchestrationBudgetReservation).where(
            OrchestrationBudgetReservation.parent_goal_id == parent.id,
            OrchestrationBudgetReservation.continuous_origin_key == origin_key,
        ))
        if existing is not None:
            return existing
        canonical = canonical_amounts(allocation, allowed=set(parent_caps(parent)))
        summary = await self.continuous_available(db, parent, policy, now)
        if any(Decimal(amount) > Decimal(summary["available"][key]) for key, amount in canonical.items()):
            raise HTTPException(status_code=409, detail="continuous_budget_wait")
        reservation = OrchestrationBudgetReservation(
            id=uuid.uuid5(uuid.NAMESPACE_URL, f"rally:continuous-reservation:{parent.id}:{origin_key}"),
            parent_goal_id=parent.id, roadmap_item_id=None, child_goal_id=child.id,
            continuous_origin_key=origin_key, allocation=canonical,
        )
        db.add(reservation)
        await db.flush()
        return reservation

    async def reserve_child(self, db, parent, item, allocation):
        existing = await db.scalar(select(OrchestrationBudgetReservation).where(
            OrchestrationBudgetReservation.roadmap_item_id == item.id
        ))
        if existing is not None:
            return existing
        canonical = canonical_amounts(allocation, allowed=set(parent_caps(parent)))
        summary = await self.remaining(db, parent)
        if any(Decimal(amount) > Decimal(summary["remaining"].get(key, "0")) for key, amount in canonical.items()):
            raise HTTPException(status_code=409, detail="Roadmap child allocation exceeds remaining parent budget")
        reservation = OrchestrationBudgetReservation(
            id=uuid.uuid5(uuid.NAMESPACE_URL, f"rally:budget-reservation:{parent.id}:{item.item_key}"),
            parent_goal_id=parent.id, roadmap_item_id=item.id, child_goal_id=item.child_goal_id,
            allocation=canonical,
        )
        db.add(reservation)
        await db.flush()
        return reservation

    async def settle_child(self, db, reservation, child_run, *, reason, conservative_cancel=True):
        if reservation.status == "settled":
            return reservation
        actual, measurement_complete = {}, True
        for dimension, allocation in reservation.allocation.items():
            if reason == "cancelled":
                spend, complete = await self.known_run_spend(db, child_run, {dimension})
                actual[dimension] = spend[dimension]
                measurement_complete = measurement_complete and complete
            else:
                actual[dimension] = (await self.measured_run_spend(db, child_run, {dimension}))[dimension]
            if reason == "cancelled" and conservative_cancel:
                actual[dimension] = format(max(Decimal(actual[dimension]), Decimal(allocation)), "f")
        if reason != "cancelled" or not conservative_cancel:
            for key, amount in actual.items():
                if Decimal(amount) > Decimal(reservation.allocation[key]):
                    raise HTTPException(status_code=409, detail=f"Child exceeded reserved {key}")
        reservation.settled_spend = actual
        reservation.measurement_complete = measurement_complete
        reservation.status = "settled"
        reservation.settlement_reason = reason
        reservation.settled_at = _utcnow()
        await db.flush()
        return reservation
