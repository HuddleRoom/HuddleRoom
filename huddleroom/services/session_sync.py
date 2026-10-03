from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun
from huddleroom.services.event_bus import emit_event


async def sync_task_from_session(db: AsyncSession, session: Session, *, settle_budget: bool = True) -> None:
    """Synchronize task status from a completed or failed session.

    If a session has been completed or failed and is linked to a task,
    update the task's status accordingly.
    """
    if session.task_id is None:
        return
    if session.status not in ("completed", "failed", "cancelled"):
        return
    result = await db.execute(select(Task).where(Task.id == session.task_id))
    task = result.scalar_one_or_none()
    if not task:
        return
    changed = task.status in ("in_progress", "ready")
    if changed:
        now = datetime.now(timezone.utc)
        if session.status == "completed":
            task.status = "done"
            task.completed_at = now
        else:
            task.status = "failed"
        await db.flush()
    # A retry can replace the task's current action while an older session is
    # still reporting terminal telemetry.  Session lineage is authoritative.
    orchestration = (session.metadata_ or {}).get("orchestration", {})
    try:
        action_id = uuid.UUID(str(orchestration.get("action_id")))
    except (TypeError, ValueError, AttributeError):
        action_id = None
    action = await db.scalar(select(OrchestrationAction).where(
        OrchestrationAction.id == action_id,
        OrchestrationAction.status == "completed",
        or_(
            (OrchestrationAction.target_type == "task") & (OrchestrationAction.target_id == task.id),
            (OrchestrationAction.target_type == "session") & (OrchestrationAction.target_id == session.id),
        ),
    )) if action_id is not None else None
    if settle_budget and action is not None and action.budget_ledger:
        run = await db.get(OrchestrationRun, action.run_id)
        goal = await db.get(OrchestrationGoal, run.goal_id) if run is not None else None
        if run is not None and goal is not None:
            from huddleroom.services.orchestration_budget_service import BudgetMeasurementError, OrchestrationBudgetService

            budget = OrchestrationBudgetService()
            dimensions = set((run.budget_state or {}).get("caps", goal.budget.get("caps", {})))
            scope = f"action:{action.id}:session:{session.id}"
            try:
                actual = {
                    dimension: format(budget._session_spend(session, dimension, allow_incomplete=True), "f")
                    for dimension in dimensions
                }
                complete = (session.metadata_ or {}).get("token_usage_complete") is not False
                await budget.settle_action_budget(
                    db, goal, run, action, actual, measurement_complete=complete,
                    observation_id=f"action:{action.id}:session:{session.id}:usage",
                )
                if complete:
                    run.active_blockers = [item for item in (run.active_blockers or []) if not (
                        isinstance(item, dict) and item.get("kind") == "budget_measurement"
                        and item.get("scope") == scope
                    )]
                else:
                    run.active_blockers = [
                        *[item for item in (run.active_blockers or []) if not (
                            isinstance(item, dict) and item.get("kind") == "budget_measurement"
                            and item.get("scope") == scope
                        )],
                        {"kind": "budget_measurement", "session_id": str(session.id),
                         "action_id": str(action.id), "scope": scope},
                    ]
            except BudgetMeasurementError as exc:
                # Keep the committed hold and surface uncertainty rather than treating
                # missing telemetry as free capacity.
                await budget.settle_action_budget(
                    db, goal, run, action, {}, measurement_complete=False,
                    observation_id=f"action:{action.id}:session:{session.id}:usage",
                )
                run.active_blockers = [
                    *[item for item in (run.active_blockers or []) if not (
                        isinstance(item, dict) and item.get("scope") == scope
                    )],
                    {"kind": "budget_measurement", "dimension": exc.dimension,
                     "session_id": str(session.id), "action_id": str(action.id),
                     "scope": scope},
                ]
    if changed:
        await emit_event(db, task.project_id, "task.status_changed", {
            "task_id": str(task.id),
            "status": task.status,
            "previous_status": "in_progress",
            "source": "session_sync",
            "project_id": str(task.project_id),
        })
