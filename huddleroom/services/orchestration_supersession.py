from __future__ import annotations

import uuid
from copy import deepcopy

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationBudgetReservation, OrchestrationGoal, OrchestrationRun
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.project_service import ProjectService
from huddleroom.services.orchestration_service import OrchestrationService


class OrchestrationSupersessionService:
    def __init__(self) -> None:
        self._orch = OrchestrationService()

    async def supersede(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
        *,
        new_goal_type: str,
        actor: str,
    ) -> OrchestrationGoal:
        caller_owns_transaction = db.in_transaction()
        if caller_owns_transaction:
            async with self._orch._lock_goal_for_baseline_transition(db, goal_id):
                return await self._supersede_locked(db, project_id, goal_id, new_goal_type, actor)

        async with self._orch._lock_goal_for_baseline_transition(db, goal_id):
            if db.get_bind().dialect.name == "sqlite":
                # SQLite needs a real outer transaction before a SAVEPOINT;
                # otherwise releasing it commits the work.
                async with db.begin():
                    return await self._supersede_locked(db, project_id, goal_id, new_goal_type, actor)
            try:
                # PostgreSQL's FOR UPDATE in the lock opens this transaction.
                # Keep its commit/rollback under that lock.
                replacement = await self._supersede_locked(db, project_id, goal_id, new_goal_type, actor)
                await db.commit()
                return replacement
            except Exception:
                await db.rollback()
                raise

    async def _supersede_locked(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
        new_goal_type: str,
        actor: str,
    ) -> OrchestrationGoal:
        goal = await db.scalar(
            select(OrchestrationGoal)
            .where(OrchestrationGoal.id == goal_id, OrchestrationGoal.project_id == project_id)
            .execution_options(populate_existing=True)
        )
        if goal is None:
            raise HTTPException(status_code=404, detail="Orchestration goal not found")

        existing = await db.scalar(
            select(OrchestrationGoal).where(OrchestrationGoal.supersedes_goal_id == goal.id)
        )
        if existing is not None:
            return existing

        run = await self._orch.get_run_for_goal(db, project_id, goal.id)
        await self._require_ready(db, goal, run)
        await ProjectService().lock_workspace_boundary(db, project_id)
        await ProjectService().require_runnable_project(db, project_id)
        original_goal_id = goal.id
        try:
            async with db.begin_nested():
                await self._orch._cancel_goal_no_commit(db, goal, run, cancelled_by=actor)
                replacement = OrchestrationGoal(
                    project_id=project_id,
                    objective=goal.objective,
                    original_request=goal.original_request,
                    success_criteria=deepcopy(goal.success_criteria),
                    constraints=deepcopy(goal.constraints),
                    budget=deepcopy(goal.budget),
                    weight=goal.weight,
                    explicit_multi_work_function=goal.explicit_multi_work_function,
                    goal_type=new_goal_type,
                    supersedes_goal_id=original_goal_id,
                    created_by_user_id=goal.created_by_user_id,
                )
                db.add(replacement)
                await db.flush()
                db.add(OrchestrationRun(
                    goal_id=replacement.id, budget_state=deepcopy(replacement.budget), baseline_authorized=False,
                ))
                await db.flush()
        except IntegrityError:
            existing = await db.scalar(
                select(OrchestrationGoal).where(OrchestrationGoal.supersedes_goal_id == original_goal_id)
            )
            if existing is None:
                raise
            return existing
        return replacement

    async def _require_ready(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun | None,
    ) -> None:
        if goal.status != "active" or run is None or run.status != "running" or run.phase not in ("baseline", "ready"):
            raise self._not_ready("goal must be active in baseline or ready phase")
        if await self._orch._orchestrated_tasks_for_run(
            db, run.id, statuses=["backlog", "ready", "in_progress", "blocked"]
        ):
            raise self._not_ready("unfinished child tasks")
        if await db.scalar(
            select(Session.id)
            .join(Task, Task.id == Session.task_id)
            .where(
                Session.status.in_(("pending", "running")),
                Task.metadata_["orchestration"]["run_id"].as_string() == str(run.id),
            )
            .limit(1)
        ) is not None:
            raise self._not_ready("active sessions")
        if await db.scalar(
            select(OrchestrationAction.id)
            .where(OrchestrationAction.run_id == run.id, OrchestrationAction.status == "reserved")
            .limit(1)
        ) is not None:
            raise self._not_ready("reserved actions")
        if await db.scalar(select(OrchestrationGoal.id).where(
            OrchestrationGoal.parent_goal_id == goal.id,
            OrchestrationGoal.status.not_in(("completed", "cancelled")),
        ).limit(1)) is not None:
            raise self._not_ready("unfinished descendants")
        if await db.scalar(select(OrchestrationBudgetReservation.id).where(
            OrchestrationBudgetReservation.parent_goal_id == goal.id,
            OrchestrationBudgetReservation.status == "active",
        ).limit(1)) is not None:
            raise self._not_ready("active child reservations")

    @staticmethod
    def _not_ready(message: str) -> HTTPException:
        return HTTPException(
            status_code=409,
            detail={"conflict": "supersession_not_ready", "message": message},
        )
