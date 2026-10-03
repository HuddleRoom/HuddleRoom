import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import SessionTransactionOrigin

from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user
from huddleroom.models.orchestration_process import PROCESS_TYPES
from huddleroom.models.user import User
from huddleroom.schemas.orchestration import (
    OrchestrationAgentReviewResponse,
    OrchestrationProcessRunResponse,
    OrchestrationProcessSkipRequest,
    OrchestrationProcessStartRequest,
)
from huddleroom.services.orchestration_agent_review_service import OrchestrationAgentReviewService
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_service import LLM_DECISION_RUN_STATUSES, OrchestrationService
from huddleroom.services.project_service import ProjectService

router = APIRouter()
process_service = OrchestrationProcessService()
orchestration_service = OrchestrationService()
agent_review_service = OrchestrationAgentReviewService()

# Deviation 15: only process types with implemented advance logic may be
# force-started. Skip has no such requirement (it's a legitimate no-op
# marker for any registered type, spec 6.5), so it stays open to all of
# PROCESS_TYPES.
STARTABLE_PROCESS_TYPES = {
    "goal_definition",
    "manager_selection",
    "agent_definition_review",
    "team_hierarchy",
    "effectiveness_review",
    "goal_closeout",
}

# Goal statuses tick() will still act on. A goal outside this set (completed,
# cancelled) must never receive a new baseline process from force-start --
# nothing would ever tick it back to terminal (review finding, MEDIUM).
_STARTABLE_GOAL_STATUSES = {"active", "blocked"}


async def _goal_or_404(db: AsyncSession, project_id: uuid.UUID, goal_id: uuid.UUID):
    goal = await orchestration_service.get_goal(db, project_id, goal_id)
    if goal is None:
        raise HTTPException(status_code=404, detail="Orchestration goal not found")
    return goal


def _ensure_known_process_type(process_type: str) -> None:
    if process_type not in PROCESS_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown process type '{process_type}'")


@router.get(
    "/goals/{goal_id}/agent-reviews",
    response_model=list[OrchestrationAgentReviewResponse],
)
async def list_agent_reviews(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    agent_id: uuid.UUID | None = None,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _goal_or_404(db, project_id, goal_id)
    return await agent_review_service.list_reviews(db, goal_id, agent_id=agent_id)


@router.get(
    "/goals/{goal_id}/processes",
    response_model=list[OrchestrationProcessRunResponse],
)
async def list_processes(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _goal_or_404(db, project_id, goal_id)
    return await process_service.list_process_runs(db, goal_id)


@router.post(
    "/goals/{goal_id}/processes/{process_type}/start",
    response_model=OrchestrationProcessRunResponse,
)
async def start_process(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    process_type: str,
    data: OrchestrationProcessStartRequest,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _ensure_known_process_type(process_type)
    if process_type not in STARTABLE_PROCESS_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"process type '{process_type}' has no implemented process logic yet",
        )
    transaction = db.sync_session.get_transaction()
    caller_owns_transaction = (
        transaction is not None
        and transaction.origin != SessionTransactionOrigin.AUTOBEGIN
    )
    # Confirm the goal exists before locking (keeps 404 semantics); the goal
    # and its active run are re-read fresh INSIDE the lock below (review
    # finding, MEDIUM) -- reading them before locking let a concurrent tick
    # complete the run, or complete the goal itself, while this request was
    # still forming its start_process() call against the stale copies.
    await _goal_or_404(db, project_id, goal_id)
    try:
        # Lock goal to serialize against concurrent tick() and force-starts on same goal.
        # This ensures a force-start doesn't race with baseline advancement or completion.
        async with orchestration_service._lock_goal_for_baseline_transition(db, goal_id):
            goal = await _goal_or_404(db, project_id, goal_id)
            if goal.status not in _STARTABLE_GOAL_STATUSES:
                raise HTTPException(
                    status_code=409,
                    detail=f"goal is '{goal.status}'; cannot start a new process",
                )
            run = await orchestration_service.get_active_run_for_goal(db, project_id, goal_id)
            if run is not None and run.status not in LLM_DECISION_RUN_STATUSES:
                raise HTTPException(
                    status_code=409,
                    detail=f"active run is '{run.status}'; not tickable",
                )
            if process_type == "agent_definition_review":
                manager_selection = await process_service.get_current(
                    db, goal.id, "manager_selection"
                )
                if manager_selection is None or manager_selection.status not in {
                    "completed",
                    "skipped",
                }:
                    raise HTTPException(
                        status_code=409,
                        detail="Manager selection must be terminal before agent definition review",
                    )
            await ProjectService().lock_workspace_boundary(db, project_id)
            await ProjectService().require_runnable_project(db, project_id)
            successor = None
            input_snapshot = (
                {"mode": "completion", "full_closeout": True}
                if process_type == "goal_closeout"
                else None
            )
            if process_type == "team_hierarchy":
                agent_review = await process_service.get_current(
                    db, goal.id, "agent_definition_review"
                )
                if agent_review is None or agent_review.status not in {"completed", "skipped"}:
                    raise HTTPException(
                        status_code=409,
                        detail="Agent definition review must be terminal before team hierarchy",
                    )
                hierarchy = await process_service.get_current(
                    db, goal.id, "team_hierarchy"
                )
                if (
                    hierarchy is not None
                    and hierarchy.status == "waiting_decision"
                    and hierarchy.outputs.get("change_requested") is True
                ):
                    hierarchy.superseded_by_id = hierarchy.id
                    await db.flush()
                    successor = await process_service.start_process(
                        db,
                        goal.id,
                        process_type=process_type,
                        trigger_reason=f"human requested: {data.reason}",
                        run_id=run.id if run is not None else None,
                        input_snapshot=input_snapshot,
                    )
                    hierarchy.superseded_by_id = successor.id
                    await db.flush()
            if successor is None:
                successor = await process_service.start_process(
                    db,
                    goal.id,
                    process_type=process_type,
                    trigger_reason=f"human requested: {data.reason}",
                    run_id=run.id if run is not None else None,
                    input_snapshot=input_snapshot,
                )
            await db.flush()
            bind = db.get_bind()
            if (
                bind is not None
                and bind.dialect.name == "sqlite"
                and not caller_owns_transaction
            ):
                await db.commit()
            return successor
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/goals/{goal_id}/processes/{process_type}/skip",
    response_model=OrchestrationProcessRunResponse,
)
async def skip_process(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    process_type: str,
    data: OrchestrationProcessSkipRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _ensure_known_process_type(process_type)
    # Confirm the goal exists before locking (keeps 404 semantics); the goal
    # and its active run are re-read fresh INSIDE the lock below (review
    # finding, MEDIUM) -- reading them before locking let a concurrent tick
    # complete the run, or complete the goal itself, while this request was
    # still forming its skip_process() call against the stale copies.
    await _goal_or_404(db, project_id, goal_id)
    try:
        # Lock goal BEFORE the process-row lock skip takes internally (review
        # finding, MEDIUM): tick() locks goal-then-process-row, and skip's own
        # repair path (orchestration_process_service.skip_process) locks the
        # process row FOR UPDATE -- without this outer goal lock, skip would
        # take process-row-then-goal on Postgres, a lock-order inversion
        # against tick's goal-then-process-row that can deadlock.
        async with orchestration_service._lock_goal_for_baseline_transition(db, goal_id):
            goal = await _goal_or_404(db, project_id, goal_id)
            run = await orchestration_service.get_active_run_for_goal(db, project_id, goal_id)
            return await process_service.skip_process(
                db,
                goal.id,
                process_type=process_type,
                skipped_by=f"human:{user.id}",
                reason=data.reason,
                run_id=run.id if run is not None else None,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
