import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user
from huddleroom.models.orchestration_process import OrchestrationWarning
from huddleroom.models.user import User
from huddleroom.schemas.orchestration import (
    OrchestrationWarningReasonRequest,
    OrchestrationWarningResponse,
)
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

router = APIRouter()
orchestration_service = OrchestrationService()
warning_service = OrchestrationWarningService()
decision_service = OrchestrationAuthorityDecisionService()


async def _goal_or_404(db: AsyncSession, project_id: uuid.UUID, goal_id: uuid.UUID):
    goal = await orchestration_service.get_goal(db, project_id, goal_id)
    if goal is None:
        raise HTTPException(status_code=404, detail="Orchestration goal not found")
    return goal


async def _warning_or_404(
    db: AsyncSession, project_id: uuid.UUID, warning_id: uuid.UUID
) -> OrchestrationWarning:
    warning = await db.get(OrchestrationWarning, warning_id)
    if warning is None:
        raise HTTPException(status_code=404, detail="Orchestration warning not found")
    await _goal_or_404(db, project_id, warning.goal_id)
    return warning


@router.get("/goals/{goal_id}/warnings", response_model=list[OrchestrationWarningResponse])
async def list_warnings(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    active_only: bool = False,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _goal_or_404(db, project_id, goal_id)
    return await warning_service.list_warnings(db, goal_id, active_only=active_only)


@router.post("/warnings/{warning_id}/acknowledge", response_model=OrchestrationWarningResponse)
async def acknowledge_warning(
    project_id: uuid.UUID,
    warning_id: uuid.UUID,
    data: OrchestrationWarningReasonRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    warning = await _warning_or_404(db, project_id, warning_id)
    try:
        decision = await decision_service.create_pending(
            db,
            warning.goal_id,
            decision_key=f"acknowledge-warning-{warning.id}",
            title=f"Acknowledge warning: {warning.warning_type}",
            question=warning.message,
            authority="human",
            options=["acknowledge"],
            recommendation="acknowledge",
            run_id=warning.run_id,
        )
        await decision_service.answer_decision(
            db,
            decision,
            selected_option="acknowledge",
            reason=data.reason,
            decided_by_user_id=user.id,
        )
        return await warning_service.acknowledge_warning(
            db, warning, acknowledged_by=f"human:{user.id}"
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/warnings/{warning_id}/resolve", response_model=OrchestrationWarningResponse)
async def resolve_warning(
    project_id: uuid.UUID,
    warning_id: uuid.UUID,
    data: OrchestrationWarningReasonRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    warning = await _warning_or_404(db, project_id, warning_id)
    try:
        async with orchestration_service._lock_goal_for_baseline_transition(db, warning.goal_id):
            return await warning_service.resolve_warning(
                db, warning, resolved_by=f"human:{user.id}", reason=data.reason
            )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
