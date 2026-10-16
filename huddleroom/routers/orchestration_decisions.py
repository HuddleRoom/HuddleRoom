import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.config import settings
from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.user import User
from huddleroom.schemas.orchestration import (
    OrchestrationAgentDefinitionReviewBatchAnswerRequest,
    OrchestrationAgentDefinitionReviewBatchAnswerResponse,
    OrchestrationAuthorityDecisionResponse,
    OrchestrationCheckpointResponse,
    OrchestrationDecisionAnswerResponse,
    OrchestrationDecisionAnswerRequest,
    OrchestrationDecisionCancelRequest,
)
from huddleroom.services.orchestration_recovery_presenter import present_decision_response, present_decision_responses
from huddleroom.services.orchestration_needs_you_presenter import attach_decision_references
from huddleroom.services.orchestration_authority_interview import build_checkpoint
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_service import OrchestrationService

from huddleroom.services import orchestration_decision_application as decision_application

router = APIRouter()
orchestration_service = OrchestrationService()
decision_service = OrchestrationAuthorityDecisionService()
_present = present_decision_responses
_present_one = present_decision_response


async def _goal_or_404(db: AsyncSession, project_id: uuid.UUID, goal_id: uuid.UUID):
    goal = await orchestration_service.get_goal(db, project_id, goal_id)
    if goal is None:
        raise HTTPException(status_code=404, detail="Orchestration goal not found")
    return goal


async def _decision_or_404(
    db: AsyncSession, goal_id: uuid.UUID, decision_id: uuid.UUID
) -> OrchestrationAuthorityDecision:
    decision = await db.get(OrchestrationAuthorityDecision, decision_id)
    if decision is None or decision.goal_id != goal_id:
        raise HTTPException(status_code=404, detail="Orchestration authority decision not found")
    return decision


@router.get("/goals/{goal_id}/decisions", response_model=list[OrchestrationAuthorityDecisionResponse])
async def list_decisions(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    status: str | None = None,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _goal_or_404(db, project_id, goal_id)
    decisions = await decision_service.list_decisions(db, goal_id, status=status)
    return await attach_decision_references(
        db, project_id=project_id, decisions=decisions, responses=await _present(db, decisions),
    )


@router.get("/goals/{goal_id}/decisions/checkpoint", response_model=OrchestrationCheckpointResponse)
async def get_checkpoint(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _goal_or_404(db, project_id, goal_id)
    pending = await decision_service.list_decisions(db, goal_id, status="pending")
    max_questions = settings.orchestration_checkpoint_max_questions
    checkpoint, deferred = build_checkpoint(pending, max_questions=max_questions)
    responses = await _present(db, list(checkpoint))
    return OrchestrationCheckpointResponse(
        goal_id=goal_id,
        items=await attach_decision_references(
            db, project_id=project_id, decisions=checkpoint, responses=responses,
        ),
        deferred_count=len(deferred),
        max_questions=max_questions,
    )



@router.post(
    "/goals/{goal_id}/decisions/agent-definition-review/batch-answer",
    response_model=OrchestrationAgentDefinitionReviewBatchAnswerResponse,
)
async def answer_agent_definition_review_batch(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    data: OrchestrationAgentDefinitionReviewBatchAnswerRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await decision_application.answer_agent_review_batch(db, project_id, goal_id, data, user, request_owns_transaction=True)


@router.post(
    "/goals/{goal_id}/decisions/{decision_id}/answer",
    response_model=OrchestrationDecisionAnswerResponse,
)
async def answer_decision(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    decision_id: uuid.UUID,
    data: OrchestrationDecisionAnswerRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await decision_application.answer_decision(db, project_id, goal_id, decision_id, data, user, request_owns_transaction=True)


@router.post(
    "/goals/{goal_id}/decisions/{decision_id}/cancel",
    response_model=OrchestrationAuthorityDecisionResponse,
)
async def cancel_decision(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    decision_id: uuid.UUID,
    data: OrchestrationDecisionCancelRequest,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _goal_or_404(db, project_id, goal_id)
    decision = await _decision_or_404(db, goal_id, decision_id)
    try:
        return await _present_one(db, await decision_service.cancel_decision(db, decision, reason=data.reason))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/goals/{goal_id}/decisions/{decision_id}", response_model=OrchestrationDecisionAnswerResponse)
async def get_decision(project_id: uuid.UUID, goal_id: uuid.UUID, decision_id: uuid.UUID,
                       _: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await _goal_or_404(db, project_id, goal_id)
    decision = await _decision_or_404(db, goal_id, decision_id)
    from huddleroom.models.orchestration import OrchestrationAction
    from huddleroom.services.orchestration_answer_receipts import receipt, runless_receipt
    action = await db.get(OrchestrationAction, decision.continuation_action_id) if decision.continuation_action_id else None
    return OrchestrationDecisionAnswerResponse(decision=await _present_one(db, decision),
        answer_receipt=receipt(action) if action and action.action_type == "apply_authority_answer" else runless_receipt(decision))
