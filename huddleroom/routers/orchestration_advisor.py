import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.config import settings
from huddleroom.database import get_db
from huddleroom.dependencies import ensure_project_exists, get_current_user
from huddleroom.models.orchestration_advisor import ProjectAdvisorTurn, advisor_allowance_used
from huddleroom.models.user import User
from huddleroom.schemas.orchestration_advisor import (
    ProjectAdvisorAllowance,
    ProjectAdvisorHistoryResponse,
    ProjectAdvisorSubmitRequest,
    ProjectAdvisorTurnResponse,
)
from huddleroom.services.orchestration_conversation_service import ConversationDomainError
from huddleroom.services.orchestration_project_advisor_service import OrchestrationProjectAdvisorService

router = APIRouter()
service = OrchestrationProjectAdvisorService()


def _advisor_error(error: ConversationDomainError) -> HTTPException:
    return HTTPException(
        status_code=error.status_code,
        detail={"code": error.code, "message": error.message},
    )


def _turn_response(turn: ProjectAdvisorTurn) -> ProjectAdvisorTurnResponse:
    return ProjectAdvisorTurnResponse(
        id=turn.id,
        question=turn.question,
        answer=turn.answer,
        citations=turn.citations or [],
        off_topic=turn.off_topic,
        status=turn.status,
        created_at=turn.created_at,
    )


async def _allowance(db: AsyncSession, project_id: uuid.UUID, actor_id: uuid.UUID) -> ProjectAdvisorAllowance:
    limit = settings.orchestration_advisor_allowance_tokens
    unlimited = limit < 0
    used = await advisor_allowance_used(db, project_id, actor_id)
    remaining = -1 if unlimited else max(0, limit - used) if used is not None else 0
    return ProjectAdvisorAllowance(
        enabled=limit != 0, unlimited=unlimited, limit=limit, remaining=remaining
    )


@router.post("/conversation", response_model=ProjectAdvisorTurnResponse)
async def submit_advisor_turn(
    project_id: uuid.UUID,
    data: ProjectAdvisorSubmitRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    try:
        turn = await service.ask(db, project_id, user.id, data.content)
    except ConversationDomainError as error:
        raise _advisor_error(error) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="advisor_unavailable") from error
    return _turn_response(turn)


@router.get("/conversation", response_model=ProjectAdvisorHistoryResponse)
async def get_advisor_history(
    project_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    turns = await service.history(db, project_id, user.id)
    return ProjectAdvisorHistoryResponse(
        items=[_turn_response(turn) for turn in turns],
        allowance=await _allowance(db, project_id, user.id),
    )
