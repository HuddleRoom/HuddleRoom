import uuid
from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user
from huddleroom.models.user import User
from huddleroom.models.session import Session as SessionModel
from huddleroom.schemas.session import SessionCreate, SessionResponse, SessionOutputResponse
from huddleroom.schemas.common import CursorPage
from huddleroom.services.session_service import SessionService
from huddleroom.services.session_service import SessionClaimAttention

# MVP: no project-level authorization — any authenticated user/agent can access any session

router = APIRouter()
service = SessionService()


async def _commit_claim_attention(db: AsyncSession, attention: SessionClaimAttention) -> None:
    await service.commit_claim_attention(db, attention)


@router.get("", response_model=CursorPage[SessionResponse])
async def list_sessions(
    agent_id: uuid.UUID | None = None,
    task_id: uuid.UUID | None = None,
    project_id: uuid.UUID | None = None,
    status: str | None = None,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    items, next_cursor = await service.list(
        db, agent_id=agent_id, task_id=task_id, project_id=project_id,
        status=status, cursor=cursor, limit=limit
    )
    return CursorPage(items=items, next_cursor=next_cursor)


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(
    data: SessionCreate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await service.create(db, data)
    except SessionClaimAttention as exc:
        await _commit_claim_attention(db, exc)
        raise


@router.get("/count")
async def count_sessions(
    project_id: uuid.UUID | None = None,
    status: str | None = None,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    conditions = []
    if project_id is not None:
        conditions.append(SessionModel.project_id == project_id)
    if status is not None:
        conditions.append(SessionModel.status == status)
    if conditions:
        result = await db.execute(select(func.count(SessionModel.id)).where(*conditions))  # pylint: disable=not-callable
    else:
        result = await db.execute(select(func.count(SessionModel.id)))  # pylint: disable=not-callable
    return {"count": result.scalar_one()}


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.get_or_404(db, session_id)


@router.post("/{session_id}/cancel", response_model=SessionResponse)
async def cancel_session(
    session_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.cancel(db, session_id)


@router.post("/{session_id}/resume", response_model=SessionResponse)
async def resume_session(
    session_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await service.resume(db, session_id)
    except SessionClaimAttention as exc:
        await _commit_claim_attention(db, exc)
        raise


@router.get("/{session_id}/output", response_model=SessionOutputResponse)
async def get_session_output(
    session_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.get_output(db, session_id)
