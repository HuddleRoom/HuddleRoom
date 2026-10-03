import uuid
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user
from huddleroom.models.user import User
from huddleroom.models.session import Session
from huddleroom.schemas.agent import AgentCreate, AgentResponse, AgentUpdate, AgentContextResponse
from huddleroom.schemas.session import SessionResponse
from huddleroom.schemas.common import CursorPage
from huddleroom.services.agent_service import AgentService

router = APIRouter()
service = AgentService()


@router.get("", response_model=CursorPage[AgentResponse])
async def list_agents(
    role: str | None = None,
    is_active: bool | None = None,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    items, next_cursor = await service.list(db, role=role, is_active=is_active, cursor=cursor, limit=limit)
    return CursorPage(items=items, next_cursor=next_cursor)


@router.post("", response_model=AgentResponse, status_code=201)
async def create_agent(
    data: AgentCreate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.create(db, data)


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.get_or_404(db, agent_id)


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: uuid.UUID,
    data: AgentUpdate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.update(db, agent_id, data)


@router.delete("/{agent_id}", status_code=204)
async def deactivate_agent(
    agent_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await service.delete(db, agent_id)


@router.get("/{agent_id}/context", response_model=AgentContextResponse)
async def get_agent_context(
    agent_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.build_context(db, agent_id)


@router.get("/{agent_id}/sessions", response_model=CursorPage[SessionResponse])
async def get_agent_sessions(
    agent_id: uuid.UUID,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from datetime import datetime
    query = select(Session).where(Session.agent_id == agent_id).order_by(Session.created_at.desc()).limit(limit + 1)
    if cursor:
        cursor_dt = datetime.fromisoformat(cursor)
        query = query.where(Session.created_at < cursor_dt)
    result = await db.execute(query)
    items = list(result.scalars().all())
    next_cursor = None
    if len(items) > limit:
        items = items[:limit]
        next_cursor = items[-1].created_at.isoformat()
    return CursorPage(items=items, next_cursor=next_cursor)
