import uuid
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user
from huddleroom.models.user import User
from huddleroom.schemas.knowledge import (
    KnowledgeCreate, KnowledgeResponse, KnowledgeUpdate,
    KnowledgeSearchRequest,
)
from huddleroom.schemas.common import CursorPage
from huddleroom.services.knowledge_service import KnowledgeService

# MVP: no project-level authorization — any authenticated user/agent can access any project's knowledge

router = APIRouter()
service = KnowledgeService()


@router.get("/projects/{project_id}/knowledge", response_model=CursorPage[KnowledgeResponse])
async def list_knowledge(
    project_id: uuid.UUID,
    content_type: str | None = None,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    items, next_cursor = await service.list(db, project_id, content_type=content_type, cursor=cursor, limit=limit)
    return CursorPage(items=items, next_cursor=next_cursor)


@router.post("/projects/{project_id}/knowledge", response_model=KnowledgeResponse, status_code=201)
async def create_knowledge(
    project_id: uuid.UUID,
    data: KnowledgeCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.create(db, project_id, data, created_by_user=current_user.id)


@router.post("/projects/{project_id}/knowledge/search")
async def search_knowledge(
    project_id: uuid.UUID,
    data: KnowledgeSearchRequest,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    results = await service.search(
        db, project_id, data.query, limit=data.limit,
        content_type=data.content_type, min_relevance_score=data.min_relevance_score
    )
    return {"results": results}


@router.get("/knowledge/{item_id}", response_model=KnowledgeResponse)
async def get_knowledge(
    item_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.get_or_404(db, item_id)


@router.put("/knowledge/{item_id}", response_model=KnowledgeResponse)
async def update_knowledge(
    item_id: uuid.UUID,
    data: KnowledgeUpdate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.update(db, item_id, data)


@router.delete("/knowledge/{item_id}", status_code=204)
async def delete_knowledge(
    item_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await service.delete(db, item_id)
