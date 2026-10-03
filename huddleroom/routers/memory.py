import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user
from huddleroom.models.user import User
from huddleroom.schemas.memory import MemoryResponse, MemorySearchRequest, MemorySearchResult
from huddleroom.schemas.common import CursorPage
from huddleroom.services.memory_service import MemoryService

router = APIRouter()
service = MemoryService()


@router.get("/projects/{project_id}/memory", response_model=CursorPage[MemoryResponse])
async def list_project_memories(
    project_id: uuid.UUID,
    agent_id: uuid.UUID | None = None,
    shared: bool | None = None,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    items, next_cursor = await service.list_memories(
        db, project_id=project_id, agent_id_filter=agent_id,
        shared_filter=shared, cursor=cursor, limit=limit,
    )
    return CursorPage(items=items, next_cursor=next_cursor)


@router.get("/projects/{project_id}/memory/{item_id}", response_model=MemoryResponse)
async def get_project_memory(
    project_id: uuid.UUID,
    item_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    item = await service.read(db, agent_id=None, project_id=project_id, item_id=item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    if item.scope == "project" and item.project_id != project_id:
        raise HTTPException(status_code=404, detail="Memory not found")
    if item.scope == "global" and not item.shared:
        raise HTTPException(status_code=404, detail="Memory not found")
    return item


@router.post("/projects/{project_id}/memory/search")
async def search_project_memories(
    project_id: uuid.UUID,
    data: MemorySearchRequest,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    results = await service.search(
        db, agent_id=None, project_id=project_id,
        query=data.query, limit=data.limit, tags=data.tags,
    )
    return {"results": results, "count": len(results)}


@router.delete("/projects/{project_id}/memory/{item_id}", status_code=204)
async def delete_project_memory(
    project_id: uuid.UUID,
    item_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    item = await service.read(db, agent_id=None, project_id=project_id, item_id=item_id)
    if item is None or item.scope != "project" or item.project_id != project_id:
        raise HTTPException(status_code=404, detail="Memory not found")
    await service.delete_any(db, item_id)


@router.get("/memory", response_model=CursorPage[MemoryResponse])
async def list_global_memories(
    scope: str = Query(default="global"),
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    items, next_cursor = await service.list_memories(
        db, project_id=None, scope="global", cursor=cursor, limit=limit,
    )
    return CursorPage(items=items, next_cursor=next_cursor)


@router.delete("/memory/{item_id}", status_code=204)
async def delete_global_memory(
    item_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    item = await service.read(db, agent_id=None, project_id=None, item_id=item_id)
    if item is None or item.scope != "global":
        raise HTTPException(status_code=404, detail="Memory not found")
    await service.delete_any(db, item_id)
