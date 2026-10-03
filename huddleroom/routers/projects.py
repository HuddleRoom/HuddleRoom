import uuid
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user
from huddleroom.models.user import User
from huddleroom.schemas.project import (
    ProjectCreate,
    ProjectResetRequest,
    ProjectResetResponse,
    ProjectResponse,
    ProjectUpdate,
)
from huddleroom.schemas.common import CursorPage
from huddleroom.services.project_service import ProjectService
from huddleroom.services.project_reset_service import ProjectResetService

router = APIRouter()
service = ProjectService()
reset_service = ProjectResetService()


@router.get("", response_model=CursorPage[ProjectResponse])
async def list_projects(
    cursor: str | None = None,
    limit: int = 50,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    items, next_cursor = await service.list(db, cursor, limit)
    return CursorPage(items=items, next_cursor=next_cursor)


@router.post("", response_model=ProjectResponse, status_code=201)
async def create_project(
    data: ProjectCreate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.create(db, data)


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.get_or_404(db, project_id)


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: uuid.UUID,
    data: ProjectUpdate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.update(db, project_id, data)


@router.delete("/{project_id}", response_model=ProjectResponse)
async def archive_project(
    project_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.archive(db, project_id)


@router.post("/{project_id}/reset", response_model=ProjectResetResponse)
async def reset_project(
    project_id: uuid.UUID,
    data: ProjectResetRequest,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await reset_service.reset(db, project_id, data.confirm_name)
