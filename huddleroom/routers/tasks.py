import uuid
from copy import deepcopy
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user, ensure_project_exists
from huddleroom.models.user import User
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.task import TaskCreate, TaskResponse, TaskUpdate, StatusPatch, TaskAssign, TaskRunRequest, TaskRunResponse
from huddleroom.schemas.session import SessionResponse
from huddleroom.schemas.common import CursorPage
from huddleroom.services.task_service import TaskService

# MVP: no project-level authorization — any authenticated user/agent can access any project's tasks

router = APIRouter()
service = TaskService()


@router.get("", response_model=CursorPage[TaskResponse])
async def list_tasks(
    project_id: uuid.UUID,
    status: str | None = None,
    assigned_to: uuid.UUID | None = None,
    parent_id: uuid.UUID | None = None,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    items, next_cursor = await service.list(db, project_id, status=status, assigned_to=assigned_to, parent_id=parent_id, cursor=cursor, limit=limit)
    return CursorPage(items=items, next_cursor=next_cursor)


@router.post("", response_model=TaskResponse, status_code=201)
async def create_task(
    project_id: uuid.UUID,
    data: TaskCreate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    return await service.create(db, project_id, data)


@router.post("/{task_id}/copy", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def copy_task(
    project_id: uuid.UUID,
    task_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    source = await service.get_or_404(db, project_id, task_id)
    return await service.create(
        db,
        project_id,
        TaskCreate(
            title=source.title,
            description=source.description,
            priority=source.priority,
            assigned_to=source.assigned_to,
            adapter_type_override=source.adapter_type_override,
            trigger=deepcopy(source.trigger),
            metadata=deepcopy(source.metadata_),
            due_at=source.due_at,
            parent_id=source.parent_id,
        ),
    )


@router.get("/count")
async def count_tasks(
    project_id: uuid.UUID,
    status: str | None = None,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    conditions = [Task.project_id == project_id]
    if status is not None:
        conditions.append(Task.status == status)
    result = await db.execute(select(func.count(Task.id)).where(*conditions))  # pylint: disable=not-callable
    return {"count": result.scalar_one()}


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    project_id: uuid.UUID,
    task_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.get_or_404(db, project_id, task_id)


@router.put("/{task_id}", response_model=TaskResponse)
async def update_task(
    project_id: uuid.UUID,
    task_id: uuid.UUID,
    data: TaskUpdate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.update(db, project_id, task_id, data)


@router.delete("/{task_id}", response_model=TaskResponse)
async def cancel_task(
    project_id: uuid.UUID,
    task_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.cancel(db, project_id, task_id)


@router.patch("/{task_id}/status", response_model=TaskResponse)
async def patch_status(
    project_id: uuid.UUID,
    task_id: uuid.UUID,
    data: StatusPatch,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await service.transition_status(db, project_id, task_id, data.status, data.reason)
    except HTTPException as exc:
        from huddleroom.services.session_service import SessionClaimAttention, SessionService
        if isinstance(exc, SessionClaimAttention):
            await SessionService.persist_claim_attention(db, exc)
            await db.commit()
        raise


@router.post("/{task_id}/assign", response_model=TaskResponse)
async def assign_task(
    project_id: uuid.UUID,
    task_id: uuid.UUID,
    data: TaskAssign,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await service.assign(db, project_id, task_id, data.agent_id)
    except HTTPException as exc:
        from huddleroom.services.session_service import SessionClaimAttention, SessionService
        if isinstance(exc, SessionClaimAttention):
            await SessionService.persist_claim_attention(db, exc)
            await db.commit()
        raise


@router.get("/{task_id}/subtasks", response_model=list[TaskResponse])
async def list_subtasks(
    project_id: uuid.UUID,
    task_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.list_subtasks(db, project_id, task_id)


async def _list_task_sessions_page(
    project_id: uuid.UUID,
    task_id: uuid.UUID,
    cursor: str | None,
    limit: int,
    db: AsyncSession,
) -> CursorPage:
    from huddleroom.services.session_service import SessionService
    items, page = await SessionService().list(db, task_id=task_id, project_id=project_id, cursor=cursor, limit=limit)
    return CursorPage(items=items, next_cursor=page)


@router.get("/{task_id}/sessions", response_model=CursorPage[SessionResponse])
async def list_task_sessions(
    project_id: uuid.UUID,
    task_id: uuid.UUID,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _list_task_sessions_page(project_id, task_id, cursor, limit, db)


@router.get("/{task_id}/runs", response_model=CursorPage[SessionResponse])
async def list_task_runs(
    project_id: uuid.UUID,
    task_id: uuid.UUID,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _list_task_sessions_page(project_id, task_id, cursor, limit, db)


@router.post("/{task_id}/run", response_model=TaskRunResponse, status_code=200)
async def run_task(
    project_id: uuid.UUID,
    task_id: uuid.UUID,
    data: TaskRunRequest,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        task, session_id = await service.run(
            db, project_id, task_id,
            adapter_type_override=data.adapter_type_override,
            context_override=data.context_override,
            model_override=data.model_override,
            timeout=data.timeout,
            max_tokens=data.max_tokens,
        )
    except HTTPException as exc:
        from huddleroom.services.session_service import SessionClaimAttention, SessionService
        if isinstance(exc, SessionClaimAttention):
            await SessionService.commit_claim_attention(db, exc)
        raise
    return TaskRunResponse(task=TaskResponse.model_validate(task), session_id=session_id)
