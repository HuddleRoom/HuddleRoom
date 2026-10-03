from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user, ensure_project_exists
from huddleroom.models.hook import Hook, HOOK_VALID_TRANSITIONS
from huddleroom.models.user import User
from huddleroom.schemas.common import CursorPage
from huddleroom.schemas.hook import HookCreate, HookResponse, HookUpdate
from huddleroom.services.hook_service import HookService

router = APIRouter()


async def _get_hook_or_404(db: AsyncSession, project_id: uuid.UUID, hook_id: uuid.UUID) -> Hook:
    result = await db.execute(
        select(Hook).where(Hook.id == hook_id, Hook.project_id == project_id)
    )
    hook = result.scalar_one_or_none()
    if hook is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Hook not found")
    return hook


@router.get("/projects/{project_id}/hooks", response_model=CursorPage[HookResponse])
async def list_hooks(
    project_id: uuid.UUID,
    status_filter: str | None = Query(default=None, alias="status"),
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    conditions = [Hook.project_id == project_id]
    if status_filter is not None:
        conditions.append(Hook.status == status_filter)
    if cursor is not None:
        parts = cursor.split("__", 1)
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid cursor")
        try:
            cursor_ts_us = int(parts[0])
            cursor_id = uuid.UUID(parts[1])
            cursor_dt = datetime.fromtimestamp(cursor_ts_us / 1_000_000.0, tz=timezone.utc)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail="Invalid cursor") from exc
        conditions.append(
            or_(
                Hook.created_at < cursor_dt,
                and_(Hook.created_at == cursor_dt, Hook.id < cursor_id),
            )
        )
    result = await db.execute(
        select(Hook)
        .where(*conditions)
        .order_by(Hook.created_at.desc(), Hook.id.desc())
        .limit(limit + 1)
    )
    items = list(result.scalars().all())
    next_cursor = None
    if len(items) > limit:
        items = items[:limit]
        last = items[-1]
        # Handle both timezone-aware and naive datetimes
        dt = last.created_at
        if dt.tzinfo is None:
            # Naive datetime - assume UTC
            dt = dt.replace(tzinfo=timezone.utc)
        cursor_ts_us = int(dt.timestamp() * 1_000_000)
        next_cursor = f"{cursor_ts_us}__{last.id}"
    return CursorPage(items=items, next_cursor=next_cursor)


@router.post("/projects/{project_id}/hooks", response_model=HookResponse, status_code=201)
async def create_hook(
    project_id: uuid.UUID,
    data: HookCreate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    hook = Hook(
        project_id=project_id,
        name=data.name,
        description=data.description,
        code=data.code,
        status="proposed",
        trigger_event=data.trigger_event,
    )
    db.add(hook)
    await db.flush()
    return hook


@router.get("/projects/{project_id}/hooks/{hook_id}", response_model=HookResponse)
async def get_hook(
    project_id: uuid.UUID,
    hook_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _get_hook_or_404(db, project_id, hook_id)


@router.patch("/projects/{project_id}/hooks/{hook_id}", response_model=HookResponse)
async def update_hook(
    project_id: uuid.UUID,
    hook_id: uuid.UUID,
    data: HookUpdate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    hook = await _get_hook_or_404(db, project_id, hook_id)
    update_data = data.model_dump(exclude_unset=True)
    return await HookService().apply_update(db, hook, update_data)


@router.delete("/projects/{project_id}/hooks/{hook_id}", status_code=204)
async def delete_hook(
    project_id: uuid.UUID,
    hook_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    hook = await _get_hook_or_404(db, project_id, hook_id)
    await db.delete(hook)
    await db.flush()
    return None
