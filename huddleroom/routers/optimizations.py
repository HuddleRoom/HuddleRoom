from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user, ensure_project_exists
from huddleroom.models.optimization import CostMetric, Optimization, OPTIMIZATION_VALID_TRANSITIONS, Pattern
from huddleroom.models.user import User
from huddleroom.schemas.common import CursorPage
from huddleroom.schemas.optimization import (
    CostMetricResponse,
    OptimizationCreate,
    OptimizationResponse,
    OptimizationUpdate,
    PatternResponse,
)
from huddleroom.services.optimization_service import OptimizationService

router = APIRouter()


@router.get("/projects/{project_id}/patterns", response_model=CursorPage[PatternResponse])
async def list_patterns(
    project_id: uuid.UUID,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    conditions = [Pattern.project_id == project_id]
    if cursor is not None:
        parts = cursor.split("__", 1)
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid cursor")
        try:
            cursor_ts_us = int(parts[0])
            cursor_id = uuid.UUID(parts[1])
            cursor_dt = datetime.fromtimestamp(cursor_ts_us / 1_000_000.0, tz=timezone.utc)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="Invalid cursor")
        conditions.append(
            or_(
                Pattern.created_at < cursor_dt,
                and_(Pattern.created_at == cursor_dt, Pattern.id < cursor_id),
            )
        )
    result = await db.execute(
        select(Pattern)
        .where(*conditions)
        .order_by(Pattern.created_at.desc(), Pattern.id.desc())
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


@router.get("/projects/{project_id}/patterns/{pattern_id}", response_model=PatternResponse)
async def get_pattern(
    project_id: uuid.UUID,
    pattern_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Pattern).where(Pattern.id == pattern_id, Pattern.project_id == project_id)
    )
    pattern = result.scalar_one_or_none()
    if pattern is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pattern not found")
    return pattern


async def _get_optimization_or_404(
    db: AsyncSession, project_id: uuid.UUID, opt_id: uuid.UUID
) -> Optimization:
    result = await db.execute(
        select(Optimization).where(Optimization.id == opt_id, Optimization.project_id == project_id)
    )
    opt = result.scalar_one_or_none()
    if opt is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Optimization not found")
    return opt


@router.get("/projects/{project_id}/optimizations", response_model=CursorPage[OptimizationResponse])
async def list_optimizations(
    project_id: uuid.UUID,
    status_filter: str | None = Query(default=None, alias="status"),
    type_filter: str | None = Query(default=None, alias="type"),
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    conditions = [Optimization.project_id == project_id]
    if status_filter is not None:
        conditions.append(Optimization.status == status_filter)
    if type_filter is not None:
        conditions.append(Optimization.type == type_filter)
    if cursor is not None:
        parts = cursor.split("__", 1)
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid cursor")
        try:
            cursor_ts_us = int(parts[0])
            cursor_id = uuid.UUID(parts[1])
            cursor_dt = datetime.fromtimestamp(cursor_ts_us / 1_000_000.0, tz=timezone.utc)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="Invalid cursor")
        conditions.append(
            or_(
                Optimization.created_at < cursor_dt,
                and_(Optimization.created_at == cursor_dt, Optimization.id < cursor_id),
            )
        )
    result = await db.execute(
        select(Optimization)
        .where(*conditions)
        .order_by(Optimization.created_at.desc(), Optimization.id.desc())
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


@router.post("/projects/{project_id}/optimizations", response_model=OptimizationResponse, status_code=201)
async def create_optimization(
    project_id: uuid.UUID,
    data: OptimizationCreate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    if data.type not in {"hook", "rule", "shortcut"}:
        raise HTTPException(status_code=422, detail="type must be one of: hook, rule, shortcut")
    if data.pattern_id is not None:
        result = await db.execute(
            select(Pattern).where(Pattern.id == data.pattern_id, Pattern.project_id == project_id)
        )
        if result.scalar_one_or_none() is None:
            raise HTTPException(status_code=404, detail="Pattern not found in this project")
    opt = Optimization(
        project_id=project_id,
        pattern_id=data.pattern_id,
        type=data.type,
        generated_code=data.generated_code,
        status="proposed",
    )
    db.add(opt)
    await db.flush()
    return opt


@router.get("/projects/{project_id}/optimizations/count")
async def count_optimizations(
    project_id: uuid.UUID,
    status_filter: str | None = Query(default=None, alias="status"),
    type_filter: str | None = Query(default=None, alias="type"),
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    conditions = [Optimization.project_id == project_id]
    if status_filter is not None:
        conditions.append(Optimization.status == status_filter)
    if type_filter is not None:
        conditions.append(Optimization.type == type_filter)
    result = await db.execute(select(func.count(Optimization.id)).where(*conditions))  # pylint: disable=not-callable
    return {"count": result.scalar_one()}


@router.get("/projects/{project_id}/optimizations/{opt_id}", response_model=OptimizationResponse)
async def get_optimization(
    project_id: uuid.UUID,
    opt_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _get_optimization_or_404(db, project_id, opt_id)


@router.patch("/projects/{project_id}/optimizations/{opt_id}", response_model=OptimizationResponse)
async def update_optimization(
    project_id: uuid.UUID,
    opt_id: uuid.UUID,
    data: OptimizationUpdate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    opt = await _get_optimization_or_404(db, project_id, opt_id)
    update_data = data.model_dump(exclude_unset=True)
    return await OptimizationService().apply_update(db, opt, update_data)


@router.delete("/projects/{project_id}/optimizations/{opt_id}", status_code=204)
async def delete_optimization(
    project_id: uuid.UUID,
    opt_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    opt = await _get_optimization_or_404(db, project_id, opt_id)
    await db.delete(opt)
    await db.flush()
    return None


@router.get("/projects/{project_id}/cost-metrics", response_model=list[CostMetricResponse])
async def list_cost_metrics(
    project_id: uuid.UUID,
    start_date: date | None = None,
    end_date: date | None = None,
    optimization_id: uuid.UUID | None = None,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    if start_date is None:
        start_date = date.today() - timedelta(days=30)
    if end_date is None:
        end_date = date.today()
    if (end_date - start_date).days > 365:
        raise HTTPException(status_code=400, detail="Date range must not exceed 365 days")
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must not be after end_date")
    conditions = [
        CostMetric.project_id == project_id,
        CostMetric.date >= start_date,
        CostMetric.date <= end_date,
    ]
    if optimization_id is not None:
        conditions.append(CostMetric.optimization_id == optimization_id)
    result = await db.execute(
        select(CostMetric).where(*conditions).order_by(CostMetric.date.asc())
    )
    return list(result.scalars().all())


