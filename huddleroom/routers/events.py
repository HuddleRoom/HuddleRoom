from __future__ import annotations

import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user
from huddleroom.models.user import User
from huddleroom.models.event_log import EventLog
from huddleroom.schemas.event import EventEmit, EventResponse
from huddleroom.schemas.common import CursorPage
from huddleroom.services.event_bus import emit_event

router = APIRouter()


@router.post("", response_model=EventResponse, status_code=201)
async def post_event(
    data: EventEmit,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    bus_event = await emit_event(
        db=db,
        project_id=data.project_id,
        event_type=data.event_type,
        payload=data.payload,
        source=data.source,
    )
    return EventResponse(
        id=bus_event.id,
        project_id=bus_event.project_id,
        event_type=bus_event.event_type,
        payload=bus_event.payload,
        source=bus_event.source,
        emitted_at=bus_event.emitted_at,
    )


@router.get("", response_model=CursorPage[EventResponse])
async def list_events(
    project_id: uuid.UUID | None = Query(default=None),
    event_type: str | None = Query(default=None),
    since: datetime | None = Query(default=None),
    cursor: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    query = select(EventLog).order_by(EventLog.emitted_at.desc(), EventLog.id.desc()).limit(limit + 1)
    if project_id:
        query = query.where(EventLog.project_id == project_id)
    if event_type:
        query = query.where(EventLog.event_type == event_type)
    if since:
        since_naive = since.replace(tzinfo=None) if since.tzinfo is not None else since
        query = query.where(EventLog.emitted_at >= since_naive)
    if cursor:
        try:
            cursor_dt_str, cursor_id_str = cursor.split("__", 1)
            cursor_dt = datetime.fromisoformat(cursor_dt_str).replace(tzinfo=None)
            cursor_id = uuid.UUID(cursor_id_str)
        except (ValueError, AttributeError) as exc:
            raise HTTPException(status_code=400, detail="Invalid cursor") from exc
        query = query.where(
            or_(
                EventLog.emitted_at < cursor_dt,
                and_(EventLog.emitted_at == cursor_dt, EventLog.id < cursor_id),
            )
        )
    result = await db.execute(query)
    items = list(result.scalars().all())
    next_cursor = None
    if len(items) > limit:
        items = items[:limit]
        last = items[-1]
        next_cursor = f"{last.emitted_at.isoformat()}__{last.id}"
    return CursorPage(
        items=[EventResponse.model_validate(e) for e in items],
        next_cursor=next_cursor,
    )
