from __future__ import annotations
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.config import settings
from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user
from huddleroom.models.session import Session
from huddleroom.models.event_log import EventLog
from huddleroom.models.user import User

# pylint: disable=not-callable

router = APIRouter()


class ConsumerStatus(BaseModel):
    running: bool
    active_connections: int | None = None


class OrchestrationHealthResponse(BaseModel):
    consumers: dict[str, ConsumerStatus]
    active_sessions: int
    event_bus_mode: str
    event_log_total: int
    debug_enabled: bool


@router.get("/health", response_model=OrchestrationHealthResponse)
async def orchestration_health(
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from huddleroom.workers.consumers import get_consumer_tasks
    from huddleroom.workers.consumers.ws_hub import get_registry

    tasks = get_consumer_tasks()
    all_names = ("ws_hub", "rule_engine", "protocol_engine", "meeting_engine", "optimizer")
    consumers: dict[str, ConsumerStatus] = {}
    for name in all_names:
        task = tasks.get(name)
        running = task is not None and not task.done()
        extra = {}
        if name == "ws_hub":
            extra["active_connections"] = get_registry().connection_count()
        consumers[name] = ConsumerStatus(running=running, **extra)

    active_sessions_result = await db.execute(
        select(func.count(Session.id)).where(Session.status.in_(["pending", "running"]))
    )
    active_sessions = active_sessions_result.scalar() or 0

    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=7)
    event_log_total_result = await db.execute(
        select(func.count(EventLog.id)).where(EventLog.emitted_at >= cutoff)  # last 7 days
    )
    event_log_total = event_log_total_result.scalar() or 0

    return OrchestrationHealthResponse(
        consumers=consumers,
        active_sessions=active_sessions,
        event_bus_mode="in_process" if settings.is_sqlite else "redis_streams",
        event_log_total=event_log_total,
        debug_enabled=settings.debug,
    )
