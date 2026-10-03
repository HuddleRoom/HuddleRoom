from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncGenerator

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.event_log import EventLog

logger = logging.getLogger(__name__)


@dataclass
class BusEvent:
    id: uuid.UUID
    project_id: uuid.UUID
    event_type: str
    payload: dict
    source: str
    emitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EventBusService:
    def __init__(self) -> None:
        # Safe in asyncio single-thread model; put() copies list before iterating.
        # List of (queue, project_id_filter or None, event_type_filter set or None)
        self._subscribers: list[tuple[asyncio.Queue, uuid.UUID | None, set[str] | None]] = []

    def put(self, event: BusEvent) -> None:
        """Fan out event to all matching subscriber queues (non-blocking)."""
        for queue, pid_filter, type_filter in list(self._subscribers):
            if pid_filter is not None and pid_filter != event.project_id:
                continue
            if type_filter is not None and event.event_type not in type_filter:
                continue
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Event bus queue full for subscriber, dropping event %s", event.event_type)

    async def subscribe(
        self,
        project_id: uuid.UUID | None = None,
        event_types: list[str] | None = None,
        maxsize: int = 1000,
    ) -> AsyncGenerator[BusEvent, None]:
        """Async generator — yields events matching the filter until cancelled."""
        queue: asyncio.Queue[BusEvent] = asyncio.Queue(maxsize=maxsize)
        type_set = set(event_types) if event_types else None
        entry = (queue, project_id, type_set)
        self._subscribers.append(entry)
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            try:
                self._subscribers.remove(entry)
            except ValueError:
                pass


_event_bus: EventBusService | None = None  # pylint: disable=invalid-name


def get_event_bus() -> EventBusService:
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBusService()
    return _event_bus


def _to_bus_event(log_entry: EventLog) -> BusEvent:
    return BusEvent(
        id=log_entry.id,
        project_id=log_entry.project_id,
        event_type=log_entry.event_type,
        payload=log_entry.payload,
        source=log_entry.source,
        emitted_at=log_entry.emitted_at,
    )


def _defer_bus_put(db: AsyncSession, bus_event: BusEvent, bus: EventBusService) -> None:
    # Defer put to after commit so consumers never see events for rolled-back txns.
    sync_session = db.sync_session
    if "pending_bus_events" not in sync_session.info:
        sync_session.info["pending_bus_events"] = []

        from sqlalchemy import event as _sa_event

        @_sa_event.listens_for(sync_session, "after_commit", once=True)
        def _dispatch_pending(session):
            for ev, b in session.info.pop("pending_bus_events", []):
                try:
                    from huddleroom.services.orchestration_supervision_scheduler import SUPPORTED_EVENTS
                    is_sqlite = session.bind is not None and session.bind.dialect.name == "sqlite"
                    # SQLite has one process and uses the in-memory bus. Postgres
                    # must cross the process boundary only through Celery.
                    if is_sqlite:
                        b.put(ev)
                    elif ev.event_type in SUPPORTED_EVENTS:
                        from huddleroom.workers.orchestration_tasks import request_supervision_wakeup  # pylint: disable=cyclic-import
                        request_supervision_wakeup(ev.id)
                except Exception:
                    logger.exception("Unable to request orchestration supervision wakeup for %s", ev.id)

    sync_session.info["pending_bus_events"].append((bus_event, bus))


async def emit_event_once(
    db: AsyncSession,
    project_id: uuid.UUID,
    event_type: str,
    payload: dict,
    *,
    source: str = "system",
    dedup_key: str | None = None,
    _bus: EventBusService | None = None,
) -> tuple[BusEvent, bool]:
    """Persist one event, returning the existing row when a dedup key already exists."""
    bus = _bus or get_event_bus()
    if dedup_key is None:
        log_entry = EventLog(
            project_id=project_id,
            event_type=event_type,
            dedup_key=None,
            payload=payload,
            source=source,
        )
        db.add(log_entry)
        await db.flush()
        bus_event = _to_bus_event(log_entry)
        _defer_bus_put(db, bus_event, bus)
        return bus_event, True

    nested = await db.begin_nested()
    try:
        log_entry = EventLog(
            project_id=project_id,
            event_type=event_type,
            dedup_key=dedup_key,
            payload=payload,
            source=source,
        )
        db.add(log_entry)
        await db.flush()
    except IntegrityError:
        await nested.rollback()
        existing = (
            await db.execute(
                select(EventLog).where(
                    EventLog.project_id == project_id,
                    EventLog.dedup_key == dedup_key,
                )
            )
        ).scalar_one()
        return _to_bus_event(existing), False

    await nested.commit()
    bus_event = _to_bus_event(log_entry)
    _defer_bus_put(db, bus_event, bus)
    return bus_event, True


async def emit_event(
    db: AsyncSession,
    project_id: uuid.UUID,
    event_type: str,
    payload: dict,
    source: str = "system",
    _bus: EventBusService | None = None,
) -> BusEvent:
    """Write EventLog row in caller's transaction and defer bus.put to after commit.

    _bus parameter is for testing only — omit in production to use singleton.
    """
    bus_event, _ = await emit_event_once(
        db,
        project_id,
        event_type,
        payload,
        source=source,
        _bus=_bus,
    )
    return bus_event
