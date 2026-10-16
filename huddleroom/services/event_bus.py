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
        *,
        ready: asyncio.Event | None = None,
    ) -> AsyncGenerator[BusEvent, None]:
        """Async generator — yields events matching the filter until cancelled."""
        queue: asyncio.Queue[BusEvent] = asyncio.Queue(maxsize=maxsize)
        type_set = set(event_types) if event_types else None
        entry = (queue, project_id, type_set)
        self._subscribers.append(entry)
        if ready is not None:
            ready.set()
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


def _bus_after_commit(session) -> None:
    root = session.get_transaction()
    scope = session.get_nested_transaction() or root
    pending = session.info.get("pending_bus_events", [])
    if scope is not None and scope.nested:
        for entry in pending:
            if entry["scope"] is scope:
                entry["scope"] = scope.parent
        return
    publish = [entry for entry in pending if entry["root"] is root]
    retained = [entry for entry in pending if entry["root"] is not root]
    if retained:
        session.info["pending_bus_events"] = retained
    else:
        session.info.pop("pending_bus_events", None)
    for entry in publish:
        ev, bus = entry["event"], entry["bus"]
        try:
            from huddleroom.services.orchestration_supervision_scheduler import SUPPORTED_EVENTS
            is_sqlite = session.bind is not None and session.bind.dialect.name == "sqlite"
            if is_sqlite:
                bus.put(ev)
            elif ev.event_type in SUPPORTED_EVENTS:
                from huddleroom.workers.orchestration_tasks import request_supervision_wakeup
                request_supervision_wakeup(ev.id)
        except Exception:
            logger.exception("Unable to request orchestration supervision wakeup for %s", ev.id)


def _bus_transaction_ended(session, transaction) -> None:
    if transaction.parent is not None and not transaction.nested:
        return  # Internal flush scopes do not own a publication queue.
    pending = session.info.get("pending_bus_events", [])
    def keep(entry):
        if transaction.parent is None:
            return entry["root"] is not transaction
        scope = entry["scope"]
        while scope is not None:
            if scope is transaction:
                return False
            scope = scope.parent
        return True
    retained = [entry for entry in pending if keep(entry)]
    if retained:
        session.info["pending_bus_events"] = retained
    else:
        session.info.pop("pending_bus_events", None)


def _defer_bus_put(db: AsyncSession, bus_event: BusEvent, bus: EventBusService) -> None:
    sync_session = db.sync_session
    root = sync_session.get_transaction()
    scope = sync_session.get_nested_transaction() or root
    sync_session.info.setdefault("pending_bus_events", []).append(
        {"event": bus_event, "bus": bus, "root": root, "scope": scope},
    )
    if not sync_session.info.get("bus_publication_hooks_installed"):
        from sqlalchemy import event as _sa_event
        _sa_event.listen(sync_session, "after_commit", _bus_after_commit)
        _sa_event.listen(sync_session, "after_transaction_end", _bus_transaction_ended)
        sync_session.info["bus_publication_hooks_installed"] = True


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
