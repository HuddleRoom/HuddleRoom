from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.services.event_bus import BusEvent, get_event_bus

logger = logging.getLogger(__name__)

_HANDLED_EVENTS = {
    "meeting.scheduled",
    "meeting.timeout",
}


class MeetingEngineConsumer:
    async def process_event(self, db: AsyncSession, event: BusEvent) -> None:
        from huddleroom.workers.meeting_tasks import (
            dispatch_meeting_timeout,
            dispatch_start_meeting,
        )

        if event.event_type == "meeting.scheduled":
            meeting_id = event.payload.get("meeting_id")
            auto_start = event.payload.get("auto_start", False)
            if meeting_id and auto_start:
                logger.info("Dispatching start_meeting for %s", meeting_id)
                dispatch_start_meeting(meeting_id, event.project_id)

        elif event.event_type == "meeting.timeout":
            meeting_id = event.payload.get("meeting_id")
            if meeting_id:
                logger.info("Dispatching meeting_timeout for %s", meeting_id)
                dispatch_meeting_timeout(meeting_id, event.project_id)


async def run_meeting_engine() -> None:
    bus = get_event_bus()
    logger.info("meeting_engine consumer started")
    from huddleroom.database import AsyncSessionLocal
    consumer = MeetingEngineConsumer()

    async for event in bus.subscribe():
        if event.event_type not in _HANDLED_EVENTS:
            continue
        try:
            async with AsyncSessionLocal() as db:
                async with db.begin():
                    await consumer.process_event(db=db, event=event)
        except Exception as exc:
            logger.error("meeting_engine error processing %s: %s", event.event_type, exc)
