import asyncio
import logging
from huddleroom.services.event_bus import get_event_bus

logger = logging.getLogger(__name__)

async def run_optimizer() -> None:
    bus = get_event_bus()
    logger.info("optimizer consumer started")
    async for event in bus.subscribe():
        try:
            # stub: optimizer processes events here
            logger.debug("optimizer received event %s (stub, no-op)", event.event_type)
        except Exception as exc:
            logger.error("optimizer error processing %s: %s", event.event_type, exc)
