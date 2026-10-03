import asyncio
import logging

from huddleroom.database import AsyncSessionLocal
from huddleroom.services.event_bus import get_event_bus
from huddleroom.services.protocol_engine import ProtocolEngineService

logger = logging.getLogger(__name__)


async def run_protocol_engine() -> None:
    bus = get_event_bus()
    engine = ProtocolEngineService(_bus=bus)
    logger.info("protocol_engine consumer started")
    async for event in bus.subscribe():
        try:
            async with AsyncSessionLocal() as db:
                async with db.begin():
                    await engine.process_event(db, event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("protocol_engine error processing %s: %s", event.event_type, exc, exc_info=True)
