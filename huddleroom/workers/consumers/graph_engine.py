import asyncio
import logging

from huddleroom.database import AsyncSessionLocal
from huddleroom.services.event_bus import get_event_bus
from huddleroom.services.graph_engine import GraphEngineService

logger = logging.getLogger(__name__)


async def run_graph_engine() -> None:
    bus = get_event_bus()
    engine = GraphEngineService(_bus=bus)
    logger.info("graph_engine consumer started")
    async for event in bus.subscribe():
        try:
            async with AsyncSessionLocal() as db:
                async with db.begin():
                    await engine.process_event(db, event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("graph_engine error processing %s: %s", event.event_type, exc, exc_info=True)
