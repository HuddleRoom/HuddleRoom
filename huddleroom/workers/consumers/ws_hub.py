from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass

from huddleroom.services.event_bus import BusEvent, get_event_bus

logger = logging.getLogger(__name__)


@dataclass
class _QueuedMessage:
    text: str
    evictable: bool


class ConnectionRegistry:
    def __init__(self) -> None:
        self._conns: dict[str, dict[str, tuple]] = {}

    def add(self, project_id: uuid.UUID, conn_id: str, websocket, event_type_filter: set[str] | None) -> None:
        pid = str(project_id)
        queue: asyncio.Queue[_QueuedMessage] = asyncio.Queue(maxsize=1000)
        sender = asyncio.create_task(self._send(project_id, conn_id, websocket, queue))
        self._conns.setdefault(pid, {})[conn_id] = (websocket, event_type_filter, queue, sender)

    async def remove(self, project_id: uuid.UUID, conn_id: str) -> None:
        pid = str(project_id)
        if pid in self._conns:
            connection = self._conns[pid].pop(conn_id, None)
            if connection:
                sender = connection[3]
                if sender is not asyncio.current_task():
                    sender.cancel()
                    await asyncio.gather(sender, return_exceptions=True)

    async def _send(
        self, project_id: uuid.UUID, conn_id: str, websocket, queue: asyncio.Queue[_QueuedMessage]
    ) -> None:
        while True:
            message = await queue.get()
            try:
                await websocket.send_text(message.text)
            except Exception:
                logger.warning("WebSocket sender failed for %s", conn_id, exc_info=True)
                try:
                    await websocket.close()
                except Exception:
                    pass
                await self.remove(project_id, conn_id)
                return
            finally:
                queue.task_done()

    async def enqueue(self, project_id: uuid.UUID, conn_id: str, text: str, *, evictable: bool) -> bool:
        connection = self._conns.get(str(project_id), {}).get(conn_id)
        if connection is None:
            return False
        websocket, _, queue, _ = connection
        if queue.full():
            kept = []
            removed = False
            while not queue.empty():
                message = queue.get_nowait()
                queue.task_done()
                if message.evictable and not removed:
                    removed = True
                else:
                    kept.append(message)
            if not removed:
                try:
                    await websocket.close(code=1013)
                except Exception:
                    logger.warning("WebSocket overload close failed for %s", conn_id, exc_info=True)
                finally:
                    await self.remove(project_id, conn_id)
                return False
            for message in kept:
                queue.put_nowait(message)
        queue.put_nowait(_QueuedMessage(text, evictable))
        return True

    async def broadcast(self, event: BusEvent) -> None:
        pid = str(event.project_id)
        conns = dict(self._conns.get(pid, {}))
        payload = json.dumps({
            "id": str(event.id),
            "project_id": pid,
            "event_type": event.event_type,
            "payload": event.payload,
            "source": event.source,
            "emitted_at": event.emitted_at.isoformat(),
        })
        dead = []
        for conn_id, (_, type_filter, _, _) in conns.items():
            if type_filter and event.event_type not in type_filter:
                continue
            try:
                await self.enqueue(event.project_id, conn_id, payload, evictable=False)
            except Exception:
                dead.append(conn_id)
        for conn_id in dead:
            await self.remove(event.project_id, conn_id)
        await asyncio.sleep(0)

    async def broadcast_response(self, event) -> None:
        payload = event.model_dump_json()
        for conn_id, (_, type_filter, _, _) in tuple(self._conns.get(str(event.project_id), {}).items()):
            if type_filter and event.event_type not in type_filter:
                continue
            try:
                await self.enqueue(event.project_id, conn_id, payload, evictable=event.event_type == "agent_response.output")
            except Exception:
                logger.warning("WebSocket response fanout failed for %s", conn_id, exc_info=True)

    def connection_count(self) -> int:
        return sum(len(v) for v in self._conns.values())


# Module-level singleton
_registry = ConnectionRegistry()


def get_registry() -> ConnectionRegistry:
    return _registry


async def run_ws_hub() -> None:
    """Consumer task: subscribe to all events and fan out to registered WS connections."""
    bus = get_event_bus()
    logger.info("ws_hub consumer started")
    async for event in bus.subscribe(project_id=None):
        try:
            await _registry.broadcast(event)
        except Exception as exc:
            logger.warning("ws_hub broadcast error: %s", exc)
