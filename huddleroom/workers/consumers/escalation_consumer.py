from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.services.graph_engine import GraphEngineService


class EscalationManager:
    async def process_expired_timeouts(self, db: AsyncSession) -> int:
        return await GraphEngineService().process_timeouts(db)
