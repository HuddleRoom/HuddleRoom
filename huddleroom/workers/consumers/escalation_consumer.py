from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.services.protocol_engine import ProtocolEngineService


class EscalationManager:
    async def process_expired_timeouts(self, db: AsyncSession) -> int:
        return await ProtocolEngineService().process_timeouts(db)
