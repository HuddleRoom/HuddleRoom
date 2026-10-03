from typing import Protocol
import uuid
from sqlalchemy.ext.asyncio import AsyncSession


class AdapterProtocol(Protocol):
    async def run(self, session_id: uuid.UUID, db: AsyncSession) -> None:
        ...
