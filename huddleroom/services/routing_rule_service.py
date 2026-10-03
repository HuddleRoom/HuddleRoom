from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.routing_rule import RoutingRule


class RoutingRuleService:
    async def apply_update(self, db: AsyncSession, rule: RoutingRule, update_data: dict) -> RoutingRule:
        _NON_NULLABLE = {"conditions", "actions", "priority", "enabled", "on_event", "name"}
        for field, value in update_data.items():
            if value is None and field in _NON_NULLABLE:
                raise HTTPException(status_code=422, detail=f"Field '{field}' cannot be null")
            setattr(rule, field, value)
        await db.flush()
        return rule
