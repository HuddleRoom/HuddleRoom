from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.optimization import Optimization, OPTIMIZATION_VALID_TRANSITIONS


class OptimizationService:
    async def apply_update(self, db: AsyncSession, opt: Optimization, update_data: dict) -> Optimization:
        if "status" in update_data:
            new_status = update_data["status"]
            allowed = OPTIMIZATION_VALID_TRANSITIONS.get(opt.status, set())
            if new_status not in allowed:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Cannot transition from '{opt.status}' to '{new_status}'. Allowed: {sorted(allowed)}",
                )
        if "type" in update_data and update_data["type"] not in {"hook", "rule", "shortcut"}:
            raise HTTPException(status_code=422, detail="type must be one of: hook, rule, shortcut")
        if "generated_code" in update_data and update_data["generated_code"] is None:
            raise HTTPException(status_code=422, detail="Field 'generated_code' cannot be null")
        for field, value in update_data.items():
            setattr(opt, field, value)
        await db.flush()
        return opt
