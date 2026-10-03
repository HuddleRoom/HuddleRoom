from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.hook import Hook, HOOK_VALID_TRANSITIONS


class HookService:
    async def apply_update(self, db: AsyncSession, hook: Hook, update_data: dict) -> Hook:
        if "status" in update_data:
            new_status = update_data["status"]
            allowed = HOOK_VALID_TRANSITIONS.get(hook.status, set())
            if new_status not in allowed:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Cannot transition from '{hook.status}' to '{new_status}'. Allowed: {sorted(allowed)}",
                )
        _NON_NULLABLE_HOOK = {"name", "trigger_event", "code"}
        for field, value in update_data.items():
            if value is None and field in _NON_NULLABLE_HOOK:
                raise HTTPException(status_code=422, detail=f"Field '{field}' cannot be null")
            setattr(hook, field, value)
        await db.flush()
        return hook
