from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user
from huddleroom.models.user import User
from huddleroom.schemas.auth import UserResponse

router = APIRouter()


@router.get("/users", response_model=list[UserResponse])
async def list_users(
    is_active: bool | None = True,  # is_active: None returns all users; True (default) returns active only; False returns inactive only
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    conditions = []
    if is_active is not None:
        conditions.append(User.is_active == is_active)
    result = await db.execute(select(User).where(*conditions))
    return list(result.scalars().all())
