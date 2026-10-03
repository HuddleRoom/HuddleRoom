from __future__ import annotations

import uuid
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException, status

from huddleroom.models.user import User
from huddleroom.models.api_key import ApiKey
from huddleroom.security import hash_password, verify_password, generate_api_key


class AuthService:
    async def create_user(
        self,
        db: AsyncSession,
        email: str,
        password: str,
        display_name: str | None = None,
        role: str = "member",
    ) -> User:
        # Check email not taken
        result = await db.execute(select(User).where(User.email == email))
        if result.scalar_one_or_none():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
        user = User(
            email=email,
            hashed_password=hash_password(password),
            display_name=display_name,
            role=role,
        )
        db.add(user)
        await db.flush()
        return user

    async def authenticate(self, db: AsyncSession, email: str, password: str) -> User | None:
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        if not user or not user.is_active:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    async def create_api_key(
        self,
        db: AsyncSession,
        label: str | None,
        user_id: uuid.UUID | None = None,
        agent_id: uuid.UUID | None = None,
        project_id: uuid.UUID | None = None,
        expires_at: datetime | None = None,
    ) -> tuple[ApiKey, str]:
        full_key, key_prefix, hashed_key = generate_api_key()
        api_key = ApiKey(
            user_id=user_id,
            agent_id=agent_id,
            project_id=project_id,
            key_prefix=key_prefix,
            hashed_key=hashed_key,
            label=label,
            expires_at=expires_at,
        )
        db.add(api_key)
        await db.flush()
        return api_key, full_key

    async def list_api_keys(self, db: AsyncSession, user_id: uuid.UUID) -> list[ApiKey]:
        result = await db.execute(select(ApiKey).where(ApiKey.user_id == user_id))
        return list(result.scalars().all())

    async def delete_api_key(self, db: AsyncSession, key_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        result = await db.execute(
            select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == user_id)
        )
        key = result.scalar_one_or_none()
        if not key:
            return False
        await db.delete(key)
        return True
