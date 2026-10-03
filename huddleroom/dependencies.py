import uuid
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.config import settings
from huddleroom.database import get_db
from huddleroom.models.agent import Agent
from huddleroom.models.api_key import ApiKey
from huddleroom.models.project import Project
from huddleroom.models.user import User
from huddleroom.security import decode_access_token, hash_api_key

_ANON_USER = User(
    id=uuid.UUID("00000000-0000-0000-0000-000000000000"),
    email="anon@local",
    hashed_password="",
    is_active=True,
    role="admin",
    created_at=datetime.now(timezone.utc),
    updated_at=datetime.now(timezone.utc),
)

_ANON_AGENT = Agent(
    id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
    name="anon-agent",
    role="agent",
    provider="local",
    model="local",
    adapter_type="api",
    capabilities=[],
    config={},
    is_active=True,
    created_at=datetime.now(timezone.utc),
    updated_at=datetime.now(timezone.utc),
)


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> User:
    if not settings.auth_enabled:
        return _ANON_USER
    authorization = request.headers.get("Authorization", "")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    token = authorization[7:]
    payload = decode_access_token(token)
    user_id_str = payload.get("sub")
    if not user_id_str:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


async def get_current_agent(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Agent:
    if not settings.auth_enabled:
        return _ANON_AGENT
    authorization = request.headers.get("Authorization", "")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid auth")
    key = authorization[7:]
    hashed = hash_api_key(key)
    result = await db.execute(select(ApiKey).where(ApiKey.hashed_key == hashed))
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    if api_key.expires_at and api_key.expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key expired")
    if api_key.agent_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Key not agent-scoped")
    api_key.last_used_at = datetime.now(timezone.utc)
    result2 = await db.execute(select(Agent).where(Agent.id == api_key.agent_id))
    agent = result2.scalar_one_or_none()
    if not agent or not agent.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Agent not found")
    return agent


async def get_caller(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> User | Agent:
    if not settings.auth_enabled:
        return _ANON_USER
    authorization = request.headers.get("Authorization", "")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No auth provided")
    token = authorization[7:]
    if not token.startswith(settings.api_key_prefix):
        try:
            payload = decode_access_token(token)
            user_id_str = payload.get("sub")
            if user_id_str:
                try:
                    user_id = uuid.UUID(user_id_str)
                except ValueError:
                    user_id = None
                if user_id:
                    result = await db.execute(select(User).where(User.id == user_id))
                    user = result.scalar_one_or_none()
                    if user and user.is_active:
                        return user
        except HTTPException:
            pass
    hashed = hash_api_key(token)
    result = await db.execute(select(ApiKey).where(ApiKey.hashed_key == hashed))
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if api_key.expires_at and api_key.expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key expired")
    if api_key.agent_id:
        result2 = await db.execute(select(Agent).where(Agent.id == api_key.agent_id))
        agent = result2.scalar_one_or_none()
        if agent and agent.is_active:
            return agent
    if api_key.user_id:
        result3 = await db.execute(select(User).where(User.id == api_key.user_id))
        user = result3.scalar_one_or_none()
        if user and user.is_active:
            return user
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")


async def ensure_project_exists(db: AsyncSession, project_id: uuid.UUID) -> None:
    """Existence check only — does NOT enforce authorization. Any authenticated caller can probe project UUIDs."""
    result = await db.execute(select(Project.id).where(Project.id == project_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
