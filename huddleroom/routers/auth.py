import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user
from huddleroom.schemas.auth import (
    ApiKeyCreate, ApiKeyCreatedResponse, ApiKeyResponse,
    LoginRequest, TokenResponse, UserResponse,
)
from huddleroom.security import create_access_token
from huddleroom.services.auth_service import AuthService
from huddleroom.models.user import User

router = APIRouter()
service = AuthService()


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(data: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await service.create_user(db, data.email, data.password)
    return user


@router.post("/login", response_model=TokenResponse)
async def login(data: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await service.authenticate(db, data.email, data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token({"sub": str(user.id)})
    return TokenResponse(access_token=token)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(current_user: User = Depends(get_current_user)):
    token = create_access_token({"sub": str(current_user.id)})
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)):
    return current_user


@router.post("/api-keys", response_model=ApiKeyCreatedResponse, status_code=201)
async def create_api_key(
    data: ApiKeyCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    api_key, full_key = await service.create_api_key(
        db,
        label=data.label,
        user_id=current_user.id,
        agent_id=data.agent_id,
        project_id=data.project_id,
        expires_at=data.expires_at,
    )
    return ApiKeyCreatedResponse(
        id=api_key.id,
        key_prefix=api_key.key_prefix,
        label=api_key.label,
        agent_id=api_key.agent_id,
        project_id=api_key.project_id,
        created_at=api_key.created_at,
        expires_at=api_key.expires_at,
        key=full_key,
    )


@router.get("/api-keys", response_model=list[ApiKeyResponse])
async def list_api_keys(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    keys = await service.list_api_keys(db, current_user.id)
    return keys


@router.delete("/api-keys/{key_id}", status_code=204)
async def delete_api_key(
    key_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    deleted = await service.delete_api_key(db, key_id, current_user.id)
    if not deleted:
        raise HTTPException(status_code=404, detail="API key not found")
