from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ApiKeyCreate(BaseModel):
    label: str | None = None
    agent_id: UUID | None = None
    project_id: UUID | None = None
    expires_at: datetime | None = None


class ApiKeyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    key_prefix: str
    label: str | None
    agent_id: UUID | None
    project_id: UUID | None
    created_at: datetime
    expires_at: datetime | None


class ApiKeyCreatedResponse(ApiKeyResponse):
    key: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    email: str
    display_name: str | None
    role: str
    is_active: bool
    created_at: datetime
