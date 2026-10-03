from datetime import datetime
import os
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator


def validate_workspace_path(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("workspace path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("workspace path is unavailable") from exc
    if not resolved.is_dir():
        raise ValueError("workspace path must be a directory")
    if not os.access(resolved, os.R_OK | os.W_OK | os.X_OK):
        raise ValueError("workspace path must be readable, writable, and searchable")
    return str(resolved)


def validate_optional_workspace_path(value: str | None) -> str | None:
    return None if value is None else validate_workspace_path(value)


class ProjectCreate(BaseModel):
    name: str
    description: str | None = None
    workspace_path: str
    config: dict | None = None

    _validate_workspace_path = field_validator("workspace_path")(validate_workspace_path)


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    workspace_path: str | None = None
    config: dict | None = None

    _validate_workspace_path = field_validator("workspace_path")(validate_optional_workspace_path)


class ProjectResetRequest(BaseModel):
    confirm_name: str


class ProjectResetResponse(BaseModel):
    cancelled_sessions: int
    cancelled_meeting_tasks: int
    deletions: dict[str, int]


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    name: str
    description: str | None
    workspace_path: str | None
    status: str
    config: dict
    created_at: datetime
    updated_at: datetime
