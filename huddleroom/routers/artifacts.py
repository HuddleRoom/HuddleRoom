from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user
from huddleroom.models.task import Task
from huddleroom.models.user import User
from huddleroom.schemas.artifact import ArtifactCreate, ArtifactResponse, ArtifactUpdate, WatchRequest, WatcherResponse
from huddleroom.services.artifact_service import ArtifactService
from huddleroom.services.event_bus import emit_event

router = APIRouter()
service = ArtifactService()


def _artifact_response(artifact) -> dict:
    return {
        "id": artifact.id,
        "project_id": artifact.project_id,
        "name": artifact.name,
        "artifact_type": artifact.artifact_type,
        "status": artifact.status,
        "path": artifact.path,
        "url": artifact.url,
        "content_hash": artifact.content_hash,
        "metadata": artifact.metadata_ or {},
        "linked_task_id": artifact.linked_task_id,
        "version": artifact.version,
        "is_breaking": artifact.is_breaking,
        "created_at": artifact.created_at,
        "updated_at": artifact.updated_at,
    }


async def _get_artifact_or_404(db: AsyncSession, artifact_id: uuid.UUID, project_id: uuid.UUID | None = None):
    artifact = await service.get(db, artifact_id)
    if artifact is None or (project_id is not None and artifact.project_id != project_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
    return artifact


async def _validate_linked_task(db: AsyncSession, project_id: uuid.UUID, linked_task_id: uuid.UUID | None) -> None:
    if linked_task_id is None:
        return
    result = await db.execute(
        select(Task.id).where(
            Task.id == linked_task_id,
            Task.project_id == project_id,
        )
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Linked task not found in project")


@router.post("/projects/{project_id}/artifacts", response_model=ArtifactResponse, status_code=201)
async def create_artifact(
    project_id: uuid.UUID,
    data: ArtifactCreate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _validate_linked_task(db, project_id, data.linked_task_id)
    artifact = await service.create(
        db,
        project_id=project_id,
        name=data.name,
        artifact_type=data.artifact_type,
        path=data.path,
        url=data.url,
        metadata=data.metadata,
        linked_task_id=data.linked_task_id,
        created_by_agent=data.created_by_agent,
        created_by_user=data.created_by_user,
    )
    return _artifact_response(artifact)


@router.get("/projects/{project_id}/artifacts", response_model=list[ArtifactResponse])
async def list_artifacts(
    project_id: uuid.UUID,
    artifact_type: str | None = None,
    status: str | None = None,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    artifacts = await service.list(db, project_id, artifact_type=artifact_type, status=status)
    return [_artifact_response(artifact) for artifact in artifacts]


@router.get("/projects/{project_id}/artifacts/{artifact_id}", response_model=ArtifactResponse)
async def get_artifact(
    project_id: uuid.UUID,
    artifact_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    artifact = await _get_artifact_or_404(db, artifact_id, project_id)
    return _artifact_response(artifact)


@router.put("/projects/{project_id}/artifacts/{artifact_id}", response_model=ArtifactResponse)
async def update_artifact(
    project_id: uuid.UUID,
    artifact_id: uuid.UUID,
    data: ArtifactUpdate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    artifact = await _get_artifact_or_404(db, artifact_id, project_id)
    updated = await service.update(
        db,
        artifact,
        name=data.name,
        status=data.status,
        path=data.path,
        url=data.url,
        metadata=data.metadata,
    )
    return _artifact_response(updated)


@router.delete("/projects/{project_id}/artifacts/{artifact_id}", status_code=204)
async def delete_artifact(
    project_id: uuid.UUID,
    artifact_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    artifact = await _get_artifact_or_404(db, artifact_id, project_id)
    artifact.status = "deleted"
    await db.flush()
    await emit_event(
        db,
        artifact.project_id,
        "artifact.deleted",
        {
            "artifact_id": str(artifact.id),
            "artifact_type": artifact.artifact_type,
            "name": artifact.name,
        },
    )
    return None


@router.post("/projects/{project_id}/artifacts/{artifact_id}/watch", response_model=WatcherResponse, status_code=201)
async def add_artifact_watcher(
    project_id: uuid.UUID,
    artifact_id: uuid.UUID,
    data: WatchRequest,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_artifact_or_404(db, artifact_id, project_id)
    return await service.add_watcher(
        db,
        artifact_id=artifact_id,
        watcher_kind=data.watcher_kind,
        watcher_id=data.watcher_id,
        event_filter=data.event_filter,
    )


@router.get("/projects/{project_id}/artifacts/{artifact_id}/watchers", response_model=list[WatcherResponse])
async def list_artifact_watchers(
    project_id: uuid.UUID,
    artifact_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_artifact_or_404(db, artifact_id, project_id)
    return await service.list_watchers(db, artifact_id)


@router.delete("/projects/{project_id}/artifacts/{artifact_id}/watch", status_code=204)
async def remove_artifact_watcher(
    project_id: uuid.UUID,
    artifact_id: uuid.UUID,
    watcher_kind: str = Query(...),
    watcher_id: uuid.UUID = Query(...),
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_artifact_or_404(db, artifact_id, project_id)
    await service.remove_watcher(db, artifact_id, watcher_kind, watcher_id)
    return None
