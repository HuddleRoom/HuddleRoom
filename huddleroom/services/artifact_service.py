from __future__ import annotations

import asyncio
import hashlib
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.artifact import Artifact, ArtifactWatcher
from huddleroom.services.event_bus import emit_event

BREAKING_CHANGE_TYPES = {"api_spec", "db_schema", "interface"}


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_breaking_change(artifact: Artifact, old_hash: str | None, new_hash: str) -> bool:
    if artifact.artifact_type not in BREAKING_CHANGE_TYPES:
        return False
    if old_hash is None:
        return False
    return old_hash != new_hash


class ArtifactService:
    async def create(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        name: str,
        artifact_type: str,
        path: str | None = None,
        url: str | None = None,
        metadata: dict | None = None,
        linked_task_id: uuid.UUID | None = None,
        created_by_agent: uuid.UUID | None = None,
        created_by_user: uuid.UUID | None = None,
    ) -> Artifact:
        artifact = Artifact(
            project_id=project_id,
            name=name,
            artifact_type=artifact_type,
            path=path,
            url=url,
            metadata_=metadata or {},
            linked_task_id=linked_task_id,
            created_by_agent=created_by_agent,
            created_by_user=created_by_user,
        )
        db.add(artifact)
        await db.flush()
        await emit_event(db, project_id, "artifact.created", {
            "artifact_id": str(artifact.id),
            "artifact_type": artifact_type,
            "name": name,
        })
        return artifact

    async def get(self, db: AsyncSession, artifact_id: uuid.UUID) -> Artifact | None:
        result = await db.execute(select(Artifact).where(Artifact.id == artifact_id))
        return result.scalar_one_or_none()

    async def list(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        artifact_type: str | None = None,
        status: str | None = None,
    ) -> list[Artifact]:
        q = select(Artifact).where(Artifact.project_id == project_id)
        if artifact_type:
            q = q.where(Artifact.artifact_type == artifact_type)
        if status:
            q = q.where(Artifact.status == status)
        result = await db.execute(q)
        return list(result.scalars().all())

    async def update(
        self,
        db: AsyncSession,
        artifact: Artifact,
        name: str | None = None,
        status: str | None = None,
        path: str | None = None,
        url: str | None = None,
        metadata: dict | None = None,
    ) -> Artifact:
        changed = False
        if name is not None:
            artifact.name = name
            changed = True
        if status is not None:
            artifact.status = status
            changed = True
        if path is not None:
            artifact.path = path
            changed = True
        if url is not None:
            artifact.url = url
            changed = True
        if metadata is not None:
            artifact.metadata_ = metadata
            changed = True
        if not changed:
            return artifact
        await db.flush()
        await emit_event(db, artifact.project_id, "artifact.content_changed", {
            "artifact_id": str(artifact.id),
            "artifact_type": artifact.artifact_type,
            "name": artifact.name,
        })
        return artifact

    async def update_content_hash(self, db: AsyncSession, artifact: Artifact) -> Artifact:
        if not artifact.path:
            return artifact
        loop = asyncio.get_running_loop()
        new_hash = await loop.run_in_executor(None, _hash_file, artifact.path)
        if new_hash == artifact.content_hash:
            return artifact

        old_hash = artifact.content_hash
        is_breaking = _detect_breaking_change(artifact, old_hash, new_hash)

        artifact.previous_hash = old_hash
        artifact.content_hash = new_hash
        artifact.version += 1
        artifact.is_breaking = is_breaking
        await db.flush()

        event_type = "artifact.breaking_change" if is_breaking else "artifact.content_changed"
        await emit_event(db, artifact.project_id, event_type, {
            "artifact_id": str(artifact.id),
            "artifact_type": artifact.artifact_type,
            "name": artifact.name,
            "previous_hash": old_hash,
            "new_hash": new_hash,
            "is_breaking": is_breaking,
        })
        return artifact

    async def add_watcher(
        self,
        db: AsyncSession,
        artifact_id: uuid.UUID,
        watcher_kind: str,
        watcher_id: uuid.UUID,
        event_filter: list[str] | None = None,
    ) -> ArtifactWatcher:
        result = await db.execute(
            select(ArtifactWatcher).where(
                ArtifactWatcher.artifact_id == artifact_id,
                ArtifactWatcher.watcher_kind == watcher_kind,
                ArtifactWatcher.watcher_id == watcher_id,
            )
        )
        existing = result.scalar_one_or_none()
        if existing:
            return existing
        watcher = ArtifactWatcher(
            artifact_id=artifact_id,
            watcher_kind=watcher_kind,
            watcher_id=watcher_id,
            event_filter=event_filter,
        )
        db.add(watcher)
        await db.flush()
        return watcher

    async def remove_watcher(
        self, db: AsyncSession, artifact_id: uuid.UUID, watcher_kind: str, watcher_id: uuid.UUID
    ) -> None:
        result = await db.execute(
            select(ArtifactWatcher).where(
                ArtifactWatcher.artifact_id == artifact_id,
                ArtifactWatcher.watcher_kind == watcher_kind,
                ArtifactWatcher.watcher_id == watcher_id,
            )
        )
        watcher = result.scalar_one_or_none()
        if watcher:
            await db.delete(watcher)
            await db.flush()

    async def list_watchers(self, db: AsyncSession, artifact_id: uuid.UUID) -> list[ArtifactWatcher]:
        result = await db.execute(
            select(ArtifactWatcher).where(ArtifactWatcher.artifact_id == artifact_id)
        )
        return list(result.scalars().all())
