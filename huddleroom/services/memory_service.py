from __future__ import annotations

import uuid
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.memory_item import MemoryItem
from huddleroom.services.embedding_service import EmbeddingService

embedding_service = EmbeddingService()


class MemoryService:
    async def write(
        self,
        db: AsyncSession,
        agent_id: uuid.UUID,
        project_id: uuid.UUID | None,
        content: str,
        tags: list[str],
        shared: bool,
        scope: str,
    ) -> MemoryItem:
        if scope == "global":
            project_id = None

        embedding = await embedding_service.generate_embedding(content)

        item = MemoryItem(
            agent_id=agent_id,
            project_id=project_id,
            scope=scope,
            content=content,
            tags=tags or [],
            embedding=embedding,
            shared=shared,
        )
        db.add(item)
        await db.flush()
        return item

    async def read(
        self,
        db: AsyncSession,
        agent_id: uuid.UUID | None,
        project_id: uuid.UUID | None,
        item_id: uuid.UUID,
    ) -> MemoryItem | None:
        result = await db.execute(select(MemoryItem).where(MemoryItem.id == item_id))
        item = result.scalar_one_or_none()
        if item is None:
            return None
        if agent_id is None:
            return item
        if not self._is_accessible(item, agent_id, project_id):
            return None
        return item

    async def delete(
        self,
        db: AsyncSession,
        agent_id: uuid.UUID,
        item_id: uuid.UUID,
    ) -> bool:
        result = await db.execute(select(MemoryItem).where(MemoryItem.id == item_id))
        item = result.scalar_one_or_none()
        if item is None or item.agent_id != agent_id:
            return False
        await db.delete(item)
        await db.flush()
        return True

    async def delete_any(
        self,
        db: AsyncSession,
        item_id: uuid.UUID,
    ) -> bool:
        """Human REST path — delete any memory regardless of ownership."""
        result = await db.execute(select(MemoryItem).where(MemoryItem.id == item_id))
        item = result.scalar_one_or_none()
        if item is None:
            return False
        await db.delete(item)
        await db.flush()
        return True

    async def search(
        self,
        db: AsyncSession,
        agent_id: uuid.UUID | None,
        project_id: uuid.UUID | None,
        query: str,
        limit: int = 5,
        tags: list[str] | None = None,
        min_relevance_score: float = 0.7,
    ) -> list:
        query_embedding = await embedding_service.generate_embedding(query)
        if query_embedding is None:
            return []

        from sqlalchemy import or_, and_

        if agent_id is not None:
            # Pre-filter to items the agent can potentially access:
            # 1. Own items in this project (project scope)
            # 2. Shared items in this project (project scope)
            # 3. Global shared items
            # 4. Own global items
            stmt = select(MemoryItem).where(
                MemoryItem.embedding.isnot(None),
                or_(
                    MemoryItem.agent_id == agent_id,
                    and_(MemoryItem.scope == "project", MemoryItem.project_id == project_id, MemoryItem.shared.is_(True)),
                    and_(MemoryItem.scope == "global", MemoryItem.shared.is_(True)),
                )
            )
        else:
            # Human inspection endpoint: restrict to the given project + global shared
            if project_id is not None:
                stmt = select(MemoryItem).where(
                    MemoryItem.embedding.isnot(None),
                    or_(
                        and_(MemoryItem.scope == "project", MemoryItem.project_id == project_id),
                        and_(MemoryItem.scope == "global", MemoryItem.shared.is_(True)),
                    )
                )
            else:
                stmt = select(MemoryItem).where(MemoryItem.embedding.isnot(None))

        result = await db.execute(stmt)
        items = result.scalars().all()

        scored = []
        for item in items:
            if agent_id is not None and not self._is_accessible(item, agent_id, project_id):
                continue
            if tags:
                item_tags = item.tags or []
                if not any(t in item_tags for t in tags):
                    continue
            if item.embedding is None:
                continue

            va = np.array(query_embedding)
            vb = np.array(item.embedding)
            dot = np.dot(va, vb)
            norm = np.linalg.norm(va) * np.linalg.norm(vb)
            score = float(dot / norm) if norm > 0 else 0.0

            if score >= min_relevance_score:
                scored.append((item, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:limit]

        from huddleroom.schemas.memory import MemorySearchResult
        return [
            MemorySearchResult(
                id=item.id,
                agent_id=item.agent_id,
                project_id=item.project_id,
                scope=item.scope,
                content=item.content,
                tags=item.tags or [],
                shared=item.shared,
                relevance_score=score,
                created_at=item.created_at,
            )
            for item, score in scored
        ]

    async def list_memories(
        self,
        db: AsyncSession,
        project_id: uuid.UUID | None,
        scope: str | None = None,
        agent_id_filter: uuid.UUID | None = None,
        shared_filter: bool | None = None,
        tags_filter: list[str] | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> tuple[list[MemoryItem], str | None]:
        from datetime import datetime
        from sqlalchemy import or_, and_

        query = select(MemoryItem).order_by(
            MemoryItem.created_at.desc(), MemoryItem.id.desc()
        ).limit(limit + 1)

        if scope == "global":
            query = query.where(MemoryItem.scope == "global")
        elif project_id is not None:
            query = query.where(
                or_(
                    and_(MemoryItem.scope == "project", MemoryItem.project_id == project_id),
                    and_(MemoryItem.scope == "global", MemoryItem.shared.is_(True)),
                )
            )

        if agent_id_filter is not None:
            query = query.where(MemoryItem.agent_id == agent_id_filter)
        if shared_filter is not None:
            query = query.where(MemoryItem.shared.is_(shared_filter))

        if cursor:
            try:
                cursor_dt_str, cursor_id_str = cursor.split("__", 1)
                cursor_dt = datetime.fromisoformat(cursor_dt_str)
                cursor_id = uuid.UUID(cursor_id_str)
            except (ValueError, AttributeError) as exc:
                from fastapi import HTTPException
                raise HTTPException(status_code=400, detail="Invalid cursor") from exc
            query = query.where(
                or_(
                    MemoryItem.created_at < cursor_dt,
                    and_(MemoryItem.created_at == cursor_dt, MemoryItem.id < cursor_id),
                )
            )

        result = await db.execute(query)
        items = list(result.scalars().all())

        if tags_filter:
            items = [item for item in items if any(t in (item.tags or []) for t in tags_filter)]

        next_cursor = None
        if len(items) > limit:
            items = items[:limit]
            last = items[-1]
            next_cursor = f"{last.created_at.isoformat()}__{last.id}"
        return items, next_cursor

    def _is_accessible(
        self,
        item: MemoryItem,
        agent_id: uuid.UUID,
        project_id: uuid.UUID | None,
    ) -> bool:
        if item.agent_id == agent_id:
            if item.scope == "global":
                return True
            if item.scope == "project":
                # Allow if project matches OR if no project context (e.g., global session)
                if item.project_id == project_id or project_id is None:
                    return True
            return False
        if item.scope == "project" and item.shared and item.project_id == project_id:
            return True
        if item.scope == "global" and item.shared:
            return True
        return False
