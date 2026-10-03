from __future__ import annotations

import uuid
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException

from huddleroom.schemas.knowledge import KnowledgeCreate, KnowledgeUpdate, KnowledgeSearchResult
from huddleroom.services.embedding_service import EmbeddingService
from huddleroom.models.knowledge_item import KnowledgeItem

embedding_service = EmbeddingService()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a)
    vb = np.array(b)
    dot = np.dot(va, vb)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm == 0:
        return 0.0
    return float(dot / norm)


class KnowledgeService:
    async def create(
        self,
        db: AsyncSession,
        project_id: uuid.UUID | None,
        data: KnowledgeCreate,
        created_by_agent: uuid.UUID | None = None,
        created_by_user: uuid.UUID | None = None,
    ) -> KnowledgeItem:
        embedding = await embedding_service.generate_embedding(data.content)

        provenance_type = "human"
        provenance_session_id = None
        if data.provenance:
            provenance_type = data.provenance.get("source_type", "human")
            src_id = data.provenance.get("source_id")
            if provenance_type == "session" and src_id:
                provenance_session_id = uuid.UUID(src_id)

        item = KnowledgeItem(
            project_id=project_id,
            title=data.title,
            content=data.content,
            content_type=data.content_type,
            tags=data.tags,
            embedding=embedding,
            provenance_type=provenance_type,
            provenance_session_id=provenance_session_id,
            created_by_agent=created_by_agent,
            created_by_user=created_by_user,
         )
        db.add(item)
        await db.flush()

        if data.supersedes:
            for old_id in data.supersedes:
                result = await db.execute(select(KnowledgeItem).where(KnowledgeItem.id == old_id))
                old_item = result.scalar_one_or_none()
                if old_item:
                    old_item.is_superseded = True

        return item

    async def get(self, db: AsyncSession, item_id: uuid.UUID) -> KnowledgeItem | None:
        result = await db.execute(select(KnowledgeItem).where(KnowledgeItem.id == item_id))
        return result.scalar_one_or_none()

    async def get_or_404(self, db: AsyncSession, item_id: uuid.UUID) -> KnowledgeItem:
        item = await self.get(db, item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Knowledge item not found")
        return item

    async def update(self, db: AsyncSession, item_id: uuid.UUID, data: KnowledgeUpdate) -> KnowledgeItem:
        item = await self.get_or_404(db, item_id)
        if data.title is not None:
            item.title = data.title
        if data.content is not None:
            item.content = data.content
            item.embedding = await embedding_service.generate_embedding(data.content)
        if data.tags is not None:
            item.tags = data.tags
        await db.flush()
        return item

    async def delete(self, db: AsyncSession, item_id: uuid.UUID) -> None:
        item = await self.get_or_404(db, item_id)
        await db.delete(item)

    async def list(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        content_type: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> tuple[list[KnowledgeItem], str | None]:
        from datetime import datetime
        from sqlalchemy import or_, and_
        query = (
            select(KnowledgeItem)
            .where(KnowledgeItem.project_id == project_id)
            .order_by(KnowledgeItem.created_at.desc(), KnowledgeItem.id.desc())
            .limit(limit + 1)
        )
        if content_type:
            query = query.where(KnowledgeItem.content_type == content_type)
        if cursor:
            try:
                cursor_dt_str, cursor_id_str = cursor.split("__", 1)
                cursor_dt = datetime.fromisoformat(cursor_dt_str)
                cursor_id = uuid.UUID(cursor_id_str)
            except (ValueError, AttributeError) as exc:
                raise HTTPException(status_code=400, detail="Invalid cursor") from exc
            query = query.where(
                or_(
                    KnowledgeItem.created_at < cursor_dt,
                    and_(KnowledgeItem.created_at == cursor_dt, KnowledgeItem.id < cursor_id),
                )
            )
        result = await db.execute(query)
        items = list(result.scalars().all())
        next_cursor = None
        if len(items) > limit:
            items = items[:limit]
            last = items[-1]
            next_cursor = f"{last.created_at.isoformat()}__{last.id}"
        return items, next_cursor

    async def search(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        query: str,
        limit: int = 10,
        content_type: str | None = None,
        min_relevance_score: float = 0.7,
    ) -> list[KnowledgeSearchResult]:
        query_embedding = await embedding_service.generate_embedding(query)
        if query_embedding is None:
            return []

        # TODO: Replace with pgvector <=> operator query when using PostgreSQL — current approach loads all items into memory.
        stmt = select(KnowledgeItem).where(
            KnowledgeItem.project_id == project_id,
            KnowledgeItem.embedding.isnot(None),
            KnowledgeItem.is_superseded.is_(False),
         )
        if content_type:
            stmt = stmt.where(KnowledgeItem.content_type == content_type)

        # TODO(pgvector): Add Postgres O(log n) vector search once embedding column is migrated
        # from JSON to pgvector Vector(1536). Requires alembic migration + pgvector extension.

        result = await db.execute(stmt)
        items = result.scalars().all()

        scored = []
        for item in items:
            score = _cosine_similarity(query_embedding, item.embedding)
            if score >= min_relevance_score:
                scored.append((item, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:limit]

        return [
            KnowledgeSearchResult(
                id=item.id,
                project_id=item.project_id,
                title=item.title,
                content=item.content,
                content_type=item.content_type,
                tags=item.tags,
                relevance_score=score,
            )
            for item, score in scored
         ]
