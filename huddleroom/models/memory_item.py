import uuid
from sqlalchemy import Boolean, DateTime, ForeignKey, Index, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from huddleroom.models.base import Base, _utcnow
from datetime import datetime


class MemoryItem(Base):
    __tablename__ = "memory_items"
    __table_args__ = (
        Index("idx_memory_agent_id", "agent_id"),
        Index("idx_memory_project_id", "project_id"),
        Index("idx_memory_scope", "scope"),
        Index("idx_memory_shared", "shared"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    agent_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"), nullable=False)
    project_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=True)
    scope: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tags: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    shared: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    def __init__(self, **kwargs):
        # Set Python-side defaults for fields that may not be provided
        if 'tags' not in kwargs:
            kwargs['tags'] = []
        if 'shared' not in kwargs:
            kwargs['shared'] = True
        super().__init__(**kwargs)
