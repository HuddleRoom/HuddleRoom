import uuid
from sqlalchemy import Boolean, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from huddleroom.models.base import Base, TimestampMixin


class KnowledgeItem(Base, TimestampMixin):
    __tablename__ = "knowledge_items"
    __table_args__ = (
        Index("idx_knowledge_project_id", "project_id"),
        Index("idx_knowledge_content_type", "content_type"),
        Index("idx_knowledge_is_superseded", "is_superseded"),
        Index("idx_knowledge_conflict_status", "conflict_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=True)
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(String, nullable=False)
    tags: Mapped[list | None] = mapped_column(JSON, nullable=True)
    embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)

    provenance_type: Mapped[str] = mapped_column(String, nullable=False, default="human")
    provenance_session_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True)
    provenance_meeting_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    provenance_graph_run_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    created_by_agent: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("agents.id", ondelete="SET NULL"), nullable=True)
    created_by_user: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    supersedes: Mapped[list | None] = mapped_column(JSON, nullable=True)
    is_superseded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    conflict_status: Mapped[str] = mapped_column(String, nullable=False, default="none")
    conflicting_item_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)

    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)
