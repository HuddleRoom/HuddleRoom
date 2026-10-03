import uuid
from datetime import datetime
from sqlalchemy import ForeignKey, Index, JSON, String, Integer, Boolean, text
from sqlalchemy.orm import Mapped, mapped_column
from huddleroom.models.base import Base, TimestampMixin, _utcnow_naive


class Artifact(Base, TimestampMixin):
    __tablename__ = "artifacts"
    __table_args__ = (
        Index("idx_artifacts_project_type", "project_id", "artifact_type"),
        Index("idx_artifacts_project", "project_id"),
        Index("idx_artifacts_status", "project_id", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    artifact_type: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, server_default="draft")
    path: Mapped[str | None] = mapped_column(String, nullable=True)
    url: Mapped[str | None] = mapped_column(String, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    previous_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)
    created_by_agent: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("agents.id", ondelete="SET NULL"), nullable=True
    )
    created_by_user: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    linked_task_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    is_breaking: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))


class ArtifactWatcher(Base):
    __tablename__ = "artifact_watchers"
    __table_args__ = (
        Index("idx_aw_artifact", "artifact_id"),
        Index("idx_aw_watcher", "watcher_kind", "watcher_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    artifact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("artifacts.id", ondelete="CASCADE"), nullable=False
    )
    watcher_kind: Mapped[str] = mapped_column(String, nullable=False)
    watcher_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    event_filter: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, default=_utcnow_naive
    )
