import uuid
from datetime import datetime
from sqlalchemy import BigInteger, Index, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from huddleroom.models.base import Base, _utcnow_naive


class EventLog(Base):
    __tablename__ = "event_log"
    __table_args__ = (
        Index("idx_event_log_project_emitted", "project_id", "emitted_at"),
        Index("idx_event_log_event_type", "event_type"),
        UniqueConstraint("project_id", "dedup_key", name="uq_event_log_project_dedup_key"),
    )

    # BigInteger on Postgres, INTEGER on SQLite so it aliases rowid and autoincrements
    # (SQLite only auto-increments a literal INTEGER PRIMARY KEY, not BIGINT).
    seq: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    id: Mapped[uuid.UUID] = mapped_column(unique=True, nullable=False, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    dedup_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    source: Mapped[str] = mapped_column(String(50), nullable=False, server_default="system")
    emitted_at: Mapped[datetime] = mapped_column(default=_utcnow_naive, nullable=False)
