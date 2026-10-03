import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, JSON, String, Text, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from huddleroom.models.base import Base, _utcnow

ADVISOR_STATUS_CHECK = "status IN ('pending', 'completed', 'failed')"


class ProjectAdvisorTurn(Base):
    """One Q&A turn of the read-only 'Ask the orchestrator' project advisor. Audit/history only."""

    __tablename__ = "project_advisor_turns"
    __table_args__ = (
        CheckConstraint(ADVISOR_STATUS_CHECK, name="ck_project_advisor_turns_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", name="fk_project_advisor_turns_project_id", ondelete="CASCADE"), nullable=False
    )
    actor_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", name="fk_project_advisor_turns_actor_id", ondelete="RESTRICT"), nullable=False
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    citations: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    off_topic: Mapped[bool] = mapped_column(nullable=False, default=False)
    tokens_used: Mapped[int] = mapped_column(nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", server_default="pending")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)


async def advisor_allowance_used(db: AsyncSession, project_id, actor_id) -> int:
    value = await db.scalar(
        select(func.coalesce(func.sum(ProjectAdvisorTurn.tokens_used), 0)).where(
            ProjectAdvisorTurn.project_id == project_id,
            ProjectAdvisorTurn.actor_id == actor_id,
            ProjectAdvisorTurn.status.in_(("completed", "failed")),
        )
    )
    return int(value)
