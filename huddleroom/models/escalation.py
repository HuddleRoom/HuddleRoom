import uuid
from datetime import datetime
from sqlalchemy import Index, JSON, String, Boolean, text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from huddleroom.models.base import Base, TimestampMixin


class EscalationChain(Base, TimestampMixin):
    __tablename__ = "escalation_chains"
    __table_args__ = (
        Index("idx_ec_project", "project_id"),
        UniqueConstraint("project_id", "name", name="uq_escalation_chains_project_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    definition: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    steps: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
