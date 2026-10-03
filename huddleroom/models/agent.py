import uuid
from sqlalchemy import JSON, Boolean, Index, String, text
from sqlalchemy.orm import Mapped, mapped_column
from huddleroom.models.base import Base, TimestampMixin


class Agent(Base, TimestampMixin):
    __tablename__ = "agents"
    __table_args__ = (
        Index("idx_agents_role", "role"),
        Index("idx_agents_is_active", "is_active"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    system_prompt: Mapped[str | None] = mapped_column(String, nullable=True)
    adapter_type: Mapped[str] = mapped_column(String, nullable=False, server_default="api")
    cli_runtime: Mapped[str | None] = mapped_column(String, nullable=True)
    capabilities: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
