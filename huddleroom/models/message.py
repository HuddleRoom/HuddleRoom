import uuid
from datetime import datetime
from sqlalchemy import CheckConstraint, ForeignKey, Index, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from huddleroom.models.base import Base, _utcnow_naive


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint(
            "sender_agent_id IS NOT NULL OR sender_user_id IS NOT NULL",
            name="ck_messages_sender",
        ),
        Index("idx_messages_channel_created", "channel_id", "created_at"),
        Index("idx_messages_sender_agent", "sender_agent_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    channel_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("channels.id", ondelete="CASCADE"), nullable=False)
    sender_agent_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("agents.id", ondelete="SET NULL"), nullable=True)
    sender_user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    message_type: Mapped[str] = mapped_column(String, nullable=False, server_default="text")
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow_naive, nullable=False)
