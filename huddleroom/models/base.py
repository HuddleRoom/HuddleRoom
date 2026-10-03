from datetime import datetime, timezone
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_naive() -> datetime:
    return _utcnow().replace(tzinfo=None)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow_naive, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow_naive, onupdate=_utcnow_naive, nullable=False
    )
