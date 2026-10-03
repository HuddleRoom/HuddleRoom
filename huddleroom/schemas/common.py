from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")


class CursorPage(BaseModel, Generic[T]):
    items: list[T]
    next_cursor: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    errors: list[dict] | None = None
