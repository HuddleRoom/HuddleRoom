"""Pydantic schemas for the "Ask the orchestrator" project advisor endpoint (T4.1).

Mirrors the field naming/serialization conventions of the existing
OrchestrationConversation* schemas (huddleroom/schemas/orchestration.py) so the
frontend types line up.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator


class ProjectAdvisorSubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        return value.strip()


class ProjectAdvisorCitation(BaseModel):
    type: str
    id: str
    goal_id: str | None = None
    label: str


class ProjectAdvisorTurnResponse(BaseModel):
    id: uuid.UUID
    question: str
    answer: str | None
    citations: list[ProjectAdvisorCitation]
    off_topic: bool
    status: str
    created_at: datetime


class ProjectAdvisorAllowance(BaseModel):
    enabled: bool
    unlimited: bool
    limit: int
    remaining: int


class ProjectAdvisorHistoryResponse(BaseModel):
    items: list[ProjectAdvisorTurnResponse]
    allowance: ProjectAdvisorAllowance
