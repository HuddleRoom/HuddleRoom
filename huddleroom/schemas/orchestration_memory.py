import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class OrchestrationMemorySectionUpsert(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_key: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    title: str = Field(min_length=1, max_length=255)
    body: str = Field(min_length=1)
    summary: str | None = None
    section_type: str = Field(default="text", max_length=50)
    always_load: bool = False
    toc_order: int = Field(default=0, ge=0, le=2_147_483_647)


class OrchestrationMemorySectionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    goal_id: uuid.UUID
    run_id: uuid.UUID | None
    section_key: str
    title: str
    section_type: str
    body: str
    summary: str | None
    always_load: bool
    toc_order: int
    created_by: str
    created_from_event_id: uuid.UUID | None
    updated_from_event_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime


class OrchestrationMemoryTocEntry(BaseModel):
    """Compact TOC entry — deliberately excludes body."""

    model_config = ConfigDict(from_attributes=True)

    section_key: str
    title: str
    summary: str | None
    section_type: str
    always_load: bool
    toc_order: int
    updated_at: datetime


class OrchestrationMemoryPrefaceCurrentProcess(BaseModel):
    process_type: str
    status: str


class OrchestrationMemoryPrefaceWarning(BaseModel):
    severity: str
    warning_type: str
    message: str | None
    acknowledged: bool


class OrchestrationMemoryPrefaceDecision(BaseModel):
    title: str | None
    authority: str
    selected_option: str | None
    overrides_recommendation: bool


class OrchestrationMemoryPrefaceSkippedProcess(BaseModel):
    process_type: str
    skipped_by: str | None


class OrchestrationMemoryPrefaceAlwaysLoaded(BaseModel):
    section_key: str
    summary: str | None


class OrchestrationMemoryPrefaceTocEntry(BaseModel):
    section_key: str
    title: str | None


class OrchestrationMemoryPreface(BaseModel):
    objective: str | None
    goal_status: str
    goal_weight: str
    run_status: str | None
    current_process: OrchestrationMemoryPrefaceCurrentProcess | None
    manager: str | None
    hierarchy: str | None
    constraints: str | None
    active_warnings: list[OrchestrationMemoryPrefaceWarning]
    recent_decisions: list[OrchestrationMemoryPrefaceDecision]
    open_blockers: list[str | None]
    skipped_processes: list[OrchestrationMemoryPrefaceSkippedProcess]
    introduction: str | None
    always_loaded: list[OrchestrationMemoryPrefaceAlwaysLoaded]
    toc: list[OrchestrationMemoryPrefaceTocEntry]


class OrchestrationMemoryOverviewResponse(BaseModel):
    toc: list[OrchestrationMemoryTocEntry] = Field(default_factory=list)
    always_loaded: list[OrchestrationMemorySectionResponse] = Field(default_factory=list)
    preface: OrchestrationMemoryPreface
