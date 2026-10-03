import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user
from huddleroom.models.user import User
from huddleroom.schemas.orchestration_memory import (
    OrchestrationMemoryOverviewResponse,
    OrchestrationMemorySectionResponse,
    OrchestrationMemorySectionUpsert,
    OrchestrationMemoryTocEntry,
)
from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService
from huddleroom.services.orchestration_memory_preface import OrchestrationMemoryPrefaceBuilder
from huddleroom.services.orchestration_service import OrchestrationService

router = APIRouter()
service = OrchestrationMemoryService()
orchestration_service = OrchestrationService()
preface_builder = OrchestrationMemoryPrefaceBuilder()


async def _goal_or_404(db: AsyncSession, project_id: uuid.UUID, goal_id: uuid.UUID):
    goal = await orchestration_service.get_goal(db, project_id, goal_id)
    if goal is None:
        raise HTTPException(status_code=404, detail="Orchestration goal not found")
    return goal


@router.get("/goals/{goal_id}/memory", response_model=OrchestrationMemoryOverviewResponse)
async def get_memory_overview(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    goal = await _goal_or_404(db, project_id, goal_id)
    run = await orchestration_service.get_run_for_goal(db, project_id, goal_id)
    sections = await service.list_sections(db, project_id, goal_id)
    return OrchestrationMemoryOverviewResponse(
        toc=[OrchestrationMemoryTocEntry.model_validate(section) for section in sections],
        always_loaded=[
            OrchestrationMemorySectionResponse.model_validate(section)
            for section in sections
            if section.always_load
        ],
        preface=await preface_builder.build(db, goal, run, section_meta=sections),
    )


@router.get("/goals/{goal_id}/memory/{section_key}", response_model=OrchestrationMemorySectionResponse)
async def get_memory_section(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    section_key: str,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _goal_or_404(db, project_id, goal_id)
    section = await service.get_section(db, project_id, goal_id, section_key)
    if section is None:
        raise HTTPException(status_code=404, detail="Memory section not found")
    return OrchestrationMemorySectionResponse.model_validate(section)


@router.post("/goals/{goal_id}/memory", response_model=OrchestrationMemorySectionResponse)
async def upsert_memory_section(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    data: OrchestrationMemorySectionUpsert,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _goal_or_404(db, project_id, goal_id)
    section = await service.upsert_section(
        db,
        project_id,
        goal_id,
        section_key=data.section_key,
        title=data.title,
        body=data.body,
        summary=data.summary,
        section_type=data.section_type,
        always_load=data.always_load,
        toc_order=data.toc_order,
        created_by=f"human:{user.id}",
    )
    return OrchestrationMemorySectionResponse.model_validate(section)
