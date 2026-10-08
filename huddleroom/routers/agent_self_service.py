import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from huddleroom.database import get_db
from huddleroom.dependencies import get_current_agent
from huddleroom.models.agent import Agent
from huddleroom.models.session import Session
from huddleroom.schemas.agent import AgentContextResponse
from huddleroom.services.agent_service import AgentService
from huddleroom.services.knowledge_service import KnowledgeService
from huddleroom.services.task_service import TaskService
from huddleroom.schemas.knowledge import KnowledgeCreate
from huddleroom.schemas.task import TaskCreate

router = APIRouter()
agent_service = AgentService()
knowledge_service = KnowledgeService()
task_service = TaskService()


# Request body for /agent/report
class KnowledgeItemReport(BaseModel):
    title: str | None = None
    content: str
    content_type: str
    tags: list[str] = []


class SubtaskReport(BaseModel):
    title: str
    description: str | None = None
    assigned_to: uuid.UUID | None = None
    priority: int = 50


class DecisionReport(BaseModel):
    statement: str
    rationale: str
    rejected_alternatives: list[str] = []


class StatusUpdateReport(BaseModel):
    task_id: uuid.UUID
    new_status: str
    reason: str | None = None


class AgentReport(BaseModel):
    session_id: uuid.UUID
    summary: str | None = None
    knowledge_items: list[KnowledgeItemReport] = []
    subtasks: list[SubtaskReport] = []
    decisions: list[DecisionReport] = []
    status_update: StatusUpdateReport | None = None


class AgentReportResponse(BaseModel):
    knowledge_created: int
    subtasks_created: int
    task_status_updated: bool


@router.get("/context", response_model=AgentContextResponse)
async def get_agent_context(
    current_agent: Agent = Depends(get_current_agent),
    db: AsyncSession = Depends(get_db),
):
    return await agent_service.build_context(db, current_agent.id)


@router.post("/report", response_model=AgentReportResponse, status_code=201)
async def post_agent_report(
    data: AgentReport,
    current_agent: Agent = Depends(get_current_agent),
    db: AsyncSession = Depends(get_db),
):
    # We need a project_id for creating knowledge + tasks
    # Get it from the session
    result = await db.execute(select(Session).where(Session.id == data.session_id))
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    project_id = session.project_id

    knowledge_created = 0
    for ki in data.knowledge_items:
        k_data = KnowledgeCreate(
            title=ki.title,
            content=ki.content,
            content_type=ki.content_type,
            tags=ki.tags,
            provenance={"source_type": "session", "source_id": str(data.session_id)},
        )
        await knowledge_service.create(
            db, project_id, k_data, created_by_agent=current_agent.id
        )
        knowledge_created += 1

    # Decisions stored as knowledge items with content_type="decision"
    for decision in data.decisions:
        k_data = KnowledgeCreate(
            title=decision.statement[:100],
            content=f"Decision: {decision.statement}\n\nRationale: {decision.rationale}\n\nRejected alternatives: {', '.join(decision.rejected_alternatives)}",
            content_type="decision",
            tags=[],
            provenance={"source_type": "session", "source_id": str(data.session_id)},
        )
        await knowledge_service.create(
            db, project_id, k_data, created_by_agent=current_agent.id
        )
        knowledge_created += 1

    subtasks_created = 0
    if project_id:
        # Find parent task from session
        parent_task_id = session.task_id
        for st in data.subtasks:
            t_data = TaskCreate(
                title=st.title,
                description=st.description,
                priority=st.priority,
                assigned_to=st.assigned_to,
                parent_id=parent_task_id,
            )
            task = await task_service.create(db, project_id, t_data)
            await task_service.auto_start(db, task)
            subtasks_created += 1

    task_status_updated = False
    if data.status_update and project_id:
        try:
            await task_service.transition_status(
                db,
                project_id,
                data.status_update.task_id,
                data.status_update.new_status,
                data.status_update.reason,
            )
            task_status_updated = True
        except HTTPException as exc:
            from huddleroom.services.session_service import SessionClaimAttention, SessionService
            if isinstance(exc, SessionClaimAttention):
                await SessionService.persist_claim_attention(db, exc)
                await db.commit()
                raise
            # 409 (invalid transition) or 404 (task not found) — best-effort, not fatal
            pass

    return AgentReportResponse(
        knowledge_created=knowledge_created,
        subtasks_created=subtasks_created,
        task_status_updated=task_status_updated,
    )
