from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user, ensure_project_exists
from huddleroom.models.graph import Graph, GraphRun, GraphRunStep, GraphRunTimeout
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.models.user import User
from huddleroom.schemas.common import CursorPage
from huddleroom.schemas.graph import (
    ActorAssignRequest,
    AdvanceRequest,
    GraphRunDetailResponse,
    GraphCreate,
    GraphRunResponse,
    GraphResponse,
    GraphSummaryResponse,
    GraphRunTimeoutResponse,
    GraphRunStepResponse,
    GraphRunSessionResponse,
    GraphRunTaskSummaryResponse,
    GraphUpdate,
)
from huddleroom.services.graph_engine import GraphEngineService
from huddleroom.services.graph_service import validate_graph_definition
from huddleroom.services.project_service import ProjectService

router = APIRouter()


async def _get_graph_for_project(db: AsyncSession, project_id: uuid.UUID, graph_id: uuid.UUID) -> Graph:
    result = await db.execute(
        select(Graph).where(
            Graph.id == graph_id,
            or_(Graph.project_id == project_id, Graph.project_id.is_(None)),
        )
    )
    graph = result.scalar_one_or_none()
    if graph is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Graph not found")
    return graph


async def _get_project_graph(db: AsyncSession, project_id: uuid.UUID, graph_id: uuid.UUID) -> Graph:
    result = await db.execute(
        select(Graph).where(Graph.id == graph_id, Graph.project_id == project_id)
    )
    graph = result.scalar_one_or_none()
    if graph is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Graph not found")
    return graph


async def _get_project_run(
    db: AsyncSession,
    project_id: uuid.UUID,
    run_id: uuid.UUID,
) -> GraphRun:
    result = await db.execute(
        select(GraphRun).where(
            GraphRun.id == run_id,
            GraphRun.project_id == project_id,
        )
    )
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Graph run not found")
    return run


def _require_status(run: GraphRun, allowed: set[str]) -> None:
    if run.status not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Graph run is {run.status}; expected one of: {allowed_text}",
        )


@router.post("/projects/{project_id}/graphs", response_model=GraphResponse, status_code=201)
async def create_graph(
    project_id: uuid.UUID,
    data: GraphCreate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        validate_graph_definition(data.definition)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    graph = Graph(
        project_id=project_id,
        name=data.name,
        version=data.version,
        description=data.description,
        definition=data.definition,
        triggers=data.triggers,
        escalation_chain=data.escalation_chain,
        loaded_from=data.loaded_from,
        is_active=True,
    )
    db.add(graph)
    await db.flush()
    return graph


@router.get("/projects/{project_id}/graphs", response_model=list[GraphResponse])
async def list_graphs(
    project_id: uuid.UUID,
    include_inactive: bool = False,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    conditions = [or_(Graph.project_id == project_id, Graph.project_id.is_(None))]
    if not include_inactive:
        conditions.append(Graph.is_active.is_(True))
    result = await db.execute(
        select(Graph).where(*conditions)
    )
    return list(result.scalars().all())


@router.get("/projects/{project_id}/graphs/{graph_id}", response_model=GraphResponse)
async def get_graph(
    project_id: uuid.UUID,
    graph_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _get_graph_for_project(db, project_id, graph_id)


@router.delete("/projects/{project_id}/graphs/{graph_id}", status_code=204)
async def deactivate_graph(
    project_id: uuid.UUID,
    graph_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    graph = await _get_project_graph(db, project_id, graph_id)
    graph.is_active = False
    await db.flush()
    return None


@router.put("/projects/{project_id}/graphs/{graph_id}", response_model=GraphResponse)
async def update_graph(
    project_id: uuid.UUID,
    graph_id: uuid.UUID,
    data: GraphUpdate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    graph = await _get_project_graph(db, project_id, graph_id)
    _NON_TERMINAL = ["active", "paused"]
    non_terminal = await db.execute(
        select(GraphRun).where(
            GraphRun.graph_id == graph_id,
            GraphRun.status.in_(_NON_TERMINAL),
        )
    )
    if non_terminal.scalars().first() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot update graph while non-terminal runs exist (active or paused)",
        )
    _NON_NULLABLE_GRAPH = {"name", "version", "definition", "triggers"}
    for field, value in data.model_dump(exclude_unset=True).items():
        if value is None and field in _NON_NULLABLE_GRAPH:
            raise HTTPException(status_code=422, detail=f"Field '{field}' cannot be null")
        if field == "definition":
            try:
                validate_graph_definition(value)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        setattr(graph, field, value)
    try:
        await db.flush()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail="A graph with this name and version already exists for the project") from exc
    return graph


@router.post("/projects/{project_id}/graphs/{graph_id}/activate", response_model=GraphResponse)
async def activate_graph(
    project_id: uuid.UUID,
    graph_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    graph = await _get_project_graph(db, project_id, graph_id)
    graph.is_active = True
    await db.flush()
    return graph


@router.get("/projects/{project_id}/graph-runs", response_model=CursorPage[GraphRunResponse])
async def list_graph_runs(
    project_id: uuid.UUID,
    status: str | None = None,
    graph_id: uuid.UUID | None = None,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    conditions = [GraphRun.project_id == project_id]
    if status is not None:
        conditions.append(GraphRun.status == status)
    if graph_id is not None:
        conditions.append(GraphRun.graph_id == graph_id)
    if cursor is not None:
        parts = cursor.split("__", 1)
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid cursor")
        try:
            cursor_dt = datetime.fromisoformat(parts[0])
            cursor_id = uuid.UUID(parts[1])
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail="Invalid cursor") from exc
        conditions.append(
            or_(
                GraphRun.created_at < cursor_dt,
                and_(GraphRun.created_at == cursor_dt, GraphRun.id < cursor_id),
            )
        )
    result = await db.execute(
        select(GraphRun)
        .where(*conditions)
        .order_by(GraphRun.created_at.desc(), GraphRun.id.desc())
        .limit(limit + 1)
    )
    items = list(result.scalars().all())
    next_cursor = None
    if len(items) > limit:
        items = items[:limit]
        last = items[-1]
        next_cursor = f"{last.created_at.isoformat()}__{last.id}"
    return CursorPage(items=items, next_cursor=next_cursor)


@router.get("/projects/{project_id}/graph-runs/count")
async def count_graph_runs(
    project_id: uuid.UUID,
    status_filter: str | None = Query(default=None, alias="status"),
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    conditions = [GraphRun.project_id == project_id]
    if status_filter is not None:
        conditions.append(GraphRun.status == status_filter)
    result = await db.execute(select(func.count(GraphRun.id)).where(*conditions))  # pylint: disable=not-callable
    return {"count": result.scalar_one()}


@router.get(
    "/projects/{project_id}/graph-runs/{run_id}/detail",
    response_model=GraphRunDetailResponse,
)
async def get_graph_run_detail(
    project_id: uuid.UUID,
    run_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    run = await _get_project_run(db, project_id, run_id)
    graph = await _get_graph_for_project(db, project_id, run.graph_id)

    steps_result = await db.execute(
        select(GraphRunStep)
        .where(GraphRunStep.graph_run_id == run.id)
        .order_by(GraphRunStep.stepped_at.asc())
    )
    sessions_result = await db.execute(
        select(Session)
        .where(
            Session.project_id == project_id,
            Session.graph_run_id == run.id,
        )
        .order_by(Session.created_at.asc())
    )
    timeouts_result = await db.execute(
        select(GraphRunTimeout)
        .where(
            GraphRunTimeout.graph_run_id == run.id,
            GraphRunTimeout.resolved.is_(False),
        )
        .order_by(GraphRunTimeout.expires_at.asc())
    )
    tasks_result = await db.execute(
        select(Task)
        .where(
            Task.project_id == project_id,
            Task.graph_run_id == run.id,
        )
        .order_by(Task.created_at.asc())
    )

    return GraphRunDetailResponse(
        run=GraphRunResponse.model_validate(run),
        graph=GraphSummaryResponse.model_validate(graph),
        steps=[
            GraphRunStepResponse.model_validate(step)
            for step in steps_result.scalars().all()
        ],
        sessions=[
            GraphRunSessionResponse.model_validate(session)
            for session in sessions_result.scalars().all()
        ],
        timeouts=[
            GraphRunTimeoutResponse.model_validate(timeout)
            for timeout in timeouts_result.scalars().all()
        ],
        tasks=[
            GraphRunTaskSummaryResponse.model_validate(task)
            for task in tasks_result.scalars().all()
        ],
    )


@router.get("/projects/{project_id}/graph-runs/{run_id}", response_model=GraphRunResponse)
async def get_graph_run(
    project_id: uuid.UUID,
    run_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _get_project_run(db, project_id, run_id)


@router.get(
    "/projects/{project_id}/graph-runs/{run_id}/steps",
    response_model=list[GraphRunStepResponse],
)
async def list_graph_run_steps(
    project_id: uuid.UUID,
    run_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_project_run(db, project_id, run_id)
    result = await db.execute(
        select(GraphRunStep)
        .where(GraphRunStep.graph_run_id == run_id)
        .order_by(GraphRunStep.stepped_at.asc())
    )
    return list(result.scalars().all())


@router.post(
    "/projects/{project_id}/graph-runs/{run_id}/actors/{role}",
    response_model=GraphRunResponse,
)
async def assign_actor(
    project_id: uuid.UUID,
    run_id: uuid.UUID,
    role: str,
    data: ActorAssignRequest,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    run = await _get_project_run(db, project_id, run_id)
    assignments = dict(run.actor_assignments or {})
    assignments[role] = {"kind": data.kind, "id": str(data.id)}
    run.actor_assignments = assignments
    await db.flush()
    return run


@router.post("/projects/{project_id}/graph-runs/{run_id}/advance", response_model=GraphRunResponse)
async def advance_graph_run(
    project_id: uuid.UUID,
    run_id: uuid.UUID,
    data: AdvanceRequest,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    run = await _get_project_run(db, project_id, run_id)
    return await GraphEngineService().advance_manually(db, run, data.to_node, data.reason)


@router.post("/projects/{project_id}/graph-runs/{run_id}/pause", response_model=GraphRunResponse)
async def pause_graph_run(
    project_id: uuid.UUID,
    run_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    run = await _get_project_run(db, project_id, run_id)
    _require_status(run, {"active"})
    run.status = "paused"
    await db.flush()
    return run


@router.post("/projects/{project_id}/graph-runs/{run_id}/resume", response_model=GraphRunResponse)
async def resume_graph_run(
    project_id: uuid.UUID,
    run_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    run = await _get_project_run(db, project_id, run_id)
    _require_status(run, {"paused"})
    await ProjectService().require_runnable_project(db, project_id)
    run.status = "active"
    await db.flush()
    return run


@router.post("/projects/{project_id}/graph-runs/{run_id}/abandon", response_model=GraphRunResponse)
async def abandon_graph_run(
    project_id: uuid.UUID,
    run_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    run = await _get_project_run(db, project_id, run_id)
    _require_status(run, {"active", "paused"})
    run.status = "failed"
    run.completed_at = datetime.now(timezone.utc)
    result = await db.execute(
        select(GraphRunTimeout).where(
            GraphRunTimeout.graph_run_id == run.id,
            GraphRunTimeout.resolved.is_(False),
        )
    )
    for timeout in result.scalars().all():
        timeout.resolved = True
        timeout.resolved_at = run.completed_at
    await db.flush()
    return run
