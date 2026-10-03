from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user, ensure_project_exists
from huddleroom.models.protocol import Protocol, ProtocolInstance, ProtocolTimeout, ProtocolTransition
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.models.user import User
from huddleroom.schemas.common import CursorPage
from huddleroom.schemas.protocol import (
    ActorAssignRequest,
    AdvanceRequest,
    ProtocolInstanceDetailResponse,
    ProtocolCreate,
    ProtocolInstanceResponse,
    ProtocolResponse,
    ProtocolSummaryResponse,
    ProtocolTimeoutResponse,
    ProtocolTransitionResponse,
    ProtocolInstanceSessionResponse,
    ProtocolInstanceTaskSummaryResponse,
    ProtocolUpdate,
)
from huddleroom.services.protocol_engine import ProtocolEngineService
from huddleroom.services.project_service import ProjectService

router = APIRouter()


async def _get_protocol_for_project(db: AsyncSession, project_id: uuid.UUID, protocol_id: uuid.UUID) -> Protocol:
    result = await db.execute(
        select(Protocol).where(
            Protocol.id == protocol_id,
            or_(Protocol.project_id == project_id, Protocol.project_id.is_(None)),
        )
    )
    protocol = result.scalar_one_or_none()
    if protocol is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Protocol not found")
    return protocol


async def _get_project_protocol(db: AsyncSession, project_id: uuid.UUID, protocol_id: uuid.UUID) -> Protocol:
    result = await db.execute(
        select(Protocol).where(Protocol.id == protocol_id, Protocol.project_id == project_id)
    )
    protocol = result.scalar_one_or_none()
    if protocol is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Protocol not found")
    return protocol


async def _get_project_instance(
    db: AsyncSession,
    project_id: uuid.UUID,
    instance_id: uuid.UUID,
) -> ProtocolInstance:
    result = await db.execute(
        select(ProtocolInstance).where(
            ProtocolInstance.id == instance_id,
            ProtocolInstance.project_id == project_id,
        )
    )
    instance = result.scalar_one_or_none()
    if instance is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Protocol instance not found")
    return instance


def _require_status(instance: ProtocolInstance, allowed: set[str]) -> None:
    if instance.status not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Protocol instance is {instance.status}; expected one of: {allowed_text}",
        )


@router.post("/projects/{project_id}/protocols", response_model=ProtocolResponse, status_code=201)
async def create_protocol(
    project_id: uuid.UUID,
    data: ProtocolCreate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    protocol = Protocol(
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
    db.add(protocol)
    await db.flush()
    return protocol


@router.get("/projects/{project_id}/protocols", response_model=list[ProtocolResponse])
async def list_protocols(
    project_id: uuid.UUID,
    include_inactive: bool = False,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    conditions = [or_(Protocol.project_id == project_id, Protocol.project_id.is_(None))]
    if not include_inactive:
        conditions.append(Protocol.is_active.is_(True))
    result = await db.execute(
        select(Protocol).where(*conditions)
    )
    return list(result.scalars().all())


@router.get("/projects/{project_id}/protocols/{protocol_id}", response_model=ProtocolResponse)
async def get_protocol(
    project_id: uuid.UUID,
    protocol_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _get_protocol_for_project(db, project_id, protocol_id)


@router.delete("/projects/{project_id}/protocols/{protocol_id}", status_code=204)
async def deactivate_protocol(
    project_id: uuid.UUID,
    protocol_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    protocol = await _get_project_protocol(db, project_id, protocol_id)
    protocol.is_active = False
    await db.flush()
    return None


@router.put("/projects/{project_id}/protocols/{protocol_id}", response_model=ProtocolResponse)
async def update_protocol(
    project_id: uuid.UUID,
    protocol_id: uuid.UUID,
    data: ProtocolUpdate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    protocol = await _get_project_protocol(db, project_id, protocol_id)
    _NON_TERMINAL = ["active", "paused"]
    non_terminal = await db.execute(
        select(ProtocolInstance).where(
            ProtocolInstance.protocol_id == protocol_id,
            ProtocolInstance.status.in_(_NON_TERMINAL),
        )
    )
    if non_terminal.scalars().first() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot update protocol while non-terminal instances exist (active or paused)",
        )
    _NON_NULLABLE_PROTOCOL = {"name", "version", "definition", "triggers"}
    for field, value in data.model_dump(exclude_unset=True).items():
        if value is None and field in _NON_NULLABLE_PROTOCOL:
            raise HTTPException(status_code=422, detail=f"Field '{field}' cannot be null")
        setattr(protocol, field, value)
    try:
        await db.flush()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail="A protocol with this name and version already exists for the project") from exc
    return protocol


@router.post("/projects/{project_id}/protocols/{protocol_id}/activate", response_model=ProtocolResponse)
async def activate_protocol(
    project_id: uuid.UUID,
    protocol_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    protocol = await _get_project_protocol(db, project_id, protocol_id)
    protocol.is_active = True
    await db.flush()
    return protocol


@router.get("/projects/{project_id}/protocol-instances", response_model=CursorPage[ProtocolInstanceResponse])
async def list_protocol_instances(
    project_id: uuid.UUID,
    status: str | None = None,
    protocol_id: uuid.UUID | None = None,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    conditions = [ProtocolInstance.project_id == project_id]
    if status is not None:
        conditions.append(ProtocolInstance.status == status)
    if protocol_id is not None:
        conditions.append(ProtocolInstance.protocol_id == protocol_id)
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
                ProtocolInstance.created_at < cursor_dt,
                and_(ProtocolInstance.created_at == cursor_dt, ProtocolInstance.id < cursor_id),
            )
        )
    result = await db.execute(
        select(ProtocolInstance)
        .where(*conditions)
        .order_by(ProtocolInstance.created_at.desc(), ProtocolInstance.id.desc())
        .limit(limit + 1)
    )
    items = list(result.scalars().all())
    next_cursor = None
    if len(items) > limit:
        items = items[:limit]
        last = items[-1]
        next_cursor = f"{last.created_at.isoformat()}__{last.id}"
    return CursorPage(items=items, next_cursor=next_cursor)


@router.get("/projects/{project_id}/protocol-instances/count")
async def count_protocol_instances(
    project_id: uuid.UUID,
    status_filter: str | None = Query(default=None, alias="status"),
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    conditions = [ProtocolInstance.project_id == project_id]
    if status_filter is not None:
        conditions.append(ProtocolInstance.status == status_filter)
    result = await db.execute(select(func.count(ProtocolInstance.id)).where(*conditions))  # pylint: disable=not-callable
    return {"count": result.scalar_one()}


@router.get(
    "/projects/{project_id}/protocol-instances/{instance_id}/detail",
    response_model=ProtocolInstanceDetailResponse,
)
async def get_protocol_instance_detail(
    project_id: uuid.UUID,
    instance_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    instance = await _get_project_instance(db, project_id, instance_id)
    protocol = await _get_protocol_for_project(db, project_id, instance.protocol_id)

    transitions_result = await db.execute(
        select(ProtocolTransition)
        .where(ProtocolTransition.protocol_instance_id == instance.id)
        .order_by(ProtocolTransition.transitioned_at.asc())
    )
    sessions_result = await db.execute(
        select(Session)
        .where(
            Session.project_id == project_id,
            Session.protocol_instance_id == instance.id,
        )
        .order_by(Session.created_at.asc())
    )
    timeouts_result = await db.execute(
        select(ProtocolTimeout)
        .where(
            ProtocolTimeout.protocol_instance_id == instance.id,
            ProtocolTimeout.resolved.is_(False),
        )
        .order_by(ProtocolTimeout.expires_at.asc())
    )
    tasks_result = await db.execute(
        select(Task)
        .where(
            Task.project_id == project_id,
            Task.protocol_instance_id == instance.id,
        )
        .order_by(Task.created_at.asc())
    )

    return ProtocolInstanceDetailResponse(
        instance=ProtocolInstanceResponse.model_validate(instance),
        protocol=ProtocolSummaryResponse.model_validate(protocol),
        transitions=[
            ProtocolTransitionResponse.model_validate(transition)
            for transition in transitions_result.scalars().all()
        ],
        sessions=[
            ProtocolInstanceSessionResponse.model_validate(session)
            for session in sessions_result.scalars().all()
        ],
        timeouts=[
            ProtocolTimeoutResponse.model_validate(timeout)
            for timeout in timeouts_result.scalars().all()
        ],
        tasks=[
            ProtocolInstanceTaskSummaryResponse.model_validate(task)
            for task in tasks_result.scalars().all()
        ],
    )


@router.get("/projects/{project_id}/protocol-instances/{instance_id}", response_model=ProtocolInstanceResponse)
async def get_protocol_instance(
    project_id: uuid.UUID,
    instance_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _get_project_instance(db, project_id, instance_id)


@router.get(
    "/projects/{project_id}/protocol-instances/{instance_id}/transitions",
    response_model=list[ProtocolTransitionResponse],
)
async def list_protocol_instance_transitions(
    project_id: uuid.UUID,
    instance_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_project_instance(db, project_id, instance_id)
    result = await db.execute(
        select(ProtocolTransition)
        .where(ProtocolTransition.protocol_instance_id == instance_id)
        .order_by(ProtocolTransition.transitioned_at.asc())
    )
    return list(result.scalars().all())


@router.post(
    "/projects/{project_id}/protocol-instances/{instance_id}/actors/{role}",
    response_model=ProtocolInstanceResponse,
)
async def assign_actor(
    project_id: uuid.UUID,
    instance_id: uuid.UUID,
    role: str,
    data: ActorAssignRequest,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    instance = await _get_project_instance(db, project_id, instance_id)
    assignments = dict(instance.actor_assignments or {})
    assignments[role] = {"kind": data.kind, "id": str(data.id)}
    instance.actor_assignments = assignments
    await db.flush()
    return instance


@router.post("/projects/{project_id}/protocol-instances/{instance_id}/advance", response_model=ProtocolInstanceResponse)
async def advance_protocol_instance(
    project_id: uuid.UUID,
    instance_id: uuid.UUID,
    data: AdvanceRequest,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    instance = await _get_project_instance(db, project_id, instance_id)
    return await ProtocolEngineService().advance_manually(db, instance, data.to_state, data.reason)


@router.post("/projects/{project_id}/protocol-instances/{instance_id}/pause", response_model=ProtocolInstanceResponse)
async def pause_protocol_instance(
    project_id: uuid.UUID,
    instance_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    instance = await _get_project_instance(db, project_id, instance_id)
    _require_status(instance, {"active"})
    instance.status = "paused"
    await db.flush()
    return instance


@router.post("/projects/{project_id}/protocol-instances/{instance_id}/resume", response_model=ProtocolInstanceResponse)
async def resume_protocol_instance(
    project_id: uuid.UUID,
    instance_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    instance = await _get_project_instance(db, project_id, instance_id)
    _require_status(instance, {"paused"})
    await ProjectService().require_runnable_project(db, project_id)
    instance.status = "active"
    await db.flush()
    return instance


@router.post("/projects/{project_id}/protocol-instances/{instance_id}/abandon", response_model=ProtocolInstanceResponse)
async def abandon_protocol_instance(
    project_id: uuid.UUID,
    instance_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    instance = await _get_project_instance(db, project_id, instance_id)
    _require_status(instance, {"active", "paused"})
    instance.status = "failed"
    instance.completed_at = datetime.now(timezone.utc)
    result = await db.execute(
        select(ProtocolTimeout).where(
            ProtocolTimeout.protocol_instance_id == instance.id,
            ProtocolTimeout.resolved.is_(False),
        )
    )
    for timeout in result.scalars().all():
        timeout.resolved = True
        timeout.resolved_at = instance.completed_at
    await db.flush()
    return instance
