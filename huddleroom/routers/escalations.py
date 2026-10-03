from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user
from huddleroom.models.escalation import EscalationChain
from huddleroom.models.user import User
from huddleroom.schemas.escalation import EscalationChainCreate, EscalationChainResponse

router = APIRouter()


async def _get_chain_or_404(
    db: AsyncSession, chain_id: uuid.UUID, project_id: uuid.UUID, *, include_global: bool = True
) -> EscalationChain:
    if include_global:
        cond = or_(EscalationChain.project_id == project_id, EscalationChain.project_id.is_(None))
    else:
        cond = EscalationChain.project_id == project_id
    result = await db.execute(
        select(EscalationChain).where(EscalationChain.id == chain_id, cond)
    )
    chain = result.scalar_one_or_none()
    if chain is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Escalation chain not found")
    return chain


@router.post("/projects/{project_id}/escalation-chains", response_model=EscalationChainResponse, status_code=201)
async def create_escalation_chain(
    project_id: uuid.UUID,
    data: EscalationChainCreate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    chain = EscalationChain(
        project_id=project_id,
        name=data.name,
        description=data.description,
        definition=data.definition,
        steps=data.steps,
        is_active=True,
    )
    db.add(chain)
    await db.flush()
    return chain


@router.get("/projects/{project_id}/escalation-chains", response_model=list[EscalationChainResponse])
async def list_escalation_chains(
    project_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(EscalationChain).where(
            or_(EscalationChain.project_id == project_id, EscalationChain.project_id.is_(None)),
            EscalationChain.is_active.is_(True),
        )
    )
    return list(result.scalars().all())


@router.get("/projects/{project_id}/escalation-chains/{chain_id}", response_model=EscalationChainResponse)
async def get_escalation_chain(
    project_id: uuid.UUID,
    chain_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _get_chain_or_404(db, chain_id, project_id)


@router.put("/projects/{project_id}/escalation-chains/{chain_id}", response_model=EscalationChainResponse)
async def update_escalation_chain(
    project_id: uuid.UUID,
    chain_id: uuid.UUID,
    data: EscalationChainCreate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    chain = await _get_chain_or_404(db, chain_id, project_id, include_global=False)
    chain.name = data.name
    chain.description = data.description
    chain.definition = data.definition
    chain.steps = data.steps
    await db.flush()
    return chain
