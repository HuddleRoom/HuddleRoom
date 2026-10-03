from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user, ensure_project_exists
from huddleroom.models.routing_rule import RoutingRule
from huddleroom.models.user import User
from huddleroom.schemas.common import CursorPage
from huddleroom.schemas.routing_rule import RoutingRuleCreate, RoutingRuleResponse, RoutingRuleUpdate
from huddleroom.services.guard_evaluator import GuardEvaluator
from huddleroom.services.routing_rule_service import RoutingRuleService

router = APIRouter()


async def _get_rule_or_404(db: AsyncSession, project_id: uuid.UUID, rule_id: uuid.UUID) -> RoutingRule:
    result = await db.execute(
        select(RoutingRule).where(RoutingRule.id == rule_id, RoutingRule.project_id == project_id)
    )
    rule = result.scalar_one_or_none()
    if rule is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rule not found")
    return rule


@router.get("/projects/{project_id}/rules", response_model=CursorPage[RoutingRuleResponse])
async def list_rules(
    project_id: uuid.UUID,
    enabled: bool | None = None,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    conditions = [RoutingRule.project_id == project_id]
    if enabled is not None:
        conditions.append(RoutingRule.enabled == enabled)
    if cursor is not None:
        parts = cursor.split("__", 1)
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid cursor")
        try:
            cursor_priority = int(parts[0])
            cursor_id = uuid.UUID(parts[1])
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="Invalid cursor")
        conditions.append(
            or_(
                RoutingRule.priority > cursor_priority,
                and_(RoutingRule.priority == cursor_priority, RoutingRule.id > cursor_id),
            )
        )
    result = await db.execute(
        select(RoutingRule)
        .where(*conditions)
        .order_by(RoutingRule.priority.asc(), RoutingRule.id.asc())
        .limit(limit + 1)
    )
    items = list(result.scalars().all())
    next_cursor = None
    if len(items) > limit:
        items = items[:limit]
        last = items[-1]
        next_cursor = f"{last.priority}__{last.id}"
    return CursorPage(items=items, next_cursor=next_cursor)


@router.post("/projects/{project_id}/rules", response_model=RoutingRuleResponse, status_code=201)
async def create_rule(
    project_id: uuid.UUID,
    data: RoutingRuleCreate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    rule = RoutingRule(
        project_id=project_id,
        name=data.name,
        description=data.description,
        priority=data.priority,
        on_event=data.on_event,
        conditions=data.conditions,
        actions=data.actions,
        enabled=data.enabled,
    )
    db.add(rule)
    await db.flush()
    return rule


@router.get("/projects/{project_id}/rules/{rule_id}", response_model=RoutingRuleResponse)
async def get_rule(
    project_id: uuid.UUID,
    rule_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _get_rule_or_404(db, project_id, rule_id)


@router.patch("/projects/{project_id}/rules/{rule_id}", response_model=RoutingRuleResponse)
async def update_rule(
    project_id: uuid.UUID,
    rule_id: uuid.UUID,
    data: RoutingRuleUpdate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rule = await _get_rule_or_404(db, project_id, rule_id)
    update_data = data.model_dump(exclude_unset=True)
    return await RoutingRuleService().apply_update(db, rule, update_data)


@router.delete("/projects/{project_id}/rules/{rule_id}", status_code=204)
async def delete_rule(
    project_id: uuid.UUID,
    rule_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rule = await _get_rule_or_404(db, project_id, rule_id)
    await db.delete(rule)
    await db.flush()
    return None


class DryRunRequest(BaseModel):
    conditions: dict
    event_payload: dict


@router.post("/projects/{project_id}/rules/dry-run")
async def dry_run_rule(
    project_id: uuid.UUID,
    data: DryRunRequest,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    matched = GuardEvaluator().evaluate(data.conditions, data.event_payload)
    return {"matched": matched}


class ReorderRequest(BaseModel):
    rule_ids: list[uuid.UUID]


@router.post("/projects/{project_id}/rules/reorder", status_code=200)
async def reorder_rules(
    project_id: uuid.UUID,
    data: ReorderRequest,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    if not data.rule_ids:
        raise HTTPException(status_code=400, detail="rule_ids must not be empty")
    if len(data.rule_ids) != len(set(data.rule_ids)):
        raise HTTPException(status_code=400, detail="rule_ids must not contain duplicates")
    result = await db.execute(
        select(RoutingRule).where(RoutingRule.project_id == project_id)
    )
    project_rules = {r.id: r for r in result.scalars().all()}
    if set(data.rule_ids) != set(project_rules.keys()):
        raise HTTPException(
            status_code=400,
            detail="rule_ids must contain exactly all rules in the project",
        )
    for priority, rid in enumerate(data.rule_ids):
        project_rules[rid].priority = priority
    await db.flush()
    return {"reordered": len(data.rule_ids)}
