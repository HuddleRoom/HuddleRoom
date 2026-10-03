"""Bounded, READ-ONLY project-context assembler for "Ask the orchestrator" (Decision 3 / T2.1).

Assembles a summarized snapshot of project state for a project-scoped advisor
LLM call. Pure reads only (SELECTs) — never mutates. Every list has an
explicit cap; every free-text field is truncated. If a source is unavailable
or empty, its list is empty rather than failing the whole assembler.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.meeting import Meeting, MeetingDecision
from huddleroom.models.orchestration import OrchestrationDecision, OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.services.orchestration_memory_preface import OrchestrationMemoryPrefaceBuilder
from huddleroom.services.orchestration_service import OrchestrationService

# Explicit numeric caps (Decision 3 / T2.1).
OPEN_GOALS_LIMIT = 5
RECENT_DECISIONS_LIMIT = 10
RECENT_MEETING_DECISIONS_LIMIT = 5
GOAL_PREFACE_LIMIT = 3
RECENT_EVENTS_LIMIT = 10

_TEXT_TRUNCATE_LIMIT = 200

_orchestration_service = OrchestrationService()
_preface_builder = OrchestrationMemoryPrefaceBuilder()


def _trunc(value: str | None, limit: int = _TEXT_TRUNCATE_LIMIT) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _decision_summary(decision: OrchestrationDecision) -> str | None:
    if decision.reason:
        return _trunc(decision.reason)
    if decision.parsed_decision:
        return _trunc(str(decision.parsed_decision))
    return None


def _authority_decision_summary(decision: OrchestrationAuthorityDecision) -> str:
    parts = [decision.title]
    if decision.selected_option:
        parts.append(f"selected: {decision.selected_option}")
    if decision.reason:
        parts.append(decision.reason)
    return _trunc(" — ".join(parts)) or ""


def _utc_timestamp(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


async def _fetch_open_goals(db: AsyncSession, project_id: uuid.UUID) -> list[OrchestrationGoal]:
    # Active goals first, then most-recently-updated across all statuses.
    result = await db.execute(
        select(OrchestrationGoal)
        .where(OrchestrationGoal.project_id == project_id)
        .order_by(
            case((OrchestrationGoal.status == "active", 0), else_=1),
            OrchestrationGoal.updated_at.desc(),
            OrchestrationGoal.id.desc(),
        )
        .limit(OPEN_GOALS_LIMIT)
    )
    return list(result.scalars().all())


async def _build_open_goals(db: AsyncSession, project_id: uuid.UUID) -> list[dict]:
    goals = await _fetch_open_goals(db, project_id)
    out = []
    for goal in goals:
        # Best-effort "current step": latest run's phase. Not a dedicated
        # column on OrchestrationGoal (see models/orchestration.py).
        run = await _orchestration_service.get_run_for_goal(db, project_id, goal.id)
        out.append(
            {
                "id": str(goal.id),
                "objective": _trunc(goal.objective),
                "status": goal.status,
                "weight": goal.weight,
                "current_step": run.phase if run is not None else None,
            }
        )
    return out


async def _build_recent_decisions(db: AsyncSession, project_id: uuid.UUID) -> list[dict]:
    execution_result = await db.execute(
        select(OrchestrationDecision, OrchestrationRun.goal_id)
        .join(OrchestrationRun, OrchestrationRun.id == OrchestrationDecision.run_id)
        .join(OrchestrationGoal, OrchestrationGoal.id == OrchestrationRun.goal_id)
        .where(OrchestrationGoal.project_id == project_id)
        .order_by(OrchestrationDecision.created_at.desc(), OrchestrationDecision.id.desc())
        .limit(RECENT_DECISIONS_LIMIT)
    )
    # pylint: disable-next=assignment-from-no-return
    authority_occurred_at = func.coalesce(
        OrchestrationAuthorityDecision.decided_at, OrchestrationAuthorityDecision.created_at
    )
    authority_result = await db.execute(
        select(OrchestrationAuthorityDecision, authority_occurred_at.label("occurred_at"))
        .join(OrchestrationGoal, OrchestrationGoal.id == OrchestrationAuthorityDecision.goal_id)
        .where(
            OrchestrationGoal.project_id == project_id,
            OrchestrationAuthorityDecision.status == "answered",
        )
        .order_by(authority_occurred_at.desc(), OrchestrationAuthorityDecision.id.desc())
        .limit(RECENT_DECISIONS_LIMIT)
    )
    decisions = [
        (
            decision.created_at,
            decision.id,
            {
                "id": str(decision.id),
                "goal_id": str(goal_id),
                "decision_type": decision.decision_type,
                "summary": _decision_summary(decision),
                "created_at": decision.created_at.isoformat(),
            },
        )
        for decision, goal_id in execution_result.all()
    ]
    decisions.extend(
        (
            occurred_at,
            decision.id,
            {
                "id": str(decision.id),
                "goal_id": str(decision.goal_id),
                "decision_type": decision.decision_key,
                "summary": _authority_decision_summary(decision),
                "created_at": occurred_at.isoformat(),
            },
        )
        for decision, occurred_at in authority_result.all()
    )
    decisions.sort(key=lambda item: (_utc_timestamp(item[0]), str(item[1])), reverse=True)
    return [decision for _, _, decision in decisions[:RECENT_DECISIONS_LIMIT]]


async def _build_recent_meeting_decisions(db: AsyncSession, project_id: uuid.UUID) -> list[dict]:
    result = await db.execute(
        select(MeetingDecision)
        .join(Meeting, Meeting.id == MeetingDecision.meeting_id)
        .where(Meeting.project_id == project_id, Meeting.status == "concluded")
        .order_by(Meeting.concluded_at.desc(), MeetingDecision.created_at.desc(), MeetingDecision.id.desc())
        .limit(RECENT_MEETING_DECISIONS_LIMIT)
    )
    return [
        {
            "meeting_id": str(decision.meeting_id),
            "question": _trunc(decision.question),
            "chosen": _trunc(decision.chosen_option),
            "rationale": _trunc(decision.rationale),
        }
        for decision in result.scalars().all()
    ]


async def _build_goal_prefaces(db: AsyncSession, project_id: uuid.UUID) -> list[dict]:
    result = await db.execute(
        select(OrchestrationGoal)
        .where(OrchestrationGoal.project_id == project_id, OrchestrationGoal.status == "active")
        .order_by(OrchestrationGoal.updated_at.desc(), OrchestrationGoal.id.desc())
        .limit(GOAL_PREFACE_LIMIT)
    )
    goals = list(result.scalars().all())
    out = []
    for goal in goals:
        run = await _orchestration_service.get_run_for_goal(db, project_id, goal.id)
        # Reuse the existing bounded memory preface (already capped to
        # PREFACE_BUDGET_CHARS with its own per-field truncation) rather than
        # re-deriving a summary here.
        preface = await _preface_builder.build(db, goal, run)
        out.append({"goal_id": str(goal.id), "preface": preface})
    return out


async def _build_recent_events(db: AsyncSession, project_id: uuid.UUID) -> list[dict]:
    # ponytail: EventLog has no free-text summary column; use event_type as
    # the summary (mirrors huddleroom/routers/events.py list query shape).
    from huddleroom.models.event_log import EventLog

    result = await db.execute(
        select(EventLog)
        .where(EventLog.project_id == project_id)
        .order_by(EventLog.emitted_at.desc(), EventLog.id.desc())
        .limit(RECENT_EVENTS_LIMIT)
    )
    return [
        {
            "type": event.event_type,
            "summary": _trunc(event.event_type),
            "created_at": event.emitted_at.isoformat(),
        }
        for event in result.scalars().all()
    ]


async def build_advisor_context(db: AsyncSession, project_id: uuid.UUID) -> dict:
    """Assemble the bounded, read-only project-context snapshot for the advisor LLM call.

    Every source is fetched independently; an empty/unavailable source yields
    an empty list rather than failing the whole assembler.
    """
    return {
        "open_goals": await _build_open_goals(db, project_id),
        "recent_decisions": await _build_recent_decisions(db, project_id),
        "recent_meeting_decisions": await _build_recent_meeting_decisions(db, project_id),
        "goal_prefaces": await _build_goal_prefaces(db, project_id),
        "recent_events": await _build_recent_events(db, project_id),
    }
