from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.orchestration import (
    GOAL_STATUS_VALUES,
    GOAL_WEIGHT_VALUES,
    OrchestrationGoal,
    OrchestrationRun,
    RUN_STATUS_VALUES,
)
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

# Spec 5.3: budget "on the order of 2-3 KB of text", configurable. Constants +
# build() kwargs, same pattern as DECISION_CONTEXT_EVENT_LIMIT (not Settings).
PREFACE_BUDGET_CHARS = 3000
PREFACE_MAX_WARNINGS = 10
PREFACE_MAX_DECISIONS = 5
PREFACE_MAX_BLOCKERS = 10
PREFACE_MAX_SKIPPED = 10

_OBJECTIVE_LIMIT = 300
_SECTION_EXCERPT_LIMIT = 300
_CONSTRAINTS_LIMIT = 400
_INTRODUCTION_LIMIT = 500
_MESSAGE_LIMIT = 160
_TITLE_LIMIT = 120
_TOC_TITLE_LIMIT = 80
_ALWAYS_LOADED_SUMMARY_LIMIT = 200
_ATTRIBUTION_LIMIT = 80

# Sections with dedicated preface fields; kept out of always_loaded excerpts
# so the same text is not paid for twice inside the budget. Still in toc.
_DEDICATED_SECTION_KEYS = frozenset({"introduction", "manager_authority", "team_hierarchy"})


def _trunc(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _normalize_dt(dt: datetime | None) -> datetime | None:
    """Convert tz-aware datetimes to naive UTC for safe sorting.

    Handles mixed-session scenarios where some objects are identity-resident
    (tz-aware) and others are reloaded (naive). Returns None if dt is None,
    converts tz-aware to naive UTC, and returns naive datetimes unchanged.
    """
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def preface_size(preface: dict) -> int:
    """Deterministic budget measure: compact JSON, no ASCII escaping."""
    return len(json.dumps(preface, ensure_ascii=False, separators=(",", ":"), default=str))


# Budget eviction order, least critical first (spec 5.3 "oldest-first
# eviction", order beyond warnings/decisions is ours — see plan Deviation 4).
# toc/always_loaded/objective_notes drop the tail (highest toc_order or newest notes) to keep the book front;
# time-ordered lists drop the oldest entry. Warnings after decisions (risk
# visibility is the point of spec 11); skipped_processes and blockers last
# (skipped_processes is diagnostic, blockers gate the tick but are already
# capped to PREFACE_MAX_BLOCKERS). current_process is last among scalars —
# derived tick-phase info, safe to drop once introduction/hierarchy/manager/
# constraints are already gone.
_EVICT_TAIL_FIRST = ("objective_notes", "toc", "always_loaded")
_EVICT_OLDEST_FIRST = ("recent_decisions", "active_warnings", "open_blockers", "skipped_processes")
_SCALAR_FALLBACK = ("introduction", "hierarchy", "manager", "constraints", "current_process")

# Irreducible skeleton for the size floor. Every evictable field is at its
# emptiest (lists [], optional scalars None, objective None — the last-resort
# loop below can shrink it all the way to None). goal_status/goal_weight/
# run_status are NOT in _SCALAR_FALLBACK — real prefaces always carry a real
# (non-null) value for these, so the floor must budget for the worst case,
# not pretend they're None. Using each enum's longest member keeps this a
# true upper bound without hardcoding string literals that could drift from
# the model.
_MIN_SKELETON: dict = {
    "objective": None,
    "objective_notes": [],
    "goal_status": max(GOAL_STATUS_VALUES, key=len),
    "goal_weight": max(GOAL_WEIGHT_VALUES, key=len),
    "run_status": max(RUN_STATUS_VALUES, key=len),
    "current_process": None,
    "manager": None,
    "hierarchy": None,
    "constraints": None,
    "active_warnings": [],
    "recent_decisions": [],
    "open_blockers": [],
    "skipped_processes": [],
    "introduction": None,
    "always_loaded": [],
    "toc": [],
}
PREFACE_MIN_CHARS = preface_size(_MIN_SKELETON)


def _enforce_budget(preface: dict, budget_chars: int) -> dict:
    if budget_chars < PREFACE_MIN_CHARS:
        raise ValueError(
            f"budget_chars={budget_chars} is below PREFACE_MIN_CHARS={PREFACE_MIN_CHARS}; "
            "the builder cannot honor a budget smaller than the irreducible skeleton"
        )
    for field in _EVICT_TAIL_FIRST:
        while preface[field] and preface_size(preface) > budget_chars:
            preface[field].pop()
    for field in _EVICT_OLDEST_FIRST:
        while preface[field] and preface_size(preface) > budget_chars:
            preface[field].pop(0)
    for field in _SCALAR_FALLBACK:
        if preface_size(preface) <= budget_chars:
            break
        preface[field] = None
    # Last resort (only reachable when budget_chars is close to
    # PREFACE_MIN_CHARS and objective is long): shrink objective itself.
    # PREFACE_MIN_CHARS already accounts for objective=None, so this loop
    # is guaranteed to terminate at or above budget_chars.
    while preface["objective"] and preface_size(preface) > budget_chars:
        preface["objective"] = preface["objective"][:-1] or None
    return preface


class OrchestrationMemoryPrefaceBuilder:
    """Always-loaded orchestrator memory preface (Spec 5.3).

    Deterministic by construction: fixed field order, per-field truncation,
    oldest-first eviction under the size budget. request_decision dedups on
    input_snapshot equality, so identical DB state must yield identical dicts.
    Orchestrator-only (spec 5): wired into the tick decision context, never
    into agent-facing surfaces.
    """

    async def build(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun | None = None,
        *,
        section_meta: Sequence[Any] | None = None,
        budget_chars: int = PREFACE_BUDGET_CHARS,
        max_warnings: int = PREFACE_MAX_WARNINGS,
        max_decisions: int = PREFACE_MAX_DECISIONS,
    ) -> dict:
        if max_warnings < 1:
            raise ValueError(f"max_warnings must be >= 1, got {max_warnings}")
        if max_decisions < 1:
            raise ValueError(f"max_decisions must be >= 1, got {max_decisions}")
        memory_service = OrchestrationMemoryService()
        process_runs = await OrchestrationProcessService().list_process_runs(db, goal.id)
        warnings = await OrchestrationWarningService().list_warnings(db, goal.id, active_only=True)
        decisions = await OrchestrationAuthorityDecisionService().list_decisions(
            db, goal.id, status="answered"
        )
        # Metadata only — no section bodies loaded here (spec 5.4: bodies can
        # be arbitrarily long; the Phase 4 check is "omitted by default").
        if section_meta is None:
            section_meta = await memory_service.list_section_metadata(db, goal.project_id, goal.id)

        active_process_runs = [
            r
            for r in process_runs
            if r.superseded_by_id is None and r.status in ("running", "waiting_decision")
        ]
        current_process = None
        if active_process_runs:
            # ponytail: process_runs is pre-sorted by DB query (started_at, created_at, id asc),
            # so active_process_runs preserves that order; take the last (most recent).
            latest = active_process_runs[-1]
            current_process = {"process_type": latest.process_type, "status": latest.status}

        skipped_processes = [
            {
                "process_type": r.process_type,
                "skipped_by": _trunc(r.skipped_by, _ATTRIBUTION_LIMIT),
            }
            for r in process_runs
            if r.superseded_by_id is None and r.status == "skipped"
        ][-PREFACE_MAX_SKIPPED:]

        # spec 5.3: "recent human or manager decisions" — team_lead/agent
        # decisions are excluded before the cap so they can never displace a
        # human/manager record.
        human_or_manager_decisions = [d for d in decisions if d.authority in ("human", "manager")]
        # list order is asked_at asc; recency for the preface is decided_at
        # (sorted() is stable, so the id-tiebroken query order breaks ties).
        # _normalize_dt handles mixed tz-aware/naive datetimes from mixed-session scenarios.
        recent_decisions = sorted(
            human_or_manager_decisions, key=lambda d: _normalize_dt(d.decided_at or d.asked_at)
        )[-max_decisions:]

        meta_by_key = {s.section_key: s for s in section_meta}

        async def _excerpt(section_key: str, limit: int) -> str | None:
            meta = meta_by_key.get(section_key)
            if meta is None:
                return None
            # Truncate the summary first, then check if non-empty. This handles the
            # case where summary is whitespace-only (e.g., "   ") which is truthy but
            # should fall through to the body-fetch path after stripping.
            truncated_summary = _trunc(meta.summary, limit)
            if truncated_summary:
                return truncated_summary
            if hasattr(meta, "body"):
                return _trunc(meta.body, limit)
            # No summary cached or it's empty after truncation: this is the only path
            # that loads a body, and only for the one section actually being excerpted.
            section = await memory_service.get_section(db, goal.project_id, goal.id, section_key)
            return _trunc(section.body if section else None, limit)

        blockers = list(run.active_blockers or []) if run is not None else []
        constraints = None
        if goal.constraints:
            constraints = _trunc(
                json.dumps(goal.constraints, ensure_ascii=False, sort_keys=True),
                _CONSTRAINTS_LIMIT,
            )

        always_loaded = []
        for s in section_meta:
            if s.always_load and s.section_key not in _DEDICATED_SECTION_KEYS:
                always_loaded.append(
                    {
                        "section_key": s.section_key,
                        "summary": await _excerpt(s.section_key, _ALWAYS_LOADED_SUMMARY_LIMIT),
                    }
                )

        objective_notes = (goal.orchestrator_context or {}).get("objective_notes", [])
        preface = {
            "objective": _trunc(goal.objective, _OBJECTIVE_LIMIT),
            "objective_notes": [_trunc(note, _OBJECTIVE_LIMIT) for note in objective_notes if note],
            "goal_status": goal.status,
            "goal_weight": goal.weight,
            "run_status": run.status if run is not None else None,
            "current_process": current_process,
            "manager": await _excerpt("manager_authority", _SECTION_EXCERPT_LIMIT),
            "hierarchy": await _excerpt("team_hierarchy", _SECTION_EXCERPT_LIMIT),
            "constraints": constraints,
            "active_warnings": [
                {
                    "severity": w.severity,
                    "warning_type": w.warning_type,
                    "message": _trunc(w.message, _MESSAGE_LIMIT),
                    "acknowledged": w.acknowledged_at is not None,
                }
                for w in warnings[-max_warnings:]
            ],
            "recent_decisions": [
                {
                    "title": _trunc(d.title, _TITLE_LIMIT),
                    "authority": d.authority,
                    "selected_option": _trunc(d.selected_option, _TITLE_LIMIT),
                    "overrides_recommendation": d.overrides_recommendation,
                }
                for d in recent_decisions
            ],
            "open_blockers": [
                _trunc(
                    json.dumps(b, ensure_ascii=False, sort_keys=True, default=str)
                    if isinstance(b, dict)
                    else str(b),
                    _MESSAGE_LIMIT,
                )
                for b in blockers[-PREFACE_MAX_BLOCKERS:]
            ],
            "skipped_processes": skipped_processes,
            "introduction": await _excerpt("introduction", _INTRODUCTION_LIMIT),
            "always_loaded": always_loaded,
            "toc": [
                {"section_key": s.section_key, "title": _trunc(s.title, _TOC_TITLE_LIMIT)}
                for s in section_meta
            ],
        }
        return _enforce_budget(preface, budget_chars)
