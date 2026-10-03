from __future__ import annotations

import copy
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.agent import Agent
from huddleroom.services.orchestration_agent_definition_analyzer import redact_semantic_payload
from huddleroom.models.orchestration_process import OrchestrationAgentReview
from huddleroom.services.orchestration_linkage import check_goal_linkage

# Snapshot field list is deterministic and code-owned (spec 9.5): the agent
# columns the review inspects. Personality, communication style, and safety
# boundaries are derived from system_prompt and description — they are not
# first-class agent columns. config carries the recommended review keys
# (config.temperature, config.reasoning_effort, config.tools) when present.
SNAPSHOT_FIELDS = (
    "name",
    "role",
    "description",
    "system_prompt",
    "provider",
    "model",
    "adapter_type",
    "cli_runtime",
    "capabilities",
    "config",
    "is_active",
)

# Fingerprint-only view of SNAPSHOT_FIELDS: runtime/routing fields (provider,
# model, adapter_type, cli_runtime, config) don't change what the review
# should conclude, so they're excluded from staleness hashing to avoid
# false-positive "definitions changed" suggestions on routing changes alone.
# definition_snapshot (audit trail) still uses the full SNAPSHOT_FIELDS.
FINGERPRINT_SNAPSHOT_FIELDS = tuple(
    f for f in SNAPSHOT_FIELDS
    if f not in {"provider", "model", "adapter_type", "cli_runtime", "config"}
)


def _validate_str_list(name: str, value: list | None) -> list:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list, got {type(value).__name__}")
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(f"{name} entries must be non-empty strings: {entry!r}")
    return value


class OrchestrationAgentReviewService:
    """Agent-definition review records (Spec 9.5, 15.4, Phase 3).

    Persistence only: the automatic review logic that judges fit and raises
    warnings is Phase 7. definition_snapshot is built here from the live
    Agent row, never caller-supplied, so a review always shows exactly what
    was reviewed (Phase 3 Deviation 3). Creation is idempotent per
    (goal_id, agent_id, source_process_run_id) when a process run is
    linked — a retried tick step must not duplicate the review; standalone
    reviews (no process run) are deliberate history and are not deduped
    (Phase 3 Deviation 5). The dedup is check-then-insert, single-writer-
    safe only — same ruling as create_warning (orchestrator tick is the
    only writer; no REST surface).

    approved_for_work_functions must be a subset of
    proposed_work_functions (Phase 3 Deviation 6): approving an agent for
    a function the review never proposed is incoherent as a record. Who
    approved and why is an authority decision (Phase 7/8), not a column
    here.
    """

    def _build_snapshot(self, agent: Agent) -> dict:
        # deepcopy, not a shallow dict comprehension: capabilities (list) and
        # config (dict) are mutable JSON columns on the live Agent instance.
        # A shallow copy would still share those nested objects, so an
        # in-place agent.capabilities.append(...) or agent.config[...] = ...
        # after the review is created would silently mutate the "frozen"
        # snapshot in memory (reviewer-flagged gap; a scalar-only field like
        # role can't expose this, capabilities/config can).
        return redact_semantic_payload(
            {field: copy.deepcopy(getattr(agent, field)) for field in SNAPSHOT_FIELDS}
        )

    @staticmethod
    def _check_approved_subset(approved: list, proposed: list) -> None:
        extra = set(approved) - set(proposed)
        if extra:
            raise ValueError(f"approved work functions not among proposed: {sorted(extra)}")

    async def create_review(
        self,
        db: AsyncSession,
        goal_id: uuid.UUID,
        *,
        agent_id: uuid.UUID,
        fit_summary: str,
        review_context: str | None = None,
        proposed_work_functions: list | None = None,
        strengths: list | None = None,
        risks: list | None = None,
        recommended_changes: list | None = None,
        approved_for_work_functions: list | None = None,
        run_id: uuid.UUID | None = None,
        source_process_run_id: uuid.UUID | None = None,
        definition_snapshot: dict | None = None,
    ) -> OrchestrationAgentReview:
        if not fit_summary or not fit_summary.strip():
            raise ValueError("fit_summary must not be empty")
        proposed = _validate_str_list("proposed_work_functions", proposed_work_functions)
        strengths = _validate_str_list("strengths", strengths)
        risks = _validate_str_list("risks", risks)
        recommended = _validate_str_list("recommended_changes", recommended_changes)
        approved = _validate_str_list("approved_for_work_functions", approved_for_work_functions)
        self._check_approved_subset(approved, proposed)
        agent = await db.get(Agent, agent_id)
        if agent is None:
            raise ValueError(f"agent {agent_id} does not exist")
        await check_goal_linkage(
            db, goal_id, run_id=run_id, source_process_run_id=source_process_run_id
        )
        if source_process_run_id is not None:
            result = await db.execute(
                select(OrchestrationAgentReview).where(
                    OrchestrationAgentReview.goal_id == goal_id,
                    OrchestrationAgentReview.agent_id == agent_id,
                    OrchestrationAgentReview.source_process_run_id == source_process_run_id,
                )
            )
            existing = result.scalars().first()
            if existing is not None:
                return existing
        review = OrchestrationAgentReview(
            goal_id=goal_id,
            run_id=run_id,
            agent_id=agent_id,
            source_process_run_id=source_process_run_id,
            review_context=review_context,
            proposed_work_functions=proposed,
            definition_snapshot=redact_semantic_payload(definition_snapshot) if definition_snapshot is not None else self._build_snapshot(agent),
            fit_summary=fit_summary,
            strengths=strengths,
            risks=risks,
            recommended_changes=recommended,
            approved_for_work_functions=approved,
        )
        db.add(review)
        await db.flush()
        return review

    async def approve_work_functions(
        self,
        db: AsyncSession,
        review: OrchestrationAgentReview,
        *,
        work_functions: list,
    ) -> OrchestrationAgentReview:
        approved = _validate_str_list("work_functions", work_functions)
        if not approved:
            raise ValueError("work_functions must not be empty")
        self._check_approved_subset(approved, review.proposed_work_functions or [])
        # Full-replace, matching Phase 1 PUT semantics; single writer
        # (orchestrator tick), so no conditional-update guard is needed yet.
        review.approved_for_work_functions = approved
        await db.flush()
        return review

    async def list_reviews(
        self,
        db: AsyncSession,
        goal_id: uuid.UUID,
        *,
        agent_id: uuid.UUID | None = None,
    ) -> list[OrchestrationAgentReview]:
        query = select(OrchestrationAgentReview).where(
            OrchestrationAgentReview.goal_id == goal_id
        )
        if agent_id is not None:
            query = query.where(OrchestrationAgentReview.agent_id == agent_id)
        query = query.order_by(
            OrchestrationAgentReview.created_at.asc(), OrchestrationAgentReview.id.asc()
        )
        result = await db.execute(query)
        return list(result.scalars().all())
