"""Shared goal-linkage validation for baseline orchestration services.

`run_id`, `source_process_run_id`, `related_gate_id`, and
`related_action_id`, and `related_authority_decision_id` are each individually
valid foreign keys even when
they belong to a different goal than the one being written to — an FK only
constrains existence, not goal membership (Phase 2 Spec Deviation 11).
This helper resolves each chain (run.goal_id, gate.run_id -> run.goal_id,
action.run_id -> run.goal_id, process_run.goal_id) and raises ValueError
on any mismatch. Consolidated from the per-service copies (Phase 2
follow-up); every kwarg defaults to None so narrow callers (process
service: run_id only) and wide callers (warning/authority/agent-review
services) share one implementation.
"""
from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGate, OrchestrationRun
from huddleroom.models.orchestration_process import (
    OrchestrationAgentReview,
    OrchestrationAuthorityDecision,
    OrchestrationProcessRun,
)


async def check_goal_linkage(
    db: AsyncSession,
    goal_id: uuid.UUID,
    *,
    run_id: uuid.UUID | None = None,
    source_process_run_id: uuid.UUID | None = None,
    related_gate_id: uuid.UUID | None = None,
    related_action_id: uuid.UUID | None = None,
    source_agent_review_id: uuid.UUID | None = None,
    related_authority_decision_id: uuid.UUID | None = None,
) -> None:
    if run_id is not None:
        run = await db.get(OrchestrationRun, run_id)
        if run is None or run.goal_id != goal_id:
            raise ValueError(f"run {run_id} does not belong to goal {goal_id}")
    if source_process_run_id is not None:
        process_run = await db.get(OrchestrationProcessRun, source_process_run_id)
        if process_run is None or process_run.goal_id != goal_id:
            raise ValueError(
                f"source_process_run {source_process_run_id} does not belong to goal {goal_id}"
            )
    if related_gate_id is not None:
        gate = await db.get(OrchestrationGate, related_gate_id)
        gate_run = await db.get(OrchestrationRun, gate.run_id) if gate else None
        if gate is None or gate_run is None or gate_run.goal_id != goal_id:
            raise ValueError(f"related_gate {related_gate_id} does not belong to goal {goal_id}")
    if related_action_id is not None:
        action = await db.get(OrchestrationAction, related_action_id)
        action_run = await db.get(OrchestrationRun, action.run_id) if action else None
        if action is None or action_run is None or action_run.goal_id != goal_id:
            raise ValueError(
                f"related_action {related_action_id} does not belong to goal {goal_id}"
            )
    if source_agent_review_id is not None:
        review = await db.get(OrchestrationAgentReview, source_agent_review_id)
        if review is None or review.goal_id != goal_id:
            raise ValueError(
                f"source_agent_review {source_agent_review_id} does not belong to goal {goal_id}"
            )
    if related_authority_decision_id is not None:
        decision = await db.get(OrchestrationAuthorityDecision, related_authority_decision_id)
        if decision is None or decision.goal_id != goal_id:
            raise ValueError(
                f"related_authority_decision {related_authority_decision_id} does not belong to goal {goal_id}"
            )
