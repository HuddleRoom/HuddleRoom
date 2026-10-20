"""Read-only, project-scoped reference projection for Needs You surfaces."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.agent import Agent
from huddleroom.models.artifact import Artifact
from huddleroom.models.meeting import Meeting
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationGate,
    OrchestrationGoal,
    OrchestrationRun,
)
from huddleroom.models.orchestration_process import (
    OrchestrationAuthorityDecision,
    OrchestrationProcessRun,
    OrchestrationWarning,
)
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import (
    NeedsYouReference,
    NeedsYouReferenceKind,
    OrchestrationAuthorityDecisionResponse,
    PlanReviewRecord,
    OrchestrationWarningResponse,
)

_QUERY_CHUNK_SIZE = 500
_TARGET_KINDS = frozenset(("task", "meeting", "agent", "goal", "gate", "process", "action", "artifact"))
_RECOVERY_EVIDENCE_KINDS = frozenset(("action", "task", "gate", "process", "agent"))
_PROCESS_LABELS = {
    "goal_definition": "Goal definition",
    "manager_selection": "Manager selection",
    "agent_definition_review": "Reviewing agent definitions",
    "team_hierarchy": "Team hierarchy",
    "effectiveness_review": "Effectiveness review",
    "goal_closeout": "Goal closeout",
}


@dataclass(frozen=True)
class NeedsYouReferenceRequest:
    kind: NeedsYouReferenceKind
    id: uuid.UUID
    goal_id: uuid.UUID | None = None
    run_id: uuid.UUID | None = None


def _chunks(values: Sequence[uuid.UUID]) -> Iterable[Sequence[uuid.UUID]]:
    for offset in range(0, len(values), _QUERY_CHUNK_SIZE):
        yield values[offset:offset + _QUERY_CHUNK_SIZE]


def _uuid(value: object) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        return None


def _criterion_label(goal: OrchestrationGoal, gate: OrchestrationGate) -> str:
    criterion = next((
        item for item in (goal.success_criteria or [])
        if isinstance(item, dict) and item.get("key") == gate.success_criterion_key
    ), None)
    description = criterion.get("description") if criterion else None
    return description.strip() if isinstance(description, str) and description.strip() else gate.success_criterion_key


def _dedupe_requests(requests: Sequence[NeedsYouReferenceRequest]) -> list[NeedsYouReferenceRequest]:
    return list({_reference_key(request): request for request in requests}.values())


def _reference_key(request: NeedsYouReferenceRequest) -> tuple[str, uuid.UUID, uuid.UUID | None]:
    return request.kind, request.id, request.run_id if request.kind == "artifact" else None


def _action_related_requests(
    action: OrchestrationAction, goal_id: uuid.UUID,
) -> list[NeedsYouReferenceRequest]:
    related: list[NeedsYouReferenceRequest] = []
    if action.target_type in _TARGET_KINDS and action.target_id is not None:
        related.append(NeedsYouReferenceRequest(
            action.target_type, action.target_id, goal_id,
            action.run_id if action.target_type == "artifact" else None,
        ))
    return related


async def resolve_needs_you_references(
    db: AsyncSession,
    *,
    project_id: uuid.UUID,
    requests: Sequence[NeedsYouReferenceRequest],
) -> list[NeedsYouReference]:
    """Resolve explicitly typed references without exposing foreign record labels."""
    ordered: dict[tuple[str, uuid.UUID, uuid.UUID | None], NeedsYouReferenceRequest] = {}
    for request in requests:
        ordered.setdefault(_reference_key(request), request)
    labels: dict[tuple[str, uuid.UUID, uuid.UUID | None], tuple[str, uuid.UUID | None]] = {}

    action_requests = [request for request in ordered.values() if request.kind == "action"]
    for chunk in _chunks([request.id for request in action_requests]):
        rows = await db.execute(
            select(OrchestrationAction, OrchestrationRun.goal_id)
            .join(OrchestrationRun, OrchestrationRun.id == OrchestrationAction.run_id)
            .join(OrchestrationGoal, OrchestrationGoal.id == OrchestrationRun.goal_id)
            .where(OrchestrationAction.id.in_(chunk), OrchestrationGoal.project_id == project_id)
        )
        for action, goal_id in rows.all():
            labels[("action", action.id, None)] = (action.action_type, goal_id)
            for related in _action_related_requests(action, goal_id):
                ordered.setdefault(
                    _reference_key(related), related,
                )

    by_kind: dict[str, list[NeedsYouReferenceRequest]] = {}
    for request in ordered.values():
        if request.kind != "action":
            by_kind.setdefault(request.kind, []).append(request)

    for chunk in _chunks([request.id for request in by_kind.get("task", [])]):
        for row in (await db.scalars(select(Task).where(Task.id.in_(chunk), Task.project_id == project_id))).all():
            labels[("task", row.id, None)] = (row.title, None)
    for chunk in _chunks([request.id for request in by_kind.get("meeting", [])]):
        for row in (await db.scalars(
            select(Meeting).where(Meeting.id.in_(chunk), Meeting.project_id == project_id)
        )).all():
            labels[("meeting", row.id, None)] = (row.title, None)
    agent_requests = by_kind.get("agent", [])
    agent_goal_ids = {request.goal_id for request in agent_requests if request.goal_id is not None}
    authorized_goal_ids: set[uuid.UUID] = set()
    for chunk in _chunks(list(agent_goal_ids)):
        authorized_goal_ids.update((await db.scalars(
            select(OrchestrationGoal.id).where(
                OrchestrationGoal.id.in_(chunk), OrchestrationGoal.project_id == project_id,
            )
        )).all())
    authorized_agent_ids = [request.id for request in agent_requests if request.goal_id in authorized_goal_ids]
    for chunk in _chunks(authorized_agent_ids):
        for row in (await db.scalars(select(Agent).where(Agent.id.in_(chunk)))).all():
            labels[("agent", row.id, None)] = (row.name, None)
    for chunk in _chunks([request.id for request in by_kind.get("goal", [])]):
        for row in (await db.scalars(
            select(OrchestrationGoal).where(
                OrchestrationGoal.id.in_(chunk), OrchestrationGoal.project_id == project_id,
            )
        )).all():
            labels[("goal", row.id, None)] = (row.objective, row.id)
    for chunk in _chunks([request.id for request in by_kind.get("process", [])]):
        rows = await db.execute(
            select(OrchestrationProcessRun, OrchestrationGoal.id)
            .join(OrchestrationGoal, OrchestrationGoal.id == OrchestrationProcessRun.goal_id)
            .where(OrchestrationProcessRun.id.in_(chunk), OrchestrationGoal.project_id == project_id)
        )
        for row, goal_id in rows.all():
            labels[("process", row.id, None)] = (_PROCESS_LABELS.get(row.process_type, row.process_type), goal_id)
    for chunk in _chunks([request.id for request in by_kind.get("gate", [])]):
        rows = await db.execute(
            select(OrchestrationGate, OrchestrationGoal)
            .join(OrchestrationRun, OrchestrationRun.id == OrchestrationGate.run_id)
            .join(OrchestrationGoal, OrchestrationGoal.id == OrchestrationRun.goal_id)
            .where(OrchestrationGate.id.in_(chunk), OrchestrationGoal.project_id == project_id)
        )
        for gate, goal in rows.all():
            labels[("gate", gate.id, None)] = (_criterion_label(goal, gate), goal.id)

    for request in by_kind.get("artifact", []):
        if request.goal_id is None or request.run_id is None:
            continue
        artifact = await db.get(Artifact, request.id)
        run = await db.get(OrchestrationRun, request.run_id)
        if artifact is None or artifact.project_id != project_id or run is None or run.goal_id != request.goal_id:
            continue
        raw = (run.plan_state or {}).get("plan_reviews", {}).get(str(artifact.id))
        try:
            record = PlanReviewRecord.model_validate(raw)
        except (TypeError, ValueError):
            continue
        if (record.identity.project_id == project_id and record.identity.goal_id == request.goal_id
                and record.identity.run_id == run.id and record.identity.artifact_id == artifact.id):
            labels[("artifact", artifact.id, run.id)] = (f"Plan candidate v{record.identity.planning_version}", request.goal_id)

    return [
        NeedsYouReference(
            kind=request.kind,
            id=request.id,
            project_id=project_id,
            goal_id=(labels.get(key) or (None, request.goal_id))[1] or request.goal_id,
            label=(labels.get(key) or (None, None))[0],
        )
        for key, request in ordered.items()
    ]


def decision_reference_requests(
    decision: OrchestrationAuthorityDecision,
    response: OrchestrationAuthorityDecisionResponse,
) -> list[NeedsYouReferenceRequest]:
    requests = [NeedsYouReferenceRequest("goal", decision.goal_id, decision.goal_id)]
    for kind, value in (
        ("process", decision.source_process_run_id),
        ("gate", decision.related_gate_id),
        ("action", decision.related_action_id),
        ("agent", decision.authority_agent_id),
    ):
        if value is not None:
            requests.append(NeedsYouReferenceRequest(kind, value, decision.goal_id))
    if decision.run_id is not None and str(decision.decision_key or "").startswith("roadmap_plan:"):
        context = decision.context
        if isinstance(context, str):
            try:
                context = json.loads(context)
            except ValueError:
                context = None
        identity = context.get("plan_review_identity") if isinstance(context, dict) else None
        artifact_id = _uuid(identity.get("artifact_id")) if isinstance(identity, dict) else None
        if artifact_id is not None:
            requests.append(NeedsYouReferenceRequest("artifact", artifact_id, decision.goal_id, decision.run_id))
    if response.recovery is not None:
        if response.recovery.source_action_id is not None:
            requests.append(NeedsYouReferenceRequest("action", response.recovery.source_action_id, decision.goal_id))
        for evidence in response.recovery.evidence or []:
            if evidence.source_kind in _RECOVERY_EVIDENCE_KINDS:
                requests.append(NeedsYouReferenceRequest(evidence.source_kind, evidence.source_id, decision.goal_id))
    return requests


def warning_reference_requests(warning: OrchestrationWarning) -> list[NeedsYouReferenceRequest]:
    requests = [NeedsYouReferenceRequest("goal", warning.goal_id, warning.goal_id)]
    for kind, value in (
        ("process", warning.source_process_run_id),
        ("gate", warning.related_gate_id),
        ("action", warning.related_action_id),
        ("agent", warning.related_agent_id),
    ):
        if value is not None:
            requests.append(NeedsYouReferenceRequest(kind, value, warning.goal_id))
    return requests


async def _expand_action_targets(
    db: AsyncSession,
    *,
    project_id: uuid.UUID,
    per_response: list[list[NeedsYouReferenceRequest]],
) -> None:
    action_ids = {
        request.id for requests in per_response for request in requests if request.kind == "action"
    }
    targets: dict[uuid.UUID, list[NeedsYouReferenceRequest]] = {}
    for chunk in _chunks(list(action_ids)):
        rows = await db.execute(
            select(OrchestrationAction, OrchestrationRun.goal_id)
            .join(OrchestrationRun, OrchestrationRun.id == OrchestrationAction.run_id)
            .join(OrchestrationGoal, OrchestrationGoal.id == OrchestrationRun.goal_id)
            .where(OrchestrationAction.id.in_(chunk), OrchestrationGoal.project_id == project_id)
        )
        for action, goal_id in rows.all():
            targets[action.id] = _action_related_requests(action, goal_id)
    for requests in per_response:
        requests.extend(
            target for request in list(requests)
            if request.kind == "action" and request.id in targets
            for target in targets[request.id]
        )
        requests[:] = _dedupe_requests(requests)


async def attach_decision_references(
    db: AsyncSession,
    *,
    project_id: uuid.UUID,
    decisions: Sequence[OrchestrationAuthorityDecision],
    responses: Sequence[OrchestrationAuthorityDecisionResponse],
) -> list[OrchestrationAuthorityDecisionResponse]:
    per_response = [decision_reference_requests(decision, response) for decision, response in zip(decisions, responses)]
    await _expand_action_targets(db, project_id=project_id, per_response=per_response)
    resolved = await resolve_needs_you_references(
        db, project_id=project_id, requests=[request for requests in per_response for request in requests],
    )
    resolved_by_key = {(reference.kind, reference.id): reference for reference in resolved}
    return [
        response.model_copy(update={
            "needs_you_references": [resolved_by_key[(request.kind, request.id)] for request in requests
                                     if (request.kind, request.id) in resolved_by_key]
        })
        for response, requests in zip(responses, per_response)
    ]


async def attach_warning_references(
    db: AsyncSession,
    *,
    project_id: uuid.UUID,
    warnings: Sequence[OrchestrationWarning],
) -> list[OrchestrationWarningResponse]:
    per_response = [warning_reference_requests(warning) for warning in warnings]
    await _expand_action_targets(db, project_id=project_id, per_response=per_response)
    resolved = await resolve_needs_you_references(
        db, project_id=project_id, requests=[request for requests in per_response for request in requests],
    )
    resolved_by_key = {(reference.kind, reference.id): reference for reference in resolved}
    return [
        OrchestrationWarningResponse.model_validate(warning).model_copy(update={
            "needs_you_references": [resolved_by_key[(request.kind, request.id)] for request in requests
                                     if (request.kind, request.id) in resolved_by_key]
        })
        for warning, requests in zip(warnings, per_response)
    ]


def typed_blocker_reference_requests(
    blockers: Sequence[object], *, goal_id: uuid.UUID, run_id: uuid.UUID | None = None,
) -> list[NeedsYouReferenceRequest]:
    requests: list[NeedsYouReferenceRequest] = []
    fields: tuple[tuple[str, NeedsYouReferenceKind], ...] = (("task_id", "task"), ("gate_id", "gate"), ("artifact_id", "artifact"))
    for blocker in blockers:
        if not isinstance(blocker, dict):
            continue
        for field, kind in fields:
            if (record_id := _uuid(blocker.get(field))) is not None:
                requests.append(NeedsYouReferenceRequest(kind, record_id, goal_id, run_id if kind == "artifact" else None))
    return requests
