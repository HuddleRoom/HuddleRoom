import uuid
from types import SimpleNamespace
from copy import deepcopy
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.config import settings
from huddleroom.database import get_db
from huddleroom.dependencies import ensure_project_exists, get_current_user
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationAgentSuggestion,
    OrchestrationDecision,
    OrchestrationEvidence,
    OrchestrationGate,
    OrchestrationGoal,
    OrchestrationRoadmapItem,
    OrchestrationRun,
    OrchestrationWait,
)
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.models.user import User
from huddleroom.schemas.common import CursorPage
from huddleroom.schemas.orchestration import (
    OrchestrationActionResponse,
    OrchestrationAgentSuggestionResponse,
    OrchestrationAuthorityDecisionResponse,
    OrchestrationDebugActionResponse,
    OrchestrationDebugRerunRequest,
    OrchestrationDebugStepRequest,
    OrchestrationDecisionResponse,
    OrchestrationEvidenceResponse,
    OrchestrationGateOverrideRequest,
    OrchestrationGateResponse,
    OrchestrationGoalCreate,
    OrchestrationContinuousPolicyUpdateRequest,
    OrchestrationContinuousStopRequest,
    OrchestrationGoalDefinitionRecoverRequest,
    OrchestrationGoalDetailResponse,
    OrchestrationGoalResponse,
    OrchestrationSupersedeRequest,
    OrchestrationGoalWeightOverrideRequest,
    OrchestrationConversationAllowanceResponse,
    OrchestrationConversationFeedbackRequest,
    OrchestrationConversationFeedbackResponse,
    OrchestrationConversationHistoryResponse,
    OrchestrationConversationInvestigationResponse,
    OrchestrationConversationSubmitRequest,
    OrchestrationConversationLearningReportResponse,
    OrchestrationConversationTurnResponse,
    OrchestrationSteeringLedgerResponse,
    OrchestrationSteeringProposalResponse,
    OrchestrationSteeringRequestResponse,
    OrchestrationSteeringSubmitRequest,
    OrchestrationSteeringTransitionResponse,
    OrchestrationRoadmapBudgetResponse,
    OrchestrationRoadmapItemResponse,
    OrchestrationRoadmapVersionResponse,
    OrchestrationRunResponse,
    OrchestrationSupervisionResponse,
    OrchestrationTickResponse,
)
from huddleroom.services.orchestration_budget_service import BudgetMeasurementError, OrchestrationBudgetService
from huddleroom.services.orchestration_debug_service import OrchestrationDebugService
from huddleroom.services.orchestration_recovery_service import OrchestrationRecoveryService
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_supersession import OrchestrationSupersessionService
from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService
from huddleroom.services.orchestration_continuous_service import OrchestrationContinuousService
from huddleroom.services.orchestration_conversation_service import OrchestrationConversationService
from huddleroom.services.orchestration_conversation_service import ConversationDomainError, ConversationTurn
from huddleroom.models.orchestration_conversation import ConversationInvestigation
from huddleroom.models.orchestration_steering import (
    OrchestrationSteeringProposal, OrchestrationSteeringRequest,
    OrchestrationSteeringResultLink, OrchestrationSteeringTransition,
)
from huddleroom.services.orchestration_conversation_investigation import conversation_allowance_used
from huddleroom.services.orchestration_steering import (
    OrchestrationSteeringService, SteeringDomainError, SteeringDraft,
)
from huddleroom.services.orchestration_conversation_learning import (
    ConversationFeedbackInput,
    ConversationLearningError,
    ConversationLearningWindow,
    OrchestrationConversationLearningService,
)

router = APIRouter()
service = OrchestrationService()
conversation_service = OrchestrationConversationService(orchestration_service=service)
steering_service = OrchestrationSteeringService(service)
learning_service = OrchestrationConversationLearningService()
debug_service = OrchestrationDebugService()


_INVESTIGATION_OPERATIONS = {"list", "read", "search"}
_INVESTIGATION_SOURCE_STATUSES = {
    "included", "restricted", "unsafe", "binary", "too_large", "changed", "omitted_by_limit",
}
_INVESTIGATION_ERROR_CODES = {
    "request_cancelled", "conversation_allowance_exhausted", "workspace_unavailable",
    "provider_outcome_unknown", "invalid_investigation_report", "interrupted_before_dispatch",
}


def _utc_timestamp(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _conversation_investigation(
    row: ConversationInvestigation | None,
) -> OrchestrationConversationInvestigationResponse | None:
    if row is None:
        return None
    manifest = row.input_manifest if isinstance(row.input_manifest, dict) else {}
    included = manifest.get("sources") if isinstance(manifest.get("sources"), list) else []
    omissions = manifest.get("omissions") if isinstance(manifest.get("omissions"), list) else []

    def source(item: object, default_status: str) -> dict | None:
        if not isinstance(item, dict):
            return None
        operation, reference = item.get("operation"), item.get("reference")
        status = item.get("status", default_status)
        freshness_at, truncated = item.get("freshness_at"), item.get("truncated", False)
        if not all((
            isinstance(operation, str) and operation in _INVESTIGATION_OPERATIONS,
            isinstance(reference, str),
            isinstance(status, str) and status in _INVESTIGATION_SOURCE_STATUSES,
            isinstance(freshness_at, str) or freshness_at is None,
            isinstance(truncated, bool),
        )):
            return None
        return {
            "reference": reference,
            "operation": operation,
            "status": status,
            "freshness_at": freshness_at,
            "truncated": truncated,
        }

    sources = [mapped for item in included if (mapped := source(item, "included")) is not None]
    sources.extend(mapped for item in omissions if (mapped := source(item, "")) is not None)
    permitted = {item["reference"] for item in sources if item["status"] == "included"}
    report = row.report if isinstance(row.report, dict) else None
    if not (
        isinstance(report, dict)
        and set(report) == {"findings", "uncertainty", "sources"}
        and isinstance(report["findings"], str)
        and isinstance(report["uncertainty"], str)
        and isinstance(report["sources"], list)
        and all(isinstance(reference, str) for reference in report["sources"])
        and len(set(report["sources"])) == len(report["sources"])
        and set(report["sources"]) <= permitted
    ):
        report = None
    error = row.error if isinstance(row.error, dict) else None
    code = error.get("code") if error else None
    return OrchestrationConversationInvestigationResponse(
        investigation_id=row.id,
        status=row.status,
        objective=row.objective,
        attempt_count=row.attempt_count,
        repair_count=row.repair_count,
        retry_count=row.retry_count,
        sources=sources,
        report=report,
        error={"code": code} if isinstance(code, str) and code in _INVESTIGATION_ERROR_CODES else None,
        started_at=_utc_timestamp(row.started_at),
        deadline_at=_utc_timestamp(row.deadline_at),
        finished_at=_utc_timestamp(row.finished_at),
        created_at=_utc_timestamp(row.created_at),
        updated_at=_utc_timestamp(row.updated_at),
    )


def _conversation_turn(
    turn: ConversationTurn, proposal: OrchestrationSteeringProposalResponse | None = None,
) -> OrchestrationConversationTurnResponse:
    message, response = turn.message, turn.response
    error = response.error if isinstance(response.error, dict) else None
    return OrchestrationConversationTurnResponse(
        message_id=message.id,
        response_id=response.id,
        client_request_id=message.client_request_id,
        sequence=message.sequence,
        actor_id=message.actor_id,
        content=message.content,
        message_created_at=message.created_at,
        status=response.status,
        run_id=response.run_id,
        answer=response.answer,
        error={"code": error["code"]} if isinstance(error and error.get("code"), str) else None,
        started_at=response.started_at,
        deadline_at=response.deadline_at,
        finished_at=response.finished_at,
        created_at=response.created_at,
        updated_at=response.updated_at,
        context_version=response.context_version,
        context_manifest=response.context_manifest,
        investigation=_conversation_investigation(turn.investigation),
        proposed_steering=proposal,
        feedback=(
            OrchestrationConversationFeedbackResponse(
                feedback_id=turn.feedback.id,
                rating=turn.feedback.rating,
                reason=turn.feedback.reason,
                created_at=_utc_timestamp(turn.feedback.created_at),
            )
            if turn.feedback is not None else None
        ),
        feedback_eligible=turn.feedback_eligible,
    )


def _conversation_error(error: ConversationDomainError) -> HTTPException:
    return HTTPException(
        status_code=error.status_code,
        detail=(
            error.message
            if error.status_code == 404
            else {"code": error.code, "message": error.message}
        ),
    )


def _learning_error(error: ConversationLearningError) -> HTTPException:
    return HTTPException(
        status_code=error.status_code,
        detail={"code": error.code, "message": error.message},
    )


def _steering_error(error: SteeringDomainError) -> HTTPException:
    return HTTPException(
        status_code=error.status_code,
        detail={"code": error.code, "message": error.message},
    )


def _steering_proposal(proposal: OrchestrationSteeringProposal) -> OrchestrationSteeringProposalResponse:
    draft = proposal.draft if isinstance(proposal.draft, dict) else {}
    return OrchestrationSteeringProposalResponse(
        proposal_id=proposal.id, response_id=proposal.response_id, status=proposal.status,
        directive=str(draft.get("directive", "")), target_type=draft.get("target_type", "goal"),
        target_id=str(draft.get("target_id", "")), scope=draft.get("scope", "run"),
        lifetime=draft.get("lifetime", "remaining_current_run"),
        impact_summary=str(draft.get("impact_summary", "")), dismissed_at=proposal.dismissed_at,
        promoted_request_id=proposal.promoted_request_id, created_at=proposal.created_at,
        updated_at=proposal.updated_at,
    )


async def _steering_request(
    db: AsyncSession, request: OrchestrationSteeringRequest,
) -> OrchestrationSteeringRequestResponse:
    transitions = (await db.scalars(select(OrchestrationSteeringTransition).where(
        OrchestrationSteeringTransition.request_id == request.id,
    ).order_by(OrchestrationSteeringTransition.sequence))).all()
    action_ids = list((await db.scalars(select(OrchestrationSteeringResultLink.action_id).where(
        OrchestrationSteeringResultLink.request_id == request.id,
    ).order_by(OrchestrationSteeringResultLink.created_at, OrchestrationSteeringResultLink.action_id))).all())
    return OrchestrationSteeringRequestResponse(
        request_id=request.id, client_request_id=request.client_request_id, sequence=request.sequence,
        directive=request.directive, target_type=request.target_type, target_id=request.target_id,
        scope=request.scope, lifetime=request.lifetime, impact_summary=request.impact_summary,
        source_proposal_id=request.source_proposal_id, supersedes_request_id=request.supersedes_request_id,
        status=request.status, reason_code=request.reason_code, submitted_at=request.submitted_at,
        considered_at=request.considered_at, finished_at=request.finished_at, updated_at=request.updated_at,
        transitions=[OrchestrationSteeringTransitionResponse(
            status=item.to_status, reason_code=item.reason_code, actor=item.actor,
            created_at=item.created_at,
        ) for item in transitions],
        result_action_ids=action_ids,
    )


async def _steering_ledger(
    db: AsyncSession, project_id: uuid.UUID, goal_id: uuid.UUID, actor_id: uuid.UUID,
) -> OrchestrationSteeringLedgerResponse:
    try:
        ledger = await steering_service.ledger(db, project_id, goal_id, actor_id)
    except SteeringDomainError as error:
        raise _steering_error(error) from error
    if not ledger.enabled:
        return OrchestrationSteeringLedgerResponse(
            enabled=False, eligibility=ledger.eligibility,
            eligibility_reason=ledger.eligibility_reason, inbox_version=0, direction_version=0,
        )
    return OrchestrationSteeringLedgerResponse(
        enabled=ledger.enabled, eligibility=ledger.eligibility,
        eligibility_reason=ledger.eligibility_reason, inbox_version=ledger.inbox_version,
        direction_version=ledger.direction_version,
        requests=[await _steering_request(db, item) for item in ledger.requests],
        proposals=[_steering_proposal(item) for item in ledger.proposals],
    )


async def _conversation_allowance(
    db: AsyncSession, goal_id: uuid.UUID, actor_id: uuid.UUID
) -> OrchestrationConversationAllowanceResponse:
    limit = settings.orchestration_conversation_allowance_tokens
    used = await conversation_allowance_used(db, goal_id, actor_id)
    return OrchestrationConversationAllowanceResponse(
        enabled=limit > 0,
        limit=limit,
        used=used,
        remaining=max(limit - used, 0),
    )


async def _detail(
    db: AsyncSession,
    goal: OrchestrationGoal,
    run: OrchestrationRun | None,
) -> OrchestrationGoalDetailResponse:
    decisions = []
    actions = []
    gates = []
    evidence = []
    agent_suggestions = []
    supervision = None
    roadmap_summary = None
    if run is not None:
        decision_rows = await db.execute(
            select(OrchestrationDecision)
            .where(OrchestrationDecision.run_id == run.id)
            .order_by(OrchestrationDecision.created_at.asc(), OrchestrationDecision.id.asc())
        )
        decisions = [
            OrchestrationDecisionResponse.model_validate(decision)
            for decision in decision_rows.scalars().all()
        ]

        action_rows = await db.execute(
            select(OrchestrationAction)
            .where(OrchestrationAction.run_id == run.id)
            .order_by(OrchestrationAction.created_at.asc(), OrchestrationAction.id.asc())
        )
        actions = [
            OrchestrationActionResponse.model_validate(action)
            for action in action_rows.scalars().all()
        ]

        gate_rows = await db.execute(
            select(OrchestrationGate)
            .where(OrchestrationGate.run_id == run.id)
            .order_by(OrchestrationGate.created_at.asc(), OrchestrationGate.id.asc())
        )
        gates = [
            OrchestrationGateResponse.model_validate(gate)
            for gate in gate_rows.scalars().all()
        ]

        evidence_rows = await db.execute(
            select(OrchestrationEvidence)
            .where(OrchestrationEvidence.run_id == run.id)
            .order_by(OrchestrationEvidence.created_at.asc(), OrchestrationEvidence.id.asc())
        )
        evidence = [
            OrchestrationEvidenceResponse.model_validate(row)
            for row in evidence_rows.scalars().all()
        ]

        suggestion_rows = await db.execute(
            select(OrchestrationAgentSuggestion)
            .where(OrchestrationAgentSuggestion.run_id == run.id)
            .order_by(
                OrchestrationAgentSuggestion.created_at.asc(),
                OrchestrationAgentSuggestion.id.asc(),
            )
        )
        agent_suggestions = [
            OrchestrationAgentSuggestionResponse.model_validate(suggestion)
            for suggestion in suggestion_rows.scalars().all()
        ]

        wait_rows = await db.execute(
            select(OrchestrationWait)
            .where(OrchestrationWait.run_id == run.id, OrchestrationWait.status == "open")
            .order_by(OrchestrationWait.created_at.asc(), OrchestrationWait.id.asc())
        )
        open_waits = wait_rows.scalars().all()
        waits = [{
            "id": str(wait.id), "owner": dict(wait.owner or {}),
            "event": (wait.awaited_event or {}).get("event_type"),
            "matcher": dict((wait.awaited_event or {}).get("matcher") or {}),
            "due_recheck_at": wait.due_recheck_at, "fallback": dict(wait.fallback or {}),
        } for wait in open_waits]
        direction = await db.scalar(
            select(OrchestrationAuthorityDecision)
            .where(OrchestrationAuthorityDecision.run_id == run.id,
                   OrchestrationAuthorityDecision.status == "pending")
            .order_by(OrchestrationAuthorityDecision.asked_at.asc(), OrchestrationAuthorityDecision.id.asc())
        )
        accepted_evidence = [item for item in evidence if item.verdict == "accepted"]
        gate = next((item for item in gates if item.status == "accepted"), None)
        if gate is None:
            gate = next((item for item in gates if item.status == "open"), None)
        criterion = None
        if gate is not None:
            criterion = {"key": gate.success_criterion_key, "status": gate.status}
            criterion.update(next((item for item in goal.success_criteria
                                   if isinstance(item, dict) and item.get("key") == gate.success_criterion_key), {}))

        recovery = (run.supervision_state or {}).get("recovery", {})
        recovery_sessions = recovery.get("sessions", {}) if isinstance(recovery, dict) else {}
        worker_rows = await db.execute(
            select(Session, Task).join(Task, Task.id == Session.task_id)
            .where(Session.project_id == goal.project_id)
            .order_by(Session.created_at.asc(), Session.id.asc())
        )
        workers = []
        recovery_service = OrchestrationRecoveryService()
        for session, task in worker_rows.all():
            owned = await recovery_service.resolve_owned_session(db, session.id)
            if owned is None or owned.action_id not in {action.id for action in actions}:
                continue
            entry = recovery_sessions.get(str(session.id), {}) if isinstance(recovery_sessions, dict) else {}
            entry = entry if isinstance(entry, dict) else {}
            workers.append({
                "session_id": str(session.id), "task_id": str(task.id), "agent_id": str(session.agent_id),
                "task_title": task.title, "session_status": session.status,
                "observed_liveness": entry.get("classification", "unknown"),
                "runner_id": entry.get("current_runner_id", entry.get("current_runner_task_id", owned.runner_task_id)),
                "observed_at": entry.get("assessed_at"),
            })
        recovery_history = [{
            "session_id": str(session_id), "classification": entry.get("classification"),
            "disposition": entry.get("disposition"),
            "backend_observation": entry.get("backend_observation"),
            "assessed_at": entry.get("assessed_at"), "action_id": entry.get("action_id"),
            "wait_id": entry.get("wait_id"),
        } for session_id, entry in recovery_sessions.items() if isinstance(entry, dict)]
        recovery_history.sort(key=lambda item: (str(item["assessed_at"] or ""), item["session_id"]))

        supervision_state = deepcopy(run.supervision_state or {})
        budget_run = SimpleNamespace(id=run.id, budget_state=deepcopy(run.budget_state or {}))
        try:
            snapshot = await OrchestrationBudgetService().snapshot_for_run(db, goal, budget_run)
            budget = {
                key: {dimension: value for dimension, value in snapshot[key].items() if value != "0"}
                for key in ("consumed", "committed", "reserved")
            }
            budget["remaining"] = dict(snapshot["remaining"])
        except BudgetMeasurementError:
            budget = {}

        if goal.goal_type == "roadmap":
            roadmap_summary = await OrchestrationRoadmapService(service).remaining_or_block(db, goal, run)

        state = supervision_state if isinstance(supervision_state, dict) else {}
        verified_progress = [dict(item) for item in state.get("verified_progress", []) if isinstance(item, dict)]
        continuous_stopped_at = (
            (goal.continuous_state or {}).get("stopped_at")
            if goal.goal_type == "continuous" and isinstance(goal.continuous_state, dict)
            else None
        )
        if goal.status == "completed":
            condition, operation, next_action, rationale = "completed", "Completed", "No action", "Goal completed"
        elif goal.status == "cancelled":
            condition, operation, next_action, rationale = "cancelled", "Cancelled", "No action", "Goal cancelled"
        elif continuous_stopped_at:
            condition, operation, next_action, rationale = "stopped", "Stopped", "No action", "Continuous policy stopped"
        elif goal.status == "paused" or run.status == "paused":
            condition, operation, next_action, rationale = "paused", "Paused", "Resume goal", "Goal is paused"
        elif direction is not None:
            condition, operation, next_action, rationale = (
                "needs_you", direction.title, "Answer pending direction", direction.question,
            )
        elif run.active_blockers:
            blocker = next((item for item in run.active_blockers if isinstance(item, dict)), {})
            reason = blocker.get("reason") or blocker.get("kind") or "A durable blocker is open"
            condition, operation, next_action, rationale = "needs_attention", "Needs attention", "Resolve blocker", str(reason)
        elif waits:
            condition, operation, next_action, rationale = (
                "waiting", f"Awaiting {waits[0]['event']}", f"Await {waits[0]['event']}",
                str(waits[0]["fallback"].get("reason", "Waiting for durable event")),
            )
        elif workers:
            condition, operation, next_action, rationale = "working", "Work in progress", "Await worker result", "An owned worker has durable supervision"
        else:
            condition, operation, next_action, rationale = "working", "Work in progress", "Continue execution", "No durable wait or direction"
        transition = None
        if condition in {"completed", "cancelled", "paused", "stopped"}:
            transition = {
                "id": str(goal.id), "key": f"goal_control:{goal.id}:{condition}", "kind": condition,
                "message": operation, "occurred_at": continuous_stopped_at or goal.updated_at,
            }
        elif condition == "needs_you" and direction is not None:
            transition = {
                "id": str(direction.id), "key": f"pending_direction:{direction.id}",
                "kind": "pending_direction", "message": direction.title, "occurred_at": direction.asked_at,
            }
        else:
            failed_gate = next((item for item in reversed(gates) if item.status == "failed"), None)
            failed_action = next((item for item in reversed(actions) if item.status == "failed"), None)
            blocker = next((item for item in run.active_blockers if isinstance(item, dict)), None)
            recovery_event = next((item for item in reversed(recovery_history)
                                   if item.get("action_id") or item.get("wait_id")), None)
            if failed_gate is not None:
                transition = {
                    "id": str(failed_gate.id), "key": f"gate:{failed_gate.id}:failed", "kind": "failed_gate",
                    "message": failed_gate.failure_reason or failed_gate.success_criterion_key,
                    "occurred_at": failed_gate.failed_at or failed_gate.updated_at,
                }
            elif failed_action is not None:
                transition = {
                    "id": str(failed_action.id), "key": f"action:{failed_action.id}:failed", "kind": "failed_action",
                    "message": failed_action.error or failed_action.action_type, "occurred_at": failed_action.updated_at,
                }
            elif blocker is not None:
                blocker_kind = str(blocker.get("kind", "attention"))
                blocker_reason = str(blocker.get("reason", blocker_kind))
                transition = {
                    "id": f"blocker:{run.id}:{blocker_kind}:{blocker_reason}",
                    "key": f"blocker:{run.id}:{blocker_kind}:{blocker_reason}", "kind": "attention_blocker",
                    "message": blocker_reason, "occurred_at": run.updated_at,
                }
            elif recovery_event is not None:
                recovery_id = recovery_event.get("action_id") or recovery_event.get("wait_id")
                transition = {
                    "id": recovery_id, "key": f"recovery:{recovery_event['session_id']}:{recovery_id}",
                    "kind": "recovery_action" if recovery_event.get("action_id") else "recovery_wait",
                    "message": str(recovery_event.get("disposition") or "recovery"),
                    "occurred_at": recovery_event.get("assessed_at"),
                }
            elif accepted_evidence:
                accepted = accepted_evidence[-1]
                transition = {
                    "id": str(accepted.id), "key": f"evidence:{accepted.id}:accepted", "kind": "accepted_evidence",
                    "message": accepted.source_type, "occurred_at": accepted.created_at,
                }
        supervision = OrchestrationSupervisionResponse(
            condition=condition, operation=operation, next_action=next_action, rationale=rationale,
            criterion=criterion, verified_progress=verified_progress, useful_learning=[],
            accepted_evidence=accepted_evidence, workers=workers, waits=waits,
            recovery_history=recovery_history,
            pending_direction=(OrchestrationAuthorityDecisionResponse.model_validate(direction)
                               if condition == "needs_you" and direction else None),
            budget=budget, transition=transition,
        )

    roadmap_version = None
    roadmap_items = None
    children = None
    budget_summary = None
    if goal.goal_type == "roadmap":
        roadmap = OrchestrationRoadmapService(service)
        version = await roadmap.current_version(db, goal.id)
        roadmap_version = OrchestrationRoadmapVersionResponse.model_validate(version) if version else None
        roadmap_items = [OrchestrationRoadmapItemResponse.model_validate(item) for item in (
            await db.scalars(select(OrchestrationRoadmapItem).where(
                OrchestrationRoadmapItem.goal_id == goal.id
            ).order_by(OrchestrationRoadmapItem.item_key))
        ).all()]
        children = list((await db.scalars(select(OrchestrationGoal.id).where(
            OrchestrationGoal.parent_goal_id == goal.id
        ).order_by(OrchestrationGoal.id))).all())
        budget_summary = (OrchestrationRoadmapBudgetResponse(**roadmap_summary)
                          if roadmap_summary is not None else None)

    run_response = None
    if run is not None:
        run_response = OrchestrationRunResponse.model_validate(
            {**OrchestrationRunResponse.model_validate(run).model_dump(),
             "condition": service.run_condition(goal, run)}
        )

    return OrchestrationGoalDetailResponse(
        goal=OrchestrationGoalResponse.model_validate(goal),
        run=run_response,
        decisions_count=len(decisions),
        decisions=decisions,
        actions_count=len(actions),
        actions=actions,
        gates_count=len(gates),
        gates=gates,
        evidence_count=len(evidence),
        evidence=evidence,
        agent_suggestions_count=len(agent_suggestions),
        agent_suggestions=agent_suggestions,
        roadmap_version=roadmap_version,
        roadmap_items=roadmap_items,
        children=children,
        budget_summary=budget_summary,
        supervision=supervision,
    )


@router.post("/goals", response_model=OrchestrationGoalDetailResponse, status_code=201)
async def create_goal(
    project_id: uuid.UUID,
    data: OrchestrationGoalCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    goal, run = await service.create_goal(db, project_id, data, created_by_user_id=user.id)
    return await _detail(db, goal, run)


@router.get("/goals", response_model=CursorPage[OrchestrationGoalResponse])
async def list_goals(
    project_id: uuid.UUID,
    status: str | None = None,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    items, next_cursor = await service.list_goals(db, project_id, status=status, cursor=cursor, limit=limit)
    return CursorPage(items=items, next_cursor=next_cursor)


@router.get("/goals/{goal_id}", response_model=OrchestrationGoalDetailResponse)
async def get_goal(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    goal = await service.get_goal(db, project_id, goal_id)
    if goal is None:
        raise HTTPException(status_code=404, detail="Orchestration goal not found")
    run = await service.get_run_for_goal(db, project_id, goal_id)
    return await _detail(db, goal, run)


@router.get(
    "/goals/{goal_id}/conversation",
    response_model=OrchestrationConversationHistoryResponse,
)
async def get_conversation(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    if await service.get_goal(db, project_id, goal_id) is None:
        raise HTTPException(status_code=404, detail="Orchestration goal not found")
    await conversation_service.recover_goal(goal_id)
    steering = await _steering_ledger(db, project_id, goal_id, user.id)
    try:
        turns = await conversation_service.history(project_id, goal_id, user.id)
    except ConversationDomainError as error:
        raise _conversation_error(error) from error
    total = len(turns)
    items = turns[-50:]
    proposals_by_response = {proposal.response_id: proposal for proposal in steering.proposals}
    return OrchestrationConversationHistoryResponse(
        items=[_conversation_turn(turn, proposals_by_response.get(turn.response.id)) for turn in items],
        total=total,
        omitted=max(total - len(items), 0),
        allowance=await _conversation_allowance(db, goal_id, user.id),
        steering=steering,
    )


@router.put(
    "/goals/{goal_id}/conversation/{response_id}/feedback",
    response_model=OrchestrationConversationFeedbackResponse,
)
async def record_conversation_feedback(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    response_id: uuid.UUID,
    data: OrchestrationConversationFeedbackRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    try:
        async with db.begin_nested():
            feedback = await learning_service.record_feedback(
                db,
                project_id,
                goal_id,
                user.id,
                response_id,
                ConversationFeedbackInput(rating=data.rating, reason=data.reason),
            )
            result = OrchestrationConversationFeedbackResponse(
                feedback_id=feedback.id,
                rating=feedback.rating,
                reason=feedback.reason,
                created_at=_utc_timestamp(feedback.created_at),
            )
        await db.commit()
        return result
    except ConversationLearningError as error:
        await db.rollback()
        raise _learning_error(error) from error
    except Exception:
        await db.rollback()
        raise


@router.get(
    "/conversation-learning",
    response_model=OrchestrationConversationLearningReportResponse,
)
async def get_conversation_learning(
    project_id: uuid.UUID,
    start_at: datetime,
    end_at: datetime,
    goal_id: uuid.UUID | None = None,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    try:
        report = await learning_service.summarize(
            db, project_id, ConversationLearningWindow(start_at, end_at, goal_id)
        )
        return OrchestrationConversationLearningReportResponse.model_validate(report.__dict__)
    except ConversationLearningError as error:
        raise _learning_error(error) from error


@router.post(
    "/goals/{goal_id}/conversation",
    response_model=OrchestrationConversationTurnResponse,
)
async def submit_conversation(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    data: OrchestrationConversationSubmitRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    if not data.content or len(data.content) > 4_000:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "conversation_invalid_content",
                "message": "Conversation content must be 1 to 4000 characters",
            },
        )
    if await service.get_goal(db, project_id, goal_id) is None:
        raise HTTPException(status_code=404, detail="Orchestration goal not found")
    await db.close()
    await conversation_service.recover_goal(goal_id)
    try:
        turn = await conversation_service.submit(
            project_id, goal_id, user.id, data.client_request_id, data.content
        )
    except ConversationDomainError as error:
        raise _conversation_error(error) from error
    return _conversation_turn(turn)


@router.post(
    "/goals/{goal_id}/conversation/steering",
    response_model=OrchestrationSteeringRequestResponse,
)
async def submit_steering(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    data: OrchestrationSteeringSubmitRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    try:
        request = await steering_service.submit(
            db, project_id, goal_id, user.id, data.client_request_id,
            SteeringDraft(
                directive=data.directive, target_type=data.target_type, target_id=data.target_id,
                scope=data.scope, lifetime=data.lifetime, impact_summary=data.impact_summary,
                source_proposal_id=data.source_proposal_id,
                supersedes_request_id=data.supersedes_request_id,
            ),
        )
        result = await _steering_request(db, request)
        await db.commit()
        return result
    except SteeringDomainError as error:
        await db.rollback()
        raise _steering_error(error) from error


@router.post(
    "/goals/{goal_id}/conversation/steering/{request_id}/withdraw",
    response_model=OrchestrationSteeringRequestResponse,
)
async def withdraw_steering(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    request_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    try:
        request = await steering_service.withdraw(db, project_id, goal_id, user.id, request_id)
        result = await _steering_request(db, request)
        await db.commit()
        return result
    except SteeringDomainError as error:
        await db.rollback()
        raise _steering_error(error) from error


@router.post(
    "/goals/{goal_id}/conversation/steering/proposals/{proposal_id}/dismiss",
    response_model=OrchestrationSteeringProposalResponse,
)
async def dismiss_steering_proposal(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    proposal_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    try:
        proposal = await steering_service.dismiss_proposal(db, project_id, goal_id, user.id, proposal_id)
        result = _steering_proposal(proposal)
        await db.commit()
        return result
    except SteeringDomainError as error:
        await db.rollback()
        raise _steering_error(error) from error


@router.post("/goals/{goal_id}/start", response_model=OrchestrationGoalDetailResponse)
async def start_goal(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    goal, run = await service.start_run(db, project_id, goal_id, actor=f"human:{user.id}")
    return await _detail(db, goal, run)


@router.post("/goals/{goal_id}/baseline/authorize", response_model=OrchestrationGoalDetailResponse)
async def authorize_baseline(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    goal, run = await service.authorize_baseline(db, project_id, goal_id, actor=f"human:{user.id}")
    return await _detail(db, goal, run)


@router.put("/goals/{goal_id}/continuous-policy", response_model=OrchestrationGoalDetailResponse)
async def update_continuous_policy(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    data: OrchestrationContinuousPolicyUpdateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    goal, run = await OrchestrationContinuousService(service).update_policy(
        db, project_id, goal_id, data.policy, actor=f"human:{user.id}",
    )
    return await _detail(db, goal, run)


@router.post("/goals/{goal_id}/stop", response_model=OrchestrationGoalDetailResponse)
async def stop_continuous_goal(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    data: OrchestrationContinuousStopRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    goal, run = await OrchestrationContinuousService(service).stop(
        db, project_id, goal_id, actor=f"human:{user.id}", reason=data.reason,
    )
    return await _detail(db, goal, run)


@router.post("/goals/{goal_id}/pause", response_model=OrchestrationGoalDetailResponse)
async def pause_goal(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    goal, run = await service.pause_goal(db, project_id, goal_id)
    return await _detail(db, goal, run)


@router.post("/goals/{goal_id}/resume", response_model=OrchestrationGoalDetailResponse)
async def resume_goal(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    goal, run = await service.resume_goal(db, project_id, goal_id)
    return await _detail(db, goal, run)


@router.post("/goals/{goal_id}/cancel", response_model=OrchestrationGoalDetailResponse)
async def cancel_goal(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    goal, run = await service.cancel_goal(
        db, project_id, goal_id, cancelled_by=f"human:{user.id}"
    )
    return await _detail(db, goal, run)


@router.post("/goals/{goal_id}/supersede", response_model=OrchestrationGoalDetailResponse, status_code=201)
async def supersede_goal(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    data: OrchestrationSupersedeRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Auth/project reads may already have autobegun this request's session.
    # Keep that transaction, its goal lock, detail reads, and commit together.
    async with service._lock_goal_for_baseline_transition(db, goal_id):
        try:
            await ensure_project_exists(db, project_id)
            replacement = await OrchestrationSupersessionService().supersede(
                db, project_id, goal_id, new_goal_type=data.goal_type, actor=f"human:{user.id}"
            )
            run = await service.get_run_for_goal(db, project_id, replacement.id)
            detail = await _detail(db, replacement, run)
            await db.commit()
            return detail
        except Exception:
            await db.rollback()
            raise


@router.post("/goals/{goal_id}/reset", response_model=OrchestrationGoalDetailResponse)
async def reset_goal(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    goal, run = await service.reset_goal(db, project_id, goal_id)
    return await _detail(db, goal, run)


@router.post("/goals/{goal_id}/override", response_model=OrchestrationGoalDetailResponse)
async def override_goal_gate(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    data: OrchestrationGateOverrideRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    goal, run = await service.override_gate(
        db,
        project_id,
        goal_id,
        gate_id=data.gate_id,
        decision=data.decision,
        reason=data.reason,
        user_id=user.id,
        evidence_metadata=data.evidence_metadata,
    )
    return await _detail(db, goal, run)


@router.post("/goals/{goal_id}/weight", response_model=OrchestrationGoalDetailResponse)
async def override_goal_weight(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    data: OrchestrationGoalWeightOverrideRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    goal = await service.override_goal_weight(
        db, project_id, goal_id, weight=data.weight, reason=data.reason, user_id=user.id
    )
    run = await service.get_run_for_goal(db, project_id, goal_id)
    return await _detail(db, goal, run)


@router.post("/runs/{run_id}/tick", response_model=OrchestrationTickResponse)
async def force_tick(
    project_id: uuid.UUID,
    run_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    await service.ensure_run_in_project(db, project_id, run_id)
    result = await service.tick(db, run_id)
    return OrchestrationTickResponse(**result)


def _require_debug_enabled() -> None:
    if not settings.debug:
        raise HTTPException(status_code=404, detail="Not Found")


@router.post("/goals/{goal_id}/baseline/step", response_model=OrchestrationDebugActionResponse)
async def baseline_step(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    data: OrchestrationDebugStepRequest,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    result = await debug_service.step(db, project_id, goal_id, data.process_type)
    return OrchestrationDebugActionResponse(action="step", **result)


@router.post("/goals/{goal_id}/baseline/rerun", response_model=OrchestrationDebugActionResponse)
async def baseline_rerun(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    data: OrchestrationDebugStepRequest,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    result = await debug_service.rerun_last(db, project_id, goal_id, data.process_type)
    return OrchestrationDebugActionResponse(action="rerun_last", **result)


@router.post("/goals/{goal_id}/baseline/retry", response_model=OrchestrationDebugActionResponse)
async def baseline_retry(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    data: OrchestrationDebugStepRequest,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    result = await debug_service.retry_failed(db, project_id, goal_id, data.process_type)
    return OrchestrationDebugActionResponse(action="retry", **result)


@router.post("/goals/{goal_id}/goal-definition/recover", response_model=OrchestrationDebugActionResponse)
async def recover_goal_definition(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    data: OrchestrationGoalDefinitionRecoverRequest,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await ensure_project_exists(db, project_id)
    result = await service.recover_goal_definition(
        db, project_id, goal_id, mode=data.mode
    )
    return OrchestrationDebugActionResponse(action="recover_goal_definition", **result)


@router.post("/goals/{goal_id}/debug/baseline/step", response_model=OrchestrationDebugActionResponse)
async def debug_baseline_step(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    data: OrchestrationDebugStepRequest,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_debug_enabled()
    await ensure_project_exists(db, project_id)
    result = await debug_service.step(db, project_id, goal_id, data.process_type)
    return OrchestrationDebugActionResponse(action="step", **result)


@router.post("/goals/{goal_id}/debug/baseline/rerun-last", response_model=OrchestrationDebugActionResponse)
async def debug_baseline_rerun_last(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    data: OrchestrationDebugRerunRequest | None = None,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_debug_enabled()
    await ensure_project_exists(db, project_id)
    result = await debug_service.rerun_last(db, project_id, goal_id, data.process_type if data else None)
    return OrchestrationDebugActionResponse(action="rerun_last", **result)
