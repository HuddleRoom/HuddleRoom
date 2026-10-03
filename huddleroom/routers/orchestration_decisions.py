import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.config import settings
from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision, OrchestrationProcessRun
from huddleroom.models.user import User
from huddleroom.schemas.orchestration import (
    OrchestrationAgentDefinitionReviewBatchAnswer,
    OrchestrationAgentDefinitionReviewBatchAnswerRequest,
    OrchestrationAgentDefinitionReviewBatchAnswerResponse,
    OrchestrationAuthorityDecisionResponse,
    OrchestrationCheckpointResponse,
    OrchestrationDecisionAnswerResponse,
    OrchestrationDecisionAnswerRequest,
    OrchestrationDecisionCancelRequest,
)
from huddleroom.services.orchestration_authority_interview import build_checkpoint
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_debug_service import OrchestrationDebugService, SUPPORTED_PROCESS_TYPES
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_service import LLM_DECISION_RUN_STATUSES, OrchestrationService
from huddleroom.services.orchestration_team_hierarchy import (
    PROPOSAL_RESOLUTION_RERUN_SENTINEL,
    TeamHierarchyProcess,
)
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService
from huddleroom.services.orchestration_team_hierarchy_analyzer import canonical_agent_create
from huddleroom.services.agent_service import AgentService
from huddleroom.services.project_service import ProjectService

router = APIRouter()
orchestration_service = OrchestrationService()
decision_service = OrchestrationAuthorityDecisionService()


async def _goal_or_404(db: AsyncSession, project_id: uuid.UUID, goal_id: uuid.UUID):
    goal = await orchestration_service.get_goal(db, project_id, goal_id)
    if goal is None:
        raise HTTPException(status_code=404, detail="Orchestration goal not found")
    return goal


async def _decision_or_404(
    db: AsyncSession, goal_id: uuid.UUID, decision_id: uuid.UUID
) -> OrchestrationAuthorityDecision:
    decision = await db.get(OrchestrationAuthorityDecision, decision_id)
    if decision is None or decision.goal_id != goal_id:
        raise HTTPException(status_code=404, detail="Orchestration authority decision not found")
    return decision


def _batch_answer_reason(item: OrchestrationAgentDefinitionReviewBatchAnswer) -> str | None:
    if item.selected_option != "edit":
        return item.reason
    return json.dumps({
        "description": item.edited_description.strip(),
        "persona": item.edited_persona.strip(),
    })


@router.get("/goals/{goal_id}/decisions", response_model=list[OrchestrationAuthorityDecisionResponse])
async def list_decisions(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    status: str | None = None,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _goal_or_404(db, project_id, goal_id)
    return await decision_service.list_decisions(db, goal_id, status=status)


@router.get("/goals/{goal_id}/decisions/checkpoint", response_model=OrchestrationCheckpointResponse)
async def get_checkpoint(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _goal_or_404(db, project_id, goal_id)
    pending = await decision_service.list_decisions(db, goal_id, status="pending")
    max_questions = settings.orchestration_checkpoint_max_questions
    checkpoint, deferred = build_checkpoint(pending, max_questions=max_questions)
    return OrchestrationCheckpointResponse(
        goal_id=goal_id,
        items=checkpoint,
        deferred_count=len(deferred),
        max_questions=max_questions,
    )


@router.post(
    "/goals/{goal_id}/decisions/agent-definition-review/batch-answer",
    response_model=OrchestrationAgentDefinitionReviewBatchAnswerResponse,
)
async def answer_agent_definition_review_batch(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    data: OrchestrationAgentDefinitionReviewBatchAnswerRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    goal = await _goal_or_404(db, project_id, goal_id)
    async with orchestration_service._lock_goal_for_baseline_transition(db, goal.id):
        await db.refresh(goal)
        active_run = await orchestration_service.get_active_run_for_goal(db, project_id, goal_id)
        if goal.status not in {"active", "blocked"} or active_run is None:
            raise HTTPException(status_code=409, detail="agent-definition review cannot advance")
        await db.refresh(active_run)
        if active_run.status not in LLM_DECISION_RUN_STATUSES:
            raise HTTPException(status_code=409, detail="agent-definition review cannot advance")
        await ProjectService().lock_workspace_boundary(db, project_id)
        await ProjectService().require_runnable_project(db, project_id)
        current = await OrchestrationProcessService().get_current(
            db, goal.id, "agent_definition_review"
        )
        if current is None or current.status != "waiting_decision":
            raise HTTPException(status_code=409, detail="agent-definition review is not awaiting a batch")
        pending_by_id = {
            decision.id: decision
            for decision in await decision_service.list_decisions(db, goal.id, status="pending")
            if decision.source_process_run_id == current.id
            and decision.decision_key.startswith("agent_definition_review:proposal:")
        }
        submitted_ids = [item.decision_id for item in data.answers]
        if len(submitted_ids) != len(set(submitted_ids)):
            raise HTTPException(status_code=400, detail="decision IDs must be unique")
        if set(submitted_ids) != set(pending_by_id):
            raise HTTPException(status_code=409, detail="pending proposal set changed")
        for item in data.answers:
            decision = pending_by_id[item.decision_id]
            if item.selected_option not in decision.options:
                raise HTTPException(status_code=400, detail="selected option is not offered")
            edit_fields = {"edited_description", "edited_persona"} & item.model_fields_set
            if item.selected_option == "edit":
                if edit_fields != {"edited_description", "edited_persona"} or not all(
                    isinstance(value, str) and value.strip()
                    for value in (item.edited_description, item.edited_persona)
                ):
                    raise HTTPException(status_code=400, detail="edit requires description and persona")
            elif edit_fields:
                raise HTTPException(status_code=400, detail="edited fields require edit")
        try:
            async with db.begin_nested():
                answered = [await decision_service.answer_decision(
                    db,
                    pending_by_id[item.decision_id],
                    selected_option=item.selected_option,
                    reason=_batch_answer_reason(item),
                    decided_by_user_id=user.id,
                ) for item in data.answers]
                process = await OrchestrationDebugService().advance_process(
                    db, goal, active_run, "agent_definition_review"
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return OrchestrationAgentDefinitionReviewBatchAnswerResponse(
            decisions=answered, process=process
        )


@router.post(
    "/goals/{goal_id}/decisions/{decision_id}/answer",
    response_model=OrchestrationDecisionAnswerResponse,
)
async def answer_decision(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    decision_id: uuid.UUID,
    data: OrchestrationDecisionAnswerRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    goal = await _goal_or_404(db, project_id, goal_id)
    decision = await _decision_or_404(db, goal_id, decision_id)
    is_agent_review = decision.decision_key.startswith("agent_definition_review:proposal:")
    legacy_fields = {"edited_description", "edited_persona"} & data.model_fields_set
    is_hierarchy_agent = decision.decision_key.startswith("team_hierarchy:agent:")
    if data.edited_agent is not None and legacy_fields:
        raise HTTPException(status_code=400, detail="edited_agent cannot be mixed with legacy edit fields")
    if legacy_fields:
        if not is_agent_review or data.selected_option != "edit":
            raise HTTPException(status_code=400, detail="legacy edit fields require agent-definition edit")
        if "edit" not in decision.options:
            raise HTTPException(status_code=400, detail="legacy edit fields require literal edit option")
        if legacy_fields != {"edited_description", "edited_persona"} or not all(
            isinstance(value, str) and value.strip()
            for value in (data.edited_description, data.edited_persona)
        ):
            raise HTTPException(status_code=400, detail="edit requires description and persona")
        data.reason = json.dumps({
            "description": data.edited_description.strip(),
            "persona": data.edited_persona.strip(),
        })
    elif is_agent_review:
        raise HTTPException(status_code=409, detail="agent-definition proposals require batch submission")
    if data.edited_agent is not None and (not is_hierarchy_agent or data.selected_option != "edit"):
        raise HTTPException(status_code=400, detail="edited_agent is only valid for a team-hierarchy agent edit")
    if is_hierarchy_agent and data.selected_option == "edit" and data.edited_agent is None:
        raise HTTPException(status_code=400, detail="team-hierarchy agent edit requires edited_agent")
    if decision.runtime_identity is not None:
        if data.contract_version is None:
            raise HTTPException(status_code=400, detail="contract_version is required for runtime decisions")
        try:
            result = await decision_service.answer_runtime_question(
                db,
                decision,
                data.selected_option,
                actor_user_id=user.id,
                contract_version=data.contract_version,
            )
            return OrchestrationDecisionAnswerResponse(
                decision=result.decision, process=None, continuation_applied=result.continuation_applied
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    if decision.status != "pending":
        raise HTTPException(status_code=409, detail=f"cannot answer decision in status '{decision.status}'")
    if decision.authority != "human":
        if decision.authority_agent_id is None:
            # Phase 2 carry-over guard: the agent this decision awaits was
            # deleted (FK SET NULL) after the decision was raised. It can
            # never be answered as-is -- surface cancel instead of a bare
            # validation error so the source process can re-raise it fresh.
            raise HTTPException(
                status_code=409,
                detail=(
                    "the agent this decision awaits was deleted; cancel it via "
                    "the cancel endpoint so its source process can re-raise a fresh decision"
                ),
            )
        raise HTTPException(
            status_code=409,
            detail=f"decision authority is '{decision.authority}'; it is answered through its agent's decision report",
        )
    try:
        async with OrchestrationService()._lock_goal_for_baseline_transition(db, decision.goal_id):
            answer_reason = data.reason
            if is_hierarchy_agent:
                current = await OrchestrationProcessService().get_current(db, goal.id, "team_hierarchy")
                if current is None or current.id != decision.source_process_run_id:
                    raise HTTPException(status_code=409, detail="team hierarchy proposal is no longer current")
                hierarchy = TeamHierarchyProcess()
                await db.refresh(goal)
                fingerprint = hierarchy._semantic_fingerprint(hierarchy._strip_volatile(await hierarchy._analysis_input(db, goal)))
                if current.input_snapshot.get("fingerprint") != fingerprint:
                    # Item 3, orchestrator override: a pending decision must
                    # never be cancelled just because inputs changed -- it
                    # stays parked. This answer attempt targets a stale
                    # snapshot, so it's rejected (409) without touching the
                    # decision; a one-time suggestion is raised (idempotent
                    # per process run) so the human sees it on the queue and
                    # can explicitly approve a rerun instead.
                    await OrchestrationWarningService().suggest_stale_inputs(
                        db,
                        goal.id,
                        process_type="team_hierarchy",
                        process_run_id=current.id,
                        step_label="team hierarchy",
                        run_id=current.run_id,
                    )
                    await db.commit()
                    raise HTTPException(status_code=409, detail="team hierarchy proposal inputs changed")
                if data.selected_option == "edit":
                    edited_agent = canonical_agent_create(data.edited_agent)
                    agent = await AgentService().create(db, edited_agent)
                    answer_reason = json.dumps({
                        "reason": data.reason, "created_agent_id": str(agent.id)
                    }, sort_keys=True)
            answered_decision = await decision_service.answer_decision(
                db,
                decision,
                selected_option=data.selected_option,
                reason=answer_reason,
                decided_by_user_id=user.id,
            )
            if is_hierarchy_agent and data.selected_option in {"edit", "reject"}:
                # Resolving one proposal in a batch obsoletes the rest --
                # this is a deliberate continuation of the same
                # decision-making process the human is already in, not a
                # background "inputs drifted" staleness (item 3's
                # one-time-suggestion rule doesn't apply here). The sentinel
                # fingerprint forces _advance_orchestrated's mismatch branch
                # to fire on the next advance_process() call below, which
                # recognizes it and reruns unconditionally instead of
                # raising a suggestion.
                current = await OrchestrationProcessService().get_current(db, goal.id, "team_hierarchy")
                await TeamHierarchyProcess()._cancel_pending_for_run(db, goal.id, current.id)
                current.input_snapshot = {"fingerprint": PROPOSAL_RESOLUTION_RERUN_SENTINEL}
            process = None
            if answered_decision.source_process_run_id is not None:
                source_process = await db.get(
                    OrchestrationProcessRun, answered_decision.source_process_run_id
                )
                if source_process is not None and source_process.process_type in SUPPORTED_PROCESS_TYPES:
                    current = await OrchestrationProcessService().get_current(
                        db, goal.id, source_process.process_type
                    )
                    if current is not None and current.id == source_process.id:
                        await db.refresh(goal)
                        active_run = await orchestration_service.get_active_run_for_goal(
                            db, project_id, goal_id
                        )
                        if goal.status in {"active", "blocked"} and active_run is not None:
                            await db.refresh(active_run)
                            if active_run.status in LLM_DECISION_RUN_STATUSES:
                                try:
                                    await ProjectService().lock_workspace_boundary(db, project_id)
                                    await ProjectService().require_runnable_project(db, project_id)
                                except HTTPException:
                                    pass
                                else:
                                    process = await OrchestrationDebugService().advance_process(
                                        db, goal, active_run, source_process.process_type
                                    )
            return OrchestrationDecisionAnswerResponse(decision=answered_decision, process=process)
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/goals/{goal_id}/decisions/{decision_id}/cancel",
    response_model=OrchestrationAuthorityDecisionResponse,
)
async def cancel_decision(
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    decision_id: uuid.UUID,
    data: OrchestrationDecisionCancelRequest,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _goal_or_404(db, project_id, goal_id)
    decision = await _decision_or_404(db, goal_id, decision_id)
    try:
        return await decision_service.cancel_decision(db, decision, reason=data.reason)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
