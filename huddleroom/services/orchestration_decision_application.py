"""Shared authority-answer application, preserving specialized validation and continuation."""

import json
import uuid

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision, OrchestrationProcessRun
from huddleroom.models.user import User
from huddleroom.schemas.orchestration import (
    OrchestrationAgentDefinitionReviewBatchAnswer,
    OrchestrationAgentDefinitionReviewBatchAnswerRequest,
    OrchestrationAgentDefinitionReviewBatchAnswerResponse,
    OrchestrationDecisionAnswerResponse,
    OrchestrationDecisionAnswerRequest,
)
from huddleroom.services.orchestration_blocker_recovery import (
    OrchestrationBlockerRecoveryService, RecoveryDiagnosisFailed, RecoveryProposalStale,
)
from huddleroom.services.orchestration_recovery_presenter import (
    GAP_PREFIX, is_recovery_decision, present_decision_response, present_decision_responses,
)
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
from huddleroom.services.agent_service import AgentService, complete_roster_handoff
from huddleroom.services.orchestration_roster_mutation import (
    RosterHandoffRequired,
    commit_before_roster_handoff,
    publish_roster_reservation,
    revalidate_roster_handoff,
    transaction_shape,
)
from huddleroom.services.project_service import ProjectService

orchestration_service = OrchestrationService()
decision_service = OrchestrationAuthorityDecisionService()
_present = present_decision_responses
_present_one = present_decision_response


async def _answer_recovery(db, goal_id, decision, data, user, reservation_callback=None):
    """Typed recovery answer: runs before (and instead of) the generic path; no effects run in the request."""
    if data.reason is not None and not data.reason.strip() and data.selected_option != "reject":
        data.reason = None
    if data.edited_agent is not None or {"edited_description", "edited_persona"} & data.model_fields_set:
        raise HTTPException(status_code=400, detail="edit fields are not valid for recovery decisions")
    service = OrchestrationBlockerRecoveryService()
    is_gap = decision.decision_key.startswith(GAP_PREFIX)
    allowed = ("capability_provided",) if is_gap else ("approve_fix", "reject")
    if data.selected_option not in allowed:
        raise HTTPException(status_code=400, detail=f"selected option must be one of: {', '.join(allowed)}")
    if data.selected_option == "reject" and not (data.reason and data.reason.strip()):
        raise HTTPException(status_code=400, detail="rejecting a fix proposal requires a reason")
    try:
        if is_gap:
            reservation = await service.answer_capability_gap(db, decision, user_id=user.id)
        else:
            reservation = await service.reserve_answer(
                db, decision, selected_option=data.selected_option, reason=data.reason, user_id=user.id)
    except (RecoveryProposalStale, RecoveryDiagnosisFailed) as exc:
        await db.commit()  # keep the flushed stale/diagnosis writes: get_db would roll back on the 409
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if reservation_callback is not None:
        await reservation_callback(db, {"kind": "recovery", "decision_ids": [str(decision.id)],
            "proposal_action_id": str(getattr(reservation, "proposal_action_id", getattr(reservation, "id", ""))),
            "wrapper_action_id": str(getattr(reservation, "first_wrapper_action_id", "") or "")})
    await db.refresh(decision)
    return OrchestrationDecisionAnswerResponse(decision=await _present_one(db, decision), process=None)


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


async def answer_agent_review_batch(
    db: AsyncSession,
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    data: OrchestrationAgentDefinitionReviewBatchAnswerRequest,
    user: User,
    *,
    reservation_callback=None,
    validation_callback=None,
    completion_callback=None,
) -> OrchestrationAgentDefinitionReviewBatchAnswerResponse:
    goal = await _goal_or_404(db, project_id, goal_id)
    owns_transaction = transaction_shape(db) in {"none", "autobegin"}
    async with orchestration_service._lock_goal_for_baseline_transition(db, goal.id):
        if validation_callback is not None:
            await validation_callback(db)
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
        prepared = None
        try:
            async with db.begin_nested():
                answered = [await decision_service.answer_decision(
                    db,
                    pending_by_id[item.decision_id],
                    selected_option=item.selected_option,
                    reason=_batch_answer_reason(item),
                    decided_by_user_id=user.id,
                ) for item in data.answers]
                try:
                    process = await OrchestrationDebugService().advance_process(
                        db, goal, active_run, "agent_definition_review"
                    )
                except RosterHandoffRequired as handoff:
                    prepared = handoff.prepared  # keep the answers: they are the reservation
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if prepared is None:
            response = OrchestrationAgentDefinitionReviewBatchAnswerResponse(
                decisions=await _present(db, answered), process=process
            )
            if completion_callback is not None:
                await completion_callback(db, response)
            return response
        if reservation_callback is not None:
            await reservation_callback(db, {"kind": "roster", "process_id": str(prepared.process_id),
                "run_id": str(prepared.origin_run_id), "decision_ids": [str(value) for value in prepared.decision_ids]})
        await publish_roster_reservation(db, prepared, owns_transaction)

    async def resume(prepared):
        async with orchestration_service._lock_goal_for_baseline_transition(db, goal_id), \
                commit_before_roster_handoff(db, owns_transaction):
            await db.refresh(goal)
            if goal.status not in {"active", "blocked"}:
                raise HTTPException(status_code=409, detail="agent-definition review cannot advance")
            run = await revalidate_roster_handoff(db, prepared)
            await ProjectService().lock_workspace_boundary(db, project_id)
            await ProjectService().require_runnable_project(db, project_id)
            try:
                process = await OrchestrationDebugService().advance_process(
                    db, goal, run, "agent_definition_review"
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            for item in answered:
                await db.refresh(item)
            response = OrchestrationAgentDefinitionReviewBatchAnswerResponse(decisions=await _present(db, answered), process=process)
            if completion_callback is not None:
                await completion_callback(db, response)
            return response

    return await complete_roster_handoff(db, prepared, resume)


async def answer_decision(
    db: AsyncSession,
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    decision_id: uuid.UUID,
    data: OrchestrationDecisionAnswerRequest,
    user: User,
    *,
    reservation_callback=None,
    validation_callback=None,
    completion_callback=None,
) -> OrchestrationDecisionAnswerResponse:
    goal = await _goal_or_404(db, project_id, goal_id)
    decision = await _decision_or_404(db, goal_id, decision_id)
    if is_recovery_decision(decision):
        async with orchestration_service._lock_goal_for_baseline_transition(db, goal_id):
            if validation_callback is not None:
                await validation_callback(db)
            response = await _answer_recovery(db, goal_id, decision, data, user, reservation_callback)
            if completion_callback is not None:
                await completion_callback(db, response)
            return response
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
            async with orchestration_service._lock_goal_for_baseline_transition(db, goal_id):
                if validation_callback is not None:
                    await validation_callback(db)
                result = await decision_service.answer_runtime_question(
                    db, decision, data.selected_option, actor_user_id=user.id,
                    contract_version=data.contract_version,
                )
                response = OrchestrationDecisionAnswerResponse(
                    decision=await _present_one(db, result.decision), process=None, continuation_applied=result.continuation_applied
                )
                if completion_callback is not None:
                    await completion_callback(db, response)
                return response
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
    owns_transaction = transaction_shape(db) in {"none", "autobegin"}

    async def advance_source_process():
        if answered_decision.source_process_run_id is None:
            return None
        source_process = await db.get(OrchestrationProcessRun, answered_decision.source_process_run_id)
        if source_process is None or source_process.process_type not in SUPPORTED_PROCESS_TYPES:
            return None
        current = await OrchestrationProcessService().get_current(db, goal.id, source_process.process_type)
        if current is None or current.id != source_process.id:
            return None
        await db.refresh(goal)
        active_run = await orchestration_service.get_active_run_for_goal(db, project_id, goal_id)
        if goal.status not in {"active", "blocked"} or active_run is None:
            return None
        await db.refresh(active_run)
        if active_run.status not in LLM_DECISION_RUN_STATUSES:
            return None
        try:
            await ProjectService().lock_workspace_boundary(db, project_id)
            await ProjectService().require_runnable_project(db, project_id)
        except HTTPException:
            return None
        return await OrchestrationDebugService().advance_process(
            db, goal, active_run, source_process.process_type
        )

    try:
        async with OrchestrationService()._lock_goal_for_baseline_transition(db, decision.goal_id):
            if validation_callback is not None:
                await validation_callback(db)
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
                    # Reserve the agent under the goal lock; the hierarchy process
                    # creates it roster-first after this lock is released.
                    edited_agent = canonical_agent_create(data.edited_agent)
                    AgentService.validate_create(edited_agent)
                    answer_reason = json.dumps({
                        "reason": data.reason, "created_agent_id": str(uuid.uuid4()),
                        "edited_agent": edited_agent.model_dump(mode="json"),
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
            try:
                process = await advance_source_process()
            except RosterHandoffRequired as handoff:
                prepared = handoff.prepared
                if reservation_callback is not None:
                    await reservation_callback(db, {"kind": "roster", "process_id": str(prepared.process_id),
                        "run_id": str(prepared.origin_run_id), "decision_ids": [str(value) for value in prepared.decision_ids]})
                await publish_roster_reservation(db, prepared, owns_transaction)
            else:
                response = OrchestrationDecisionAnswerResponse(decision=await _present_one(db, answered_decision), process=process)
                if completion_callback is not None:
                    await completion_callback(db, response)
                return response

        async def resume(prepared):
            async with OrchestrationService()._lock_goal_for_baseline_transition(db, goal_id), \
                    commit_before_roster_handoff(db, owns_transaction):
                await revalidate_roster_handoff(db, prepared)
                await db.refresh(answered_decision)
                process = await advance_source_process()
                response = OrchestrationDecisionAnswerResponse(decision=await _present_one(db, answered_decision), process=process)
                if completion_callback is not None:
                    await completion_callback(db, response)
                return response

        return await complete_roster_handoff(db, prepared, resume)
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

