"""Shared authority-answer application, preserving specialized validation and continuation."""

import json
import uuid

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
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
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_team_hierarchy import TeamHierarchyProcess
from huddleroom.services.orchestration_team_hierarchy_analyzer import canonical_agent_create
from huddleroom.services.agent_service import AgentService

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
    request_owns_transaction=False,
) -> OrchestrationAgentDefinitionReviewBatchAnswerResponse:
    from huddleroom.services.orchestration_answer_receipts import accept_answer, acceptance_owns, canonical_answer_submission
    owns = acceptance_owns(db, request_owns_transaction)
    await _goal_or_404(db, project_id, goal_id)
    submitted_ids = [item.decision_id for item in data.answers]
    if len(submitted_ids) != len(set(submitted_ids)):
        raise HTTPException(400, "decision IDs must be unique")
    decisions = [await _decision_or_404(db, goal_id, value) for value in submitted_ids]
    answers = []
    for item in data.answers:
        edit_fields = {"edited_description", "edited_persona"} & item.model_fields_set
        if item.selected_option == "edit":
            if edit_fields != {"edited_description", "edited_persona"} or not all(
                isinstance(value, str) and value.strip() for value in (item.edited_description, item.edited_persona)
            ):
                raise HTTPException(400, "edit requires description and persona")
        elif edit_fields:
            raise HTTPException(400, "edited fields require edit")
        answers.append({"decision_id": str(item.decision_id), "selected_option": item.selected_option,
                        "reason": _batch_answer_reason(item)})

    submission = canonical_answer_submission(decisions, data, user.id)
    if owns:
        await db.commit()  # End only this invocation's preflight before the acceptance primitive owns its root.
    try:
        return await accept_answer(db, project_id, goal_id, submission, user,
            validation_callback=validation_callback, reservation_callback=reservation_callback,
            completion_callback=completion_callback)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


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
    request_owns_transaction=False,
) -> OrchestrationDecisionAnswerResponse:
    from huddleroom.services.orchestration_answer_receipts import accept_answer, acceptance_owns, canonical_answer_submission
    owns = acceptance_owns(db, request_owns_transaction)
    await _goal_or_404(db, project_id, goal_id)
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
                    contract_version=data.contract_version, reason=data.reason,
                )
                response = OrchestrationDecisionAnswerResponse(
                    decision=await _present_one(db, result.decision), process=None, continuation_applied=result.continuation_applied
                )
                if completion_callback is not None:
                    await completion_callback(db, response)
                return response
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    submission = canonical_answer_submission([decision], data, user.id)
    if owns:
        await db.commit()
    try:
        return await accept_answer(db, project_id, goal_id, submission, user,
            validation_callback=validation_callback, reservation_callback=reservation_callback,
            completion_callback=completion_callback)
    except (ValueError, ValidationError) as exc:
        raise HTTPException(400, str(exc)) from exc


async def validate_ordinary_submission(db, goal, decisions, submission, answers):
    """Fresh domain guards for the canonical ordinary acceptance writer transaction."""
    for decision in decisions:
        if is_recovery_decision(decision) or decision.runtime_identity is not None:
            raise HTTPException(409, "decision requires its specialized answer path")
        if decision.authority != "human":
            if decision.authority_agent_id is None:
                raise HTTPException(409, "the agent this decision awaits was deleted; cancel it via the cancel endpoint so its source process can re-raise a fresh decision")
            raise HTTPException(409, "decision must be answered through its designated agent")
    if submission.kind == "batch":
        current = await OrchestrationProcessService().get_current(db, goal.id, "agent_definition_review")
        if current is None or current.status != "waiting_decision":
            raise HTTPException(409, "agent-definition review is not awaiting a batch")
        pending = [d for d in await decision_service.list_decisions(db, goal.id, status="pending")
                   if d.source_process_run_id == current.id
                   and d.decision_key.startswith("agent_definition_review:proposal:")]
        if set(submission.decision_ids) != {d.id for d in pending}:
            raise HTTPException(409, "pending proposal set changed")
    for decision, answer in zip(decisions, answers):
        if not decision.decision_key.startswith("team_hierarchy:agent:"):
            continue
        current = await OrchestrationProcessService().get_current(db, goal.id, "team_hierarchy")
        if current is None or current.id != decision.source_process_run_id:
            raise HTTPException(409, "team hierarchy proposal is no longer current")
        hierarchy = TeamHierarchyProcess()
        fingerprint = hierarchy._semantic_fingerprint(hierarchy._strip_volatile(await hierarchy._analysis_input(db, goal)))
        if current.input_snapshot.get("fingerprint") != fingerprint:
            stale = HTTPException(409, "team hierarchy proposal inputs changed")
            # The accepting owner persists the rerun suggestion after rolling this rejection back.
            stale.stale_inputs = {"goal_id": goal.id, "process_run_id": current.id, "run_id": current.run_id}
            raise stale
        if answer["selected_option"] == "edit":
            from huddleroom.schemas.agent import AgentCreate
            edited_agent = canonical_agent_create(AgentCreate.model_validate(answer["edited_agent"]))
            AgentService.validate_create(edited_agent)
            answer["answer_reason"] = json.dumps({"reason": answer.get("reason"), "created_agent_id": str(uuid.uuid4()),
                "edited_agent": edited_agent.model_dump(mode="json")}, sort_keys=True)


from huddleroom.services.orchestration_answer_receipts import (  # noqa: E402
    accept_answer, canonical_answer_submission, drain_answer_applications, record_answer_effect, retry_answer_application,
)
