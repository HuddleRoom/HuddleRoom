"""Durable ordinary human-answer acceptance and source-process application."""

import logging
import uuid
from contextvars import ContextVar
from contextlib import contextmanager
from copy import deepcopy

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import SessionTransactionOrigin

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationRun, OrchestrationGoal
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision, OrchestrationProcessRun
from huddleroom.schemas.orchestration import AnswerSubmission, AnswerReceipt, AnswerApplicationPass
from huddleroom.services.orchestration_control import acquire_control_writer

logger = logging.getLogger(__name__)

_application = ContextVar("authority_answer_application", default=None)


def receipt(action):
    contract = action.dispatch_contract or {}
    submission = action.request["submission"]
    return AnswerReceipt(action_id=action.id, decision_ids=submission["decision_ids"],
                         actor_user_id=submission["actor_user_id"], run_id=submission["run_id"],
                         source_process_run_id=submission.get("source_process_run_id"),
                         submission=submission, contracts=submission["contracts"],
                         application_state=contract.get("application_state", "pending"),
                         reason=contract.get("reason"))


def runless_receipt(decision):
    """Context metadata is evidence only when it matches the actual native answered row."""
    import json
    try:
        context = json.loads(decision.context or "null")
        saved = context.get("_ordinary_answer_acceptance") if isinstance(context, dict) else None
        if not isinstance(saved, dict) or saved.get("kind") != "ordinary_runless_answer":
            return None
        submission = saved["submission"]
        member_ids = submission["decision_ids"]
        if (decision.status != "answered" or decision.decided_at is None or decision.authority != "human"
                or submission.get("kind") != "single" or len(member_ids) != 1
                or len(submission["answers"]) != len(member_ids) or len(submission["contracts"]) != len(member_ids)
                or decision.run_id is not None or decision.source_process_run_id is not None
                or submission.get("run_id") is not None or submission.get("source_process_run_id") is not None
                or len(member_ids) != len(set(member_ids)) or str(decision.id) not in member_ids
                or decision.decided_by_user_id is None or decision.decided_by_agent_id is not None
                or str(decision.decided_by_user_id) != submission["actor_user_id"]):
            return None
        index = member_ids.index(str(decision.id))
        answer = submission["answers"][index]
        if (answer["decision_id"] != str(decision.id) or answer["selected_option"] != decision.selected_option
                or answer.get("reason") != decision.reason or submission["contracts"][index] != contract(decision)):
            return None
        return AnswerReceipt(action_id=None, run_id=None, decision_ids=member_ids,
            actor_user_id=submission["actor_user_id"], source_process_run_id=None,
            submission=submission, contracts=submission["contracts"], application_state="applied")
    except (ValueError, TypeError, KeyError, IndexError):
        return None


def contract(decision):
    return {"id": str(decision.id), "key": decision.decision_key, "options": decision.options,
            "authority": decision.authority, "authority_agent_id": str(decision.authority_agent_id or ""),
            "source_process_run_id": str(decision.source_process_run_id or ""),
            "run_id": str(decision.run_id or ""), "contract_version": decision.contract_version}


def canonical_answer_submission(decisions, data, actor_user_id):
    """Pure reviewed input identity; generated native identities are deliberately excluded."""
    import json
    is_batch = hasattr(data, "answers")
    items = data.answers if is_batch else [data]
    answers = []
    for decision, item in zip(decisions, items):
        reason = item.reason
        if is_batch and item.selected_option == "edit":
            reason = json.dumps({"description": item.edited_description.strip(),
                                 "persona": item.edited_persona.strip()})
        value = {"decision_id": str(decision.id), "selected_option": item.selected_option, "reason": reason}
        if not is_batch:
            value.update(contract_version=item.contract_version,
                         edited_agent=item.edited_agent.model_dump(mode="json") if item.edited_agent else None)
        answers.append(value)
    return AnswerSubmission(kind="batch" if is_batch else "single", actor_user_id=actor_user_id, decision_ids=[d.id for d in decisions],
        source_process_run_id=decisions[0].source_process_run_id, run_id=decisions[0].run_id,
        answers=answers, contracts=[contract(d) for d in decisions])


@contextmanager
def application_context(action_id):
    token = _application.set(action_id)
    try:
        yield
    finally:
        _application.reset(token)


async def record_answer_effect(db, application_id, effect_key, target_type, target_id):
    """Link a native logical effect before its publication, in that same transaction."""
    action_id = application_id
    if action_id is None:
        return
    action = await db.get(OrchestrationAction, action_id, populate_existing=True)
    effects = dict((action.dispatch_contract or {}).get("effects", {}))
    value = {"target_type": target_type, "target_id": str(target_id)}
    if effect_key in effects and effects[effect_key] != value:
        raise HTTPException(409, "answer application effect identity changed")
    effects[effect_key] = value
    action.dispatch_contract = {**(action.dispatch_contract or {}), "effects": effects}
    await db.flush()


async def record_current_answer_effect(db, effect_key, target_type, target_id):
    await record_answer_effect(db, _application.get(), effect_key, target_type, target_id)


async def accept_answer(db, project_id, goal_id, submission, user, *, validation_callback=None,
                        reservation_callback=None, completion_callback=None):
    """Accept a frozen scoped ordinary submission; this is not a second HTTP answer API."""
    from huddleroom.services.orchestration_decision_application import validate_ordinary_submission
    from huddleroom.services.orchestration_recovery_presenter import present_decision_response, present_decision_responses
    from huddleroom.schemas.orchestration import (
        OrchestrationDecisionAnswerResponse, OrchestrationAgentDefinitionReviewBatchAnswerResponse,
    )
    owns = acceptance_owns(db)
    try:
        if submission.actor_user_id != user.id:
            raise HTTPException(409, "answer submission actor changed")
        member_ids = submission.decision_ids
        if (not member_ids or len(member_ids) != len(set(member_ids))
                or (submission.kind == "single" and len(member_ids) != 1)
                or len(submission.answers) != len(member_ids) or len(submission.contracts) != len(member_ids)):
            raise HTTPException(409, "answer submission member shape changed")
        goal = await db.get(OrchestrationGoal, goal_id)
        if goal is None or goal.project_id != project_id:
            raise HTTPException(404, "Orchestration goal not found")
        decisions = [await db.get(OrchestrationAuthorityDecision, value) for value in submission.decision_ids]
        if not decisions or any(d is None or d.goal_id != goal_id for d in decisions):
            raise HTTPException(404, "Orchestration authority decision not found")
        answers = deepcopy(submission.answers)
        if len(answers) != len(decisions) or any(answer.get("decision_id") != str(d.id)
                                               for d, answer in zip(decisions, answers)):
            raise HTTPException(409, "answer submission member identity changed")
        async def validate():
            await validate_ordinary_submission(db, goal, decisions, submission, answers)
        async def response(items, value):
            if submission.kind == "batch":
                return OrchestrationAgentDefinitionReviewBatchAnswerResponse(
                    decisions=await present_decision_responses(db, items), answer_receipt=value)
            return OrchestrationDecisionAnswerResponse(decision=await present_decision_response(db, items[0]),
                                                       answer_receipt=value)
        return await _accept_answer(db, decisions, answers, user, owns=owns, submission_input=submission,
            validate=validate, validation_callback=validation_callback, reservation_callback=reservation_callback,
            completion_callback=completion_callback, make_response=response)
    except BaseException as exc:
        if owns:
            await db.rollback()
            stale = getattr(exc, "stale_inputs", None)
            if stale:  # rejected stale proposal: keep only the one-time rerun suggestion
                # Callers that own the transaction themselves (e.g. chat actions) get no warning
                # here; the tick/advance path re-suggests it.
                try:
                    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService
                    await OrchestrationWarningService().suggest_stale_inputs(
                        db, stale["goal_id"], process_type="team_hierarchy",
                        process_run_id=stale["process_run_id"], step_label="team hierarchy", run_id=stale["run_id"])
                    await db.commit()
                except Exception:
                    logger.warning("stale-inputs suggestion failed", exc_info=True)
                    await db.rollback()
        raise


async def _accept_answer(db, decisions, answers, user, *, owns, submission_input=None, validate=None,
                        validation_callback=None, reservation_callback=None, completion_callback=None,
                        make_response=None):
    """Answers, obligation, event and request receipt share one writer transaction."""
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
    first = decisions[0]
    linked = await db.get(OrchestrationAction, first.continuation_action_id) if first.continuation_action_id else None
    source = await db.get(OrchestrationProcessRun, first.source_process_run_id) if first.source_process_run_id else None
    run_id = linked.run_id if linked and linked.action_type == "apply_authority_answer" else first.run_id
    if run_id is None and source is not None:
        run_id = source.run_id
    if run_id is None and first.source_process_run_id is None:
        run_id = await db.scalar(select(OrchestrationRun.id).where(OrchestrationRun.goal_id == first.goal_id)
                                 .order_by(OrchestrationRun.created_at.desc()).limit(1))
    existing_runless = runless_receipt(first)
    if existing_runless is not None or (run_id is None and first.source_process_run_id is None):
        return await _accept_runless_answer(db, decisions, answers, user, owns=owns,
            submission=submission_input, validate=validate, validation_callback=validation_callback,
            reservation_callback=reservation_callback, completion_callback=completion_callback,
            make_response=make_response)
    if run_id is None:
        raise HTTPException(409, "answer application requires an orchestration run")
    await acquire_control_writer(db, run_id)
    run = await db.get(OrchestrationRun, run_id, populate_existing=True)
    if run is None or run.goal_id != first.goal_id:
        raise HTTPException(409, "answer run scope changed")
    for decision in decisions:
        await db.refresh(decision)
    canonical = submission_input
    if canonical.actor_user_id != user.id or canonical.contracts != [contract(d) for d in decisions]:
        raise HTTPException(409, "answer submission contract changed")
    if canonical.source_process_run_id != first.source_process_run_id or canonical.run_id not in {None, run_id}:
        raise HTTPException(409, "answer submission source identity changed")
    submission = canonical.model_copy(update={"run_id": run_id}).model_dump(mode="json")
    key = "apply_authority_answer:" + ":".join(sorted(str(d.id) for d in decisions))
    # The deterministic key is bounded independently of batch size.
    if len(key) > 255:
        import hashlib
        key = "apply_authority_answer:" + hashlib.sha256(key.encode()).hexdigest()
    action = await db.scalar(select(OrchestrationAction).where(OrchestrationAction.run_id == run_id,
                                                             OrchestrationAction.idempotency_key == key))
    if action is not None:
        if action.request.get("submission") != submission:
            raise HTTPException(409, "answer submission conflicts with saved answer")
        response = await make_response(decisions, receipt(action))
        if owns:
            await db.commit()
        return response
    nested = await db.begin_nested()
    try:
        from huddleroom.services.orchestration_baseline_runtime import require_baseline_available
        require_baseline_available(first.goal_id)
        if validation_callback is not None:
            await validation_callback(db)
        if first.source_process_run_id is not None:
            source = await db.get(OrchestrationProcessRun, first.source_process_run_id, populate_existing=True)
            if source is None or source.goal_id != first.goal_id or source.run_id != run_id:
                raise HTTPException(409, "answer source run changed")
            if source.superseded_by_id is not None:
                raise HTTPException(409, "answer source process is superseded")
            if any(d.source_process_run_id != source.id or d.run_id not in {None, run_id} for d in decisions):
                raise HTTPException(409, "answer batch source identity changed")
        if validate is not None:
            await validate()
        action = OrchestrationAction(run_id=run_id, action_type="apply_authority_answer",
            idempotency_key=key, request={"submission": deepcopy(submission), "accepted_answers": deepcopy(answers)},
            dispatch_contract={"application_state": "pending", "effects": {}}, status="reserved")
        db.add(action)
        await db.flush()
        for decision, answer in zip(decisions, answers):
            await OrchestrationAuthorityDecisionService().answer_decision(db, decision,
                selected_option=answer["selected_option"], reason=answer.get("answer_reason", answer.get("reason")),
                decided_by_user_id=user.id, control_writer=True)
            decision.continuation_action_id = action.id
            if decision.decision_key.startswith("team_hierarchy:agent:") and answer.get("answer_reason"):
                import json
                generated_id = json.loads(answer["answer_reason"]).get("created_agent_id")
                if generated_id:
                    await record_answer_effect(db, action.id, f"created_agent_id:{decision.id}",
                                               "agent", uuid.UUID(generated_id))
        if reservation_callback is not None:
            await reservation_callback(db, {"kind": "answer", "action_id": str(action.id),
                "decision_ids": [str(d.id) for d in decisions], "run_id": str(run_id)})
        response = await make_response(decisions, receipt(action))
        if completion_callback is not None:
            await completion_callback(db, response)
        await db.flush()
        await nested.commit()
        if owns:
            await db.commit()
        return response
    except BaseException:
        if nested.is_active:
            await nested.rollback()
        if owns:
            await db.rollback()
        raise


async def _accept_runless_answer(db, decisions, answers, user, *, owns, submission, validate,
                                  validation_callback, reservation_callback, completion_callback, make_response):
    import json
    from huddleroom.services.project_service import ProjectService
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
    from huddleroom.services.orchestration_baseline_runtime import require_baseline_available
    goal = await db.get(OrchestrationGoal, decisions[0].goal_id)
    await ProjectService().lock_workspace_boundary(db, goal.project_id)
    for decision in decisions:
        await db.refresh(decision)
    canonical = submission.model_dump(mode="json")
    saved = runless_receipt(decisions[0])
    if saved is not None:
        if saved.submission != canonical or any(runless_receipt(d) != saved for d in decisions):
            raise HTTPException(409, "answer submission conflicts with saved answer")
        response = await make_response(decisions, saved)
        if owns:
            await db.commit()
        return response
    if (submission.actor_user_id != user.id or submission.contracts != [contract(d) for d in decisions]
            or submission.run_id is not None or submission.source_process_run_id is not None
            or any(d.run_id is not None or d.source_process_run_id is not None for d in decisions)
            or await db.scalar(select(OrchestrationRun.id).where(OrchestrationRun.goal_id == goal.id).limit(1)) is not None):
        raise HTTPException(409, "runless answer scope changed")
    require_baseline_available(goal.id)
    nested = await db.begin_nested()
    try:
        if validation_callback is not None:
            await validation_callback(db)
        await validate()
        for decision, answer in zip(decisions, answers):
            await OrchestrationAuthorityDecisionService().answer_decision(db, decision,
                selected_option=answer["selected_option"], reason=answer.get("reason"),
                decided_by_user_id=user.id, control_writer=True)
            try:
                original = json.loads(decision.context or "null")
            except (ValueError, TypeError):
                original = None
            context = original if isinstance(original, dict) else {"_original_context": decision.context}
            decision.context = json.dumps({**context, "_ordinary_answer_acceptance": {
                "kind": "ordinary_runless_answer", "submission": canonical}}, sort_keys=True)
        if reservation_callback is not None:
            await reservation_callback(db, {"kind": "answer", "action_id": None,
                "decision_ids": [str(d.id) for d in decisions], "run_id": None})
        response = await make_response(decisions, runless_receipt(decisions[0]))
        if completion_callback is not None:
            await completion_callback(db, response)
        await db.flush()
        await nested.commit()
        if owns:
            await db.commit()
        return response
    except BaseException:
        if nested.is_active:
            await nested.rollback()
        if owns:
            await db.rollback()
        raise


def acceptance_owns(db, request_owns_transaction=False):
    root = db.sync_session.get_transaction()
    return root is None or (request_owns_transaction and not db.in_nested_transaction()
                            and root.origin == SessionTransactionOrigin.AUTOBEGIN)


async def retry_answer_application(db, application_id):
    """Explicitly requeue one failed saved obligation after its native guards pass."""
    owns = acceptance_owns(db)
    try:
        return await _retry_answer_application(db, application_id, owns=owns)
    except BaseException:
        if owns:
            await db.rollback()
        raise


async def _retry_answer_application(db, application_id, *, owns):
    from huddleroom.services.orchestration_baseline_runtime import (
        require_baseline_available, require_control_admission, BaselinePaused,
    )
    action = await db.get(OrchestrationAction, application_id)
    if action is None or action.action_type != "apply_authority_answer":
        raise HTTPException(404, "Answer application not found")
    await acquire_control_writer(db, action.run_id)
    await db.refresh(action)
    run = await db.get(OrchestrationRun, action.run_id, populate_existing=True)
    require_baseline_available(run.goal_id)
    try:
        await require_control_admission(db, run.id)
    except BaselinePaused:
        raise HTTPException(409, "Orchestration admission paused") from None
    source_id = action.request["submission"].get("source_process_run_id")
    if source_id:
        source = await db.get(OrchestrationProcessRun, uuid.UUID(source_id), populate_existing=True)
        effects = (action.dispatch_contract or {}).get("effects", {})
        completed = f"complete_process:{source_id}" in effects
        successor = effects.get(f"source_successor:{source.process_type}") if source else None
        linked_successor = await db.get(OrchestrationProcessRun, uuid.UUID(successor["target_id"])) if successor else None
        if (source is None or source.run_id != run.id or source.goal_id != run.goal_id
                or (source.superseded_by_id is not None and not completed
                    and (linked_successor is None or linked_successor.run_id != run.id
                         or linked_successor.goal_id != run.goal_id
                         or source.superseded_by_id != linked_successor.id))):
            raise HTTPException(409, "Answer source process changed")
        if completed:
            action.status = "completed"
            action.dispatch_contract = {**(action.dispatch_contract or {}), "application_state": "applied", "reason": None}
            from datetime import datetime, timezone
            for decision_id in action.request["submission"]["decision_ids"]:
                decision = await db.get(OrchestrationAuthorityDecision, uuid.UUID(decision_id))
                decision.continuation_applied_at = decision.continuation_applied_at or datetime.now(timezone.utc)
    if action.status == "failed":
        action.status = "reserved"
        action.dispatch_contract = {**(action.dispatch_contract or {}), "application_state": "pending", "reason": None}
        await db.flush()
    response = receipt(action)
    if owns:
        await db.commit()
    return response


async def drain_answer_applications(db, run_id):
    """Consume saved ordinary answers; never repeat request callbacks or human acceptance."""
    from huddleroom.services.orchestration_service import OrchestrationService
    from huddleroom.services.orchestration_debug_service import OrchestrationDebugService, SUPPORTED_PROCESS_TYPES
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_baseline_runtime import baseline_operation, require_control_admission
    from huddleroom.services.orchestration_roster_mutation import RosterHandoffRequired, commit_before_roster_handoff
    from huddleroom.services.agent_service import complete_roster_handoff
    service = OrchestrationService()
    owns = await service._prepare_control_operation(db, run_id=run_id)
    result = AnswerApplicationPass()
    ids = list(await db.scalars(select(OrchestrationAction.id).where(OrchestrationAction.run_id == run_id,
        OrchestrationAction.action_type == "apply_authority_answer", OrchestrationAction.status == "reserved")
        .order_by(OrchestrationAction.created_at, OrchestrationAction.id).limit(32)))
    if owns:
        await db.commit()
    for action_id in ids:
        run = await db.get(OrchestrationRun, run_id)
        goal_id = run.goal_id
        if owns:
            await db.commit()
        async def apply(_prepared=None):
            with application_context(action_id):
                async with service._lock_goal_for_baseline_transition(db, goal_id), \
                        baseline_operation(db, goal_id, owns_transaction=owns, tick=True, run_id=run_id), \
                        commit_before_roster_handoff(db, owns):
                    await require_control_admission(db, run_id)
                    if _prepared is not None:
                        from huddleroom.services.orchestration_roster_mutation import revalidate_roster_handoff
                        await revalidate_roster_handoff(db, _prepared)
                    action = await db.get(OrchestrationAction, action_id, populate_existing=True)
                    if action.status != "reserved":
                        return
                    submission = action.request["submission"]
                    source_id = submission.get("source_process_run_id")
                    source = await db.get(OrchestrationProcessRun, uuid.UUID(source_id), populate_existing=True) if source_id else None
                    goal = await db.get(OrchestrationGoal, goal_id, populate_existing=True)
                    run = await db.get(OrchestrationRun, run_id, populate_existing=True)
                    if source_id and source is None:
                        action.dispatch_contract = {**(action.dispatch_contract or {}),
                            "application_state": "blocked", "reason": "source_process_missing"}
                        action.status = "failed"
                        return
                    if source is not None and source.process_type in SUPPORTED_PROCESS_TYPES:
                        source_id_value = source.id
                        linked_successor = False
                        current = await OrchestrationProcessService().get_current(db, goal_id, source.process_type)
                        effects = (action.dispatch_contract or {}).get("effects", {})
                        if current is None or current.id != source.id:
                            successor = effects.get(f"source_successor:{source.process_type}")
                            if (current is not None and successor is not None
                                    and successor["target_id"] == str(current.id) and current.run_id == run_id):
                                # Resume only the successor already linked by this application's native write.
                                source = current
                                linked_successor = True
                            elif f"complete_process:{source.id}" not in effects:
                                action.dispatch_contract = {**(action.dispatch_contract or {}),
                                    "application_state": "blocked", "reason": "source_process_superseded"}
                                action.status = "failed"
                                return
                        if (
                            not linked_successor
                            and f"complete_process:{source.id}" not in effects
                            and source.superseded_by_id is None
                        ):
                            with application_context(action_id):
                                if source.process_type == "team_hierarchy":
                                    from huddleroom.services.orchestration_team_hierarchy import (
                                        TeamHierarchyProcess, PROPOSAL_RESOLUTION_RERUN_SENTINEL,
                                    )
                                    if any(item["selected_option"] in {"edit", "reject"}
                                           for item in submission["answers"]):
                                        await TeamHierarchyProcess()._cancel_pending_for_run(db, goal_id, source.id)
                                        source.input_snapshot = {"fingerprint": PROPOSAL_RESOLUTION_RERUN_SENTINEL}
                                await OrchestrationDebugService().advance_process(db, goal, run, source.process_type)
                                await db.refresh(action)
                                effects = dict((action.dispatch_contract or {}).get("effects", {}))
                        original = await db.get(OrchestrationProcessRun, source_id_value, populate_existing=True)
                        successor = ((action.dispatch_contract or {}).get("effects", {})
                                     .get(f"source_successor:{source.process_type}"))
                        current = await OrchestrationProcessService().get_current(db, goal_id, source.process_type)
                        linked_successor = linked_successor or (
                            original is not None
                            and isinstance(successor, dict)
                            and successor.get("target_id") == str(current.id if current is not None else "")
                            and current is not None
                            and current.run_id == run_id
                            and original.superseded_by_id == current.id
                        )
                        completed = f"complete_process:{source_id_value}" in (
                            (action.dispatch_contract or {}).get("effects", {})
                        ) or (original is not None and original.status in {"completed", "skipped"})
                        continuation = effects.get(f"continuation_decision:{source_id_value}")
                        hierarchy_continuation = False
                        if (
                            source.process_type == "team_hierarchy"
                            and isinstance(continuation, dict)
                            and continuation.get("target_type") == "decision"
                            and isinstance(continuation.get("target_id"), str)
                            and current is not None
                            and current.id == source_id_value
                            and current.run_id == run_id
                        ):
                            try:
                                target_id = uuid.UUID(continuation["target_id"])
                            except (TypeError, ValueError):
                                target_id = None
                            target = await db.get(OrchestrationAuthorityDecision, target_id) if target_id else None
                            accepted = await db.get(
                                OrchestrationAuthorityDecision, uuid.UUID(submission["decision_ids"][0])
                            )
                            hierarchy_continuation = bool(
                                target is not None
                                and accepted is not None
                                and accepted.decision_key.startswith("team_hierarchy:agent:")
                                and target.decision_key == "team_hierarchy:approval"
                                and target.goal_id == goal_id
                                and target.run_id == run_id
                                and target.source_process_run_id == source_id_value
                                and target.status in {"pending", "answered"}
                            )
                        if not completed and not linked_successor and not hierarchy_continuation:
                            action.dispatch_contract = {**(action.dispatch_contract or {}),
                                "application_state": "blocked", "reason": "source_process_did_not_advance"}
                            action.status = "failed"
                            return
                    action.dispatch_contract = {**(action.dispatch_contract or {}), "application_state": "applied"}
                    action.status = "completed"
                    from datetime import datetime, timezone
                    for decision_id in submission["decision_ids"]:
                        decision = await db.get(OrchestrationAuthorityDecision, uuid.UUID(decision_id), populate_existing=True)
                        decision.continuation_applied_at = datetime.now(timezone.utc)
                    await db.flush()
        try:
            try:
                await apply()
            except RosterHandoffRequired as handoff:
                with application_context(action_id):
                    await complete_roster_handoff(db, handoff.prepared, apply)
            if owns:
                await db.commit()
            action = await db.get(OrchestrationAction, action_id, populate_existing=True)
            (result.applied if action.status == "completed" else result.failed).append(action_id)
            if (action.dispatch_contract or {}).get("reason"):
                result.reasons[str(action_id)] = action.dispatch_contract["reason"]
        except Exception as exc:
            if not owns:
                raise  # An external caller alone decides the outcome of its transaction.
            await db.rollback()
            from huddleroom.services.orchestration_llm_decision_adapter import _safe_completion_error
            if isinstance(exc, HTTPException):
                reason = str(exc.detail)[:1000]
                retryable = exc.status_code == 409 and exc.detail in (
                    "Orchestration admission paused", "Baseline operation discarded by reset",
                )
            else:
                reason = _safe_completion_error(exc)
                retryable = isinstance(exc, OperationalError) and any(s in str(exc).lower() for s in ("locked", "busy"))  # e.g. "database is locked": retry on next drain
            await acquire_control_writer(db, run_id)
            action = await db.get(OrchestrationAction, action_id, populate_existing=True)
            if action.status != "completed":
                action.dispatch_contract = {**(action.dispatch_contract or {}),
                    "application_state": "pending" if retryable else "failed", "reason": reason}
                action.status = "reserved" if retryable else "failed"
            await db.commit()
            (result.pending if retryable else result.failed).append(action_id)
            result.reasons[str(action_id)] = reason
        if owns:
            await db.commit()
    return result
