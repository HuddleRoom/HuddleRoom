"""Finite human action facade. Persisted provider output alone is never authority."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from sqlalchemy.orm import SessionTransactionOrigin
import dataclasses
import hashlib
import json
import re
import uuid

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select

from huddleroom.database import AsyncSessionLocal
from huddleroom.models.orchestration_chat import ProjectChatTurn
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.orchestration_steering import OrchestrationSteeringProposal
from huddleroom.models.user import User
from huddleroom.schemas.orchestration_chat import ChatActionDraft, ChatActionReceipt, ChatInterpretation
from huddleroom.schemas.orchestration import OrchestrationDecisionAnswerRequest, OrchestrationAgentDefinitionReviewBatchAnswerRequest
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_authority_service import effective_authorization
from huddleroom.services.orchestration_steering import OrchestrationSteeringService, SteeringDraft
from huddleroom.services.orchestration_chat_interpretation import hierarchy_instruction_text, parse_chat_interpretation
from huddleroom.services import orchestration_decision_application as decisions

ACTION_LOCKS = {}

@asynccontextmanager
async def no_goal_lock():
    yield

CONFIRM_OPERATIONS = {"start_execution", "restart_execution", "pause", "cancel"}
DECISION_OPERATIONS = {"answer_authority", "answer_runtime_question", "approve_recovery", "reject_recovery", "answer_hierarchy", "answer_agent_definition_batch"}


def conflict(code, message):
    return HTTPException(409, detail={"conflict": code, "message": message})


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


class OrchestrationChatActionService:
    def __init__(self, session_factory=AsyncSessionLocal, orchestration=None):
        self._session_factory = session_factory
        self._orchestration = orchestration or OrchestrationService()
        self._steering = OrchestrationSteeringService(self._orchestration)

    def _instruction(self, turn, action):
        if turn.project_id != action.project_id:
            raise conflict("chat_invalid_target", "Action belongs to another project")
        try:
            interpretation = ChatInterpretation.model_validate(turn.interpretation or {})
        except ValidationError as exc:
            raise conflict("chat_instruction_required", "Validated current instruction is required") from exc
        checked = parse_chat_interpretation(interpretation.model_dump_json(), turn.content, turn.context_goal_id)
        if checked.intent != "instruction" or checked.operation != action.operation or checked.goal_id != action.goal_id:
            raise conflict("chat_instruction_required", "A clear current instruction for this target is required")
        if checked.payload != action.payload:
            raise conflict("chat_instruction_required", "Action payload differs from the submitted instruction")
        instruction_content = hierarchy_instruction_text(turn.content, checked)
        text = instruction_content.casefold()
        if action.operation == "submit_steering" and action.payload.lifetime == "future_runs" and not re.search(r"\bfuture\s+runs?\b", text):
            raise conflict("chat_instruction_required", "Future-run direction requires explicit lifetime")
        if action.operation == "submit_steering" and action.payload.lifetime == "selected_item" and not re.search(r"\b(item|task)\s+only\b|\bonly\s+(this|the)\s+(item|task)\b", text):
            raise conflict("chat_instruction_required", "Item-only direction requires explicit lifetime")
        if action.operation == "submit_steering" and (not action.payload.directive or action.payload.directive.casefold() not in text):
            raise conflict("chat_instruction_required", "Steering must preserve the explicit submitted directive")
        if action.payload.target_type in {"task", "plan_item"} and (not action.payload.target_id or str(action.payload.target_id) not in text):
            raise conflict("chat_instruction_required", "Identify the intended item explicitly")
        if action.operation in {"promote_steering", "withdraw_steering", "dismiss_proposal"} and (not action.target_id or str(action.target_id) not in text):
            raise conflict("chat_instruction_required", "Identify the steering record explicitly")
        if action.operation == "start_baseline" and "baseline" not in text:
            raise conflict("chat_instruction_required", "Specify baseline authorization or execution Start")
        if action.operation in {"start_execution", "restart_execution"} and not re.search(r"\b(execution|work|spend)\b", text):
            raise conflict("chat_instruction_required", "Specify baseline authorization or execution Start")
        if action.operation in DECISION_OPERATIONS:
            command = re.split(r"\b(?:because|reason)\b", text, maxsplit=1)[0]
            if re.search(r"\b(not|never|dont|don't|unless|maybe|might|instead|if)\b", command):
                raise conflict("chat_instruction_required", "Clarify the intended answer without conflicting instructions")
            ids = [item.decision_id for item in action.payload.answers] if action.operation == "answer_agent_definition_batch" else [action.target_id]
            if not ids or any(item is None or str(item).casefold() not in text for item in ids):
                raise conflict("chat_instruction_required", "Identify each decision explicitly before answering")
            options = [item.selected_option for item in action.payload.answers] if action.operation == "answer_agent_definition_batch" else [action.payload.selected_option]
            aliases = {"approve": {"approve", "approve_fix", "accept"}, "accept": {"accept", "approve"}, "reject": {"reject"}}
            clauses = [part.strip() for part in re.split(r"[;\n]", command)]
            for decision_id, option in zip(ids, options):
                pair = rf"(?:answer|respond(?: to)?)\s+(?:decision\s+)?{re.escape(str(decision_id))}\s*[:=]?\s+{re.escape(option or '')}"
                natural = rf"(approve|accept|reject)\s+(?:decision\s+)?{re.escape(str(decision_id))}"
                matched = False
                for clause in clauses:
                    clause = re.sub(r"^please\s+", "", clause).strip().rstrip(".,")
                    if re.fullmatch(pair, clause):
                        matched = True
                    match = re.fullmatch(natural, clause)
                    if match and option in aliases[match.group(1)]:
                        matched = True
                    explicit_recovery = rf"(?:approve|reject)\s+(?:decision\s+)?{re.escape(str(decision_id))}\s+{re.escape(option or '')}"
                    if re.fullmatch(explicit_recovery, clause):
                        matched = True
                if not option or not matched:
                    raise conflict("chat_instruction_required", "Clarify the offered answer for each identified decision")
            verb = re.sub(r"^please\s+", "", text).split(" ", 1)[0]
            if action.operation == "approve_recovery" and verb != "approve" or action.operation == "reject_recovery" and verb != "reject":
                raise conflict("chat_instruction_required", "Approval and rejection must match the current instruction")
            if action.payload.reason and action.payload.reason.casefold() not in text:
                raise conflict("chat_instruction_required", "The reason must be explicitly supplied")
            edits = [action.payload.edited_description, action.payload.edited_persona]
            if action.payload.edited_agent is not None:
                if action.operation != "answer_hierarchy" or instruction_content == turn.content:
                    raise conflict("chat_instruction_required", "Supply the complete edited agent explicitly")
            for item in action.payload.answers:
                edits.extend([item.reason, item.edited_description, item.edited_persona])
            if any(value and value.casefold() not in text for value in edits):
                raise conflict("chat_instruction_required", "Required reasons and edits must be explicitly supplied")

    async def _validate_natural_options(self, db, turn, action):
        if action.operation not in DECISION_OPERATIONS:
            return
        from huddleroom.services.orchestration_authority_service import _option_keys
        aliases = {"approve": {"approve", "approve_fix", "accept"}, "accept": {"accept", "approve"}, "reject": {"reject"}}
        interpretation = ChatInterpretation.model_validate(turn.interpretation)
        instruction_content = hierarchy_instruction_text(turn.content, interpretation)
        text = re.split(r"\b(?:because|reason)\b", instruction_content.casefold(), maxsplit=1)[0]
        items = [(item.decision_id, item.selected_option) for item in action.payload.answers] if action.operation == "answer_agent_definition_batch" else [(action.target_id, action.payload.selected_option)]
        for decision_id, option in items:
            for clause in re.split(r"[;\n]", text):
                clause = re.sub(r"^please\s+", "", clause.strip()).rstrip(".,")
                match = re.fullmatch(rf"(approve|accept|reject)\s+(?:decision\s+)?{re.escape(str(decision_id))}", clause)
                if match:
                    decision = await db.get(OrchestrationAuthorityDecision, decision_id, populate_existing=True)
                    offered = _option_keys(decision.options) & aliases[match.group(1)] if decision else set()
                    if offered != {option}:
                        raise conflict("chat_instruction_required", "Approval answer is ambiguous among the offered options")

    async def _snapshot(self, db, action):
        goal = await self._orchestration.get_goal(db, action.project_id, action.goal_id)
        if goal is None:
            raise HTTPException(404, "Orchestration goal not found")
        await db.refresh(goal)
        run = await self._orchestration.get_active_run_for_goal(db, action.project_id, action.goal_id)
        if run is not None:
            await db.refresh(run)
        state = {"goal_status": goal.status, "run_id": str(run.id) if run else None,
                 "run_status": run.status if run else None, "phase": run.phase if run else None}
        if action.operation == "pause":
            state["control_version"] = run.control_version if run else None
        if action.operation in CONFIRM_OPERATIONS or action.operation == "start_baseline":
            state["contract"] = {key: getattr(goal, key, None) for key in (
                "goal_type", "planning_mode", "objective", "constraints", "budget", "authority_model",
                "manager_agent_id", "manager_user_id", "continuous_policy", "continuous_state")}
            if run:
                authorization = await effective_authorization(db, goal, run)
                state["authorization"] = dataclasses.asdict(authorization) if dataclasses.is_dataclass(authorization) else authorization.model_dump(mode="json")
        if action.operation in DECISION_OPERATIONS:
            ids = [item.decision_id for item in action.payload.answers] if action.operation == "answer_agent_definition_batch" else [action.target_id]
            state["decisions"] = []
            for decision_id in ids:
                decision = await db.get(OrchestrationAuthorityDecision, decision_id)
                if decision is None or decision.goal_id != goal.id:
                    raise HTTPException(404, "Authority decision not found")
                await db.refresh(decision)
                state["decisions"].append({key: getattr(decision, key, None) for key in (
                    "id", "status", "options", "authority", "runtime_identity", "source_process_run_id", "decision_key", "context")})
        if action.operation in {"submit_steering", "promote_steering"} and run:
            state["steering"] = dataclasses.asdict(await self._steering.current_versions(db, goal, run))
        return goal, run, state

    async def prepare(self, db, turn: ProjectChatTurn, action: ChatActionDraft, *, submission_validation=None) -> ChatActionDraft:
        self._instruction(turn, action)
        await self._validate_natural_options(db, turn, action)
        if turn.context_goal_id != action.goal_id or (turn.interpretation or {}).get("named_goal"):
            goal = await self._orchestration.get_goal(db, action.project_id, action.goal_id)
            named = (turn.interpretation or {}).get("named_goal")
            title = getattr(goal, "objective", None) if goal else None
            from huddleroom.models.orchestration import OrchestrationGoal
            matches = list((await db.scalars(select(OrchestrationGoal.id).where(
                OrchestrationGoal.project_id == action.project_id, OrchestrationGoal.objective == named))).all()) if named else []
            if not named or named.casefold() not in turn.content.casefold() or named != title or matches != [action.goal_id]:
                raise conflict("chat_invalid_target", "Named target must match the submitted goal explicitly")
        preparation_lock = no_goal_lock() if action.operation == "pause" else self._orchestration._lock_goal_for_baseline_transition(db, action.goal_id)
        async with preparation_lock:
            if submission_validation is not None:
                await submission_validation(db)
            _, run, state = await self._snapshot(db, action)
            if action.run_id is not None and (run is None or run.id != action.run_id):
                raise conflict("chat_stale_review", "Submitted run has changed")
            prepared = action.model_copy(deep=True)
            prepared.run_id = run.id if run else None
            prepared.frozen_versions = {"state": fingerprint(state)}
            if action.operation == "pause" and run is not None:
                prepared.frozen_versions["control_version"] = str(run.control_version)
            prepared.requires_confirmation = action.operation in CONFIRM_OPERATIONS
            prepared.status = "awaiting_confirmation" if prepared.requires_confirmation else "prepared"
            if prepared.requires_confirmation:
                notice = "Cancels this goal and its children, tasks and sessions." if action.operation == "cancel" else "Pauses work; baseline authorization is cleared during baseline." if action.operation == "pause" else "Authorizes model spend and execution under these limits."
                prepared.summary = notice + "\n" + json.dumps(state, sort_keys=True, default=str)
            if len(prepared.summary) > 12000:
                raise conflict("chat_review_too_large", "Review exceeds the bounded conversation; use the existing control")
            turn.action_draft = prepared.model_dump(mode="json")
            await db.flush()
            return prepared

    async def execute(self, db, turn: ProjectChatTurn, action: ChatActionDraft) -> ChatActionReceipt:
        return await self._execute(db, turn, action, confirmed=False)

    async def _execute(self, db, turn, action, *, confirmed):
        lock = ACTION_LOCKS.setdefault(action.action_id, asyncio.Lock())
        async with lock:
            transaction = db.sync_session.get_transaction()
            owns = transaction is None or transaction.origin == SessionTransactionOrigin.AUTOBEGIN
            goal_lock = no_goal_lock() if action.operation in DECISION_OPERATIONS | {"pause"} else self._orchestration._lock_goal_for_baseline_transition(db, action.goal_id)
            async with goal_lock:
                await db.refresh(turn)
                receipt = await self._execute_locked(db, turn, action, confirmed=confirmed, owns_transaction=owns)
                if owns:
                    await db.commit()
                return receipt

    async def _execute_locked(self, db, turn, action, *, confirmed, owns_transaction=False):
        persisted = await db.get(ProjectChatTurn, turn.id)
        if persisted is None or persisted.actor_id != turn.actor_id or persisted.project_id != action.project_id:
            raise HTTPException(404, "Conversation action not found")
        if persisted.action_draft != action.model_dump(mode="json"):
            raise conflict("chat_idempotency_conflict", "Action differs from its immutable reviewed payload")
        if persisted.action_receipt:
            receipt = ChatActionReceipt.model_validate(persisted.action_receipt)
            if receipt.action_id != action.action_id:
                raise conflict("chat_idempotency_conflict", "Turn already has a different action")
            if receipt.status == "interrupted_unknown" and receipt.source_kind == "authority_decision":
                decision = await db.get(OrchestrationAuthorityDecision, receipt.source_record_id)
                answered = decision is not None and decision.goal_id == action.goal_id and decision.status == "answered" and decision.selected_option == action.payload.selected_option
                if action.operation == "answer_agent_definition_batch":
                    answered = True
                    for item in action.payload.answers:
                        member = await db.get(OrchestrationAuthorityDecision, item.decision_id, populate_existing=True)
                        answered = answered and member is not None and member.goal_id == action.goal_id and member.status == "answered" and member.selected_option == item.selected_option
                if answered:
                    receipt.status = "recorded"
                    receipt.message = "Answer durably recorded; continuation remains managed by its existing reservation."
                    persisted.action_receipt = receipt.model_dump(mode="json")
                    await db.flush()
            return receipt
        self._instruction(persisted, action)
        await self._validate_natural_options(db, persisted, action)
        if persisted.action_draft != action.model_dump(mode="json"):
            raise conflict("chat_idempotency_conflict", "Action differs from its immutable reviewed payload")
        if action.requires_confirmation and not confirmed:
            return ChatActionReceipt(action_id=action.action_id, operation=action.operation, project_id=action.project_id,
                goal_id=action.goal_id, run_id=action.run_id, status="pending", message="Review and confirm this action.")
        goal_lock = no_goal_lock() if action.operation in DECISION_OPERATIONS | {"pause"} else self._orchestration._lock_goal_for_baseline_transition(db, action.goal_id)
        async with goal_lock:
            if action.operation == "pause" and action.run_id is not None:
                from huddleroom.services.orchestration_control import acquire_control_writer
                await acquire_control_writer(db, action.run_id)
            await db.refresh(persisted)
            if persisted.action_receipt:
                return ChatActionReceipt.model_validate(persisted.action_receipt)
            if persisted.action_draft != action.model_dump(mode="json"):
                raise conflict("chat_idempotency_conflict", "Reviewed action changed")
            goal, run, state = await self._snapshot(db, action)
            if (run.id if run else None) != action.run_id or fingerprint(state) != action.frozen_versions.get("state"):
                if confirmed and (run.id if run else None) == action.run_id:
                    refreshed = action.model_copy(deep=True)
                    refreshed.action_id = uuid.uuid4()
                    refreshed.frozen_versions = {"state": fingerprint(state)}
                    if action.operation == "pause" and run is not None:
                        refreshed.frozen_versions["control_version"] = str(run.control_version)
                    refreshed.summary = "Limits changed. Review and confirm these current limits.\n" + json.dumps(state, sort_keys=True, default=str)
                    if len(refreshed.summary) > 12000:
                        raise conflict("chat_review_too_large", "Use the existing control to review these limits")
                    persisted.action_draft = refreshed.model_dump(mode="json")
                    persisted.frozen_context = {key: value for key, value in (persisted.frozen_context or {}).items() if key != "confirmation_request_id"}
                    await db.flush()
                    return ChatActionReceipt(action_id=refreshed.action_id, operation=action.operation,
                        project_id=action.project_id, goal_id=action.goal_id, run_id=action.run_id,
                        status="pending", error_code="chat_stale_review", message=refreshed.summary)
                raise conflict("chat_stale_review", "Submitted run or action state changed; no action was applied")
            if action.operation in {"start_execution", "restart_execution"}:
                if goal.goal_type == "continuous" and not goal.continuous_policy:
                    raise conflict("chat_start_blocked", "Continuous goals require an accepted recurring policy")
                authorization = state.get("authorization") or {}
                if not authorization.get("allowed") and authorization.get("required_decision") != "start":
                    raise conflict("chat_start_blocked", "Start is blocked by the current authorization envelope")
            user = await db.get(User, persisted.actor_id)
            if user is None:
                raise HTTPException(403, "Human actor not found")
            receipt = ChatActionReceipt(action_id=action.action_id, operation=action.operation, project_id=action.project_id,
                goal_id=action.goal_id, run_id=action.run_id, status="completed", message="Action completed.")
            # Specialized continuations may publish an existing durable reservation before returning.
            # Persist a linked unknown receipt first: a crash must reconcile, never dispatch again.
            if action.operation in DECISION_OPERATIONS:
                receipt.status = "interrupted_unknown"
                receipt.source_kind = "authority_decision"
                receipt.source_record_id = action.target_id or (action.payload.answers[0].decision_id if action.payload.answers else None)
                persisted.action_receipt = receipt.model_dump(mode="json")
                await db.flush()
            await self._dispatch(db, persisted, action, user, receipt, owns_transaction=owns_transaction)
            persisted.action_receipt = receipt.model_dump(mode="json")
            persisted.status = "completed"
            await db.flush()
            return receipt

    async def _dispatch(self, db, turn, action, user, receipt, *, owns_transaction=False):
        op, payload = action.operation, action.payload
        async def validation_callback(session):
            await session.refresh(turn)
            if turn.action_draft != action.model_dump(mode="json"):
                raise conflict("chat_idempotency_conflict", "Reviewed action changed")
            _, current_run, current_state = await self._snapshot(session, action)
            if (current_run.id if current_run else None) != action.run_id or fingerprint(current_state) != action.frozen_versions.get("state"):
                raise conflict("chat_stale_review", "Decision target changed; no answer was applied")
        async def reservation_callback(session, linkage):
            turn.frozen_context = {**(turn.frozen_context or {}), "action_reservation": linkage}
            turn.action_receipt = receipt.model_dump(mode="json")
            await session.flush()
        async def completion_callback(session, result):
            receipt.status = "recorded"
            receipt.message = "Answer saved; continuation pending." if getattr(result, "answer_receipt", None) is not None else "Answer recorded." if getattr(result, "continuation_applied", None) is not False else "Answer recorded without continuation."
            turn.action_receipt = receipt.model_dump(mode="json")
            turn.status = "completed"
            await session.flush()
            if owns_transaction and getattr(result, "answer_receipt", None) is None:
                await session.commit()
        project_id, goal_id = action.project_id, action.goal_id
        if op == "start_baseline":
            await self._orchestration.authorize_baseline(db, project_id, goal_id, actor=f"human:{user.id}")
        elif op in {"start_execution", "restart_execution"}:
            await self._orchestration.start_run(db, project_id, goal_id, actor=f"human:{user.id}", commit=False)
        elif op == "pause":
            await self._orchestration.pause_goal(
                db, project_id, goal_id, commit=False, actor_user_id=user.id, run_id=action.run_id,
                expected_control_version=int(action.frozen_versions["control_version"]), request_id=action.action_id,
            )
        elif op == "resume":
            await self._orchestration.resume_goal(db, project_id, goal_id)
        elif op == "cancel":
            await self._orchestration.cancel_goal(db, project_id, goal_id, cancelled_by=f"human:{user.id}", commit=False)
        elif op in {"submit_steering", "promote_steering"}:
            directive = payload.directive
            if op == "promote_steering":
                proposal = await db.get(OrchestrationSteeringProposal, action.target_id)
                if proposal is None or proposal.goal_id != goal_id or proposal.actor_id != user.id:
                    raise HTTPException(404, "Steering proposal not found")
                original = proposal.draft or {}
                expected = {"directive": directive, "target_type": payload.target_type or "goal",
                    "target_id": str(payload.target_id or goal_id), "scope": payload.scope or "run",
                    "lifetime": payload.lifetime or "remaining_current_run", "impact_summary": payload.impact_summary or ""}
                if any(str(original.get(key, "")) != str(value) for key, value in expected.items()):
                    raise conflict("chat_instruction_required", "Promotion must accept the complete existing proposal exactly")
            if not directive:
                raise conflict("chat_instruction_required", "A steering directive is required")
            request = await self._steering.submit(db, project_id, goal_id, user.id, turn.client_request_id,
                SteeringDraft(directive=directive, target_type=payload.target_type or "goal",
                    target_id=str(payload.target_id or goal_id), scope=payload.scope or "run",
                    lifetime=payload.lifetime or "remaining_current_run", impact_summary=payload.impact_summary or "",
                    source_proposal_id=action.target_id if op == "promote_steering" else None,
                    supersedes_request_id=payload.supersedes_request_id))
            receipt.status, receipt.source_kind, receipt.source_record_id = "pending", "steering_request", request.id
            receipt.message = "Direction submitted; work is not complete."
        elif op == "withdraw_steering":
            request = await self._steering.withdraw(db, project_id, goal_id, user.id, action.target_id)
            receipt.source_kind, receipt.source_record_id = "steering_request", request.id
        elif op == "dismiss_proposal":
            proposal = await self._steering.dismiss_proposal(db, project_id, goal_id, user.id, action.target_id)
            receipt.source_kind, receipt.source_record_id = "steering_proposal", proposal.id
        elif op == "answer_agent_definition_batch":
            await decisions.answer_agent_review_batch(db, project_id, goal_id,
                OrchestrationAgentDefinitionReviewBatchAnswerRequest(answers=payload.answers), user, reservation_callback=reservation_callback, validation_callback=validation_callback, completion_callback=completion_callback)
            receipt.status = "recorded"
        elif op in DECISION_OPERATIONS:
            values = payload.model_dump(exclude_none=True)
            allowed = {"selected_option", "reason", "contract_version", "edited_agent", "edited_description", "edited_persona"}
            data = OrchestrationDecisionAnswerRequest(**{key: value for key, value in values.items() if key in allowed})
            decision = await db.get(OrchestrationAuthorityDecision, action.target_id)
            from huddleroom.services.orchestration_recovery_presenter import is_recovery_decision
            hierarchy = decision.decision_key.startswith("team_hierarchy:")
            if hierarchy != (op == "answer_hierarchy"):
                raise conflict("chat_invalid_operation", "Hierarchy proposal requires its specialized action")
            recovery = is_recovery_decision(decision)
            if recovery != (op in {"approve_recovery", "reject_recovery"}):
                raise conflict("chat_invalid_operation", "Decision requires its specialized action")
            if op == "approve_recovery" and data.selected_option not in {"approve_fix", "capability_provided"} or op == "reject_recovery" and data.selected_option != "reject":
                raise conflict("chat_invalid_operation", "Recovery answer does not match the instruction")
            if (decision.runtime_identity is not None) != (op == "answer_runtime_question"):
                raise conflict("chat_invalid_operation", "Runtime question requires its specialized action")
            result = await decisions.answer_decision(db, project_id, goal_id, action.target_id, data, user, reservation_callback=reservation_callback, validation_callback=validation_callback, completion_callback=completion_callback)
            receipt.status = "recorded"
            receipt.message = "Answer saved; continuation pending." if result.answer_receipt is not None else "Answer recorded." if result.continuation_applied is not False else "Answer recorded without continuation."
        else:
            raise conflict("chat_unsupported", "Use the existing control for this operation")

    async def confirm(self, project_id, actor_id, action_id, client_request_id) -> ChatActionReceipt:
        async with self._session_factory() as db:
            rows = (await db.scalars(select(ProjectChatTurn).where(
                ProjectChatTurn.project_id == project_id, ProjectChatTurn.actor_id == actor_id,
                ProjectChatTurn.action_draft.is_not(None)))).all()
            turn = next((row for row in rows if row.action_draft.get("action_id") == str(action_id)), None)
            if turn is None:
                raise HTTPException(404, "Reviewed action not found")
            action = ChatActionDraft.model_validate(turn.action_draft)
            if action.status != "awaiting_confirmation" or not action.summary:
                raise conflict("chat_confirmation_required", "This action has not been shown for confirmation")
            previous = (turn.frozen_context or {}).get("confirmation_request_id")
            if previous is not None and previous != str(client_request_id):
                # Duplicate acceptance may converge only to an already durable receipt.
                if not turn.action_receipt:
                    raise conflict("chat_idempotency_conflict", "Action has another confirmation in progress")
            turn.frozen_context = {**(turn.frozen_context or {}), "confirmation_request_id": str(client_request_id)}
            return await self._execute(db, turn, action, confirmed=True)
