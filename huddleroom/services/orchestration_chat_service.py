"""Durable unified conversation facade using existing admission and action gates."""
import asyncio
import uuid
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from huddleroom.database import AsyncSessionLocal
from huddleroom.models.project import Project
from huddleroom.models.orchestration import OrchestrationGoal
from huddleroom.models.orchestration_chat import ProjectChatTurn
from huddleroom.models.orchestration_conversation import ConversationResponse, ConversationMessage
from huddleroom.schemas.orchestration_chat import ChatSubmitRequest, ChatActionDraft
from huddleroom.services.orchestration_chat_history import history as project_history
from huddleroom.services.orchestration_chat_interpretation import OrchestrationChatInterpretationService, goal_chat_source_id
from huddleroom.services.orchestration_chat_actions import OrchestrationChatActionService, fingerprint, ACTION_LOCKS
from huddleroom.services.orchestration_conversation_service import ConversationDomainError

REQUEST_LOCKS = {}


class OrchestrationChatService:
    def __init__(self, session_factory=AsyncSessionLocal, completion_fn=None, lookup_fn=None,
                 interpretation=None, actions=None):
        self._session_factory = session_factory
        self._interpretation = interpretation or OrchestrationChatInterpretationService(session_factory, completion_fn, lookup_fn)
        self._actions = actions or OrchestrationChatActionService(session_factory)

    async def history(self, project_id, actor_id):
        async with self._session_factory() as db:
            return await project_history(db, project_id, actor_id)

    async def _response(self, project_id, actor_id, turn_id):
        result = await self.history(project_id, actor_id)
        return next(item for item in result.items if item.id == str(turn_id))

    async def _state(self, db, goal):
        run = await self._actions._orchestration.get_active_run_for_goal(db, goal.project_id, goal.id)
        state = {key: getattr(goal, key) for key in (
            "status", "objective", "constraints", "budget", "authority_model", "manager_user_id",
            "manager_agent_id", "goal_type", "planning_mode", "continuous_policy", "continuous_state")}
        state["run"] = {key: getattr(run, key) for key in (
            "id", "status", "phase", "plan_state", "baseline_authorized")} if run else None
        return run.id if run else None, fingerprint(state)

    async def submit(self, project_id, actor_id, data: ChatSubmitRequest):
        lock = REQUEST_LOCKS.setdefault((project_id, actor_id, data.client_request_id), asyncio.Lock())
        async with lock:
            try:
                return await self._submit(project_id, actor_id, data)
            except (ConversationDomainError, HTTPException) as error:
                async with self._session_factory() as db:
                    turn = await db.scalar(select(ProjectChatTurn).where(
                        ProjectChatTurn.project_id == project_id, ProjectChatTurn.actor_id == actor_id,
                        ProjectChatTurn.client_request_id == data.client_request_id))
                    matching = turn and turn.content == data.content and (
                        turn.frozen_context.get("submitted_context_goal_id") == (str(data.context_goal_id) if data.context_goal_id else None)
                        and turn.frozen_context.get("submitted_action_id") == (str(data.action_id) if data.action_id else None)
                        and turn.frozen_context.get("submitted_start_scope") == data.start_scope)
                    if matching and turn.status in {"pending", "running"}:
                        detail = error.detail if isinstance(error, HTTPException) else {"code": error.code, "message": error.message}
                        turn.error = detail if isinstance(detail, dict) else {"message": str(detail)}
                        turn.status = "failed"
                        await db.commit()
                raise

    async def _submit(self, project_id, actor_id, data):
        identity = {"submitted_context_goal_id": str(data.context_goal_id) if data.context_goal_id else None,
                    "submitted_action_id": str(data.action_id) if data.action_id else None,
                    "submitted_start_scope": data.start_scope}
        if data.action_id is not None and data.start_scope is not None:
            raise ConversationDomainError("chat_idempotency_conflict", 409, "Confirmation cannot change the reviewed Start scope")
        async with self._session_factory() as db:
            existing = await db.scalar(select(ProjectChatTurn).where(
                ProjectChatTurn.project_id == project_id, ProjectChatTurn.actor_id == actor_id,
                ProjectChatTurn.client_request_id == data.client_request_id))
            if existing:
                if existing.content != data.content or any(existing.frozen_context.get(k) != v for k, v in identity.items()):
                    raise ConversationDomainError("chat_idempotency_conflict", 409, "Request identity has different content or context")
                turn_id = existing.id
                if existing.source_kind == "action" and existing.status == "pending":
                    candidates = list((await db.scalars(select(ProjectChatTurn).where(
                        ProjectChatTurn.project_id == project_id, ProjectChatTurn.actor_id == actor_id,
                        ProjectChatTurn.action_draft.is_not(None)))).all())
                    origin = existing.frozen_context.get("bound_turn_id")
                    bound = next((row for row in candidates if str(row.id) == origin), None) if origin else next(
                        (row for row in candidates if row.action_draft.get("action_id") == identity["submitted_action_id"]), None)
                    if bound and bound.action_receipt:
                        if bound.action_receipt.get("status") == "interrupted_unknown":
                            await self._actions.execute(db, bound, ChatActionDraft.model_validate(bound.action_draft))
                        existing.action_receipt = bound.action_receipt
                        existing.answer = bound.action_receipt.get("message")
                        existing.status = "interrupted_unknown" if bound.action_receipt.get("status") == "interrupted_unknown" else "completed"
                    elif bound and bound.action_draft.get("action_id") != identity["submitted_action_id"]:
                        existing.answer = bound.action_draft.get("summary")
                        existing.status = "completed"
                    else:
                        existing.status = "interrupted_unknown"
                        existing.error = {"code": "chat_action_unknown", "message": "Action outcome requires review; it was not redispatched."}
                    await db.commit()
                    return await self._response(project_id, actor_id, turn_id)
                if existing.action_draft:
                    draft = ChatActionDraft.model_validate(existing.action_draft)
                    if (existing.action_receipt and existing.action_receipt.get("status") != "interrupted_unknown") or (draft.requires_confirmation and not existing.action_receipt) or draft.status == "failed":
                        return await self._response(project_id, actor_id, turn_id)
                    receipt = await self._actions.execute(db, existing, draft)
                    existing.status = "interrupted_unknown" if receipt.status == "interrupted_unknown" else "completed"
                    existing.answer = receipt.message
                    await db.commit()
                    return await self._response(project_id, actor_id, turn_id)
                recover = existing.status in {"pending", "running"}
                if not recover:
                    return await self._response(project_id, actor_id, turn_id)
            else:
                if await db.get(Project, project_id) is None:
                    raise ConversationDomainError("project_not_found", 404, "Project not found")
                goals = list((await db.scalars(select(OrchestrationGoal).where(OrchestrationGoal.project_id == project_id))).all())
                selected = next((g for g in goals if g.id == data.context_goal_id), None)
                if data.context_goal_id and selected is None and not data.action_id:
                    raise ConversationDomainError("goal_not_found", 404, "Goal not found in project")
                named = [g for g in goals if g.objective and g.objective.casefold() in data.content.casefold()]
                ambiguous = len(named) > 1
                if len(named) == 1:
                    selected = named[0]
                snapshots = {}
                for goal in goals:
                    run_id, state = await self._state(db, goal)
                    snapshots[str(goal.id)] = {"run_id": str(run_id) if run_id else None, "state": state}
                bound = None
                if data.action_id:
                    candidates = list((await db.scalars(select(ProjectChatTurn).where(
                        ProjectChatTurn.project_id == project_id, ProjectChatTurn.actor_id == actor_id,
                        ProjectChatTurn.action_draft.is_not(None)))).all())
                    bound = next((row for row in candidates if row.action_draft.get("action_id") == str(data.action_id)), None)
                    if bound is None:
                        raise ConversationDomainError("chat_action_not_found", 404, "Reviewed action not found")
                    selected = next((g for g in goals if str(g.id) == bound.action_draft.get("goal_id")), None)
                    if selected is None:
                        raise ConversationDomainError("goal_not_found", 404, "Reviewed goal no longer exists")
                    if ambiguous or (named and any(g.id != selected.id for g in named)):
                        raise ConversationDomainError("chat_target_conflict", 409, "Named goal conflicts with the reviewed action target")
                turn = ProjectChatTurn(project_id=project_id, actor_id=actor_id,
                    client_request_id=data.client_request_id, content=data.content,
                    context_goal_id=selected.id if selected else None,
                    run_id=uuid.UUID(snapshots[str(selected.id)]["run_id"]) if selected and snapshots[str(selected.id)]["run_id"] else None,
                    frozen_context={**identity, "targets": snapshots, "bound_turn_id": str(bound.id) if bound else None},
                    source_kind="goal_conversation" if selected else "project_advisor")
                if selected:
                    turn.source_record_id = goal_chat_source_id(selected.id, actor_id, data.client_request_id)
                db.add(turn)
                try:
                    await db.commit()
                except IntegrityError:
                    await db.rollback()
                    return await self._submit(project_id, actor_id, data)
                turn_id = turn.id
                if ambiguous and bound is None:
                    turn.source_kind = "clarification"
                    turn.status = "completed"
                    turn.answer = "Please identify one intended goal."
                    await db.commit()
                    return await self._response(project_id, actor_id, turn_id)
                if bound is not None and data.content.casefold() in {"confirm", "yes", "dismiss", "no"}:
                    turn.source_kind = "action"
                    if data.content.casefold() in {"dismiss", "no"}:
                        async with ACTION_LOCKS.setdefault(data.action_id, asyncio.Lock()):
                            await db.refresh(bound)
                            if bound.action_receipt:
                                raise ConversationDomainError("chat_action_completed", 409, "Action already has an outcome")
                            draft = ChatActionDraft.model_validate(bound.action_draft)
                            if draft.status != "awaiting_confirmation":
                                raise ConversationDomainError("chat_confirmation_required", 409, "Action is not awaiting confirmation")
                            draft.status = "failed"
                            bound.action_draft = draft.model_dump(mode="json")
                            turn.answer = "Reviewed action dismissed."
                            turn.status = "completed"
                            await db.commit()
                    else:
                        await db.commit()
                        receipt = await self._actions.confirm(project_id, actor_id, data.action_id, data.client_request_id)
                        await db.refresh(turn)
                        turn.action_receipt = receipt.model_dump(mode="json")
                        turn.answer = receipt.message
                        turn.status = "completed" if receipt.status != "interrupted_unknown" else "interrupted_unknown"
                        await db.commit()
                    return await self._response(project_id, actor_id, turn_id)
        try:
            result = await self._interpretation.interpret(turn_id)
            async with self._session_factory() as db:
                turn = await db.get(ProjectChatTurn, turn_id)
                turn.interpretation = result.model_dump(mode="json")
                turn.answer = result.answer or result.clarification
                if turn.source_kind == "goal_conversation":
                    source = await db.get(ConversationResponse, turn.source_record_id)
                    if source is not None:
                        from huddleroom.services.orchestration_chat_interpretation import sync_goal_chat_provider
                        message = await db.get(ConversationMessage, source.message_id)
                        await sync_goal_chat_provider(db, turn.id, source, message)
                        if source.error:
                            turn.error = source.error
                if turn.provider_status in {"interrupted_unknown", "failed", "pending", "running"}:
                    turn.status = turn.provider_status
                elif result.intent == "instruction" and result.operation:
                    goals = list((await db.scalars(select(OrchestrationGoal).where(OrchestrationGoal.project_id == project_id))).all())
                    if result.named_goal:
                        matches = [g for g in goals if g.objective == result.named_goal and result.named_goal.casefold() in turn.content.casefold()]
                        if len(matches) != 1:
                            raise ConversationDomainError("chat_invalid_target", 409, "Please identify one intended goal")
                        result.goal_id = matches[0].id
                        turn.interpretation = result.model_dump(mode="json")
                    if turn.frozen_context.get("submitted_action_id") and result.goal_id != turn.context_goal_id:
                        raise ConversationDomainError("chat_target_conflict", 409, "Instruction conflicts with the reviewed action target")
                    goal = next((g for g in goals if g.id == result.goal_id), None)
                    captured = turn.frozen_context["targets"].get(str(result.goal_id))
                    if goal is None or captured is None:
                        raise ConversationDomainError("chat_invalid_target", 409, "Target was not present at submission")
                    run_id, state = await self._state(db, goal)
                    if captured != {"run_id": str(run_id) if run_id else None, "state": state}:
                        raise ConversationDomainError("chat_stale_submission", 409, "Goal state changed; submit a fresh instruction")
                    draft = ChatActionDraft(action_id=uuid.uuid5(turn.id, "action"), operation=result.operation,
                        project_id=project_id, goal_id=goal.id, run_id=run_id,
                        target_id=result.payload.target_id, payload=result.payload,
                        summary=result.answer or "Requested action")
                    if result.operation in {"start_execution", "restart_execution"}:
                        draft.payload.start_scope = data.start_scope or "execution"
                    elif data.start_scope is not None:
                        raise ConversationDomainError("chat_invalid_operation", 409, "Scope selection requires a Start action")
                    async def validate_submission(session):
                        await session.refresh(goal)
                        current_run_id, current_state = await self._state(session, goal)
                        if captured != {"run_id": str(current_run_id) if current_run_id else None, "state": current_state}:
                            raise ConversationDomainError("chat_stale_submission", 409, "Goal state changed; submit a fresh instruction")
                    draft = await self._actions.prepare(db, turn, draft, submission_validation=validate_submission)
                    await db.commit()
                    if not draft.requires_confirmation:
                        receipt = await self._actions.execute(db, turn, draft)
                        turn.answer = receipt.message or turn.answer
                        turn.status = "interrupted_unknown" if receipt.status == "interrupted_unknown" else "completed"
                    else:
                        turn.status = "completed"
                        turn.answer = draft.summary
                else:
                    turn.status = "completed"
                await db.commit()
        except (ConversationDomainError, HTTPException) as error:
            async with self._session_factory() as db:
                turn = await db.get(ProjectChatTurn, turn_id)
                turn.status = "failed"
                detail = error.detail if isinstance(error, HTTPException) else {"code": error.code, "message": error.message}
                turn.error = detail if isinstance(detail, dict) else {"message": str(detail)}
                turn.answer = turn.error.get("message", "Request blocked")
                await db.commit()
        return await self._response(project_id, actor_id, turn_id)
