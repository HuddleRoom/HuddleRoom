"""Read-only projection of durable chat and identifiable original history."""
import uuid
from datetime import timezone
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from huddleroom.models.project import Project
from huddleroom.models.orchestration import OrchestrationGoal
from huddleroom.models.orchestration_advisor import ProjectAdvisorTurn
from huddleroom.models.orchestration_chat import ProjectChatTurn
from huddleroom.models.orchestration_conversation import (
    ConversationMessage, ConversationResponse, ConversationInvestigation, ConversationFeedback,
)
from huddleroom.models.orchestration_steering import OrchestrationSteeringProposal
from huddleroom.schemas.orchestration_advisor import ProjectAdvisorCitation
from huddleroom.schemas.orchestration_chat import ChatHistoryResponse, ChatTurnResponse, ChatSourceReference, ChatActionDraft
from huddleroom.services.orchestration_conversation_service import ConversationDomainError, ConversationTurn


def _sort_key(item: ChatTurnResponse):
    created = item.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    sequence = item.goal_turn.sequence if item.goal_turn else 0
    return created, str(item.context_goal_id or ""), sequence, item.id


def _citation_goals(citations):
    goals = []
    for citation in citations:
        if citation.goal_id:
            try:
                goals.append(uuid.UUID(citation.goal_id))
            except ValueError:
                continue
    return goals


async def history(db: AsyncSession, project_id: uuid.UUID, actor_id: uuid.UUID) -> ChatHistoryResponse:
    # Reuse the existing sanitized presenters and gate-aware ledger. They do not
    # dispatch effects; local imports avoid introducing router import cycles.
    from huddleroom.routers.orchestration_advisor import _allowance
    from huddleroom.routers.orchestration_goals import (
        _conversation_turn, _conversation_allowance, _steering_ledger, _steering_proposal,
    )
    if await db.get(Project, project_id) is None:
        raise ConversationDomainError("project_not_found", 404, "Project not found")
    goals = list((await db.scalars(select(OrchestrationGoal).where(
        OrchestrationGoal.project_id == project_id,
    ).order_by(OrchestrationGoal.created_at, OrchestrationGoal.id))).all())
    goal_ids = {goal.id for goal in goals}
    indexed = list((await db.scalars(select(ProjectChatTurn).where(
        ProjectChatTurn.project_id == project_id, ProjectChatTurn.actor_id == actor_id,
    ))).all())
    advisors = list((await db.scalars(select(ProjectAdvisorTurn).where(
        ProjectAdvisorTurn.project_id == project_id, ProjectAdvisorTurn.actor_id == actor_id,
    ))).all())
    advisor_by_id = {row.id: row for row in advisors}
    rows = (await db.execute(select(
        ConversationMessage, ConversationResponse, ConversationInvestigation, ConversationFeedback,
    ).select_from(ConversationMessage)
        .join(ConversationResponse, ConversationResponse.message_id == ConversationMessage.id)
        .join(OrchestrationGoal, OrchestrationGoal.id == ConversationMessage.goal_id)
        .outerjoin(ConversationInvestigation, ConversationInvestigation.response_id == ConversationResponse.id)
        .outerjoin(ConversationFeedback, (ConversationFeedback.response_id == ConversationResponse.id)
                   & (ConversationFeedback.actor_id == actor_id))
        .where(OrchestrationGoal.project_id == project_id))).all()
    proposals = list((await db.scalars(select(OrchestrationSteeringProposal).where(
        OrchestrationSteeringProposal.goal_id.in_(goal_ids),
    ))).all()) if goal_ids else []
    proposal_by_response = {row.response_id: row for row in proposals}
    goal_turns = {}
    for message, response, investigation, feedback in rows:
        proposal = proposal_by_response.get(response.id)
        projected = _conversation_turn(ConversationTurn(
            message=message, response=response, investigation=investigation, feedback=feedback,
            feedback_eligible=(message.actor_id == actor_id and response.status == "completed"
                               and bool((response.answer or "").strip()) and feedback is None),
        ), _steering_proposal(proposal) if proposal else None)
        goal_turns[response.id] = (message, projected)
    indexed_by_id = {str(row.id): row for row in indexed}
    items = []
    consumed = set()
    for row in indexed:
        item = ChatTurnResponse(
            id=str(row.id), client_request_id=row.client_request_id, actor_id=row.actor_id,
            content=row.content, answer=row.answer, status=row.status, created_at=row.created_at,
            context_goal_id=row.context_goal_id, run_id=row.run_id,
            source_references=row.source_references or [], error=row.error,
            actions=[row.action_draft] if row.action_draft else [],
            receipts=[row.action_receipt] if row.action_receipt else [],
            accounting_snapshot=row.accounting_snapshot or {},
        )
        if row.source_kind == "action" and not row.action_draft:
            origin = indexed_by_id.get((row.frozen_context or {}).get("bound_turn_id"))
            if origin and origin.action_draft:
                item.actions = [ChatActionDraft.model_validate(origin.action_draft)]
        if row.source_kind == "project_advisor" and row.source_record_id in advisor_by_id:
            source = advisor_by_id[row.source_record_id]
            if not row.action_draft and not row.action_receipt and not row.error:
                item.answer = source.answer
            item.citations = [ProjectAdvisorCitation.model_validate(citation) for citation in source.citations or []]
            item.off_topic = source.off_topic
            if row.status in {"pending", "running"}:
                item.status = source.status
            consumed.add((row.source_kind, source.id))
        elif row.source_kind == "goal_conversation" and row.source_record_id in goal_turns:
            message, source = goal_turns[row.source_record_id]
            item.goal_turn = source
            if not row.action_draft and not row.action_receipt and not row.error:
                item.answer = source.answer
            if row.status in {"pending", "running"}:
                item.status = source.status
            if not row.error:
                item.error = source.error.model_dump() if source.error else None
            item.context_goal_id = message.goal_id
            consumed.add((row.source_kind, source.response_id))
        if row.source_kind in {"project_advisor", "goal_conversation"}:
            ref = ChatSourceReference(kind=row.source_kind, id=row.source_record_id, goal_id=item.context_goal_id)
            if ref not in item.source_references:
                item.source_references.append(ref)
        item.source_goal_ids = list(dict.fromkeys(
            ([item.context_goal_id] if item.context_goal_id else [])
            + _citation_goals(item.citations)
        ))
        items.append(item)
    for row in advisors:
        if ("project_advisor", row.id) in consumed:
            continue
        advisor_item = ChatTurnResponse(
            id=f"project_advisor:{row.id}", actor_id=row.actor_id, content=row.question,
            answer=row.answer, status=row.status, created_at=row.created_at, historical=True,
            citations=row.citations or [], off_topic=row.off_topic,
            source_references=[ChatSourceReference(kind="project_advisor", id=row.id)],
        )
        advisor_item.source_goal_ids = list(dict.fromkeys(
            _citation_goals(advisor_item.citations)
        ))
        items.append(advisor_item)
    for response_id, (message, projected) in goal_turns.items():
        if ("goal_conversation", response_id) in consumed:
            continue
        items.append(ChatTurnResponse(
            id=f"goal_conversation:{response_id}", client_request_id=message.client_request_id,
            actor_id=message.actor_id, content=message.content, answer=projected.answer,
            status=projected.status, created_at=message.created_at, context_goal_id=message.goal_id,
            run_id=projected.run_id, historical=True, goal_turn=projected,
            source_goal_ids=[message.goal_id],
            source_references=[ChatSourceReference(kind="goal_conversation", id=response_id, goal_id=message.goal_id)],
            error=projected.error.model_dump() if projected.error else None,
        ))
    items.sort(key=_sort_key)
    return ChatHistoryResponse(
        items=items, total=len(items), allowance=await _allowance(db, project_id, actor_id),
        goal_allowances={str(goal.id): await _conversation_allowance(db, goal.id, actor_id) for goal in goals},
        steering={str(goal.id): await _steering_ledger(db, project_id, goal.id, actor_id) for goal in goals},
    )
