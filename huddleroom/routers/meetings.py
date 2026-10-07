from __future__ import annotations

import base64
import binascii
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select, func, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.dependencies import get_current_user, ensure_project_exists
from huddleroom.database import get_db
from huddleroom.models.meeting import (
    Meeting, MeetingAgendaItem, MeetingDecision, MeetingTurn,
    MeetingActionItem, MeetingParticipantSignal, MeetingRequest,
)
from huddleroom.models.user import User
from huddleroom.schemas.common import CursorPage
from huddleroom.schemas.meeting import (
    MeetingCreate, MeetingResponse, AgendaItemResponse,
    TurnResponse, HumanTurnCreate, HumanOverrideCreate, VetoDecisionCreate,
    DecisionResponse, MeetingCopyRequest, ActionItemResponse,
    AgendaItemCreate, AgendaItemUpdate, ActionItemUpdate,
    GrantTurnRequest, SignalCreate, SignalResponse,
    AdvanceAgendaItemRequest, AgentMeetingRequestCreate, AgentMeetingRequestResponse,
    FinalReviewResponse, FinalReviewSubmit, VetoCreate,
)
from huddleroom.services.meeting_service import MeetingService, MeetingTransitionError
from huddleroom.services.project_service import ProjectService
from huddleroom.services.event_bus import emit_event

router = APIRouter(tags=["meetings"])
_svc = MeetingService()


def _encode_cursor(created_at: datetime, meeting_id: uuid.UUID) -> str:
    if created_at.tzinfo is not None:
        created_at = created_at.astimezone(timezone.utc).replace(tzinfo=None)
    raw = f"{created_at.isoformat()}|{meeting_id}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


def _decode_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
    try:
        raw = base64.urlsafe_b64decode(cursor.encode()).decode()
        ts_str, id_str = raw.split("|", 1)
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt, uuid.UUID(id_str)
    except (ValueError, UnicodeDecodeError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="Invalid cursor") from exc


async def _get_meeting_or_404(meeting_id: uuid.UUID, db: AsyncSession) -> Meeting:
    meeting = await db.get(Meeting, meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting


async def _load_agenda_items(db: AsyncSession, meeting_id: uuid.UUID) -> list[MeetingAgendaItem]:
    result = await db.execute(
        select(MeetingAgendaItem)
        .where(MeetingAgendaItem.meeting_id == meeting_id)
        .order_by(MeetingAgendaItem.order)
    )
    return list(result.scalars().all())


async def _activate_runnable_meeting(db: AsyncSession, meeting: Meeting) -> Meeting:
    if meeting.status == "scheduled":
        claimed = await _svc.claim_scheduled_meeting(db, meeting.id)
        if claimed is not None:
            meeting = claimed
    elif meeting.status == "preparing":
        project_svc = ProjectService()
        await project_svc.lock_workspace_boundary(db, meeting.project_id)
        await project_svc.require_runnable_project(db, meeting.project_id)
    if meeting.status == "preparing":
        await _svc.transition_to_active(db=db, meeting=meeting)
        await db.flush()
    return meeting


def _meeting_to_response(meeting: Meeting, agenda_items: list[MeetingAgendaItem]) -> MeetingResponse:
    items = [AgendaItemResponse.model_validate(i) for i in agenda_items]
    data = MeetingResponse.model_validate(meeting)
    data.agenda_items = items
    return data


@router.post(
    "/projects/{project_id}/meetings",
    response_model=MeetingResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_meeting(
    project_id: uuid.UUID,
    body: MeetingCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MeetingResponse:
    meeting = await _svc.create_meeting(
        db=db,
        project_id=project_id,
        title=body.title,
        meeting_type=body.meeting_type,
        participant_agent_ids=[str(a) for a in body.participant_agent_ids],
        participant_user_ids=[str(u) for u in body.participant_user_ids],
        agenda_items=[item.model_dump() for item in body.agenda_items],
        turn_strategy=body.turn_strategy,
        deadlock_strategy=body.deadlock_strategy,
        max_duration_minutes=body.max_duration_minutes,
        veto_window_hours=body.veto_window_hours,
        auto_start=body.auto_start,
        scheduled_at=body.scheduled_at,
        created_by_user_id=current_user.id,
        source_task_id=body.source_task_id,
        source_graph_run_id=body.source_graph_run_id,
        organizer_agent_id=body.organizer_agent_id,
        organizer_user_id=body.organizer_user_id,
        planner_agent_id=body.planner_agent_id,
        signal_check_enabled=body.signal_check_enabled,
    )
    await emit_event(
        db,
        project_id,
        "meeting.scheduled",
        {"meeting_id": str(meeting.id), "auto_start": meeting.auto_start},
    )
    items = await _load_agenda_items(db, meeting.id)
    return _meeting_to_response(meeting, items)


@router.get("/projects/{project_id}/meetings/count")
async def count_meetings(
    project_id: uuid.UUID,
    status: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await ensure_project_exists(db, project_id)
    conditions = [Meeting.project_id == project_id]
    if status is not None:
        conditions.append(Meeting.status == status)
    result = await db.execute(select(func.count(Meeting.id)).where(*conditions))  # pylint: disable=not-callable
    return {"count": result.scalar_one()}


@router.get("/projects/{project_id}/meetings", response_model=CursorPage[MeetingResponse])
async def list_meetings(
    project_id: uuid.UUID,
    status_filter: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None, max_length=512),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> CursorPage[MeetingResponse]:
    await ensure_project_exists(db, project_id)
    q = select(Meeting).where(Meeting.project_id == project_id)
    VALID_STATUSES = {"scheduled", "preparing", "active", "concluding", "concluded", "cancelled"}
    if status_filter:
        if status_filter not in VALID_STATUSES:
            raise HTTPException(status_code=400, detail=f"Invalid status_filter: {status_filter!r}")
        q = q.where(Meeting.status == status_filter)
    if cursor:
        cursor_created_at, cursor_id = _decode_cursor(cursor)
        q = q.where(
            or_(
                Meeting.created_at < cursor_created_at,
                and_(
                    Meeting.created_at == cursor_created_at,
                    Meeting.id < cursor_id,
                ),
            )
        )
    q = q.order_by(Meeting.created_at.desc(), Meeting.id.desc()).limit(limit + 1)
    meetings = list((await db.execute(q)).scalars().all())

    has_more = len(meetings) > limit
    if has_more:
        meetings = meetings[:limit]

    if not meetings:
        return CursorPage(items=[], next_cursor=None)

    meeting_ids = [meeting.id for meeting in meetings]
    agenda_items = list(
        (
            await db.execute(
                select(MeetingAgendaItem)
                .where(MeetingAgendaItem.meeting_id.in_(meeting_ids))
                .order_by(MeetingAgendaItem.meeting_id, MeetingAgendaItem.order)
            )
        ).scalars().all()
    )

    items_by_meeting: dict[uuid.UUID, list[MeetingAgendaItem]] = {}
    for item in agenda_items:
        items_by_meeting.setdefault(item.meeting_id, []).append(item)

    last = meetings[-1]
    next_cursor = _encode_cursor(last.created_at, last.id) if has_more else None

    return CursorPage(
        items=[_meeting_to_response(meeting, items_by_meeting.get(meeting.id, [])) for meeting in meetings],
        next_cursor=next_cursor,
    )


@router.get("/meetings/{meeting_id}", response_model=MeetingResponse)
async def get_meeting(
    meeting_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MeetingResponse:
    meeting = await _get_meeting_or_404(meeting_id, db)
    items = await _load_agenda_items(db, meeting.id)
    return _meeting_to_response(meeting, items)


@router.post("/meetings/{meeting_id}/copy", response_model=MeetingResponse, status_code=status.HTTP_201_CREATED)
async def copy_meeting(
    meeting_id: uuid.UUID,
    body: MeetingCopyRequest | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MeetingResponse:
    source = await _get_meeting_or_404(meeting_id, db)
    copy_body = body or MeetingCopyRequest()
    meeting = await _svc.copy_meeting(
        db=db,
        source_meeting=source,
        created_by_user_id=current_user.id,
        title=copy_body.title,
        scheduled_at=copy_body.scheduled_at,
        auto_start=copy_body.auto_start,
    )
    await emit_event(
        db,
        source.project_id,
        "meeting.scheduled",
        {"meeting_id": str(meeting.id), "auto_start": meeting.auto_start},
    )
    items = await _load_agenda_items(db, meeting.id)
    return _meeting_to_response(meeting, items)


@router.delete("/meetings/{meeting_id}", response_model=MeetingResponse)
async def cancel_meeting(
    meeting_id: uuid.UUID,
    reason: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MeetingResponse:
    meeting = await _get_meeting_or_404(meeting_id, db)
    try:
        await _svc.cancel_meeting(db=db, meeting=meeting, reason=reason)
    except MeetingTransitionError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    items = await _load_agenda_items(db, meeting.id)
    return _meeting_to_response(meeting, items)


@router.post("/meetings/{meeting_id}/human-turn", response_model=TurnResponse, status_code=201)
async def submit_human_turn(
    meeting_id: uuid.UUID,
    body: HumanTurnCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TurnResponse:
    meeting = await _get_meeting_or_404(meeting_id, db)
    meeting = await _activate_runnable_meeting(db, meeting)
    if meeting.status != "active":
        raise HTTPException(status_code=400, detail="Meeting is not active")

    count_result = await db.execute(
        select(func.count(MeetingTurn.id)).where(MeetingTurn.meeting_id == meeting_id)  # pylint: disable=not-callable
    )
    turn_number = (count_result.scalar_one() or 0) + 1

    current_item = await _svc.get_current_agenda_item(db=db, meeting_id=meeting_id)
    await ProjectService().require_runnable_project(db, meeting.project_id)

    turn = MeetingTurn(
        meeting_id=meeting_id,
        agenda_item_id=current_item.id if current_item else None,
        turn_number=turn_number,
        round_number=current_item.current_round if current_item else 1,
        speaker_user_id=current_user.id,
        content=body.content,
        references=body.references,
        is_human_turn=True,
    )
    db.add(turn)
    await _svc.log_event(db, meeting_id, "human_turn", {"turn_number": turn_number})
    await db.flush()
    await emit_event(
        db=db,
        project_id=meeting.project_id,
        event_type="meeting.human_turn",
        payload={
            "meeting_id": str(meeting.id),
            "turn_id": str(turn.id),
            "turn_number": turn.turn_number,
        },
    )
    return TurnResponse.model_validate(turn)


@router.post("/meetings/{meeting_id}/override", response_model=DecisionResponse)
async def human_override(
    meeting_id: uuid.UUID,
    body: HumanOverrideCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DecisionResponse:
    meeting = await _get_meeting_or_404(meeting_id, db)
    if meeting.status != "active":
        raise HTTPException(status_code=400, detail="Meeting is not active")

    item = await db.get(MeetingAgendaItem, body.agenda_item_id)
    if not item or item.meeting_id != meeting_id:
        raise HTTPException(status_code=404, detail="Agenda item not found")

    decision = MeetingDecision(
        meeting_id=meeting_id,
        agenda_item_id=body.agenda_item_id,
        title=item.title,
        question=item.question,
        chosen_option=body.decision,
        rationale=body.reason,
        decided_by="human_override",
    )
    db.add(decision)
    await db.flush()
    item.status = "resolved"
    item.resolved_at = datetime.now(timezone.utc)

    await _svc.log_event(
        db, meeting_id, "human_override",
        {"agenda_item_id": str(body.agenda_item_id), "decision": body.decision},
        actor_user_id=current_user.id,
    )
    await _svc.advance_agenda(db=db, meeting=meeting, completed_item=item, resolution="resolved")
    return DecisionResponse.model_validate(decision)


@router.post("/meetings/{meeting_id}/end", response_model=MeetingResponse)
async def end_meeting(
    meeting_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MeetingResponse:
    from huddleroom.workers.meeting_tasks import dispatch_finalize_meeting

    meeting = await _get_meeting_or_404(meeting_id, db)
    if meeting.status in {"cancelled", "concluded"}:
        raise HTTPException(status_code=400, detail="Meeting is already terminal")
    if meeting.status not in {"preparing", "active", "concluding"}:
        raise HTTPException(status_code=400, detail="Only preparing, active, or concluding meetings can be ended")
    try:
        if meeting.status == "preparing":
            await _svc.transition_to_active(db=db, meeting=meeting)
        if meeting.status == "active":
            await _svc.transition_to_concluding(db=db, meeting=meeting)
        meeting.is_partial = True
    except MeetingTransitionError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    await _svc.log_event(db, meeting_id, "human_end", {}, actor_user_id=current_user.id)
    items = await _load_agenda_items(db, meeting.id)
    response = _meeting_to_response(meeting, items)
    await db.commit()
    dispatch_finalize_meeting(str(meeting_id), meeting.project_id)
    return response


@router.post("/meetings/{meeting_id}/resume", response_model=MeetingResponse)
async def resume_meeting(
    meeting_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MeetingResponse:
    from huddleroom.workers.meeting_tasks import dispatch_resume_meeting_turn

    meeting = await _get_meeting_or_404(meeting_id, db)

    if meeting.status != "active" or not meeting.resume_state.get("failed"):
        raise HTTPException(
            status_code=409,
            detail="Meeting must be active and have a failed turn to resume"
        )

    items = await _load_agenda_items(db, meeting.id)
    response = _meeting_to_response(meeting, items)
    await db.commit()
    dispatch_resume_meeting_turn(str(meeting_id), meeting.project_id)
    return response


@router.get(
    "/meetings/{meeting_id}/final-review",
    response_model=FinalReviewResponse | None,
)
async def get_final_review(
    meeting_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FinalReviewResponse | None:
    from huddleroom.services.meeting_outcome import MeetingOutcomeService

    meeting = await _get_meeting_or_404(meeting_id, db)
    pending = await MeetingOutcomeService().get_pending_final_review(db=db, meeting=meeting)
    return FinalReviewResponse.model_validate(pending.payload) if pending else None


@router.post("/meetings/{meeting_id}/final-review", status_code=status.HTTP_204_NO_CONTENT)
async def submit_final_review(
    meeting_id: uuid.UUID,
    body: FinalReviewSubmit,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Response:
    from huddleroom.services.meeting_outcome import MeetingOutcomeService
    from huddleroom.workers.meeting_tasks import dispatch_finalize_meeting

    meeting = await _get_meeting_or_404(meeting_id, db)
    if meeting.organizer_user_id != current_user.id and str(current_user.id) not in (
        meeting.participant_user_ids or []
    ):
        raise HTTPException(status_code=403, detail="Only meeting participants can complete final review")

    try:
        completed = await MeetingOutcomeService().complete_final_review(
            db=db,
            meeting=meeting,
            decisions_made=body.decisions_made,
            decisions_clear=body.decisions_clear,
            action_items_needed=body.action_items_needed,
            action_items=body.action_items,
            actor_user_id=current_user.id,
        )
    except MeetingTransitionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    await db.commit()
    if completed:
        dispatch_finalize_meeting(str(meeting_id), meeting.project_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/meetings/{meeting_id}/veto-decision", response_model=DecisionResponse)
async def veto_decision(
    meeting_id: uuid.UUID,
    body: VetoDecisionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DecisionResponse:
    meeting = await _get_meeting_or_404(meeting_id, db)
    decision = await db.get(MeetingDecision, body.decision_id)
    if not decision or decision.meeting_id != meeting_id:
        raise HTTPException(status_code=404, detail="Decision not found")
    if decision.is_vetoed:
        raise HTTPException(status_code=400, detail="Decision already vetoed")

    decision_created_at = decision.created_at
    if decision_created_at.tzinfo is None:
        decision_created_at = decision_created_at.replace(tzinfo=timezone.utc)
    veto_deadline = decision_created_at + timedelta(hours=meeting.veto_window_hours)
    if datetime.now(timezone.utc) > veto_deadline:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Veto window of {meeting.veto_window_hours}h has expired. "
                f"Decision was made at {decision.created_at.isoformat()}."
            ),
        )

    decision.is_vetoed = True
    decision.veto_reason = body.reason
    decision.vetoed_by_user_id = current_user.id
    decision.vetoed_at = datetime.now(timezone.utc)

    await _svc.log_event(
        db, meeting_id, "veto",
        {"decision_id": str(body.decision_id), "reason": body.reason},
        actor_user_id=current_user.id,
    )
    return DecisionResponse.model_validate(decision)


@router.post("/meetings/{meeting_id}/decisions/{decision_id}/veto", response_model=DecisionResponse)
async def veto_decision_canonical(
    meeting_id: uuid.UUID,
    decision_id: uuid.UUID,
    body: VetoCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DecisionResponse:
    meeting = await _get_meeting_or_404(meeting_id, db)
    decision = await db.get(MeetingDecision, decision_id)
    if not decision or decision.meeting_id != meeting_id:
        raise HTTPException(status_code=404, detail="Decision not found")
    if decision.is_vetoed:
        raise HTTPException(status_code=400, detail="Decision already vetoed")

    decision_created_at = decision.created_at
    if decision_created_at.tzinfo is None:
        decision_created_at = decision_created_at.replace(tzinfo=timezone.utc)
    veto_deadline = decision_created_at + timedelta(hours=meeting.veto_window_hours)
    if datetime.now(timezone.utc) > veto_deadline:
        raise HTTPException(
            status_code=400,
            detail=f"Veto window of {meeting.veto_window_hours}h has expired.",
        )

    decision.is_vetoed = True
    decision.veto_reason = body.reason
    decision.vetoed_by_user_id = current_user.id
    decision.vetoed_at = datetime.now(timezone.utc)

    await _svc.log_event(
        db, meeting_id, "veto",
        {"decision_id": str(decision_id), "reason": body.reason},
        actor_user_id=current_user.id,
    )
    return DecisionResponse.model_validate(decision)


@router.get("/meetings/{meeting_id}/turns", response_model=list[TurnResponse])
async def list_turns(
    meeting_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[TurnResponse]:
    await _get_meeting_or_404(meeting_id, db)
    result = await db.execute(
        select(MeetingTurn)
        .where(MeetingTurn.meeting_id == meeting_id)
        .order_by(MeetingTurn.turn_number)
    )
    return [TurnResponse.model_validate(t) for t in result.scalars().all()]


@router.get("/meetings/{meeting_id}/decisions", response_model=list[DecisionResponse])
async def list_decisions(
    meeting_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[DecisionResponse]:
    await _get_meeting_or_404(meeting_id, db)
    result = await db.execute(
        select(MeetingDecision).where(MeetingDecision.meeting_id == meeting_id).order_by(MeetingDecision.created_at)
    )
    return [DecisionResponse.model_validate(d) for d in result.scalars().all()]


@router.get("/meetings/{meeting_id}/agenda", response_model=list[AgendaItemResponse])
async def list_agenda_items(
    meeting_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[AgendaItemResponse]:
    await _get_meeting_or_404(meeting_id, db)
    items = await _load_agenda_items(db, meeting_id)
    return [AgendaItemResponse.model_validate(i) for i in items]


@router.post("/meetings/{meeting_id}/agenda", response_model=AgendaItemResponse, status_code=201)
async def add_agenda_item(
    meeting_id: uuid.UUID,
    body: AgendaItemCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AgendaItemResponse:
    meeting = await _get_meeting_or_404(meeting_id, db)
    if meeting.status not in {"scheduled", "preparing"}:
        raise HTTPException(status_code=400, detail="Agenda items can only be added before meeting is active")
    item = MeetingAgendaItem(
        meeting_id=meeting_id,
        order=body.order,
        title=body.title,
        description=body.description,
        question=body.question,
        options=body.options,
        artifact_url=body.artifact_url,
        turn_order=[str(t) for t in body.turn_order] if body.turn_order else None,
        max_rounds=body.max_rounds,
        requires_approval=body.requires_approval,
        creates_graph=body.creates_graph,
    )
    db.add(item)
    await db.flush()
    return AgendaItemResponse.model_validate(item)


@router.patch("/meetings/{meeting_id}/agenda/{item_id}", response_model=AgendaItemResponse)
async def update_agenda_item(
    meeting_id: uuid.UUID,
    item_id: uuid.UUID,
    body: AgendaItemUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AgendaItemResponse:
    meeting = await _get_meeting_or_404(meeting_id, db)
    if meeting.status not in {"scheduled", "preparing"}:
        raise HTTPException(status_code=400, detail="Agenda items can only be updated before meeting is active")
    item = await db.get(MeetingAgendaItem, item_id)
    if not item or item.meeting_id != meeting_id:
        raise HTTPException(status_code=404, detail="Agenda item not found")
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(item, field, value)
    await db.flush()
    return AgendaItemResponse.model_validate(item)


@router.post("/meetings/{meeting_id}/agenda/{item_id}/advance", response_model=AgendaItemResponse)
async def advance_agenda_item(
    meeting_id: uuid.UUID,
    item_id: uuid.UUID,
    body: AdvanceAgendaItemRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AgendaItemResponse:
    meeting = await _get_meeting_or_404(meeting_id, db)
    if meeting.status != "active":
        raise HTTPException(status_code=400, detail="Meeting must be active to advance agenda items")
    if meeting.turn_strategy == "organizer_controlled" and meeting.organizer_user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only the organizer can advance agenda items directly")
    item = await db.get(MeetingAgendaItem, item_id)
    if not item or item.meeting_id != meeting_id:
        raise HTTPException(status_code=404, detail="Agenda item not found")
    await _svc.advance_agenda_item_by_organizer(
        db=db, meeting=meeting, item=item, resolution=body.resolution
    )
    await db.flush()
    return AgendaItemResponse.model_validate(item)


@router.post("/meetings/{meeting_id}/grant-turn", response_model=MeetingResponse)
async def grant_turn(
    meeting_id: uuid.UUID,
    body: GrantTurnRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MeetingResponse:
    from huddleroom.workers.meeting_tasks import dispatch_run_meeting_turn
    meeting = await _get_meeting_or_404(meeting_id, db)
    if meeting.organizer_user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only the human organizer can grant turns")
    try:
        await _svc.grant_turn(db=db, meeting=meeting, agent_id=body.participant_agent_id)
    except MeetingTransitionError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    await db.commit()
    dispatch_run_meeting_turn(str(meeting_id), meeting.project_id)
    meeting = await db.get(Meeting, meeting_id)
    items = await _load_agenda_items(db, meeting_id)
    return _meeting_to_response(meeting, items)


@router.post("/meetings/{meeting_id}/signal", response_model=SignalResponse, status_code=201)
async def submit_signal(
    meeting_id: uuid.UUID,
    body: SignalCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SignalResponse:
    meeting = await _get_meeting_or_404(meeting_id, db)
    if str(body.agent_id) not in meeting.participant_agent_ids:
        raise HTTPException(status_code=403, detail="Agent is not a participant in this meeting")
    signal = await _svc.add_signal(
        db=db,
        meeting_id=meeting_id,
        agent_id=body.agent_id,
        signal_type=body.signal_type,
        message=body.message,
    )
    await emit_event(
        db=db,
        project_id=meeting.project_id,
        event_type="meeting.signal",
        payload={
            "meeting_id": str(meeting_id),
            "agent_id": str(body.agent_id),
            "signal_type": body.signal_type,
            "message": body.message,
        },
    )
    return SignalResponse.model_validate(signal)


@router.get("/meetings/{meeting_id}/signals", response_model=list[SignalResponse])
async def list_signals(
    meeting_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[SignalResponse]:
    await _get_meeting_or_404(meeting_id, db)
    signals = await _svc.get_pending_signals(db=db, meeting_id=meeting_id)
    return [SignalResponse.model_validate(s) for s in signals]


@router.get("/meetings/{meeting_id}/action-items", response_model=list[ActionItemResponse])
async def list_action_items(
    meeting_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[ActionItemResponse]:
    await _get_meeting_or_404(meeting_id, db)
    result = await db.execute(
        select(MeetingActionItem)
        .where(MeetingActionItem.meeting_id == meeting_id)
        .order_by(MeetingActionItem.created_at)
    )
    return [ActionItemResponse.model_validate(a) for a in result.scalars().all()]


@router.patch("/meetings/{meeting_id}/action-items/{aid}", response_model=ActionItemResponse)
async def update_action_item(
    meeting_id: uuid.UUID,
    aid: uuid.UUID,
    body: ActionItemUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ActionItemResponse:
    await _get_meeting_or_404(meeting_id, db)
    item = await db.get(MeetingActionItem, aid)
    if not item or item.meeting_id != meeting_id:
        raise HTTPException(status_code=404, detail="Action item not found")
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(item, field, value)
    await db.flush()
    return ActionItemResponse.model_validate(item)


@router.post("/agent/request-meeting", response_model=AgentMeetingRequestResponse, status_code=201)
async def agent_request_meeting(
    body: AgentMeetingRequestCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AgentMeetingRequestResponse:
    request = MeetingRequest(
        project_id=body.project_id,
        requesting_agent_id=body.requesting_agent_id,
        title=body.title,
        reason=body.reason,
        meeting_type=body.meeting_type,
        suggested_participant_agent_ids=[str(a) for a in body.suggested_participant_agent_ids],
    )
    db.add(request)
    await db.flush()
    await emit_event(
        db=db,
        project_id=body.project_id,
        event_type="meeting.request_pending",
        payload={
            "request_id": str(request.id),
            "agent_id": str(body.requesting_agent_id),
            "title": body.title,
        },
    )
    return AgentMeetingRequestResponse.model_validate(request)
