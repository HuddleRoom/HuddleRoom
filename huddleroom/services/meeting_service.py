from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingEvent, MeetingParticipantSignal
from huddleroom.models.project import Project
from huddleroom.services.project_service import ProjectService

logger = logging.getLogger(__name__)

_VALID_TRANSITIONS: dict[str, list[str]] = {
    "scheduled": ["preparing", "cancelled"],
    "preparing": ["active", "cancelled"],
    "active": ["concluding", "cancelled"],
    "concluding": ["concluded", "cancelled"],
    "concluded": [],
    "cancelled": [],
}


class MeetingTransitionError(Exception):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MeetingService:

    async def claim_scheduled_meeting(
        self, db: AsyncSession, meeting_id: uuid.UUID
    ) -> Meeting | None:
        """Atomically validate the project and claim a scheduled meeting for execution."""
        await db.execute(
            update(Project)
            .where(Project.id == select(Meeting.project_id).where(Meeting.id == meeting_id).scalar_subquery())
            .values(id=Project.id)
        )
        meeting = await db.get(Meeting, meeting_id, populate_existing=True)
        if not meeting or meeting.status != "scheduled":
            return None
        await ProjectService().require_runnable_project(db, meeting.project_id)
        await self.transition_to_preparing(db, meeting)
        return meeting

    async def create_meeting(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        title: str,
        meeting_type: str,
        participant_agent_ids: list[str],
        agenda_items: list[dict],
        *,
        participant_user_ids: list[str] | None = None,
        turn_strategy: str = "round_robin",
        deadlock_strategy: str = "human_intervention",
        max_duration_minutes: int = 30,
        veto_window_hours: int | None = None,
        auto_start: bool = True,
        scheduled_at: datetime | None = None,
        created_by_agent_id: uuid.UUID | None = None,
        created_by_user_id: uuid.UUID | None = None,
        created_by_trigger: bool = False,
        trigger_reason: str | None = None,
        source_task_id: uuid.UUID | None = None,
        source_protocol_instance_id: uuid.UUID | None = None,
        organizer_agent_id: uuid.UUID | None = None,
        organizer_user_id: uuid.UUID | None = None,
        planner_agent_id: uuid.UUID | None = None,
        signal_check_enabled: bool = False,
    ) -> Meeting:
        effective_veto_window_hours = (
            0 if meeting_type in {"decision", "standup"} and veto_window_hours is None
            else 24 if veto_window_hours is None
            else veto_window_hours
        )
        meeting = Meeting(
            project_id=project_id,
            title=title,
            meeting_type=meeting_type,
            participant_agent_ids=[str(a) for a in participant_agent_ids],
            participant_user_ids=[str(u) for u in (participant_user_ids or [])],
            turn_strategy=turn_strategy,
            deadlock_strategy=deadlock_strategy,
            max_duration_minutes=max_duration_minutes,
            veto_window_hours=effective_veto_window_hours,
            auto_start=auto_start,
            scheduled_at=scheduled_at,
            created_by_agent_id=created_by_agent_id,
            created_by_user_id=created_by_user_id,
            created_by_trigger=created_by_trigger,
            trigger_reason=trigger_reason,
            source_task_id=source_task_id,
            source_protocol_instance_id=source_protocol_instance_id,
            organizer_agent_id=organizer_agent_id,
            organizer_user_id=organizer_user_id,
            planner_agent_id=planner_agent_id,
            signal_check_enabled=signal_check_enabled,
        )
        db.add(meeting)
        await db.flush()

        for i, item_data in enumerate(agenda_items):
            item = MeetingAgendaItem(
                meeting_id=meeting.id,
                order=item_data.get("order", i + 1),
                title=item_data["title"],
                description=item_data.get("description"),
                question=item_data.get("question"),
                options=item_data.get("options"),
                artifact_url=item_data.get("artifact_url"),
                turn_order=item_data.get("turn_order"),
                max_rounds=item_data.get("max_rounds", 3),
                requires_approval=item_data.get("requires_approval", False),
                creates_protocol=item_data.get("creates_protocol", False),
            )
            db.add(item)

        await db.flush()
        await self._log_event(db, meeting.id, "state_transition", {"to": "scheduled"})
        return meeting

    async def copy_meeting(
        self,
        db: AsyncSession,
        source_meeting: Meeting,
        *,
        created_by_user_id: uuid.UUID | None = None,
        title: str | None = None,
        scheduled_at: datetime | None = None,
        auto_start: bool | None = None,
    ) -> Meeting:
        result = await db.execute(
            select(MeetingAgendaItem)
            .where(MeetingAgendaItem.meeting_id == source_meeting.id)
            .order_by(MeetingAgendaItem.order)
        )
        agenda_items = [
            {
                "order": item.order,
                "title": item.title,
                "description": item.description,
                "question": item.question,
                "options": item.options,
                "artifact_url": item.artifact_url,
                "turn_order": item.turn_order,
                "max_rounds": item.max_rounds,
                "requires_approval": item.requires_approval,
                "creates_protocol": item.creates_protocol,
            }
            for item in result.scalars().all()
        ]

        return await self.create_meeting(
            db=db,
            project_id=source_meeting.project_id,
            title=title or source_meeting.title,
            meeting_type=source_meeting.meeting_type,
            participant_agent_ids=list(source_meeting.participant_agent_ids or []),
            participant_user_ids=list(source_meeting.participant_user_ids or []),
            agenda_items=agenda_items,
            turn_strategy=source_meeting.turn_strategy,
            deadlock_strategy=source_meeting.deadlock_strategy,
            max_duration_minutes=source_meeting.max_duration_minutes,
            veto_window_hours=source_meeting.veto_window_hours,
            auto_start=source_meeting.auto_start if auto_start is None else auto_start,
            scheduled_at=scheduled_at,
            created_by_user_id=created_by_user_id,
        )

    async def transition_to_preparing(self, db: AsyncSession, meeting: Meeting) -> None:
        self._assert_transition(meeting, "preparing")
        meeting.status = "preparing"
        meeting.preparing_started_at = _utcnow()
        await self._log_event(db, meeting.id, "state_transition", {"from": "scheduled", "to": "preparing"})

    async def transition_to_active(self, db: AsyncSession, meeting: Meeting) -> None:
        self._assert_transition(meeting, "active")
        meeting.status = "active"
        meeting.active_started_at = _utcnow()
        # Activate first agenda item
        first = await self._first_pending_item(db, meeting.id)
        if first:
            first.status = "active"
            if first.started_at is None:
                first.started_at = _utcnow()
        await self._log_event(db, meeting.id, "state_transition", {"from": "preparing", "to": "active"})

    async def transition_to_concluding(self, db: AsyncSession, meeting: Meeting) -> None:
        self._assert_transition(meeting, "concluding")
        meeting.status = "concluding"
        meeting.concluding_started_at = _utcnow()
        await self._log_event(db, meeting.id, "state_transition", {"from": "active", "to": "concluding"})

    async def transition_to_concluded(self, db: AsyncSession, meeting: Meeting) -> None:
        self._assert_transition(meeting, "concluded")
        meeting.status = "concluded"
        meeting.concluded_at = _utcnow()
        await self._log_event(db, meeting.id, "state_transition", {"from": "concluding", "to": "concluded"})

    async def cancel_meeting(self, db: AsyncSession, meeting: Meeting, reason: str | None = None) -> None:
        old_status = meeting.status
        self._assert_transition(meeting, "cancelled")
        meeting.status = "cancelled"
        meeting.cancelled_at = _utcnow()
        meeting.cancelled_reason = reason
        await self._log_event(db, meeting.id, "state_transition", {
            "from": old_status, "to": "cancelled", "reason": reason
        })

    async def log_event(
        self,
        db: AsyncSession,
        meeting_id: uuid.UUID,
        event_type: str,
        payload: dict | None = None,
        actor_agent_id: uuid.UUID | None = None,
        actor_user_id: uuid.UUID | None = None,
    ) -> None:
        await self._log_event(
            db=db,
            meeting_id=meeting_id,
            event_type=event_type,
            payload=payload,
            actor_agent_id=actor_agent_id,
            actor_user_id=actor_user_id,
        )

    async def get_current_agenda_item(
        self, db: AsyncSession, meeting_id: uuid.UUID
    ) -> MeetingAgendaItem | None:
        result = await db.execute(
            select(MeetingAgendaItem)
            .where(MeetingAgendaItem.meeting_id == meeting_id, MeetingAgendaItem.status == "active")
            .order_by(MeetingAgendaItem.order)
            .limit(1)
        )
        active = result.scalar_one_or_none()
        if active:
            return active
        # Activate first pending item
        first = await self._first_pending_item(db, meeting_id)
        if first:
            first.status = "active"
            if first.started_at is None:
                first.started_at = _utcnow()
            await db.flush()
        return first

    async def advance_agenda(
        self,
        db: AsyncSession,
        meeting: Meeting,
        completed_item: MeetingAgendaItem,
        resolution: str,  # 'resolved' | 'unresolved' | 'tabled' | 'abandoned'
        outcome: dict | None = None,
    ) -> MeetingAgendaItem | None:
        completed_item.status = resolution
        if outcome:
            completed_item.resolution_kind = outcome.get("resolution_kind")
            completed_item.resolution_summary = outcome.get("resolution_summary")
            completed_item.required_followup = outcome.get("required_followup")
            completed_item.participants_heard = outcome.get("participants_heard")
        if resolution == "resolved":
            completed_item.resolved_at = _utcnow()
        await self._log_event(
            db,
            meeting.id,
            "agenda_item_completed",
            {
                "agenda_item_id": str(completed_item.id),
                "resolution": resolution,
                "resolution_kind": completed_item.resolution_kind,
                "resolution_summary": completed_item.resolution_summary,
                "required_followup": completed_item.required_followup,
                "participants_heard": completed_item.participants_heard,
            },
        )
        next_item = await self._first_pending_item(db, meeting.id)
        if next_item:
            next_item.status = "active"
            if next_item.started_at is None:
                next_item.started_at = _utcnow()
            await self._log_event(
                db,
                meeting.id,
                "agenda_item_advanced",
                {
                    "from_agenda_item_id": str(completed_item.id),
                    "to_agenda_item_id": str(next_item.id),
                    "resolution": resolution,
                },
            )
            await db.flush()
            return next_item
        # No more items — transition to concluding
        await self._log_event(
            db,
            meeting.id,
            "agenda_completed_all",
            {
                "final_agenda_item_id": str(completed_item.id),
                "resolution": resolution,
            },
        )
        await self.transition_to_concluding(db=db, meeting=meeting)
        await db.flush()
        return None

    async def is_round_complete(
        self, db: AsyncSession, meeting_id: uuid.UUID, agenda_item_id: uuid.UUID, round_number: int
    ) -> bool:
        from huddleroom.models.meeting import MeetingTurn
        from sqlalchemy import func as sa_func

        meeting = await db.get(Meeting, meeting_id)
        if not meeting:
            return False

        filters = [
            MeetingTurn.meeting_id == meeting_id,
            MeetingTurn.agenda_item_id == agenda_item_id,
            MeetingTurn.round_number == round_number,
            MeetingTurn.is_human_turn.is_(False),
        ]
        if meeting.meeting_type == "decision":
            filters.append(
                or_(
                    MeetingTurn.moderator_note.is_(None),
                    ~MeetingTurn.moderator_note.startswith("validation_error:"),
                )
            )

        result = await db.execute(
            select(sa_func.count(MeetingTurn.id)).where(*filters)  # pylint: disable=not-callable
        )
        turn_count = result.scalar_one()
        participant_count = len(meeting.participant_agent_ids)
        return participant_count > 0 and turn_count >= participant_count

    def next_speaker_round_robin(
        self, participant_ids: list[str], turn_number: int
    ) -> str:
        if not participant_ids:
            raise ValueError("No participants")
        return participant_ids[turn_number % len(participant_ids)]

    def next_speaker_agenda_driven(
        self, item: MeetingAgendaItem, turn_number_in_item: int
    ) -> str | None:
        turn_order = item.turn_order or []
        if not turn_order:
            return None
        if turn_number_in_item >= len(turn_order):
            return None
        return str(turn_order[turn_number_in_item])

    async def grant_turn(
        self, db: AsyncSession, meeting: Meeting, agent_id: uuid.UUID
    ) -> None:
        if meeting.status != "active":
            raise MeetingTransitionError("Cannot grant turn: meeting not active")
        if meeting.turn_strategy != "organizer_controlled":
            raise MeetingTransitionError("Cannot grant turn: strategy is not organizer_controlled")
        meeting.pending_grant_agent_id = agent_id
        await self._log_event(db, meeting.id, "grant_turn", {"agent_id": str(agent_id)})

    async def clear_grant(self, db: AsyncSession, meeting: Meeting) -> None:
        meeting.pending_grant_agent_id = None

    async def get_pending_signals(
        self, db: AsyncSession, meeting_id: uuid.UUID
    ) -> list[MeetingParticipantSignal]:
        result = await db.execute(
            select(MeetingParticipantSignal).where(
                MeetingParticipantSignal.meeting_id == meeting_id,
                MeetingParticipantSignal.acknowledged_at.is_(None),
            ).order_by(MeetingParticipantSignal.created_at)
        )
        return list(result.scalars().all())

    async def add_signal(
        self,
        db: AsyncSession,
        meeting_id: uuid.UUID,
        agent_id: uuid.UUID,
        signal_type: str,
        message: str | None,
    ) -> MeetingParticipantSignal:
        signal = MeetingParticipantSignal(
            meeting_id=meeting_id,
            agent_id=agent_id,
            signal_type=signal_type,
            message=message,
        )
        db.add(signal)
        await db.flush()
        return signal

    async def acknowledge_agent_signals(
        self, db: AsyncSession, meeting_id: uuid.UUID, agent_id: uuid.UUID
    ) -> None:
        now = _utcnow()
        result = await db.execute(
            select(MeetingParticipantSignal).where(
                MeetingParticipantSignal.meeting_id == meeting_id,
                MeetingParticipantSignal.agent_id == agent_id,
                MeetingParticipantSignal.acknowledged_at.is_(None),
            )
        )
        for signal in result.scalars().all():
            signal.acknowledged_at = now

    async def advance_agenda_item_by_organizer(
        self,
        db: AsyncSession,
        meeting: Meeting,
        item: MeetingAgendaItem,
        resolution: str,
    ) -> MeetingAgendaItem | None:
        await self._log_event(
            db, meeting.id, "organizer_advance",
            {"agenda_item_id": str(item.id), "resolution": resolution}
        )
        return await self.advance_agenda(db=db, meeting=meeting, completed_item=item, resolution=resolution)

    def _assert_transition(self, meeting: Meeting, target: str) -> None:
        allowed = _VALID_TRANSITIONS.get(meeting.status, [])
        if target not in allowed:
            raise MeetingTransitionError(
                f"Cannot transition meeting from '{meeting.status}' to '{target}'"
            )

    async def _first_pending_item(
        self, db: AsyncSession, meeting_id: uuid.UUID
    ) -> MeetingAgendaItem | None:
        result = await db.execute(
            select(MeetingAgendaItem)
            .where(
                MeetingAgendaItem.meeting_id == meeting_id,
                MeetingAgendaItem.status == "pending",
            )
            .order_by(MeetingAgendaItem.order)
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def _log_event(
        self,
        db: AsyncSession,
        meeting_id: uuid.UUID,
        event_type: str,
        payload: dict | None = None,
        actor_agent_id: uuid.UUID | None = None,
        actor_user_id: uuid.UUID | None = None,
    ) -> None:
        event = MeetingEvent(
            meeting_id=meeting_id,
            event_type=event_type,
            payload=payload,
            actor_agent_id=actor_agent_id,
            actor_user_id=actor_user_id,
        )
        db.add(event)
