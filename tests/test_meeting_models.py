import pytest
from sqlalchemy import text


@pytest.mark.asyncio
async def test_meeting_insert_defaults(db_session, test_project):
    from huddleroom.models.meeting import Meeting
    m = Meeting(
        project_id=test_project.id,
        title="Decision on API design",
        meeting_type="decision",
        participant_agent_ids=[],
    )
    db_session.add(m)
    await db_session.flush()

    assert m.id is not None
    assert m.status == "scheduled"
    assert m.turn_strategy == "round_robin"
    assert m.deadlock_strategy == "human_intervention"
    assert m.max_duration_minutes == 30
    assert m.veto_window_hours == 24
    assert m.is_partial is False or m.is_partial == 0
    assert m.created_by_trigger is False or m.created_by_trigger == 0
    assert m.created_at is not None


def test_meeting_create_schema_validates():
    from huddleroom.schemas.meeting import MeetingCreate, AgendaItemCreate
    import uuid
    data = MeetingCreate(
        title="API Design Decision",
        meeting_type="decision",
        participant_agent_ids=[uuid.uuid4()],
        agenda_items=[
            AgendaItemCreate(
                order=1,
                title="REST vs GraphQL",
                question="Which API style to use?",
                options=["REST", "GraphQL"],
                max_rounds=3,
            )
        ],
    )
    assert data.turn_strategy == "round_robin"
    assert data.max_duration_minutes == 30
    assert data.veto_window_hours is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("meeting_type", "veto_window_hours", "expected_veto_window"),
    [
        ("decision", 0, 0),
        ("standup", None, 0),
        ("standup", 2, 2),
    ],
)
async def test_meeting_service_create_veto_window_handling(db_session, test_project, test_agent, meeting_type, veto_window_hours, expected_veto_window):
    from huddleroom.services.meeting_service import MeetingService

    kwargs = {
        "db": db_session,
        "project_id": test_project.id,
        "title": f"Meeting {meeting_type}",
        "meeting_type": meeting_type,
        "participant_agent_ids": [str(test_agent.id)],
        "agenda_items": [{"order": 1, "title": "Item", "max_rounds": 1}],
    }
    if veto_window_hours is not None:
        kwargs["veto_window_hours"] = veto_window_hours

    meeting = await MeetingService().create_meeting(**kwargs)

    assert meeting.veto_window_hours == expected_veto_window


def test_meeting_create_schema_rejects_invalid_type():
    from huddleroom.schemas.meeting import MeetingCreate
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        MeetingCreate(
            title="Bad",
            meeting_type="invalid_type",
            participant_agent_ids=[],
        )


@pytest.mark.asyncio
async def test_meeting_service_create_and_prepare(db_session, test_project, test_agent):
    from huddleroom.services.meeting_service import MeetingService
    svc = MeetingService()

    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Arch Decision",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "REST vs GraphQL", "max_rounds": 2}],
    )
    assert meeting.status == "scheduled"
    assert len(meeting.participant_agent_ids) == 1

    await svc.transition_to_preparing(db=db_session, meeting=meeting)
    await db_session.flush()
    assert meeting.status == "preparing"
    assert meeting.preparing_started_at is not None


@pytest.mark.asyncio
async def test_meeting_service_full_lifecycle(db_session, test_project, test_agent):
    from huddleroom.services.meeting_service import MeetingService
    svc = MeetingService()

    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Lifecycle Test",
        meeting_type="standup",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[],
    )

    await svc.transition_to_preparing(db=db_session, meeting=meeting)
    await svc.transition_to_active(db=db_session, meeting=meeting)
    assert meeting.status == "active"
    assert meeting.active_started_at is not None

    await svc.transition_to_concluding(db=db_session, meeting=meeting)
    assert meeting.status == "concluding"

    await svc.transition_to_concluded(db=db_session, meeting=meeting)
    assert meeting.status == "concluded"
    assert meeting.concluded_at is not None


@pytest.mark.asyncio
async def test_meeting_service_cancel(db_session, test_project):
    from huddleroom.services.meeting_service import MeetingService
    svc = MeetingService()

    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Cancelled Meeting",
        meeting_type="adhoc",
        participant_agent_ids=[],
        agenda_items=[],
    )
    await svc.cancel_meeting(db=db_session, meeting=meeting, reason="No longer needed")
    assert meeting.status == "cancelled"
    assert meeting.cancelled_reason == "No longer needed"
    assert meeting.cancelled_at is not None


@pytest.mark.asyncio
async def test_meeting_service_invalid_transition_raises(db_session, test_project):
    from huddleroom.services.meeting_service import MeetingService, MeetingTransitionError
    svc = MeetingService()

    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Bad Transition",
        meeting_type="decision",
        participant_agent_ids=[],
        agenda_items=[],
    )
    # Cannot jump from scheduled straight to active
    with pytest.raises(MeetingTransitionError):
        await svc.transition_to_active(db=db_session, meeting=meeting)


@pytest.mark.asyncio
async def test_meeting_service_get_current_agenda_item(db_session, test_project):
    from huddleroom.services.meeting_service import MeetingService
    svc = MeetingService()

    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Agenda Test",
        meeting_type="decision",
        participant_agent_ids=[],
        agenda_items=[
            {"order": 1, "title": "Item 1", "max_rounds": 2},
            {"order": 2, "title": "Item 2", "max_rounds": 2},
        ],
    )
    item = await svc.get_current_agenda_item(db=db_session, meeting_id=meeting.id)
    assert item is not None
    assert item.title == "Item 1"
    assert item.status == "active"


@pytest.mark.asyncio
async def test_meeting_service_advance_agenda(db_session, test_project):
    from huddleroom.services.meeting_service import MeetingService
    from huddleroom.models.meeting import MeetingEvent
    from sqlalchemy import select
    svc = MeetingService()

    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Advance Test",
        meeting_type="decision",
        participant_agent_ids=[],
        agenda_items=[
            {"order": 1, "title": "Item 1", "max_rounds": 2},
            {"order": 2, "title": "Item 2", "max_rounds": 2},
        ],
    )
    item1 = await svc.get_current_agenda_item(db=db_session, meeting_id=meeting.id)
    await svc.advance_agenda(
        db=db_session,
        meeting=meeting,
        completed_item=item1,
        resolution="resolved",
        outcome={
            "resolution_kind": "consensus",
            "resolution_summary": "Item 1 was resolved with consensus.",
            "required_followup": "Implement the agreed approach.",
            "participants_heard": [],
        },
    )
    await db_session.flush()

    item2 = await svc.get_current_agenda_item(db=db_session, meeting_id=meeting.id)
    assert item2 is not None
    assert item2.title == "Item 2"
    assert item1.resolution_kind == "consensus"
    assert item1.resolution_summary == "Item 1 was resolved with consensus."
    assert item1.required_followup == "Implement the agreed approach."
    assert item2.started_at is not None

    events = (
        await db_session.execute(
            select(MeetingEvent)
            .where(MeetingEvent.meeting_id == meeting.id)
            .order_by(MeetingEvent.created_at)
        )
    ).scalars().all()
    event_types = [event.event_type for event in events]
    assert "agenda_item_completed" in event_types
    assert "agenda_item_advanced" in event_types
