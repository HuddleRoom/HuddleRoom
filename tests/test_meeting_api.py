import pytest
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch, MagicMock

from huddleroom.models.meeting import MeetingActionItem as _MeetingActionItem


@contextmanager
def patch_streaming_acompletion(service_module, content):
    """Patch a service's litellm.acompletion to stream `content`, and patch the
    agent_response_stream reconstruction seam to rebuild it into a usable response.
    """

    class _Stream:
        def __init__(self):
            self._chunks = [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=content, reasoning_content=None)
                        )
                    ]
                )
            ]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._chunks:
                return self._chunks.pop()
            raise StopAsyncIteration

    async def mock_acompletion(**_kwargs):
        return _Stream()

    def mock_builder(chunks, messages):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=content, reasoning_content=None, tool_calls=None
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
        )

    with patch(f"{service_module}.litellm.acompletion", new=mock_acompletion), patch(
        "huddleroom.services.agent_response_stream.litellm.stream_chunk_builder", new=mock_builder
    ):
        yield


@pytest.mark.asyncio
async def test_extract_action_items_with_decisions(db_session, test_project, test_agent):
    """extract_action_items maps depends_on_decision_id when LLM returns depends_on_decision_title."""
    if not hasattr(_MeetingActionItem, "depends_on_decision_id"):
        pytest.skip("MeetingActionItem.depends_on_decision_id field absent")

    from huddleroom.services.meeting_service import MeetingService
    from huddleroom.services.meeting_outcome import MeetingOutcomeService
    from huddleroom.models.meeting import MeetingDecision, MeetingAgendaItem, MeetingTurn
    from sqlalchemy import select

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Decision Dependency Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "API style", "max_rounds": 1}],
    )
    await svc.transition_to_preparing(db=db_session, meeting=meeting)
    await svc.transition_to_active(db=db_session, meeting=meeting)

    turn = MeetingTurn(
        meeting_id=meeting.id,
        turn_number=1,
        round_number=1,
        speaker_agent_id=test_agent.id,
        content="We should go with REST.",
        references=[],
    )
    db_session.add(turn)
    await db_session.flush()
    await svc.transition_to_concluding(db=db_session, meeting=meeting)

    # Create a real decision
    result = await db_session.execute(
        select(MeetingAgendaItem).where(MeetingAgendaItem.meeting_id == meeting.id).limit(1)
    )
    agenda_item = result.scalar_one()
    decision = MeetingDecision(
        meeting_id=meeting.id,
        agenda_item_id=agenda_item.id,
        title="API choice",
        chosen_option="REST",
        rationale="REST is simpler.",
        decided_by="consensus",
        confidence=0.9,
    )
    db_session.add(decision)
    await db_session.flush()

    llm_payload = (
        f'[{{"description": "Implement REST endpoints", '
        f'"assignee_agent_name": "{test_agent.name}", '
        f'"priority": 75, "deadline_days": 5, '
        f'"depends_on_decision_title": "API choice"}}]'
    )

    outcome_svc = MeetingOutcomeService()
    with patch_streaming_acompletion("huddleroom.services.meeting_outcome", llm_payload):
        action_items = await outcome_svc.extract_action_items(
            db=db_session, meeting=meeting, decisions=[decision]
        )

    assert len(action_items) == 1
    assert action_items[0].description == "Implement REST endpoints"
    assert action_items[0].depends_on_decision_id == decision.id, (
        f"Expected depends_on_decision_id={decision.id}, got {action_items[0].depends_on_decision_id}"
    )


@pytest.mark.asyncio
async def test_outcome_service_extracts_action_items(db_session, test_project, test_agent):
    from huddleroom.services.meeting_service import MeetingService
    from huddleroom.services.meeting_outcome import MeetingOutcomeService
    from huddleroom.models.meeting import MeetingTurn

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Outcome Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "REST vs GraphQL", "max_rounds": 1}],
    )
    await svc.transition_to_preparing(db=db_session, meeting=meeting)
    await svc.transition_to_active(db=db_session, meeting=meeting)

    turn = MeetingTurn(
        meeting_id=meeting.id,
        turn_number=1,
        round_number=1,
        speaker_agent_id=test_agent.id,
        content="We should update the API spec. I will do it by Friday.",
        references=[],
    )
    db_session.add(turn)
    await db_session.flush()
    await svc.transition_to_concluding(db=db_session, meeting=meeting)

    llm_payload = (
        f'[{{"description": "Update API spec", "assignee_agent_name": "{test_agent.name}", '
        f'"priority": 80, "deadline_days": 3}}]'
    )

    outcome_svc = MeetingOutcomeService()
    with patch_streaming_acompletion("huddleroom.services.meeting_outcome", llm_payload):
        action_items = await outcome_svc.extract_action_items(db=db_session, meeting=meeting)

    assert len(action_items) == 1
    assert action_items[0].description == "Update API spec"
    assert action_items[0].assignee_agent_id == test_agent.id
    assert action_items[0].priority == 80


@pytest.mark.asyncio
async def test_outcome_service_falls_back_to_review_followup_action_items(db_session, test_project, test_agent):
    from sqlalchemy import select

    from huddleroom.models.meeting import MeetingAgendaItem, MeetingTurn
    from huddleroom.services.meeting_outcome import MeetingOutcomeService
    from huddleroom.services.meeting_service import MeetingService

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Security Review",
        meeting_type="review",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[
            {"order": 1, "title": "Auth mechanism review", "max_rounds": 2},
            {"order": 2, "title": "Password storage review", "max_rounds": 2},
        ],
    )
    await svc.transition_to_preparing(db=db_session, meeting=meeting)
    await svc.transition_to_active(db=db_session, meeting=meeting)

    turn = MeetingTurn(
        meeting_id=meeting.id,
        turn_number=1,
        round_number=1,
        speaker_agent_id=test_agent.id,
        content="Basic Auth and MD5 both require remediation before approval.",
        references=[],
    )
    db_session.add(turn)
    await db_session.flush()
    await svc.transition_to_concluding(db=db_session, meeting=meeting)

    result = await db_session.execute(
        select(MeetingAgendaItem)
        .where(MeetingAgendaItem.meeting_id == meeting.id)
        .order_by(MeetingAgendaItem.order)
    )
    agenda_items = list(result.scalars().all())
    agenda_items[0].status = "completed"
    agenda_items[0].resolution_kind = "approved_with_followups"
    agenda_items[0].resolution_summary = "Conditional approval pending stronger auth."
    agenda_items[0].required_followup = "Replace Basic Auth with token-based authentication."
    agenda_items[1].status = "completed"
    agenda_items[1].resolution_kind = "rejected"
    agenda_items[1].resolution_summary = "MD5 storage is not acceptable."
    agenda_items[1].required_followup = "Migrate password hashes to Argon2id and rotate credentials."
    await db_session.flush()

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "[]"

    outcome_svc = MeetingOutcomeService()
    with patch("huddleroom.services.meeting_outcome.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_resp
        action_items = await outcome_svc.extract_action_items(db=db_session, meeting=meeting)

    descriptions = [item.description for item in action_items]
    assert len(action_items) == 2
    assert "Replace Basic Auth with token-based authentication." in descriptions
    assert "Migrate password hashes to Argon2id and rotate credentials." in descriptions
    assert all(item.priority == 75 for item in action_items)


@pytest.mark.asyncio
async def test_outcome_service_creates_tasks_from_action_items(db_session, test_project, test_agent):
    from huddleroom.services.meeting_service import MeetingService
    from huddleroom.services.meeting_outcome import MeetingOutcomeService
    from huddleroom.models.meeting import MeetingActionItem, MeetingTurn

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Task Creation Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "Item", "max_rounds": 1}],
    )
    await svc.transition_to_preparing(db=db_session, meeting=meeting)
    await svc.transition_to_active(db=db_session, meeting=meeting)
    turn = MeetingTurn(
        meeting_id=meeting.id, turn_number=1, round_number=1,
        speaker_agent_id=test_agent.id, content="Do the thing.", references=[],
    )
    db_session.add(turn)
    await db_session.flush()

    action_item = MeetingActionItem(
        meeting_id=meeting.id,
        description="Update API spec",
        assignee_agent_id=test_agent.id,
        priority=80,
    )
    db_session.add(action_item)
    await db_session.flush()

    outcome_svc = MeetingOutcomeService()
    created_tasks = await outcome_svc.create_tasks_from_action_items(
        db=db_session, meeting=meeting, action_items=[action_item]
    )

    assert len(created_tasks) == 1
    assert created_tasks[0].title == "Update API spec"
    assert created_tasks[0].assigned_to == test_agent.id
    assert action_item.status == "task_created"
    assert action_item.task_id == created_tasks[0].id


@pytest.mark.asyncio
async def test_outcome_service_writes_knowledge_items(db_session, test_project, test_agent):
    from huddleroom.services.meeting_service import MeetingService
    from huddleroom.services.meeting_outcome import MeetingOutcomeService
    from huddleroom.models.meeting import MeetingDecision, MeetingAgendaItem, MeetingTurn
    from sqlalchemy import select
    from huddleroom.models.knowledge_item import KnowledgeItem

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Knowledge Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "REST vs GraphQL", "max_rounds": 1}],
    )
    await svc.transition_to_preparing(db=db_session, meeting=meeting)
    await svc.transition_to_active(db=db_session, meeting=meeting)
    turn = MeetingTurn(
        meeting_id=meeting.id, turn_number=1, round_number=1,
        speaker_agent_id=test_agent.id, content="We chose REST.", references=[],
    )
    db_session.add(turn)
    await db_session.flush()
    await svc.transition_to_concluding(db=db_session, meeting=meeting)

    item = await db_session.execute(
        select(MeetingAgendaItem).where(MeetingAgendaItem.meeting_id == meeting.id).limit(1)
    )
    item = item.scalar_one()
    decision = MeetingDecision(
        meeting_id=meeting.id,
        agenda_item_id=item.id,
        title="REST vs GraphQL",
        chosen_option="REST",
        rationale="REST is simpler and more widely supported.",
        decided_by="consensus",
        confidence=0.95,
    )
    db_session.add(decision)
    await db_session.flush()

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "The team decided to use REST for the API."

    outcome_svc = MeetingOutcomeService()
    with patch("huddleroom.services.meeting_outcome.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_resp
        kis = await outcome_svc.write_knowledge_items(
            db=db_session, meeting=meeting, decisions=[decision]
        )

    assert len(kis) == 2  # one per decision + one summary
    result = await db_session.execute(
        select(KnowledgeItem).where(KnowledgeItem.project_id == test_project.id)
    )
    persisted = result.scalars().all()
    assert any(ki.content_type == "decision" for ki in persisted)
    assert any(ki.content_type == "summary" for ki in persisted)


from datetime import datetime, timezone, timedelta


@pytest.mark.parametrize("scenario", [
    {
        "name": "simple_decision",
        "payload": {
            "title": "API Design Decision",
            "meeting_type": "decision",
            "participant_agent_ids": None,  # will be filled from test_agent
            "agenda_items": [
                {
                    "order": 1,
                    "title": "REST vs GraphQL",
                    "question": "Which API style?",
                    "options": ["REST", "GraphQL"],
                    "max_rounds": 2,
                }
            ],
            "max_duration_minutes": 30,
            "turn_strategy": "round_robin",
            "auto_start": False,
        },
        "assertions": {
            "title": "API Design Decision",
            "status": "scheduled",
            "meeting_type": "decision",
            "agenda_items_len": 1,
        }
    },
    {
        "name": "balanced_dashboard",
        "payload": {
            "title": "Weekly Architecture Sync",
            "meeting_type": "review",
            "participant_agent_ids": None,  # will be filled from test_agent
            "agenda_items": [
                {
                    "order": 1,
                    "title": "Review API boundary",
                    "description": "Check whether the router shape still fits the service layer.",
                    "question": "Do we keep the current boundary?",
                    "options": ["Keep", "Refine"],
                    "max_rounds": 2,
                },
                {
                    "order": 2,
                    "title": "Agree follow-up",
                    "description": "Capture next action if changes are needed.",
                    "max_rounds": 1,
                },
            ],
            "max_duration_minutes": 45,
            "veto_window_hours": 0,
            "turn_strategy": "agenda_driven",
            "deadlock_strategy": "majority_rules",
            "auto_start": True,
        },
        "assertions": {
            "meeting_type": "review",
            "max_duration_minutes": 45,
            "veto_window_hours": 0,
            "turn_strategy": "agenda_driven",
            "deadlock_strategy": "majority_rules",
            "auto_start": True,
            "agenda_items_len": 2,
            "first_title": "Review API boundary",
            "first_options": ["Keep", "Refine"],
        }
    }
])
@pytest.mark.asyncio
async def test_create_meeting_endpoint(client, auth_headers, test_project, test_agent, scenario):
    payload = scenario["payload"].copy()
    if payload["participant_agent_ids"] is None:
        payload["participant_agent_ids"] = [str(test_agent.id)]

    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/meetings",
        json=payload,
        headers=auth_headers,
    )
    assert resp.status_code == 201
    data = resp.json()

    assertions = scenario["assertions"]
    if "title" in assertions:
        assert data["title"] == assertions["title"]
    if "status" in assertions:
        assert data["status"] == assertions["status"]
    if "meeting_type" in assertions:
        assert data["meeting_type"] == assertions["meeting_type"]
    if "participant_agent_ids" in assertions:
        assert data["participant_agent_ids"] == assertions["participant_agent_ids"]
    if "max_duration_minutes" in assertions:
        assert data["max_duration_minutes"] == assertions["max_duration_minutes"]
    if "veto_window_hours" in assertions:
        assert data["veto_window_hours"] == assertions["veto_window_hours"]
    if "turn_strategy" in assertions:
        assert data["turn_strategy"] == assertions["turn_strategy"]
    if "deadlock_strategy" in assertions:
        assert data["deadlock_strategy"] == assertions["deadlock_strategy"]
    if "auto_start" in assertions:
        assert data["auto_start"] == assertions["auto_start"]
    if "agenda_items_len" in assertions:
        assert len(data["agenda_items"]) == assertions["agenda_items_len"]
    if "first_title" in assertions:
        assert data["agenda_items"][0]["title"] == assertions["first_title"]
    if "first_options" in assertions:
        assert data["agenda_items"][0]["options"] == assertions["first_options"]


@pytest.mark.asyncio
async def test_create_meeting_endpoint_accepts_organizer_settings(
    client, auth_headers, test_project, test_agent, test_user
):
    payload = {
        "title": "Organizer-Controlled Review",
        "meeting_type": "review",
        "participant_agent_ids": [str(test_agent.id)],
        "agenda_items": [{"order": 1, "title": "Pick next speaker", "max_rounds": 2}],
        "turn_strategy": "organizer_controlled",
        "organizer_agent_id": str(test_agent.id),
        "organizer_user_id": str(test_user.id),
        "auto_start": False,
    }
    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/meetings",
        json=payload,
        headers=auth_headers,
    )

    assert resp.status_code == 201
    data = resp.json()
    assert data["turn_strategy"] == "organizer_controlled"
    assert data["organizer_agent_id"] == str(test_agent.id)
    assert data["organizer_user_id"] == str(test_user.id)


@pytest.mark.asyncio
async def test_create_meeting_endpoint_defaults_standup_veto_window_to_zero(
    client, auth_headers, test_project, test_agent
):
    payload = {
        "title": "Daily standup",
        "meeting_type": "standup",
        "participant_agent_ids": [str(test_agent.id)],
        "agenda_items": [{"order": 1, "title": "Updates", "max_rounds": 1}],
        "auto_start": False,
    }
    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/meetings",
        json=payload,
        headers=auth_headers,
    )

    assert resp.status_code == 201
    data = resp.json()
    assert data["meeting_type"] == "standup"
    assert data["veto_window_hours"] == 0


@pytest.mark.asyncio
async def test_list_meetings_endpoint(client, auth_headers, test_project, test_agent):
    for title in ("Meeting A", "Meeting B"):
        await client.post(
            f"/api/v1/projects/{test_project.id}/meetings",
            json={
                "title": title,
                "meeting_type": "standup",
                "participant_agent_ids": [],
                "auto_start": False,
            },
            headers=auth_headers,
        )

    resp = await client.get(
        f"/api/v1/projects/{test_project.id}/meetings",
        headers=auth_headers,
    )
    assert resp.status_code == 200
    meetings = resp.json()
    assert len(meetings) >= 2


@pytest.mark.asyncio
async def test_get_meeting_endpoint(client, auth_headers, test_project, test_agent):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/meetings",
        json={
            "title": "Get Test",
            "meeting_type": "adhoc",
            "participant_agent_ids": [],
            "auto_start": False,
        },
        headers=auth_headers,
    )
    meeting_id = create_resp.json()["id"]

    resp = await client.get(
        f"/api/v1/meetings/{meeting_id}",
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["id"] == meeting_id


@pytest.mark.asyncio
async def test_copy_meeting_endpoint_returns_new_scheduled_meeting(
    client, auth_headers, test_project, test_agent
):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/meetings",
        json={
            "title": "Copy Source",
            "meeting_type": "review",
            "participant_agent_ids": [str(test_agent.id)],
            "agenda_items": [
                {
                    "order": 1,
                    "title": "Review API boundary",
                    "description": "Check the handler split.",
                    "question": "Keep the current split?",
                    "options": ["Keep", "Refine"],
                    "artifact_url": "https://example.com/design-doc",
                    "max_rounds": 2,
                }
            ],
            "max_duration_minutes": 45,
            "turn_strategy": "agenda_driven",
            "deadlock_strategy": "majority_rules",
            "auto_start": False,
        },
        headers=auth_headers,
    )
    source = create_resp.json()

    copy_resp = await client.post(
        f"/api/v1/meetings/{source['id']}/copy",
        headers=auth_headers,
    )

    assert copy_resp.status_code == 201
    copied = copy_resp.json()
    assert copied["id"] != source["id"]
    assert copied["title"] == source["title"]
    assert copied["meeting_type"] == source["meeting_type"]
    assert copied["participant_agent_ids"] == source["participant_agent_ids"]
    assert copied["turn_strategy"] == source["turn_strategy"]
    assert copied["deadlock_strategy"] == source["deadlock_strategy"]
    assert copied["max_duration_minutes"] == source["max_duration_minutes"]
    assert copied["status"] == "scheduled"
    assert copied["summary"] is None
    assert copied["agenda_items"][0]["id"] != source["agenda_items"][0]["id"]
    assert copied["agenda_items"][0]["title"] == source["agenda_items"][0]["title"]
    assert copied["agenda_items"][0]["description"] == source["agenda_items"][0]["description"]
    assert copied["agenda_items"][0]["question"] == source["agenda_items"][0]["question"]
    assert copied["agenda_items"][0]["options"] == source["agenda_items"][0]["options"]
    assert copied["agenda_items"][0]["artifact_url"] == source["agenda_items"][0]["artifact_url"]
    assert copied["agenda_items"][0]["status"] == "pending"


@pytest.mark.asyncio
async def test_cancel_meeting_endpoint(client, auth_headers, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/meetings",
        json={
            "title": "Cancel Test",
            "meeting_type": "adhoc",
            "participant_agent_ids": [],
            "auto_start": False,
        },
        headers=auth_headers,
    )
    meeting_id = create_resp.json()["id"]

    resp = await client.delete(
        f"/api/v1/meetings/{meeting_id}",
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_human_turn_endpoint(client, auth_headers, runnable_project, test_agent, db_session):
    from huddleroom.services.meeting_service import MeetingService

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=runnable_project.id,
        title="Human Turn Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "Item 1", "max_rounds": 2}],
    )
    await svc.transition_to_preparing(db=db_session, meeting=meeting)
    await svc.transition_to_active(db=db_session, meeting=meeting)
    await db_session.flush()

    resp = await client.post(
        f"/api/v1/meetings/{meeting.id}/human-turn",
        json={"content": "I want to add that our SLA requirements favor REST."},
        headers=auth_headers,
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["is_human_turn"] is True
    assert "SLA" in data["content"]


@pytest.mark.asyncio
async def test_human_turn_endpoint_auto_starts_scheduled_meeting(
    client, auth_headers, runnable_project, test_agent, db_session
):
    from huddleroom.services.meeting_service import MeetingService

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=runnable_project.id,
        title="Scheduled Human Turn Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "Item 1", "max_rounds": 2}],
        auto_start=True,
    )
    await db_session.flush()

    resp = await client.post(
        f"/api/v1/meetings/{meeting.id}/human-turn",
        json={"content": "Starting this from the dashboard should activate the meeting."},
        headers=auth_headers,
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["is_human_turn"] is True

    await db_session.refresh(meeting)
    assert meeting.status == "active"


@pytest.mark.asyncio
async def test_human_turn_endpoint_rejects_scheduled_meeting_without_workspace(
    client, auth_headers, test_project, test_agent, db_session
):
    from sqlalchemy import func, select

    from huddleroom.models.meeting import MeetingTurn
    from huddleroom.services.meeting_service import MeetingService

    test_project.workspace_path = None
    meeting = await MeetingService().create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Blocked Scheduled Human Turn",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "Item 1", "max_rounds": 2}],
        auto_start=True,
    )
    await db_session.flush()

    response = await client.post(
        f"/api/v1/meetings/{meeting.id}/human-turn",
        json={"content": "This must not activate an unrunnable project."},
        headers=auth_headers,
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": {"code": "project_not_runnable", "reason": "workspace_unset"}
    }
    await db_session.refresh(meeting)
    assert meeting.status == "scheduled"
    assert await db_session.scalar(
        select(func.count(MeetingTurn.id)).where(MeetingTurn.meeting_id == meeting.id)
    ) == 0


@pytest.mark.asyncio
async def test_human_override_endpoint(client, auth_headers, test_project, test_agent, db_session):
    from huddleroom.services.meeting_service import MeetingService
    from sqlalchemy import select
    from huddleroom.models.meeting import MeetingAgendaItem

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Override Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "REST vs GraphQL", "max_rounds": 2}],
    )
    await svc.transition_to_preparing(db=db_session, meeting=meeting)
    await svc.transition_to_active(db=db_session, meeting=meeting)
    await db_session.flush()

    result = await db_session.execute(
        select(MeetingAgendaItem).where(MeetingAgendaItem.meeting_id == meeting.id).limit(1)
    )
    item = result.scalar_one()

    resp = await client.post(
        f"/api/v1/meetings/{meeting.id}/override",
        json={
            "agenda_item_id": str(item.id),
            "decision": "We will use REST.",
            "reason": "Timeline constraints.",
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["decided_by"] == "human_override"


@pytest.mark.asyncio
async def test_veto_decision_endpoint(client, auth_headers, test_project, test_agent, db_session):
    from huddleroom.services.meeting_service import MeetingService
    from huddleroom.models.meeting import MeetingDecision, MeetingAgendaItem
    from sqlalchemy import select

    svc = MeetingService()
    meeting = await svc.create_meeting(
        db=db_session,
        project_id=test_project.id,
        title="Veto Test",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "REST vs GraphQL", "max_rounds": 1}],
        veto_window_hours=24,
    )
    await svc.transition_to_preparing(db=db_session, meeting=meeting)
    await svc.transition_to_active(db=db_session, meeting=meeting)
    await svc.transition_to_concluding(db=db_session, meeting=meeting)
    await svc.transition_to_concluded(db=db_session, meeting=meeting)

    result = await db_session.execute(
        select(MeetingAgendaItem).where(MeetingAgendaItem.meeting_id == meeting.id).limit(1)
    )
    item = result.scalar_one()
    decision = MeetingDecision(
        meeting_id=meeting.id,
        agenda_item_id=item.id,
        title="Use REST",
        chosen_option="REST",
        rationale="Simpler.",
        decided_by="consensus",
    )
    db_session.add(decision)
    await db_session.flush()

    resp = await client.post(
        f"/api/v1/meetings/{meeting.id}/veto-decision",
        json={"decision_id": str(decision.id), "reason": "Conflicts with security policy."},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["is_vetoed"] is True


@pytest.mark.asyncio
async def test_blocked_task_triggers_meeting(db_session, test_project, test_agent):
    from huddleroom.workers.scheduler import check_blocked_tasks_for_meetings
    from huddleroom.models.task import Task

    # Create a blocked task older than 4 hours
    blocked = Task(
        project_id=test_project.id,
        title="Blocked Task",
        status="blocked",
        assigned_to=test_agent.id,
        priority=70,
    )
    db_session.add(blocked)
    await db_session.flush()

    # Manually set updated_at to 5 hours ago to simulate long-blocked task
    from sqlalchemy import update
    from huddleroom.models.task import Task as TaskModel
    await db_session.execute(
        update(TaskModel)
        .where(TaskModel.id == blocked.id)
        .values(updated_at=datetime.now(timezone.utc) - timedelta(hours=5))
    )
    await db_session.flush()

    with patch("huddleroom.database.AsyncSessionLocal") as mock_factory:
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=db_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("huddleroom.services.meeting_service.MeetingService") as mock_svc_cls:
            mock_svc = AsyncMock()
            mock_svc.create_meeting = AsyncMock(
                return_value=MagicMock(id=uuid.uuid4(), auto_start=True)
            )
            mock_svc_cls.return_value = mock_svc
            await check_blocked_tasks_for_meetings()

    mock_svc.create_meeting.assert_called_once()
    call_kwargs = mock_svc.create_meeting.call_args.kwargs
    assert call_kwargs["meeting_type"] == "adhoc"


@pytest.mark.asyncio
async def test_blocked_task_meeting_emits_scheduled_event_once(db_session, test_project, test_agent):
    from huddleroom.models.event_log import EventLog
    from huddleroom.models.meeting import Meeting
    from huddleroom.models.task import Task
    from huddleroom.workers.scheduler import check_blocked_tasks_for_meetings
    from sqlalchemy import select, update

    blocked = Task(
        project_id=test_project.id,
        title="Blocked Task Event",
        status="blocked",
        assigned_to=test_agent.id,
        priority=70,
    )
    db_session.add(blocked)
    await db_session.flush()
    await db_session.execute(
        update(Task)
        .where(Task.id == blocked.id)
        .values(updated_at=datetime.now(timezone.utc) - timedelta(hours=5))
    )

    with patch("huddleroom.database.AsyncSessionLocal") as mock_factory:
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=db_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch.object(db_session, "commit", new=AsyncMock(side_effect=db_session.flush)):
            await check_blocked_tasks_for_meetings()
            await check_blocked_tasks_for_meetings()

            meetings = list(
                (
                    await db_session.execute(select(Meeting).where(Meeting.source_task_id == blocked.id))
                ).scalars().all()
            )
            events = list(
                (
                    await db_session.execute(
                        select(EventLog).where(
                            EventLog.project_id == test_project.id,
                            EventLog.event_type == "meeting.scheduled",
                        )
                    )
                ).scalars().all()
            )
            assert len(meetings) == 1
            assert len(events) == 1

            meetings[0].status = "concluded"
            await db_session.flush()
            await check_blocked_tasks_for_meetings()
            await check_blocked_tasks_for_meetings()

    meetings = list(
        (
            await db_session.execute(select(Meeting).where(Meeting.source_task_id == blocked.id))
        ).scalars().all()
    )
    events = list(
        (
            await db_session.execute(
                select(EventLog).where(
                    EventLog.project_id == test_project.id,
                    EventLog.event_type == "meeting.scheduled",
                )
            )
        ).scalars().all()
    )

    assert len(meetings) == 2
    assert len(events) == 2
    for meeting in meetings:
        assert meeting.auto_start is True
        event = next(event for event in events if event.payload["meeting_id"] == str(meeting.id))
        assert event.dedup_key == f"meeting.scheduled:meeting:{meeting.id}"
        assert event.payload == {
            "meeting_id": str(meeting.id),
            "auto_start": True,
        }
