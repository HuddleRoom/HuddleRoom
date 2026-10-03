"""Public steering endpoint contract."""

import uuid
from datetime import timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from huddleroom.database import get_db
from huddleroom.dependencies import get_current_user
from huddleroom.main import create_app
import huddleroom.routers.orchestration_goals as goals_router
from huddleroom.schemas.orchestration import OrchestrationSteeringSubmitRequest
from huddleroom.models import OrchestrationSteeringRequest, OrchestrationSteeringTransition
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationDecision, OrchestrationGoal
from huddleroom.models.orchestration_steering import OrchestrationSteeringState
from huddleroom.models.event_log import EventLog
from huddleroom.models.base import _utcnow
from huddleroom.models.user import User
from huddleroom.security import create_access_token, hash_password
from huddleroom.models.orchestration_conversation import (
    ConversationMessage, ConversationResponse, conversation_response_id,
)
from huddleroom.models.orchestration_steering import OrchestrationSteeringProposal
from huddleroom.services.orchestration_conversation_service import OrchestrationConversationService
from huddleroom.services.orchestration_service import OrchestrationService


@pytest.fixture(autouse=True)
def conversation_service(test_engine, monkeypatch):
    async def completion(**_kwargs):
        return {"choices": [{"message": {"content": "Safe answer"}}], "usage": {}}

    monkeypatch.setattr(
        goals_router, "conversation_service", OrchestrationConversationService(
            async_sessionmaker(test_engine, expire_on_commit=False), completion,
            orchestration_service=OrchestrationService(),
        ),
    )


@pytest.fixture
async def steering_client(test_engine, test_user):
    app = create_app()
    factory = async_sessionmaker(test_engine, expire_on_commit=False)

    async def override_get_db():
        async with factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db
    async def override_get_current_user():
        return test_user
    app.dependency_overrides[get_current_user] = override_get_current_user
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as value:
        yield value


def _payload(goal_id: uuid.UUID, request_id: uuid.UUID | None = None) -> dict:
    return {
        "client_request_id": str(request_id or uuid.uuid4()),
        "directive": "Prioritize validation before other unstarted work.",
        "target_type": "goal",
        "target_id": str(goal_id),
        "scope": "run",
        "lifetime": "remaining_current_run",
        "impact_summary": "Affects future decisions for unstarted work in this run.",
        "source_proposal_id": None,
        "supersedes_request_id": None,
    }


async def _steering_counts(test_engine):
    async with async_sessionmaker(test_engine, expire_on_commit=False)() as db:
        return (
            await db.scalar(select(func.count(OrchestrationSteeringRequest.id))),
            await db.scalar(select(func.count(OrchestrationSteeringTransition.id))),
            await db.scalar(select(func.count(OrchestrationSteeringState.id))),
            await db.scalar(select(func.count(EventLog.id)).where(
                EventLog.event_type == "orchestration.steering_changed"
            )),
            await db.scalar(select(func.count(OrchestrationDecision.id))),
            await db.scalar(select(func.count(OrchestrationAction.id))),
        )


async def _authorize(goal, run, db_session):
    goal.status = "active"
    run.status = "running"
    run.phase = "authorized"
    await db_session.commit()


async def test_disabled_and_invalid_submissions_create_no_steering_rows(
    steering_client, conversation_goal_run, auth_headers, db_session, monkeypatch, test_engine
):
    goal, run = conversation_goal_run
    await _authorize(goal, run, db_session)
    async with async_sessionmaker(test_engine, expire_on_commit=False)() as check:
        assert await check.get(type(goal), goal.id) is not None
    url = f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation/steering"
    before = await _steering_counts(test_engine)

    monkeypatch.setattr("huddleroom.services.orchestration_steering.settings.orchestration_conversation_steering_enabled", False)
    disabled = await steering_client.post(url, json=_payload(goal.id), headers=auth_headers)
    assert disabled.status_code == 404
    assert disabled.json()["detail"]["code"] == "steering_disabled"
    assert await _steering_counts(test_engine) == before

    monkeypatch.setattr("huddleroom.services.orchestration_steering.settings.orchestration_conversation_steering_enabled", True)
    invalid = await steering_client.post(url, json={**_payload(goal.id), "extra": True}, headers=auth_headers)
    assert invalid.status_code == 422
    assert await _steering_counts(test_engine) == before

    enabled = await steering_client.post(url, json=_payload(goal.id), headers=auth_headers)
    assert enabled.status_code == 200
    monkeypatch.setattr("huddleroom.services.orchestration_steering.settings.orchestration_conversation_steering_enabled", False)
    history = await steering_client.get(
        f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation",
        headers=auth_headers,
    )
    assert history.json()["steering"] == {
        "enabled": False, "eligibility": "active", "eligibility_reason": None,
        "inbox_version": 0, "direction_version": 0, "requests": [], "proposals": [],
    }


@pytest.mark.parametrize("field", ["directive", "impact_summary", "target_id"])
@pytest.mark.parametrize("value", [None, {"not": "text"}])
async def test_non_string_steering_fields_are_422_without_side_effects(
    steering_client, conversation_goal_run, db_session, monkeypatch, test_engine, field, value
):
    goal, run = conversation_goal_run
    await _authorize(goal, run, db_session)
    monkeypatch.setattr("huddleroom.services.orchestration_steering.settings.orchestration_conversation_steering_enabled", True)
    before = await _steering_counts(test_engine)
    response = await steering_client.post(
        f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation/steering",
        json={**_payload(goal.id), field: value},
    )
    assert response.status_code == 422
    assert await _steering_counts(test_engine) == before


@pytest.mark.unsupported_mode
async def test_http_unauthenticated_and_non_owner_submissions_have_no_side_effects(
    conversation_goal_run, db_session, monkeypatch, test_engine
):
    goal, run = conversation_goal_run
    outsider = User(
        email=f"outsider-{uuid.uuid4()}@example.com", hashed_password=hash_password("password"),
        display_name="Outsider", role="member",
    )
    db_session.add(outsider)
    await db_session.flush()
    await _authorize(goal, run, db_session)
    monkeypatch.setattr("huddleroom.dependencies.settings.auth_enabled", True)
    monkeypatch.setattr("huddleroom.services.orchestration_steering.settings.orchestration_conversation_steering_enabled", True)
    app = create_app()
    factory = async_sessionmaker(test_engine, expire_on_commit=False)

    async def override_get_db():
        async with factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db
    url = f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation/steering"
    before = await _steering_counts(test_engine)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        unauthenticated = await http.post(url, json=_payload(goal.id))
        non_owner = await http.post(
            url, json=_payload(goal.id),
            headers={"Authorization": f"Bearer {create_access_token({'sub': str(outsider.id)})}"},
        )
    assert unauthenticated.status_code == 401
    assert non_owner.status_code == 403
    assert non_owner.json()["detail"]["code"] == "steering_forbidden"
    assert await _steering_counts(test_engine) == before


async def test_active_submit_replays_and_withdraws_pending_request(
    steering_client, conversation_goal_run, auth_headers, db_session, monkeypatch, test_engine
):
    goal, run = conversation_goal_run
    await _authorize(goal, run, db_session)
    monkeypatch.setattr("huddleroom.services.orchestration_steering.settings.orchestration_conversation_steering_enabled", True)
    url = f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation/steering"
    request_id = uuid.uuid4()

    submitted = await steering_client.post(url, json=_payload(goal.id, request_id), headers=auth_headers)
    assert submitted.status_code == 200
    body = submitted.json()
    assert set(body) == {
        "request_id", "client_request_id", "sequence", "directive", "target_type", "target_id",
        "scope", "lifetime", "impact_summary", "source_proposal_id", "supersedes_request_id",
        "status", "reason_code", "submitted_at", "considered_at", "finished_at", "updated_at",
        "transitions", "result_action_ids",
    }
    assert body["status"] == "pending"
    assert body["result_action_ids"] == []
    assert body["transitions"][0]["status"] == "pending"
    assert (await _steering_counts(test_engine))[:2] == (1, 1)

    replay = await steering_client.post(url, json=_payload(goal.id, request_id), headers=auth_headers)
    assert replay.status_code == 200
    assert replay.json()["request_id"] == body["request_id"]
    assert (await _steering_counts(test_engine))[:2] == (1, 1)

    conflict = await steering_client.post(url, json={**_payload(goal.id, request_id), "directive": "Different"}, headers=auth_headers)
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "steering_idempotency_conflict"

    withdrawn = await steering_client.post(f"{url}/{body['request_id']}/withdraw", headers=auth_headers)
    assert withdrawn.status_code == 200
    assert withdrawn.json()["status"] == "withdrawn"
    assert (await _steering_counts(test_engine))[:2] == (1, 2)
    not_pending = await steering_client.post(f"{url}/{body['request_id']}/withdraw", headers=auth_headers)
    assert not_pending.status_code == 409
    assert not_pending.json()["detail"]["code"] == "steering_not_pending"


async def test_history_projects_disabled_empty_steering_ledger(
    steering_client, conversation_goal_run, auth_headers, db_session, test_engine, monkeypatch
):
    goal, run = conversation_goal_run
    await _authorize(goal, run, db_session)
    async with async_sessionmaker(test_engine, expire_on_commit=False)() as check:
        assert await check.get(type(goal), goal.id) is not None
    monkeypatch.setattr("huddleroom.services.orchestration_steering.settings.orchestration_conversation_steering_enabled", False)

    response = await steering_client.get(
        f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation",
        headers=auth_headers,
    )

    assert response.status_code == 200, response.text
    assert response.json()["steering"] == {
        "enabled": False,
        "eligibility": "active",
        "eligibility_reason": None,
        "inbox_version": 0,
        "direction_version": 0,
        "requests": [],
        "proposals": [],
    }


@pytest.mark.parametrize(
    ("field", "value"), [
        ("directive", " " + "x" * 4_000 + " "),
        ("impact_summary", " " + "x" * 1_000 + " "),
        ("target_id", " " + "x" * 255 + " "),
    ],
)
async def test_submit_strips_before_enforcing_bounds(
    steering_client, conversation_goal_run, db_session, monkeypatch, field, value
):
    goal, run = conversation_goal_run
    await _authorize(goal, run, db_session)
    monkeypatch.setattr("huddleroom.services.orchestration_steering.settings.orchestration_conversation_steering_enabled", True)
    payload = _payload(goal.id)
    payload[field] = value
    if field == "target_id":
        parsed = OrchestrationSteeringSubmitRequest.model_validate(payload)
        assert parsed.target_id == value.strip()
        return

    response = await steering_client.post(
        f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation/steering",
        json=payload,
    )

    assert response.status_code == 200
    assert response.json()[field] == value.strip()


async def test_history_attaches_only_matching_public_proposal(
    steering_client, conversation_goal_run, db_session, monkeypatch, test_engine
):
    goal, run = conversation_goal_run
    message = ConversationMessage(
        id=uuid.uuid4(), goal_id=goal.id, actor_id=goal.created_by_user_id,
        client_request_id=uuid.uuid4(), sequence=1, content="question",
    )
    now = _utcnow()
    response = ConversationResponse(
        id=conversation_response_id(message.id), message_id=message.id, run_id=run.id,
        status="completed", dossier={}, context_manifest={}, context_version="v1",
        provider_request_id="safe", answer="answer", started_at=now - timedelta(seconds=2),
        deadline_at=now - timedelta(seconds=1), finished_at=now,
    )
    proposal = OrchestrationSteeringProposal(
        id=uuid.uuid4(), response_id=response.id, goal_id=goal.id, actor_id=goal.created_by_user_id,
        status="proposed", draft={
            "directive": "Validate first", "target_type": "goal", "target_id": str(goal.id),
            "scope": "run", "lifetime": "remaining_current_run", "impact_summary": "Safe impact",
        },
    )
    other_message = ConversationMessage(
        id=uuid.uuid4(), goal_id=goal.id, actor_id=goal.created_by_user_id,
        client_request_id=uuid.uuid4(), sequence=2, content="other question",
    )
    other_response = ConversationResponse(
        id=conversation_response_id(other_message.id), message_id=other_message.id, run_id=run.id,
        status="completed", dossier={}, context_manifest={}, context_version="v1",
        provider_request_id="safe-other", answer="other answer", started_at=now - timedelta(seconds=2),
        deadline_at=now - timedelta(seconds=1), finished_at=now,
    )
    db_session.add_all((message, response, proposal, other_message, other_response))
    await db_session.flush()
    await _authorize(goal, run, db_session)
    monkeypatch.setattr("huddleroom.services.orchestration_steering.settings.orchestration_conversation_steering_enabled", True)

    history = await steering_client.get(
        f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation"
    )

    assert history.status_code == 200
    turn = history.json()["items"][0]
    proposal_body = turn["proposed_steering"]
    assert {key: value for key, value in proposal_body.items() if key not in {"created_at", "updated_at"}} == {
        "proposal_id": str(proposal.id), "response_id": str(response.id), "status": "proposed",
        "directive": "Validate first", "target_type": "goal", "target_id": str(goal.id),
        "scope": "run", "lifetime": "remaining_current_run", "impact_summary": "Safe impact",
        "dismissed_at": None, "promoted_request_id": None,
    }
    assert proposal_body["created_at"] and proposal_body["updated_at"]
    assert history.json()["items"][1]["proposed_steering"] is None

    before_dismiss = await _steering_counts(test_engine)
    dismissed = await steering_client.post(
        f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation/steering/proposals/{proposal.id}/dismiss"
    )
    assert dismissed.status_code == 200
    assert dismissed.json()["status"] == "dismissed"
    assert await _steering_counts(test_engine) == before_dismiss


@pytest.mark.parametrize(
    ("goal_status", "run_status", "phase", "expected"), [
        ("active", "running", "ready", 409),
        ("completed", "completed", "completed", 409),
        ("paused", "paused", "authorized", 200),
    ],
)
async def test_lifecycle_accepts_only_active_or_paused_without_side_effects(
    steering_client, conversation_goal_run, db_session, monkeypatch, test_engine,
    goal_status, run_status, phase, expected,
):
    goal, run = conversation_goal_run
    goal.status, run.status, run.phase = goal_status, run_status, phase
    await db_session.commit()
    monkeypatch.setattr("huddleroom.services.orchestration_steering.settings.orchestration_conversation_steering_enabled", True)
    response = await steering_client.post(
        f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation/steering",
        json=_payload(goal.id),
    )
    assert response.status_code == expected
    assert await _steering_counts(test_engine) == (
        (1, 1, 1, 1, 0, 0) if expected == 200 else (0, 0, 0, 0, 0, 0)
    )
    if expected == 200:
        assert response.json()["status"] == "pending"


async def test_conversation_history_without_run_keeps_chat_readable_and_steering_empty(
    steering_client, conversation_goal_run, db_session, test_user,
):
    original_goal, _run = conversation_goal_run
    goal = OrchestrationGoal(
        project_id=original_goal.project_id, objective="No run yet", success_criteria=[],
        constraints={}, budget={}, manager_user_id=test_user.id,
    )
    db_session.add(goal)
    await db_session.commit()
    url = f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation"

    response = await steering_client.get(url)
    assert response.status_code == 200
    assert response.json()["items"] == []
    assert response.json()["steering"] == {
        "enabled": True, "eligibility": "unstarted", "eligibility_reason": "steering_ineligible",
        "inbox_version": 0, "direction_version": 0, "requests": [], "proposals": [],
    }
