"""RED contract for the typed conversation feedback and learning API boundary."""

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import huddleroom.dependencies as dependencies
import huddleroom.routers.orchestration_goals as goals_router
from huddleroom.config import Settings, settings
from huddleroom.database import get_db
from huddleroom.main import create_app
from huddleroom.models.orchestration import OrchestrationGoal
from huddleroom.models.project import Project
from huddleroom.models.orchestration_conversation import (
    ConversationFeedback,
    ConversationMessage,
    ConversationResponse,
    conversation_response_id,
)
from huddleroom.services.orchestration_conversation_learning import ConversationLearningError


REASONS = (
    "unanswered",
    "incorrect",
    "missing_context",
    "stale_context",
    "unclear",
    "too_limited",
    "other",
)


@pytest.fixture
async def learning_client(test_engine):
    app = create_app()
    factory = async_sessionmaker(test_engine, expire_on_commit=False)

    async def override_get_db():
        async with factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


async def _response(db, goal, run, actor_id, sequence, *, status="completed", answer="answer", now=None):
    now = now or datetime.now(timezone.utc)
    message = ConversationMessage(
        goal_id=goal.id,
        actor_id=actor_id,
        client_request_id=uuid.uuid4(),
        sequence=sequence,
        content=f"private-question-{sequence}",
        created_at=now,
    )
    db.add(message)
    await db.flush()
    response = ConversationResponse(
        id=conversation_response_id(message.id),
        message_id=message.id,
        run_id=run.id,
        status=status,
        dossier={"private": "never report"},
        context_manifest={},
        context_version="learning-api-v1",
        provider_request_id=f"learning-api-{sequence}",
        answer=answer,
        started_at=now - timedelta(seconds=60) if status in {"running", "completed", "interrupted_unknown"} else None,
        deadline_at=now if status in {"running", "completed", "interrupted_unknown"} else None,
        finished_at=now if status in {"completed", "failed", "interrupted_unknown"} else None,
    )
    db.add(response)
    await db.flush()
    return response


async def _feedback_count(test_engine):
    async with async_sessionmaker(test_engine, expire_on_commit=False)() as db:
        return await db.scalar(select(func.count()).select_from(ConversationFeedback))


def _feedback_url(goal, response_id):
    return (
        f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}"
        f"/conversation/{response_id}/feedback"
    )


def _report_url(project_id):
    return f"/api/v1/projects/{project_id}/orchestration/conversation-learning"


def _assert_zero_report(body, start_at, end_at):
    assert set(body) == {
        "window", "sample", "chat", "investigations", "context_limits", "steering", "operator_feedback",
    }
    assert body["window"]["start_at"] == start_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    assert body["window"]["end_at"] == end_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    generated_at = datetime.fromisoformat(body["window"]["generated_at"].replace("Z", "+00:00"))
    assert generated_at.tzinfo is not None and generated_at.utcoffset() == timedelta(0)
    assert body["window"]["goal_filtered"] is False
    assert body["sample"] == {"goals": 0, "operators": 0, "questions": 0}
    assert body["chat"] == {
        "status_counts": {"pending": 0, "running": 0, "completed": 0, "failed": 0, "interrupted_unknown": 0},
        "answered": 0,
        "terminal_without_answer": 0,
        "latency_seconds": {"count": 0, "invalid": 0, "median": None, "p95": None},
    }
    assert body["investigations"] == {
        "triggered": 0,
        "accounted_terminal": 0,
        "status_counts": {key: 0 for key in (
            "pending", "running", "completed", "limited", "failed", "cancelled", "unavailable", "interrupted_unknown",
        )},
        "attempts": 0,
        "repairs": 0,
        "retries": 0,
        "accumulated_tokens": 0,
        "latency_seconds": {"count": 0, "invalid": 0, "median": None, "p95": None},
    }
    assert body["context_limits"] == {
        "turns_truncated": 0,
        "turns_with_omissions": 0,
        "omitted_records": 0,
        "truncated_source_counts": {key: 0 for key in (
            "goal", "run", "accepted_plan", "decisions", "actions", "gates", "evidence", "processes",
            "warnings", "memory", "artifact", "agents", "prior_turns",
        )},
        "investigation_omission_status_counts": {key: 0 for key in (
            "restricted", "unsafe", "binary", "too_large", "changed", "omitted_by_limit",
        )},
    }
    assert body["steering"] == {
        "proposal_status_counts": {"proposed": 0, "dismissed": 0, "promoted": 0},
        "request_status_counts": {key: 0 for key in (
            "pending", "being_considered", "applied", "deferred", "rejected", "superseded", "needs_clarification", "withdrawn",
        )},
        "request_reason_counts": {key: 0 for key in (
            "submitted", "considering", "run_changed", "steering_ineligible", "target_already_started",
            "supersedes_required", "invalid_supersedes_request", "superseded", "applied", "withdrawn",
        )},
        "direct_requests": 0,
        "proposal_derived_requests": 0,
        "requests_with_result_actions": 0,
        "result_links": 0,
        "submit_to_considered_seconds": {"count": 0, "invalid": 0, "median": None, "p95": None},
        "submit_to_finished_seconds": {"count": 0, "invalid": 0, "median": None, "p95": None},
    }
    assert body["operator_feedback"] == {
        "rated": 0,
        "unrated_answered": 0,
        "helpful": 0,
        "not_helpful": 0,
        "not_helpful_reason_counts": {reason: 0 for reason in REASONS},
    }


async def test_feedback_create_and_exact_replay_return_the_same_typed_row(
    learning_client, conversation_goal_run, db_session
):
    goal, run = conversation_goal_run
    response = await _response(db_session, goal, run, uuid.UUID(int=0), 1)
    await db_session.commit()

    body = {"rating": "not_helpful", "reason": "unclear"}
    created = await learning_client.put(_feedback_url(goal, response.id), json=body)
    replayed = await learning_client.put(_feedback_url(goal, response.id), json=body)

    assert created.status_code == replayed.status_code == 200
    assert created.json() == replayed.json()
    assert set(created.json()) == {"feedback_id", "rating", "reason", "created_at"}
    assert created.json()["rating"] == "not_helpful"
    assert created.json()["reason"] == "unclear"
    assert await _feedback_count(db_session.bind) == 1


async def test_feedback_conflict_and_lineage_errors_preserve_immutable_history(
    learning_client, conversation_goal_run, db_session, test_user
):
    goal, run = conversation_goal_run
    owned = await _response(db_session, goal, run, uuid.UUID(int=0), 1)
    other_owner = await _response(db_session, goal, run, test_user.id, 2)
    await db_session.commit()
    url = _feedback_url(goal, owned.id)

    assert (await learning_client.put(url, json={"rating": "helpful"})).status_code == 200
    conflict = await learning_client.put(url, json={"rating": "not_helpful", "reason": "incorrect"})
    forbidden = await learning_client.put(
        _feedback_url(goal, other_owner.id), json={"rating": "helpful"}
    )
    missing = await learning_client.put(
        _feedback_url(goal, uuid.uuid4()), json={"rating": "helpful"}
    )

    assert (conflict.status_code, conflict.json()["detail"]["code"]) == (409, "conversation_feedback_already_recorded")
    assert (forbidden.status_code, forbidden.json()["detail"]["code"]) == (403, "conversation_feedback_forbidden")
    assert (missing.status_code, missing.json()["detail"]) == (
        404,
        {"code": "conversation_feedback_not_found", "message": "Conversation response not found"},
    )
    assert await _feedback_count(db_session.bind) == 1
    async with async_sessionmaker(db_session.bind, expire_on_commit=False)() as db:
        saved = await db.scalar(select(ConversationFeedback))
    assert (saved.rating, saved.reason) == ("helpful", None)


@pytest.mark.parametrize("reason", REASONS)
async def test_feedback_accepts_every_fixed_negative_reason(
    learning_client, conversation_goal_run, db_session, reason
):
    goal, run = conversation_goal_run
    response = await _response(db_session, goal, run, uuid.UUID(int=0), REASONS.index(reason) + 1)
    await db_session.commit()

    result = await learning_client.put(
        _feedback_url(goal, response.id), json={"rating": "not_helpful", "reason": reason}
    )

    assert result.status_code == 200
    assert result.json()["reason"] == reason


@pytest.mark.parametrize(
    "payload",
    (
        {"rating": "helpful", "reason": "unclear"},
        {"rating": "not_helpful"},
        {"rating": "not_helpful", "reason": None},
        {"rating": "unknown"},
        {"rating": "not_helpful", "reason": "unknown"},
        {"rating": "helpful", "unexpected": True},
    ),
)
async def test_feedback_request_rejects_invalid_or_extra_values_before_writing(
    learning_client, conversation_goal_run, db_session, payload
):
    goal, run = conversation_goal_run
    response = await _response(db_session, goal, run, uuid.UUID(int=0), 1)
    await db_session.commit()

    result = await learning_client.put(_feedback_url(goal, response.id), json=payload)

    assert result.status_code == 422
    assert await _feedback_count(db_session.bind) == 0


@pytest.mark.parametrize("status,answer", (
    ("pending", "answer"), ("running", "answer"), ("failed", "answer"),
    ("interrupted_unknown", "answer"), ("completed", None), ("completed", ""), ("completed", " \t "),
))
async def test_feedback_rejects_the_ineligible_lifecycle_and_blank_answer_matrix(
    learning_client, conversation_goal_run, db_session, status, answer
):
    goal, run = conversation_goal_run
    response = await _response(db_session, goal, run, uuid.UUID(int=0), 1, status=status, answer=answer)
    await db_session.commit()

    result = await learning_client.put(_feedback_url(goal, response.id), json={"rating": "helpful"})

    assert (result.status_code, result.json()["detail"]) == (
        409, {"code": "conversation_feedback_ineligible", "message": "Conversation feedback is ineligible"},
    )
    assert await _feedback_count(db_session.bind) == 0


async def test_feedback_historical_owner_answer_ignores_current_runtime_flags(
    learning_client, conversation_goal_run, db_session, monkeypatch
):
    goal, run = conversation_goal_run
    response = await _response(db_session, goal, run, uuid.UUID(int=0), 1)
    await db_session.commit()
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", 0)
    monkeypatch.setattr(settings, "orchestration_conversation_investigation_enabled", False)
    monkeypatch.setattr(settings, "orchestration_conversation_steering_enabled", False)

    result = await learning_client.put(_feedback_url(goal, response.id), json={"rating": "helpful"})

    assert result.status_code == 200
    assert result.json()["rating"] == "helpful"


async def test_feedback_rolls_back_staged_writes_before_commit_on_response_or_domain_error(
    learning_client, conversation_goal_run, db_session, monkeypatch
):
    goal, run = conversation_goal_run
    response = await _response(db_session, goal, run, uuid.UUID(int=0), 1)
    await db_session.commit()
    commits, rollbacks = [], []
    rollback = AsyncSession.rollback

    async def track_commit(self, *_args, **_kwargs):
        commits.append(self)

    async def track_rollback(self, *_args, **_kwargs):
        rollbacks.append(self)
        return await rollback(self)

    def invalid_response(**_kwargs):
        raise ValueError("response serialization failed")

    monkeypatch.setattr(AsyncSession, "commit", track_commit)
    monkeypatch.setattr(AsyncSession, "rollback", track_rollback)
    monkeypatch.setattr(goals_router, "OrchestrationConversationFeedbackResponse", invalid_response)

    with pytest.raises(ValueError, match="response serialization failed"):
        await learning_client.put(_feedback_url(goal, response.id), json={"rating": "helpful"})

    assert commits == []
    assert rollbacks
    assert await _feedback_count(db_session.bind) == 0

    async def stage_then_fail(db, *_args, **_kwargs):
        db.add(ConversationFeedback(
            id=uuid.uuid4(), response_id=response.id, actor_id=uuid.UUID(int=0),
            rating="helpful", reason=None,
        ))
        raise ConversationLearningError("staged_error", 409, "staged domain error")

    monkeypatch.setattr(goals_router.learning_service, "record_feedback", stage_then_fail)
    domain = await learning_client.put(_feedback_url(goal, response.id), json={"rating": "helpful"})

    assert (domain.status_code, domain.json()["detail"]["code"]) == (409, "staged_error")
    assert commits == []
    assert len(rollbacks) >= 2
    assert await _feedback_count(db_session.bind) == 0


async def test_learning_report_has_exact_zero_and_nonzero_content_free_fixed_shapes(
    learning_client, conversation_goal_run, db_session, concurrent_sessions
):
    goal, run = conversation_goal_run
    await db_session.commit()
    start_at = datetime.now(timezone.utc) - timedelta(hours=1)
    end_at = datetime.now(timezone.utc) + timedelta(hours=1)
    params = {"start_at": start_at.isoformat(), "end_at": end_at.isoformat()}

    zero = await learning_client.get(_report_url(goal.project_id), params=params)
    assert zero.status_code == 200
    _assert_zero_report(zero.json(), start_at, end_at)

    writer, _ = concurrent_sessions
    response = await _response(writer, goal, run, uuid.UUID(int=0), 1, answer="PRIVATE_ANSWER")
    writer.add(ConversationFeedback(
        id=uuid.uuid4(), response_id=response.id, actor_id=uuid.UUID(int=0), rating="helpful", reason=None,
    ))
    await writer.commit()
    nonzero = await learning_client.get(_report_url(goal.project_id), params=params)

    assert nonzero.status_code == 200
    body = nonzero.json()
    assert body["sample"] == {"goals": 1, "operators": 1, "questions": 1}
    assert body["chat"]["status_counts"] == {
        "pending": 0, "running": 0, "completed": 1, "failed": 0, "interrupted_unknown": 0,
    }
    assert body["chat"]["answered"] == 1
    assert body["operator_feedback"] == {
        "rated": 1, "unrated_answered": 0, "helpful": 1, "not_helpful": 0,
        "not_helpful_reason_counts": {reason: 0 for reason in REASONS},
    }
    assert "PRIVATE_ANSWER" not in json.dumps(body)
    assert str(goal.id) not in json.dumps(body)


@pytest.mark.parametrize(
    "start_at,end_at",
    (
        ("2026-09-01T00:00:00", "2026-09-02T00:00:00Z"),
        ("2026-09-02T00:00:00Z", "2026-09-02T00:00:00Z"),
        ("2026-09-03T00:00:00Z", "2026-09-02T00:00:00Z"),
        ("2026-01-01T00:00:00Z", "2026-04-02T00:00:01Z"),
    ),
)
async def test_learning_report_rejects_invalid_windows_before_aggregation(
    learning_client, conversation_goal_run, db_session, start_at, end_at
):
    goal, _ = conversation_goal_run
    await db_session.commit()
    result = await learning_client.get(
        _report_url(goal.project_id), params={"start_at": start_at, "end_at": end_at}
    )

    assert (result.status_code, result.json()["detail"]) == (
        422,
        {"code": "conversation_learning_invalid_window", "message": "Conversation learning window is invalid"},
    )


@pytest.mark.unsupported_mode
async def test_learning_routes_authenticate_before_project_or_service_access(
    learning_client, conversation_goal_run, monkeypatch
):
    goal, _ = conversation_goal_run
    called = False

    async def forbidden_project_lookup(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("authentication must run first")

    monkeypatch.setattr(goals_router, "ensure_project_exists", forbidden_project_lookup)
    monkeypatch.setattr(dependencies, "settings", Settings(auth_enabled=True))
    start_at = datetime.now(timezone.utc)
    end_at = start_at + timedelta(minutes=1)
    requests = (
        learning_client.put(
            _feedback_url(goal, uuid.uuid4()), json={"rating": "helpful"}
        ),
        learning_client.get(
            _report_url(goal.project_id), params={"start_at": start_at.isoformat(), "end_at": end_at.isoformat()}
        ),
    )
    responses = await asyncio.gather(*requests)

    assert [response.status_code for response in responses] == [401, 401]
    assert called is False


async def test_learning_report_uses_existing_project_boundary_and_validates_goal_and_half_open_window(
    learning_client, conversation_goal_run, db_session
):
    goal, run = conversation_goal_run
    start_at = datetime.now(timezone.utc) - timedelta(hours=1)
    end_at = datetime.now(timezone.utc)
    other_project = Project(name="other learning project", config={})
    db_session.add(other_project)
    await db_session.flush()
    other_goal = OrchestrationGoal(project_id=other_project.id, objective="other learning goal")
    db_session.add(other_goal)
    await _response(db_session, goal, run, uuid.UUID(int=0), 1, now=end_at)
    await db_session.commit()
    params = {"start_at": start_at.isoformat(), "end_at": end_at.isoformat()}

    missing_project = await learning_client.get(_report_url(uuid.uuid4()), params=params)
    wrong_goal = await learning_client.get(
        _report_url(goal.project_id), params={**params, "goal_id": str(other_goal.id)}
    )
    half_open = await learning_client.get(_report_url(goal.project_id), params=params)

    assert missing_project.status_code == 404
    assert missing_project.json()["detail"] == "Project not found"
    assert (wrong_goal.status_code, wrong_goal.json()["detail"]["code"]) == (
        404, "conversation_learning_goal_not_found",
    )
    assert half_open.status_code == 200
    _assert_zero_report(half_open.json(), start_at, end_at)


async def test_learning_report_normalizes_offset_timestamps_and_never_commits(
    learning_client, conversation_goal_run, db_session, monkeypatch
):
    goal, _ = conversation_goal_run
    await db_session.commit()
    start_at = datetime(2026, 9, 1, 3, tzinfo=timezone(timedelta(hours=3)))
    end_at = datetime(2026, 9, 2, 3, tzinfo=timezone(timedelta(hours=3)))

    async def forbidden_commit(*_args, **_kwargs):
        raise AssertionError("learning report must not commit")

    monkeypatch.setattr(AsyncSession, "commit", forbidden_commit)
    result = await learning_client.get(
        _report_url(goal.project_id), params={"start_at": start_at.isoformat(), "end_at": end_at.isoformat()}
    )

    assert result.status_code == 200
    body = result.json()
    assert datetime.fromisoformat(body["window"]["start_at"].replace("Z", "+00:00")) == start_at.astimezone(timezone.utc)
    assert datetime.fromisoformat(body["window"]["end_at"].replace("Z", "+00:00")) == end_at.astimezone(timezone.utc)
