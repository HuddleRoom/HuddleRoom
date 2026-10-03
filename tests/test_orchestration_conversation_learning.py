"""Service contract for immutable, measurement-only conversation feedback."""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import FrozenInstanceError, asdict, is_dataclass
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select

from huddleroom.config import settings
from huddleroom.models import (
    Artifact,
    ConversationFeedback,
    EventLog,
    OrchestrationAction,
    OrchestrationDecision,
    OrchestrationEvidence,
    Task,
)
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_conversation import (
    ConversationInvestigation,
    ConversationInvestigationReservation,
    ConversationMessage,
    ConversationReservation,
    ConversationResponse,
    conversation_feedback_id,
    conversation_response_id,
)
from huddleroom.models.orchestration_memory import OrchestrationMemorySection
from huddleroom.models.orchestration_steering import (
    OrchestrationSteeringProposal,
    OrchestrationSteeringRequest,
    OrchestrationSteeringResultLink,
    OrchestrationSteeringState,
    OrchestrationSteeringTransition,
)
from huddleroom.models.project import Project
from huddleroom.models.user import User
from huddleroom.services.orchestration_conversation_learning import (
    ConversationFeedbackInput,
    ConversationLearningError,
    ConversationLearningReport,
    ConversationLearningWindow,
    OrchestrationConversationLearningService,
)


REASONS = (
    "unanswered",
    "incorrect",
    "missing_context",
    "stale_context",
    "unclear",
    "too_limited",
    "other",
)


async def _response(
    db,
    goal,
    run,
    actor,
    sequence,
    *,
    status="completed",
    answer="A completed answer.",
):
    now = datetime.now(timezone.utc)
    message = ConversationMessage(
        goal_id=goal.id,
        actor_id=actor.id,
        client_request_id=uuid.uuid4(),
        sequence=sequence,
        content="question",
    )
    db.add(message)
    await db.flush()
    started_at = deadline_at = finished_at = None
    if status in {"running", "completed", "interrupted_unknown"}:
        started_at = deadline_at = now
    if status in {"completed", "failed", "interrupted_unknown"}:
        finished_at = now
    response = ConversationResponse(
        id=conversation_response_id(message.id),
        message_id=message.id,
        run_id=run.id,
        status=status,
        dossier={},
        context_manifest={},
        context_version="feedback-test-v1",
        provider_request_id=f"rally-chat:learning-{sequence}",
        answer=answer,
        started_at=started_at,
        deadline_at=deadline_at,
        finished_at=finished_at,
    )
    db.add(response)
    await db.flush()
    return message, response


async def _other_user(db):
    user = User(email=f"feedback-{uuid.uuid4()}@example.test", hashed_password="test")
    db.add(user)
    await db.flush()
    return user


async def _feedback_count(db):
    return await db.scalar(select(func.count()).select_from(ConversationFeedback))


def _force_feedback_collision(monkeypatch, first, second, feedback_id):
    """Make both pre-checks miss, then let the second insert lose to the first."""
    missing = asyncio.Barrier(2)
    first_committed = asyncio.Event()
    feedback_reads = []
    for session in (first, second):
        get = session.get

        async def synchronized_get(entity, ident, *args, _get=get, **kwargs):
            existing = await _get(entity, ident, *args, **kwargs)
            if entity is ConversationFeedback and ident == feedback_id:
                feedback_reads.append(existing)
                if existing is None:
                    await missing.wait()
            return existing

        monkeypatch.setattr(session, "get", synchronized_get)

    begin_nested = second.begin_nested

    @asynccontextmanager
    async def second_after_first_commit():
        await first_committed.wait()
        async with begin_nested():
            yield

    monkeypatch.setattr(second, "begin_nested", second_after_first_commit)
    return first_committed, feedback_reads


async def _expect_error(awaitable, code, status_code):
    with pytest.raises(ConversationLearningError) as captured:
        await awaitable
    assert captured.value.code == code
    assert captured.value.status_code == status_code
    assert captured.value.message


async def _durable_counts(db):
    models = (
        EventLog,
        OrchestrationAction,
        OrchestrationDecision,
        Task,
        OrchestrationEvidence,
        OrchestrationMemorySection,
        Artifact,
        ConversationMessage,
        ConversationResponse,
        ConversationReservation,
        ConversationInvestigation,
        ConversationInvestigationReservation,
        OrchestrationSteeringProposal,
        OrchestrationSteeringRequest,
        OrchestrationSteeringTransition,
        OrchestrationSteeringResultLink,
        OrchestrationSteeringState,
        OrchestrationGoal,
        OrchestrationRun,
    )
    return {
        model.__tablename__: await db.scalar(select(func.count()).select_from(model))
        for model in models
    }


async def _durable_rows(db):
    models = (
        EventLog,
        OrchestrationAction,
        OrchestrationDecision,
        Task,
        OrchestrationEvidence,
        OrchestrationMemorySection,
        Artifact,
        ConversationMessage,
        ConversationResponse,
        ConversationReservation,
        ConversationInvestigation,
        ConversationInvestigationReservation,
        OrchestrationSteeringProposal,
        OrchestrationSteeringRequest,
        OrchestrationSteeringTransition,
        OrchestrationSteeringResultLink,
        OrchestrationSteeringState,
        OrchestrationGoal,
        OrchestrationRun,
    )
    snapshots = {}
    for model in models:
        primary_key = tuple(model.__table__.primary_key.columns)
        rows = await db.execute(select(*model.__table__.columns).order_by(*primary_key))
        snapshots[model.__tablename__] = tuple(tuple(row) for row in rows.all())
    return snapshots


def _service():
    return OrchestrationConversationLearningService()


def test_learning_public_input_and_error_contracts_are_stable():
    feedback = ConversationFeedbackInput("helpful", None)
    error = ConversationLearningError("test_code", 418, "safe message")

    assert is_dataclass(feedback)
    with pytest.raises(FrozenInstanceError):
        feedback.rating = "not_helpful"
    assert (error.code, error.status_code, error.message) == ("test_code", 418, "safe message")


@pytest.mark.asyncio
async def test_record_feedback_creates_helpful_row_with_null_reason(
    db_session, conversation_goal_run, test_user
):
    goal, run = conversation_goal_run
    message, response = await _response(db_session, goal, run, test_user, 1)

    recorded = await _service().record_feedback(
        db_session,
        goal.project_id,
        goal.id,
        message.actor_id,
        response.id,
        ConversationFeedbackInput("helpful", None),
    )

    assert (recorded.id, recorded.response_id, recorded.actor_id) == (
        conversation_feedback_id(response.id, message.actor_id),
        response.id,
        message.actor_id,
    )
    assert (recorded.rating, recorded.reason) == ("helpful", None)
    assert await _feedback_count(db_session) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", REASONS)
async def test_record_feedback_creates_each_allowed_not_helpful_reason(
    db_session, conversation_goal_run, test_user, reason
):
    goal, run = conversation_goal_run
    message, response = await _response(db_session, goal, run, test_user, 1)

    recorded = await _service().record_feedback(
        db_session,
        goal.project_id,
        goal.id,
        message.actor_id,
        response.id,
        ConversationFeedbackInput("not_helpful", reason),
    )

    assert (recorded.rating, recorded.reason) == ("not_helpful", reason)
    assert await _feedback_count(db_session) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rating", "reason"),
    (
        ("helpful", "unclear"),
        ("helpful", ""),
        ("not_helpful", None),
        ("not_helpful", ""),
        ("not_helpful", "unknown"),
        ("not_helpful", []),
        ("unknown", None),
        ("unknown", "unclear"),
        (None, None),
        (None, "unclear"),
    ),
)
async def test_record_feedback_rejects_invalid_shapes_before_writing(
    db_session, conversation_goal_run, test_user, rating, reason
):
    goal, run = conversation_goal_run
    message, response = await _response(db_session, goal, run, test_user, 1)

    await _expect_error(
        _service().record_feedback(
            db_session,
            goal.project_id,
            goal.id,
            message.actor_id,
            response.id,
            ConversationFeedbackInput(rating, reason),
        ),
        "conversation_feedback_ineligible",
        409,
    )

    assert await _feedback_count(db_session) == 0


@pytest.mark.asyncio
async def test_record_feedback_replays_an_exact_existing_body_without_another_row(
    db_session, conversation_goal_run, test_user
):
    goal, run = conversation_goal_run
    message, response = await _response(db_session, goal, run, test_user, 1)
    feedback = ConversationFeedbackInput("not_helpful", "missing_context")

    recorded = await _service().record_feedback(
        db_session, goal.project_id, goal.id, message.actor_id, response.id, feedback
    )
    replayed = await _service().record_feedback(
        db_session, goal.project_id, goal.id, message.actor_id, response.id, feedback
    )

    assert replayed.id == recorded.id == conversation_feedback_id(response.id, message.actor_id)
    assert await _feedback_count(db_session) == 1


@pytest.mark.asyncio
async def test_record_feedback_keeps_original_body_when_replay_conflicts(
    db_session, conversation_goal_run, test_user
):
    goal, run = conversation_goal_run
    message, response = await _response(db_session, goal, run, test_user, 1)
    original = ConversationFeedbackInput("helpful", None)
    await _service().record_feedback(
        db_session, goal.project_id, goal.id, message.actor_id, response.id, original
    )

    await _expect_error(
        _service().record_feedback(
            db_session,
            goal.project_id,
            goal.id,
            message.actor_id,
            response.id,
            ConversationFeedbackInput("not_helpful", "incorrect"),
        ),
        "conversation_feedback_already_recorded",
        409,
    )

    saved = await db_session.get(
        ConversationFeedback, conversation_feedback_id(response.id, message.actor_id)
    )
    assert (saved.rating, saved.reason) == ("helpful", None)
    assert await _feedback_count(db_session) == 1


@pytest.mark.asyncio
async def test_record_feedback_accepts_only_the_requested_response_project_goal_lineage(
    db_session, conversation_goal_run, test_user
):
    goal, run = conversation_goal_run
    message, response = await _response(db_session, goal, run, test_user, 1)
    other_project = Project(name="Other project", config={})
    db_session.add(other_project)
    await db_session.flush()
    other_goal = OrchestrationGoal(project_id=other_project.id, objective="Other goal")
    db_session.add(other_goal)
    await db_session.flush()

    recorded = await _service().record_feedback(
        db_session,
        goal.project_id,
        goal.id,
        message.actor_id,
        response.id,
        ConversationFeedbackInput("helpful", None),
    )
    assert recorded.response_id == response.id

    for project_id, goal_id, response_id in (
        (goal.project_id, goal.id, uuid.uuid4()),
        (other_project.id, goal.id, response.id),
        (goal.project_id, other_goal.id, response.id),
    ):
        await _expect_error(
            _service().record_feedback(
                db_session,
                project_id,
                goal_id,
                message.actor_id,
                response_id,
                ConversationFeedbackInput("helpful", None),
            ),
            "conversation_feedback_not_found",
            404,
        )


@pytest.mark.asyncio
async def test_record_feedback_rejects_a_non_owner_without_writing(
    db_session, conversation_goal_run, test_user
):
    goal, run = conversation_goal_run
    message, response = await _response(db_session, goal, run, test_user, 1)
    other = await _other_user(db_session)

    await _expect_error(
        _service().record_feedback(
            db_session,
            goal.project_id,
            goal.id,
            other.id,
            response.id,
            ConversationFeedbackInput("helpful", None),
        ),
        "conversation_feedback_forbidden",
        403,
    )

    assert await _feedback_count(db_session) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ("pending", "running", "failed", "interrupted_unknown"))
async def test_record_feedback_rejects_non_completed_responses(
    db_session, conversation_goal_run, test_user, status
):
    goal, run = conversation_goal_run
    message, response = await _response(db_session, goal, run, test_user, 1, status=status)

    await _expect_error(
        _service().record_feedback(
            db_session,
            goal.project_id,
            goal.id,
            message.actor_id,
            response.id,
            ConversationFeedbackInput("helpful", None),
        ),
        "conversation_feedback_ineligible",
        409,
    )

    assert await _feedback_count(db_session) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", (None, "", " \n\t "))
async def test_record_feedback_rejects_completed_blank_answers(
    db_session, conversation_goal_run, test_user, answer
):
    goal, run = conversation_goal_run
    message, response = await _response(db_session, goal, run, test_user, 1, answer=answer)

    await _expect_error(
        _service().record_feedback(
            db_session,
            goal.project_id,
            goal.id,
            message.actor_id,
            response.id,
            ConversationFeedbackInput("helpful", None),
        ),
        "conversation_feedback_ineligible",
        409,
    )


@pytest.mark.asyncio
async def test_historical_completed_answer_ignores_current_runtime_flags(
    db_session, conversation_goal_run, test_user, monkeypatch
):
    goal, run = conversation_goal_run
    message, response = await _response(db_session, goal, run, test_user, 1)
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", 0)
    monkeypatch.setattr(settings, "orchestration_conversation_investigation_enabled", False)
    monkeypatch.setattr(settings, "orchestration_conversation_steering_enabled", False)

    recorded = await _service().record_feedback(
        db_session,
        goal.project_id,
        goal.id,
        message.actor_id,
        response.id,
        ConversationFeedbackInput("helpful", None),
    )

    assert recorded.response_id == response.id


@pytest.mark.asyncio
async def test_record_feedback_changes_only_feedback_and_never_runtime_state(
    db_session, conversation_goal_run, test_user
):
    goal, run = conversation_goal_run
    message, response = await _response(db_session, goal, run, test_user, 1)
    before_counts = await _durable_counts(db_session)
    before_rows = await _durable_rows(db_session)

    created = await _service().record_feedback(
        db_session,
        goal.project_id,
        goal.id,
        message.actor_id,
        response.id,
        ConversationFeedbackInput("helpful", None),
    )
    replayed = await _service().record_feedback(
        db_session,
        goal.project_id,
        goal.id,
        message.actor_id,
        response.id,
        ConversationFeedbackInput("helpful", None),
    )

    assert created.id == replayed.id
    assert await _feedback_count(db_session) == 1
    assert await _durable_counts(db_session) == before_counts
    assert await _durable_rows(db_session) == before_rows


@pytest.mark.asyncio
async def test_record_feedback_never_calls_runtime_provider_event_or_scheduler_seams(
    db_session, conversation_goal_run, test_user, monkeypatch
):
    from huddleroom.services import event_bus
    from huddleroom.services.orchestration_conversation_investigation import (
        ConversationInvestigationService,
        conversation_allowance_used,
    )
    from huddleroom.services.orchestration_conversation_service import OrchestrationConversationService
    from huddleroom.services.orchestration_service import OrchestrationService
    from huddleroom.services.orchestration_supervision_scheduler import OrchestrationSupervisionScheduler

    goal, run = conversation_goal_run
    message, response = await _response(db_session, goal, run, test_user, 1)

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("record_feedback must remain measurement-only")

    monkeypatch.setattr(event_bus, "emit_event", forbidden)
    monkeypatch.setattr(event_bus, "emit_event_once", forbidden)
    monkeypatch.setattr(ConversationInvestigationService, "recover_goal", forbidden)
    monkeypatch.setattr(ConversationInvestigationService, "recover_all", forbidden)
    monkeypatch.setattr(OrchestrationConversationService, "recover_goal", forbidden)
    monkeypatch.setattr(OrchestrationConversationService, "recover_all", forbidden)
    monkeypatch.setattr(OrchestrationService, "tick", forbidden)
    monkeypatch.setattr(OrchestrationService, "_lock_goal_for_baseline_transition", forbidden)
    monkeypatch.setattr(OrchestrationSupervisionScheduler, "record_event", forbidden)
    monkeypatch.setattr(OrchestrationSupervisionScheduler, "_tick", forbidden)
    monkeypatch.setattr(
        "huddleroom.services.orchestration_conversation_investigation.conversation_allowance_used",
        forbidden,
    )

    recorded = await _service().record_feedback(
        db_session,
        goal.project_id,
        goal.id,
        message.actor_id,
        response.id,
        ConversationFeedbackInput("helpful", None),
    )

    assert recorded.response_id == response.id
    assert conversation_allowance_used


@pytest.mark.asyncio
async def test_concurrent_same_body_submissions_converge_on_one_feedback_row(
    test_engine, conversation_goal_run, test_user, db_session, concurrent_sessions, monkeypatch
):
    """Independent sessions exercise the unique row contract without sleeps."""
    goal, run = conversation_goal_run
    message, response = await _response(db_session, goal, run, test_user, 1)
    await db_session.commit()
    first, second = concurrent_sessions
    feedback = ConversationFeedbackInput("not_helpful", "unclear")
    feedback_id = conversation_feedback_id(response.id, message.actor_id)
    first_committed, feedback_reads = _force_feedback_collision(
        monkeypatch, first, second, feedback_id
    )

    async def submit_first():
        recorded = await _service().record_feedback(
            first, goal.project_id, goal.id, message.actor_id, response.id, feedback
        )
        await first.commit()
        first_committed.set()
        return recorded

    async def submit_second():
        recorded = await _service().record_feedback(
            second, goal.project_id, goal.id, message.actor_id, response.id, feedback
        )
        await second.commit()
        return recorded

    recorded, replayed = await asyncio.gather(submit_first(), submit_second())
    assert len(feedback_reads) == 3
    assert sum(row is None for row in feedback_reads) == 2
    async with first.begin():
        assert await _feedback_count(first) == 1
    assert recorded.id == replayed.id == feedback_id


@pytest.mark.asyncio
async def test_concurrent_different_body_submissions_keep_one_immutable_winner(
    conversation_goal_run, test_user, db_session, concurrent_sessions, monkeypatch
):
    """The loser must surface the stable immutable-conflict error, never overwrite."""
    goal, run = conversation_goal_run
    message, response = await _response(db_session, goal, run, test_user, 1)
    await db_session.commit()
    first, second = concurrent_sessions
    feedback_id = conversation_feedback_id(response.id, message.actor_id)
    first_committed, feedback_reads = _force_feedback_collision(
        monkeypatch, first, second, feedback_id
    )

    async def submit_first():
        recorded = await _service().record_feedback(
            first,
            goal.project_id,
            goal.id,
            message.actor_id,
            response.id,
            ConversationFeedbackInput("helpful", None),
        )
        await first.commit()
        first_committed.set()
        return ("recorded", recorded)

    async def submit_second():
        with pytest.raises(ConversationLearningError) as captured:
            await _service().record_feedback(
                second,
                goal.project_id,
                goal.id,
                message.actor_id,
                response.id,
                ConversationFeedbackInput("not_helpful", "unclear"),
            )
        marker = User(email=f"feedback-loser-{uuid.uuid4()}@example.test", hashed_password="test")
        second.add(marker)
        await second.commit()
        return ("error", captured.value, marker.id)

    first_result, second_result = await asyncio.gather(submit_first(), submit_second())

    assert first_result[0] == "recorded"
    assert second_result[0] == "error"
    assert len(feedback_reads) == 3
    assert sum(row is None for row in feedback_reads) == 2
    error = second_result[1]
    assert (error.code, error.status_code) == ("conversation_feedback_already_recorded", 409)
    async with first.begin():
        stored = await first.get(ConversationFeedback, feedback_id)
        assert await first.get(User, second_result[2]) is not None
    winner = first_result[1]
    assert (stored.rating, stored.reason) == (winner.rating, winner.reason)


def _utc(day, hour=0, minute=0):
    return datetime(2026, 9, day, hour, minute, tzinfo=timezone.utc)


async def _turn(
    db,
    goal,
    run,
    actor,
    sequence,
    created_at,
    *,
    status="completed",
    answer="answer",
    started_at=None,
    finished_at=None,
    context_manifest=None,
):
    message = ConversationMessage(
        goal_id=goal.id,
        actor_id=actor.id,
        client_request_id=uuid.uuid4(),
        sequence=sequence,
        content=f"private-message-{sequence}",
        created_at=created_at,
    )
    db.add(message)
    await db.flush()
    response = ConversationResponse(
        id=conversation_response_id(message.id),
        message_id=message.id,
        run_id=run.id,
        status=status,
        dossier={},
        context_manifest=context_manifest or {},
        context_version=f"learning-window-{sequence}",
        provider_request_id=f"learning-window-{message.id}",
        answer=answer,
        started_at=started_at,
        deadline_at=started_at,
        finished_at=finished_at,
        created_at=created_at,
    )
    db.add(response)
    await db.flush()
    return message, response


async def _fixture_feedback(db, response, actor, rating, reason, created_at):
    db.add(
        ConversationFeedback(
            id=conversation_feedback_id(response.id, actor.id),
            response_id=response.id,
            actor_id=actor.id,
            rating=rating,
            reason=reason,
            created_at=created_at,
        )
    )
    await db.flush()


async def _fixture_investigation(
    db,
    response,
    goal,
    actor,
    suffix,
    created_at,
    *,
    status,
    attempts,
    repairs,
    retries,
    tokens,
    started_at=None,
    finished_at=None,
    input_manifest=None,
):
    investigation = ConversationInvestigation(
        id=uuid.uuid4(),
        response_id=response.id,
        goal_id=goal.id,
        actor_id=actor.id,
        context_version=f"investigation-{suffix}",
        status=status,
        objective=f"private-investigation-objective-{suffix}",
        scope=[],
        input_manifest=input_manifest or {},
        provider_identity=f"private-investigation-reference-{suffix}",
        provider_request_id=f"investigation-request-{suffix}" if attempts else None,
        attempt_count=attempts,
        repair_count=repairs,
        retry_count=retries,
        accumulated_tokens=tokens,
        report={"private_report": f"private-investigation-report-{suffix}"},
        started_at=started_at,
        deadline_at=started_at,
        finished_at=finished_at,
        created_at=created_at,
    )
    db.add(investigation)
    await db.flush()
    return investigation


async def _fixture_proposal(db, response, goal, actor, suffix, created_at, *, status="proposed"):
    proposal = OrchestrationSteeringProposal(
        id=uuid.uuid4(),
        response_id=response.id,
        goal_id=goal.id,
        actor_id=actor.id,
        status=status,
        draft={"directive": f"private-proposal-{suffix}"},
        dismissed_at=_utc(20) if status == "dismissed" else None,
        promoted_request_id=uuid.uuid4() if status == "promoted" else None,
        created_at=created_at,
    )
    db.add(proposal)
    await db.flush()
    return proposal


async def _fixture_request(
    db, goal, run, actor, sequence, submitted_at, *, source_proposal_id=None
):
    request = OrchestrationSteeringRequest(
        id=uuid.uuid4(),
        goal_id=goal.id,
        actor_id=actor.id,
        client_request_id=uuid.uuid4(),
        sequence=sequence,
        submitted_run_id=run.id,
        directive=f"private-directive-{sequence}",
        target_type="goal",
        target_id=str(goal.id),
        scope="goal",
        lifetime="future_runs",
        impact_summary=f"private-impact-{sequence}",
        source_proposal_id=source_proposal_id,
        status="applied",
        reason_code="applied",
        contract_version="private-contract",
        plan_version="private-plan",
        submitted_at=submitted_at,
    )
    db.add(request)
    await db.flush()
    return request


async def _fixture_transition(db, request, sequence, status, reason, created_at):
    db.add(
        OrchestrationSteeringTransition(
            id=uuid.uuid4(),
            request_id=request.id,
            sequence=sequence,
            from_status=None,
            to_status=status,
            reason_code=reason,
            actor="private-transition-actor",
            created_at=created_at,
        )
    )
    await db.flush()


async def _fixture_result_link(db, request, run, suffix, created_at):
    decision = OrchestrationDecision(
        id=uuid.uuid4(),
        run_id=run.id,
        decision_type="private-decision",
        validator_status="accepted",
        created_at=created_at,
    )
    db.add(decision)
    await db.flush()
    action = OrchestrationAction(
        id=uuid.uuid4(),
        run_id=run.id,
        decision_id=decision.id,
        idempotency_key=f"learning-result-{suffix}",
        action_type="private-action",
        status="completed",
        created_at=created_at,
    )
    db.add(action)
    await db.flush()
    db.add(
        OrchestrationSteeringResultLink(
            id=uuid.uuid4(),
            request_id=request.id,
            decision_id=decision.id,
            action_id=action.id,
            created_at=created_at,
        )
    )
    await db.flush()


def _zero_report(start_at, end_at, generated_at, *, goal_filtered=False):
    return {
        "window": {
            "start_at": start_at,
            "end_at": end_at,
            "generated_at": generated_at,
            "goal_filtered": goal_filtered,
        },
        "sample": {"goals": 0, "operators": 0, "questions": 0},
        "chat": {
            "status_counts": {
                "pending": 0,
                "running": 0,
                "completed": 0,
                "failed": 0,
                "interrupted_unknown": 0,
            },
            "answered": 0,
            "terminal_without_answer": 0,
            "latency_seconds": {"count": 0, "invalid": 0, "median": None, "p95": None},
        },
        "investigations": {
            "triggered": 0,
            "accounted_terminal": 0,
            "status_counts": {
                "pending": 0,
                "running": 0,
                "completed": 0,
                "limited": 0,
                "failed": 0,
                "cancelled": 0,
                "unavailable": 0,
                "interrupted_unknown": 0,
            },
            "attempts": 0,
            "repairs": 0,
            "retries": 0,
            "accumulated_tokens": 0,
            "latency_seconds": {"count": 0, "invalid": 0, "median": None, "p95": None},
        },
        "context_limits": {
            "turns_truncated": 0,
            "turns_with_omissions": 0,
            "omitted_records": 0,
            "truncated_source_counts": {
                "goal": 0,
                "run": 0,
                "accepted_plan": 0,
                "decisions": 0,
                "actions": 0,
                "gates": 0,
                "evidence": 0,
                "processes": 0,
                "warnings": 0,
                "memory": 0,
                "artifact": 0,
                "agents": 0,
                "prior_turns": 0,
            },
            "investigation_omission_status_counts": {
                "restricted": 0,
                "unsafe": 0,
                "binary": 0,
                "too_large": 0,
                "changed": 0,
                "omitted_by_limit": 0,
            },
        },
        "steering": {
            "proposal_status_counts": {"proposed": 0, "dismissed": 0, "promoted": 0},
            "request_status_counts": {
                "pending": 0,
                "being_considered": 0,
                "applied": 0,
                "deferred": 0,
                "rejected": 0,
                "superseded": 0,
                "needs_clarification": 0,
                "withdrawn": 0,
            },
            "request_reason_counts": {
                "submitted": 0,
                "considering": 0,
                "run_changed": 0,
                "steering_ineligible": 0,
                "target_already_started": 0,
                "supersedes_required": 0,
                "invalid_supersedes_request": 0,
                "superseded": 0,
                "applied": 0,
                "withdrawn": 0,
            },
            "direct_requests": 0,
            "proposal_derived_requests": 0,
            "requests_with_result_actions": 0,
            "result_links": 0,
            "submit_to_considered_seconds": {
                "count": 0,
                "invalid": 0,
                "median": None,
                "p95": None,
            },
            "submit_to_finished_seconds": {
                "count": 0,
                "invalid": 0,
                "median": None,
                "p95": None,
            },
        },
        "operator_feedback": {
            "rated": 0,
            "unrated_answered": 0,
            "helpful": 0,
            "not_helpful": 0,
            "not_helpful_reason_counts": {
                "unanswered": 0,
                "incorrect": 0,
                "missing_context": 0,
                "stale_context": 0,
                "unclear": 0,
                "too_limited": 0,
                "other": 0,
            },
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "window",
    (
        lambda: ConversationLearningWindow(datetime(2026, 9, 1), _utc(2)),
        lambda: ConversationLearningWindow(_utc(1), datetime(2026, 9, 2)),
    ),
)
async def test_summarize_rejects_naive_window_endpoints_before_aggregation(
    db_session, conversation_goal_run, window
):
    goal, _ = conversation_goal_run

    await _expect_error(
        _service().summarize(db_session, goal.project_id, window()),
        "conversation_learning_invalid_window",
        422,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "start_at,end_at",
    (
        (_utc(2), _utc(2)),
        (_utc(3), _utc(2)),
        (_utc(1), _utc(1) + timedelta(days=90, seconds=1)),
    ),
)
async def test_summarize_rejects_empty_reversed_and_oversize_windows(
    db_session, conversation_goal_run, start_at, end_at
):
    goal, _ = conversation_goal_run

    await _expect_error(
        _service().summarize(
            db_session, goal.project_id, ConversationLearningWindow(start_at, end_at)
        ),
        "conversation_learning_invalid_window",
        422,
    )


@pytest.mark.asyncio
async def test_summarize_normalizes_offsets_and_accepts_an_exact_ninety_day_window(
    db_session, conversation_goal_run
):
    goal, _ = conversation_goal_run
    offset = timezone(timedelta(hours=3))
    start_at = datetime(2026, 9, 1, 3, tzinfo=offset)
    end_at = start_at + timedelta(days=90)

    report = await _service().summarize(
        db_session, goal.project_id, ConversationLearningWindow(start_at, end_at)
    )

    assert report.window["start_at"] == datetime(2026, 9, 1, tzinfo=timezone.utc)
    assert report.window["end_at"] == datetime(2026, 11, 30, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_summarize_rejects_a_goal_from_another_project_before_aggregation(
    db_session, conversation_goal_run
):
    goal, _ = conversation_goal_run
    other_project = Project(name="private-other-project", config={})
    db_session.add(other_project)
    await db_session.flush()
    other_goal = OrchestrationGoal(project_id=other_project.id, objective="private-other-goal")
    db_session.add(other_goal)
    await db_session.flush()

    await _expect_error(
        _service().summarize(
            db_session,
            goal.project_id,
            ConversationLearningWindow(_utc(1), _utc(2), other_goal.id),
        ),
        "conversation_learning_goal_not_found",
        404,
    )


@pytest.mark.asyncio
async def test_summarize_returns_the_exact_zero_filled_report_for_an_empty_window(
    db_session, conversation_goal_run
):
    goal, _ = conversation_goal_run
    start_at, end_at = _utc(1), _utc(2)

    report = await _service().summarize(
        db_session, goal.project_id, ConversationLearningWindow(start_at, end_at)
    )

    assert isinstance(report, ConversationLearningReport)
    generated_at = report.window["generated_at"]
    assert generated_at.tzinfo is not None
    assert generated_at.utcoffset() == timedelta(0)
    assert asdict(report) == _zero_report(start_at, end_at, generated_at)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ("failed", "interrupted_unknown"))
async def test_summarize_does_not_call_terminal_nonblank_answers_missing(
    db_session, conversation_goal_run, test_user, status
):
    goal, run = conversation_goal_run
    start_at, end_at = _utc(1), _utc(2)
    await _turn(
        db_session,
        goal,
        run,
        test_user,
        1,
        _utc(1, 1),
        status=status,
        answer="persisted but not completed",
        started_at=_utc(1, 1) if status == "interrupted_unknown" else None,
        finished_at=_utc(1, 2),
    )

    report = await _service().summarize(
        db_session, goal.project_id, ConversationLearningWindow(start_at, end_at)
    )

    assert report.chat["terminal_without_answer"] == 0


@pytest.mark.asyncio
async def test_summarize_ignores_unknown_legacy_transition_reasons(
    db_session, conversation_goal_run, test_user
):
    goal, run = conversation_goal_run
    start_at, end_at = _utc(1), _utc(2)
    request = await _fixture_request(db_session, goal, run, test_user, 1, _utc(1, 1))
    await _fixture_transition(db_session, request, 1, "applied", "legacy_reason", _utc(1, 2))

    report = await _service().summarize(
        db_session, goal.project_id, ConversationLearningWindow(start_at, end_at)
    )

    assert report.steering["request_status_counts"]["applied"] == 1
    assert report.steering["request_reason_counts"] == {
        reason: 0 for reason in report.steering["request_reason_counts"]
    }


@pytest.mark.asyncio
async def test_summarize_reconciles_a_bounded_as_of_disclosure_safe_mixed_fixture(
    db_session, conversation_goal_run, test_user
):
    goal, run = conversation_goal_run
    start_at, end_at = _utc(1), datetime(2026, 10, 1, tzinfo=timezone.utc)
    other_actor = await _other_user(db_session)
    other_goal = OrchestrationGoal(project_id=goal.project_id, objective="private-second-goal")
    other_project = Project(name="private-isolated-project", config={})
    db_session.add_all((other_goal, other_project))
    await db_session.flush()
    other_run = OrchestrationRun(goal_id=other_goal.id, started_at=start_at)
    isolated_goal = OrchestrationGoal(project_id=other_project.id, objective="private-isolated-goal")
    db_session.add_all((other_run, isolated_goal))
    await db_session.flush()
    isolated_run = OrchestrationRun(goal_id=isolated_goal.id, started_at=start_at)
    db_session.add(isolated_run)
    await db_session.flush()

    manifest = {
        "truncated": True,
        "sources": [
            {"source": "goal", "truncated": True, "omitted": 2, "references": ["private-reference"]},
            {"source": "run", "truncated": False, "omitted": 1},
            {"source": "unknown", "truncated": True, "omitted": 99},
            "malformed-source",
        ],
    }
    _, first = await _turn(
        db_session,
        goal,
        run,
        test_user,
        1,
        _utc(2),
        answer="private-answer-helpful",
        started_at=_utc(2, 1),
        finished_at=_utc(2, 2),
        context_manifest=manifest,
    )
    _, blank = await _turn(
        db_session, goal, run, test_user, 2, _utc(3), answer=" \n\t ", started_at=_utc(3, 1), finished_at=_utc(3, 1, 30)
    )
    _, failed = await _turn(
        db_session, goal, run, test_user, 3, _utc(4), status="failed", answer=None, started_at=_utc(4, 2), finished_at=_utc(4, 1)
    )
    _, interrupted = await _turn(
        db_session, goal, run, test_user, 4, _utc(5), status="interrupted_unknown", answer=None, started_at=_utc(5, 1), finished_at=_utc(5, 2)
    )
    negative_turns = []
    for sequence, reason in enumerate(REASONS, 5):
        _, response = await _turn(
            db_session,
            goal,
            run,
            test_user,
            sequence,
            _utc(sequence + 1),
            answer=f"private-answer-{reason}",
            started_at=_utc(sequence + 1, 1),
            finished_at=_utc(sequence + 1, 1, 1),
        )
        negative_turns.append((response, reason))
    _, unrated = await _turn(
        db_session, goal, run, test_user, 12, _utc(13), answer="private-unrated-answer", started_at=_utc(13, 1), finished_at=_utc(13, 2)
    )
    _, active = await _turn(
        db_session,
        goal,
        run,
        test_user,
        13,
        _utc(14),
        answer="private-late-answer",
        started_at=_utc(30, 12),
        finished_at=datetime(2026, 10, 2, tzinfo=timezone.utc),
    )
    _, late_pending = await _turn(
        db_session,
        goal,
        run,
        test_user,
        14,
        _utc(15),
        answer="private-pending-answer",
        started_at=datetime(2026, 10, 2, tzinfo=timezone.utc),
        finished_at=datetime(2026, 10, 3, tzinfo=timezone.utc),
    )
    await _turn(
        db_session, goal, run, test_user, 15, datetime(2026, 8, 31, tzinfo=timezone.utc), answer="private-before-window", started_at=_utc(1), finished_at=_utc(1, 1)
    )
    await _turn(
        db_session, goal, run, test_user, 16, end_at, answer="private-at-end", started_at=end_at, finished_at=datetime(2026, 10, 2, tzinfo=timezone.utc)
    )
    await _turn(
        db_session, other_goal, other_run, other_actor, 1, _utc(16), status="pending", answer=None
    )
    await _turn(
        db_session, isolated_goal, isolated_run, other_actor, 1, _utc(16), answer="private-isolated-answer", started_at=_utc(16, 1), finished_at=_utc(16, 2)
    )

    await _fixture_feedback(db_session, first, test_user, "helpful", None, _utc(3))
    for response, reason in negative_turns:
        await _fixture_feedback(db_session, response, test_user, "not_helpful", reason, _utc(20))
    await _fixture_feedback(db_session, unrated, test_user, "not_helpful", "incorrect", datetime(2026, 10, 2, tzinfo=timezone.utc))

    omission_manifest = {
        "sources": ["malformed"],
        "omissions": [
            {"status": status, "reference": f"private-investigation-reference-{status}"}
            for status in ("restricted", "unsafe", "binary", "too_large", "changed", "omitted_by_limit")
        ] + [{"status": "unknown"}, "malformed-omission"],
    }
    await _fixture_investigation(
        db_session, first, goal, test_user, "completed", _utc(10), status="completed", attempts=2, repairs=1, retries=0, tokens=100, started_at=_utc(10, 1), finished_at=_utc(10, 3), input_manifest=omission_manifest
    )
    await _fixture_investigation(
        db_session, blank, goal, test_user, "limited", _utc(11), status="limited", attempts=0, repairs=0, retries=0, tokens=0, finished_at=_utc(11, 3)
    )
    await _fixture_investigation(
        db_session, failed, goal, test_user, "failed", _utc(12), status="failed", attempts=1, repairs=0, retries=1, tokens=50, started_at=_utc(12, 3), finished_at=_utc(12, 2)
    )
    await _fixture_investigation(
        db_session, interrupted, goal, test_user, "active", _utc(13), status="completed", attempts=2, repairs=1, retries=0, tokens=10_000, started_at=_utc(20), finished_at=datetime(2026, 10, 2, tzinfo=timezone.utc)
    )
    await _fixture_investigation(
        db_session, active, goal, test_user, "at-end", end_at, status="completed", attempts=1, repairs=0, retries=0, tokens=1, started_at=end_at, finished_at=datetime(2026, 10, 2, tzinfo=timezone.utc)
    )

    proposed = await _fixture_proposal(db_session, first, goal, test_user, "proposed", _utc(17))
    await _fixture_proposal(db_session, blank, goal, test_user, "dismissed", _utc(18), status="dismissed")
    promoted = await _fixture_proposal(db_session, negative_turns[0][0], goal, test_user, "promoted", _utc(19))
    late_promoted = await _fixture_proposal(db_session, negative_turns[1][0], goal, test_user, "late-promoted", _utc(20))
    requests = [
        await _fixture_request(
            db_session,
            goal,
            run,
            test_user,
            sequence,
            _utc(20 + sequence, 12),
            source_proposal_id=promoted.id if sequence in {2, 4, 6, 8} else None,
        )
        for sequence in range(1, 11)
    ]
    status_reason = (
        ("pending", "submitted"),
        ("being_considered", "considering"),
        ("applied", "applied"),
        ("deferred", "run_changed"),
        ("rejected", "steering_ineligible"),
        ("superseded", "target_already_started"),
        ("needs_clarification", "supersedes_required"),
        ("withdrawn", "invalid_supersedes_request"),
        ("pending", "superseded"),
        ("pending", "withdrawn"),
    )
    for request, (status, reason) in zip(requests, status_reason, strict=True):
        submitted = request.submitted_at
        if request is requests[1]:
            await _fixture_transition(db_session, request, 1, status, reason, submitted + timedelta(minutes=10))
            await _fixture_transition(db_session, request, 2, "applied", "applied", datetime(2026, 10, 2, tzinfo=timezone.utc))
        elif request is requests[2]:
            await _fixture_transition(db_session, request, 1, "being_considered", "considering", submitted - timedelta(minutes=5))
            await _fixture_transition(db_session, request, 2, status, reason, submitted + timedelta(minutes=20))
        else:
            await _fixture_transition(db_session, request, 1, status, reason, submitted + timedelta(minutes=10))
    promoted.status, promoted.promoted_request_id = "promoted", requests[1].id
    late_request = await _fixture_request(
        db_session,
        goal,
        run,
        test_user,
        11,
        end_at,
        source_proposal_id=late_promoted.id,
    )
    late_promoted.status, late_promoted.promoted_request_id = "promoted", late_request.id
    await db_session.flush()

    await _fixture_result_link(db_session, requests[2], run, "one", _utc(25))
    await _fixture_result_link(db_session, requests[3], run, "two", _utc(25))
    await _fixture_result_link(db_session, requests[3], run, "three", _utc(25))
    await _fixture_result_link(db_session, requests[1], run, "late", datetime(2026, 10, 2, tzinfo=timezone.utc))

    report = await _service().summarize(
        db_session, goal.project_id, ConversationLearningWindow(start_at, end_at)
    )
    generated_at = report.window["generated_at"]
    expected = _zero_report(start_at, end_at, generated_at)
    expected["sample"] = {"goals": 2, "operators": 2, "questions": 15}
    expected["chat"] = {
        "status_counts": {"pending": 2, "running": 1, "completed": 10, "failed": 1, "interrupted_unknown": 1},
        "answered": 9,
        "terminal_without_answer": 3,
        "latency_seconds": {"count": 11, "invalid": 1, "median": 60.0, "p95": 3600.0},
    }
    expected["investigations"] = {
        "triggered": 4,
        "accounted_terminal": 3,
        "status_counts": {"pending": 0, "running": 1, "completed": 1, "limited": 1, "failed": 1, "cancelled": 0, "unavailable": 0, "interrupted_unknown": 0},
        "attempts": 3,
        "repairs": 1,
        "retries": 1,
        "accumulated_tokens": 150,
        "latency_seconds": {"count": 1, "invalid": 1, "median": 7200.0, "p95": 7200.0},
    }
    expected["context_limits"] = {
        "turns_truncated": 1,
        "turns_with_omissions": 1,
        "omitted_records": 3,
        "truncated_source_counts": {**expected["context_limits"]["truncated_source_counts"], "goal": 1},
        "investigation_omission_status_counts": {
            "restricted": 1,
            "unsafe": 1,
            "binary": 1,
            "too_large": 1,
            "changed": 1,
            "omitted_by_limit": 1,
        },
    }
    expected["steering"] = {
        "proposal_status_counts": {"proposed": 2, "dismissed": 1, "promoted": 1},
        "request_status_counts": {"pending": 3, "being_considered": 1, "applied": 1, "deferred": 1, "rejected": 1, "superseded": 1, "needs_clarification": 1, "withdrawn": 1},
        "request_reason_counts": {
            "submitted": 1,
            "considering": 1,
            "run_changed": 1,
            "steering_ineligible": 1,
            "target_already_started": 1,
            "supersedes_required": 1,
            "invalid_supersedes_request": 1,
            "superseded": 1,
            "applied": 1,
            "withdrawn": 1,
        },
        "direct_requests": 6,
        "proposal_derived_requests": 4,
        "requests_with_result_actions": 2,
        "result_links": 3,
        "submit_to_considered_seconds": {"count": 1, "invalid": 1, "median": 600.0, "p95": 600.0},
        "submit_to_finished_seconds": {"count": 6, "invalid": 0, "median": 600.0, "p95": 1200.0},
    }
    expected["operator_feedback"] = {
        "rated": 8,
        "unrated_answered": 1,
        "helpful": 1,
        "not_helpful": 7,
        "not_helpful_reason_counts": {reason: 1 for reason in REASONS},
    }

    assert generated_at.tzinfo is not None and generated_at.utcoffset() == timedelta(0)
    assert asdict(report) == expected

    filtered = await _service().summarize(
        db_session, goal.project_id, ConversationLearningWindow(start_at, end_at, goal.id)
    )
    assert filtered.window["goal_filtered"] is True
    assert filtered.sample == {"goals": 1, "operators": 1, "questions": 14}
    assert filtered.chat["status_counts"]["pending"] == 1

    serialized = json.dumps(asdict(report), default=str, sort_keys=True)
    secrets = {
        "private-message-1",
        "private-answer-helpful",
        "private-investigation-objective-completed",
        "private-investigation-report-completed",
        "private-investigation-reference-restricted",
        "private-directive-1",
        "private-impact-1",
        goal.objective,
        other_goal.objective,
        str(goal.id),
        str(other_goal.id),
        str(test_user.id),
        str(other_actor.id),
        str(proposed.id),
        str(requests[0].id),
    }
    assert all(secret not in serialized for secret in secrets)
