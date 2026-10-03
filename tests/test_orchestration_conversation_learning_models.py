import os
import sqlite3
import subprocess
import sys
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError


FEEDBACK_REASONS = (
    "unanswered",
    "incorrect",
    "missing_context",
    "stale_context",
    "unclear",
    "too_limited",
    "other",
)


def _alembic(env, *command):
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *command],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def _migration_version(connection):
    return connection.execute("SELECT version_num FROM alembic_version").fetchone()[0]


def test_feedback_identity_is_stable_and_namespaced():
    from huddleroom.models.orchestration_conversation import conversation_feedback_id

    response_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
    actor_id = uuid.UUID("22222222-2222-2222-2222-222222222222")

    assert conversation_feedback_id(response_id, actor_id) == uuid.UUID(
        "d76a0b97-77d8-5357-a6eb-f65037de6b42"
    )
    assert conversation_feedback_id(response_id, actor_id) == conversation_feedback_id(
        response_id, actor_id
    )
    assert conversation_feedback_id(response_id, actor_id) != conversation_feedback_id(
        response_id, uuid.uuid4()
    )
    assert conversation_feedback_id(response_id, actor_id) != conversation_feedback_id(
        uuid.uuid4(), actor_id
    )


def test_feedback_model_has_only_the_immutable_schema_columns():
    from huddleroom.models import ConversationFeedback

    table = ConversationFeedback.__table__
    assert set(table.columns.keys()) == {
        "id",
        "response_id",
        "actor_id",
        "rating",
        "reason",
        "created_at",
    }
    assert table.c.id.primary_key is True
    assert table.c.response_id.nullable is False
    assert table.c.actor_id.nullable is False
    assert table.c.rating.nullable is False
    assert table.c.reason.nullable is True
    assert table.c.created_at.nullable is False
    assert "updated_at" not in table.c

    foreign_keys = {
        (foreign_key.parent.name, foreign_key.column.table.name, foreign_key.column.name,
         foreign_key.ondelete)
        for foreign_key in table.foreign_keys
    }
    assert foreign_keys == {
        ("response_id", "orchestration_conversation_responses", "id", "CASCADE"),
        ("actor_id", "users", "id", "RESTRICT"),
    }
    assert any(
        set(constraint.columns.keys()) == {"response_id", "actor_id"}
        for constraint in table.constraints
        if constraint.__class__.__name__ == "UniqueConstraint"
    )


async def _completed_response(db_session, conversation_goal_run, test_user, sequence):
    from huddleroom.models.orchestration_conversation import (
        ConversationMessage,
        ConversationResponse,
        conversation_response_id,
    )

    goal, run = conversation_goal_run
    now = datetime.now(timezone.utc)
    message = ConversationMessage(
        goal_id=goal.id,
        actor_id=test_user.id,
        client_request_id=uuid.uuid4(),
        sequence=sequence,
        content="question",
    )
    db_session.add(message)
    await db_session.flush()
    response = ConversationResponse(
        id=conversation_response_id(message.id),
        message_id=message.id,
        run_id=run.id,
        status="completed",
        dossier={},
        context_manifest={},
        context_version="context-v1",
        provider_request_id=f"rally-chat:feedback-{sequence}",
        answer="A completed answer.",
        started_at=now,
        deadline_at=now,
        finished_at=now,
    )
    db_session.add(response)
    await db_session.flush()
    return response


@pytest.mark.asyncio
async def test_feedback_persists_helpful_and_every_allowed_negative_reason(
    db_session, conversation_goal_run, test_user
):
    from huddleroom.models import ConversationFeedback
    from huddleroom.models.orchestration_conversation import conversation_feedback_id

    helpful_response = await _completed_response(
        db_session, conversation_goal_run, test_user, 1
    )
    db_session.add(
        ConversationFeedback(
            id=conversation_feedback_id(helpful_response.id, test_user.id),
            response_id=helpful_response.id,
            actor_id=test_user.id,
            rating="helpful",
            reason=None,
        )
    )
    for sequence, reason in enumerate(FEEDBACK_REASONS, start=2):
        response = await _completed_response(
            db_session, conversation_goal_run, test_user, sequence
        )
        db_session.add(
            ConversationFeedback(
                id=conversation_feedback_id(response.id, test_user.id),
                response_id=response.id,
                actor_id=test_user.id,
                rating="not_helpful",
                reason=reason,
            )
        )
    await db_session.flush()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rating", "reason"),
    tuple(("helpful", reason) for reason in FEEDBACK_REASONS)
    + (
        ("not_helpful", None),
        ("not_helpful", "not_an_allowed_reason"),
        ("not_a_rating", None),
    ),
)
async def test_feedback_constraints_reject_invalid_rating_and_reason_pairs(
    db_session, conversation_goal_run, test_user, rating, reason
):
    from huddleroom.models import ConversationFeedback
    from huddleroom.models.orchestration_conversation import conversation_feedback_id

    response = await _completed_response(db_session, conversation_goal_run, test_user, 1)
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            db_session.add(
                ConversationFeedback(
                    id=conversation_feedback_id(response.id, test_user.id),
                    response_id=response.id,
                    actor_id=test_user.id,
                    rating=rating,
                    reason=reason,
                )
            )
            await db_session.flush()


@pytest.mark.asyncio
async def test_feedback_unique_constraint_rejects_second_row_for_response_and_actor(
    db_session, conversation_goal_run, test_user
):
    from huddleroom.models import ConversationFeedback
    from huddleroom.models.orchestration_conversation import conversation_feedback_id

    response = await _completed_response(db_session, conversation_goal_run, test_user, 1)
    db_session.add(
        ConversationFeedback(
            id=conversation_feedback_id(response.id, test_user.id),
            response_id=response.id,
            actor_id=test_user.id,
            rating="helpful",
            reason=None,
        )
    )
    await db_session.flush()

    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            db_session.add(
                ConversationFeedback(
                    id=uuid.uuid4(),
                    response_id=response.id,
                    actor_id=test_user.id,
                    rating="not_helpful",
                    reason="unclear",
                )
            )
            await db_session.flush()


@pytest.mark.asyncio
async def test_deleting_response_cascades_to_its_feedback(
    db_session, conversation_goal_run, test_user
):
    from sqlalchemy import select

    from huddleroom.models import ConversationFeedback
    from huddleroom.models.orchestration_conversation import conversation_feedback_id

    response = await _completed_response(db_session, conversation_goal_run, test_user, 1)
    feedback_id = conversation_feedback_id(response.id, test_user.id)
    db_session.add(
        ConversationFeedback(
            id=feedback_id,
            response_id=response.id,
            actor_id=test_user.id,
            rating="helpful",
            reason=None,
        )
    )
    await db_session.flush()

    await db_session.delete(response)
    await db_session.flush()

    assert (await db_session.execute(
        select(ConversationFeedback.id).where(ConversationFeedback.id == feedback_id)
    )).scalar_one_or_none() is None


def test_feedback_user_foreign_key_restricts_deletion(tmp_path):
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import Session

    from huddleroom.models import ConversationFeedback, OrchestrationGoal, Project, User
    from huddleroom.models.orchestration_conversation import (
        ConversationMessage,
        ConversationResponse,
        conversation_feedback_id,
        conversation_response_id,
    )

    db_path = tmp_path / "feedback-restrict.db"
    env = {**os.environ, "RALLY_DATABASE_URL": f"sqlite+aiosqlite:///{db_path}"}
    _alembic(env, "upgrade", "044")
    engine = create_engine(f"sqlite:///{db_path}")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys = ON")

    with Session(engine) as session:
        owner = User(email="feedback-owner@example.test", hashed_password="secret")
        actor = User(email="feedback-actor@example.test", hashed_password="secret")
        project = Project(name="Feedback restrict")
        session.add_all((owner, actor, project))
        session.flush()
        goal = OrchestrationGoal(project_id=project.id, objective="Feedback restrict")
        session.add(goal)
        session.flush()
        message = ConversationMessage(
            goal_id=goal.id,
            actor_id=owner.id,
            client_request_id=uuid.uuid4(),
            sequence=1,
            content="question",
        )
        session.add(message)
        session.flush()
        response = ConversationResponse(
            id=conversation_response_id(message.id),
            message_id=message.id,
            dossier={},
            context_manifest={},
            context_version="context-v1",
            provider_request_id="rally-chat:feedback-restrict",
        )
        session.add(response)
        session.flush()
        session.add(
            ConversationFeedback(
                id=conversation_feedback_id(response.id, actor.id),
                response_id=response.id,
                actor_id=actor.id,
                rating="helpful",
                reason=None,
            )
        )
        session.commit()
        actor_id = actor.id

    with Session(engine) as session:
        session.delete(session.get(User, actor_id))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
    engine.dispose()


def test_migration_044_round_trip_creates_only_feedback_schema(tmp_path):
    db_path = tmp_path / "conversation_feedback.db"
    env = {**os.environ, "RALLY_DATABASE_URL": f"sqlite+aiosqlite:///{db_path}"}

    _alembic(env, "upgrade", "043")
    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "orchestration_conversation_feedback" not in tables
        assert _migration_version(connection) == "043"

    _alembic(env, "upgrade", "044")
    with sqlite3.connect(db_path) as connection:
        assert _migration_version(connection) == "044"
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(orchestration_conversation_feedback)")
        }
        assert columns == {"id", "response_id", "actor_id", "rating", "reason", "created_at"}
        foreign_keys = {
            (row[3], row[2], row[4], row[6].upper())
            for row in connection.execute("PRAGMA foreign_key_list(orchestration_conversation_feedback)")
        }
        assert foreign_keys == {
            ("response_id", "orchestration_conversation_responses", "id", "CASCADE"),
            ("actor_id", "users", "id", "RESTRICT"),
        }
        schema_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='orchestration_conversation_feedback'"
        ).fetchone()[0]
        for constraint in (
            "uq_orch_conversation_feedback_response_actor",
            "ck_orch_conversation_feedback_rating",
            "ck_orch_conversation_feedback_reason",
        ):
            assert constraint in schema_sql

    _alembic(env, "downgrade", "043")
    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "orchestration_conversation_feedback" not in tables
        assert _migration_version(connection) == "043"

    _alembic(env, "upgrade", "044")
    with sqlite3.connect(db_path) as connection:
        assert _migration_version(connection) == "044"
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='orchestration_conversation_feedback'"
        ).fetchone() == ("orchestration_conversation_feedback",)
