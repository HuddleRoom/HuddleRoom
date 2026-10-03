import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
import uuid

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError


def _alembic(env, *command):
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *command],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def _utc(value):
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def test_migration_041_round_trip_creates_conversation_schema(tmp_path):
    db_path = tmp_path / "conversation.db"
    env = {**os.environ, "RALLY_DATABASE_URL": f"sqlite+aiosqlite:///{db_path}"}
    _alembic(env, "upgrade", "040")
    _alembic(env, "upgrade", "041")

    with sqlite3.connect(db_path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {
            "orchestration_conversation_messages",
            "orchestration_conversation_responses",
            "orchestration_conversation_reservations",
        } <= tables
        columns = {
            table: {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            for table in tables & {
                "orchestration_conversation_messages",
                "orchestration_conversation_responses",
                "orchestration_conversation_reservations",
            }
        }
        assert columns["orchestration_conversation_messages"] == {
            "id", "goal_id", "actor_id", "client_request_id", "sequence", "content", "created_at"
        }
        assert columns["orchestration_conversation_responses"] == {
            "id", "message_id", "run_id", "status", "dossier", "context_manifest", "context_version",
            "provider_request_id", "answer", "error", "started_at", "deadline_at", "finished_at",
            "created_at", "updated_at",
        }
        assert columns["orchestration_conversation_reservations"] == {
            "id", "response_id", "goal_id", "actor_id", "ceiling_snapshot", "reserved_tokens",
            "settled_tokens", "released_tokens", "status", "committed_at", "settled_at", "released_at",
            "created_at", "updated_at",
        }
        message_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='orchestration_conversation_messages'"
        ).fetchone()[0]
        response_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='orchestration_conversation_responses'"
        ).fetchone()[0]
        reservation_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='orchestration_conversation_reservations'"
        ).fetchone()[0]
        for name in (
            "uq_orch_conversation_messages_goal_actor_request",
            "uq_orch_conversation_messages_goal_sequence",
            "fk_orch_conversation_messages_goal_id",
            "fk_orch_conversation_messages_actor_id",
        ):
            assert name in message_sql
        for name in (
            "uq_orch_conversation_responses_message_id",
            "uq_orch_conversation_responses_provider_request_id",
            "fk_orch_conversation_responses_message_id",
            "ck_orch_conversation_responses_status",
            "ck_orch_conversation_responses_lifecycle",
        ):
            assert name in response_sql
        response_fk_columns = {
            row[3]
            for row in connection.execute("PRAGMA foreign_key_list(orchestration_conversation_responses)")
        }
        assert "run_id" not in response_fk_columns
        for name in (
            "uq_orch_conversation_reservations_response_id",
            "fk_orch_conversation_reservations_response_id",
            "fk_orch_conversation_reservations_goal_id",
            "fk_orch_conversation_reservations_actor_id",
            "ck_orch_conversation_reservations_amounts",
            "ck_orch_conversation_reservations_status",
            "ck_orch_conversation_reservations_lifecycle",
        ):
            assert name in reservation_sql
        indexes = {
            row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }
        assert {
            "idx_orch_conversation_messages_goal_sequence",
            "idx_orch_conversation_responses_status_deadline",
            "idx_orch_conversation_reservations_goal_actor_status",
        } <= indexes

    _alembic(env, "downgrade", "040")
    with sqlite3.connect(db_path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "orchestration_conversation_messages" not in tables
        assert "orchestration_conversation_responses" not in tables
        assert "orchestration_conversation_reservations" not in tables


def test_migration_042_round_trip_creates_investigation_schema(tmp_path):
    db_path = tmp_path / "investigation.db"
    env = {**os.environ, "RALLY_DATABASE_URL": f"sqlite+aiosqlite:///{db_path}"}
    _alembic(env, "upgrade", "041")
    _alembic(env, "upgrade", "042")
    with sqlite3.connect(db_path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {
            "orchestration_conversation_investigations",
            "orchestration_conversation_investigation_reservations",
        } <= tables
        columns = {
            table: {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            for table in {
                "orchestration_conversation_investigations",
                "orchestration_conversation_investigation_reservations",
            }
        }
        assert columns["orchestration_conversation_investigations"] == {
            "id", "response_id", "goal_id", "actor_id", "context_version", "status",
            "objective", "scope", "input_manifest", "provider_identity",
            "provider_request_id", "attempt_count", "repair_count", "retry_count",
            "accumulated_tokens", "report", "error", "started_at", "deadline_at",
            "finished_at", "cancelled_at", "created_at", "updated_at",
        }
        assert columns["orchestration_conversation_investigation_reservations"] == {
            "id", "investigation_id", "goal_id", "actor_id", "ceiling_snapshot", "reserved_tokens",
            "settled_tokens", "released_tokens", "status", "committed_at", "settled_at", "released_at",
            "created_at", "updated_at",
        }
        investigation_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='orchestration_conversation_investigations'"
        ).fetchone()[0]
        reservation_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='orchestration_conversation_investigation_reservations'"
        ).fetchone()[0]
        for name in (
            "uq_orch_conversation_investigations_response",
            "uq_orch_conversation_investigations_provider_identity",
            "uq_orch_conversation_investigations_provider_request",
            "fk_orch_conversation_investigations_response_id",
            "fk_orch_conversation_investigations_goal_id",
            "fk_orch_conversation_investigations_actor_id",
            "ck_orch_conversation_investigations_status",
            "ck_orch_conversation_investigations_counters",
            "ck_orch_conversation_investigations_lifecycle",
        ):
            assert name in investigation_sql
        for name in (
            "uq_orch_conversation_investigation_reservations_investigation",
            "fk_orch_conv_inv_res_investigation_id",
            "fk_orch_conversation_investigation_reservations_goal_id",
            "fk_orch_conversation_investigation_reservations_actor_id",
            "ck_orch_conversation_investigation_reservations_amounts",
            "ck_orch_conversation_investigation_reservations_status",
            "ck_orch_conversation_investigation_reservations_lifecycle",
        ):
            assert name in reservation_sql
        investigation_fks = {
            (row[3], row[2], row[4], row[6].upper())
            for row in connection.execute(
                "PRAGMA foreign_key_list(orchestration_conversation_investigations)"
            )
        }
        assert {
            ("response_id", "orchestration_conversation_responses", "id", "CASCADE"),
            ("goal_id", "orchestration_goals", "id", "CASCADE"),
            ("actor_id", "users", "id", "RESTRICT"),
        } <= investigation_fks
        reservation_fks = {
            (row[3], row[2], row[4], row[6].upper())
            for row in connection.execute(
                "PRAGMA foreign_key_list(orchestration_conversation_investigation_reservations)"
            )
        }
        assert {
            ("investigation_id", "orchestration_conversation_investigations", "id", "CASCADE"),
            ("goal_id", "orchestration_goals", "id", "CASCADE"),
            ("actor_id", "users", "id", "RESTRICT"),
        } <= reservation_fks
        indexes = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        assert {
            "idx_orch_conversation_investigations_goal_status_deadline",
            "idx_orch_conv_inv_res_goal_actor_status",
        } <= indexes
    _alembic(env, "downgrade", "041")
    with sqlite3.connect(db_path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "orchestration_conversation_investigations" not in tables
        assert "orchestration_conversation_investigation_reservations" not in tables


def test_conversation_identities_are_stable_and_provider_scoped():
    from huddleroom.models.orchestration_conversation import (
        conversation_message_id,
        conversation_provider_request_id,
        conversation_reservation_id,
        conversation_response_id,
    )

    goal_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
    actor_id = uuid.UUID("22222222-2222-2222-2222-222222222222")
    request_id = uuid.UUID("33333333-3333-3333-3333-333333333333")

    message_id = conversation_message_id(goal_id, actor_id, request_id)
    assert message_id == conversation_message_id(goal_id, actor_id, request_id)
    assert message_id != conversation_message_id(goal_id, actor_id, uuid.uuid4())

    response_id = conversation_response_id(message_id)
    assert response_id == conversation_response_id(message_id)
    assert conversation_reservation_id(response_id) == conversation_reservation_id(response_id)
    assert conversation_provider_request_id(response_id) == f"rally-chat:{response_id}"


def test_investigation_identities_are_stable_and_attempt_scoped():
    from huddleroom.models.orchestration_conversation import (
        conversation_investigation_id,
        conversation_investigation_provider_identity,
        conversation_investigation_provider_request_id,
        conversation_investigation_reservation_id,
    )

    response_id = uuid.UUID("44444444-4444-4444-8444-444444444444")
    investigation_id = conversation_investigation_id(response_id, "context-v1")
    assert investigation_id == uuid.UUID("d0fe38ef-1c5a-54ee-92b9-537d11ee9dbc")
    assert investigation_id != conversation_investigation_id(response_id, "context-v2")
    assert conversation_investigation_reservation_id(investigation_id) == uuid.UUID(
        "f80e07ed-5a92-5644-8b56-d0f44a08770c"
    )
    assert conversation_investigation_provider_identity(investigation_id) == (
        "rally-chat-investigation:d0fe38ef-1c5a-54ee-92b9-537d11ee9dbc"
    )
    assert conversation_investigation_provider_request_id(investigation_id, 1) == (
        "rally-chat-investigation:d0fe38ef-1c5a-54ee-92b9-537d11ee9dbc:1"
    )
    assert conversation_investigation_provider_request_id(investigation_id, 2) == (
        "rally-chat-investigation:d0fe38ef-1c5a-54ee-92b9-537d11ee9dbc:2"
    )
    for attempt in (0, 3):
        with pytest.raises(ValueError, match="investigation attempt must be 1 or 2"):
            conversation_investigation_provider_request_id(investigation_id, attempt)


def test_conversation_allowance_setting_defaults_enables_and_rejects_negative(monkeypatch):
    from huddleroom.config import Settings

    monkeypatch.delenv("RALLY_ORCHESTRATION_CONVERSATION_ALLOWANCE_TOKENS", raising=False)
    assert Settings(_env_file=None).orchestration_conversation_allowance_tokens == 50000

    monkeypatch.setenv("RALLY_ORCHESTRATION_CONVERSATION_ALLOWANCE_TOKENS", "12")
    assert Settings(_env_file=None).orchestration_conversation_allowance_tokens == 12

    monkeypatch.setenv("RALLY_ORCHESTRATION_CONVERSATION_ALLOWANCE_TOKENS", "-1")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_investigation_setting_is_default_on(monkeypatch):
    from huddleroom.config import Settings

    monkeypatch.delenv("RALLY_ORCHESTRATION_CONVERSATION_INVESTIGATION_ENABLED", raising=False)
    assert Settings(_env_file=None).orchestration_conversation_investigation_enabled is True
    monkeypatch.setenv("RALLY_ORCHESTRATION_CONVERSATION_INVESTIGATION_ENABLED", "false")
    assert Settings(_env_file=None).orchestration_conversation_investigation_enabled is False


@pytest.mark.asyncio
async def test_conversation_models_persist_a_valid_goal_owned_turn(db_session, conversation_goal_run, test_user):
    from huddleroom.models.orchestration_conversation import (
        ConversationMessage,
        ConversationReservation,
        ConversationResponse,
        conversation_message_id,
        conversation_provider_request_id,
        conversation_reservation_id,
        conversation_response_id,
    )

    goal, run = conversation_goal_run
    request_id = uuid.uuid4()
    message_id = conversation_message_id(goal.id, test_user.id, request_id)
    response_id = conversation_response_id(message_id)
    response = ConversationResponse(
        id=response_id,
        message_id=message_id,
        run_id=run.id,
        dossier={},
        context_manifest={},
        context_version="context-v1",
        provider_request_id=conversation_provider_request_id(response_id),
    )
    db_session.add(
        ConversationMessage(
            id=message_id,
            goal_id=goal.id,
            actor_id=test_user.id,
            client_request_id=request_id,
            sequence=1,
            content="question",
        )
    )
    await db_session.flush()
    db_session.add(response)
    await db_session.flush()
    db_session.add(
        ConversationReservation(
            id=conversation_reservation_id(response_id),
            response_id=response_id,
            goal_id=goal.id,
            actor_id=test_user.id,
            ceiling_snapshot=1000,
            reserved_tokens=864,
        )
    )
    await db_session.flush()
    assert response.status == "pending"


@pytest.mark.asyncio
async def test_investigation_models_persist_every_valid_lifecycle(
    db_session, conversation_goal_run, test_user
):
    from huddleroom.models import ConversationInvestigation, ConversationInvestigationReservation
    from huddleroom.models.orchestration_conversation import (
        ConversationMessage,
        ConversationResponse,
        conversation_investigation_id,
        conversation_investigation_provider_identity,
        conversation_investigation_reservation_id,
        conversation_response_id,
    )

    goal, run = conversation_goal_run
    now = datetime.now(timezone.utc)
    lifecycles = (
        ("pending", {}, "reserved", {}),
        ("running", {
            "attempt_count": 2, "repair_count": 1, "accumulated_tokens": 1,
            "started_at": now, "deadline_at": now,
        }, "committed", {"committed_at": now}),
        ("completed", {"attempt_count": 2, "retry_count": 1, "finished_at": now}, "settled", {
            "settled_tokens": 7, "released_tokens": 3, "committed_at": now,
            "settled_at": now, "released_at": now,
        }),
        ("limited", {"finished_at": now}, "released", {"released_tokens": 10, "released_at": now}),
        ("failed", {"attempt_count": 1, "finished_at": now}, "held_unknown", {"committed_at": now}),
        ("cancelled", {"finished_at": now, "cancelled_at": now}, "reserved", {}),
        ("unavailable", {"finished_at": now}, "reserved", {}),
        ("interrupted_unknown", {"attempt_count": 1, "finished_at": now}, "reserved", {}),
    )
    for sequence, (status, investigation_values, reservation_status, reservation_values) in enumerate(
        lifecycles, start=1
    ):
        message = ConversationMessage(
            goal_id=goal.id, actor_id=test_user.id, client_request_id=uuid.uuid4(),
            sequence=sequence, content="question",
        )
        db_session.add(message)
        await db_session.flush()
        response = ConversationResponse(
            id=conversation_response_id(message.id), message_id=message.id, run_id=run.id,
            dossier={}, context_manifest={}, context_version="context-v1",
            provider_request_id=f"rally-chat:investigation-{sequence}",
        )
        db_session.add(response)
        await db_session.flush()
        investigation_id = conversation_investigation_id(response.id, response.context_version)
        db_session.add(
            ConversationInvestigation(
                id=investigation_id, response_id=response.id, goal_id=goal.id, actor_id=test_user.id,
                context_version=response.context_version, status=status, objective="investigate", scope=[],
                input_manifest={}, provider_identity=conversation_investigation_provider_identity(investigation_id),
                **investigation_values,
            )
        )
        await db_session.flush()
        db_session.add(
            ConversationInvestigationReservation(
                id=conversation_investigation_reservation_id(investigation_id), investigation_id=investigation_id,
                goal_id=goal.id, actor_id=test_user.id, ceiling_snapshot=10, reserved_tokens=10,
                status=reservation_status, **reservation_values,
            )
        )
        await db_session.flush()


@pytest.mark.asyncio
async def test_investigation_constraints_reject_invalid_records(
    db_session, conversation_goal_run, test_user
):
    from huddleroom.models import ConversationInvestigation, ConversationInvestigationReservation
    from huddleroom.models.orchestration_conversation import (
        ConversationMessage,
        ConversationResponse,
        conversation_investigation_id,
        conversation_investigation_provider_identity,
        conversation_investigation_reservation_id,
        conversation_response_id,
    )

    goal, run = conversation_goal_run
    now = datetime.now(timezone.utc)

    async def response_for(sequence):
        message = ConversationMessage(
            goal_id=goal.id, actor_id=test_user.id, client_request_id=uuid.uuid4(),
            sequence=sequence, content="question",
        )
        db_session.add(message)
        await db_session.flush()
        response = ConversationResponse(
            id=conversation_response_id(message.id), message_id=message.id, run_id=run.id,
            dossier={}, context_manifest={}, context_version="context-v1",
            provider_request_id=f"rally-chat:investigation-{sequence}",
        )
        db_session.add(response)
        await db_session.flush()
        return response

    def investigation_for(response, **values):
        investigation_id = values.pop(
            "id", conversation_investigation_id(response.id, response.context_version)
        )
        provider_identity = values.pop(
            "provider_identity", conversation_investigation_provider_identity(investigation_id)
        )
        return ConversationInvestigation(
            id=investigation_id, response_id=response.id, goal_id=goal.id, actor_id=test_user.id,
            context_version=response.context_version, objective="investigate", scope=[], input_manifest={},
            provider_identity=provider_identity, **values,
        )

    response = await response_for(1)
    investigation = investigation_for(response)
    db_session.add(investigation)
    await db_session.flush()
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            duplicate_id = uuid.uuid4()
            db_session.add(
                investigation_for(
                    response,
                    id=duplicate_id,
                    provider_identity=conversation_investigation_provider_identity(duplicate_id),
                )
            )
            await db_session.flush()

    for sequence, values in (
        (2, {"status": "unknown"}),
        (3, {"attempt_count": -1}),
        (4, {"attempt_count": 3}),
        (5, {"status": "running", "attempt_count": 0, "started_at": now, "deadline_at": now}),
        (6, {"status": "running", "attempt_count": 1, "started_at": now, "deadline_at": now, "repair_count": -1}),
        (7, {"status": "running", "attempt_count": 1, "started_at": now, "deadline_at": now, "repair_count": 2}),
        (8, {"status": "running", "attempt_count": 1, "started_at": now, "deadline_at": now, "retry_count": -1}),
        (9, {"status": "running", "attempt_count": 1, "started_at": now, "deadline_at": now, "retry_count": 2}),
        (10, {"status": "running", "attempt_count": 2, "started_at": now, "deadline_at": now, "repair_count": 1, "retry_count": 1}),
        (11, {"accumulated_tokens": -1}),
        (12, {"attempt_count": 0, "repair_count": 1}),
    ):
        with pytest.raises(IntegrityError):
            async with db_session.begin_nested():
                db_session.add(investigation_for(await response_for(sequence), **values))
                await db_session.flush()

    for sequence, amount in enumerate(
        ("ceiling_snapshot", "reserved_tokens", "settled_tokens", "released_tokens"), start=13
    ):
        response = await response_for(sequence)
        invalid_investigation = investigation_for(response)
        db_session.add(invalid_investigation)
        await db_session.flush()
        amounts = {
            "ceiling_snapshot": 0,
            "reserved_tokens": 0,
            "settled_tokens": 0,
            "released_tokens": 0,
        }
        amounts[amount] = -1
        with pytest.raises(IntegrityError):
            async with db_session.begin_nested():
                db_session.add(
                    ConversationInvestigationReservation(
                        id=conversation_investigation_reservation_id(invalid_investigation.id),
                        investigation_id=invalid_investigation.id, goal_id=goal.id, actor_id=test_user.id,
                        **amounts,
                    )
                )
                await db_session.flush()

    response = await response_for(17)
    invalid_investigation = investigation_for(response)
    db_session.add(invalid_investigation)
    await db_session.flush()
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            db_session.add(
                ConversationInvestigationReservation(
                    id=conversation_investigation_reservation_id(invalid_investigation.id),
                    investigation_id=invalid_investigation.id, goal_id=goal.id, actor_id=test_user.id,
                    ceiling_snapshot=0, reserved_tokens=0, status="committed",
                )
            )
            await db_session.flush()


@pytest.mark.asyncio
async def test_conversation_constraints_reject_invalid_links_and_lifecycles(
    db_session, conversation_goal_run, test_user
):
    from huddleroom.models.orchestration_conversation import (
        ConversationMessage,
        ConversationReservation,
        ConversationResponse,
    )

    goal, _ = conversation_goal_run
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            db_session.add(
                ConversationMessage(
                    goal_id=uuid.uuid4(), actor_id=test_user.id, client_request_id=uuid.uuid4(),
                    sequence=1, content="invalid goal",
                )
            )
            await db_session.flush()

    message = ConversationMessage(
        goal_id=goal.id, actor_id=test_user.id, client_request_id=uuid.uuid4(), sequence=2, content="question"
    )
    db_session.add(message)
    await db_session.flush()
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            db_session.add(
                ConversationResponse(
                    message_id=message.id, dossier={}, context_manifest={}, context_version="v1",
                    provider_request_id="rally-chat:invalid", status="running",
                )
            )
            await db_session.flush()

    response = ConversationResponse(
        message_id=message.id, dossier={}, context_manifest={}, context_version="v1",
        provider_request_id="rally-chat:valid",
    )
    db_session.add(response)
    await db_session.flush()
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            db_session.add(
                ConversationReservation(
                    response_id=response.id, goal_id=goal.id, actor_id=test_user.id,
                    ceiling_snapshot=1, reserved_tokens=1, settled_tokens=2,
                )
            )
            await db_session.flush()


@pytest.mark.asyncio
async def test_settled_reservation_records_the_released_remainder(
    db_session, conversation_goal_run, test_user
):
    from huddleroom.models.orchestration_conversation import (
        ConversationMessage,
        ConversationReservation,
        ConversationResponse,
    )

    goal, _ = conversation_goal_run
    message = ConversationMessage(
        goal_id=goal.id, actor_id=test_user.id, client_request_id=uuid.uuid4(), sequence=1, content="question"
    )
    db_session.add(message)
    await db_session.flush()
    response = ConversationResponse(
        message_id=message.id, dossier={}, context_manifest={}, context_version="v1", provider_request_id="rally-chat:settled"
    )
    db_session.add(response)
    await db_session.flush()
    now = datetime.now(timezone.utc)
    db_session.add(
        ConversationReservation(
            response_id=response.id, goal_id=goal.id, actor_id=test_user.id, ceiling_snapshot=10,
            reserved_tokens=10, settled_tokens=7, released_tokens=3, status="settled",
            committed_at=now, settled_at=now, released_at=now,
        )
    )
    await db_session.flush()


@pytest.mark.asyncio
async def test_orm_schema_allows_only_unclaimed_failed_investigations(
    db_session, conversation_goal_run, test_engine, test_user
):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from huddleroom.models import ConversationInvestigation, ConversationInvestigationReservation
    from huddleroom.models.orchestration_conversation import (
        ConversationMessage,
        ConversationResponse,
        conversation_investigation_id,
        conversation_investigation_provider_identity,
        conversation_investigation_provider_request_id,
        conversation_investigation_reservation_id,
        conversation_response_id,
    )

    goal, run = conversation_goal_run
    now = datetime.now(timezone.utc)
    session_factory = async_sessionmaker(test_engine, expire_on_commit=False)

    async def investigation_for(session, sequence, **values):
        message = ConversationMessage(
            goal_id=goal.id,
            actor_id=test_user.id,
            client_request_id=uuid.uuid4(),
            sequence=sequence,
            content="question",
        )
        session.add(message)
        await session.flush()
        response = ConversationResponse(
            id=conversation_response_id(message.id),
            message_id=message.id,
            run_id=run.id,
            dossier={},
            context_manifest={},
            context_version="context-v1",
            provider_request_id=f"rally-chat:failed-zero-{sequence}",
        )
        session.add(response)
        await session.flush()
        investigation_id = conversation_investigation_id(response.id, response.context_version)
        defaults = {
            "status": "pending",
            "attempt_count": 0,
            "repair_count": 0,
            "retry_count": 0,
            "accumulated_tokens": 0,
            "provider_request_id": None,
            "started_at": None,
            "deadline_at": None,
            "finished_at": None,
        }
        defaults.update(values)
        if defaults["attempt_count"] and defaults["provider_request_id"] is None:
            defaults["provider_request_id"] = conversation_investigation_provider_request_id(
                investigation_id, defaults["attempt_count"]
            )
        investigation = ConversationInvestigation(
            id=investigation_id,
            response_id=response.id,
            goal_id=goal.id,
            actor_id=test_user.id,
            context_version=response.context_version,
            objective="investigate",
            scope=[],
            input_manifest={},
            provider_identity=conversation_investigation_provider_identity(investigation_id),
            **defaults,
        )
        return investigation, response

    accepted, response = await investigation_for(db_session, 1)
    reservation = ConversationInvestigationReservation(
        id=conversation_investigation_reservation_id(accepted.id),
        investigation_id=accepted.id,
        goal_id=goal.id,
        actor_id=test_user.id,
        ceiling_snapshot=10,
        reserved_tokens=10,
    )
    db_session.add_all((accepted, reservation))
    await db_session.flush()
    await db_session.commit()

    async with session_factory() as session:
        pending = await session.get(ConversationInvestigation, accepted.id)
        pending_response = await session.get(ConversationResponse, response.id)
        pending_reservation = await session.get(ConversationInvestigationReservation, reservation.id)
        assert (pending.status, pending_response.status, pending_reservation.status) == (
            "pending", "pending", "reserved",
        )
        pending.status = "failed"
        pending.error = {"code": "interrupted_before_dispatch"}
        pending.finished_at = now
        pending_response.status = "failed"
        pending_response.error = {"code": "investigation_interrupted"}
        pending_response.finished_at = now
        pending_reservation.status = "released"
        pending_reservation.released_tokens = pending_reservation.reserved_tokens
        pending_reservation.released_at = now
        await session.commit()

    async with session_factory() as session:
        failed = await session.get(ConversationInvestigation, accepted.id)
        failed_response = await session.get(ConversationResponse, response.id)
        released = await session.get(ConversationInvestigationReservation, reservation.id)
        assert (
            failed.status,
            failed.attempt_count,
            failed.repair_count,
            failed.retry_count,
            failed.accumulated_tokens,
            failed.provider_request_id,
            failed.started_at,
            failed.deadline_at,
            _utc(failed.finished_at),
            failed.error,
        ) == (
            "failed", 0, 0, 0, 0, None, None, None, now,
            {"code": "interrupted_before_dispatch"},
        )
        assert (
            failed_response.status,
            failed_response.error,
            failed_response.answer,
            _utc(failed_response.finished_at),
        ) == ("failed", {"code": "investigation_interrupted"}, None, now)
        assert (
            released.status,
            released.settled_tokens,
            released.released_tokens,
            _utc(released.released_at),
        ) == ("released", 0, released.reserved_tokens, now)

    async def commit_investigation(sequence, **values):
        async with session_factory() as session:
            investigation, _ = await investigation_for(session, sequence, **values)
            session.add(investigation)
            await session.commit()
            return investigation.id

    attempted_ids = []
    for sequence, attempt, repair in ((2, 1, 0), (3, 2, 1)):
        attempted_ids.append(await commit_investigation(
            sequence,
            status="failed",
            attempt_count=attempt,
            repair_count=repair,
            started_at=now,
            deadline_at=now,
            finished_at=now,
        ))
    async with session_factory() as session:
        assert [
            (await session.get(ConversationInvestigation, investigation_id)).attempt_count
            for investigation_id in attempted_ids
        ] == [1, 2]

    for sequence, (values, constraint) in enumerate(
        (
            ({"provider_request_id": "rally-chat-investigation:claimed:1"}, "lifecycle"),
            ({"started_at": now}, "lifecycle"),
            ({"deadline_at": now}, "lifecycle"),
            ({"repair_count": 1}, "counters"),
            ({"retry_count": 1}, "counters"),
            ({"accumulated_tokens": 1}, "lifecycle"),
            ({"finished_at": None}, "lifecycle"),
            ({"status": "completed", "finished_at": now}, "lifecycle"),
            ({"status": "interrupted_unknown", "finished_at": now}, "lifecycle"),
        ),
        start=4,
    ):
        with pytest.raises(IntegrityError, match=f"ck_orch_conversation_investigations_{constraint}"):
            await commit_investigation(sequence, **({"status": "failed", "finished_at": now} | values))


def test_migration_042_schema_allows_only_unclaimed_failed_investigations(tmp_path):
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import Session

    from huddleroom.models import (
        ConversationInvestigation,
        ConversationInvestigationReservation,
        OrchestrationGoal,
        Project,
        User,
    )
    from huddleroom.models.orchestration_conversation import (
        ConversationMessage,
        ConversationResponse,
        conversation_investigation_id,
        conversation_investigation_provider_identity,
        conversation_investigation_provider_request_id,
        conversation_investigation_reservation_id,
        conversation_response_id,
    )

    db_path = tmp_path / "failed-zero.db"
    env = {**os.environ, "RALLY_DATABASE_URL": f"sqlite+aiosqlite:///{db_path}"}
    _alembic(env, "upgrade", "041")
    _alembic(env, "upgrade", "042")
    now = datetime.now(timezone.utc)
    engine = create_engine(f"sqlite:///{db_path}")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys = ON")

    with Session(engine) as session:
        actor = User(email="failed-zero@example.test", hashed_password="secret")
        project = Project(name="Failed zero")
        session.add_all((actor, project))
        session.flush()
        goal = OrchestrationGoal(project_id=project.id, objective="Investigate")
        session.add(goal)
        session.flush()
        actor_id = actor.id
        goal_id = goal.id
        session.commit()

    def investigation_for(session, sequence, **values):
        message = ConversationMessage(
            goal_id=goal_id,
            actor_id=actor_id,
            client_request_id=uuid.uuid4(),
            sequence=sequence,
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
            provider_request_id=f"rally-chat:migrated-failed-zero-{sequence}",
        )
        session.add(response)
        session.flush()
        investigation_id = conversation_investigation_id(response.id, response.context_version)
        defaults = {
            "status": "pending",
            "attempt_count": 0,
            "repair_count": 0,
            "retry_count": 0,
            "accumulated_tokens": 0,
            "provider_request_id": None,
            "started_at": None,
            "deadline_at": None,
            "finished_at": None,
        }
        defaults.update(values)
        if defaults["attempt_count"] and defaults["provider_request_id"] is None:
            defaults["provider_request_id"] = conversation_investigation_provider_request_id(
                investigation_id, defaults["attempt_count"]
            )
        return ConversationInvestigation(
            id=investigation_id,
            response_id=response.id,
            goal_id=goal_id,
            actor_id=actor_id,
            context_version=response.context_version,
            objective="investigate",
            scope=[],
            input_manifest={},
            provider_identity=conversation_investigation_provider_identity(investigation_id),
            **defaults,
        ), response

    with Session(engine) as session:
        pending, response = investigation_for(session, 1)
        reservation = ConversationInvestigationReservation(
            id=conversation_investigation_reservation_id(pending.id),
            investigation_id=pending.id,
            goal_id=goal_id,
            actor_id=actor_id,
            ceiling_snapshot=10,
            reserved_tokens=10,
        )
        session.add_all((pending, reservation))
        pending_id = pending.id
        response_id = response.id
        reservation_id = reservation.id
        session.commit()

    with Session(engine) as session:
        pending = session.get(ConversationInvestigation, pending_id)
        pending_response = session.get(ConversationResponse, response_id)
        pending_reservation = session.get(ConversationInvestigationReservation, reservation_id)
        assert (pending.status, pending_response.status, pending_reservation.status) == (
            "pending", "pending", "reserved",
        )
        pending.status = "failed"
        pending.error = {"code": "interrupted_before_dispatch"}
        pending.finished_at = now
        pending_response.status = "failed"
        pending_response.error = {"code": "investigation_interrupted"}
        pending_response.finished_at = now
        pending_reservation.status = "released"
        pending_reservation.released_tokens = pending_reservation.reserved_tokens
        pending_reservation.released_at = now
        session.commit()

    with Session(engine) as session:
        failed = session.get(ConversationInvestigation, pending_id)
        failed_response = session.get(ConversationResponse, response_id)
        released = session.get(ConversationInvestigationReservation, reservation_id)
        assert (
            failed.status,
            failed.attempt_count,
            failed.repair_count,
            failed.retry_count,
            failed.accumulated_tokens,
            failed.provider_request_id,
            failed.started_at,
            failed.deadline_at,
            _utc(failed.finished_at),
            failed.error,
        ) == (
            "failed", 0, 0, 0, 0, None, None, None, now,
            {"code": "interrupted_before_dispatch"},
        )
        assert (
            failed_response.status,
            failed_response.error,
            failed_response.answer,
            _utc(failed_response.finished_at),
        ) == ("failed", {"code": "investigation_interrupted"}, None, now)
        assert (
            released.status,
            released.settled_tokens,
            released.released_tokens,
            _utc(released.released_at),
        ) == ("released", 0, released.reserved_tokens, now)

    attempted_ids = []
    for sequence, attempt, repair in ((2, 1, 0), (3, 2, 1)):
        with Session(engine) as session:
            investigation, _ = investigation_for(
                session,
                sequence,
                status="failed",
                attempt_count=attempt,
                repair_count=repair,
                started_at=now,
                deadline_at=now,
                finished_at=now,
            )
            session.add(investigation)
            attempted_ids.append(investigation.id)
            session.commit()

    with Session(engine) as session:
        assert [session.get(ConversationInvestigation, investigation_id).attempt_count
                for investigation_id in attempted_ids] == [1, 2]

    for sequence, (values, constraint) in enumerate(
        (
            ({"provider_request_id": "rally-chat-investigation:claimed:1"}, "lifecycle"),
            ({"started_at": now}, "lifecycle"),
            ({"deadline_at": now}, "lifecycle"),
            ({"repair_count": 1}, "counters"),
            ({"retry_count": 1}, "counters"),
            ({"accumulated_tokens": 1}, "lifecycle"),
            ({"finished_at": None}, "lifecycle"),
            ({"status": "completed", "finished_at": now}, "lifecycle"),
            ({"status": "interrupted_unknown", "finished_at": now}, "lifecycle"),
        ),
        start=4,
    ):
        with pytest.raises(IntegrityError, match=f"ck_orch_conversation_investigations_{constraint}"):
            with Session(engine) as session:
                investigation, _ = investigation_for(
                    session, sequence, **({"status": "failed", "finished_at": now} | values)
                )
                session.add(investigation)
                session.commit()
    engine.dispose()
