import asyncio
import json
import logging
import threading
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration import OrchestrationGoal
from huddleroom.models.orchestration_conversation import (
    ConversationInvestigation,
    ConversationInvestigationReservation,
    ConversationMessage,
    ConversationReservation,
    ConversationResponse,
    conversation_reservation_id,
    conversation_investigation_reservation_id,
)
from huddleroom.models.project import Project
from huddleroom.config import settings
from huddleroom.services.orchestration_conversation_dossier import ConversationDossierBuilder
from huddleroom.services.orchestration_conversation_investigation import (
    ConversationInvestigationService,
    REQUEST_INVESTIGATION_TOOL,
    parse_investigation_request,
)
from huddleroom.services.orchestration_conversation_service import (
    ConversationDomainError,
    OrchestrationConversationService,
)
from huddleroom.services.orchestration_service import OrchestrationService

# Tests intentionally exercise durable transition primitives directly.
# pylint: disable=protected-access

@pytest.fixture(autouse=True)
def conversation_enabled(monkeypatch):
    monkeypatch.setattr(
        "huddleroom.services.orchestration_conversation_service.settings.orchestration_conversation_allowance_tokens",
        10_000,
    )


def _factory(test_engine):
    return async_sessionmaker(test_engine, expire_on_commit=False)


async def _chat_demand(test_engine, goal, content):
    async with _factory(test_engine)() as db:
        saved_goal = await db.get(type(goal), goal.id)
        run = await OrchestrationService().get_run_for_goal(db, goal.project_id, goal.id)
        pairs = (await db.execute(
            select(ConversationMessage, ConversationResponse)
            .join(ConversationResponse)
            .where(ConversationMessage.goal_id == goal.id, ConversationResponse.status == "completed")
            .order_by(ConversationMessage.sequence)
        )).all()
        build = await ConversationDossierBuilder(db).build(saved_goal, run, content, pairs)
    return len(json.dumps(build.provider_messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")) + 64 + 800


async def _set_workspace(db_session, project_id, workspace):
    project = await db_session.get(Project, project_id)
    project.workspace_path = str(workspace.resolve())


def _tool_lookup_result():
    return {
        "choices": [{"message": {"content": None, "tool_calls": [{"function": {
            "name": "request_investigation",
            "arguments": json.dumps({
                "objective": "Check the release gate",
                "requests": [{"operation": "read", "path": "risk.txt", "query": None}],
            }),
        }}]}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }


def _service(test_engine, lookup=None):
    async def completion(**_kwargs):
        raise AssertionError("recovery must not dispatch completion")

    return OrchestrationConversationService(
        _factory(test_engine), completion, lookup, OrchestrationService()
    )


def _tracked_service(test_engine, completion, lookup):
    factory = _factory(test_engine)
    state = {
        "sessions": 0, "transactions": 0, "locks": 0,
        "session_entries": 0, "transaction_entries": 0, "lock_entries": 0,
    }

    class TrackedSession:
        def __init__(self, session):
            self._session = session

        def __getattr__(self, name):
            return getattr(self._session, name)

        def begin(self):
            @asynccontextmanager
            async def transaction_scope():
                state["transactions"] += 1
                state["transaction_entries"] += 1
                try:
                    async with self._session.begin() as transaction:
                        yield transaction
                finally:
                    state["transactions"] -= 1

            return transaction_scope()

    def tracked_factory():
        @asynccontextmanager
        async def session_scope():
            state["sessions"] += 1
            state["session_entries"] += 1
            try:
                async with factory() as session:
                    yield TrackedSession(session)
            finally:
                state["sessions"] -= 1

        return session_scope()

    tracked_factory.kw = factory.kw

    orchestration = OrchestrationService()
    original_lock = orchestration._lock_goal_for_baseline_transition

    @asynccontextmanager
    async def tracked_lock(db, goal_id):
        async with original_lock(db, goal_id):
            state["locks"] += 1
            state["lock_entries"] += 1
            try:
                yield
            finally:
                state["locks"] -= 1

    orchestration._lock_goal_for_baseline_transition = tracked_lock
    return OrchestrationConversationService(tracked_factory, completion, lookup, orchestration), state


async def _pending(service, goal, actor_id, *, abandoned=False):
    turn, created = await service._prepare(
        goal.project_id, goal.id, actor_id, uuid.uuid4(), "Question"
    )
    assert created is True
    if abandoned:
        async with _factory(service._session_factory.kw["bind"])() as db:
            response = await db.get(ConversationResponse, turn.response.id)
            response.created_at = _utcnow() - timedelta(seconds=121)
            await db.commit()
    return turn.response.id


async def _running(service, goal, actor_id):
    response_id = await _pending(service, goal, actor_id)
    assert await service._claim(goal.id, response_id)
    return response_id


async def _expired_enabled(service, goal, actor_id):
    turn, created = await service._prepare(
        goal.project_id, goal.id, actor_id, uuid.uuid4(), "Investigate gate"
    )
    assert created is True and await service._claim(goal.id, turn.response.id)
    async with _factory(service._session_factory.kw["bind"])() as db:
        response = await db.get(ConversationResponse, turn.response.id)
        response.deadline_at = _utcnow() - timedelta(seconds=1)
        await db.commit()
    return turn.response.id


async def _terminal_chat_snapshot(test_engine, response_id):
    def normalized(value):
        if value is None:
            return None
        return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)

    async with _factory(test_engine)() as db:
        response = await db.get(ConversationResponse, response_id)
        reservation = await db.get(ConversationReservation, conversation_reservation_id(response_id))
        return (
            response.id, response.status, response.answer, response.error,
            response.provider_request_id, response.context_version,
            normalized(response.started_at), normalized(response.deadline_at),
            normalized(response.finished_at), normalized(response.updated_at),
            reservation.id, reservation.goal_id, reservation.actor_id,
            reservation.ceiling_snapshot, reservation.reserved_tokens,
            reservation.status, reservation.settled_tokens, reservation.released_tokens,
            normalized(reservation.committed_at), normalized(reservation.settled_at),
            normalized(reservation.released_at), normalized(reservation.updated_at),
        )


def _snapshot_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if isinstance(value, datetime):
        return (value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)).isoformat()
    return value.isoformat() if hasattr(value, "isoformat") else value


async def _conversation_state(db):
    """All durable conversation rows, ordered and value-complete."""
    rows = []
    for model in (
        ConversationMessage,
        ConversationResponse,
        ConversationReservation,
        ConversationInvestigation,
        ConversationInvestigationReservation,
    ):
        records = (await db.scalars(select(model).order_by(model.id))).all()
        rows.append(tuple(
            tuple((column.name, _snapshot_value(getattr(record, column.name)))
                  for column in model.__table__.columns)
            for record in records
        ))
    return tuple(rows)


@pytest.mark.asyncio
async def test_recover_goal_releases_undispatched_pair(
    test_engine, conversation_goal_run, test_user, db_session
):
    goal, _ = conversation_goal_run
    await db_session.commit()
    service = _service(test_engine)
    response_id = await _pending(service, goal, test_user.id, abandoned=True)

    await service.recover_goal(goal.id)

    async with _factory(test_engine)() as db:
        response = await db.get(ConversationResponse, response_id)
        reservation = await db.get(
            ConversationReservation, conversation_reservation_id(response_id)
        )
        assert response.status == "failed"
        assert response.error == {"code": "interrupted_before_dispatch"}
        assert response.finished_at is not None
        assert (reservation.status, reservation.released_tokens) == (
            "released",
            reservation.reserved_tokens,
        )


@pytest.mark.asyncio
async def test_recover_goal_keeps_fresh_undispatched_pair(
    test_engine, conversation_goal_run, test_user, db_session
):
    goal, _ = conversation_goal_run
    await db_session.commit()
    service = _service(test_engine)
    response_id = await _pending(service, goal, test_user.id)

    await service.recover_goal(goal.id)

    async with _factory(test_engine)() as db:
        response = await db.get(ConversationResponse, response_id)
        reservation = await db.get(
            ConversationReservation, conversation_reservation_id(response_id)
        )
        assert (response.status, reservation.status) == ("pending", "reserved")


@pytest.mark.asyncio
async def test_recover_goal_leaves_nonexpired_running_pair_unchanged(
    test_engine, conversation_goal_run, test_user, db_session
):
    goal, _ = conversation_goal_run
    await db_session.commit()
    service = _service(test_engine)
    response_id = await _running(service, goal, test_user.id)

    await service.recover_goal(goal.id)

    async with _factory(test_engine)() as db:
        response = await db.get(ConversationResponse, response_id)
        reservation = await db.get(
            ConversationReservation, conversation_reservation_id(response_id)
        )
        assert (response.status, reservation.status) == ("running", "committed")


@pytest.mark.asyncio
async def test_pending_attempt_that_loses_release_race_is_not_looked_up(
    test_engine, conversation_goal_run, test_user, db_session, monkeypatch
):
    goal, _ = conversation_goal_run
    await db_session.commit()
    looked_up = []

    async def lookup(provider_request_id):
        looked_up.append(provider_request_id)
        return None

    service = _service(test_engine, lookup)
    await _pending(service, goal, test_user.id, abandoned=True)

    async def lost_release_race(_goal_id, _response_id):
        return False

    monkeypatch.setattr(service, "_release_interrupted", lost_release_race)
    await service.recover_goal(goal.id)

    assert not looked_up


@pytest.mark.asyncio
async def test_expired_recovery_looks_up_outside_session_and_lock_then_settles(
    test_engine, conversation_goal_run, test_user, db_session
):
    goal, _ = conversation_goal_run
    await db_session.commit()
    state = {"sessions": 0, "locks": 0}
    factory = _factory(test_engine)

    def tracked_factory():
        @asynccontextmanager
        async def scope():
            state["sessions"] += 1
            try:
                async with factory() as db:
                    yield db
            finally:
                state["sessions"] -= 1

        return scope()

    orchestration = OrchestrationService()
    original_lock = orchestration._lock_goal_for_baseline_transition

    @asynccontextmanager
    async def tracked_lock(db, goal_id):
        async with original_lock(db, goal_id):
            state["locks"] += 1
            try:
                yield
            finally:
                state["locks"] -= 1

    orchestration._lock_goal_for_baseline_transition = tracked_lock

    async def lookup(_provider_request_id):
        assert state == {"sessions": 0, "locks": 0}
        return {
            "choices": [{"message": {"content": "Recovered"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        }

    async def completion(**_kwargs):
        raise AssertionError("recovery must not dispatch completion")

    service = OrchestrationConversationService(
        tracked_factory, completion, lookup, orchestration
    )
    response_id = await _running(service, goal, test_user.id)
    async with factory() as db:
        response = await db.get(ConversationResponse, response_id)
        response.deadline_at = _utcnow() - timedelta(seconds=1)
        await db.commit()

    await service.recover_goal(goal.id)

    async with factory() as db:
        response = await db.get(ConversationResponse, response_id)
        assert (response.status, response.answer) == ("completed", "Recovered")


@pytest.mark.asyncio
async def test_expired_recovery_without_result_holds_unknown_and_terminal_rows_stay_immutable(
    test_engine, conversation_goal_run, test_user, db_session
):
    goal, _ = conversation_goal_run
    await db_session.commit()
    service = _service(test_engine)
    response_id = await _running(service, goal, test_user.id)
    async with _factory(test_engine)() as db:
        response = await db.get(ConversationResponse, response_id)
        response.deadline_at = _utcnow() - timedelta(seconds=1)
        await db.commit()

    await service.recover_goal(goal.id)
    await service.recover_goal(goal.id)

    async with _factory(test_engine)() as db:
        response = await db.get(ConversationResponse, response_id)
        reservation = await db.get(
            ConversationReservation, conversation_reservation_id(response_id)
        )
        assert response.status == "interrupted_unknown"
        assert response.error == {"code": "provider_outcome_unknown"}
        assert reservation.status == "held_unknown"


@pytest.mark.asyncio
async def test_goal_scoped_recovery_then_all_goal_recovery(
    test_engine, conversation_goal_run, test_user, db_session
):
    goal_a, _ = conversation_goal_run
    goal_b = OrchestrationGoal(
        project_id=goal_a.project_id,
        objective="Second conversation fixture goal",
        success_criteria=[],
        constraints={},
        budget={},
        created_by_user_id=test_user.id,
    )
    db_session.add(goal_b)
    await db_session.flush()
    await db_session.commit()
    service = _service(test_engine)
    response_a_id = await _pending(service, goal_a, test_user.id, abandoned=True)
    response_b_id = await _pending(service, goal_b, test_user.id, abandoned=True)

    await service.recover_goal(goal_a.id)

    async with _factory(test_engine)() as db:
        response_a = await db.get(ConversationResponse, response_a_id)
        response_b = await db.get(ConversationResponse, response_b_id)
        reservation_a = await db.get(
            ConversationReservation, conversation_reservation_id(response_a_id)
        )
        reservation_b = await db.get(
            ConversationReservation, conversation_reservation_id(response_b_id)
        )
        assert (response_a.status, reservation_a.status) == ("failed", "released")
        assert (response_b.status, reservation_b.status) == ("pending", "reserved")

    response_a_retry_id = await _pending(service, goal_a, test_user.id, abandoned=True)
    await service.recover_all()

    async with _factory(test_engine)() as db:
        response_a_retry = await db.get(ConversationResponse, response_a_retry_id)
        response_b = await db.get(ConversationResponse, response_b_id)
        reservation_a_retry = await db.get(
            ConversationReservation,
            conversation_reservation_id(response_a_retry_id),
        )
        reservation_b = await db.get(
            ConversationReservation, conversation_reservation_id(response_b_id)
        )
        assert (response_a_retry.status, reservation_a_retry.status) == (
            "failed",
            "released",
        )
        assert (response_b.status, reservation_b.status) == ("failed", "released")


@pytest.mark.asyncio
async def test_enabled_response_recovery_uses_persisted_snapshot_after_setting_disables(
    test_engine, conversation_goal_run, test_user, db_session, tmp_path, monkeypatch
):
    goal, _ = conversation_goal_run
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "risk.txt").write_text("gate pending\n", encoding="utf-8")
    await _set_workspace(db_session, goal.project_id, workspace)
    await db_session.commit()
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", 50_000)
    monkeypatch.setattr(settings, "orchestration_conversation_investigation_enabled", True)
    calls, lookups = [], []

    async def completion(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": json.dumps({
            "findings": "Gate pending.", "uncertainty": "", "sources": ["risk.txt#L1-L1"],
        })}}], "usage": {"prompt_tokens": 2, "completion_tokens": 2}}

    async def lookup(provider_request_id):
        lookups.append(provider_request_id)
        return _tool_lookup_result()

    service = OrchestrationConversationService(
        _factory(test_engine), completion, lookup, OrchestrationService()
    )
    pending, created = await service._prepare(
        goal.project_id, goal.id, test_user.id, uuid.uuid4(), "Investigate gate"
    )
    assert created is True
    assert pending.response.dossier["_conversation_runtime"] == {"investigation_enabled": True}
    assert await service._claim(goal.id, pending.response.id)
    async with _factory(test_engine)() as db:
        response = await db.get(ConversationResponse, pending.response.id)
        response.deadline_at = _utcnow() - timedelta(seconds=1)
        await db.commit()
    monkeypatch.setattr(settings, "orchestration_conversation_investigation_enabled", False)

    await service.recover_goal(goal.id)
    await service.recover_goal(goal.id)

    async with _factory(test_engine)() as db:
        response = await db.get(ConversationResponse, pending.response.id)
        investigations = (await db.scalars(select(ConversationInvestigation))).all()
        assert (response.status, response.answer) == ("completed", "Gate pending.")
        assert len(investigations) == 1
        assert investigations[0].response_id == pending.response.id
    assert lookups == [pending.response.provider_request_id]
    assert len(calls) == 1
    assert "tools" not in calls[0] and "tool_choice" not in calls[0]


@pytest.mark.asyncio
async def test_enabled_chat_recovery_lookup_timeout_holds_persisted_authority_once(
    test_engine, conversation_goal_run, test_user, db_session, monkeypatch,
):
    """A recovery lookup has the same finite unknown boundary as live enabled Chat."""
    goal, _ = conversation_goal_run
    await db_session.commit()
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", 50_000)
    monkeypatch.setattr(settings, "orchestration_conversation_investigation_enabled", True)
    import huddleroom.services.orchestration_conversation_investigation as investigation_module
    monkeypatch.setattr(investigation_module, "_PROVIDER_LOOKUP_TIMEOUT_SECONDS", 0.05, raising=False)
    entered, release, exited = asyncio.Event(), asyncio.Event(), asyncio.Event()
    lookups, state = [], None

    async def lookup(provider_request_id):
        assert {key: state[key] for key in ("sessions", "transactions", "locks")} == {
            "sessions": 0, "transactions": 0, "locks": 0,
        }
        lookups.append(provider_request_id)
        entered.set()
        try:
            await release.wait()
        finally:
            exited.set()
        return _tool_lookup_result()

    async def completion(**_kwargs):
        raise AssertionError("recovery must not dispatch completion")

    service, state = _tracked_service(test_engine, completion, lookup)
    response_id = await _expired_enabled(service, goal, test_user.id)
    expected_demand = await _chat_demand(test_engine, goal, "Investigate gate")
    async with _factory(test_engine)() as db:
        before = await db.get(ConversationResponse, response_id)
        authority = (before.provider_request_id, before.context_version, before.dossier)
    task = asyncio.create_task(service.recover_goal(goal.id))
    try:
        await asyncio.wait_for(entered.wait(), timeout=0.3)
        await asyncio.wait_for(task, timeout=0.3)
        assert exited.is_set()
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.3)
        assert exited.is_set()
    async with _factory(test_engine)() as db:
        response = await db.get(ConversationResponse, response_id)
        reservation = await db.get(ConversationReservation, conversation_reservation_id(response_id))
        assert response.provider_request_id == authority[0]
        assert lookups == [authority[0]]
        assert (response.provider_request_id, response.context_version, response.dossier) == authority
        assert (response.status, response.answer, response.error) == (
            "interrupted_unknown", None, {"code": "provider_outcome_unknown"},
        )
        assert (
            reservation.id, reservation.response_id, reservation.goal_id, reservation.actor_id,
            reservation.reserved_tokens, reservation.ceiling_snapshot,
            reservation.status, reservation.settled_tokens, reservation.released_tokens,
        ) == (
            conversation_reservation_id(response_id), response_id, goal.id, test_user.id,
            expected_demand, 50_000, "held_unknown", 0, 0,
        )
    await service.recover_goal(goal.id)
    assert len(lookups) == 1
    assert all(state[name] > 0 for name in ("session_entries", "transaction_entries", "lock_entries"))


@pytest.mark.parametrize("kind", ("investigation", "invalid", "unknown", "invalid_untrusted"))
@pytest.mark.asyncio
async def test_enabled_response_lookup_classifies_with_immutable_identity_and_accounting(
    kind, test_engine, conversation_goal_run, test_user, db_session, tmp_path, monkeypatch,
):
    """Recovery must use the prepared enabled authority, never a fresh Chat call."""
    goal, _ = conversation_goal_run
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "risk.txt").write_text("gate pending\n", encoding="utf-8")
    await _set_workspace(db_session, goal.project_id, workspace)
    await db_session.commit()
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", 50_000)
    monkeypatch.setattr(settings, "orchestration_conversation_investigation_enabled", True)
    lookup_ids, completion_ids, lookup_snapshots, completion_boundaries = [], [], [], []

    def raw():
        if kind == "investigation":
            return _tool_lookup_result()
        if kind == "unknown":
            return {"choices": [{"message": {"content": " ", "tool_calls": []}}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
        usage = {"prompt_tokens": "bad", "completion_tokens": 2} if kind == "invalid_untrusted" else {"prompt_tokens": 3, "completion_tokens": 2}
        return {"choices": [{"message": {"content": "also answer", "tool_calls": [{"function": {
            "name": "request_investigation", "arguments": "{}",
        }}]}}], "usage": usage}

    async def lookup(provider_request_id):
        lookup_ids.append(provider_request_id)
        async with _factory(test_engine)() as db:
            response = await db.get(ConversationResponse, response_id)
            message = await db.get(ConversationMessage, response.message_id)
            lookup_snapshots.append((
                {key: state[key] for key in ("sessions", "transactions", "locks")},
                response.provider_request_id, message.goal_id, message.actor_id,
                response.context_version, response.dossier.get("_conversation_runtime"),
            ))
        return raw()

    async def completion(**kwargs):
        completion_ids.append(kwargs["litellm_call_id"])
        completion_boundaries.append(
            {key: state[key] for key in ("sessions", "transactions", "locks")}
        )
        return {"choices": [{"message": {"content": json.dumps({
            "findings": "Gate pending.", "uncertainty": "", "sources": ["risk.txt#L1-L1"],
        })}}], "usage": {"prompt_tokens": 2, "completion_tokens": 2}}

    service, state = _tracked_service(test_engine, completion, lookup)
    prepared, created = await service._prepare(
        goal.project_id, goal.id, test_user.id, uuid.uuid4(), "Investigate gate"
    )
    assert created is True
    response_id = prepared.response.id
    assert await service._claim(goal.id, response_id)
    async with _factory(test_engine)() as db:
        response = await db.get(ConversationResponse, response_id)
        response.deadline_at = _utcnow() - timedelta(seconds=1)
        await db.commit()
    await service.recover_goal(goal.id)
    await service.recover_goal(goal.id)
    await service.recover_all()
    async with _factory(test_engine)() as db:
        response = await db.get(ConversationResponse, response_id)
        reservation = await db.get(ConversationReservation, conversation_reservation_id(response_id))
        investigation = await db.scalar(select(ConversationInvestigation))
        if kind == "investigation":
            investigation_reservation = await db.get(
                ConversationInvestigationReservation,
                conversation_investigation_reservation_id(investigation.id),
            )
            assert (response.status, response.answer, reservation.status, reservation.settled_tokens) == (
                "completed", "Gate pending.", "settled", 5,
            )
            assert (investigation.status, investigation.response_id, investigation.goal_id, investigation.actor_id) == (
                "completed", response_id, goal.id, test_user.id,
            )
            assert (investigation_reservation.status, investigation_reservation.settled_tokens) == ("settled", 4)
            assert completion_ids == [investigation.provider_request_id]
        else:
            expected_status = "interrupted_unknown" if kind == "unknown" else "failed"
            expected_error = {"code": "provider_outcome_unknown"} if kind == "unknown" else {"code": "invalid_investigation_request"}
            expected_reservation = "settled" if kind == "invalid" else "held_unknown"
            assert (response.status, response.answer, response.error, reservation.status) == (
                expected_status, None, expected_error, expected_reservation,
            )
            if kind == "invalid":
                assert reservation.settled_tokens == 5
                assert reservation.released_tokens == reservation.reserved_tokens - 5
            assert investigation is None and completion_ids == []
    assert lookup_ids == [prepared.response.provider_request_id]
    assert lookup_snapshots == [(
        {"sessions": 0, "transactions": 0, "locks": 0},
        prepared.response.provider_request_id, goal.id, test_user.id,
        prepared.response.context_version, {"investigation_enabled": True},
    )]
    assert completion_boundaries == ([{"sessions": 0, "transactions": 0, "locks": 0}]
                                      if kind == "investigation" else [])
    assert all(state[name] > 0 for name in ("session_entries", "transaction_entries", "lock_entries"))


@pytest.mark.parametrize("after_collection", (False, True))
@pytest.mark.asyncio
async def test_concurrent_recovery_lost_tool_current_keeps_terminal_winner_and_delegates(
    after_collection, test_engine, conversation_goal_run, test_user, db_session, tmp_path, monkeypatch,
):
    """A terminal sibling must win over a stale valid tool continuation."""
    goal, _ = conversation_goal_run
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "risk.txt").write_text("gate pending\n", encoding="utf-8")
    await _set_workspace(db_session, goal.project_id, workspace)
    await db_session.commit()
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", 50_000)
    monkeypatch.setattr(settings, "orchestration_conversation_investigation_enabled", True)
    lookup_entered, lookup_release = asyncio.Event(), asyncio.Event()
    reader_entered, reader_release, reader_exited = threading.Event(), threading.Event(), threading.Event()
    lookup_ids, recovery_calls, investigator_calls = [], [], []

    async def lookup(provider_request_id):
        assert all(state[name] == 0 for name in ("sessions", "transactions", "locks"))
        lookup_ids.append(provider_request_id)
        if len(lookup_ids) == 1:
            if after_collection:
                return _tool_lookup_result()
            lookup_entered.set()
            await lookup_release.wait()
            return _tool_lookup_result()
        return None

    async def completion(**kwargs):
        investigator_calls.append(kwargs)
        raise AssertionError("lost-current recovery must not call investigator")

    service, state = _tracked_service(test_engine, completion, lookup)
    real = service._investigations
    reader = real._reader

    class RecordingInvestigationRecovery:
        def __getattr__(self, name):
            return getattr(real, name)

        async def recover_goal(self, goal_id):
            recovery_calls.append(goal_id)
            return await real.recover_goal(goal_id)

    class BlockingReader:
        def collect(self, root, request):
            try:
                reader_entered.set()
                assert reader_release.wait(timeout=1)
                return reader.collect(root, request)
            finally:
                reader_exited.set()

    if after_collection:
        real._reader = BlockingReader()
    service._investigations = RecordingInvestigationRecovery()
    response_id = await _expired_enabled(service, goal, test_user.id)
    tool = asyncio.create_task(service.recover_goal(goal.id))
    winner = None
    try:
        if after_collection:
            assert await asyncio.to_thread(reader_entered.wait, 1)
        else:
            await asyncio.wait_for(lookup_entered.wait(), timeout=1)
        winner = asyncio.create_task(service.recover_goal(goal.id))
        await asyncio.wait_for(winner, timeout=1)
        winner_snapshot = await _terminal_chat_snapshot(test_engine, response_id)
        assert winner_snapshot[1:4] == ("interrupted_unknown", None, {"code": "provider_outcome_unknown"})
        assert winner_snapshot[15:18] == ("held_unknown", 0, 0)
        if after_collection:
            reader_release.set()
        else:
            lookup_release.set()
        outcomes = await asyncio.wait_for(
            asyncio.gather(tool, winner, return_exceptions=True), timeout=1
        )
        assert all(state[name] > 0 for name in ("session_entries", "transaction_entries", "lock_entries"))
        assert await _terminal_chat_snapshot(test_engine, response_id) == winner_snapshot
        async with _factory(test_engine)() as db:
            assert await db.scalar(select(ConversationInvestigation)) is None
            assert await db.scalar(select(ConversationInvestigationReservation)) is None
        await asyncio.wait_for(service.recover_goal(goal.id), timeout=1)
        assert await _terminal_chat_snapshot(test_engine, response_id) == winner_snapshot
        assert lookup_ids == [winner_snapshot[4], winner_snapshot[4]]
        assert investigator_calls == []
        assert outcomes == [None, None]
        assert recovery_calls == [goal.id, goal.id, goal.id]
    finally:
        lookup_release.set()
        reader_release.set()
        try:
            tasks = [task for task in (tool, winner) if task is not None]
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=1)
        finally:
            if after_collection:
                assert await asyncio.to_thread(reader_exited.wait, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ("context_version", "provider_request_id", "dossier"))
@pytest.mark.parametrize("during_collection", (False, True))
async def test_recovery_rethrows_real_ineligible_tool_continuation_without_terminal_winner(
    field, during_collection, test_engine, conversation_goal_run, test_user, db_session, tmp_path, monkeypatch,
):
    goal, _ = conversation_goal_run
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "risk.txt").write_text("gate pending\n", encoding="utf-8")
    await _set_workspace(db_session, goal.project_id, workspace)
    await db_session.commit()
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", 50_000)
    monkeypatch.setattr(settings, "orchestration_conversation_investigation_enabled", True)
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    lookup_entered, lookup_release = asyncio.Event(), asyncio.Event()
    lookup_ids, investigator_calls, state = [], [], None

    async def lookup(provider_request_id):
        assert all(state[name] == 0 for name in ("sessions", "transactions", "locks"))
        lookup_ids.append(provider_request_id)
        if not during_collection:
            lookup_entered.set()
            await lookup_release.wait()
        return _tool_lookup_result()

    async def completion(**kwargs):
        investigator_calls.append(kwargs)
        raise AssertionError("ineligible recovery must not dispatch investigator")

    service, state = _tracked_service(test_engine, completion, lookup)
    real = service._investigations
    reader = real._reader

    class BlockingReader:
        def collect(self, root, request):
            try:
                entered.set()
                assert release.wait(timeout=1)
                return reader.collect(root, request)
            finally:
                exited.set()

    if during_collection:
        real._reader = BlockingReader()
    response_id = await _expired_enabled(service, goal, test_user.id)
    recovery = asyncio.create_task(service.recover_goal(goal.id))
    try:
        if during_collection:
            assert await asyncio.to_thread(entered.wait, 1)
        else:
            await asyncio.wait_for(lookup_entered.wait(), timeout=1)
        async with _factory(test_engine)() as db:
            response = await db.get(ConversationResponse, response_id)
            captured = (
                response.id, response.context_version, response.provider_request_id,
                json.dumps(response.dossier, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
            )
            if field == "context_version":
                response.context_version = "drifted-context"
            elif field == "provider_request_id":
                response.provider_request_id = "drifted-provider-request"
            else:
                response.dossier = {**response.dossier, "drifted": True}
            await db.commit()
            drifted = await _conversation_state(db)
        if during_collection:
            release.set()
        else:
            lookup_release.set()
        with pytest.raises(ConversationDomainError, match="conversation_investigation_ineligible"):
            await asyncio.wait_for(recovery, timeout=1)
        assert all(state[name] > 0 for name in ("session_entries", "transaction_entries", "lock_entries"))
        async with _factory(test_engine)() as db:
            response = await db.get(ConversationResponse, response_id)
            reservation = await db.get(ConversationReservation, conversation_reservation_id(response_id))
            assert captured[0] == response.id
            assert captured[2].startswith("rally-chat:")
            assert json.loads(captured[3])["_conversation_runtime"] == {"investigation_enabled": True}
            assert (
                response.context_version, response.provider_request_id,
                json.dumps(response.dossier, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
            ) != captured[1:]
            assert await _conversation_state(db) == drifted
            assert (reservation.status, reservation.settled_tokens, reservation.released_tokens) == (
                "committed", 0, 0,
            )
            assert await db.scalar(select(ConversationInvestigation)) is None
            assert await db.scalar(select(ConversationInvestigationReservation)) is None
        assert lookup_ids == [captured[2]]
        assert investigator_calls == []
    finally:
        release.set()
        lookup_release.set()
        try:
            if not recovery.done():
                recovery.cancel()
                with suppress(asyncio.CancelledError):
                    await asyncio.wait_for(recovery, timeout=1)
        finally:
            if during_collection:
                assert await asyncio.to_thread(exited.wait, 1)


@pytest.mark.asyncio
async def test_recover_all_continues_after_lost_current_and_runs_global_investigation_recovery(
    test_engine, conversation_goal_run, test_user, db_session, monkeypatch,
):
    goal, _ = conversation_goal_run
    other_goal = OrchestrationGoal(
        project_id=goal.project_id, objective="Other recovery goal", success_criteria=[],
        constraints={}, budget={}, created_by_user_id=test_user.id,
    )
    db_session.add(other_goal)
    await db_session.flush()
    await db_session.commit()
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", 50_000)
    monkeypatch.setattr(settings, "orchestration_conversation_investigation_enabled", True)
    entered, release = asyncio.Event(), asyncio.Event()
    goal_calls, global_calls = [], 0
    state = None

    async def completion(**_kwargs):
        raise AssertionError("lost-current recovery must not dispatch investigator")

    async def lookup(provider_request_id):
        assert all(state[name] == 0 for name in ("sessions", "transactions", "locks"))
        if provider_request_id == main_provider_request_id:
            if not entered.is_set():
                entered.set()
                await release.wait()
                return _tool_lookup_result()
            return None
        if provider_request_id == other_provider_request_id:
            return {"choices": [{"message": {"content": "Other recovered"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        raise AssertionError("unexpected provider request")

    service, state = _tracked_service(test_engine, completion, lookup)
    main_response_id = await _expired_enabled(service, goal, test_user.id)
    other_response_id = await _running(service, other_goal, test_user.id)
    async with _factory(test_engine)() as db:
        main = await db.get(ConversationResponse, main_response_id)
        other = await db.get(ConversationResponse, other_response_id)
        main_provider_request_id = main.provider_request_id
        other_provider_request_id = other.provider_request_id
        other.deadline_at = _utcnow() - timedelta(seconds=1)
        await db.commit()
    real = service._investigations

    class RecordingRecovery:
        def __getattr__(self, name):
            return getattr(real, name)

        async def recover_goal(self, goal_id):
            goal_calls.append(goal_id)
            return await real.recover_goal(goal_id)

        async def recover_all(self):
            nonlocal global_calls
            global_calls += 1
            return await real.recover_all()

    service._investigations = RecordingRecovery()
    global_recovery = asyncio.create_task(service.recover_all())
    winner = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        winner = asyncio.create_task(service.recover_goal(goal.id))
        await asyncio.wait_for(winner, timeout=1)
        winner_snapshot = await _terminal_chat_snapshot(test_engine, main_response_id)
        release.set()
        outcomes = await asyncio.wait_for(
            asyncio.gather(global_recovery, winner, return_exceptions=True), timeout=1
        )
        assert all(state[name] > 0 for name in ("session_entries", "transaction_entries", "lock_entries"))
        async with _factory(test_engine)() as db:
            other = await db.get(ConversationResponse, other_response_id)
            assert (other.status, other.answer) == ("completed", "Other recovered")
        assert await _terminal_chat_snapshot(test_engine, main_response_id) == winner_snapshot
        assert outcomes == [None, None]
        assert global_calls == 1
        assert goal_calls.count(goal.id) >= 2 and other_goal.id in goal_calls
    finally:
        release.set()
        tasks = [task for task in (global_recovery, winner) if task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=1)


@pytest.mark.asyncio
async def test_disabled_historical_response_never_accepts_recovered_tool_after_enable(
    test_engine, conversation_goal_run, test_user, db_session, monkeypatch
):
    goal, _ = conversation_goal_run
    await db_session.commit()
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", 10_000)
    monkeypatch.setattr(settings, "orchestration_conversation_investigation_enabled", False)
    looked_up = []

    async def lookup(provider_request_id):
        looked_up.append(provider_request_id)
        return _tool_lookup_result()

    service = _service(test_engine, lookup)
    pending, created = await service._prepare(
        goal.project_id, goal.id, test_user.id, uuid.uuid4(), "Question"
    )
    assert created is True
    assert "_conversation_runtime" not in pending.response.dossier
    assert await service._claim(goal.id, pending.response.id)
    async with _factory(test_engine)() as db:
        response = await db.get(ConversationResponse, pending.response.id)
        response.deadline_at = _utcnow() - timedelta(seconds=1)
        await db.commit()
    monkeypatch.setattr(settings, "orchestration_conversation_investigation_enabled", True)

    await service.recover_goal(goal.id)
    async with _factory(test_engine)() as db:
        response = await db.get(ConversationResponse, pending.response.id)
        assert (response.status, response.error) == ("interrupted_unknown", {"code": "provider_outcome_unknown"})
        assert await db.scalar(select(ConversationInvestigation)) is None
    assert len(looked_up) == 1


@pytest.mark.asyncio
async def test_provider_exception_lookup_continues_enabled_tool_trigger_without_chat_retry(
    test_engine, conversation_goal_run, test_user, db_session, tmp_path, monkeypatch
):
    goal, _ = conversation_goal_run
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "risk.txt").write_text("gate pending\n", encoding="utf-8")
    await _set_workspace(db_session, goal.project_id, workspace)
    await db_session.commit()
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", 50_000)
    monkeypatch.setattr(settings, "orchestration_conversation_investigation_enabled", True)
    calls, lookups = [], []

    async def completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise TimeoutError("ambiguous chat delivery")
        return {"choices": [{"message": {"content": json.dumps({
            "findings": "Gate pending.", "uncertainty": "", "sources": ["risk.txt#L1-L1"],
        })}}], "usage": {"prompt_tokens": 2, "completion_tokens": 2}}

    async def lookup(provider_request_id):
        lookups.append(provider_request_id)
        return _tool_lookup_result()

    turn = await OrchestrationConversationService(
        _factory(test_engine), completion, lookup, OrchestrationService()
    ).submit(goal.project_id, goal.id, test_user.id, uuid.uuid4(), "Investigate gate")
    assert (turn.response.status, turn.response.answer) == ("completed", "Gate pending.")
    assert turn.investigation.report["findings"] == "Gate pending."
    assert len(lookups) == 1 and len(calls) == 2
    assert calls[0]["tools"] == [REQUEST_INVESTIGATION_TOOL]
    assert "tools" not in calls[1]


@pytest.mark.asyncio
async def test_response_recovery_always_delegates_investigation_recovery(
    test_engine, conversation_goal_run, test_user, db_session
):
    goal, _ = conversation_goal_run
    await db_session.commit()
    service = _service(test_engine)

    class InvestigationRecovery:
        def __init__(self):
            self.goal_ids = []

        async def recover_goal(self, goal_id):
            self.goal_ids.append(goal_id)

    investigation_recovery = InvestigationRecovery()
    service._investigations = investigation_recovery
    await service.recover_goal(goal.id)
    assert investigation_recovery.goal_ids == [goal.id]


@pytest.mark.asyncio
async def test_recover_all_delegates_investigation_recovery_without_response_candidates(
    test_engine, conversation_goal_run, test_user, db_session
):
    goal, _ = conversation_goal_run
    await db_session.commit()
    service = _service(test_engine)

    class InvestigationRecovery:
        def __init__(self):
            self.calls = 0

        async def recover_all(self):
            self.calls += 1

    investigation_recovery = InvestigationRecovery()
    service._investigations = investigation_recovery
    await service.recover_all()
    assert investigation_recovery.calls == 1


@pytest.mark.asyncio
async def test_recovery_adopts_existing_expired_investigation_through_real_investigation_service(
    test_engine, conversation_goal_run, test_user, db_session, tmp_path, monkeypatch
):
    goal, _ = conversation_goal_run
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "risk.txt").write_text("gate pending\n", encoding="utf-8")
    await _set_workspace(db_session, goal.project_id, workspace)
    await db_session.commit()
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", 50_000)

    async def seed_completion(**_kwargs):
        return {"choices": [{"message": {"content": "seed"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    seed = await OrchestrationConversationService(
        _factory(test_engine), seed_completion, orchestration_service=OrchestrationService()
    ).submit(goal.project_id, goal.id, test_user.id, uuid.uuid4(), "Seed")
    async with _factory(test_engine)() as db:
        response = await db.get(ConversationResponse, seed.response.id)
        response.status, response.answer, response.finished_at = "running", None, None
        response.deadline_at = _utcnow() + timedelta(seconds=120)
        await db.commit()
    request = parse_investigation_request(json.dumps({
        "objective": "Inspect", "requests": [{"operation": "read", "path": "risk.txt", "query": None}],
    }))
    lookups = []

    async def lookup(provider_request_id):
        lookups.append(provider_request_id)
        return {"choices": [{"message": {"content": json.dumps({
            "findings": "Gate pending.", "uncertainty": "", "sources": ["risk.txt#L1-L1"],
        })}}], "usage": {"prompt_tokens": 2, "completion_tokens": 2}}

    investigator = ConversationInvestigationService(
        _factory(test_engine), lambda **_kwargs: None, lookup, OrchestrationService()
    )
    prepared, _ = await investigator._prepare(
        goal.project_id, goal.id, test_user.id, seed.response.id, seed.response.context_version, request
    )
    claimed = await investigator._claim(goal.id, prepared.id, repair=False)
    async with _factory(test_engine)() as db:
        row = await db.get(ConversationInvestigation, claimed.id)
        chat_reservation = await db.get(
            ConversationReservation, conversation_reservation_id(seed.response.id)
        )
        assert (chat_reservation.status, chat_reservation.settled_tokens) == ("settled", 2)
        row.deadline_at = _utcnow() - timedelta(seconds=1)
        await db.commit()
    conversation = _service(test_engine)
    conversation._investigations = investigator
    await conversation.recover_goal(goal.id)
    async with _factory(test_engine)() as db:
        row = await db.get(ConversationInvestigation, claimed.id)
        response = await db.get(ConversationResponse, seed.response.id)
        snapshot = (
            row.id, row.response_id, row.goal_id, row.actor_id, row.context_version,
            row.status, row.report, response.status, response.answer, response.error,
        )
        assert snapshot == (
            claimed.id, seed.response.id, goal.id, test_user.id, seed.response.context_version,
            "completed", {"findings": "Gate pending.", "uncertainty": "", "sources": ["risk.txt#L1-L1"]},
            "completed", "Gate pending.", None,
        )
    await conversation.recover_all()
    async with _factory(test_engine)() as db:
        row = await db.get(ConversationInvestigation, claimed.id)
        response = await db.get(ConversationResponse, seed.response.id)
        assert (
            row.id, row.response_id, row.goal_id, row.actor_id, row.context_version,
            row.status, row.report, response.status, response.answer, response.error,
        ) == snapshot
    assert lookups == [claimed.provider_request_id]


@pytest.mark.asyncio
async def test_lifespan_logs_conversation_recovery_failure_and_starts(monkeypatch, caplog):
    import sqlalchemy.ext.asyncio
    from huddleroom import main
    import huddleroom.routers.orchestration_goals as goals_router
    import huddleroom.services.agent_response_relay as relay

    class DbCheck:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def begin(self):
            return self

        async def execute(self, *_args, **_kwargs):
            return None

    class BrokenRecovery:
        async def recover_all(self):
            raise RuntimeError("recovery broke")

    ready = asyncio.Event()

    async def hub(_ready):
        _ready.set()
        ready.set()
        await asyncio.Future()

    monkeypatch.setattr(main, "settings", type("Settings", (), {
        "auth_enabled": False,
        "is_sqlite": True,
        "jwt_secret": "safe",
        "cors_origins": ["https://example.test"],
    })())
    monkeypatch.setattr(main, "validate_supported_settings", lambda _settings: None)
    monkeypatch.setattr(relay, "settings", type("RelaySettings", (), {"is_sqlite": True})())
    monkeypatch.setattr(main, "init_db", lambda: asyncio.sleep(0))
    monkeypatch.setattr(main, "register_litellm_debug_logger", lambda: None)
    monkeypatch.setattr(sqlalchemy.ext.asyncio, "AsyncSession", DbCheck)
    monkeypatch.setattr(
        goals_router, "conversation_service", BrokenRecovery(), raising=False
    )
    monkeypatch.setattr(relay, "run_agent_response_hub", hub)
    monkeypatch.setattr(relay, "close_agent_response_relay", lambda: asyncio.sleep(0))
    async def idle():
        await asyncio.Future()
    consumer_tasks = {}
    monkeypatch.setattr("huddleroom.workers.scheduler.start_scheduler", lambda: object())
    monkeypatch.setattr("huddleroom.workers.scheduler.stop_scheduler", lambda: None)
    monkeypatch.setattr("huddleroom.workers.consumers.get_consumer_tasks", lambda: consumer_tasks)
    monkeypatch.setattr("huddleroom.workers.orchestration_tasks.run_orchestration_event_supervisor", idle)
    for module in ("ws_hub", "rule_engine", "protocol_engine", "meeting_engine", "optimizer"):
        monkeypatch.setattr(f"huddleroom.workers.consumers.{module}.run_{module}", idle)

    with caplog.at_level(logging.WARNING, logger="huddleroom.main"):
        async with main.lifespan(None):
            assert ready.is_set()
    assert "Conversation recovery failed" in caplog.text
