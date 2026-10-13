"""Every interpretation call retains its admitted scope and durable source identity."""
import json
import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from huddleroom.models.orchestration_chat import ProjectChatTurn
from huddleroom.config import settings
from huddleroom.models.orchestration_advisor import ProjectAdvisorTurn, advisor_allowance_used
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationAction
from huddleroom.services.orchestration_conversation_service import ConversationDomainError
from huddleroom.models.orchestration_conversation import (
    conversation_message_id, conversation_provider_request_id, conversation_response_id,
)
from tests.test_project_advisor_service import advisor_session, advisor_user, advisor_project

pytestmark = pytest.mark.asyncio


def _response(payload, *, usage=5):
    answer = payload if isinstance(payload, str) else json.dumps(payload)
    return {"choices": [{"message": {"content": json.dumps({
        "answer": answer, "citations": [], "off_topic": False,
    })}}], "usage": {"prompt_tokens": usage, "completion_tokens": usage}}


async def _new_turn(db, project_id, actor_id, content="What's the project state?", goal_id=None):
    request_id = uuid.uuid4()
    source_id = uuid.uuid4() if goal_id is None else conversation_response_id(
        conversation_message_id(goal_id, actor_id, request_id))
    row = ProjectChatTurn(
        project_id=project_id, actor_id=actor_id, client_request_id=request_id, content=content,
        context_goal_id=goal_id, source_kind="project_advisor" if goal_id is None else "goal_conversation",
        source_record_id=source_id,
        provider_request_id=f"project-chat:{source_id}" if goal_id is None else conversation_provider_request_id(source_id),
        provider_status="pending", deadline_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        accounting_snapshot={"scope": "project" if goal_id is None else "goal", "goal_id": str(goal_id) if goal_id else None},
    )
    db.add(row)
    await db.commit()
    return row


def _service(engine, completion):
    from huddleroom.services.orchestration_chat_interpretation import OrchestrationChatInterpretationService
    return OrchestrationChatInterpretationService(
        session_factory=async_sessionmaker(engine, expire_on_commit=False), completion_fn=completion,
    )


async def test_admission_counts_calls_and_allows_prepared_confirmation(
    test_engine, advisor_session, advisor_project, advisor_user, monkeypatch
):
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 100_000)
    turn = await _new_turn(advisor_session, advisor_project.id, advisor_user.id)
    calls = []
    factory = async_sessionmaker(test_engine, expire_on_commit=False)

    async def completion(**kwargs):
        async with factory() as db:
            stored = await db.get(ProjectChatTurn, turn.id)
            assert stored.source_record_id == turn.source_record_id
            assert stored.provider_request_id == turn.provider_request_id
            assert stored.provider_status == "running"
            assert stored.accounting_snapshot["scope"] == "project_advisor"
            assert stored.accounting_snapshot["calls_admitted"] == len(calls) + 1
            assert stored.deadline_at is not None
        calls.append(kwargs)
        if len(calls) == 1:
            return {"choices": [{"message": {"content": "not-json"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 5}}
        return _response({"intent": "question", "answer": "Current project state"})

    service = _service(test_engine, completion)
    result = await service.interpret(turn.id)
    assert result.intent == "question" and result.operation is None
    assert len(calls) == 2
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 0)
    replay = await service.interpret(turn.id)
    assert replay == result and len(calls) == 2
    async with factory() as db:
        source = await db.get(ProjectAdvisorTurn, turn.source_record_id)
        assert source.status == "completed" and source.tokens_used == 20
        assert await advisor_allowance_used(db, turn.project_id, turn.actor_id) == 20
        assert await db.scalar(select(func.count()).select_from(ProjectAdvisorTurn).where(
            ProjectAdvisorTurn.project_id == turn.project_id)) == 1
        stored = await db.get(ProjectChatTurn, turn.id)
        assert stored.provider_status == "completed"
        assert stored.accounting_snapshot["tokens_observed"] == source.tokens_used
        assert stored.accounting_snapshot["unknown_usage"] is False


@pytest.mark.parametrize("kind", ["disabled", "exhausted", "unknown"])
async def test_project_disabled_exhausted_unknown_prevent_new_calls(
    kind, test_engine, advisor_session, advisor_project, advisor_user, monkeypatch
):
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 0 if kind == "disabled" else 1000)
    if kind != "disabled":
        advisor_session.add(ProjectAdvisorTurn(project_id=advisor_project.id, actor_id=advisor_user.id,
            question="old", answer="old", status="completed", tokens_used=None if kind == "unknown" else 1000))
        await advisor_session.commit()
    turn = await _new_turn(advisor_session, advisor_project.id, advisor_user.id)
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        raise AssertionError("provider must not be called without allowance")

    try:
        result = await _service(test_engine, completion).interpret(turn.id)
        assert result.operation is None
    except ConversationDomainError:
        pass
    assert calls == []


async def test_unknown_project_provider_outcome_is_never_redispatched(
    test_engine, advisor_session, advisor_project, advisor_user, monkeypatch
):
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 100_000)
    turn = await _new_turn(advisor_session, advisor_project.id, advisor_user.id)
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("unverifiable provider outcome")

    service = _service(test_engine, completion)
    for _ in range(2):
        try:
            result = await service.interpret(turn.id)
            assert result.operation is None
        except (RuntimeError, ConversationDomainError):
            pass
    assert len(calls) == 1
    async with async_sessionmaker(test_engine, expire_on_commit=False)() as db:
        stored = await db.get(ProjectChatTurn, turn.id)
        assert stored.provider_status in {"failed", "interrupted_unknown"}
        assert await advisor_allowance_used(db, turn.project_id, turn.actor_id) is None


async def test_project_followups_use_eight_completed_pairs_and_fresh_context(
    test_engine, advisor_session, advisor_project, advisor_user, monkeypatch
):
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 100_000)
    now = datetime(2026, 10, 13, tzinfo=timezone.utc)
    for i in range(10):
        advisor_session.add(ProjectChatTurn(project_id=advisor_project.id, actor_id=advisor_user.id,
            client_request_id=uuid.uuid4(), content=f"question-{i}", answer=f"answer-{i}",
            status="completed", created_at=now + timedelta(seconds=i)))
    advisor_session.add(ProjectChatTurn(project_id=advisor_project.id, actor_id=advisor_user.id,
        client_request_id=uuid.uuid4(), content="unfinished-exclude", answer="unfinished-answer", status="running"))
    goal = OrchestrationGoal(project_id=advisor_project.id, objective="Old captured goal description", status="active")
    advisor_session.add(goal)
    await advisor_session.commit()
    turn = await _new_turn(advisor_session, advisor_project.id, advisor_user.id)
    goal.objective = "Fresh current goal description"
    await advisor_session.commit()
    requests = []

    async def completion(**kwargs):
        requests.append(kwargs)
        return _response({"intent": "question", "answer": "Grounded answer"})

    await _service(test_engine, completion).interpret(turn.id)
    serialized = json.dumps(requests[0]["messages"])
    assert "question-0" not in serialized and "question-1" not in serialized
    assert all(f"question-{i}" in serialized and f"answer-{i}" in serialized for i in range(2, 10))
    assert "unfinished-exclude" not in serialized
    assert "Fresh current goal description" in serialized
    assert "Old captured goal description" not in serialized
    assert 'older_pairs_omitted' in serialized


@pytest.mark.parametrize("limit", [0, 100_000])
async def test_advisor_source_committed_before_index_link_recovers_without_dispatch(
    limit, test_engine, advisor_session, advisor_project, advisor_user, monkeypatch
):
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", limit)
    turn = await _new_turn(advisor_session, advisor_project.id, advisor_user.id)
    advisor_session.add(ProjectAdvisorTurn(id=turn.source_record_id, project_id=turn.project_id,
        actor_id=turn.actor_id, question=turn.content, answer="Committed answer", status="completed", tokens_used=10))
    await advisor_session.commit()

    async def completion(**_kwargs):
        raise AssertionError("committed source must never dispatch again")

    result = await _service(test_engine, completion).interpret(turn.id)
    assert result.operation is None and result.answer == "Committed answer"
    async with async_sessionmaker(test_engine, expire_on_commit=False)() as db:
        assert await db.scalar(select(func.count()).select_from(ProjectAdvisorTurn).where(
            ProjectAdvisorTurn.id == turn.source_record_id)) == 1
        source = await db.get(ProjectAdvisorTurn, turn.source_record_id)
        assert source.answer == "Committed answer" and source.tokens_used == 10
        stored = await db.get(ProjectChatTurn, turn.id)
        assert stored.provider_status == "completed" and stored.answer == source.answer
        assert stored.accounting_snapshot["tokens_observed"] == source.tokens_used
        assert stored.accounting_snapshot["unknown_usage"] is False


async def test_project_repair_denied_when_first_response_usage_consumes_pool(
    test_engine, advisor_session, advisor_project, advisor_user, monkeypatch
):
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 100_000)
    turn = await _new_turn(advisor_session, advisor_project.id, advisor_user.id)
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "not-json"}}],
                "usage": {"prompt_tokens": 100_000, "completion_tokens": 0}}

    with pytest.raises(ConversationDomainError):
        await _service(test_engine, completion).interpret(turn.id)
    assert len(calls) == 1
    async with async_sessionmaker(test_engine, expire_on_commit=False)() as db:
        source = await db.get(ProjectAdvisorTurn, turn.source_record_id)
        assert source.tokens_used == 100_000
        stored = await db.get(ProjectChatTurn, turn.id)
        assert stored.accounting_snapshot["calls_admitted"] == 1
        assert stored.accounting_snapshot["tokens_observed"] == source.tokens_used
        assert stored.accounting_snapshot["unknown_usage"] is False


async def test_project_unknown_usage_denies_repair_without_free_call(
    test_engine, advisor_session, advisor_project, advisor_user, monkeypatch
):
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 100_000)
    turn = await _new_turn(advisor_session, advisor_project.id, advisor_user.id)
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "not-json"}}]}

    with pytest.raises(ConversationDomainError):
        await _service(test_engine, completion).interpret(turn.id)
    assert len(calls) == 1
    async with async_sessionmaker(test_engine, expire_on_commit=False)() as db:
        assert (await db.get(ProjectAdvisorTurn, turn.source_record_id)).tokens_used is None
        stored = await db.get(ProjectChatTurn, turn.id)
        assert stored.accounting_snapshot["unknown_usage"] is True
        assert stored.accounting_snapshot["calls_admitted"] == 1


async def test_goal_context_charges_goal_pool_and_keeps_project_disabled(
    test_engine, conversation_goal_run, test_user, db_session, monkeypatch
):
    from huddleroom.models.orchestration_conversation import ConversationResponse, ConversationReservation
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 0)
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", 100_000)
    goal, run = conversation_goal_run
    await db_session.commit()
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with factory() as db:
        turn = await _new_turn(db, goal.project_id, test_user.id, "Resume this goal", goal.id)
    calls = []

    async def completion(**kwargs):
        async with factory() as db:
            response = await db.get(ConversationResponse, turn.source_record_id)
            reservation = await db.scalar(select(ConversationReservation).where(
                ConversationReservation.response_id == turn.source_record_id))
            assert response.status == "running" and reservation.status == "committed"
            assert response.dossier["_conversation_runtime"]["chat_turn_id"] == str(turn.id)
        calls.append(kwargs)
        return {"choices": [{"message": {"content": json.dumps({
            "intent": "instruction", "operation": "resume", "goal_id": str(goal.id),
        })}}], "usage": {"prompt_tokens": 2, "completion_tokens": 3}}

    result = await _service(test_engine, completion).interpret(turn.id)
    assert result.operation == "resume" and len(calls) == 1
    async with factory() as db:
        response = await db.get(ConversationResponse, turn.source_record_id)
        frozen = response.dossier.copy()
        assert response.status == "completed"
        reservation = await db.scalar(select(ConversationReservation).where(
            ConversationReservation.response_id == turn.source_record_id))
        assert reservation.status == "settled" and reservation.settled_tokens == 5
        assert await db.scalar(select(func.count()).select_from(ProjectAdvisorTurn)) == 0
        assert await db.scalar(select(func.count()).select_from(OrchestrationAction)) == 0
    await _service(test_engine, completion).interpret(turn.id)
    async with factory() as db:
        assert (await db.get(ConversationResponse, turn.source_record_id)).dossier == frozen
    assert len(calls) == 1


@pytest.mark.parametrize("limit", [0, 1])
async def test_goal_disabled_or_exhausted_never_swaps_to_advisor(
    limit, test_engine, conversation_goal_run, test_user, db_session, monkeypatch
):
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 100_000)
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", limit)
    goal, _ = conversation_goal_run
    await db_session.commit()
    async with async_sessionmaker(test_engine, expire_on_commit=False)() as db:
        turn = await _new_turn(db, goal.project_id, test_user.id, "Resume this goal", goal.id)
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        raise AssertionError("goal allowance failure cannot swap charge scope")

    with pytest.raises(ConversationDomainError):
        await _service(test_engine, completion).interpret(turn.id)
    assert calls == []


async def test_hanging_advisor_times_out_releases_claim_lock_and_never_redispatches(
    test_engine, advisor_session, advisor_project, advisor_user, monkeypatch
):
    import huddleroom.services.orchestration_project_advisor_service as advisor_module
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 100_000)
    turn = await _new_turn(advisor_session, advisor_project.id, advisor_user.id)
    next_turn = await _new_turn(advisor_session, advisor_project.id, advisor_user.id, "New question")
    original_wait_for = asyncio.wait_for

    async def fast_deadline(awaitable, timeout):
        return await original_wait_for(awaitable, timeout=0.02 if timeout == 120 else timeout)

    monkeypatch.setattr(advisor_module.asyncio, "wait_for", fast_deadline)
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        async with async_sessionmaker(test_engine, expire_on_commit=False)() as db:
            stored = await db.get(ProjectChatTurn, turn.id)
            assert stored.deadline_at is not None and stored.provider_status == "running"
        await asyncio.Event().wait()

    service = _service(test_engine, completion)
    with pytest.raises(TimeoutError):
        await service.interpret(turn.id)
    replay = await original_wait_for(service.interpret(turn.id), timeout=1)
    assert replay.operation is None
    with pytest.raises(ConversationDomainError):
        await original_wait_for(service.interpret(next_turn.id), timeout=1)
    assert len(calls) == 1
    async with async_sessionmaker(test_engine, expire_on_commit=False)() as db:
        stored = await db.get(ProjectChatTurn, turn.id)
        assert stored.provider_status == "interrupted_unknown"
        assert stored.accounting_snapshot["unknown_usage"] is True
        assert (await db.get(ProjectAdvisorTurn, turn.source_record_id)).tokens_used is None


async def test_goal_chat_keeps_investigation_continuation_and_reservations(
    test_engine, conversation_goal_run, test_user, db_session, tmp_path, monkeypatch
):
    from tests.test_orchestration_conversation_service import _tool_result, _set_workspace
    from huddleroom.models.orchestration_conversation import (
        ConversationInvestigation, ConversationInvestigationReservation, ConversationReservation,
    )
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", 100_000)
    monkeypatch.setattr(settings, "orchestration_conversation_investigation_enabled", True)
    goal, _ = conversation_goal_run
    workspace = tmp_path / "chat-investigation"
    workspace.mkdir()
    (workspace / "risk.txt").write_text("gate pending\n", encoding="utf-8")
    await _set_workspace(db_session, goal.project_id, workspace)
    await db_session.commit()
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with factory() as db:
        turn = await _new_turn(db, goal.project_id, test_user.id, "Investigate risk", goal.id)
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _tool_result()
        return {"choices": [{"message": {"content": json.dumps({
            "findings": "Gate is pending.", "uncertainty": "", "sources": ["risk.txt#L1-L1"],
        })}}], "usage": {"prompt_tokens": 6, "completion_tokens": 3}}

    service = _service(test_engine, completion)
    result = await service.interpret(turn.id)
    replay = await service.interpret(turn.id)
    assert result.answer == "Gate is pending." and result.operation is None
    assert replay == result and len(calls) == 2
    async with factory() as db:
        investigation = await db.scalar(select(ConversationInvestigation).where(
            ConversationInvestigation.response_id == turn.source_record_id))
        assert investigation.status == "completed" and investigation.report["findings"] == result.answer
        first = await db.scalar(select(ConversationReservation).where(
            ConversationReservation.response_id == turn.source_record_id))
        second = await db.scalar(select(ConversationInvestigationReservation).where(
            ConversationInvestigationReservation.investigation_id == investigation.id))
        assert first.status == second.status == "settled"
        assert first.settled_tokens == second.settled_tokens == 9


async def test_concurrent_project_turns_cannot_overadmit_same_pool(
    test_engine, advisor_session, advisor_project, advisor_user, monkeypatch
):
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 100_000)
    first = await _new_turn(advisor_session, advisor_project.id, advisor_user.id, "First question")
    second = await _new_turn(advisor_session, advisor_project.id, advisor_user.id, "Second question")
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        entered.set()
        await release.wait()
        return _response({"intent": "question", "answer": "Done"}, usage=50_000)

    service = _service(test_engine, completion)
    pending_first = asyncio.create_task(service.interpret(first.id))
    await asyncio.wait_for(entered.wait(), timeout=2)
    pending_second = asyncio.create_task(service.interpret(second.id))
    release.set()
    results = await asyncio.gather(pending_first, pending_second, return_exceptions=True)
    assert len(calls) == 1
    assert sum(isinstance(result, ConversationDomainError) for result in results) == 1
