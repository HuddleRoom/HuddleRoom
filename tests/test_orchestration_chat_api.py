"""Additive chat facade binds human replies and durable sources without replaying effects."""
import asyncio
import json
import uuid

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from huddleroom.config import settings
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_chat import ProjectChatTurn
from huddleroom.schemas.orchestration_chat import ChatInterpretation, ChatSubmitRequest
from huddleroom.services.orchestration_conversation_service import ConversationDomainError
from tests.test_orchestration_chat_actions import _seed, _prepare, _states, _ready
from tests.test_orchestration_roster_mutation import sessions, seed_goal_run, seed_project_user, approve_semantic_reviews

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("safe_goal_analysis")]


@pytest.mark.parametrize("limit", [0, 1])
@pytest.mark.parametrize("historical_unknown", [False, True])
async def test_spr209_goal_chat_ignores_allowance(sessions, tmp_path, monkeypatch, limit, historical_unknown):
    from huddleroom.models.orchestration_advisor import ProjectAdvisorTurn
    from huddleroom.models.orchestration_conversation import ConversationReservation, ConversationResponse
    project, user, goal, run = await _seed(sessions, tmp_path)
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", 100_000)
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 0)
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1 and historical_unknown:
            raise TimeoutError("unknown historical provider outcome")
        return {"choices": [{"message": {"content": json.dumps({"intent": "question", "answer": "Current state"})}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3}}

    facade = service(sessions, completion_fn=completion)
    old = await facade.submit(project.id, user.id, request("Old question", goal.id))
    async with sessions() as db:
        old_row = await db.get(ProjectChatTurn, uuid.UUID(old.id))
        old_source = old_row.source_record_id
        old_reservation = await db.scalar(select(ConversationReservation).where(ConversationReservation.response_id == old_source))
        old_state = (old_reservation.status, old_reservation.reserved_tokens, old_reservation.ceiling_snapshot)
        execution_before = dict((await db.get(OrchestrationRun, run.id)).budget_state)
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", limit)
    data = request("What changed?", goal.id)
    fresh = await facade.submit(project.id, user.id, data)
    assert fresh.answer == "Current state"
    replay = await facade.submit(project.id, user.id, data)
    assert replay.id == fresh.id and len(calls) == 2
    with pytest.raises((HTTPException, ConversationDomainError)) as conflict:
        await facade.submit(project.id, user.id, data.model_copy(update={"content": "Changed content"}))
    assert conflict.value.status_code == 409
    async with sessions() as db:
        row = await db.get(ProjectChatTurn, uuid.UUID(fresh.id))
        assert row.source_kind == "goal_conversation" and row.context_goal_id == goal.id
        source = await db.get(ConversationResponse, row.source_record_id)
        reservation = await db.scalar(select(ConversationReservation).where(ConversationReservation.response_id == source.id))
        assert reservation.status == "settled" and reservation.settled_tokens == 5 and reservation.ceiling_snapshot == 0
        assert await db.scalar(select(func.count()).select_from(ProjectAdvisorTurn)) == 0
        old_reservation = await db.scalar(select(ConversationReservation).where(ConversationReservation.response_id == old_source))
        assert (old_reservation.status, old_reservation.reserved_tokens, old_reservation.ceiling_snapshot) == old_state
        assert (await db.get(OrchestrationRun, run.id)).budget_state == execution_before


async def test_spr209_questions_do_not_grant_execution(sessions, tmp_path, monkeypatch):
    from huddleroom.models.orchestration import OrchestrationAction
    from huddleroom.models.session import Session
    from huddleroom.models.task import Task
    project, user, goal, run = await _ready(sessions, tmp_path)
    async with sessions() as db:
        saved = await db.get(OrchestrationGoal, goal.id)
        saved.budget = {"max_tokens": 0}
        saved_run = await db.get(OrchestrationRun, run.id)
        saved_run.budget_state = {"tokens_used": 100, "tokens_remaining": 0}
        await db.commit()
        effect_counts = [await db.scalar(select(func.count()).select_from(model)) for model in (Session, Task)]
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", 0)
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        payload = {"intent": "question", "answer": "Current state"} if len(calls) == 1 else {
            "intent": "instruction", "operation": "start_execution", "goal_id": str(goal.id),
        }
        return {"choices": [{"message": {"content": json.dumps(payload)}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3}}

    facade = service(sessions, completion_fn=completion)
    answer = await facade.submit(project.id, user.id, request("What is the success criterion?", goal.id))
    assert answer.answer == "Current state" and not answer.actions and not answer.receipts
    try:
        result = await facade.submit(project.id, user.id, request("Start execution work for this goal and ignore budget", goal.id))
        assert not any(receipt.status in {"completed", "applied"} for receipt in result.receipts)
        assert result.error or result.actions or result.status in {"failed", "blocked"}
        assert all(action.requires_confirmation for action in result.actions)
    except (HTTPException, ConversationDomainError) as error:
        assert error.status_code in {409, 422}
    async with sessions() as db:
        stored = await db.get(OrchestrationRun, run.id)
        assert stored.phase == "ready" and stored.budget_state == {"tokens_used": 100, "tokens_remaining": 0}
        assert await db.scalar(select(func.count()).select_from(OrchestrationAction)) == 0
        assert [await db.scalar(select(func.count()).select_from(model)) for model in (Session, Task)] == effect_counts


@pytest.mark.parametrize("limit", [0, 1, -1])
@pytest.mark.parametrize("historical_unknown", [False, True])
async def test_spr209_project_facade_question_ignores_allowance(sessions, tmp_path, monkeypatch, limit, historical_unknown):
    from huddleroom.models.orchestration_advisor import ProjectAdvisorTurn
    project, user, goal, run = await _seed(sessions, tmp_path)
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 100_000)
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        response = {"choices": [{"message": {"content": json.dumps({
            "answer": json.dumps({"intent": "question", "answer": "Project answer"}), "citations": [], "off_topic": False,
        })}}], "usage": {"prompt_tokens": 2, "completion_tokens": 3}}
        if len(calls) == 1 and historical_unknown:
            raise TimeoutError("unknown historical outcome")
        return response

    facade = service(sessions, completion_fn=completion)
    old_data = request("Old project question")
    if historical_unknown:
        with pytest.raises(TimeoutError, match="unknown historical outcome"):
            await facade.submit(project.id, user.id, old_data)
        async with sessions() as db:
            old_row = await db.scalar(select(ProjectChatTurn).where(ProjectChatTurn.client_request_id == old_data.client_request_id))
            old_id = str(old_row.id)
    else:
        old_id = (await facade.submit(project.id, user.id, old_data)).id
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", limit)
    replay_old = await facade.submit(project.id, user.id, old_data)
    assert replay_old.id == old_id and len(calls) == 1
    data = request("Fresh project question")
    fresh = await facade.submit(project.id, user.id, data)
    replay = await facade.submit(project.id, user.id, data)
    assert fresh.answer == "Project answer" and replay.id == fresh.id and len(calls) == 2
    assert not fresh.actions and not fresh.receipts
    async with sessions() as db:
        row = await db.get(ProjectChatTurn, uuid.UUID(fresh.id))
        assert row.source_kind == "project_advisor" and row.context_goal_id is None
        source = await db.get(ProjectAdvisorTurn, row.source_record_id)
        assert source.status == "completed" and source.tokens_used == 5
        assert row.accounting_snapshot["limit_tokens"] == -1
        assert row.accounting_snapshot["calls_admitted"] == 1 and row.accounting_snapshot["tokens_observed"] == 5
        assert row.accounting_snapshot["unknown_usage"] is False
        old_row = await db.get(ProjectChatTurn, uuid.UUID(old_id))
        old_source = await db.get(ProjectAdvisorTurn, old_row.source_record_id)
        assert old_source.status == ("failed" if historical_unknown else "completed")
        assert old_source.tokens_used == (None if historical_unknown else 5)
        assert await db.scalar(select(func.count()).select_from(OrchestrationGoal)) == 1
        assert await db.scalar(select(func.count()).select_from(OrchestrationRun)) == 1
        assert (await db.get(OrchestrationRun, run.id)).phase == "baseline"
        assert (await db.get(OrchestrationGoal, goal.id)).status == "active"


class Interpreter:
    def __init__(self, sessions, result=None):
        self.sessions = sessions
        self.result = result or ChatInterpretation(intent="question", answer="Grounded answer")
        self.calls = []

    async def interpret(self, turn_id):
        async with self.sessions() as db:
            row = await db.get(ProjectChatTurn, turn_id)
            self.calls.append((row.id, row.context_goal_id, row.run_id, row.content))
            row.interpretation = self.result.model_dump(mode="json")
            await db.commit()
        return self.result


def service(sessions, **kwargs):
    from huddleroom.services.orchestration_chat_service import OrchestrationChatService
    return OrchestrationChatService(session_factory=sessions, **kwargs)


def request(content, goal=None, action=None, client=None):
    return ChatSubmitRequest(content=content, context_goal_id=goal, action_id=action,
        client_request_id=client or uuid.uuid4())


async def test_answer_only_duplicate_identity_and_conflicting_submission(sessions, tmp_path):
    project, user, goal, run = await _seed(sessions, tmp_path)
    interpreter = Interpreter(sessions)
    facade = service(sessions, interpretation=interpreter)
    data = request("Why is this goal running?", goal.id)
    first = await facade.submit(project.id, user.id, data)
    replay = await facade.submit(project.id, user.id, data)
    assert first.id == replay.id and first.answer == "Grounded answer"
    assert len(interpreter.calls) == 1
    for changed in [data.model_copy(update={"content": "Different question"}),
                    data.model_copy(update={"context_goal_id": None}),
                    data.model_copy(update={"action_id": uuid.uuid4()})]:
        with pytest.raises((HTTPException, ConversationDomainError)) as error:
            await facade.submit(project.id, user.id, changed)
        assert error.value.status_code == 409
    assert (await _states(sessions, goal, run))[0].status == "active"
    async with sessions() as db:
        assert await db.scalar(select(func.count()).select_from(ProjectChatTurn).where(
            ProjectChatTurn.project_id == project.id)) == 1


@pytest.mark.parametrize("reply,bound", [("yes", False), ("confirm", False),
    ("yes?", True), ('"yes"', True), ("If I say yes", True),
    ("yes, but pause a different goal", True), ("why should I confirm?", True),
    ("action id only", True)])
async def test_non_authorizing_replies_are_inert(sessions, tmp_path, reply, bound):
    project, user, goal, run = await _seed(sessions, tmp_path)
    _, action = await _prepare(sessions, user, goal, "pause")
    interpreter = Interpreter(sessions)
    result = await service(sessions, interpretation=interpreter).submit(project.id, user.id,
        request(reply, goal.id, action.action_id if bound else None))
    assert not any(item.status in {"applied", "completed"} for item in result.receipts)
    assert (await _states(sessions, goal, run))[0].status == "active"
    if bound:
        assert len(interpreter.calls) == 1


@pytest.mark.parametrize("reply", ["yes", "confirm", " YES "])
async def test_bound_acceptance_uses_no_model_and_preserves_target_after_navigation(sessions, tmp_path, monkeypatch, reply):
    project, user, goal, run = await _seed(sessions, tmp_path)
    _, action = await _prepare(sessions, user, goal, "pause")
    async with sessions() as db:
        other, other_run = await seed_goal_run(db, project, user=user, objective="Other goal")
        await db.commit()
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 0)
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", 0)
    interpreter = Interpreter(sessions)
    facade = service(sessions, interpretation=interpreter)
    data = request(reply, uuid.uuid4() if reply == "confirm" else other.id, action.action_id)
    first = await facade.submit(project.id, user.id, data)
    second = await facade.submit(project.id, user.id, data)
    assert first.id == second.id and interpreter.calls == []
    assert first.receipts[0].goal_id == goal.id and first.receipts[0].run_id == run.id
    assert (await _states(sessions, goal, run))[0].status == "paused"
    assert (await _states(sessions, other, other_run))[0].status == "active"


@pytest.mark.parametrize("reply", ["no", "dismiss"])
async def test_bound_dismissal_never_applies_or_calls_model(sessions, tmp_path, reply):
    project, user, goal, run = await _seed(sessions, tmp_path)
    _, action = await _prepare(sessions, user, goal, "pause")
    interpreter = Interpreter(sessions)
    facade = service(sessions, interpretation=interpreter)
    await facade.submit(project.id, user.id, request(reply, goal.id, action.action_id))
    assert interpreter.calls == [] and (await _states(sessions, goal, run))[0].status == "active"
    try:
        result = await facade.submit(project.id, user.id, request("confirm", goal.id, action.action_id))
        assert not any(item.status == "completed" for item in result.receipts)
    except (HTTPException, ConversationDomainError) as error:
        assert error.status_code in {404, 409}
    assert (await _states(sessions, goal, run))[0].status == "active"


@pytest.mark.parametrize("foreign", ["actor", "project", "unknown_action"])
async def test_foreign_bound_action_is_denied_before_interpretation(sessions, tmp_path, foreign):
    project, user, goal, run = await _seed(sessions, tmp_path)
    _, action = await _prepare(sessions, user, goal, "pause")
    async with sessions() as db:
        other_project, other_user = await seed_project_user(db, tmp_path)
        await db.commit()
    interpreter = Interpreter(sessions)
    with pytest.raises((HTTPException, ConversationDomainError)) as error:
        await service(sessions, interpretation=interpreter).submit(
            other_project.id if foreign == "project" else project.id,
            other_user.id if foreign == "actor" else user.id,
            request("confirm", goal.id if foreign != "project" else None,
                uuid.uuid4() if foreign == "unknown_action" else action.action_id))
    assert error.value.status_code in {403, 404, 409}
    assert interpreter.calls == [] and (await _states(sessions, goal, run))[0].status == "active"


async def test_multiple_pending_unbound_yes_is_ambiguous(sessions, tmp_path):
    project, user, goal, run = await _seed(sessions, tmp_path)
    await _prepare(sessions, user, goal, "pause")
    await _prepare(sessions, user, goal, "cancel")
    result = await service(sessions, interpretation=Interpreter(sessions)).submit(project.id, user.id, request("yes", goal.id))
    assert not any(receipt.status == "completed" for receipt in result.receipts)
    assert (await _states(sessions, goal, run))[0].status == "active"


async def test_running_duplicate_does_not_duplicate_interpretation(sessions, tmp_path):
    project, user, goal, _ = await _seed(sessions, tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    interpreter = Interpreter(sessions)
    original = interpreter.interpret

    async def held(turn_id):
        entered.set()
        await release.wait()
        return await original(turn_id)

    interpreter.interpret = held
    facade = service(sessions, interpretation=interpreter)
    data = request("What changed?", goal.id)
    first = asyncio.create_task(facade.submit(project.id, user.id, data))
    await asyncio.wait_for(entered.wait(), timeout=2)
    second = asyncio.create_task(facade.submit(project.id, user.id, data))
    release.set()
    results = await asyncio.gather(first, second)
    assert results[0].id == results[1].id and len(interpreter.calls) == 1


async def test_additive_http_route_and_safe_history(sessions, tmp_path, monkeypatch):
    from huddleroom.database import get_db
    from huddleroom.dependencies import get_current_user
    from huddleroom.main import create_app
    import huddleroom.routers.orchestration_chat as router

    project, user, goal, _ = await _seed(sessions, tmp_path)
    facade = service(sessions, interpretation=Interpreter(sessions))
    monkeypatch.setattr(router, "chat_service", facade)
    app = create_app()

    async def current_user():
        return user

    async def database():
        async with sessions() as db:
            yield db

    app.dependency_overrides[get_current_user] = current_user
    app.dependency_overrides[get_db] = database
    path = f"/api/v1/projects/{project.id}/orchestration/chat"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(path, json=request("What changed?", goal.id).model_dump(mode="json"))
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Grounded answer" and body["context_goal_id"] == str(goal.id)
        history = await client.get(path)
        assert history.status_code == 200 and history.json()["total"] >= 1
        assert any(item["id"] == body["id"] for item in history.json()["items"])
        assert "interpretation" not in body and "provider_request_id" not in body
        invalid = await client.post(path, json={"client_request_id": str(uuid.uuid4()), "content": " "})
        assert invalid.status_code == 422


@pytest.mark.parametrize("scope", ["project", "goal", "named_goal"])
async def test_provider_pool_routing_is_declared_before_dispatch(sessions, tmp_path, monkeypatch, scope):
    from huddleroom.models.orchestration_advisor import ProjectAdvisorTurn
    from huddleroom.models.orchestration_conversation import ConversationReservation, ConversationResponse

    project, user, goal, _ = await _seed(sessions, tmp_path)
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 100_000 if scope == "project" else 0)
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", 0 if scope == "project" else 100_000)
    calls = []

    async def completion(**kwargs):
        async with sessions() as db:
            row = await db.scalar(select(ProjectChatTurn).where(ProjectChatTurn.project_id == project.id))
            assert row.provider_request_id and row.source_record_id
            if scope == "project":
                assert row.accounting_snapshot["scope"] == "project_advisor"
            else:
                reservation = await db.scalar(select(ConversationReservation).where(
                    ConversationReservation.response_id == row.source_record_id))
                assert reservation is not None and reservation.status == "committed"
                source = await db.get(ConversationResponse, row.source_record_id)
                assert row.provider_request_id == source.provider_request_id == kwargs["litellm_call_id"]
                assert row.deadline_at == source.deadline_at and row.deadline_at is not None
                assert row.provider_status == source.status == "running"
                assert row.accounting_snapshot["scope"] == "goal_conversation"
        calls.append(kwargs)
        interpretation = {"intent": "question", "answer": "Current state"}
        body = {"answer": json.dumps(interpretation), "citations": [], "off_topic": False} if scope == "project" else interpretation
        return {"choices": [{"message": {"content": json.dumps(body)}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3}}

    content = f"What is the state of {goal.objective}?" if scope == "named_goal" else "What is the state?"
    result = await service(sessions, completion_fn=completion).submit(project.id, user.id,
        request(content, goal.id if scope == "goal" else None))
    assert len(calls) == 1 and result.answer == "Current state"
    async with sessions() as db:
        assert await db.scalar(select(func.count()).select_from(ProjectAdvisorTurn).where(
            ProjectAdvisorTurn.project_id == project.id)) == int(scope == "project")


@pytest.mark.parametrize("scope", ["project", "goal"])
async def test_retired_disabled_pool_preserves_declared_source(sessions, tmp_path, monkeypatch, scope):
    project, user, goal, run = await _seed(sessions, tmp_path)
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 0 if scope == "project" else 100_000)
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", 0 if scope == "goal" else 100_000)
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        interpretation = {"intent": "question", "answer": "Current state"}
        payload = {"answer": json.dumps(interpretation), "citations": [], "off_topic": False} if scope == "project" else interpretation
        return {"choices": [{"message": {"content": json.dumps(payload)}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3}}

    result = await service(sessions, completion_fn=completion).submit(project.id, user.id,
        request("What changed?", goal.id if scope == "goal" else None))
    assert result.answer == "Current state" and len(calls) == 1
    assert (await _states(sessions, goal, run))[0].status == "active"
    async with sessions() as db:
        row = await db.get(ProjectChatTurn, uuid.UUID(result.id))
        assert row.source_kind == ("project_advisor" if scope == "project" else "goal_conversation")


@pytest.mark.parametrize("style", ["question", "quoted", "hypothetical"])
async def test_named_objective_mention_is_not_action_authority(sessions, tmp_path, monkeypatch, style):
    project, user, goal, run = await _seed(sessions, tmp_path)
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 0)
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", 100_000)
    messages = {"question": f"Should I pause {goal.objective}?",
        "quoted": f'"Pause {goal.objective}"', "hypothetical": f"If I pause {goal.objective}"}
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": json.dumps({"intent": "instruction", "operation": "pause",
            "goal_id": str(goal.id), "named_goal": goal.objective})}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3}}

    result = await service(sessions, completion_fn=completion).submit(project.id, user.id, request(messages[style]))
    assert len(calls) == 1 and not result.actions and not result.receipts
    assert (await _states(sessions, goal, run))[0].status == "active"


async def test_named_goal_override_and_ambiguous_objective_are_frozen(sessions, tmp_path):
    project, user, goal, run = await _seed(sessions, tmp_path)
    async with sessions() as db:
        other, other_run = await seed_goal_run(db, project, user=user, objective="Unique other goal")
        await db.commit()
    interpreter = Interpreter(sessions, ChatInterpretation(intent="instruction", operation="pause",
        goal_id=other.id, named_goal=other.objective))
    result = await service(sessions, interpretation=interpreter).submit(project.id, user.id,
        request(f"Pause {other.objective}", goal.id))
    assert interpreter.calls[0][1:3] == (other.id, other_run.id)
    assert result.actions[0].goal_id == other.id and result.actions[0].run_id == other_run.id
    assert (await _states(sessions, goal, run))[0].status == "active"
    async with sessions() as db:
        duplicate, _ = await seed_goal_run(db, project, user=user, objective=other.objective)
        await db.commit()
    ambiguous = await service(sessions, interpretation=interpreter).submit(project.id, user.id,
        request(f"Pause {other.objective}", goal.id))
    assert not ambiguous.actions and not any(item.status == "completed" for item in ambiguous.receipts)
    assert (await _states(sessions, other, other_run))[0].status == "active"


@pytest.mark.parametrize("investigation_status", ["completed", "running", "failed"])
async def test_facade_history_preserves_private_sources_investigation_and_feedback(sessions, investigation_status):
    from tests.test_orchestration_chat_history import _seed_sources
    from huddleroom.models.orchestration_conversation import ConversationInvestigation

    async with sessions() as db:
        seeded = await _seed_sources(db)
        investigation = await db.scalar(select(ConversationInvestigation).where(
            ConversationInvestigation.response_id == seeded["responses"][0]))
        investigation.status = investigation_status
        if investigation_status == "running":
            investigation.finished_at = None
        investigation.report = {"findings": "Verified gate status", "uncertainty": "", "sources": []} if investigation_status == "completed" else None
        investigation.error = {"code": "invalid_investigation_report"} if investigation_status == "failed" else None
        await db.commit()
        original_report = investigation.report
    facade = service(sessions, interpretation=Interpreter(sessions))
    first = await facade.history(seeded["project"], seeded["actors"][0])
    other = await facade.history(seeded["project"], seeded["actors"][1])
    contents = {item.content: item for item in first.items}
    assert "private A" in contents and "private B" not in contents and "foreign advisor" not in contents
    assert "private B" in {item.content for item in other.items}
    assert contents["shared 1"].goal_turn.feedback.feedback_id == seeded["feedback"][0]
    projected = contents["shared 1"].goal_turn.investigation
    assert projected.status == investigation_status and projected.attempt_count == 1
    if investigation_status == "completed":
        assert projected.report.findings == "Verified gate status"
    elif investigation_status == "failed":
        assert projected.error is not None
    assert contents["shared 2"].goal_turn.feedback_eligible is False
    assert next(item for item in other.items if item.content == "shared 2").goal_turn.feedback_eligible is True
    assert contents["shared 1"].goal_turn.proposed_steering.status == "proposed"
    async with sessions() as db:
        saved = await db.get(ConversationInvestigation, investigation.id)
        assert saved.report == original_report and saved.attempt_count == 1


@pytest.mark.parametrize("boundary", ["advisor_publication", "pause", "cancel", "start_execution"])
async def test_crash_recovery_has_one_source_and_effect(sessions, tmp_path, monkeypatch, boundary):
    from huddleroom.models.orchestration_advisor import ProjectAdvisorTurn
    from huddleroom.services.orchestration_chat_actions import OrchestrationChatActionService

    if boundary == "advisor_publication":
        project, user, _, _ = await _seed(sessions, tmp_path)
        monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 100_000)
        calls = []

        async def completion(**kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": json.dumps({"answer": json.dumps({
                "intent": "question", "answer": "Durable advisor answer"}), "citations": [], "off_topic": False})}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3}}

        facade = service(sessions, completion_fn=completion)
        data = request("What changed?")
        original = facade._interpretation.interpret

        async def crash_after_source(turn_id):
            await original(turn_id)
            raise RuntimeError("crash after advisor source publication")

        monkeypatch.setattr(facade._interpretation, "interpret", crash_after_source)
        try:
            await facade.submit(project.id, user.id, data)
        except RuntimeError:
            pass
        replay = await service(sessions, completion_fn=completion).submit(project.id, user.id, data)
        assert replay.answer == "Durable advisor answer" and len(calls) == 1
        async with sessions() as db:
            assert await db.scalar(select(func.count()).select_from(ProjectAdvisorTurn).where(
                ProjectAdvisorTurn.project_id == project.id)) == 1
            assert await db.scalar(select(func.count()).select_from(ProjectChatTurn).where(
                ProjectChatTurn.project_id == project.id)) == 1
        return
    project, user, goal, run = await (_ready(sessions, tmp_path) if boundary == "start_execution" else _seed(sessions, tmp_path))
    _, draft = await _prepare(sessions, user, goal, boundary,
        content="Start execution work for this goal" if boundary == "start_execution" else None)
    actions = OrchestrationChatActionService(session_factory=sessions)
    original_confirm = actions.confirm
    effects = []

    async def crash_after_effect(*args, **kwargs):
        result = await original_confirm(*args, **kwargs)
        effects.append(result)
        raise RuntimeError("crash after committed effect before reply")

    monkeypatch.setattr(actions, "confirm", crash_after_effect)
    facade = service(sessions, interpretation=Interpreter(sessions), actions=actions)
    data = request("confirm", goal.id, draft.action_id)
    try:
        await facade.submit(project.id, user.id, data)
    except RuntimeError:
        pass
    replay = await service(sessions, interpretation=Interpreter(sessions)).submit(project.id, user.id, data)
    assert replay.receipts[0].status == "completed" and len(effects) == 1
    stored_goal, stored_run = await _states(sessions, goal, run)
    assert stored_run.phase == "authorized" if boundary == "start_execution" else stored_goal.status == ("paused" if boundary == "pause" else "cancelled")


@pytest.mark.parametrize("kind", ["agent_batch", "hierarchy"])
async def test_crash_recovery_has_one_source_and_effect_handoffs(sessions, tmp_path, monkeypatch, approve_semantic_reviews, kind):
    from huddleroom.services import agent_service, orchestration_decision_application as application
    from huddleroom.models.agent import Agent
    from huddleroom.models.orchestration import OrchestrationAction
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision, OrchestrationProcessRun
    from tests.test_orchestration_chat_actions import _loaded_goal_user
    from tests.test_orchestration_roster_mutation import seed_definition_proposal, seed_hierarchy_proposal

    seeded = await (seed_definition_proposal(sessions, tmp_path) if kind == "agent_batch"
        else seed_hierarchy_proposal(sessions, tmp_path, monkeypatch))
    goal, user = await _loaded_goal_user(sessions, seeded)
    operation = "answer_agent_definition_batch" if kind == "agent_batch" else "answer_hierarchy"
    payload = {"answers": [{"decision_id": str(seeded.decision_id), "selected_option": "approve"}]} if kind == "agent_batch" else {
        "selected_option": "approve", "target_id": str(seeded.decision_id)}
    interpreter = Interpreter(sessions, ChatInterpretation(intent="instruction", operation=operation,
        goal_id=goal.id, payload=payload))
    dispatches = []

    async def crash(*args, **kwargs):
        dispatches.append(True)
        raise RuntimeError("crash after reservation before roster handoff")

    original_handoff = agent_service.complete_roster_handoff
    monkeypatch.setattr(agent_service, "complete_roster_handoff", crash)
    facade = service(sessions, interpretation=interpreter)
    data = request(f"Answer {seeded.decision_id} approve", goal.id)
    await facade.submit(goal.project_id, user.id, data)
    async with sessions() as consumer:
        failed = await application.drain_answer_applications(consumer, seeded.run_id)
        await consumer.commit()
    replay = await facade.submit(goal.project_id, user.id, data)
    assert len(dispatches) == 1 and len(interpreter.calls) == 1
    assert replay.receipts and replay.receipts[0].status in {"recorded", "interrupted_unknown"}
    async with sessions() as db:
        turns = (await db.scalars(select(ProjectChatTurn).where(ProjectChatTurn.project_id == goal.project_id))).all()
        assert len(turns) == 1 and turns[0].frozen_context["action_reservation"]
        applications = list(await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == seeded.run_id, OrchestrationAction.action_type == "apply_authority_answer")))
        assert len(applications) == 1
        saved_application = applications[0]
        action_id = saved_application.id
        assert failed.failed == [action_id] and saved_application.status == "failed"
        effects = saved_application.dispatch_contract["effects"]
        assert len([key for key in effects if key.startswith("roster_reservation:")]) == 1
        assert not any(key.startswith("roster_agent:") for key in effects)
        decision = await db.get(OrchestrationAuthorityDecision, seeded.decision_id)
        assert decision.status == "answered" and decision.continuation_action_id == action_id
        source_id = decision.source_process_run_id
    # Failed continuations require explicit retry; neither chat replay nor a sweep reaccepts the answer.
    async with sessions() as consumer:
        skipped = await application.drain_answer_applications(consumer, seeded.run_id)
        await consumer.commit()
    assert not skipped.applied and not skipped.failed and len(dispatches) == 1
    completed_handoffs = []
    async def complete_once(*args, **kwargs):
        result = await original_handoff(*args, **kwargs)
        completed_handoffs.append(True)
        return result
    monkeypatch.setattr(agent_service, "complete_roster_handoff", complete_once)
    async with sessions() as retry:
        await application.retry_answer_application(retry, action_id)
        await retry.commit()
    async with sessions() as consumer:
        applied = await application.drain_answer_applications(consumer, seeded.run_id)
        await consumer.commit()
    assert applied.applied == [action_id] and completed_handoffs == [True]
    async with sessions() as db:
        saved_application = await db.get(OrchestrationAction, action_id)
        assert saved_application.status == "completed"
        if kind == "agent_batch":
            assert (await db.get(OrchestrationProcessRun, source_id)).status == "completed"
        effects = dict(saved_application.dispatch_contract["effects"])
        roster_effects = [effect for key, effect in effects.items() if key.startswith("roster_agent:")]
        assert len(roster_effects) == 1
        assert await db.get(Agent, uuid.UUID(roster_effects[0]["target_id"])) is not None
        agent_count = await db.scalar(select(func.count()).select_from(Agent))
    await facade.submit(goal.project_id, user.id, data)
    async with sessions() as consumer:
        await application.drain_answer_applications(consumer, seeded.run_id)
        await consumer.commit()
        assert await consumer.scalar(select(func.count()).select_from(Agent)) == agent_count
        assert (await consumer.get(OrchestrationAction, action_id)).dispatch_contract["effects"] == effects
        assert await consumer.scalar(select(func.count()).select_from(ProjectChatTurn)) == 1
    assert completed_handoffs == [True] and len(interpreter.calls) == 1


async def test_frozen_submission_run_replacement_cannot_prepare_for_new_run(sessions, tmp_path):
    project, user, goal, run = await _seed(sessions, tmp_path)
    interpreter = Interpreter(sessions, ChatInterpretation(intent="instruction", operation="pause", goal_id=goal.id))
    original = interpreter.interpret

    async def replace_run(turn_id):
        async with sessions() as db:
            old = await db.get(OrchestrationRun, run.id)
            old.status = "cancelled"
            new = OrchestrationRun(goal_id=goal.id, status="running", phase="baseline")
            db.add(new)
            await db.commit()
        return await original(turn_id)

    interpreter.interpret = replace_run
    try:
        result = await service(sessions, interpretation=interpreter).submit(project.id, user.id, request("Pause this goal", goal.id))
        assert not result.actions or all(action.run_id == run.id for action in result.actions)
        assert not any(receipt.status == "completed" for receipt in result.receipts)
    except HTTPException as error:
        assert error.status_code == 409
    assert (await _states(sessions, goal, run))[0].status == "active"


async def test_crash_recovery_has_one_source_and_effect_recovery_reservation(sessions, tmp_path, monkeypatch):
    from tests.test_orchestration_recovery_authority import diagnosed, actions_of
    from tests.test_orchestration_roster_mutation import make_agent
    from huddleroom.services.orchestration_chat_actions import OrchestrationChatActionService

    project, user, goal, run = await _seed(sessions, tmp_path)
    async with sessions() as db:
        agent = make_agent(name=f"planner-{uuid.uuid4()}", capabilities=["planning"])
        db.add(agent)
        await db.flush()
        await db.commit()
        _, _, proposal, decision = await diagnosed(db, await db.get(OrchestrationGoal, goal.id),
            await db.get(OrchestrationRun, run.id), agent, owns_setup=True)
        await db.commit()
    interpreter = Interpreter(sessions, ChatInterpretation(intent="instruction", operation="approve_recovery",
        goal_id=goal.id, payload={"target_id": str(decision.id), "selected_option": "approve_fix"}))
    actions = OrchestrationChatActionService(session_factory=sessions)
    original = actions.execute
    effects = []

    async def crash_after_reservation(*args, **kwargs):
        receipt = await original(*args, **kwargs)
        effects.append(receipt)
        raise RuntimeError("crash after durable recovery reservation before reply")

    monkeypatch.setattr(actions, "execute", crash_after_reservation)
    data = request(f"Approve {decision.id} approve_fix", goal.id)
    try:
        await service(sessions, interpretation=interpreter, actions=actions).submit(project.id, user.id, data)
    except RuntimeError:
        pass
    replay = await service(sessions, interpretation=interpreter).submit(project.id, user.id, data)
    assert replay.receipts[0].status == "recorded" and len(effects) == len(interpreter.calls) == 1
    async with sessions() as db:
        wrappers = await actions_of(db, run, "recovery_apply")
        assert len(wrappers) == 1
        turn = await db.scalar(select(ProjectChatTurn).where(ProjectChatTurn.project_id == project.id))
        assert str(proposal.id) in str(turn.frozen_context["action_reservation"])


@pytest.mark.parametrize("boundary", ["pre_dispatch", "admitted_unknown", "completed_source"])
async def test_goal_source_crash_recovery_keeps_one_source_and_reservation(sessions, tmp_path, monkeypatch, boundary):
    from huddleroom.models.orchestration_conversation import ConversationMessage, ConversationResponse, ConversationReservation
    from huddleroom.models.orchestration_conversation import ConversationInvestigation, ConversationInvestigationReservation
    from tests.test_orchestration_conversation_service import _tool_result
    from pathlib import Path

    project, user, goal, _ = await _seed(sessions, tmp_path)
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", 100_000)
    monkeypatch.setattr(settings, "orchestration_conversation_investigation_enabled", True)
    (Path(project.workspace_path) / "risk.txt").write_text("gate pending\n", encoding="utf-8")
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        async with sessions() as db:
            row = await db.scalar(select(ProjectChatTurn).where(ProjectChatTurn.project_id == project.id))
            source = await db.get(ConversationResponse, row.source_record_id)
            assert row.provider_request_id == source.provider_request_id
            assert row.deadline_at == source.deadline_at and row.deadline_at is not None
        if boundary == "admitted_unknown":
            raise RuntimeError("provider outcome is unknown after admission")
        if boundary == "completed_source":
            if len(calls) == 1:
                return _tool_result()
            return {"choices": [{"message": {"content": json.dumps({"findings": "Durable goal answer",
                "uncertainty": "", "sources": ["risk.txt#L1-L1"]})}}],
                "usage": {"prompt_tokens": 6, "completion_tokens": 3}}
        return {"choices": [{"message": {"content": json.dumps({"intent": "question", "answer": "Durable goal answer"})}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3}}

    facade = service(sessions, completion_fn=completion)
    original = facade._interpretation.interpret

    async def crash(turn_id):
        if boundary != "pre_dispatch":
            await original(turn_id)
        raise RuntimeError("crash before facade publication")

    if boundary != "admitted_unknown":
        monkeypatch.setattr(facade._interpretation, "interpret", crash)
    data = request("What changed?", goal.id)
    try:
        await facade.submit(project.id, user.id, data)
    except RuntimeError:
        pass
    replay = await service(sessions, completion_fn=completion).submit(project.id, user.id, data)
    assert len(calls) == (2 if boundary == "completed_source" else 1)
    if boundary == "admitted_unknown":
        assert replay.status in {"failed", "interrupted_unknown"}
    else:
        assert replay.answer == "Durable goal answer" and replay.status == "completed"
    async with sessions() as db:
        messages = (await db.scalars(select(ConversationMessage).where(ConversationMessage.goal_id == goal.id))).all()
        assert len(messages) == 1
        responses = (await db.scalars(select(ConversationResponse).where(ConversationResponse.message_id == messages[0].id))).all()
        assert len(responses) == 1
        reservations = (await db.scalars(select(ConversationReservation).where(ConversationReservation.response_id == responses[0].id))).all()
        assert len(reservations) == 1
        assert reservations[0].status == ("held_unknown" if boundary == "admitted_unknown" else "settled")
        turn = await db.scalar(select(ProjectChatTurn).where(ProjectChatTurn.project_id == project.id))
        assert turn.source_record_id == responses[0].id
        assert turn.provider_request_id == responses[0].provider_request_id
        if boundary == "completed_source":
            investigations = (await db.scalars(select(ConversationInvestigation).where(
                ConversationInvestigation.response_id == responses[0].id))).all()
            assert len(investigations) == 1 and investigations[0].report["findings"] == "Durable goal answer"
            assert replay.goal_turn.investigation.report.findings == "Durable goal answer"
            investigation_reservations = (await db.scalars(select(ConversationInvestigationReservation).where(
                ConversationInvestigationReservation.investigation_id == investigations[0].id))).all()
            assert len(investigation_reservations) == 1 and investigation_reservations[0].status == "settled"


async def test_bound_stale_review_crash_recovers_new_review_without_authorizing(sessions, tmp_path, monkeypatch):
    from huddleroom.services.orchestration_chat_actions import OrchestrationChatActionService

    project, user, goal, run = await _ready(sessions, tmp_path)
    _, action = await _prepare(sessions, user, goal, "start_execution", content="Start execution work for this goal")
    async with sessions() as db:
        saved = await db.get(OrchestrationGoal, goal.id)
        saved.budget = {"max_tokens": 200}
        await db.commit()
    actions = OrchestrationChatActionService(session_factory=sessions)
    original = actions.confirm

    async def crash(*args, **kwargs):
        result = await original(*args, **kwargs)
        assert result.status == "pending" and result.error_code == "chat_stale_review"
        raise RuntimeError("crash after regenerated review before reply publication")

    monkeypatch.setattr(actions, "confirm", crash)
    data = request("confirm", goal.id, action.action_id)
    try:
        await service(sessions, interpretation=Interpreter(sessions), actions=actions).submit(project.id, user.id, data)
    except RuntimeError:
        pass
    replay = await service(sessions, interpretation=Interpreter(sessions)).submit(project.id, user.id, data)
    assert replay.actions and replay.actions[0].action_id != action.action_id
    assert replay.actions[0].requires_confirmation
    assert (await _states(sessions, goal, run))[1].phase == "ready"
    # The response may show the linked review, but durable storage has one authoritative draft.
    async with sessions() as db:
        rows = (await db.scalars(select(ProjectChatTurn).where(ProjectChatTurn.project_id == project.id))).all()
        assert sum(row.action_draft is not None and row.action_draft["action_id"] == str(replay.actions[0].action_id) for row in rows) == 1


async def test_bound_action_correction_cannot_cross_to_named_other_goal(sessions, tmp_path):
    project, user, goal, run = await _seed(sessions, tmp_path)
    _, action = await _prepare(sessions, user, goal, "pause")
    async with sessions() as db:
        other, other_run = await seed_goal_run(db, project, user=user, objective="Distinct Beta")
        other.status, other_run.status = "paused", "paused"
        await db.commit()
    interpreter = Interpreter(sessions, ChatInterpretation(intent="instruction", operation="resume",
        goal_id=other.id, named_goal=other.objective))
    try:
        result = await service(sessions, interpretation=interpreter).submit(project.id, user.id,
            request(f"Resume {other.objective}", other.id, action.action_id))
        assert not result.actions and not any(receipt.status == "completed" for receipt in result.receipts)
    except ConversationDomainError as error:
        assert error.status_code == 409 and error.code == "chat_target_conflict"
    assert (await _states(sessions, goal, run))[0].status == "active"
    assert (await _states(sessions, other, other_run))[0].status == "paused"


async def test_known_confirmation_denial_replays_as_denial(sessions, tmp_path):
    project, user, goal, run = await _seed(sessions, tmp_path)
    _, action = await _prepare(sessions, user, goal, "pause")
    facade = service(sessions, interpretation=Interpreter(sessions))
    await facade.submit(project.id, user.id, request("dismiss", goal.id, action.action_id))
    data = request("confirm", goal.id, action.action_id)
    try:
        first = await facade.submit(project.id, user.id, data)
        assert first.status == "failed" and first.error
    except (HTTPException, ConversationDomainError) as error:
        assert error.status_code == 409
    replay = await facade.submit(project.id, user.id, data)
    assert replay.status == "failed" and replay.error
    assert (await _states(sessions, goal, run))[0].status == "active"


async def test_known_goal_provider_failure_preserves_settled_usage_on_index_and_replay(sessions, tmp_path, monkeypatch):
    from huddleroom.models.orchestration_conversation import ConversationResponse, ConversationReservation

    project, user, goal, _ = await _seed(sessions, tmp_path)
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", 100_000)
    monkeypatch.setattr(settings, "orchestration_conversation_investigation_enabled", True)
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": None, "tool_calls": [{"function": {
            "name": "request_investigation", "arguments": '{"objective":"Missing required requests"}',
        }}]}}], "usage": {"prompt_tokens": 2, "completion_tokens": 3}}

    data = request("What changed?", goal.id)
    facade = service(sessions, completion_fn=completion)
    first = await facade.submit(project.id, user.id, data)
    replay = await facade.submit(project.id, user.id, data)
    assert first.status == replay.status == "failed" and len(calls) == 1
    async with sessions() as db:
        row = await db.scalar(select(ProjectChatTurn).where(ProjectChatTurn.project_id == project.id))
        source = await db.get(ConversationResponse, row.source_record_id)
        reservation = await db.scalar(select(ConversationReservation).where(ConversationReservation.response_id == source.id))
        assert source.status == "failed" and reservation.status == "settled" and reservation.settled_tokens == 5
        assert row.accounting_snapshot["tokens_observed"] == 5
        assert row.accounting_snapshot["unknown_usage"] is False
