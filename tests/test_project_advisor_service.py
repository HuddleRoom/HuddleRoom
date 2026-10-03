"""Unit tests for OrchestrationProjectAdvisorService (T3.3).

Uses a bare (non-transaction-wrapped) session rather than the shared
db_session fixture: the service commits internally (mirrors
OrchestrationConversationService._settle/_fail_known — durable persistence
of a failed turn must survive a subsequent raise), which is incompatible
with db_session's `async with session.begin(): ... rollback()` wrapper.
"""
import asyncio
import json
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from huddleroom.config import settings
from huddleroom.models.project import Project
from huddleroom.models.user import User
from huddleroom.security import hash_password
from huddleroom.services.orchestration_conversation_service import ConversationDomainError
from huddleroom.services.orchestration_project_advisor_service import OrchestrationProjectAdvisorService


@pytest_asyncio.fixture
async def advisor_session(test_engine):
    session_factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def advisor_user(advisor_session):
    user = User(
        email=f"advisor-{uuid.uuid4()}@example.com",
        hashed_password=hash_password("testpassword"),
        display_name="Advisor Test User",
        role="member",
    )
    advisor_session.add(user)
    await advisor_session.commit()
    return user


@pytest_asyncio.fixture
async def advisor_project(advisor_session):
    project = Project(name="Advisor Test Project", description="A test project", config={})
    advisor_session.add(project)
    await advisor_session.commit()
    return project


def _canned_response(payload: dict, prompt_tokens: int = 10, completion_tokens: int = 20) -> dict:
    return {
        "choices": [{"message": {"content": json.dumps(payload)}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def _fake_completion_fn(response: dict):
    async def _completion_fn(**kwargs):
        return response
    return _completion_fn


def _always_bad_completion_fn():
    async def _completion_fn(**kwargs):
        return {
            "choices": [{"message": {"content": "not json"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
    return _completion_fn


@pytest.mark.asyncio
async def test_ask_succeeds_under_allowance(advisor_session, advisor_user, advisor_project, monkeypatch):
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 1000)
    payload = {
        "answer": "Two goals are active.",
        "citations": [{"type": "goal", "id": "g1", "label": "Ship feature"}],
        "off_topic": False,
    }
    service = OrchestrationProjectAdvisorService(completion_fn=_fake_completion_fn(_canned_response(payload)))

    turn = await service.ask(advisor_session, advisor_project.id, advisor_user.id, "What's the state of my project?")

    assert turn.status == "completed"
    assert turn.answer == "Two goals are active."
    assert turn.citations == [{"type": "goal", "id": "g1", "label": "Ship feature"}]
    assert turn.off_topic is False
    assert turn.tokens_used == 30


@pytest.mark.asyncio
async def test_ask_accepts_fenced_json_response(advisor_session, advisor_user, advisor_project, monkeypatch):
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 1000)
    response = _canned_response({"answer": "Two goals are active.", "citations": [], "off_topic": False})
    response["choices"][0]["message"]["content"] = ' ```json\n{"answer":"Two goals are active.","citations":[],"off_topic":false}\n``` '

    turn = await OrchestrationProjectAdvisorService(completion_fn=_fake_completion_fn(response)).ask(
        advisor_session, advisor_project.id, advisor_user.id, "What's the state?"
    )

    assert (turn.status, turn.answer, turn.citations) == ("completed", "Two goals are active.", [])


@pytest.mark.asyncio
async def test_ask_raises_when_allowance_at_limit(advisor_session, advisor_user, advisor_project, monkeypatch):
    from huddleroom.models.orchestration_advisor import ProjectAdvisorTurn

    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 100)
    advisor_session.add(
        ProjectAdvisorTurn(
            project_id=advisor_project.id, actor_id=advisor_user.id,
            question="q", answer="a", status="completed", tokens_used=100,
        )
    )
    await advisor_session.commit()
    service = OrchestrationProjectAdvisorService(completion_fn=_fake_completion_fn(_canned_response({})))

    with pytest.raises(ConversationDomainError) as exc_info:
        await service.ask(advisor_session, advisor_project.id, advisor_user.id, "What's next?")

    assert exc_info.value.code == "advisor_allowance_exhausted"


@pytest.mark.asyncio
async def test_ask_succeeds_unlimited_even_past_usage(advisor_session, advisor_user, advisor_project, monkeypatch):
    """limit=-1 (unlimited) never raises exhausted, regardless of prior usage."""
    from huddleroom.models.orchestration_advisor import ProjectAdvisorTurn

    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", -1)
    advisor_session.add(
        ProjectAdvisorTurn(
            project_id=advisor_project.id, actor_id=advisor_user.id,
            question="q", answer="a", status="completed", tokens_used=10_000_000,
        )
    )
    await advisor_session.commit()
    payload = {"answer": "All good.", "citations": [], "off_topic": False}
    service = OrchestrationProjectAdvisorService(completion_fn=_fake_completion_fn(_canned_response(payload)))

    turn = await service.ask(advisor_session, advisor_project.id, advisor_user.id, "Anything urgent?")

    assert turn.status == "completed"


@pytest.mark.asyncio
async def test_finite_allowance_serializes_concurrent_asks(advisor_session, advisor_user, advisor_project, monkeypatch):
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 30)
    factory = async_sessionmaker(advisor_session.bind, expire_on_commit=False)
    payload = {"answer": "All good.", "citations": [], "off_topic": False}

    async def completion(**_kwargs):
        await asyncio.sleep(0)
        return _canned_response(payload)

    async def ask_once():
        async with factory() as session:
            return await OrchestrationProjectAdvisorService(completion_fn=completion).ask(
                session, advisor_project.id, advisor_user.id, "Anything urgent?"
            )

    first, second = await asyncio.gather(ask_once(), ask_once(), return_exceptions=True)

    assert sum(isinstance(result, ConversationDomainError) for result in (first, second)) == 1
    assert sum(getattr(result, "tokens_used", 0) for result in (first, second)) == 30


@pytest.mark.asyncio
async def test_ask_raises_when_allowance_disabled(advisor_session, advisor_user, advisor_project, monkeypatch):
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 0)
    service = OrchestrationProjectAdvisorService(completion_fn=_fake_completion_fn(_canned_response({})))

    with pytest.raises(ConversationDomainError) as exc_info:
        await service.ask(advisor_session, advisor_project.id, advisor_user.id, "What's next?")

    assert exc_info.value.code == "advisor_disabled"


@pytest.mark.asyncio
async def test_ask_raises_on_empty_question(advisor_session, advisor_user, advisor_project, monkeypatch):
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 1000)
    service = OrchestrationProjectAdvisorService(completion_fn=_fake_completion_fn(_canned_response({})))

    with pytest.raises(ConversationDomainError) as exc_info:
        await service.ask(advisor_session, advisor_project.id, advisor_user.id, "   ")

    assert exc_info.value.code == "invalid_content"


@pytest.mark.asyncio
async def test_ask_persists_off_topic_deflection_with_no_citations(
    advisor_session, advisor_user, advisor_project, monkeypatch
):
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 1000)
    payload = {
        "answer": "That's outside what I can help with here.",
        "citations": [],
        "off_topic": True,
    }
    service = OrchestrationProjectAdvisorService(completion_fn=_fake_completion_fn(_canned_response(payload)))

    turn = await service.ask(advisor_session, advisor_project.id, advisor_user.id, "What's the weather today?")

    assert turn.status == "completed"
    assert turn.off_topic is True
    assert turn.citations == []


@pytest.mark.asyncio
async def test_ask_persists_failed_turn_and_reraises_on_repair_exhaustion(
    advisor_session, advisor_user, advisor_project, monkeypatch
):
    from huddleroom.models.orchestration_advisor import ProjectAdvisorTurn

    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 1000)
    service = OrchestrationProjectAdvisorService(completion_fn=_always_bad_completion_fn())

    with pytest.raises(ValueError):
        await service.ask(advisor_session, advisor_project.id, advisor_user.id, "What's the state of my project?")

    rows = (
        await advisor_session.execute(
            select(ProjectAdvisorTurn).where(ProjectAdvisorTurn.project_id == advisor_project.id)
        )
    ).scalars().all()
    assert len(rows) == 1
    assert rows[0].status == "failed"
    assert rows[0].error
    assert rows[0].answer is None
    assert rows[0].tokens_used == 90


@pytest.mark.asyncio
async def test_ask_repairs_invalid_decision_citation_and_counts_each_attempt(
    advisor_session, advisor_user, advisor_project, monkeypatch
):
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 1_000)
    async def context(*_args):
        return {
            "recent_decisions": [{"id": "decision-1", "goal_id": "goal-1"}],
            "open_goals": [], "recent_meeting_decisions": [], "goal_prefaces": [], "recent_events": [],
        }
    monkeypatch.setattr(
        "huddleroom.services.orchestration_project_advisor_service.build_advisor_context",
        context,
    )
    responses = iter([
        _canned_response({
            "answer": "A decision was made.",
            "citations": [{"type": "decision", "id": "decision-1", "label": "Decision"}],
            "off_topic": False,
        }, 3, 4),
        _canned_response({
            "answer": "A decision was made.",
            "citations": [{"type": "decision", "id": "decision-1", "goal_id": "goal-1", "label": "Decision"}],
            "off_topic": False,
        }, 5, 6),
    ])

    async def completion(**_kwargs):
        return next(responses)

    turn = await OrchestrationProjectAdvisorService(completion_fn=completion).ask(
        advisor_session, advisor_project.id, advisor_user.id, "What was decided?"
    )

    assert turn.citations == [{"type": "decision", "id": "decision-1", "goal_id": "goal-1", "label": "Decision"}]
    assert turn.tokens_used == 18


@pytest.mark.asyncio
async def test_ask_rejects_decision_citation_with_mismatched_goal(
    advisor_session, advisor_user, advisor_project, monkeypatch
):
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 1_000)
    async def context(*_args):
        return {
            "recent_decisions": [{"id": "decision-1", "goal_id": "goal-1"}],
            "open_goals": [], "recent_meeting_decisions": [], "goal_prefaces": [], "recent_events": [],
        }
    monkeypatch.setattr(
        "huddleroom.services.orchestration_project_advisor_service.build_advisor_context",
        context,
    )
    payload = {
        "answer": "A decision was made.",
        "citations": [{"type": "decision", "id": "decision-1", "goal_id": "wrong-goal", "label": "Decision"}],
        "off_topic": False,
    }

    with pytest.raises(ValueError, match="decision citation"):
        await OrchestrationProjectAdvisorService(completion_fn=_fake_completion_fn(_canned_response(payload))).ask(
            advisor_session, advisor_project.id, advisor_user.id, "What was decided?"
        )


@pytest.mark.asyncio
async def test_ask_only_writes_the_single_advisor_turn_row(
    advisor_session, advisor_user, advisor_project, monkeypatch
):
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 1000)
    payload = {"answer": "All quiet.", "citations": [], "off_topic": False}
    service = OrchestrationProjectAdvisorService(completion_fn=_fake_completion_fn(_canned_response(payload)))

    added_objects = []
    original_add = advisor_session.add

    def _tracking_add(instance):
        added_objects.append(instance)
        return original_add(instance)

    monkeypatch.setattr(advisor_session, "add", _tracking_add)

    await service.ask(advisor_session, advisor_project.id, advisor_user.id, "What's the state of my project?")

    from huddleroom.models.orchestration_advisor import ProjectAdvisorTurn

    assert len(added_objects) == 1
    assert isinstance(added_objects[0], ProjectAdvisorTurn)


@pytest.mark.asyncio
async def test_history_returns_last_20_turns_newest_first(
    advisor_session, advisor_user, advisor_project, monkeypatch
):
    from huddleroom.models.orchestration_advisor import ProjectAdvisorTurn

    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 1000)
    for i in range(25):
        advisor_session.add(
            ProjectAdvisorTurn(
                project_id=advisor_project.id, actor_id=advisor_user.id,
                question=f"q{i}", answer=f"a{i}", status="completed", tokens_used=1,
            )
        )
    await advisor_session.commit()

    service = OrchestrationProjectAdvisorService(completion_fn=_fake_completion_fn(_canned_response({})))
    turns = await service.history(advisor_session, advisor_project.id, advisor_user.id)

    assert len(turns) == 20
    assert turns[0].question == "q24"
    assert turns[-1].question == "q5"
