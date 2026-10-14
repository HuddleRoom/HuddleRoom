"""Endpoint contract tests for the project advisor router."""
import json
import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker

import huddleroom.routers.orchestration_advisor as advisor_router
from huddleroom.database import get_db
from huddleroom.main import create_app
from huddleroom.models.project import Project
from huddleroom.services.orchestration_conversation_service import ConversationDomainError
from huddleroom.services.orchestration_project_advisor_service import OrchestrationProjectAdvisorService


def _canned_completion(payload: dict):
    async def completion(**_kwargs):
        return {
            "choices": [{"message": {"content": json.dumps(payload)}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }
    return completion


@pytest.fixture
def advisor_service(monkeypatch):
    """Swap the module-level service instance the router uses; tests patch its completion_fn."""
    def _install(completion_fn):
        svc = OrchestrationProjectAdvisorService(completion_fn=completion_fn)
        monkeypatch.setattr(advisor_router, "service", svc)
        return svc
    return _install


@pytest.fixture
async def advisor_client(test_engine):
    app = create_app()
    factory = async_sessionmaker(test_engine, expire_on_commit=False)

    async def override_get_db():
        async with factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


@pytest.fixture
async def advisor_project(test_engine):
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with factory() as db:
        project = Project(name="Advisor Endpoint Project", description="desc", config={})
        db.add(project)
        await db.commit()
        await db.refresh(project)
        return project


async def test_submit_returns_answer_and_citations(
    advisor_client, advisor_project, advisor_service, monkeypatch
):
    monkeypatch.setattr(
        "huddleroom.services.orchestration_project_advisor_service.settings.orchestration_advisor_allowance_tokens",
        1_000,
    )
    advisor_service(_canned_completion({
        "answer": "The project has one open goal.",
        "citations": [{"type": "goal", "id": str(uuid.uuid4()), "label": "Ship feature"}],
        "off_topic": False,
    }))

    response = await advisor_client.post(
        f"/api/v1/projects/{advisor_project.id}/orchestration/conversation",
        json={"content": "What's the state of my project?"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == "The project has one open goal."
    assert body["off_topic"] is False
    assert len(body["citations"]) == 1
    assert body["citations"][0]["type"] == "goal"
    assert body["citations"][0]["goal_id"] is None
    assert body["status"] == "completed"


async def test_submit_empty_content_returns_422(advisor_client, advisor_project, advisor_service, monkeypatch):
    monkeypatch.setattr(
        "huddleroom.services.orchestration_project_advisor_service.settings.orchestration_advisor_allowance_tokens",
        1_000,
    )
    advisor_service(_canned_completion({"answer": "n/a", "citations": [], "off_topic": False}))

    response = await advisor_client.post(
        f"/api/v1/projects/{advisor_project.id}/orchestration/conversation",
        json={"content": "   "},
    )

    assert response.status_code == 422, response.text


async def test_submit_retired_disabled_allowance_returns_answer(advisor_client, advisor_project, advisor_service, monkeypatch):
    monkeypatch.setattr(
        "huddleroom.services.orchestration_project_advisor_service.settings.orchestration_advisor_allowance_tokens",
        0,
    )
    advisor_service(_canned_completion({"answer": "n/a", "citations": [], "off_topic": False}))

    response = await advisor_client.post(
        f"/api/v1/projects/{advisor_project.id}/orchestration/conversation",
        json={"content": "What's the state of my project?"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["answer"] == "n/a"


async def test_submit_unlimited_allowance_never_exhausts(
    advisor_client, advisor_project, advisor_service, monkeypatch
):
    monkeypatch.setattr(
        "huddleroom.services.orchestration_project_advisor_service.settings.orchestration_advisor_allowance_tokens",
        -1,
    )
    advisor_service(_canned_completion({"answer": "n/a", "citations": [], "off_topic": False}))

    async def fake_used(_db, _project_id, _actor_id):
        return 999_999_999


    response = await advisor_client.post(
        f"/api/v1/projects/{advisor_project.id}/orchestration/conversation",
        json={"content": "What's the state of my project?"},
    )

    assert response.status_code == 200, response.text


async def test_submit_retired_exhausted_allowance_returns_answer(
    advisor_client, advisor_project, advisor_service, monkeypatch
):
    monkeypatch.setattr(
        "huddleroom.services.orchestration_project_advisor_service.settings.orchestration_advisor_allowance_tokens",
        1,
    )
    svc = advisor_service(_canned_completion({"answer": "n/a", "citations": [], "off_topic": False}))

    async def fake_used(_db, _project_id, _actor_id):
        return 5


    response = await advisor_client.post(
        f"/api/v1/projects/{advisor_project.id}/orchestration/conversation",
        json={"content": "What's the state of my project?"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["answer"] == "n/a"


async def test_submit_repair_exhaustion_returns_503(
    advisor_client, advisor_project, advisor_service, monkeypatch
):
    monkeypatch.setattr(
        "huddleroom.services.orchestration_project_advisor_service.settings.orchestration_advisor_allowance_tokens",
        1_000,
    )

    svc = advisor_service(_canned_completion({"answer": "n/a", "citations": [], "off_topic": False}))

    async def broken_ask(*_args, **_kwargs):
        raise RuntimeError("repair exhausted")

    monkeypatch.setattr(svc, "ask", broken_ask)

    response = await advisor_client.post(
        f"/api/v1/projects/{advisor_project.id}/orchestration/conversation",
        json={"content": "What's the state of my project?"},
    )

    assert response.status_code == 503, response.text
    assert response.json()["detail"] == "advisor_unavailable"


async def test_get_history_returns_turns_and_allowance(
    advisor_client, advisor_project, advisor_service, monkeypatch
):
    monkeypatch.setattr(
        "huddleroom.services.orchestration_project_advisor_service.settings.orchestration_advisor_allowance_tokens",
        1_000,
    )
    monkeypatch.setattr(
        "huddleroom.config.settings.orchestration_advisor_allowance_tokens",
        1_000,
    )
    advisor_service(_canned_completion({
        "answer": "One open goal.",
        "citations": [],
        "off_topic": False,
    }))

    submit = await advisor_client.post(
        f"/api/v1/projects/{advisor_project.id}/orchestration/conversation",
        json={"content": "What's the state of my project?"},
    )
    assert submit.status_code == 200, submit.text

    response = await advisor_client.get(
        f"/api/v1/projects/{advisor_project.id}/orchestration/conversation",
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["items"]) == 1
    assert body["items"][0]["answer"] == "One open goal."
    assert body["allowance"]["enabled"] is True
    assert body["allowance"]["unlimited"] is True
    assert body["allowance"]["limit"] == -1
    assert body["allowance"]["remaining"] == -1


async def test_get_history_allowance_unlimited_snapshot(
    advisor_client, advisor_project, advisor_service, monkeypatch
):
    monkeypatch.setattr(
        "huddleroom.config.settings.orchestration_advisor_allowance_tokens",
        -1,
    )

    response = await advisor_client.get(
        f"/api/v1/projects/{advisor_project.id}/orchestration/conversation",
    )

    assert response.status_code == 200, response.text
    allowance = response.json()["allowance"]
    assert allowance["enabled"] is True
    assert allowance["unlimited"] is True
    assert allowance["limit"] == -1
    assert allowance["remaining"] == -1


async def test_get_history_allowance_disabled_snapshot(
    advisor_client, advisor_project, advisor_service, monkeypatch
):
    monkeypatch.setattr(
        "huddleroom.config.settings.orchestration_advisor_allowance_tokens",
        0,
    )

    response = await advisor_client.get(
        f"/api/v1/projects/{advisor_project.id}/orchestration/conversation",
    )

    assert response.status_code == 200, response.text
    allowance = response.json()["allowance"]
    assert allowance["enabled"] is True
    assert allowance["unlimited"] is True


async def test_get_history_reports_zero_remaining_for_unknown_finite_usage(
    advisor_client, advisor_project, monkeypatch
):
    monkeypatch.setattr(
        "huddleroom.config.settings.orchestration_advisor_allowance_tokens", 1_000,
    )

    async def unknown_usage(*_args):
        return None


    response = await advisor_client.get(
        f"/api/v1/projects/{advisor_project.id}/orchestration/conversation",
    )

    assert response.status_code == 200, response.text
    assert response.json()["allowance"]["remaining"] == -1


@pytest.mark.parametrize("limit", [0, 1, -1])
@pytest.mark.parametrize("historical_unknown", [False, True])
async def test_spr209_advisor_direct_question_ignores_allowance(
    advisor_client, advisor_project, advisor_service, monkeypatch, limit, historical_unknown,
):
    from huddleroom.config import settings
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 100_000)
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        response = {"choices": [{"message": {"content": json.dumps({
            "answer": "Fresh answer", "citations": [], "off_topic": False})}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10}}
        if len(calls) == 1 and historical_unknown:
            response.pop("usage")
        return response

    advisor_service(completion)
    url = f"/api/v1/projects/{advisor_project.id}/orchestration/conversation"
    old = await advisor_client.post(url, json={"content": "Old question"})
    assert old.status_code == 200
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", limit)
    fresh = await advisor_client.post(url, json={"content": "Fresh question"})
    assert fresh.status_code == 200 and fresh.json()["answer"] == "Fresh answer" and len(calls) == 2
    history = await advisor_client.get(url)
    assert history.status_code == 200
    assert len(history.json()["items"]) == 2
    assert history.json()["allowance"] == {"enabled": True, "unlimited": True, "limit": -1, "remaining": -1}


@pytest.mark.parametrize("limit", [0, 1, -1])
async def test_spr209_project_history_is_unlimited(advisor_client, advisor_project, monkeypatch, limit):
    from huddleroom.config import settings
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", limit)
    response = await advisor_client.get(f"/api/v1/projects/{advisor_project.id}/orchestration/conversation")
    assert response.status_code == 200
    assert response.json()["allowance"] == {"enabled": True, "unlimited": True, "limit": -1, "remaining": -1}
