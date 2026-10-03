"""Endpoint contract tests for the project advisor router (T4.3)."""
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


async def test_submit_disabled_allowance_returns_409(advisor_client, advisor_project, advisor_service, monkeypatch):
    monkeypatch.setattr(
        "huddleroom.services.orchestration_project_advisor_service.settings.orchestration_advisor_allowance_tokens",
        0,
    )
    advisor_service(_canned_completion({"answer": "n/a", "citations": [], "off_topic": False}))

    response = await advisor_client.post(
        f"/api/v1/projects/{advisor_project.id}/orchestration/conversation",
        json={"content": "What's the state of my project?"},
    )

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "advisor_disabled"


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

    monkeypatch.setattr(
        "huddleroom.services.orchestration_project_advisor_service.advisor_allowance_used", fake_used
    )

    response = await advisor_client.post(
        f"/api/v1/projects/{advisor_project.id}/orchestration/conversation",
        json={"content": "What's the state of my project?"},
    )

    assert response.status_code == 200, response.text


async def test_submit_exhausted_allowance_returns_429(
    advisor_client, advisor_project, advisor_service, monkeypatch
):
    monkeypatch.setattr(
        "huddleroom.services.orchestration_project_advisor_service.settings.orchestration_advisor_allowance_tokens",
        1,
    )
    svc = advisor_service(_canned_completion({"answer": "n/a", "citations": [], "off_topic": False}))

    async def fake_used(_db, _project_id, _actor_id):
        return 5

    monkeypatch.setattr(advisor_router, "advisor_allowance_used", fake_used)
    monkeypatch.setattr(
        "huddleroom.services.orchestration_project_advisor_service.advisor_allowance_used", fake_used
    )

    response = await advisor_client.post(
        f"/api/v1/projects/{advisor_project.id}/orchestration/conversation",
        json={"content": "What's the state of my project?"},
    )

    assert response.status_code == 429, response.text
    assert response.json()["detail"]["code"] == "advisor_allowance_exhausted"


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
        "huddleroom.routers.orchestration_advisor.settings.orchestration_advisor_allowance_tokens",
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
    assert body["allowance"]["unlimited"] is False
    assert body["allowance"]["limit"] == 1_000
    assert body["allowance"]["remaining"] == 1_000 - 15


async def test_get_history_allowance_unlimited_snapshot(
    advisor_client, advisor_project, advisor_service, monkeypatch
):
    monkeypatch.setattr(
        "huddleroom.routers.orchestration_advisor.settings.orchestration_advisor_allowance_tokens",
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
        "huddleroom.routers.orchestration_advisor.settings.orchestration_advisor_allowance_tokens",
        0,
    )

    response = await advisor_client.get(
        f"/api/v1/projects/{advisor_project.id}/orchestration/conversation",
    )

    assert response.status_code == 200, response.text
    allowance = response.json()["allowance"]
    assert allowance["enabled"] is False
    assert allowance["unlimited"] is False
