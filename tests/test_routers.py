"""Integration tests for Rally API routers using the AsyncClient + SQLite test DB."""
from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.channel import Channel
from huddleroom.models.session import Session as SessionModel
from huddleroom.schemas.task import TaskCreate
from huddleroom.services.task_service import TaskService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _register_and_login(client: AsyncClient) -> dict:
    """Register a fresh user and return auth headers."""
    email = f"user-{uuid.uuid4()}@example.com"
    password = "testpass123"
    r = await client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert r.status_code == 201, r.text
    r = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ===========================================================================
# Health check
# ===========================================================================

@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.parametrize(
    "markers",
    [
        ["Create Agent", "new-agent-modal", "createAgent()"],
        ["new-meeting-modal", "openNewMeetingModal()", "createMeeting()"],
        ["active-meeting-project-id", "Meeting ID:", "Project ID:"],
    ],
    ids=[
        "create_agent_controls",
        "meeting_modal_controls",
        "meeting_identity_fields",
    ],
)
def test_dashboard_asset_includes_expected_markers(markers):
    dashboard_html = Path("huddleroom/static/dev-dashboard/index.html").read_text()
    for marker in markers:
        assert marker in dashboard_html


# ===========================================================================
# Auth router
# ===========================================================================

@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_auth_register(client: AsyncClient):
    email = f"reg-{uuid.uuid4()}@example.com"
    r = await client.post("/api/v1/auth/register", json={"email": email, "password": "pw"})
    assert r.status_code == 201
    data = r.json()
    assert data["email"] == email
    assert "id" in data


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_auth_login_valid(client: AsyncClient):
    email = f"login-{uuid.uuid4()}@example.com"
    await client.post("/api/v1/auth/register", json={"email": email, "password": "pw123"})
    r = await client.post("/api/v1/auth/login", json={"email": email, "password": "pw123"})
    assert r.status_code == 200
    assert "access_token" in r.json()


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_auth_login_invalid_password(client: AsyncClient):
    email = f"bad-{uuid.uuid4()}@example.com"
    await client.post("/api/v1/auth/register", json={"email": email, "password": "correct"})
    r = await client.post("/api/v1/auth/login", json={"email": email, "password": "wrong"})
    assert r.status_code == 401
    assert r.json()["error"] == "http_error"
    assert r.json()["detail"] == "Invalid credentials"


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_auth_me(client: AsyncClient):
    headers = await _register_and_login(client)
    r = await client.get("/api/v1/auth/me", headers=headers)
    assert r.status_code == 200
    assert "email" in r.json()


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_auth_me_unauthenticated(client: AsyncClient):
    from huddleroom.config import Settings
    with patch("huddleroom.dependencies.settings", Settings(auth_enabled=True)):
        r = await client.get("/api/v1/auth/me")
        assert r.status_code == 401


@pytest.mark.asyncio
async def test_no_auth_by_default(client: AsyncClient):
    """When RALLY_AUTH_ENABLED is unset, protected routes work without a token."""
    from huddleroom.config import Settings
    with patch("huddleroom.dependencies.settings", Settings(auth_enabled=False)):
        r = await client.get("/api/v1/projects")
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_public_config_endpoint(client: AsyncClient):
    from huddleroom.config import Settings
    with patch("huddleroom.main.settings", Settings(auth_enabled=False)):
        r = await client.get("/api/v1/config")
    assert r.status_code == 200
    assert "auth_enabled" in r.json()
    assert r.json()["auth_enabled"] is False


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_auth_refresh(client: AsyncClient):
    headers = await _register_and_login(client)
    r = await client.post("/api/v1/auth/refresh", headers=headers)
    assert r.status_code == 200
    assert "access_token" in r.json()


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_api_key_create_list_delete(client: AsyncClient):
    headers = await _register_and_login(client)

    # Create
    r = await client.post("/api/v1/auth/api-keys", json={"label": "my-key"}, headers=headers)
    assert r.status_code == 201
    key_data = r.json()
    assert "key" in key_data
    key_id = key_data["id"]

    # List
    r = await client.get("/api/v1/auth/api-keys", headers=headers)
    assert r.status_code == 200
    ids = [k["id"] for k in r.json()]
    assert key_id in ids

    # Delete
    r = await client.delete(f"/api/v1/auth/api-keys/{key_id}", headers=headers)
    assert r.status_code == 204


# ===========================================================================
# Projects router
# ===========================================================================

@pytest.mark.asyncio
async def test_project_create_list_get_update_delete(client: AsyncClient, tmp_path: Path):
    headers = await _register_and_login(client)

    # Create
    r = await client.post(
        "/api/v1/projects",
        json={"name": "My Project", "description": "desc", "workspace_path": str(tmp_path)},
        headers=headers,
    )
    assert r.status_code == 201
    proj = r.json()
    assert proj["name"] == "My Project"
    pid = proj["id"]

    # Get
    r = await client.get(f"/api/v1/projects/{pid}", headers=headers)
    assert r.status_code == 200
    assert r.json()["id"] == pid

    # List
    r = await client.get("/api/v1/projects", headers=headers)
    assert r.status_code == 200
    ids = [p["id"] for p in r.json()["items"]]
    assert pid in ids

    # Update
    r = await client.put(f"/api/v1/projects/{pid}", json={"name": "Updated"}, headers=headers)
    assert r.status_code == 200
    assert r.json()["name"] == "Updated"

    # Archive (DELETE)
    r = await client.delete(f"/api/v1/projects/{pid}", headers=headers)
    assert r.status_code == 200
    assert r.json()["status"] == "archived"


@pytest.mark.asyncio
async def test_project_get_not_found(client: AsyncClient):
    headers = await _register_and_login(client)
    r = await client.get(f"/api/v1/projects/{uuid.uuid4()}", headers=headers)
    assert r.status_code == 404
    assert r.json()["error"] == "http_error"
    assert r.json()["detail"] == "Project not found"


@pytest.mark.asyncio
async def test_project_create_validation_error_includes_reason(client: AsyncClient):
    headers = await _register_and_login(client)
    r = await client.post(
        "/api/v1/projects",
        json={},
        headers=headers,
    )
    assert r.status_code == 422
    data = r.json()
    assert data["error"] == "validation_error"
    assert data["detail"] == "Validation failed"
    assert isinstance(data["errors"], list)
    assert data["errors"]


# ===========================================================================
# Agents router
# ===========================================================================

_AGENT_PAYLOAD = {
    "name": "test-agent",
    "role": "developer",
    "provider": "openai",
    "model": "gpt-4o-mini",
    "adapter_type": "api",
}


@pytest.mark.asyncio
async def test_agent_create_list_get_update_deactivate(client: AsyncClient):
    headers = await _register_and_login(client)

    payload = {**_AGENT_PAYLOAD, "name": f"agent-{uuid.uuid4()}"}

    # Create
    r = await client.post("/api/v1/agents", json=payload, headers=headers)
    assert r.status_code == 201
    agent = r.json()
    aid = agent["id"]
    assert agent["name"] == payload["name"]

    # Get
    r = await client.get(f"/api/v1/agents/{aid}", headers=headers)
    assert r.status_code == 200
    assert r.json()["id"] == aid

    # List
    r = await client.get("/api/v1/agents", headers=headers)
    assert r.status_code == 200
    ids = [a["id"] for a in r.json()["items"]]
    assert aid in ids

    # Update
    r = await client.put(f"/api/v1/agents/{aid}", json={"role": "reviewer"}, headers=headers)
    assert r.status_code == 200
    assert r.json()["role"] == "reviewer"

    # Deactivate
    r = await client.delete(f"/api/v1/agents/{aid}", headers=headers)
    assert r.status_code == 204


@pytest.mark.asyncio
async def test_agent_get_context(client: AsyncClient):
    headers = await _register_and_login(client)
    payload = {**_AGENT_PAYLOAD, "name": f"ctx-agent-{uuid.uuid4()}"}
    r = await client.post("/api/v1/agents", json=payload, headers=headers)
    aid = r.json()["id"]

    r = await client.get(f"/api/v1/agents/{aid}/context", headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert data["agent"]["id"] == aid
    assert "current_tasks" in data


@pytest.mark.asyncio
async def test_agent_not_found(client: AsyncClient):
    headers = await _register_and_login(client)
    r = await client.get(f"/api/v1/agents/{uuid.uuid4()}", headers=headers)
    assert r.status_code == 404


# ===========================================================================
# Tasks router
# ===========================================================================

async def _create_project(client: AsyncClient, headers: dict, workspace: Path) -> str:
    r = await client.post(
        "/api/v1/projects",
        json={"name": f"proj-{uuid.uuid4()}", "workspace_path": str(workspace)},
        headers=headers,
    )
    assert r.status_code == 201
    return r.json()["id"]


@pytest.mark.asyncio
async def test_task_crud(client: AsyncClient, tmp_path: Path):
    headers = await _register_and_login(client)
    pid = await _create_project(client, headers, tmp_path)

    # Create
    r = await client.post(
        f"/api/v1/projects/{pid}/tasks",
        json={"title": "My Task"},
        headers=headers,
    )
    assert r.status_code == 201
    task = r.json()
    tid = task["id"]
    assert task["title"] == "My Task"

    # Get
    r = await client.get(f"/api/v1/projects/{pid}/tasks/{tid}", headers=headers)
    assert r.status_code == 200
    assert r.json()["id"] == tid

    # List
    r = await client.get(f"/api/v1/projects/{pid}/tasks", params={"project_id": pid}, headers=headers)
    assert r.status_code == 200
    ids = [t["id"] for t in r.json()["items"]]
    assert tid in ids

    # Update
    r = await client.put(
        f"/api/v1/projects/{pid}/tasks/{tid}",
        json={"title": "Updated Task"},
        headers=headers,
    )
    assert r.status_code == 200
    assert r.json()["title"] == "Updated Task"


@pytest.mark.asyncio
async def test_task_status_patch(client: AsyncClient, tmp_path: Path):
    headers = await _register_and_login(client)
    pid = await _create_project(client, headers, tmp_path)
    r = await client.post(
        f"/api/v1/projects/{pid}/tasks",
        json={"title": "Status Task"},
        headers=headers,
    )
    tid = r.json()["id"]

    # backlog -> ready
    r = await client.patch(
        f"/api/v1/projects/{pid}/tasks/{tid}/status",
        json={"status": "ready"},
        headers=headers,
    )
    assert r.status_code == 200

    # ready -> in_progress
    r = await client.patch(
        f"/api/v1/projects/{pid}/tasks/{tid}/status",
        json={"status": "in_progress"},
        headers=headers,
    )
    assert r.status_code == 200
    assert r.json()["status"] == "in_progress"


@pytest.mark.asyncio
async def test_task_status_patch_invalid_transition(client: AsyncClient, tmp_path: Path):
    headers = await _register_and_login(client)
    pid = await _create_project(client, headers, tmp_path)
    r = await client.post(
        f"/api/v1/projects/{pid}/tasks",
        json={"title": "Bad Transition Task"},
        headers=headers,
    )
    tid = r.json()["id"]

    # ready -> done is not valid
    r = await client.patch(
        f"/api/v1/projects/{pid}/tasks/{tid}/status",
        json={"status": "done"},
        headers=headers,
    )
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_task_assign(client: AsyncClient, tmp_path: Path):
    headers = await _register_and_login(client)
    pid = await _create_project(client, headers, tmp_path)

    # Create agent
    payload = {**_AGENT_PAYLOAD, "name": f"assign-agent-{uuid.uuid4()}"}
    r = await client.post("/api/v1/agents", json=payload, headers=headers)
    aid = r.json()["id"]

    # Create task
    r = await client.post(
        f"/api/v1/projects/{pid}/tasks",
        json={"title": "Assignable Task"},
        headers=headers,
    )
    tid = r.json()["id"]

    # Assign
    r = await client.post(
        f"/api/v1/projects/{pid}/tasks/{tid}/assign",
        json={"agent_id": aid},
        headers=headers,
    )
    assert r.status_code == 200
    assert r.json()["assigned_to"] == aid


@pytest.mark.asyncio
async def test_task_not_found(client: AsyncClient, tmp_path: Path):
    headers = await _register_and_login(client)
    pid = await _create_project(client, headers, tmp_path)
    r = await client.get(f"/api/v1/projects/{pid}/tasks/{uuid.uuid4()}", headers=headers)
    assert r.status_code == 404


# ===========================================================================
# Sessions router
# ===========================================================================

@pytest.mark.asyncio
async def test_session_list(client: AsyncClient, db_session: AsyncSession, test_project, test_agent):
    """GET /api/v1/sessions returns an empty page initially."""
    headers = await _register_and_login(client)
    r = await client.get("/api/v1/sessions", headers=headers)
    assert r.status_code == 200
    assert "items" in r.json()


@pytest.mark.asyncio
async def test_session_create_and_get(client: AsyncClient, runnable_project, test_agent):
    headers = await _register_and_login(client)

    with patch(
        "huddleroom.workers.task_runner.dispatch_session",
        new=AsyncMock(return_value="mocked-task"),
    ):
        r = await client.post(
            "/api/v1/sessions",
            json={
                "agent_id": str(test_agent.id),
                "project_id": str(runnable_project.id),
            },
            headers=headers,
        )
    assert r.status_code == 201
    session = r.json()
    sid = session["id"]

    r = await client.get(f"/api/v1/sessions/{sid}", headers=headers)
    assert r.status_code == 200
    assert r.json()["id"] == sid


@pytest.mark.asyncio
async def test_session_cancel(client: AsyncClient, db_session: AsyncSession, test_project, test_agent):
    headers = await _register_and_login(client)

    # Insert a pending session directly into the test DB
    raw = SessionModel(
        agent_id=test_agent.id,
        project_id=test_project.id,
        adapter_type="api",
        status="pending",
        input_context={},
        metadata_={},
    )
    db_session.add(raw)
    await db_session.flush()

    r = await client.post(f"/api/v1/sessions/{raw.id}/cancel", headers=headers)
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_session_get_output(client: AsyncClient, db_session: AsyncSession, test_project, test_agent):
    headers = await _register_and_login(client)
    raw = SessionModel(
        agent_id=test_agent.id,
        project_id=test_project.id,
        adapter_type="api",
        status="completed",
        input_context={},
        output="done result",
        metadata_={},
    )
    db_session.add(raw)
    await db_session.flush()

    r = await client.get(f"/api/v1/sessions/{raw.id}/output", headers=headers)
    assert r.status_code == 200
    assert r.json()["output"] == "done result"


@pytest.mark.asyncio
async def test_session_not_found(client: AsyncClient):
    headers = await _register_and_login(client)
    r = await client.get(f"/api/v1/sessions/{uuid.uuid4()}", headers=headers)
    assert r.status_code == 404


# ===========================================================================
# Channels + Messages routers
# ===========================================================================

@pytest.mark.asyncio
async def test_channel_create_list_get(client: AsyncClient, tmp_path: Path):
    headers = await _register_and_login(client)
    pid = await _create_project(client, headers, tmp_path)

    # Create
    r = await client.post(
        f"/api/v1/projects/{pid}/channels",
        json={"name": "general", "channel_type": "general"},
        headers=headers,
    )
    assert r.status_code == 201
    ch = r.json()
    cid = ch["id"]
    assert ch["name"] == "general"

    # List
    r = await client.get(f"/api/v1/projects/{pid}/channels", headers=headers)
    assert r.status_code == 200
    ids = [c["id"] for c in r.json()]
    assert cid in ids

    # Get
    r = await client.get(f"/api/v1/channels/{cid}", headers=headers)
    assert r.status_code == 200
    assert r.json()["id"] == cid


@pytest.mark.asyncio
async def test_channel_get_not_found(client: AsyncClient):
    headers = await _register_and_login(client)
    r = await client.get(f"/api/v1/channels/{uuid.uuid4()}", headers=headers)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_message_create_and_list(client: AsyncClient, tmp_path: Path):
    headers = await _register_and_login(client)
    pid = await _create_project(client, headers, tmp_path)

    # Create channel
    r = await client.post(
        f"/api/v1/projects/{pid}/channels",
        json={"name": "msg-ch", "channel_type": "general"},
        headers=headers,
    )
    cid = r.json()["id"]

    # Post message
    r = await client.post(
        f"/api/v1/channels/{cid}/messages",
        json={"content": "Hello!", "message_type": "text"},
        headers=headers,
    )
    assert r.status_code == 201
    msg = r.json()
    assert msg["content"] == "Hello!"

    # List messages
    r = await client.get(f"/api/v1/channels/{cid}/messages", headers=headers)
    assert r.status_code == 200
    contents = [m["content"] for m in r.json()["items"]]
    assert "Hello!" in contents


@pytest.mark.asyncio
async def test_message_channel_not_found(client: AsyncClient):
    headers = await _register_and_login(client)
    r = await client.get(f"/api/v1/channels/{uuid.uuid4()}/messages", headers=headers)
    assert r.status_code == 404


# ===========================================================================
# Knowledge router
# ===========================================================================

def _fake_embed(text: str) -> list[float]:
    return [1.0] + [0.0] * 1535


@pytest.mark.asyncio
async def test_knowledge_create_list_get_update_delete(client: AsyncClient, tmp_path: Path):
    headers = await _register_and_login(client)
    pid = await _create_project(client, headers, tmp_path)

    with patch(
        "huddleroom.services.knowledge_service.embedding_service.generate_embedding",
        new=AsyncMock(return_value=_fake_embed("x")),
    ):
        # Create
        r = await client.post(
            f"/api/v1/projects/{pid}/knowledge",
            json={"content": "Important fact", "content_type": "text"},
            headers=headers,
        )
        assert r.status_code == 201
        kid = r.json()["id"]

        # List
        r = await client.get(f"/api/v1/projects/{pid}/knowledge", headers=headers)
        assert r.status_code == 200
        ids = [k["id"] for k in r.json()["items"]]
        assert kid in ids

        # Get
        r = await client.get(f"/api/v1/knowledge/{kid}", headers=headers)
        assert r.status_code == 200
        assert r.json()["id"] == kid

        # Update
        r = await client.put(
            f"/api/v1/knowledge/{kid}",
            json={"content": "Updated fact"},
            headers=headers,
        )
        assert r.status_code == 200
        assert r.json()["content"] == "Updated fact"

        # Delete
        r = await client.delete(f"/api/v1/knowledge/{kid}", headers=headers)
        assert r.status_code == 204


@pytest.mark.asyncio
async def test_knowledge_search(client: AsyncClient, tmp_path: Path):
    headers = await _register_and_login(client)
    pid = await _create_project(client, headers, tmp_path)

    fake_vec = [1.0] + [0.0] * 1535
    with patch(
        "huddleroom.services.knowledge_service.embedding_service.generate_embedding",
        new=AsyncMock(return_value=fake_vec),
    ):
        await client.post(
            f"/api/v1/projects/{pid}/knowledge",
            json={"content": "Searchable content", "content_type": "text"},
            headers=headers,
        )
        r = await client.post(
            f"/api/v1/projects/{pid}/knowledge/search",
            json={"query": "searchable", "min_relevance_score": 0.0},
            headers=headers,
        )
    assert r.status_code == 200
    assert "results" in r.json()
    assert len(r.json()["results"]) >= 1
