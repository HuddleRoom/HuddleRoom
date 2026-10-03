import uuid
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_list_project_memories(client, auth_headers, test_project, test_agent, db_session):
    from huddleroom.services.memory_service import MemoryService
    svc = MemoryService()
    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=[0.1] * 10):
        await svc.write(db_session, test_agent.id, test_project.id, "test memory", ["decision"], True, "project")
    await db_session.flush()

    resp = await client.get(f"/api/v1/projects/{test_project.id}/memory", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert len(data["items"]) >= 1


@pytest.mark.asyncio
async def test_get_memory(client, auth_headers, test_project, test_agent, db_session):
    from huddleroom.services.memory_service import MemoryService
    svc = MemoryService()
    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=[0.1] * 10):
        item = await svc.write(db_session, test_agent.id, test_project.id, "get me", [], True, "project")
    await db_session.flush()

    resp = await client.get(f"/api/v1/projects/{test_project.id}/memory/{item.id}", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["content"] == "get me"


@pytest.mark.asyncio
async def test_get_memory_not_found(client, auth_headers, test_project):
    resp = await client.get(f"/api/v1/projects/{test_project.id}/memory/{uuid.uuid4()}", headers=auth_headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_search_memories(client, auth_headers, test_project, test_agent, db_session):
    from huddleroom.services.memory_service import MemoryService
    svc = MemoryService()
    embedding = [1.0, 0.0, 0.0]
    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=embedding):
        await svc.write(db_session, test_agent.id, test_project.id, "searchable", ["decision"], True, "project")
    await db_session.flush()

    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=embedding):
        resp = await client.post(
            f"/api/v1/projects/{test_project.id}/memory/search",
            json={"query": "test"},
            headers=auth_headers,
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data


@pytest.mark.asyncio
async def test_delete_project_memory(client, auth_headers, test_project, test_agent, db_session):
    from huddleroom.services.memory_service import MemoryService
    svc = MemoryService()
    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=[0.1] * 10):
        item = await svc.write(db_session, test_agent.id, test_project.id, "to delete", [], True, "project")
    await db_session.flush()

    resp = await client.delete(f"/api/v1/projects/{test_project.id}/memory/{item.id}", headers=auth_headers)
    assert resp.status_code == 204


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_delete_global_memory_non_admin(client, auth_headers, test_agent, db_session):
    from huddleroom.services.memory_service import MemoryService
    svc = MemoryService()
    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=[0.1] * 10):
        item = await svc.write(db_session, test_agent.id, None, "global mem", [], True, "global")
    await db_session.flush()

    resp = await client.delete(f"/api/v1/memory/{item.id}", headers=auth_headers)
    assert resp.status_code == 403


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_delete_global_memory_admin(client, test_agent, db_session):
    from huddleroom.services.memory_service import MemoryService
    from huddleroom.models.user import User
    from huddleroom.security import create_access_token, hash_password

    # Create admin user
    admin_user = User(
        email=f"admin-{uuid.uuid4()}@example.com",
        hashed_password=hash_password("testpassword"),
        display_name="Admin User",
        role="admin",
    )
    db_session.add(admin_user)
    await db_session.flush()

    # Create auth headers for admin user
    token = create_access_token({"sub": str(admin_user.id)})
    admin_headers = {"Authorization": f"Bearer {token}"}

    svc = MemoryService()
    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=[0.1] * 10):
        item = await svc.write(db_session, test_agent.id, None, "global mem", [], True, "global")
    await db_session.flush()

    resp = await client.delete(f"/api/v1/memory/{item.id}", headers=admin_headers)
    assert resp.status_code == 204


@pytest.mark.asyncio
async def test_list_global_memories(client, auth_headers, test_agent, db_session):
    from huddleroom.services.memory_service import MemoryService
    svc = MemoryService()
    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=[0.1] * 10):
        await svc.write(db_session, test_agent.id, None, "global mem", [], True, "global")
    await db_session.flush()

    resp = await client.get("/api/v1/memory?scope=global", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
