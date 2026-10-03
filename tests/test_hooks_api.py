import uuid
import pytest


@pytest.mark.asyncio
async def test_create_hook(client, test_project):
    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/hooks",
        json={
            "name": "auto-assign",
            "trigger_event": "task.created",
            "code": "def run(event): pass",
            "description": "Auto-assign tasks",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "auto-assign"
    assert data["status"] == "proposed"
    assert data["execution_count"] == 0
    assert data["error_count"] == 0


@pytest.mark.asyncio
async def test_list_hooks(client, test_project):
    await client.post(
        f"/api/v1/projects/{test_project.id}/hooks",
        json={"name": "h1", "trigger_event": "e1", "code": "pass"},
    )
    resp = await client.get(f"/api/v1/projects/{test_project.id}/hooks")
    assert resp.status_code == 200
    assert len(resp.json()["items"]) >= 1


@pytest.mark.asyncio
async def test_list_hooks_invalid_project(client):
    resp = await client.get(f"/api/v1/projects/{uuid.uuid4()}/hooks")
    assert resp.status_code == 404
    assert "Project not found" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_list_hooks_filter_status(client, test_project):
    resp = await client.get(f"/api/v1/projects/{test_project.id}/hooks?status=active")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_get_hook(client, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/hooks",
        json={"name": "h-get", "trigger_event": "e1", "code": "pass"},
    )
    hook_id = create_resp.json()["id"]
    resp = await client.get(f"/api/v1/projects/{test_project.id}/hooks/{hook_id}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "h-get"


@pytest.mark.asyncio
async def test_update_hook_fields(client, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/hooks",
        json={"name": "h-update", "trigger_event": "e1", "code": "pass"},
    )
    hook_id = create_resp.json()["id"]
    resp = await client.patch(
        f"/api/v1/projects/{test_project.id}/hooks/{hook_id}",
        json={"name": "h-updated", "code": "def run(): return True"},
    )
    assert resp.status_code == 200
    assert resp.json()["name"] == "h-updated"


@pytest.mark.asyncio
async def test_hook_status_transition_valid(client, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/hooks",
        json={"name": "h-trans", "trigger_event": "e1", "code": "pass"},
    )
    hook_id = create_resp.json()["id"]
    resp = await client.patch(
        f"/api/v1/projects/{test_project.id}/hooks/{hook_id}",
        json={"status": "active"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "active"


@pytest.mark.asyncio
async def test_hook_status_transition_invalid(client, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/hooks",
        json={"name": "h-bad-trans", "trigger_event": "e1", "code": "pass"},
    )
    hook_id = create_resp.json()["id"]
    resp = await client.patch(
        f"/api/v1/projects/{test_project.id}/hooks/{hook_id}",
        json={"status": "shadow"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_delete_hook(client, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/hooks",
        json={"name": "h-delete", "trigger_event": "e1", "code": "pass"},
    )
    hook_id = create_resp.json()["id"]
    resp = await client.delete(f"/api/v1/projects/{test_project.id}/hooks/{hook_id}")
    assert resp.status_code == 204
    get_resp = await client.get(f"/api/v1/projects/{test_project.id}/hooks/{hook_id}")
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_create_hook_invalid_project(client):
    resp = await client.post(
        f"/api/v1/projects/{uuid.uuid4()}/hooks",
        json={"name": "h-bad", "trigger_event": "e1", "code": "pass"},
    )
    assert resp.status_code == 404
    assert "Project not found" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_hooks_pagination_same_timestamp(client, test_project, db_session):
    """Create multiple hooks with same timestamp, verify composite cursor doesn't skip."""
    from huddleroom.models.hook import Hook
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    hooks = []
    for i in range(5):
        h = Hook(
            project_id=test_project.id,
            name=f"h-page-{i}",
            code="pass",
            trigger_event="e1",
            status="proposed",
            created_at=now,
            updated_at=now,
        )
        db_session.add(h)
        hooks.append(h)
    await db_session.flush()

    resp1 = await client.get(f"/api/v1/projects/{test_project.id}/hooks?limit=2")
    assert resp1.status_code == 200, f"Response error: {resp1.json()}"
    page1 = resp1.json()
    assert len(page1["items"]) == 2, f"Expected 2 items, got {len(page1['items'])}"
    assert page1["next_cursor"] is not None

    resp2 = await client.get(
        f"/api/v1/projects/{test_project.id}/hooks?limit=2&cursor={page1['next_cursor']}"
    )
    assert resp2.status_code == 200, f"Response error: {resp2.json()}"
    page2 = resp2.json()
    assert len(page2["items"]) == 2, f"Expected 2 items in page 2, got {len(page2['items'])} with cursor {page1['next_cursor']}"

    resp3 = await client.get(
        f"/api/v1/projects/{test_project.id}/hooks?limit=2&cursor={page2['next_cursor']}"
    )
    assert resp3.status_code == 200, f"Response error: {resp3.json()}"
    page3 = resp3.json()
    assert len(page3["items"]) == 1
    assert page3["next_cursor"] is None

    all_ids = [h["id"] for h in page1["items"] + page2["items"] + page3["items"]]
    assert len(all_ids) == len(set(all_ids)) == 5
