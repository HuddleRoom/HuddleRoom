import uuid
import pytest


@pytest.mark.asyncio
async def test_create_rule(client, test_project):
    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/rules",
        json={
            "name": "route-errors",
            "on_event": "task.failed",
            "conditions": {"status": "failed"},
            "actions": {"notify": "admin"},
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "route-errors"
    assert data["on_event"] == "task.failed"
    assert data["enabled"] is True
    assert data["priority"] == 0


@pytest.mark.asyncio
async def test_list_rules(client, test_project):
    await client.post(
        f"/api/v1/projects/{test_project.id}/rules",
        json={"name": "r1", "on_event": "e1", "conditions": {}, "actions": {}},
    )
    resp = await client.get(f"/api/v1/projects/{test_project.id}/rules")
    assert resp.status_code == 200
    assert len(resp.json()["items"]) >= 1


@pytest.mark.asyncio
async def test_list_rules_invalid_project(client):
    resp = await client.get(f"/api/v1/projects/{uuid.uuid4()}/rules")
    assert resp.status_code == 404
    assert "Project not found" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_list_rules_filter_enabled(client, test_project):
    await client.post(
        f"/api/v1/projects/{test_project.id}/rules",
        json={"name": "disabled-rule", "on_event": "e1", "conditions": {}, "actions": {}, "enabled": False},
    )
    resp = await client.get(f"/api/v1/projects/{test_project.id}/rules?enabled=false")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert all(not item["enabled"] for item in items)


@pytest.mark.asyncio
async def test_get_rule(client, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/rules",
        json={"name": "r-get", "on_event": "e1", "conditions": {}, "actions": {}},
    )
    rule_id = create_resp.json()["id"]
    resp = await client.get(f"/api/v1/projects/{test_project.id}/rules/{rule_id}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "r-get"


@pytest.mark.asyncio
async def test_get_rule_not_found(client, test_project):
    resp = await client.get(f"/api/v1/projects/{test_project.id}/rules/{uuid.uuid4()}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_rule(client, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/rules",
        json={"name": "r-update", "on_event": "e1", "conditions": {}, "actions": {}},
    )
    rule_id = create_resp.json()["id"]
    resp = await client.patch(
        f"/api/v1/projects/{test_project.id}/rules/{rule_id}",
        json={"name": "r-updated", "enabled": False},
    )
    assert resp.status_code == 200
    assert resp.json()["name"] == "r-updated"
    assert resp.json()["enabled"] is False


@pytest.mark.asyncio
async def test_delete_rule(client, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/rules",
        json={"name": "r-delete", "on_event": "e1", "conditions": {}, "actions": {}},
    )
    rule_id = create_resp.json()["id"]
    resp = await client.delete(f"/api/v1/projects/{test_project.id}/rules/{rule_id}")
    assert resp.status_code == 204
    get_resp = await client.get(f"/api/v1/projects/{test_project.id}/rules/{rule_id}")
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_reorder_rules(client, test_project):
    r1 = await client.post(
        f"/api/v1/projects/{test_project.id}/rules",
        json={"name": "r-first", "on_event": "e1", "conditions": {}, "actions": {}},
    )
    r2 = await client.post(
        f"/api/v1/projects/{test_project.id}/rules",
        json={"name": "r-second", "on_event": "e2", "conditions": {}, "actions": {}},
    )
    id1 = r1.json()["id"]
    id2 = r2.json()["id"]
    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/rules/reorder",
        json={"rule_ids": [id2, id1]},
    )
    assert resp.status_code == 200

    get1 = await client.get(f"/api/v1/projects/{test_project.id}/rules/{id1}")
    get2 = await client.get(f"/api/v1/projects/{test_project.id}/rules/{id2}")
    assert get2.json()["priority"] == 0
    assert get1.json()["priority"] == 1


@pytest.mark.asyncio
async def test_reorder_rules_partial_rejected(client, test_project):
    r1 = await client.post(
        f"/api/v1/projects/{test_project.id}/rules",
        json={"name": "r-partial-1", "on_event": "e1", "conditions": {}, "actions": {}},
    )
    await client.post(
        f"/api/v1/projects/{test_project.id}/rules",
        json={"name": "r-partial-2", "on_event": "e2", "conditions": {}, "actions": {}},
    )
    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/rules/reorder",
        json={"rule_ids": [r1.json()["id"]]},
    )
    assert resp.status_code == 400
    assert "exactly all rules" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_create_rule_invalid_project(client):
    resp = await client.post(
        f"/api/v1/projects/{uuid.uuid4()}/rules",
        json={"name": "r-bad-proj", "on_event": "e1", "conditions": {}, "actions": {}},
    )
    assert resp.status_code == 404
    assert "Project not found" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_reorder_rules_duplicates_rejected(client, test_project):
    r1 = await client.post(
        f"/api/v1/projects/{test_project.id}/rules",
        json={"name": "r-dup", "on_event": "e1", "conditions": {}, "actions": {}},
    )
    rid = r1.json()["id"]
    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/rules/reorder",
        json={"rule_ids": [rid, rid]},
    )
    assert resp.status_code == 400
    assert "duplicates" in resp.json()["detail"]
