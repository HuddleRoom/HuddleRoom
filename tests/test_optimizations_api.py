import uuid
import pytest


@pytest.mark.asyncio
async def test_list_patterns_empty(client, test_project):
    resp = await client.get(f"/api/v1/projects/{test_project.id}/patterns")
    assert resp.status_code == 200
    assert resp.json()["items"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("method,endpoint", [
    ("GET", "patterns"),
    ("GET", "optimizations"),
    ("GET", "cost-metrics"),
    ("POST", "optimizations"),
])
async def test_invalid_project_endpoints(client, method, endpoint):
    project_id = uuid.uuid4()
    if method == "GET":
        resp = await client.get(f"/api/v1/projects/{project_id}/{endpoint}")
    else:
        resp = await client.post(
            f"/api/v1/projects/{project_id}/{endpoint}",
            json={"type": "hook", "generated_code": "pass"},
        )
    assert resp.status_code == 404
    assert "Project not found" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_get_pattern_not_found(client, test_project):
    resp = await client.get(f"/api/v1/projects/{test_project.id}/patterns/{uuid.uuid4()}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_optimization(client, test_project):
    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/optimizations",
        json={
            "type": "hook",
            "generated_code": "def optimize(): pass",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["type"] == "hook"
    assert data["status"] == "proposed"
    assert data["fire_count"] == 0
    assert data["error_rate"] == 0.0


@pytest.mark.asyncio
async def test_create_optimization_invalid_type(client, test_project):
    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/optimizations",
        json={"type": "invalid", "generated_code": "pass"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_list_optimizations(client, test_project):
    await client.post(
        f"/api/v1/projects/{test_project.id}/optimizations",
        json={"type": "rule", "generated_code": "pass"},
    )
    resp = await client.get(f"/api/v1/projects/{test_project.id}/optimizations")
    assert resp.status_code == 200
    assert len(resp.json()["items"]) >= 1




@pytest.mark.asyncio
async def test_optimization_status_transition_valid(client, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/optimizations",
        json={"type": "hook", "generated_code": "pass"},
    )
    opt_id = create_resp.json()["id"]
    resp = await client.patch(
        f"/api/v1/projects/{test_project.id}/optimizations/{opt_id}",
        json={"status": "approved"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "approved"
    resp = await client.patch(
        f"/api/v1/projects/{test_project.id}/optimizations/{opt_id}",
        json={"status": "active"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "active"


@pytest.mark.asyncio
async def test_optimization_status_transition_invalid(client, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/optimizations",
        json={"type": "hook", "generated_code": "pass"},
    )
    opt_id = create_resp.json()["id"]
    resp = await client.patch(
        f"/api/v1/projects/{test_project.id}/optimizations/{opt_id}",
        json={"status": "active"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_optimization_rejected_terminal(client, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/optimizations",
        json={"type": "shortcut", "generated_code": "pass"},
    )
    opt_id = create_resp.json()["id"]
    await client.patch(
        f"/api/v1/projects/{test_project.id}/optimizations/{opt_id}",
        json={"status": "rejected"},
    )
    resp = await client.patch(
        f"/api/v1/projects/{test_project.id}/optimizations/{opt_id}",
        json={"status": "proposed"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_delete_optimization(client, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/optimizations",
        json={"type": "hook", "generated_code": "pass"},
    )
    opt_id = create_resp.json()["id"]
    resp = await client.delete(f"/api/v1/projects/{test_project.id}/optimizations/{opt_id}")
    assert resp.status_code == 204


@pytest.mark.asyncio
async def test_list_cost_metrics_empty(client, test_project):
    resp = await client.get(f"/api/v1/projects/{test_project.id}/cost-metrics")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_optimizations_pagination_same_timestamp(client, test_project, db_session):
    """Composite cursor handles same-timestamp items without skips or duplicates."""
    from huddleroom.models.optimization import Optimization
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    for i in range(5):
        opt = Optimization(
            project_id=test_project.id,
            type="hook",
            generated_code=f"pass # {i}",
            status="proposed",
            created_at=now,
            updated_at=now,
        )
        db_session.add(opt)
    await db_session.flush()

    resp1 = await client.get(f"/api/v1/projects/{test_project.id}/optimizations?limit=2")
    assert resp1.status_code == 200
    page1 = resp1.json()
    assert len(page1["items"]) == 2
    assert page1["next_cursor"] is not None

    resp2 = await client.get(
        f"/api/v1/projects/{test_project.id}/optimizations?limit=2&cursor={page1['next_cursor']}"
    )
    page2 = resp2.json()
    assert len(page2["items"]) == 2, f"Expected 2 items in page 2, got {len(page2['items'])}"

    resp3 = await client.get(
        f"/api/v1/projects/{test_project.id}/optimizations?limit=2&cursor={page2['next_cursor']}"
    )
    page3 = resp3.json()
    assert len(page3["items"]) == 1
    assert page3["next_cursor"] is None

    all_ids = [o["id"] for o in page1["items"] + page2["items"] + page3["items"]]
    assert len(all_ids) == len(set(all_ids)) == 5
