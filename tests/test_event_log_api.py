import pytest
import uuid
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_post_event_returns_201(client: AsyncClient, test_project, auth_headers):
    resp = await client.post(
        "/api/v1/events",
        json={
            "project_id": str(test_project.id),
            "event_type": "custom.event",
            "payload": {"key": "value"},
            "source": "agent",
        },
        headers=auth_headers,
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["event_type"] == "custom.event"
    assert "id" in data
    assert "emitted_at" in data


@pytest.mark.asyncio
async def test_get_events_returns_list(client: AsyncClient, test_project, auth_headers):
    # Post an event first
    await client.post(
        "/api/v1/events",
        json={
            "project_id": str(test_project.id),
            "event_type": "task.created",
            "payload": {},
            "source": "system",
        },
        headers=auth_headers,
    )
    resp = await client.get(
        f"/api/v1/events?project_id={test_project.id}",
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert len(data["items"]) >= 1
    assert data["items"][0]["event_type"] == "task.created"


@pytest.mark.asyncio
async def test_get_events_filters_by_event_type(client: AsyncClient, test_project, auth_headers):
    for etype in ["task.created", "session.started"]:
        await client.post(
            "/api/v1/events",
            json={"project_id": str(test_project.id), "event_type": etype, "payload": {}, "source": "system"},
            headers=auth_headers,
        )
    resp = await client.get(
        f"/api/v1/events?project_id={test_project.id}&event_type=session.started",
        headers=auth_headers,
    )
    data = resp.json()
    assert all(i["event_type"] == "session.started" for i in data["items"])
