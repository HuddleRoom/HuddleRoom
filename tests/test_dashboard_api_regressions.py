"""Tests for dashboard API regressions:
  dry-run endpoint
  cost-metric date filters
  graph runs pagination (same-timestamp tiebreak)
  user is_active filtering
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.graph import Graph, GraphRun
from huddleroom.models.user import User
from huddleroom.security import hash_password


# ---------------------------------------------------------------------------
# Dry-run endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dry_run_match(client, test_project):
    """Conditions that exactly match the payload return matched=true."""
    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/rules/dry-run",
        json={
            "conditions": {"status": "failed"},
            "event_payload": {"status": "failed"},
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"matched": True}


@pytest.mark.asyncio
async def test_dry_run_no_match(client, test_project):
    """Conditions that do not match the payload return matched=false."""
    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/rules/dry-run",
        json={
            "conditions": {"status": "failed"},
            "event_payload": {"status": "succeeded"},
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"matched": False}


@pytest.mark.asyncio
async def test_dry_run_nonexistent_project(client):
    """A non-existent project UUID returns 404."""
    resp = await client.post(
        f"/api/v1/projects/{uuid.uuid4()}/rules/dry-run",
        json={
            "conditions": {"status": "failed"},
            "event_payload": {"status": "failed"},
        },
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Cost-metric date filters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cost_metrics_start_after_end(client, test_project):
    """start_date after end_date should return 400."""
    resp = await client.get(
        f"/api/v1/projects/{test_project.id}/cost-metrics",
        params={"start_date": "2025-06-01", "end_date": "2025-01-01"},
    )
    assert resp.status_code == 400
    assert "start_date" in resp.json()["detail"].lower() or "after" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_cost_metrics_range_too_wide(client, test_project):
    """Date range exceeding 365 days should return 400."""
    start = date(2024, 1, 1)
    end = start + timedelta(days=366)
    resp = await client.get(
        f"/api/v1/projects/{test_project.id}/cost-metrics",
        params={"start_date": start.isoformat(), "end_date": end.isoformat()},
    )
    assert resp.status_code == 400
    assert "365" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_cost_metrics_valid_range(client, test_project):
    """A valid date range (within 365 days, start <= end) should return 200."""
    start = date(2025, 1, 1)
    end = date(2025, 3, 31)
    resp = await client.get(
        f"/api/v1/projects/{test_project.id}/cost-metrics",
        params={"start_date": start.isoformat(), "end_date": end.isoformat()},
    )
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# Graph runs pagination (same-timestamp tiebreak)
# ---------------------------------------------------------------------------


async def _make_graph(db: AsyncSession, project_id: uuid.UUID) -> Graph:
    """Helper: create a minimal Graph for a given project."""
    graph = Graph(
        project_id=project_id,
        name=f"graph-{uuid.uuid4()}",
        version="1.0",
        definition={},
        triggers=[],
        is_active=True,
    )
    db.add(graph)
    await db.flush()
    return graph


async def _make_run(
    db: AsyncSession,
    project_id: uuid.UUID,
    graph_id: uuid.UUID,
    created_at: datetime,
) -> GraphRun:
    """Helper: create a GraphRun with an explicit created_at."""
    run = GraphRun(
        graph_id=graph_id,
        project_id=project_id,
        current_node="start",
        status="active",
        actor_assignments={},
        context={},
        started_at=datetime.now(timezone.utc),
    )
    db.add(run)
    await db.flush()
    # Override created_at after flush
    run.created_at = created_at
    await db.flush()
    return run


@pytest.mark.asyncio
async def test_graph_runs_cursor_tiebreak(client, test_project, db_session):
    """Two runs with identical created_at must not produce duplicates across pages."""
    graph = await _make_graph(db_session, test_project.id)

    # Use a fixed timestamp for both runs
    shared_ts = datetime(2025, 1, 15, 12, 0, 0)

    run1 = await _make_run(db_session, test_project.id, graph.id, shared_ts)
    run2 = await _make_run(db_session, test_project.id, graph.id, shared_ts)

    # Page 1: limit=1
    resp1 = await client.get(
        f"/api/v1/projects/{test_project.id}/graph-runs",
        params={"limit": 1},
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert len(data1["items"]) == 1
    next_cursor = data1.get("next_cursor")
    assert next_cursor is not None, "Expected a next_cursor for page 2"

    # Page 2: use cursor
    resp2 = await client.get(
        f"/api/v1/projects/{test_project.id}/graph-runs",
        params={"limit": 1, "cursor": next_cursor},
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert len(data2["items"]) == 1

    # The two pages must have different items
    ids_page1 = {item["id"] for item in data1["items"]}
    ids_page2 = {item["id"] for item in data2["items"]}
    assert ids_page1.isdisjoint(ids_page2), "Duplicate item found across pages with same-timestamp cursor"


@pytest.mark.asyncio
async def test_graph_runs_invalid_cursor(client, test_project):
    """A garbage cursor string should return 400."""
    resp = await client.get(
        f"/api/v1/projects/{test_project.id}/graph-runs",
        params={"cursor": "garbage"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# User is_active filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("filter_param,expect_active,user_suffix", [
    ({"is_active": "true"}, True, "active"),
    ({"is_active": "false"}, False, "inactive"),
    ({}, True, "default"),
])
async def test_user_list_is_active_filter(client, db_session, filter_param, expect_active, user_suffix):
    """Test is_active filtering: explicit true/false and default (active only)."""
    active_user = User(
        email=f"active-{user_suffix}-{uuid.uuid4()}@example.com",
        hashed_password=hash_password("pw"),
        display_name=f"Active User {user_suffix}",
        role="member",
        is_active=True,
    )
    inactive_user = User(
        email=f"inactive-{user_suffix}-{uuid.uuid4()}@example.com",
        hashed_password=hash_password("pw"),
        display_name=f"Inactive User {user_suffix}",
        role="member",
        is_active=False,
    )
    db_session.add(active_user)
    db_session.add(inactive_user)
    await db_session.flush()

    resp = await client.get("/api/v1/users", params=filter_param)
    assert resp.status_code == 200
    users = resp.json()
    returned_ids = {u["id"] for u in users}

    if expect_active:
        assert all(u["is_active"] is True for u in users), "Should only return active users"
        assert str(active_user.id) in returned_ids
        assert str(inactive_user.id) not in returned_ids
    else:
        assert all(u["is_active"] is False for u in users), "Should only return inactive users"
        assert str(inactive_user.id) in returned_ids
        assert str(active_user.id) not in returned_ids


# ---------------------------------------------------------------------------
# graph-runs/count filtered by status query param
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_runs_count_status_filter(client, test_project, db_session):
    """?status=active must filter; total count must differ from filtered count."""
    graph = await _make_graph(db_session, test_project.id)
    ts = datetime.now(timezone.utc)
    run_active = GraphRun(
        graph_id=graph.id,
        project_id=test_project.id,
        current_node="start",
        status="active",
        actor_assignments={},
        context={},
        started_at=ts,
    )
    run_concluded = GraphRun(
        graph_id=graph.id,
        project_id=test_project.id,
        current_node="end",
        status="concluded",
        actor_assignments={},
        context={},
        started_at=ts,
    )
    db_session.add(run_active)
    db_session.add(run_concluded)
    await db_session.flush()

    resp_all = await client.get(f"/api/v1/projects/{test_project.id}/graph-runs/count")
    resp_active = await client.get(
        f"/api/v1/projects/{test_project.id}/graph-runs/count",
        params={"status": "active"},
    )
    resp_concluded = await client.get(
        f"/api/v1/projects/{test_project.id}/graph-runs/count",
        params={"status": "concluded"},
    )

    assert resp_all.status_code == 200
    assert resp_active.status_code == 200
    assert resp_concluded.status_code == 200

    count_all = resp_all.json()["count"]
    count_active = resp_active.json()["count"]
    count_concluded = resp_concluded.json()["count"]

    assert count_active >= 1
    assert count_concluded >= 1
    assert count_all >= count_active + count_concluded
