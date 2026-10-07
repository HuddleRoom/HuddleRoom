import uuid
from datetime import datetime, timezone

import pytest

from huddleroom.models.graph import Graph, GraphRun, GraphRunStep


SAMPLE_DEFINITION = {
    "nodes": {
        "start": {"edges": [{"to": "end", "name": "finish"}]},
        "end": {},
    },
    "start_node": "start",
    "terminal_nodes": {"success": ["end"], "failure": []},
}


@pytest.mark.asyncio
async def test_graph_response_includes_definition(client, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/graphs",
        json={
            "name": "test-def",
            "definition": SAMPLE_DEFINITION,
            "triggers": [],
        },
    )
    assert create_resp.status_code == 201
    data = create_resp.json()
    assert "definition" in data
    assert data["definition"]["start_node"] == "start"
    assert data["definition"]["nodes"]["start"]["edges"][0]["name"] == "finish"


@pytest.mark.asyncio
async def test_update_graph(client, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/graphs",
        json={"name": "graph-update", "definition": SAMPLE_DEFINITION, "triggers": []},
    )
    graph_id = create_resp.json()["id"]
    new_def = {
        "nodes": {"a": {"edges": [{"to": "b", "name": "go"}]}, "b": {}},
        "start_node": "a",
        "terminal_nodes": {"success": ["b"], "failure": []},
    }
    resp = await client.put(
        f"/api/v1/projects/{test_project.id}/graphs/{graph_id}",
        json={"definition": new_def, "description": "updated"},
    )
    assert resp.status_code == 200
    assert resp.json()["description"] == "updated"
    assert resp.json()["definition"]["start_node"] == "a"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["active", "paused"])
async def test_update_graph_blocked_by_non_terminal_run(client, test_project, db_session, status):
    from huddleroom.models.graph import Graph, GraphRun
    graph = Graph(
        project_id=test_project.id,
        name=f"graph-blocked-{status}",
        definition=SAMPLE_DEFINITION,
        triggers=[],
    )
    db_session.add(graph)
    await db_session.flush()
    run = GraphRun(
        graph_id=graph.id,
        project_id=test_project.id,
        current_node="start",
        status=status,
    )
    db_session.add(run)
    await db_session.flush()
    resp = await client.put(
        f"/api/v1/projects/{test_project.id}/graphs/{graph.id}",
        json={"description": "should fail"},
    )
    assert resp.status_code == 409
    assert "non-terminal" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_update_graph_allowed_with_terminal_runs(client, test_project, db_session):
    from huddleroom.models.graph import Graph, GraphRun
    graph = Graph(
        project_id=test_project.id,
        name="graph-terminal-ok",
        definition=SAMPLE_DEFINITION,
        triggers=[],
    )
    db_session.add(graph)
    await db_session.flush()
    run = GraphRun(
        graph_id=graph.id,
        project_id=test_project.id,
        current_node="end",
        status="completed",
    )
    db_session.add(run)
    await db_session.flush()
    resp = await client.put(
        f"/api/v1/projects/{test_project.id}/graphs/{graph.id}",
        json={"description": "allowed with completed run"},
    )
    assert resp.status_code == 200
    assert resp.json()["description"] == "allowed with completed run"


@pytest.mark.asyncio
async def test_activate_graph(client, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/graphs",
        json={"name": "graph-activate", "definition": SAMPLE_DEFINITION, "triggers": []},
    )
    graph_id = create_resp.json()["id"]
    # Deactivate
    await client.delete(f"/api/v1/projects/{test_project.id}/graphs/{graph_id}")
    # Verify inactive via include_inactive
    list_resp = await client.get(f"/api/v1/projects/{test_project.id}/graphs?include_inactive=true")
    found = [p for p in list_resp.json() if p["id"] == graph_id]
    assert len(found) == 1
    assert found[0]["is_active"] is False
    # Reactivate
    resp = await client.post(f"/api/v1/projects/{test_project.id}/graphs/{graph_id}/activate")
    assert resp.status_code == 200
    assert resp.json()["is_active"] is True


@pytest.mark.asyncio
async def test_list_graphs_include_inactive(client, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/graphs",
        json={"name": "graph-inactive-list", "definition": SAMPLE_DEFINITION, "triggers": []},
    )
    graph_id = create_resp.json()["id"]
    await client.delete(f"/api/v1/projects/{test_project.id}/graphs/{graph_id}")
    # Without include_inactive — should not appear
    resp = await client.get(f"/api/v1/projects/{test_project.id}/graphs")
    ids = [p["id"] for p in resp.json()]
    assert graph_id not in ids
    # With include_inactive — should appear
    resp = await client.get(f"/api/v1/projects/{test_project.id}/graphs?include_inactive=true")
    ids = [p["id"] for p in resp.json()]
    assert graph_id in ids


@pytest.mark.asyncio
async def test_list_graphs_invalid_project(client):
    resp = await client.get(f"/api/v1/projects/{uuid.uuid4()}/graphs")
    assert resp.status_code == 404
    assert "Project not found" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_list_graph_runs_filter_by_graph_id(client, test_project):
    resp = await client.get(
        f"/api/v1/projects/{test_project.id}/graph-runs?graph_id={uuid.uuid4()}"
    )
    assert resp.status_code == 200
    assert resp.json()["items"] == []


@pytest.mark.asyncio
async def test_list_graph_runs_invalid_project(client):
    resp = await client.get(f"/api/v1/projects/{uuid.uuid4()}/graph-runs")
    assert resp.status_code == 404
    assert "Project not found" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_graph_run_step_response_fields(client, test_project, db_session):
    stepped_at = datetime(2026, 6, 24, 12, 30, tzinfo=timezone.utc)
    graph = Graph(
        project_id=test_project.id,
        name=f"graph-step-fields-{uuid.uuid4()}",
        definition=SAMPLE_DEFINITION,
        triggers=[],
    )
    db_session.add(graph)
    await db_session.flush()
    run = GraphRun(
        graph_id=graph.id,
        project_id=test_project.id,
        current_node="start",
        status="active",
        actor_assignments={},
        context={},
    )
    db_session.add(run)
    await db_session.flush()
    step = GraphRunStep(
        graph_run_id=run.id,
        from_node="start",
        to_node="end",
        edge_name="finish",
        actions_executed=[],
        stepped_at=stepped_at,
    )
    db_session.add(step)
    await db_session.flush()

    resp = await client.get(
        f"/api/v1/projects/{test_project.id}/graph-runs/{run.id}/steps"
    )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["edge_name"] == "finish"
    assert data[0]["stepped_at"] == stepped_at.isoformat().replace("+00:00", "Z")
    assert data[0]["from_node"] == "start"
    assert data[0]["to_node"] == "end"
    assert "event_type" not in data[0]
    assert "created_at" not in data[0]


_LEGACY_DEFINITION = {
    "states": {"a": {}},
    "initial_state": "a",
    "terminal_states": {"success": ["a"]},
}


@pytest.mark.asyncio
async def test_create_graph_with_legacy_keys_rejected(client, test_project):
    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/graphs",
        json={"name": "legacy-create", "definition": _LEGACY_DEFINITION, "triggers": []},
    )
    assert resp.status_code == 422
    assert "legacy" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_update_graph_with_legacy_keys_rejected(client, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/graphs",
        json={"name": "legacy-update", "definition": SAMPLE_DEFINITION, "triggers": []},
    )
    graph_id = create_resp.json()["id"]
    resp = await client.put(
        f"/api/v1/projects/{test_project.id}/graphs/{graph_id}",
        json={"definition": _LEGACY_DEFINITION},
    )
    assert resp.status_code == 422
    assert "legacy" in resp.json()["detail"].lower()
