import pytest

from huddleroom.services.graph_service import validate_graph_definition


def _definition():
    return {
        "start_node": "start",
        "terminal_nodes": {"success": ["end"], "failure": []},
        "nodes": {
            "start": {"edges": [{"to": "end", "trigger_event": "task.completed"}]},
            "end": {},
        },
    }


@pytest.mark.parametrize("edges", [[], [{"from": "start", "to": "end", "event": "task.completed"}]])
def test_validate_graph_rejects_top_level_edges(edges):
    definition = _definition()
    definition["edges"] = edges
    with pytest.raises(ValueError, match="node-level edges"):
        validate_graph_definition(definition)


def test_validate_graph_accepts_node_level_edges():
    validate_graph_definition(_definition())


@pytest.mark.asyncio
async def test_create_graph_rejects_top_level_edges_with_422(client, test_project):
    definition = _definition()
    definition["edges"] = [{"from": "start", "to": "end", "event": "task.completed"}]
    response = await client.post(
        f"/api/v1/projects/{test_project.id}/graphs",
        json={"name": "invalid-shape", "definition": definition, "triggers": []},
    )
    assert response.status_code == 422
    assert "node-level edges" in response.json()["detail"]


@pytest.mark.asyncio
async def test_update_graph_rejects_top_level_edges_with_422(client, test_project):
    response = await client.post(
        f"/api/v1/projects/{test_project.id}/graphs",
        json={"name": "valid-shape", "definition": _definition(), "triggers": []},
    )
    assert response.status_code == 201
    graph_id = response.json()["id"]
    definition = _definition()
    definition["edges"] = []
    response = await client.put(
        f"/api/v1/projects/{test_project.id}/graphs/{graph_id}",
        json={"definition": definition},
    )
    assert response.status_code == 422
    assert "node-level edges" in response.json()["detail"]
