import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from huddleroom.models.graph import Graph
from huddleroom.routers.graphs import create_graph
from huddleroom.schemas.graph import GraphCreate


DEFINITION = {
    "nodes": {"start": {"edges": [{"to": "end", "trigger_event": "task.completed"}]}, "end": {}},
    "start_node": "start",
    "terminal_nodes": {"success": ["end"], "failure": []},
}


@pytest.mark.asyncio
async def test_create_duplicate_graph_returns_409(client, test_project, db_session):
    url = f"/api/v1/projects/{test_project.id}/graphs"
    payload = {"name": "duplicate", "version": "1.0", "definition": DEFINITION, "triggers": []}
    first = await client.post(url, json=payload)
    assert first.status_code == 201

    duplicate = await client.post(url, json=payload)
    assert duplicate.status_code == 409
    assert "name and version already exists" in duplicate.json()["detail"]
    graphs = (await db_session.execute(select(Graph).where(Graph.project_id == test_project.id))).scalars().all()
    assert len(graphs) == 1
    assert str(graphs[0].id) == first.json()["id"]


@pytest.mark.asyncio
async def test_create_duplicate_global_graph_returns_409(db_session):
    data = GraphCreate(name="global", definition=DEFINITION, triggers=[])
    first = await create_graph(None, data, None, db_session)
    with pytest.raises(HTTPException) as exc_info:
        await create_graph(None, data, None, db_session)
    assert exc_info.value.status_code == 409
    graphs = (await db_session.execute(select(Graph).where(Graph.project_id.is_(None)))).scalars().all()
    assert [graph.id for graph in graphs] == [first.id]


@pytest.mark.asyncio
async def test_create_graph_duplicate_key_is_scoped_to_project_and_version(db_session):
    data = GraphCreate(name="scoped", definition=DEFINITION, triggers=[])
    project_id = uuid.uuid4()
    first = await create_graph(project_id, data, None, db_session)
    other_project = await create_graph(uuid.uuid4(), data, None, db_session)
    global_graph = await create_graph(None, data, None, db_session)
    next_version = await create_graph(project_id, data.model_copy(update={"version": "2.0"}), None, db_session)
    assert len({first.id, other_project.id, global_graph.id, next_version.id}) == 4
