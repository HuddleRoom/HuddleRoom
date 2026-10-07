import pytest
import uuid
from sqlalchemy import select, text


@pytest.mark.asyncio
async def test_graph_insert_defaults(db_session):
    from huddleroom.models.graph import Graph
    graph = Graph(
        name="test_graph",
        definition={"nodes": {}},
        triggers=[],
    )
    db_session.add(graph)
    await db_session.flush()

    assert graph.id is not None
    assert graph.is_active is True or graph.is_active == 1  # SQLite may return int
    assert graph.version == "1.0"
    assert graph.created_at is not None
    assert graph.updated_at is not None


@pytest.mark.asyncio
async def test_artifact_insert_defaults(db_session, test_project):
    from huddleroom.models.artifact import Artifact
    artifact = Artifact(
        project_id=test_project.id,
        name="test-artifact",
        artifact_type="pull_request",
    )
    db_session.add(artifact)
    await db_session.flush()

    assert artifact.id is not None
    assert artifact.status == "draft"
    assert artifact.version == 1
    assert artifact.is_breaking is False or artifact.is_breaking == 0
    assert artifact.created_at is not None
