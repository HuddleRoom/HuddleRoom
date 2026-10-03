import pytest
import uuid
from sqlalchemy import select, text


@pytest.mark.asyncio
async def test_protocol_insert_defaults(db_session):
    from huddleroom.models.protocol import Protocol
    proto = Protocol(
        name="test_proto",
        definition={"states": {}},
        triggers=[],
    )
    db_session.add(proto)
    await db_session.flush()

    assert proto.id is not None
    assert proto.is_active is True or proto.is_active == 1  # SQLite may return int
    assert proto.version == "1.0"
    assert proto.created_at is not None
    assert proto.updated_at is not None


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
