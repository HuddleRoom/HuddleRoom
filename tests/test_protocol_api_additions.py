import uuid
from datetime import datetime, timezone

import pytest

from huddleroom.models.protocol import Protocol, ProtocolInstance, ProtocolTransition


SAMPLE_DEFINITION = {
    "states": {"start": {"initial": True}, "end": {"terminal": True}},
    "transitions": [{"from": "start", "to": "end", "name": "finish"}],
}


@pytest.mark.asyncio
async def test_protocol_response_includes_definition(client, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/protocols",
        json={
            "name": "test-def",
            "definition": SAMPLE_DEFINITION,
            "triggers": [],
        },
    )
    assert create_resp.status_code == 201
    data = create_resp.json()
    assert "definition" in data
    assert data["definition"]["states"]["start"]["initial"] is True


@pytest.mark.asyncio
async def test_update_protocol(client, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/protocols",
        json={"name": "proto-update", "definition": SAMPLE_DEFINITION, "triggers": []},
    )
    proto_id = create_resp.json()["id"]
    new_def = {"states": {"a": {"initial": True}, "b": {"terminal": True}}, "transitions": []}
    resp = await client.put(
        f"/api/v1/projects/{test_project.id}/protocols/{proto_id}",
        json={"definition": new_def, "description": "updated"},
    )
    assert resp.status_code == 200
    assert resp.json()["description"] == "updated"
    assert resp.json()["definition"]["states"]["a"]["initial"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["active", "paused"])
async def test_update_protocol_blocked_by_non_terminal_instance(client, test_project, db_session, status):
    from huddleroom.models.protocol import Protocol, ProtocolInstance
    protocol = Protocol(
        project_id=test_project.id,
        name=f"proto-blocked-{status}",
        definition=SAMPLE_DEFINITION,
        triggers=[],
    )
    db_session.add(protocol)
    await db_session.flush()
    instance = ProtocolInstance(
        protocol_id=protocol.id,
        project_id=test_project.id,
        current_state="start",
        status=status,
    )
    db_session.add(instance)
    await db_session.flush()
    resp = await client.put(
        f"/api/v1/projects/{test_project.id}/protocols/{protocol.id}",
        json={"description": "should fail"},
    )
    assert resp.status_code == 409
    assert "non-terminal" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_update_protocol_allowed_with_terminal_instances(client, test_project, db_session):
    from huddleroom.models.protocol import Protocol, ProtocolInstance
    protocol = Protocol(
        project_id=test_project.id,
        name="proto-terminal-ok",
        definition=SAMPLE_DEFINITION,
        triggers=[],
    )
    db_session.add(protocol)
    await db_session.flush()
    instance = ProtocolInstance(
        protocol_id=protocol.id,
        project_id=test_project.id,
        current_state="end",
        status="completed",
    )
    db_session.add(instance)
    await db_session.flush()
    resp = await client.put(
        f"/api/v1/projects/{test_project.id}/protocols/{protocol.id}",
        json={"description": "allowed with completed instance"},
    )
    assert resp.status_code == 200
    assert resp.json()["description"] == "allowed with completed instance"


@pytest.mark.asyncio
async def test_activate_protocol(client, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/protocols",
        json={"name": "proto-activate", "definition": SAMPLE_DEFINITION, "triggers": []},
    )
    proto_id = create_resp.json()["id"]
    # Deactivate
    await client.delete(f"/api/v1/projects/{test_project.id}/protocols/{proto_id}")
    # Verify inactive via include_inactive
    list_resp = await client.get(f"/api/v1/projects/{test_project.id}/protocols?include_inactive=true")
    found = [p for p in list_resp.json() if p["id"] == proto_id]
    assert len(found) == 1
    assert found[0]["is_active"] is False
    # Reactivate
    resp = await client.post(f"/api/v1/projects/{test_project.id}/protocols/{proto_id}/activate")
    assert resp.status_code == 200
    assert resp.json()["is_active"] is True


@pytest.mark.asyncio
async def test_list_protocols_include_inactive(client, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/protocols",
        json={"name": "proto-inactive-list", "definition": SAMPLE_DEFINITION, "triggers": []},
    )
    proto_id = create_resp.json()["id"]
    await client.delete(f"/api/v1/projects/{test_project.id}/protocols/{proto_id}")
    # Without include_inactive — should not appear
    resp = await client.get(f"/api/v1/projects/{test_project.id}/protocols")
    ids = [p["id"] for p in resp.json()]
    assert proto_id not in ids
    # With include_inactive — should appear
    resp = await client.get(f"/api/v1/projects/{test_project.id}/protocols?include_inactive=true")
    ids = [p["id"] for p in resp.json()]
    assert proto_id in ids


@pytest.mark.asyncio
async def test_list_protocols_invalid_project(client):
    resp = await client.get(f"/api/v1/projects/{uuid.uuid4()}/protocols")
    assert resp.status_code == 404
    assert "Project not found" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_list_protocol_instances_filter_by_protocol_id(client, test_project):
    resp = await client.get(
        f"/api/v1/projects/{test_project.id}/protocol-instances?protocol_id={uuid.uuid4()}"
    )
    assert resp.status_code == 200
    assert resp.json()["items"] == []


@pytest.mark.asyncio
async def test_list_protocol_instances_invalid_project(client):
    resp = await client.get(f"/api/v1/projects/{uuid.uuid4()}/protocol-instances")
    assert resp.status_code == 404
    assert "Project not found" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_protocol_transition_response_includes_dashboard_compat_fields(client, test_project, db_session):
    transitioned_at = datetime(2026, 6, 24, 12, 30, tzinfo=timezone.utc)
    protocol = Protocol(
        project_id=test_project.id,
        name=f"proto-transition-compat-{uuid.uuid4()}",
        definition=SAMPLE_DEFINITION,
        triggers=[],
    )
    db_session.add(protocol)
    await db_session.flush()
    instance = ProtocolInstance(
        protocol_id=protocol.id,
        project_id=test_project.id,
        current_state="start",
        status="active",
        actor_assignments={},
        context={},
    )
    db_session.add(instance)
    await db_session.flush()
    transition = ProtocolTransition(
        protocol_instance_id=instance.id,
        from_state="start",
        to_state="end",
        transition_name="finish",
        actions_executed=[],
        transitioned_at=transitioned_at,
    )
    db_session.add(transition)
    await db_session.flush()

    resp = await client.get(
        f"/api/v1/projects/{test_project.id}/protocol-instances/{instance.id}/transitions"
    )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["transition_name"] == "finish"
    assert data[0]["transitioned_at"] == transitioned_at.isoformat().replace("+00:00", "Z")
    assert data[0]["event_type"] == "finish"
    assert data[0]["created_at"] == transitioned_at.isoformat().replace("+00:00", "Z")
