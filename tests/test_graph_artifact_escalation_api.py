from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import select

from huddleroom.models.artifact import Artifact
from huddleroom.models.escalation import EscalationChain
from huddleroom.models.event_log import EventLog
from huddleroom.models.project import Project
from huddleroom.models.graph import Graph, GraphRun, GraphRunTimeout, GraphRunStep
from huddleroom.models.session import Session
from huddleroom.models.task import Task


def graph_payload(name: str = "test_proto") -> dict:
    return {
        "name": name,
        "version": "1.0",
        "description": "Graph API test template",
        "definition": {
            "name": name,
            "triggers": [{"event_type": "task.created"}],
            "nodes": {
                "waiting": {"edges": [{"name": "finish", "to": "done"}]},
                "done": {"edges": []},
            },
            "start_node": "waiting",
            "terminal_nodes": {"success": ["done"], "failure": []},
        },
        "triggers": [{"event_type": "task.created"}],
        "escalation_chain": "standard_dev_escalation",
    }


def escalation_chain_payload(name: str = "my_chain") -> dict:
    return {
        "name": name,
        "description": "Test escalation chain",
        "definition": {"name": name, "steps": []},
        "steps": [
            {
                "step": 1,
                "label": "Human notify",
                "timeout": "2h",
                "action": "human_notify",
                "participants": [{"kind": "human"}],
                "message_template": "Alert!",
                "on_timeout": "fail",
            }
        ],
    }


async def create_graph_run(db_session, test_project) -> GraphRun:
    graph_name = f"run_proto_{uuid4().hex}"
    proto = Graph(
        project_id=test_project.id,
        name=graph_name,
        version="1.0",
        definition=graph_payload(graph_name)["definition"],
        triggers=[],
        escalation_chain="standard_dev_escalation",
        is_active=True,
    )
    db_session.add(proto)
    await db_session.flush()

    run = GraphRun(
        graph_id=proto.id,
        project_id=test_project.id,
        current_node="waiting",
        status="active",
        actor_assignments={},
        context={},
    )
    db_session.add(run)
    await db_session.flush()
    return run


@pytest.mark.asyncio
async def test_create_list_get_and_deactivate_graph_template(client, auth_headers, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/graphs",
        json=graph_payload(),
        headers=auth_headers,
    )
    assert create_resp.status_code == 201
    created = create_resp.json()
    assert created["name"] == "test_proto"
    assert created["version"] == "1.0"
    assert created["is_active"] is True

    list_resp = await client.get(
        f"/api/v1/projects/{test_project.id}/graphs",
        headers=auth_headers,
    )
    assert list_resp.status_code == 200
    assert any(item["id"] == created["id"] for item in list_resp.json())

    get_resp = await client.get(
        f"/api/v1/projects/{test_project.id}/graphs/{created['id']}",
        headers=auth_headers,
    )
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == created["id"]

    delete_resp = await client.delete(
        f"/api/v1/projects/{test_project.id}/graphs/{created['id']}",
        headers=auth_headers,
    )
    assert delete_resp.status_code == 204

    inactive_resp = await client.get(
        f"/api/v1/projects/{test_project.id}/graphs/{created['id']}",
        headers=auth_headers,
    )
    assert inactive_resp.status_code == 200
    assert inactive_resp.json()["is_active"] is False


@pytest.mark.asyncio
async def test_list_and_get_graph_runs(client, auth_headers, test_project, db_session):
    run = await create_graph_run(db_session, test_project)

    list_resp = await client.get(
        f"/api/v1/projects/{test_project.id}/graph-runs",
        headers=auth_headers,
    )
    assert list_resp.status_code == 200
    assert any(item["id"] == str(run.id) for item in list_resp.json()["items"])

    get_resp = await client.get(
        f"/api/v1/projects/{test_project.id}/graph-runs/{run.id}",
        headers=auth_headers,
    )
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["id"] == str(run.id)
    assert data["current_node"] == "waiting"
    assert data["status"] == "active"


@pytest.mark.asyncio
async def test_list_graph_run_steps_ordered_oldest_first(
    client,
    auth_headers,
    test_project,
    db_session,
):
    run = await create_graph_run(db_session, test_project)
    now = datetime.now(timezone.utc)
    newer = GraphRunStep(
        graph_run_id=run.id,
        from_node="review",
        to_node="done",
        edge_name="approve",
        trigger_reason="second",
        actions_executed=[{"type": "notify"}],
        stepped_at=now,
    )
    older = GraphRunStep(
        graph_run_id=run.id,
        from_node="waiting",
        to_node="review",
        edge_name="submit",
        trigger_reason="first",
        actions_executed=[],
        stepped_at=now - timedelta(minutes=5),
    )
    db_session.add_all([newer, older])
    await db_session.flush()

    resp = await client.get(
        f"/api/v1/projects/{test_project.id}/graph-runs/{run.id}/steps",
        headers=auth_headers,
    )

    assert resp.status_code == 200
    data = resp.json()
    assert [item["trigger_reason"] for item in data] == ["first", "second"]
    assert data[0]["from_node"] == "waiting"
    assert data[1]["actions_executed"] == [{"type": "notify"}]


@pytest.mark.asyncio
async def test_get_graph_run_detail_includes_graph_steps_sessions_tasks_and_unresolved_timeouts(
    client,
    auth_headers,
    test_project,
    test_agent,
    db_session,
):
    run = await create_graph_run(db_session, test_project)
    graph = await db_session.get(Graph, run.graph_id)
    linked_task = Task(
        project_id=test_project.id,
        graph_run_id=run.id,
        title="Investigate graph output",
        description="",
        status="in_progress",
        priority=10,
        metadata_={},
    )
    db_session.add(linked_task)
    await db_session.flush()
    run.linked_task_id = linked_task.id

    older_step = GraphRunStep(
        graph_run_id=run.id,
        from_node="waiting",
        to_node="review",
        edge_name="submit",
        trigger_reason="first",
        actions_executed=[],
        stepped_at=datetime.now(timezone.utc) - timedelta(minutes=10),
    )
    newer_step = GraphRunStep(
        graph_run_id=run.id,
        from_node="review",
        to_node="done",
        edge_name="approve",
        trigger_reason="second",
        actions_executed=[{"type": "notify"}],
        stepped_at=datetime.now(timezone.utc) - timedelta(minutes=5),
    )
    unresolved_timeout = GraphRunTimeout(
        graph_run_id=run.id,
        node_name="review",
        timeout_action="escalate",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        resolved=False,
    )
    resolved_timeout = GraphRunTimeout(
        graph_run_id=run.id,
        node_name="waiting",
        timeout_action="fail",
        expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        resolved=True,
        resolved_at=datetime.now(timezone.utc) - timedelta(minutes=30),
    )
    session = Session(
        task_id=linked_task.id,
        agent_id=test_agent.id,
        project_id=test_project.id,
        graph_run_id=run.id,
        adapter_type="codex",
        status="failed",
        input_context={"step": "review"},
        output="full output for debugging",
        error="runner crashed",
        origin="graph",
        started_at=datetime.now(timezone.utc) - timedelta(minutes=4),
        ended_at=datetime.now(timezone.utc) - timedelta(minutes=3),
    )
    db_session.add_all([older_step, newer_step, unresolved_timeout, resolved_timeout, session])
    await db_session.flush()

    resp = await client.get(
        f"/api/v1/projects/{test_project.id}/graph-runs/{run.id}/detail",
        headers=auth_headers,
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["run"]["id"] == str(run.id)
    assert data["graph"] == {
        "id": str(graph.id),
        "name": graph.name,
        "version": graph.version,
    }
    assert [item["trigger_reason"] for item in data["steps"]] == ["first", "second"]
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["id"] == str(session.id)
    assert data["sessions"][0]["task_id"] == str(linked_task.id)
    assert data["sessions"][0]["agent_id"] == str(test_agent.id)
    assert data["sessions"][0]["status"] == "failed"
    assert data["sessions"][0]["origin"] == "graph"
    assert data["sessions"][0]["graph_run_id"] == str(run.id)
    assert data["sessions"][0]["error"] == "runner crashed"
    assert data["sessions"][0]["output"] == "full output for debugging"
    assert [item["id"] for item in data["timeouts"]] == [str(unresolved_timeout.id)]
    assert data["tasks"] == [
        {
            "id": str(linked_task.id),
            "title": "Investigate graph output",
            "status": "in_progress",
            "parent_id": None,
            "graph_run_id": str(run.id),
        }
    ]


@pytest.mark.asyncio
async def test_graph_run_detail_route_is_not_shadowed_by_basic_run_route(
    client,
    auth_headers,
    test_project,
    db_session,
):
    run = await create_graph_run(db_session, test_project)

    detail_resp = await client.get(
        f"/api/v1/projects/{test_project.id}/graph-runs/{run.id}/detail",
        headers=auth_headers,
    )
    basic_resp = await client.get(
        f"/api/v1/projects/{test_project.id}/graph-runs/{run.id}",
        headers=auth_headers,
    )

    assert detail_resp.status_code == 200
    assert detail_resp.json()["run"]["id"] == str(run.id)
    assert basic_resp.status_code == 200
    assert basic_resp.json()["id"] == str(run.id)


@pytest.mark.asyncio
async def test_graph_run_detail_rejects_cross_project_access(client, auth_headers, test_project, db_session):
    other_project = Project(name="Other Project", description="", config={})
    db_session.add(other_project)
    await db_session.flush()
    run = await create_graph_run(db_session, other_project)

    resp = await client.get(
        f"/api/v1/projects/{test_project.id}/graph-runs/{run.id}/detail",
        headers=auth_headers,
    )

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_assign_actor_to_graph_run(client, auth_headers, test_project, test_agent, db_session):
    run = await create_graph_run(db_session, test_project)

    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/graph-runs/{run.id}/actors/reviewer",
        json={"kind": "agent", "id": str(test_agent.id)},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["actor_assignments"]["reviewer"] == {"kind": "agent", "id": str(test_agent.id)}


@pytest.mark.asyncio
async def test_manual_advance_pause_resume_and_abandon_graph_run(
    client,
    auth_headers,
    runnable_project,
    db_session,
):
    run = await create_graph_run(db_session, runnable_project)

    advance_resp = await client.post(
        f"/api/v1/projects/{runnable_project.id}/graph-runs/{run.id}/advance",
        json={"to_node": "done", "reason": "manual QA override"},
        headers=auth_headers,
    )
    assert advance_resp.status_code == 200
    assert advance_resp.json()["current_node"] == "done"

    run = await create_graph_run(db_session, runnable_project)
    pause_resp = await client.post(
        f"/api/v1/projects/{runnable_project.id}/graph-runs/{run.id}/pause",
        headers=auth_headers,
    )
    assert pause_resp.status_code == 200
    assert pause_resp.json()["status"] == "paused"

    resume_resp = await client.post(
        f"/api/v1/projects/{runnable_project.id}/graph-runs/{run.id}/resume",
        headers=auth_headers,
    )
    assert resume_resp.status_code == 200
    assert resume_resp.json()["status"] == "active"

    abandon_resp = await client.post(
        f"/api/v1/projects/{runnable_project.id}/graph-runs/{run.id}/abandon",
        headers=auth_headers,
    )
    assert abandon_resp.status_code == 200
    assert abandon_resp.json()["status"] == "failed"
    assert abandon_resp.json()["completed_at"] is not None


@pytest.mark.asyncio
async def test_create_list_get_and_update_artifact(client, auth_headers, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/artifacts",
        json={
            "name": "my-pr",
            "artifact_type": "pull_request",
            "url": "https://github.com/org/repo/pull/1",
            "metadata": {"branch": "feature/x", "pr_id": "1"},
        },
        headers=auth_headers,
    )
    assert create_resp.status_code == 201
    created = create_resp.json()
    assert created["name"] == "my-pr"
    assert created["artifact_type"] == "pull_request"
    assert created["status"] == "draft"
    assert created["metadata"]["branch"] == "feature/x"

    list_resp = await client.get(
        f"/api/v1/projects/{test_project.id}/artifacts",
        headers=auth_headers,
    )
    assert list_resp.status_code == 200
    assert any(item["id"] == created["id"] for item in list_resp.json())

    get_resp = await client.get(
        f"/api/v1/projects/{test_project.id}/artifacts/{created['id']}",
        headers=auth_headers,
    )
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == created["id"]

    update_resp = await client.put(
        f"/api/v1/projects/{test_project.id}/artifacts/{created['id']}",
        json={"name": "my-pr-updated", "status": "published", "metadata": {"branch": "feature/y"}},
        headers=auth_headers,
    )
    assert update_resp.status_code == 200
    updated = update_resp.json()
    assert updated["name"] == "my-pr-updated"
    assert updated["status"] == "published"
    assert updated["metadata"]["branch"] == "feature/y"


@pytest.mark.asyncio
async def test_delete_artifact_marks_deleted_and_emits_event(client, auth_headers, test_project, db_session):
    artifact = Artifact(
        project_id=test_project.id,
        name="old-pr",
        artifact_type="pull_request",
        url="https://github.com/org/repo/pull/2",
        metadata_={},
    )
    db_session.add(artifact)
    await db_session.flush()

    resp = await client.delete(f"/api/v1/projects/{test_project.id}/artifacts/{artifact.id}", headers=auth_headers)

    assert resp.status_code == 204

    artifact_result = await db_session.execute(select(Artifact).where(Artifact.id == artifact.id))
    deleted_artifact = artifact_result.scalar_one()
    assert deleted_artifact.status == "deleted"

    event_result = await db_session.execute(
        select(EventLog).where(
            EventLog.project_id == test_project.id,
            EventLog.event_type == "artifact.deleted",
        )
    )
    event = event_result.scalar_one()
    assert event.payload["artifact_id"] == str(artifact.id)
    assert event.payload["artifact_type"] == "pull_request"
    assert event.payload["name"] == "old-pr"


@pytest.mark.asyncio
async def test_create_artifact_rejects_cross_project_linked_task(client, auth_headers, test_project, db_session):
    other_project = Project(name="Other Project", description="", config={})
    db_session.add(other_project)
    await db_session.flush()

    other_task = Task(
        project_id=other_project.id,
        title="Other task",
        description="",
        status="backlog",
        priority=50,
        metadata_={},
    )
    db_session.add(other_task)
    await db_session.flush()

    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/artifacts",
        json={
            "name": "cross-linked",
            "artifact_type": "pull_request",
            "metadata": {},
            "linked_task_id": str(other_task.id),
        },
        headers=auth_headers,
    )

    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_add_list_and_remove_artifact_watcher(client, auth_headers, test_project, test_agent, db_session):
    artifact = Artifact(
        project_id=test_project.id,
        name="api-spec",
        artifact_type="api_spec",
        url="https://example.test/openapi.yaml",
        metadata_={},
    )
    db_session.add(artifact)
    await db_session.flush()

    add_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/artifacts/{artifact.id}/watch",
        json={
            "watcher_kind": "agent",
            "watcher_id": str(test_agent.id),
            "event_filter": ["artifact.content_changed"],
        },
        headers=auth_headers,
    )
    assert add_resp.status_code == 201
    assert add_resp.json()["watcher_id"] == str(test_agent.id)

    list_resp = await client.get(
        f"/api/v1/projects/{test_project.id}/artifacts/{artifact.id}/watchers",
        headers=auth_headers,
    )
    assert list_resp.status_code == 200
    watchers = list_resp.json()
    assert len(watchers) == 1
    assert watchers[0]["watcher_kind"] == "agent"
    assert watchers[0]["watcher_id"] == str(test_agent.id)

    remove_resp = await client.delete(
        f"/api/v1/projects/{test_project.id}/artifacts/{artifact.id}/watch",
        params={"watcher_kind": "agent", "watcher_id": str(test_agent.id)},
        headers=auth_headers,
    )
    assert remove_resp.status_code == 204

    empty_resp = await client.get(
        f"/api/v1/projects/{test_project.id}/artifacts/{artifact.id}/watchers",
        headers=auth_headers,
    )
    assert empty_resp.status_code == 200
    assert empty_resp.json() == []


@pytest.mark.asyncio
async def test_create_list_get_and_update_escalation_chain(client, auth_headers, test_project):
    create_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/escalation-chains",
        json=escalation_chain_payload(),
        headers=auth_headers,
    )
    assert create_resp.status_code == 201
    created = create_resp.json()
    assert created["name"] == "my_chain"
    assert len(created["steps"]) == 1
    assert created["is_active"] is True

    list_resp = await client.get(
        f"/api/v1/projects/{test_project.id}/escalation-chains",
        headers=auth_headers,
    )
    assert list_resp.status_code == 200
    assert any(item["id"] == created["id"] for item in list_resp.json())

    get_resp = await client.get(
        f"/api/v1/projects/{test_project.id}/escalation-chains/{created['id']}",
        headers=auth_headers,
    )
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == created["id"]

    updated_payload = escalation_chain_payload("my_chain_updated")
    updated_payload["steps"][0]["message_template"] = "Updated alert!"
    update_resp = await client.put(
        f"/api/v1/projects/{test_project.id}/escalation-chains/{created['id']}",
        json=updated_payload,
        headers=auth_headers,
    )
    assert update_resp.status_code == 200
    updated = update_resp.json()
    assert updated["name"] == "my_chain_updated"
    assert updated["steps"][0]["message_template"] == "Updated alert!"


@pytest.mark.asyncio
async def test_project_scoped_routes_reject_cross_project_resources(client, auth_headers, test_project, db_session):
    other_project = Project(name="Other Project", description="", config={})
    db_session.add(other_project)
    await db_session.flush()

    run = await create_graph_run(db_session, other_project)
    artifact = Artifact(project_id=other_project.id, name="other-artifact", artifact_type="pull_request", metadata_={})
    chain = EscalationChain(
        project_id=other_project.id,
        name="other_chain",
        description="",
        definition={},
        steps=[],
    )
    db_session.add_all([artifact, chain])
    await db_session.flush()

    run_resp = await client.get(
        f"/api/v1/projects/{test_project.id}/graph-runs/{run.id}",
        headers=auth_headers,
    )
    artifact_resp = await client.get(
        f"/api/v1/projects/{test_project.id}/artifacts/{artifact.id}",
        headers=auth_headers,
    )
    chain_resp = await client.get(
        f"/api/v1/projects/{test_project.id}/escalation-chains/{chain.id}",
        headers=auth_headers,
    )

    assert run_resp.status_code == 404
    assert artifact_resp.status_code == 404
    assert chain_resp.status_code == 404


@pytest.mark.asyncio
async def test_project_graph_delete_does_not_deactivate_global_graph(client, auth_headers, test_project, db_session):
    global_graph = Graph(
        project_id=None,
        name="global_proto",
        version="1.0",
        definition=graph_payload("global_proto")["definition"],
        triggers=[],
        is_active=True,
    )
    db_session.add(global_graph)
    await db_session.flush()

    resp = await client.delete(
        f"/api/v1/projects/{test_project.id}/graphs/{global_graph.id}",
        headers=auth_headers,
    )

    assert resp.status_code == 404
    refreshed = await db_session.get(Graph, global_graph.id)
    assert refreshed.is_active is True


@pytest.mark.asyncio
async def test_graph_run_lifecycle_rejects_invalid_status_changes(client, auth_headers, test_project, db_session):
    run = await create_graph_run(db_session, test_project)
    run.status = "completed"
    await db_session.flush()

    pause_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/graph-runs/{run.id}/pause",
        headers=auth_headers,
    )
    resume_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/graph-runs/{run.id}/resume",
        headers=auth_headers,
    )
    abandon_resp = await client.post(
        f"/api/v1/projects/{test_project.id}/graph-runs/{run.id}/abandon",
        headers=auth_headers,
    )

    assert pause_resp.status_code == 409
    assert resume_resp.status_code == 409
    assert abandon_resp.status_code == 409


@pytest.mark.asyncio
async def test_escalation_manager_fires_on_expired_timeout(db_session, runnable_project):
    from huddleroom.workers.consumers.escalation_consumer import EscalationManager

    chain = EscalationChain(
        project_id=None,
        name="test_chain_escalate",
        description="",
        definition={},
        steps=[
            {
                "step": 1,
                "label": "Human notify",
                "timeout": "1h",
                "action": "human_notify",
                "participants": [{"kind": "human"}],
                "message_template": "Graph {{graph_name}} needs attention in node {{current_node}}.",
                "on_timeout": "fail",
            }
        ],
    )
    db_session.add(chain)

    proto = Graph(
        project_id=None,
        name="chain_test_proto",
        version="1.0",
        definition={
            "nodes": {},
            "start_node": "waiting",
            "terminal_nodes": {"success": [], "failure": ["failed"]},
        },
        triggers=[],
        escalation_chain="test_chain_escalate",
        is_active=True,
    )
    db_session.add(proto)
    await db_session.flush()

    run = GraphRun(
        graph_id=proto.id,
        project_id=runnable_project.id,
        current_node="waiting",
        status="active",
        actor_assignments={},
        context={},
    )
    db_session.add(run)
    await db_session.flush()

    timeout = GraphRunTimeout(
        graph_run_id=run.id,
        node_name="waiting",
        timeout_action="escalate",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        resolved=False,
    )
    db_session.add(timeout)
    await db_session.flush()

    processed = await EscalationManager().process_expired_timeouts(db_session)
    await db_session.flush()

    assert processed == 1

    timeout_result = await db_session.execute(select(GraphRunTimeout).where(GraphRunTimeout.id == timeout.id))
    updated_timeout = timeout_result.scalar_one()
    assert updated_timeout.resolved is True
    assert updated_timeout.resolved_at is not None

    run_result = await db_session.execute(select(GraphRun).where(GraphRun.id == run.id))
    updated_run = run_result.scalar_one()
    assert updated_run.escalation_step == 1

    event_result = await db_session.execute(
        select(EventLog).where(
            EventLog.project_id == runnable_project.id,
            EventLog.event_type == "system.escalation_alert",
        )
    )
    alert = event_result.scalar_one()
    assert alert.payload["graph_run_id"] == str(run.id)
    assert "chain_test_proto" in alert.payload["message"]
