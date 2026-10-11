from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from huddleroom.models.escalation import EscalationChain
from huddleroom.models.event_log import EventLog
from huddleroom.models.graph import Graph, GraphRun, GraphRunTimeout
from huddleroom.models.task import Task
from huddleroom.services.graph_engine import GraphEngineService


@pytest.mark.asyncio
async def test_timeout_escalation_resolves_templates_and_keeps_legacy_tokens(db_session, test_project, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    test_project.workspace_path = str(workspace.resolve())
    chain = EscalationChain(
        project_id=test_project.id,
        name="template-escalation",
        steps=[{
            "step": 1,
            "action": "human_notify",
            "message_template": (
                "Run {{graph_run.id}}: {{task.title}} ({{task.id}}); "
                "{{graph_name}} at {{current_node}} ({{graph_run_id}}); "
                "{{dashboard_url}}; {{unknown_token}}"
            ),
        }],
    )
    graph = Graph(
        project_id=test_project.id,
        name="template-graph",
        version="1.0",
        definition={"nodes": {"waiting": {}}, "start_node": "waiting"},
        triggers=[],
        escalation_chain=chain.name,
    )
    task = Task(project_id=test_project.id, title="Review the release")
    db_session.add_all([chain, graph, task])
    await db_session.flush()
    run = GraphRun(
        graph_id=graph.id,
        project_id=test_project.id,
        linked_task_id=task.id,
        current_node="waiting",
        status="active",
        actor_assignments={},
        context={"dashboard_url": "https://example.test/dashboard"},
    )
    db_session.add(run)
    await db_session.flush()
    timeout = GraphRunTimeout(
        graph_run_id=run.id,
        node_name="waiting",
        timeout_action="escalate",
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    db_session.add(timeout)
    await db_session.flush()

    await GraphEngineService().process_timeouts(db_session)

    alert = (await db_session.execute(select(EventLog).where(
        EventLog.project_id == test_project.id,
        EventLog.event_type == "system.escalation_alert",
    ))).scalar_one()
    assert alert.payload["message"] == (
        f"Run {run.id}: {task.title} ({task.id}); "
        f"{graph.name} at waiting ({run.id}); "
        "https://example.test/dashboard; {{unknown_token}}"
    )
    assert timeout.resolved is True
