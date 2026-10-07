from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import select

from huddleroom.models.agent import Agent
from huddleroom.models.artifact import Artifact
from huddleroom.models.channel import Channel
from huddleroom.models.event_log import EventLog
from huddleroom.models.knowledge_item import KnowledgeItem
from huddleroom.models.message import Message
from huddleroom.models.graph import Graph, GraphRun, GraphRunTimeout, GraphRunStep
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.graph_service import GraphService


@pytest_asyncio.fixture(autouse=True)
async def _runnable_graph_project(db_session, test_project, tmp_path: Path):
    workspace = tmp_path / "graph-workspace"
    workspace.mkdir()
    test_project.workspace_path = str(workspace.resolve())
    await db_session.flush()


@pytest.mark.asyncio
async def test_resolve_actor_by_role(db_session):
    from huddleroom.services.actor_resolver import ActorResolver

    agent = Agent(
        name=f"reviewer-{uuid.uuid4()}",
        role="reviewer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_review"],
        config={},
    )
    db_session.add(agent)
    await db_session.flush()

    reviewers = list(
        (
            await db_session.execute(
                select(Agent).where(
                    Agent.role == "reviewer",
                    Agent.is_active.is_(True),
                )
            )
        ).scalars().all()
    )
    expected = sorted(reviewers, key=lambda reviewer: reviewer.name)[0]

    result = await ActorResolver().resolve_role(db_session, role_value="reviewer")
    assert result is not None
    assert result.id == expected.id


@pytest.mark.asyncio
async def test_resolve_actor_by_capabilities(db_session):
    from huddleroom.services.actor_resolver import ActorResolver

    agent = Agent(
        name=f"arch-{uuid.uuid4()}",
        role="architect",
        provider="openai",
        model="gpt-4o",
        adapter_type="api",
        capabilities=["architecture_review", "planning"],
        config={},
    )
    db_session.add(agent)
    await db_session.flush()

    result = await ActorResolver().resolve_auto(db_session, required_capabilities=["architecture_review"])
    assert result is not None
    assert result.id == agent.id


@pytest.mark.asyncio
async def test_resolve_actor_auto_prefers_event_author_when_capable(db_session):
    from huddleroom.services.actor_resolver import ActorResolver

    preferred = Agent(
        name=f"author-{uuid.uuid4()}",
        role="developer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_review"],
        config={},
    )
    fallback = Agent(
        name=f"fallback-{uuid.uuid4()}",
        role="reviewer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_review"],
        config={},
    )
    db_session.add_all([preferred, fallback])
    await db_session.flush()

    result = await ActorResolver().resolve_actor_slot(
        db_session,
        {"assignment": "auto", "required_capabilities": ["code_review"]},
        triggering_event_payload={"author_agent_id": str(preferred.id)},
    )
    assert result == {"kind": "agent", "id": str(preferred.id), "name": preferred.name}


@pytest.mark.asyncio
async def test_resolve_actor_auto_rejects_event_author_without_capabilities(db_session):
    from huddleroom.services.actor_resolver import ActorResolver

    preferred = Agent(
        name=f"author-{uuid.uuid4()}",
        role="developer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["planning"],
        config={},
    )
    fallback = Agent(
        name=f"fallback-{uuid.uuid4()}",
        role="reviewer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_review"],
        config={},
    )
    db_session.add_all([preferred, fallback])
    await db_session.flush()

    result = await ActorResolver().resolve_actor_slot(
        db_session,
        {"assignment": "auto", "required_capabilities": ["code_review"]},
        triggering_event_payload={"author_agent_id": str(preferred.id)},
    )
    assert result == {"kind": "agent", "id": str(fallback.id), "name": fallback.name}


@pytest.mark.asyncio
async def test_resolve_actor_no_match(db_session):
    from huddleroom.services.actor_resolver import ActorResolver

    result = await ActorResolver().resolve_role(db_session, role_value="nonexistent_role_xyz")
    assert result is None


@pytest.mark.asyncio
async def test_template_resolver_basic(db_session, test_project):
    from huddleroom.services.template_resolver import TemplateResolver

    proto = Graph(project_id=None, name="test_proto", version="1.0", definition={}, triggers=[], is_active=True)
    db_session.add(proto)
    await db_session.flush()

    artifact = Artifact(
        project_id=test_project.id,
        name="my-pr",
        artifact_type="pull_request",
        url="http://example.com",
        metadata_={"branch": "feature/x"},
    )
    db_session.add(artifact)
    await db_session.flush()

    run = GraphRun(
        graph_id=proto.id,
        project_id=test_project.id,
        current_node="opened",
        status="active",
        artifact_id=artifact.id,
        actor_assignments={},
        context={"pr_title": "My PR"},
    )
    db_session.add(run)
    await db_session.flush()

    resolver = TemplateResolver()
    assert await resolver.resolve(
        db_session,
        "PR opened: {{artifact.name}}. Branch: {{artifact.metadata.branch}}",
        run,
    ) == "PR opened: my-pr. Branch: feature/x"
    assert await resolver.resolve(db_session, "{{graph_run.artifact_id}}", run) == str(artifact.id)


def test_guard_evaluator_simple_match():
    from huddleroom.services.guard_evaluator import GuardEvaluator

    assert GuardEvaluator().evaluate({"artifact_id": "abc-123"}, {"artifact_id": "abc-123", "result": "approved"}) is True


def test_guard_evaluator_no_match():
    from huddleroom.services.guard_evaluator import GuardEvaluator

    assert GuardEvaluator().evaluate({"artifact_id": "different-id"}, {"artifact_id": "abc-123"}) is False


def test_guard_evaluator_operator_gte():
    from huddleroom.services.guard_evaluator import GuardEvaluator

    assert GuardEvaluator().evaluate({"priority": {"operator": "gte", "value": 50}}, {"priority": 80}) is True


def test_guard_evaluator_operator_contains():
    from huddleroom.services.guard_evaluator import GuardEvaluator

    assert GuardEvaluator().evaluate({"tags": {"operator": "contains", "value": "bug"}}, {"tags": ["bug", "critical"]}) is True


@pytest.mark.parametrize(
    ("guard", "payload"),
    [
        ({"priority": {"operator": "gte", "value": "high"}}, {"priority": 80}),
        ({"priority": {"operator": "lte", "value": 50}}, {"priority": "urgent"}),
    ],
)
def test_guard_evaluator_gte_lte_malformed_values_return_false(guard, payload):
    from huddleroom.services.guard_evaluator import GuardEvaluator

    assert GuardEvaluator().evaluate(guard, payload) is False


def test_guard_evaluator_empty_guard_always_passes():
    from huddleroom.services.guard_evaluator import GuardEvaluator

    evaluator = GuardEvaluator()
    assert evaluator.evaluate({}, {"anything": "here"}) is True
    assert evaluator.evaluate(None, {}) is True


@pytest.mark.asyncio
async def test_action_emit_event(db_session, test_project):
    from huddleroom.services.action_executor import ActionExecutor

    proto = Graph(project_id=None, name="t1", version="1.0", definition={}, triggers=[], is_active=True)
    db_session.add(proto)
    await db_session.flush()

    run = GraphRun(
        graph_id=proto.id,
        project_id=test_project.id,
        current_node="s1",
        status="active",
        actor_assignments={},
        context={},
    )
    db_session.add(run)
    await db_session.flush()

    await ActionExecutor().execute(
        db_session,
        {"action_type": "emit_event", "event_type": "task.created", "payload_overrides": {"note": "test"}},
        run,
    )
    await db_session.flush()

    result = await db_session.execute(select(EventLog).where(EventLog.event_type == "task.created"))
    event = result.scalar_one()
    assert event.payload["note"] == "test"
    assert event.payload["graph_run_id"] == str(run.id)


@pytest.mark.asyncio
async def test_action_emit_event_includes_linked_task_and_artifact_correlation(db_session, test_project):
    from huddleroom.services.action_executor import ActionExecutor

    proto = Graph(project_id=None, name="t-emit-correlated", version="1.0", definition={}, triggers=[], is_active=True)
    task = Task(project_id=test_project.id, title="Parent feature", status="in_progress")
    artifact = Artifact(
        project_id=test_project.id,
        name="feature-pr",
        artifact_type="pull_request",
        url="http://example.com/pr/1",
        metadata_={},
    )
    db_session.add_all([proto, task, artifact])
    await db_session.flush()

    run = GraphRun(
        graph_id=proto.id,
        project_id=test_project.id,
        current_node="in_review",
        status="active",
        linked_task_id=task.id,
        artifact_id=artifact.id,
        actor_assignments={},
        context={},
    )
    db_session.add(run)
    await db_session.flush()

    await ActionExecutor().execute(
        db_session,
        {"action_type": "emit_event", "event_type": "code.pr_opened"},
        run,
    )
    await db_session.flush()

    event = (
        await db_session.execute(
            select(EventLog).where(
                EventLog.event_type == "code.pr_opened",
                EventLog.source == "graph",
            )
        )
    ).scalar_one()
    assert event.payload["graph_run_id"] == str(run.id)
    assert event.payload["task_id"] == str(task.id)
    assert event.payload["artifact_id"] == str(artifact.id)


@pytest.mark.asyncio
async def test_action_post_message(db_session, test_project, test_user):
    from huddleroom.services.action_executor import ActionExecutor

    channel = Channel(project_id=test_project.id, name="general", channel_type="general", members=[])
    db_session.add(channel)
    await db_session.flush()

    proto = Graph(project_id=None, name="t2", version="1.0", definition={}, triggers=[], is_active=True)
    db_session.add(proto)
    await db_session.flush()

    run = GraphRun(
        graph_id=proto.id,
        project_id=test_project.id,
        current_node="s1",
        status="active",
        actor_assignments={},
        context={"system_user_id": str(test_user.id)},
    )
    db_session.add(run)
    await db_session.flush()

    await ActionExecutor().execute(
        db_session,
        {"action_type": "post_message", "channel": "general", "template": "Hello from graph"},
        run,
    )
    await db_session.flush()

    result = await db_session.execute(select(Message).where(Message.channel_id == channel.id))
    messages = list(result.scalars().all())
    assert len(messages) == 1
    assert "Hello from graph" in messages[0].content
    assert messages[0].sender_user_id == test_user.id


@pytest.mark.asyncio
async def test_action_create_session_without_resolved_actor_creates_no_task_or_session(db_session, test_project):
    from huddleroom.services.action_executor import ActionExecutor

    proto = Graph(project_id=None, name="t-create-session", version="1.0", definition={}, triggers=[], is_active=True)
    db_session.add(proto)
    await db_session.flush()

    run = GraphRun(
        graph_id=proto.id,
        project_id=test_project.id,
        current_node="ready_for_review",
        status="active",
        actor_assignments={},
        context={},
    )
    db_session.add(run)
    await db_session.flush()

    result = await ActionExecutor().execute(
        db_session,
        {"action_type": "create_session", "actor": "reviewer", "task_title": "Review PR"},
        run,
    )
    await db_session.flush()

    tasks = list((await db_session.execute(select(Task).where(Task.graph_run_id == run.id))).scalars().all())
    sessions = list((await db_session.execute(select(Session).where(Session.graph_run_id == run.id))).scalars().all())
    assert result == {"action_type": "create_session", "status": "actor_not_found"}
    assert tasks == []
    assert sessions == []


@pytest.mark.asyncio
async def test_action_create_session_with_resolved_actor_creates_task_and_session(db_session, test_project, monkeypatch):
    from huddleroom.services.action_executor import ActionExecutor

    async def fake_dispatch_session(
        _session_id: str,
        _adapter_type: str,
        _project_id: uuid.UUID,
        task_id: str | None = None,
    ) -> str:
        return "runner-123"

    monkeypatch.setattr("huddleroom.workers.task_runner.dispatch_session", fake_dispatch_session)

    reviewer = Agent(
        name=f"reviewer-{uuid.uuid4()}",
        role="reviewer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_review"],
        config={},
    )
    proto = Graph(project_id=None, name="t-create-session-ok", version="1.0", definition={}, triggers=[], is_active=True)
    db_session.add_all([reviewer, proto])
    await db_session.flush()

    run = GraphRun(
        graph_id=proto.id,
        project_id=test_project.id,
        current_node="ready_for_review",
        status="active",
        actor_assignments={"reviewer": {"kind": "agent", "id": str(reviewer.id), "name": reviewer.name}},
        context={},
    )
    db_session.add(run)
    await db_session.flush()

    result = await ActionExecutor().execute(
        db_session,
        {"action_type": "create_session", "actor": "reviewer", "task_title": "Review PR"},
        run,
    )
    await db_session.flush()

    tasks = list((await db_session.execute(select(Task).where(Task.graph_run_id == run.id))).scalars().all())
    sessions = list((await db_session.execute(select(Session).where(Session.task_id == tasks[0].id))).scalars().all())
    assert result["action_type"] == "create_session"
    assert "session_id" in result
    assert len(tasks) == 1
    assert len(sessions) == 1
    assert sessions[0].agent_id == reviewer.id
    assert sessions[0].runner_task_id is not None
    assert sessions[0].graph_run_id == run.id


@pytest.mark.asyncio
async def test_action_complete_task_does_not_bypass_invalid_transition(db_session, test_project):
    from huddleroom.services.action_executor import ActionExecutor

    proto = Graph(project_id=None, name="t-complete", version="1.0", definition={}, triggers=[], is_active=True)
    task = Task(project_id=test_project.id, title="Blocked completion", status="archived")
    db_session.add_all([proto, task])
    await db_session.flush()

    run = GraphRun(
        graph_id=proto.id,
        project_id=test_project.id,
        current_node="merged",
        status="completed",
        linked_task_id=task.id,
        actor_assignments={},
        context={},
    )
    db_session.add(run)
    await db_session.flush()

    result = await ActionExecutor().execute(
        db_session,
        {"action_type": "complete_task", "task_id": str(task.id)},
        run,
    )
    await db_session.refresh(task)

    assert result["action_type"] == "complete_task"
    assert result["status"] == "invalid_transition"
    assert task.status == "archived"
    assert task.completed_at is None


@pytest.mark.asyncio
async def test_engine_creates_run_on_trigger(db_session, test_project, test_user):
    from huddleroom.services.graph_engine import GraphEngineService

    proto = await GraphService().load_from_yaml(db_session, Path("workspace/graphs/code_review.yaml"))
    reviewer = Agent(
        name=f"reviewer-{uuid.uuid4()}",
        role="reviewer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_review"],
        config={},
    )
    merger = Agent(
        name=f"pm-{uuid.uuid4()}",
        role="pm",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_merge"],
        config={},
    )
    db_session.add_all(
        [
            reviewer,
            merger,
            Channel(project_id=test_project.id, name="general", channel_type="general", members=[]),
        ]
    )
    await db_session.flush()

    engine = GraphEngineService()
    from huddleroom.services.event_bus import BusEvent

    event = BusEvent(
        id=uuid.uuid4(),
        project_id=test_project.id,
        event_type="code.pr_opened",
        payload={"artifact_id": str(uuid.uuid4()), "pr_id": "42"},
        source="system",
    )
    await engine.process_event(db_session, event)
    await db_session.flush()

    result = await db_session.execute(
        select(GraphRun).where(
            GraphRun.graph_id == proto.id,
            GraphRun.project_id == test_project.id,
        )
    )
    runs = list(result.scalars().all())
    assert len(runs) == 1
    assert runs[0].current_node == "opened"
    assert runs[0].status == "active"


@pytest.mark.asyncio
async def test_engine_starts_feature_graph_from_task_service_created_event_metadata(db_session, test_project, monkeypatch):
    from huddleroom.schemas.task import TaskCreate
    from huddleroom.services.event_bus import BusEvent
    from huddleroom.services.graph_engine import GraphEngineService
    from huddleroom.services.task_service import TaskService

    async def fake_dispatch_session(_session_id: str, _adapter_type: str, _project_id: uuid.UUID) -> str:
        return "runner-feature"

    monkeypatch.setattr("huddleroom.workers.task_runner.dispatch_session", fake_dispatch_session)

    proto = await GraphService().load_from_yaml(db_session, Path("workspace/graphs/feature_development.yaml"))
    pm = Agent(
        name=f"pm-{uuid.uuid4()}",
        role="pm",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["planning", "spec_writing"],
        config={},
    )
    developer = Agent(
        name=f"developer-{uuid.uuid4()}",
        role="developer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_write"],
        config={},
    )
    reviewer = Agent(
        name=f"reviewer-{uuid.uuid4()}",
        role="reviewer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_review"],
        config={},
    )
    db_session.add_all([pm, developer, reviewer])
    await db_session.flush()

    task = await TaskService().create(
        db_session,
        test_project.id,
        TaskCreate(title="Build metadata-triggered feature", metadata={"graph": "feature_development"}),
    )
    task_event = (
        await db_session.execute(
            select(EventLog).where(
                EventLog.event_type == "task.created",
                EventLog.payload["task_id"].as_string() == str(task.id),
            )
        )
    ).scalar_one()

    await GraphEngineService().process_event(
        db_session,
        BusEvent(
            id=task_event.id,
            project_id=test_project.id,
            event_type=task_event.event_type,
            payload=task_event.payload,
            source=task_event.source,
            emitted_at=task_event.emitted_at,
        ),
    )
    await db_session.flush()

    runs = list(
        (
            await db_session.execute(
                select(GraphRun).where(
                    GraphRun.graph_id == proto.id,
                    GraphRun.project_id == test_project.id,
                    GraphRun.linked_task_id == task.id,
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(runs) == 1
    assert runs[0].current_node == "spec_required"


@pytest.mark.asyncio
async def test_feature_development_completes_from_real_task_done_status_changed_payload(db_session, test_project):
    from huddleroom.services.event_bus import BusEvent
    from huddleroom.services.graph_engine import GraphEngineService
    from huddleroom.services.task_service import TaskService

    proto = await GraphService().load_from_yaml(db_session, Path("workspace/graphs/feature_development.yaml"))
    task = Task(project_id=test_project.id, title="Deploy merged feature", status="in_progress")
    db_session.add(task)
    await db_session.flush()

    run = GraphRun(
        graph_id=proto.id,
        project_id=test_project.id,
        linked_task_id=task.id,
        current_node="deployment_ready",
        status="active",
        actor_assignments={},
        context={},
    )
    db_session.add(run)
    await db_session.flush()

    await TaskService().transition_status(db_session, test_project.id, task.id, "done")
    status_changed = (
        await db_session.execute(
            select(EventLog).where(
                EventLog.project_id == test_project.id,
                EventLog.event_type == "task.status_changed",
                EventLog.payload["task_id"].as_string() == str(task.id),
            )
        )
    ).scalar_one()

    assert status_changed.payload["status"] == "done"
    assert "new_status" not in status_changed.payload

    await GraphEngineService().process_event(
        db_session,
        BusEvent(
            id=status_changed.id,
            project_id=test_project.id,
            event_type=status_changed.event_type,
            payload=status_changed.payload,
            source=status_changed.source,
            emitted_at=status_changed.emitted_at,
        ),
    )
    await db_session.flush()
    await db_session.refresh(run)

    step = (
        await db_session.execute(
            select(GraphRunStep).where(
                GraphRunStep.graph_run_id == run.id,
                GraphRunStep.edge_name == "deployed",
            )
        )
    ).scalar_one()
    assert run.current_node == "complete"
    assert run.status == "completed"
    assert run.completed_at is not None
    assert step.from_node == "deployment_ready"
    assert step.to_node == "complete"


@pytest.mark.asyncio
async def test_engine_deduplicates_same_trigger_for_same_artifact(db_session, test_project):
    from huddleroom.services.event_bus import BusEvent
    from huddleroom.services.graph_engine import GraphEngineService

    proto = await GraphService().load_from_yaml(db_session, Path("workspace/graphs/code_review.yaml"))
    reviewer = Agent(
        name=f"reviewer-{uuid.uuid4()}",
        role="reviewer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_review"],
        config={},
    )
    merger = Agent(
        name=f"pm-{uuid.uuid4()}",
        role="pm",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_merge"],
        config={},
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="dedupe-pr",
        artifact_type="pull_request",
        url="http://example.com",
        metadata_={"branch": "feature/dedupe"},
    )
    db_session.add_all(
        [reviewer, merger, artifact, Channel(project_id=test_project.id, name="general", channel_type="general", members=[])]
    )
    await db_session.flush()

    engine = GraphEngineService()
    first_event = BusEvent(
        id=uuid.uuid4(),
        project_id=test_project.id,
        event_type="code.pr_opened",
        payload={"artifact_id": str(artifact.id)},
        source="system",
    )
    second_event = BusEvent(
        id=uuid.uuid4(),
        project_id=test_project.id,
        event_type="code.pr_opened",
        payload={"artifact_id": str(artifact.id)},
        source="system",
    )

    await engine.process_event(db_session, first_event)
    await engine.process_event(db_session, second_event)
    await db_session.flush()

    runs = list(
        (
            await db_session.execute(
                select(GraphRun).where(
                    GraphRun.graph_id == proto.id,
                    GraphRun.project_id == test_project.id,
                    GraphRun.artifact_id == artifact.id,
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(runs) == 1


@pytest.mark.asyncio
async def test_engine_advances_node_on_event(db_session, test_project, test_user):
    from huddleroom.services.event_bus import BusEvent
    from huddleroom.services.graph_engine import GraphEngineService

    proto = await GraphService().load_from_yaml(db_session, Path("workspace/graphs/code_review.yaml"))
    reviewer = Agent(
        name=f"reviewer-{uuid.uuid4()}",
        role="reviewer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_review"],
        config={},
    )
    merger = Agent(
        name=f"pm-{uuid.uuid4()}",
        role="pm",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_merge"],
        config={},
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="test-pr",
        artifact_type="pull_request",
        url="http://example.com",
        metadata_={"branch": "feature/x"},
    )
    db_session.add_all(
        [
            reviewer,
            merger,
            artifact,
            Channel(project_id=test_project.id, name="general", channel_type="general", members=[]),
        ]
    )
    await db_session.flush()

    engine = GraphEngineService()
    await engine.process_event(
        db_session,
        BusEvent(
            id=uuid.uuid4(),
            project_id=test_project.id,
            event_type="code.pr_opened",
            payload={"artifact_id": str(artifact.id)},
            source="system",
        ),
    )
    await db_session.flush()

    await engine.process_event(
        db_session,
        BusEvent(
            id=uuid.uuid4(),
            project_id=test_project.id,
            event_type="test.passed",
            payload={"artifact_id": str(artifact.id)},
            source="system",
        ),
    )
    await db_session.flush()

    result = await db_session.execute(
        select(GraphRun).where(
            GraphRun.graph_id == proto.id,
            GraphRun.project_id == test_project.id,
        )
    )
    run = result.scalar_one()
    assert run.current_node == "ready_for_review"

    timeouts = list(
        (
            await db_session.execute(
                select(GraphRunTimeout).where(GraphRunTimeout.graph_run_id == run.id)
            )
        )
        .scalars()
        .all()
    )
    resolved = [timeout for timeout in timeouts if timeout.resolved]
    unresolved = [timeout for timeout in timeouts if not timeout.resolved]
    assert len(resolved) == 1
    assert resolved[0].node_name == "opened"
    assert len(unresolved) == 1
    assert unresolved[0].node_name == "ready_for_review"


@pytest.mark.asyncio
async def test_engine_completes_graph_on_merge(db_session, test_project, monkeypatch):
    from huddleroom.services.event_bus import BusEvent
    from huddleroom.services.graph_engine import GraphEngineService

    async def fake_dispatch_session(_session_id: str, _adapter_type: str, _project_id: uuid.UUID) -> str:
        return "runner-456"

    monkeypatch.setattr("huddleroom.workers.task_runner.dispatch_session", fake_dispatch_session)

    proto = await GraphService().load_from_yaml(db_session, Path("workspace/graphs/code_review.yaml"))
    reviewer = Agent(
        name=f"reviewer-{uuid.uuid4()}",
        role="reviewer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_review"],
        config={},
    )
    merger = Agent(
        name=f"pm-{uuid.uuid4()}",
        role="pm",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_merge"],
        config={},
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="merge-pr",
        artifact_type="pull_request",
        url="http://example.com",
        metadata_={"branch": "feature/merge"},
    )
    linked_task = Task(project_id=test_project.id, title="Ship PR", status="in_progress")
    db_session.add_all(
        [reviewer, merger, artifact, linked_task, Channel(project_id=test_project.id, name="general", channel_type="general", members=[])]
    )
    await db_session.flush()

    engine = GraphEngineService()
    artifact_id = str(artifact.id)
    for event_type in ("code.pr_opened", "test.passed", "review.approved", "code.pr_merged"):
        await engine.process_event(
            db_session,
            BusEvent(
                id=uuid.uuid4(),
                project_id=test_project.id,
                event_type=event_type,
                payload={"artifact_id": artifact_id, "task_id": str(linked_task.id)},
                source="system",
            ),
        )
    await db_session.flush()

    run = (
        await db_session.execute(
            select(GraphRun).where(
                GraphRun.graph_id == proto.id,
                GraphRun.project_id == test_project.id,
            )
        )
    ).scalar_one()
    await db_session.refresh(linked_task)

    completed_event = (
        await db_session.execute(
            select(EventLog).where(
                EventLog.project_id == test_project.id,
                EventLog.event_type == "graph.run_completed",
            )
        )
    ).scalar_one()
    decisions = list(
        (
            await db_session.execute(
                select(KnowledgeItem).where(KnowledgeItem.provenance_graph_run_id == run.id)
            )
        )
        .scalars()
        .all()
    )

    assert run.current_node == "merged"
    assert run.status == "completed"
    assert run.completed_at is not None
    assert completed_event.payload["to_node"] == "merged"
    assert linked_task.status == "done"
    assert linked_task.completed_at is not None
    assert len(decisions) == 1
    assert decisions[0].content_type == "decision"


@pytest.mark.asyncio
async def test_engine_session_completed_guard_uses_session_metadata_when_payload_only_ids(db_session, test_project):
    from huddleroom.services.event_bus import BusEvent
    from huddleroom.services.graph_engine import GraphEngineService

    proto = await GraphService().load_from_yaml(db_session, Path("workspace/graphs/architecture_decision.yaml"))
    architect = Agent(
        name=f"architect-{uuid.uuid4()}",
        role="architect",
        provider="openai",
        model="gpt-4o",
        adapter_type="api",
        capabilities=["architecture_review", "planning"],
        config={},
    )
    task = Task(project_id=test_project.id, title="Choose event enrichment approach", status="in_progress")
    db_session.add_all([architect, task])
    await db_session.flush()

    run = GraphRun(
        graph_id=proto.id,
        project_id=test_project.id,
        linked_task_id=task.id,
        current_node="proposal_submitted",
        status="active",
        actor_assignments={"architect": {"kind": "agent", "id": str(architect.id), "name": architect.name}},
        context={},
    )
    db_session.add(run)
    await db_session.flush()

    session = Session(
        agent_id=architect.id,
        task_id=task.id,
        project_id=test_project.id,
        graph_run_id=run.id,
        adapter_type="api",
        status="completed",
        metadata_={"recommendation": "recommend_meeting"},
        origin="graph",
    )
    db_session.add(session)
    await db_session.flush()

    await GraphEngineService().process_event(
        db_session,
        BusEvent(
            id=uuid.uuid4(),
            project_id=test_project.id,
            event_type="session.completed",
            payload={"session_id": str(session.id), "task_id": str(task.id)},
            source="system",
        ),
    )
    await db_session.flush()
    await db_session.refresh(run)

    step = (
        await db_session.execute(
            select(GraphRunStep).where(
                GraphRunStep.graph_run_id == run.id,
                GraphRunStep.edge_name == "recommend_meeting",
            )
        )
    ).scalar_one()
    assert run.current_node == "meeting_scheduled"
    assert step.from_node == "proposal_submitted"
    assert step.to_node == "meeting_scheduled"


@pytest.mark.asyncio
async def test_engine_session_metadata_takes_precedence_over_linked_task_metadata(db_session, test_project):
    from huddleroom.services.event_bus import BusEvent
    from huddleroom.services.graph_engine import GraphEngineService

    proto = await GraphService().load_from_yaml(db_session, Path("workspace/graphs/architecture_decision.yaml"))
    architect = Agent(
        name=f"architect-conflict-{uuid.uuid4()}",
        role="architect",
        provider="openai",
        model="gpt-4o",
        adapter_type="api",
        capabilities=["architecture_review", "planning"],
        config={},
    )
    task = Task(
        project_id=test_project.id,
        title="Choose event metadata precedence",
        status="in_progress",
        metadata_={"recommendation": "reject"},
    )
    db_session.add_all([architect, task])
    await db_session.flush()

    run = GraphRun(
        graph_id=proto.id,
        project_id=test_project.id,
        linked_task_id=task.id,
        current_node="proposal_submitted",
        status="active",
        actor_assignments={"architect": {"kind": "agent", "id": str(architect.id), "name": architect.name}},
        context={},
    )
    db_session.add(run)
    await db_session.flush()

    session = Session(
        agent_id=architect.id,
        task_id=task.id,
        project_id=test_project.id,
        graph_run_id=run.id,
        adapter_type="api",
        status="completed",
        metadata_={"recommendation": "recommend_meeting"},
        origin="graph",
    )
    db_session.add(session)
    await db_session.flush()

    await GraphEngineService().process_event(
        db_session,
        BusEvent(
            id=uuid.uuid4(),
            project_id=test_project.id,
            event_type="session.completed",
            payload={"session_id": str(session.id), "task_id": str(task.id)},
            source="system",
        ),
    )
    await db_session.flush()
    await db_session.refresh(run)

    assert run.current_node == "meeting_scheduled"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("output", "expected_verdict"),
    [
        ("APPROVE", "approved"),
        ("CHANGES_REQUESTED", "changes_requested"),
        ("Looks mostly fine, but I left comments.", "commented"),
        ("DO NOT APPROVE", "commented"),
    ],
)
async def test_engine_enriches_graph_review_session_completed(db_session, test_project, output, expected_verdict):
    from huddleroom.services.event_bus import BusEvent
    from huddleroom.services.graph_engine import GraphEngineService

    proto = await GraphService().load_from_yaml(db_session, Path("workspace/graphs/code_review.yaml"))
    reviewer = Agent(
        name=f"reviewer-{uuid.uuid4()}",
        role="reviewer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_review"],
        config={},
    )
    task = Task(project_id=test_project.id, title="Review PR", status="in_progress")
    db_session.add_all([reviewer, task])
    await db_session.flush()

    run = GraphRun(
        graph_id=proto.id,
        project_id=test_project.id,
        linked_task_id=task.id,
        current_node="ready_for_review",
        status="active",
        actor_assignments={"reviewer": {"kind": "agent", "id": str(reviewer.id), "name": reviewer.name}},
        context={},
    )
    db_session.add(run)
    await db_session.flush()

    session = Session(
        agent_id=reviewer.id,
        task_id=task.id,
        project_id=test_project.id,
        graph_run_id=run.id,
        adapter_type="api",
        status="completed",
        output=output,
        metadata_={},
        origin="graph",
    )
    db_session.add(session)
    await db_session.flush()

    engine = GraphEngineService()
    captured: dict[str, dict] = {}

    async def fake_check_triggers(_db, event):
        captured["payload"] = event.payload

    async def fake_evaluate_runs(_db, _event):
        return None

    engine._check_triggers = fake_check_triggers
    engine._evaluate_active_runs = fake_evaluate_runs

    await engine.process_event(
        db_session,
        BusEvent(
            id=uuid.uuid4(),
            project_id=test_project.id,
            event_type="session.completed",
            payload={"session_id": str(session.id)},
            source="system",
        ),
    )

    assert captured["payload"]["session"]["output"] == output
    assert captured["payload"]["review_outcome"] == {
        "verdict": expected_verdict,
        "raw_output": output,
    }


@pytest.mark.parametrize(
    ("raw_output", "expected_verdict"),
    [
        ("APPROVED", "approved"),
        ("REQUEST CHANGES", "changes_requested"),
        ("REQUESTED CHANGES", "changes_requested"),
    ],
)
def test_engine_review_outcome_parser_recognizes_conservative_variants(raw_output, expected_verdict):
    from huddleroom.services.graph_engine import GraphEngineService

    assert GraphEngineService()._parse_review_outcome(raw_output) == {
        "verdict": expected_verdict,
        "raw_output": raw_output,
    }


@pytest.mark.asyncio
async def test_engine_does_not_enrich_non_code_review_session_completed_with_review_fields(db_session, test_project):
    from huddleroom.services.event_bus import BusEvent
    from huddleroom.services.graph_engine import GraphEngineService

    proto = Graph(
        project_id=None,
        name=f"general_review_{uuid.uuid4()}",
        version="1.0",
        definition={"nodes": {"ready_for_review": {"edges": []}}},
        triggers=[],
        is_active=True,
    )
    reviewer = Agent(
        name=f"reviewer-generic-{uuid.uuid4()}",
        role="reviewer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_review"],
        config={},
    )
    task = Task(project_id=test_project.id, title="Generic review task", status="in_progress")
    db_session.add_all([proto, reviewer, task])
    await db_session.flush()

    run = GraphRun(
        graph_id=proto.id,
        project_id=test_project.id,
        linked_task_id=task.id,
        current_node="ready_for_review",
        status="active",
        actor_assignments={"reviewer": {"kind": "agent", "id": str(reviewer.id), "name": reviewer.name}},
        context={},
    )
    db_session.add(run)
    await db_session.flush()

    session = Session(
        agent_id=reviewer.id,
        task_id=task.id,
        project_id=test_project.id,
        graph_run_id=run.id,
        adapter_type="api",
        status="completed",
        output="APPROVE",
        metadata_={},
        origin="graph",
    )
    db_session.add(session)
    await db_session.flush()

    engine = GraphEngineService()
    captured: dict[str, dict] = {}

    async def fake_check_triggers(_db, event):
        captured["payload"] = event.payload

    async def fake_evaluate_runs(_db, _event):
        return None

    engine._check_triggers = fake_check_triggers
    engine._evaluate_active_runs = fake_evaluate_runs

    await engine.process_event(
        db_session,
        BusEvent(
            id=uuid.uuid4(),
            project_id=test_project.id,
            event_type="session.completed",
            payload={"session_id": str(session.id)},
            source="system",
        ),
    )

    assert "output" not in captured["payload"]["session"]
    assert "review_outcome" not in captured["payload"]


@pytest.mark.asyncio
async def test_engine_does_not_enrich_non_graph_origin_code_review_session_completed(db_session, test_project):
    from huddleroom.services.event_bus import BusEvent
    from huddleroom.services.graph_engine import GraphEngineService

    proto = await GraphService().load_from_yaml(db_session, Path("workspace/graphs/code_review.yaml"))
    reviewer = Agent(
        name=f"reviewer-manual-origin-{uuid.uuid4()}",
        role="reviewer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_review"],
        config={},
    )
    task = Task(project_id=test_project.id, title="Manual origin review PR", status="in_progress")
    db_session.add_all([reviewer, task])
    await db_session.flush()

    run = GraphRun(
        graph_id=proto.id,
        project_id=test_project.id,
        linked_task_id=task.id,
        current_node="ready_for_review",
        status="active",
        actor_assignments={"reviewer": {"kind": "agent", "id": str(reviewer.id), "name": reviewer.name}},
        context={},
    )
    db_session.add(run)
    await db_session.flush()

    session = Session(
        agent_id=reviewer.id,
        task_id=task.id,
        project_id=test_project.id,
        graph_run_id=run.id,
        adapter_type="api",
        status="completed",
        output="APPROVE",
        metadata_={},
        origin="api",
    )
    db_session.add(session)
    await db_session.flush()

    engine = GraphEngineService()
    captured: dict[str, dict] = {}

    async def fake_check_triggers(_db, event):
        captured["payload"] = event.payload

    async def fake_evaluate_runs(_db, _event):
        return None

    engine._check_triggers = fake_check_triggers
    engine._evaluate_active_runs = fake_evaluate_runs

    await engine.process_event(
        db_session,
        BusEvent(
            id=uuid.uuid4(),
            project_id=test_project.id,
            event_type="session.completed",
            payload={"session_id": str(session.id)},
            source="system",
        ),
    )

    assert "output" not in captured["payload"]["session"]
    assert "review_outcome" not in captured["payload"]


@pytest.mark.asyncio
async def test_engine_does_not_enrich_or_auto_emit_review_for_wrong_assigned_actor(db_session, test_project):
    from huddleroom.services.event_bus import BusEvent
    from huddleroom.services.graph_engine import GraphEngineService

    proto = await GraphService().load_from_yaml(db_session, Path("workspace/graphs/code_review.yaml"))
    assigned_reviewer = Agent(
        name=f"reviewer-assigned-{uuid.uuid4()}",
        role="reviewer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_review"],
        config={},
    )
    wrong_reviewer = Agent(
        name=f"reviewer-wrong-{uuid.uuid4()}",
        role="reviewer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_review"],
        config={},
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="wrong-reviewer-pr",
        artifact_type="pull_request",
        url="http://example.com/pr/wrong-reviewer",
        metadata_={"branch": "feature/wrong-reviewer"},
    )
    task = Task(project_id=test_project.id, title="Review wrong reviewer PR", status="in_progress")
    db_session.add_all([assigned_reviewer, wrong_reviewer, artifact, task])
    await db_session.flush()

    run = GraphRun(
        graph_id=proto.id,
        project_id=test_project.id,
        linked_task_id=task.id,
        artifact_id=artifact.id,
        current_node="ready_for_review",
        status="active",
        actor_assignments={"reviewer": {"kind": "agent", "id": str(assigned_reviewer.id), "name": assigned_reviewer.name}},
        context={},
    )
    db_session.add(run)
    await db_session.flush()

    session = Session(
        agent_id=wrong_reviewer.id,
        task_id=task.id,
        project_id=test_project.id,
        graph_run_id=run.id,
        adapter_type="api",
        status="completed",
        output="APPROVED",
        metadata_={},
        origin="graph",
    )
    db_session.add(session)
    await db_session.flush()

    captured: dict[str, dict] = {}
    engine = GraphEngineService()
    original_check_triggers = engine._check_triggers

    async def fake_check_triggers(_db, event):
        captured["payload"] = event.payload

    async def fake_evaluate_runs(_db, _event):
        return None

    engine._check_triggers = fake_check_triggers
    engine._evaluate_active_runs = fake_evaluate_runs

    await engine.process_event(
        db_session,
        BusEvent(
            id=uuid.uuid4(),
            project_id=test_project.id,
            event_type="session.completed",
            payload={"session_id": str(session.id)},
            source="system",
        ),
    )
    await db_session.flush()
    await db_session.refresh(run)
    await db_session.refresh(session)

    assert captured["payload"]["session"]["id"] == str(session.id)
    assert "output" not in captured["payload"]["session"]
    assert "review_outcome" not in captured["payload"]
    assert run.current_node == "ready_for_review"
    assert "auto_review_graph_event" not in session.metadata_

    engine._check_triggers = original_check_triggers
    result = await db_session.execute(
        select(EventLog).where(
            EventLog.project_id == test_project.id,
            EventLog.event_type.in_(["review.approved", "review.changes_requested"]),
        )
    )
    assert result.scalars().all() == []


@pytest.mark.asyncio
async def test_engine_auto_emits_review_approved_and_advances_code_review_run(db_session, test_project):
    from huddleroom.services.event_bus import BusEvent
    from huddleroom.services.graph_engine import GraphEngineService

    proto = await GraphService().load_from_yaml(db_session, Path("workspace/graphs/code_review.yaml"))
    reviewer = Agent(
        name=f"reviewer-auto-approved-{uuid.uuid4()}",
        role="reviewer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_review"],
        config={},
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="auto-approved-pr",
        artifact_type="pull_request",
        url="http://example.com/pr/approved",
        metadata_={"branch": "feature/approved"},
    )
    task = Task(project_id=test_project.id, title="Review auto approved PR", status="in_progress")
    db_session.add_all([reviewer, artifact, task])
    await db_session.flush()

    run = GraphRun(
        graph_id=proto.id,
        project_id=test_project.id,
        linked_task_id=task.id,
        artifact_id=artifact.id,
        current_node="ready_for_review",
        status="active",
        actor_assignments={"reviewer": {"kind": "agent", "id": str(reviewer.id), "name": reviewer.name}},
        context={},
    )
    db_session.add(run)
    await db_session.flush()

    session = Session(
        agent_id=reviewer.id,
        task_id=task.id,
        project_id=test_project.id,
        graph_run_id=run.id,
        adapter_type="api",
        status="completed",
        output="APPROVE",
        metadata_={},
        origin="graph",
    )
    db_session.add(session)
    await db_session.flush()

    engine = GraphEngineService()
    completion_event = BusEvent(
        id=uuid.uuid4(),
        project_id=test_project.id,
        event_type="session.completed",
        payload={"session_id": str(session.id)},
        source="system",
    )

    await engine.process_event(db_session, completion_event)
    await engine.process_event(db_session, completion_event)
    await db_session.flush()
    await db_session.refresh(run)
    await db_session.refresh(session)

    assert run.current_node == "approved"
    assert session.metadata_["auto_review_graph_event"]["event_type"] == "review.approved"
    result = await db_session.execute(
        select(EventLog).where(
            EventLog.project_id == test_project.id,
            EventLog.event_type == "review.approved",
        )
    )
    events = result.scalars().all()
    assert len(events) == 1
    assert events[0].payload["artifact_id"] == str(artifact.id)
    assert events[0].payload["graph_run_id"] == str(run.id)


@pytest.mark.asyncio
async def test_engine_auto_emits_review_changes_requested_and_advances_code_review_run(
    db_session, test_project
):
    from huddleroom.services.event_bus import BusEvent
    from huddleroom.services.graph_engine import GraphEngineService

    proto = await GraphService().load_from_yaml(db_session, Path("workspace/graphs/code_review.yaml"))
    reviewer = Agent(
        name=f"reviewer-auto-changes-{uuid.uuid4()}",
        role="reviewer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_review"],
        config={},
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="auto-changes-pr",
        artifact_type="pull_request",
        url="http://example.com/pr/changes",
        metadata_={"branch": "feature/changes"},
    )
    task = Task(project_id=test_project.id, title="Review auto changes PR", status="in_progress")
    db_session.add_all([reviewer, artifact, task])
    await db_session.flush()

    run = GraphRun(
        graph_id=proto.id,
        project_id=test_project.id,
        linked_task_id=task.id,
        artifact_id=artifact.id,
        current_node="ready_for_review",
        status="active",
        actor_assignments={"reviewer": {"kind": "agent", "id": str(reviewer.id), "name": reviewer.name}},
        context={},
    )
    db_session.add(run)
    await db_session.flush()

    session = Session(
        agent_id=reviewer.id,
        task_id=task.id,
        project_id=test_project.id,
        graph_run_id=run.id,
        adapter_type="api",
        status="completed",
        output="CHANGES_REQUESTED",
        metadata_={},
        origin="graph",
    )
    db_session.add(session)
    await db_session.flush()

    await GraphEngineService().process_event(
        db_session,
        BusEvent(
            id=uuid.uuid4(),
            project_id=test_project.id,
            event_type="session.completed",
            payload={"session_id": str(session.id)},
            source="system",
        ),
    )
    await db_session.flush()
    await db_session.refresh(run)

    assert run.current_node == "awaiting_revision"
    result = await db_session.execute(
        select(EventLog).where(
            EventLog.project_id == test_project.id,
            EventLog.event_type == "review.changes_requested",
        )
    )
    event = result.scalar_one()
    assert event.payload["artifact_id"] == str(artifact.id)
    assert event.payload["graph_run_id"] == str(run.id)


@pytest.mark.asyncio
async def test_engine_auto_review_deduplicates_persisted_event_when_session_completed_reprocessed(
    db_session, test_project
):
    from huddleroom.services.event_bus import BusEvent
    from huddleroom.services.graph_engine import GraphEngineService

    proto = await GraphService().load_from_yaml(db_session, Path("workspace/graphs/code_review.yaml"))
    reviewer = Agent(
        name=f"reviewer-auto-dedup-{uuid.uuid4()}",
        role="reviewer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_review"],
        config={},
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="auto-dedup-pr",
        artifact_type="pull_request",
        url="http://example.com/pr/dedup",
        metadata_={"branch": "feature/dedup"},
    )
    task = Task(project_id=test_project.id, title="Review auto dedup PR", status="in_progress")
    db_session.add_all([reviewer, artifact, task])
    await db_session.flush()

    run = GraphRun(
        graph_id=proto.id,
        project_id=test_project.id,
        linked_task_id=task.id,
        artifact_id=artifact.id,
        current_node="ready_for_review",
        status="active",
        actor_assignments={"reviewer": {"kind": "agent", "id": str(reviewer.id), "name": reviewer.name}},
        context={},
    )
    db_session.add(run)
    await db_session.flush()

    session = Session(
        agent_id=reviewer.id,
        task_id=task.id,
        project_id=test_project.id,
        graph_run_id=run.id,
        adapter_type="api",
        status="completed",
        output="APPROVE",
        metadata_={},
        origin="graph",
    )
    db_session.add(session)
    await db_session.flush()

    engine = GraphEngineService()
    completion_event = BusEvent(
        id=uuid.uuid4(),
        project_id=test_project.id,
        event_type="session.completed",
        payload={"session_id": str(session.id)},
        source="system",
    )

    await engine.process_event(db_session, completion_event)
    session.metadata_ = {}
    await db_session.flush()

    await engine.process_event(db_session, completion_event)
    await db_session.flush()
    await db_session.refresh(run)
    await db_session.refresh(session)

    assert run.current_node == "approved"

    result = await db_session.execute(
        select(EventLog).where(
            EventLog.project_id == test_project.id,
            EventLog.event_type == "review.approved",
        )
    )
    events = result.scalars().all()
    assert len(events) == 1
    assert events[0].payload["session_id"] == str(session.id)
    assert events[0].dedup_key == f"derived_review:review.approved:{session.id}"


@pytest.mark.asyncio
async def test_engine_does_not_auto_emit_review_event_for_commented_output(db_session, test_project):
    from huddleroom.services.event_bus import BusEvent
    from huddleroom.services.graph_engine import GraphEngineService

    proto = await GraphService().load_from_yaml(db_session, Path("workspace/graphs/code_review.yaml"))
    reviewer = Agent(
        name=f"reviewer-auto-comment-{uuid.uuid4()}",
        role="reviewer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_review"],
        config={},
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="auto-comment-pr",
        artifact_type="pull_request",
        url="http://example.com/pr/comment",
        metadata_={"branch": "feature/comment"},
    )
    task = Task(project_id=test_project.id, title="Review auto comment PR", status="in_progress")
    db_session.add_all([reviewer, artifact, task])
    await db_session.flush()

    run = GraphRun(
        graph_id=proto.id,
        project_id=test_project.id,
        linked_task_id=task.id,
        artifact_id=artifact.id,
        current_node="ready_for_review",
        status="active",
        actor_assignments={"reviewer": {"kind": "agent", "id": str(reviewer.id), "name": reviewer.name}},
        context={},
    )
    db_session.add(run)
    await db_session.flush()

    session = Session(
        agent_id=reviewer.id,
        task_id=task.id,
        project_id=test_project.id,
        graph_run_id=run.id,
        adapter_type="api",
        status="completed",
        output="I left comments, but no approval yet.",
        metadata_={},
        origin="graph",
    )
    db_session.add(session)
    await db_session.flush()

    await GraphEngineService().process_event(
        db_session,
        BusEvent(
            id=uuid.uuid4(),
            project_id=test_project.id,
            event_type="session.completed",
            payload={"session_id": str(session.id)},
            source="system",
        ),
    )
    await db_session.flush()
    await db_session.refresh(run)
    await db_session.refresh(session)

    assert run.current_node == "ready_for_review"
    assert "auto_review_graph_event" not in session.metadata_
    result = await db_session.execute(
        select(EventLog).where(
            EventLog.project_id == test_project.id,
            EventLog.event_type.in_(["review.approved", "review.changes_requested"]),
        )
    )
    assert result.scalars().all() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("origin, graph_name", [("api", "code_review"), ("graph", None)])
async def test_engine_does_not_auto_emit_review_events_for_non_qualifying_sessions(
    db_session, test_project, origin, graph_name
):
    from huddleroom.services.event_bus import BusEvent
    from huddleroom.services.graph_engine import GraphEngineService

    if graph_name == "code_review":
        proto = await GraphService().load_from_yaml(db_session, Path("workspace/graphs/code_review.yaml"))
    else:
        proto = Graph(
            project_id=None,
            name=f"generic_review_{uuid.uuid4()}",
            version="1.0",
            definition={"nodes": {"ready_for_review": {"edges": []}}},
            triggers=[],
            is_active=True,
        )
        db_session.add(proto)
        await db_session.flush()

    reviewer = Agent(
        name=f"reviewer-non-qualifying-{uuid.uuid4()}",
        role="reviewer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_review"],
        config={},
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="non-qualifying-pr",
        artifact_type="pull_request",
        url="http://example.com/pr/non-qualifying",
        metadata_={"branch": "feature/non-qualifying"},
    )
    task = Task(project_id=test_project.id, title="Review non qualifying PR", status="in_progress")
    db_session.add_all([reviewer, artifact, task])
    await db_session.flush()

    run = GraphRun(
        graph_id=proto.id,
        project_id=test_project.id,
        linked_task_id=task.id,
        artifact_id=artifact.id,
        current_node="ready_for_review",
        status="active",
        actor_assignments={"reviewer": {"kind": "agent", "id": str(reviewer.id), "name": reviewer.name}},
        context={},
    )
    db_session.add(run)
    await db_session.flush()

    session = Session(
        agent_id=reviewer.id,
        task_id=task.id,
        project_id=test_project.id,
        graph_run_id=run.id,
        adapter_type="api",
        status="completed",
        output="APPROVE",
        metadata_={},
        origin=origin,
    )
    db_session.add(session)
    await db_session.flush()

    await GraphEngineService().process_event(
        db_session,
        BusEvent(
            id=uuid.uuid4(),
            project_id=test_project.id,
            event_type="session.completed",
            payload={"session_id": str(session.id)},
            source="system",
        ),
    )
    await db_session.flush()
    await db_session.refresh(run)
    await db_session.refresh(session)

    assert run.current_node == "ready_for_review"
    assert "auto_review_graph_event" not in session.metadata_
    result = await db_session.execute(
        select(EventLog).where(
            EventLog.project_id == test_project.id,
            EventLog.event_type.in_(["review.approved", "review.changes_requested"]),
        )
    )
    assert result.scalars().all() == []


@pytest.mark.asyncio
async def test_engine_guard_mismatch_does_not_step(db_session, test_project):
    from huddleroom.services.event_bus import BusEvent
    from huddleroom.services.graph_engine import GraphEngineService

    proto = await GraphService().load_from_yaml(db_session, Path("workspace/graphs/code_review.yaml"))
    reviewer = Agent(
        name=f"reviewer-{uuid.uuid4()}",
        role="reviewer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_review"],
        config={},
    )
    merger = Agent(
        name=f"pm-{uuid.uuid4()}",
        role="pm",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_merge"],
        config={},
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="guard-pr",
        artifact_type="pull_request",
        url="http://example.com",
        metadata_={"branch": "feature/guard"},
    )
    db_session.add_all(
        [reviewer, merger, artifact, Channel(project_id=test_project.id, name="general", channel_type="general", members=[])]
    )
    await db_session.flush()

    engine = GraphEngineService()
    await engine.process_event(
        db_session,
        BusEvent(
            id=uuid.uuid4(),
            project_id=test_project.id,
            event_type="code.pr_opened",
            payload={"artifact_id": str(artifact.id)},
            source="system",
        ),
    )
    await engine.process_event(
        db_session,
        BusEvent(
            id=uuid.uuid4(),
            project_id=test_project.id,
            event_type="test.passed",
            payload={"artifact_id": str(uuid.uuid4())},
            source="system",
        ),
    )
    await db_session.flush()

    run = (
        await db_session.execute(
            select(GraphRun).where(
                GraphRun.graph_id == proto.id,
                GraphRun.project_id == test_project.id,
            )
        )
    ).scalar_one()
    steps = list(
        (
            await db_session.execute(
                select(GraphRunStep).where(GraphRunStep.graph_run_id == run.id)
            )
        )
        .scalars()
        .all()
    )
    assert run.current_node == "opened"
    assert len(steps) == 1


@pytest.mark.asyncio
async def test_engine_processes_expired_timeouts_once(db_session, test_project):
    from huddleroom.services.graph_engine import GraphEngineService

    proto = Graph(
        project_id=None,
        name="timeout-regression",
        version="1.0",
        definition={"nodes": {"waiting": {}}, "start_node": "waiting"},
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

    timeout = GraphRunTimeout(
        graph_run_id=run.id,
        node_name="waiting",
        timeout_action="escalate",
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    db_session.add(timeout)
    await db_session.flush()

    engine = GraphEngineService()
    await engine.process_timeouts(db_session)
    await db_session.flush()
    await db_session.refresh(timeout)

    first_events = list(
        (
            await db_session.execute(
                select(EventLog).where(
                    EventLog.project_id == test_project.id,
                    EventLog.event_type.in_(["graph.run_escalated", "system.escalation_alert"]),
                )
            )
        )
        .scalars()
        .all()
    )
    assert timeout.resolved is True
    assert timeout.resolved_at is not None
    assert len(first_events) == 1
    assert first_events[0].payload["graph_run_id"] == str(run.id)

    await engine.process_timeouts(db_session)
    await db_session.flush()

    second_events = list(
        (
            await db_session.execute(
                select(EventLog).where(
                    EventLog.project_id == test_project.id,
                    EventLog.event_type.in_(["graph.run_escalated", "system.escalation_alert"]),
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(second_events) == 1


@pytest.mark.asyncio
async def test_advance_manually_records_step(db_session, test_project):
    from fastapi import HTTPException
    from huddleroom.services.graph_engine import GraphEngineService

    engine = GraphEngineService()
    with pytest.raises(HTTPException, match="Graph run not found"):
        await engine.advance_manually(db_session, uuid.uuid4(), "merged")

    proto = Graph(
        project_id=None,
        name="manual",
        version="1.0",
        definition={"nodes": {"opened": {}, "merged": {}}, "terminal_nodes": {"success": ["merged"]}},
        triggers=[],
        is_active=True,
    )
    db_session.add(proto)
    await db_session.flush()

    run = GraphRun(
        graph_id=proto.id,
        project_id=test_project.id,
        current_node="opened",
        status="active",
        actor_assignments={},
        context={},
    )
    db_session.add(run)
    await db_session.flush()

    updated = await engine.advance_manually(db_session, run.id, "merged", reason="manual override")
    await db_session.flush()

    steps = list(
        (
            await db_session.execute(
                select(GraphRunStep).where(GraphRunStep.graph_run_id == run.id)
            )
        )
        .scalars()
        .all()
    )
    assert updated.current_node == "merged"
    assert updated.status == "completed"
    assert steps[-1].edge_name == "manual_advance"
    assert steps[-1].trigger_reason == "manual override"


@pytest.mark.asyncio
async def test_advance_manually_rejects_invalid_target_node(db_session, test_project):
    from fastapi import HTTPException
    from huddleroom.services.graph_engine import GraphEngineService

    proto = Graph(
        project_id=None,
        name="manual-invalid-target",
        version="1.0",
        definition={"nodes": {"opened": {}, "merged": {}}, "terminal_nodes": {"success": ["merged"]}},
        triggers=[],
        is_active=True,
    )
    db_session.add(proto)
    await db_session.flush()

    run = GraphRun(
        graph_id=proto.id,
        project_id=test_project.id,
        current_node="opened",
        status="active",
        actor_assignments={},
        context={},
    )
    db_session.add(run)
    await db_session.flush()

    with pytest.raises(HTTPException):
        await GraphEngineService().advance_manually(db_session, run.id, "not_a_node")


@pytest.mark.asyncio
async def test_advance_manually_rejects_inactive_runs(db_session, test_project):
    from fastapi import HTTPException
    from huddleroom.services.graph_engine import GraphEngineService

    proto = Graph(
        project_id=None,
        name="manual-inactive",
        version="1.0",
        definition={"nodes": {"opened": {}, "merged": {}}, "terminal_nodes": {"success": ["merged"]}},
        triggers=[],
        is_active=True,
    )
    db_session.add(proto)
    await db_session.flush()

    run = GraphRun(
        graph_id=proto.id,
        project_id=test_project.id,
        current_node="merged",
        status="completed",
        actor_assignments={},
        context={},
        completed_at=datetime.now(timezone.utc),
    )
    db_session.add(run)
    await db_session.flush()

    with pytest.raises(HTTPException):
        await GraphEngineService().advance_manually(db_session, run.id, "opened")
