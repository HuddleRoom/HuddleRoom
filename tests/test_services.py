"""Unit tests for TaskService, AuthService, KnowledgeService, and cron trigger evaluation."""
from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from huddleroom.models.agent import Agent
from huddleroom.models.project import Project
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.models.user import User
from huddleroom.models.graph import Graph, GraphRun
from huddleroom.schemas.agent import AgentCreate, AgentUpdate
from huddleroom.schemas.channel import ChannelCreate
from huddleroom.schemas.knowledge import KnowledgeCreate
from huddleroom.schemas.project import ProjectCreate, ProjectUpdate
from huddleroom.schemas.session import SessionCreate, SessionResponse
from huddleroom.schemas.task import TaskCreate, TaskUpdate
from huddleroom.security import hash_password
from huddleroom.services.agent_service import AgentService
from huddleroom.services.auth_service import AuthService
from huddleroom.services.channel_service import ChannelService
from huddleroom.services.knowledge_service import KnowledgeService
from huddleroom.services.message_service import MessageService
from huddleroom.services.project_service import ProjectService
from huddleroom.services.session_service import SessionService
from huddleroom.services.task_service import TaskService
from huddleroom.workers.trigger_tasks import evaluate_cron_triggers_async


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _make_task(db: AsyncSession, project_id: uuid.UUID, status: str = "ready") -> Task:
    """Create a Task row with an explicit status (bypassing server_default)."""
    svc = TaskService()
    task = await svc.create(db, project_id, TaskCreate(title="Test Task"))
    # server_default only fires on INSERT; force the value in-memory too
    task.status = status
    await db.flush()
    return task


# ===========================================================================
# 1. TaskService.transition_status — state machine
# ===========================================================================

@pytest.mark.asyncio
async def test_task_valid_transition(db_session: AsyncSession, test_project):
    """ready -> in_progress is a valid transition and sets started_at."""
    task = await _make_task(db_session, test_project.id, status="ready")
    svc = TaskService()

    updated = await svc.transition_status(db_session, test_project.id, task.id, "in_progress")

    assert updated.status == "in_progress"
    assert updated.started_at is not None


@pytest.mark.asyncio
async def test_task_invalid_transition(db_session: AsyncSession, test_project):
    """ready -> done is NOT a valid transition; expect HTTPException 409."""
    task = await _make_task(db_session, test_project.id, status="ready")
    svc = TaskService()

    with pytest.raises(HTTPException) as exc_info:
        await svc.transition_status(db_session, test_project.id, task.id, "done")

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_task_transition_to_done_sets_completed_at(db_session: AsyncSession, test_project):
    """Transitioning in_progress -> done sets completed_at."""
    task = await _make_task(db_session, test_project.id, status="in_progress")
    svc = TaskService()

    updated = await svc.transition_status(db_session, test_project.id, task.id, "done")

    assert updated.status == "done"
    assert updated.completed_at is not None


# ===========================================================================
# 2. AuthService.authenticate — wrong password rejection
# ===========================================================================

@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_auth_wrong_password_rejected(db_session: AsyncSession):
    """authenticate returns None when the password is wrong."""
    svc = AuthService()
    email = f"user-{uuid.uuid4()}@example.com"
    await svc.create_user(db_session, email=email, password="correct_password")

    result = await svc.authenticate(db_session, email=email, password="wrong_password")

    assert result is None


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_auth_valid_credentials_succeed(db_session: AsyncSession):
    """authenticate returns the User object when credentials are correct."""
    svc = AuthService()
    email = f"user-{uuid.uuid4()}@example.com"
    created = await svc.create_user(db_session, email=email, password="secret123")

    result = await svc.authenticate(db_session, email=email, password="secret123")

    assert result is not None
    assert isinstance(result, User)
    assert result.id == created.id
    assert result.email == email


# ===========================================================================
# 3. KnowledgeService.search — relevance filtering
# ===========================================================================

def _fake_embedding(text: str) -> list[float]:
    """Return a deterministic unit vector so cosine similarity is reproducible."""
    # Use a fixed non-zero vector; cosine sim of identical vectors = 1.0
    vec = [1.0] + [0.0] * 1535
    return vec


@pytest.mark.asyncio
async def test_knowledge_min_relevance_filter(db_session: AsyncSession, test_project):
    """
    With min_relevance_score=0.99 only near-identical vectors pass.
    With min_relevance_score=0.0 all items are returned.
    """
    svc = KnowledgeService()
    fake_vec = _fake_embedding("anything")

    with patch(
        "huddleroom.services.knowledge_service.embedding_service.generate_embedding",
        new=AsyncMock(return_value=fake_vec),
    ):
        # Create two knowledge items
        await svc.create(
            db_session,
            test_project.id,
            KnowledgeCreate(content="Alpha content", content_type="text"),
        )
        await svc.create(
            db_session,
            test_project.id,
            KnowledgeCreate(content="Beta content", content_type="text"),
        )

        # All embeddings are identical to query embedding → cosine sim == 1.0
        # high threshold: should return items (all score == 1.0 >= 0.99)
        high_results = await svc.search(
            db_session, test_project.id, query="anything", min_relevance_score=0.99
        )

        # low threshold: should also return all items
        low_results = await svc.search(
            db_session, test_project.id, query="anything", min_relevance_score=0.0
        )

    # Both should return the same 2 items since all cosine sims are 1.0
    assert len(low_results) >= 2
    assert len(high_results) >= 2
    # All scores must be >= the respective threshold
    assert all(r.relevance_score >= 0.99 for r in high_results)
    assert all(r.relevance_score >= 0.0 for r in low_results)


@pytest.mark.asyncio
async def test_knowledge_min_relevance_filter_excludes_low_scores(db_session: AsyncSession, test_project):
    """
    When query embedding is orthogonal to stored embeddings the score is 0,
    so items are excluded when min_relevance_score > 0.
    """
    svc = KnowledgeService()
    store_vec = [1.0] + [0.0] * 1535   # stored item vector
    query_vec = [0.0, 1.0] + [0.0] * 1534  # orthogonal → cosine sim = 0.0

    # Patch for create: return store_vec; for search query: return query_vec
    call_count = {"n": 0}

    async def side_effect(text: str) -> list[float]:
        call_count["n"] += 1
        # First two calls are for create(), subsequent calls are for search()
        if call_count["n"] <= 1:
            return store_vec
        return query_vec

    with patch(
        "huddleroom.services.knowledge_service.embedding_service.generate_embedding",
        new=side_effect,
    ):
        await svc.create(
            db_session,
            test_project.id,
            KnowledgeCreate(content="Stored content", content_type="text"),
        )

        results = await svc.search(
            db_session, test_project.id, query="unrelated", min_relevance_score=0.5
        )

    assert len(results) == 0


@pytest.mark.asyncio
async def test_knowledge_search_returns_results(db_session: AsyncSession, test_project):
    """search returns at least one result when content exists and embeddings match."""
    svc = KnowledgeService()
    fake_vec = _fake_embedding("anything")

    with patch(
        "huddleroom.services.knowledge_service.embedding_service.generate_embedding",
        new=AsyncMock(return_value=fake_vec),
    ):
        await svc.create(
            db_session,
            test_project.id,
            KnowledgeCreate(content="Machine learning basics", content_type="text"),
        )

        results = await svc.search(
            db_session,
            test_project.id,
            query="machine learning",
            min_relevance_score=0.0,
        )

    assert len(results) >= 1
    assert any("Machine learning" in r.content for r in results)


# ===========================================================================
# 4. evaluate_cron_triggers_async — cron trigger evaluation
# ===========================================================================

def _make_session_factory(db_session: AsyncSession):
    """Return a session factory that yields the test session (no new connection)."""
    @asynccontextmanager
    async def _factory():
        yield db_session
    return _factory


@pytest.mark.asyncio
async def test_cron_task_creates_session(db_session: AsyncSession, runnable_project, test_agent):
    """A ready cron task with spec '* * * * *' should enqueue a session."""
    from sqlalchemy import select as sa_select
    task = Task(
        project_id=runnable_project.id,
        title="Cron Task",
        status="ready",
        assigned_to=test_agent.id,
        trigger={"type": "cron", "spec": "* * * * *"},
    )
    db_session.add(task)
    await db_session.flush()

    with patch("huddleroom.workers.task_runner.dispatch_session", new=AsyncMock(return_value="test-task-id")):
        evaluated, enqueued = await evaluate_cron_triggers_async(
            session_factory=_make_session_factory(db_session)
        )

    assert evaluated >= 1
    assert enqueued >= 1

    result = await db_session.execute(sa_select(Session).where(Session.task_id == task.id))
    sessions = result.scalars().all()
    assert len(sessions) >= 1


@pytest.mark.asyncio
async def test_non_cron_task_skipped(db_session: AsyncSession, test_project, test_agent):
    """A task with trigger type 'webhook' should not create a session."""
    task = Task(
        project_id=test_project.id,
        title="Webhook Task",
        status="ready",
        assigned_to=test_agent.id,
        trigger={"type": "webhook", "url": "http://example.com"},
    )
    db_session.add(task)
    await db_session.flush()

    _, enqueued = await evaluate_cron_triggers_async(
        session_factory=_make_session_factory(db_session)
    )

    # webhook tasks don't count as cron — no session created for them
    result = await db_session.execute(
        __import__("sqlalchemy").select(Session).where(Session.task_id == task.id)
    )
    assert len(result.scalars().all()) == 0


@pytest.mark.asyncio
async def test_task_without_trigger_skipped(db_session: AsyncSession, test_project, test_agent):
    """A task with no trigger should never create a session via cron evaluation."""
    task = Task(
        project_id=test_project.id,
        title="Manual Task",
        status="ready",
        assigned_to=test_agent.id,
        trigger=None,
    )
    db_session.add(task)
    await db_session.flush()

    _, enqueued = await evaluate_cron_triggers_async(
        session_factory=_make_session_factory(db_session)
    )

    result = await db_session.execute(
        __import__("sqlalchemy").select(Session).where(Session.task_id == task.id)
    )
    assert len(result.scalars().all()) == 0


@pytest.mark.asyncio
async def test_pending_session_prevents_duplicate(db_session: AsyncSession, test_project, test_agent):
    """If a pending/running session already exists, no duplicate should be created."""
    task = Task(
        project_id=test_project.id,
        title="Cron Task No Dup",
        status="ready",
        assigned_to=test_agent.id,
        trigger={"type": "cron", "spec": "* * * * *"},
    )
    db_session.add(task)
    await db_session.flush()

    # Pre-create a pending session
    existing = Session(
        task_id=task.id,
        agent_id=test_agent.id,
        project_id=test_project.id,
        adapter_type="api",
        status="pending",
        input_context={},
        metadata_={},
    )
    db_session.add(existing)
    await db_session.flush()

    _, enqueued = await evaluate_cron_triggers_async(
        session_factory=_make_session_factory(db_session)
    )

    assert enqueued == 0


# ===========================================================================
# 5. ProjectService — CRUD
# ===========================================================================

@pytest.mark.asyncio
async def test_project_create_and_get(db_session: AsyncSession, tmp_path):
    """create persists a project row; get retrieves it by id."""
    svc = ProjectService()
    project = await svc.create(
        db_session,
        ProjectCreate(name="My Project", description="desc", workspace_path=str(tmp_path)),
    )
    assert project.id is not None
    assert project.name == "My Project"

    fetched = await svc.get(db_session, project.id)
    assert fetched is not None
    assert fetched.id == project.id


@pytest.mark.asyncio
async def test_project_get_or_404_missing(db_session: AsyncSession):
    """get_or_404 raises HTTPException 404 for unknown id."""
    svc = ProjectService()
    with pytest.raises(HTTPException) as exc_info:
        await svc.get_or_404(db_session, uuid.uuid4())
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_project_update(db_session: AsyncSession, tmp_path):
    """update modifies an existing project's fields."""
    svc = ProjectService()
    project = await svc.create(
        db_session,
        ProjectCreate(name="Old Name", workspace_path=str(tmp_path)),
    )
    updated = await svc.update(db_session, project.id, ProjectUpdate(name="New Name"))
    assert updated.name == "New Name"


@pytest.mark.asyncio
async def test_project_archive(db_session: AsyncSession, tmp_path):
    """archive sets status to 'archived'."""
    svc = ProjectService()
    project = await svc.create(
        db_session,
        ProjectCreate(name="Archivable Project", workspace_path=str(tmp_path)),
    )
    archived = await svc.archive(db_session, project.id)
    assert archived.status == "archived"


@pytest.mark.asyncio
async def test_project_list(db_session: AsyncSession, tmp_path):
    """list returns projects in descending created_at order."""
    svc = ProjectService()
    await svc.create(db_session, ProjectCreate(name="Project A", workspace_path=str(tmp_path)))
    await svc.create(db_session, ProjectCreate(name="Project B", workspace_path=str(tmp_path)))
    items, next_cursor = await svc.list(db_session, limit=50)
    names = [p.name for p in items]
    assert "Project A" in names
    assert "Project B" in names
    assert next_cursor is None  # fewer than limit + 1 items


# ===========================================================================
# 6. AgentService — CRUD + build_context
# ===========================================================================

def _agent_create_data(**kwargs) -> AgentCreate:
    defaults = dict(
        name=f"agent-{uuid.uuid4()}",
        role="developer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
    )
    defaults.update(kwargs)
    return AgentCreate(**defaults)


@pytest.mark.asyncio
async def test_agent_create_and_get(db_session: AsyncSession):
    """create persists an agent; get retrieves it."""
    svc = AgentService()
    agent = await svc.create(db_session, _agent_create_data(name="test-agent-unique"))
    assert agent.id is not None

    fetched = await svc.get(db_session, agent.id)
    assert fetched is not None
    assert fetched.id == agent.id


@pytest.mark.asyncio
@pytest.mark.parametrize("cli_runtime", [None, "", "   "], ids=["missing", "empty", "blank"])
async def test_agent_create_cli_requires_nonblank_runtime(db_session: AsyncSession, cli_runtime: str | None):
    svc = AgentService()

    with pytest.raises(HTTPException) as exc_info:
        await svc.create(db_session, _agent_create_data(adapter_type="cli", cli_runtime=cli_runtime))

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("config_runtime", ["", "   "], ids=["empty_config", "blank_config"])
async def test_agent_create_cli_rejects_blank_config_runtime(
    db_session: AsyncSession, config_runtime: str
):
    svc = AgentService()

    with pytest.raises(HTTPException) as exc_info:
        await svc.create(db_session, _agent_create_data(
            adapter_type="cli", cli_runtime="codex", config={"cli_runtime": config_runtime}
        ))

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_agent_create_cli_uses_config_runtime_override(db_session: AsyncSession):
    svc = AgentService()

    agent = await svc.create(db_session, _agent_create_data(
        adapter_type="cli", cli_runtime=None, config={"cli_runtime": "opencode"}
    ))

    assert agent.config["cli_runtime"] == "opencode"


@pytest.mark.asyncio
async def test_agent_get_or_404_missing(db_session: AsyncSession):
    """get_or_404 raises 404 for non-existent id."""
    svc = AgentService()
    with pytest.raises(HTTPException) as exc_info:
        await svc.get_or_404(db_session, uuid.uuid4())
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_agent_update(db_session: AsyncSession):
    """update modifies agent fields."""
    svc = AgentService()
    agent = await svc.create(db_session, _agent_create_data())
    updated = await svc.update(db_session, agent.id, AgentUpdate(role="reviewer"))
    assert updated.role == "reviewer"


@pytest.mark.parametrize(
    "initial_config,initial_cli_runtime,update_payload,expected_config",
    [
        # Case 1: cli_runtime at top level only
        ({}, "claude-code", {}, {}),
        # Case 2: cli_runtime in config dict
        ({"cli_runtime": "codex", "other": "value"}, "claude-code", {}, {"other": "value"}),
        # Case 3: conflicting cli_runtime in both places + payload
        ({"cli_runtime": "aider"}, "claude-code", {"cli_runtime": "codex"}, {}),
        # Case 4: cli_runtime in update payload config
        ({"existing": "kept"}, "claude-code", {"config": {"cli_runtime": "codex", "other": "value"}}, {"other": "value"}),
    ],
    ids=[
        "clears_top_level_cli_runtime",
        "clears_config_cli_runtime",
        "ignores_conflicting_payload_cli_runtime",
        "strips_cli_runtime_from_payload_config",
    ],
)
@pytest.mark.asyncio
async def test_agent_update_switch_to_api_clears_cli_runtime(
    db_session: AsyncSession, initial_config, initial_cli_runtime, update_payload, expected_config
):
    """Switching to API adapter must clear all CLI runtime variants."""
    svc = AgentService()

    # Build initial agent data
    agent_data = _agent_create_data(adapter_type="cli", cli_runtime=initial_cli_runtime)
    if initial_config:
        agent_data.config = initial_config

    agent = await svc.create(db_session, agent_data)

    # Build update payload
    update_data = AgentUpdate(adapter_type="api", **update_payload)
    updated = await svc.update(db_session, agent.id, update_data)

    assert updated.adapter_type == "api"
    assert updated.cli_runtime is None
    assert updated.config == expected_config


@pytest.mark.asyncio
async def test_agent_update_switch_to_cli_keeps_cli_runtime(db_session: AsyncSession):
    """Switching an agent to the CLI adapter must preserve an explicitly provided CLI runtime."""
    svc = AgentService()
    agent = await svc.create(db_session, _agent_create_data(adapter_type="api"))

    updated = await svc.update(
        db_session,
        agent.id,
        AgentUpdate(adapter_type="cli", cli_runtime="codex"),
    )

    assert updated.adapter_type == "cli"
    assert updated.cli_runtime == "codex"


@pytest.mark.asyncio
@pytest.mark.parametrize("update", [AgentUpdate(adapter_type="cli"), AgentUpdate(adapter_type="cli", cli_runtime="")])
async def test_agent_update_switch_to_cli_requires_nonblank_runtime(db_session: AsyncSession, update: AgentUpdate):
    svc = AgentService()
    agent = await svc.create(db_session, _agent_create_data(adapter_type="api"))

    with pytest.raises(HTTPException) as exc_info:
        await svc.update(db_session, agent.id, update)

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_agent_update_cli_rejects_explicit_blank_runtime(db_session: AsyncSession):
    svc = AgentService()
    agent = await svc.create(db_session, _agent_create_data(adapter_type="cli", cli_runtime="codex"))

    with pytest.raises(HTTPException) as exc_info:
        await svc.update(db_session, agent.id, AgentUpdate(cli_runtime=" "))

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_agent_update_cli_rejects_blank_config_runtime(db_session: AsyncSession):
    svc = AgentService()
    agent = await svc.create(db_session, _agent_create_data(adapter_type="cli", cli_runtime="codex"))

    with pytest.raises(HTTPException) as exc_info:
        await svc.update(db_session, agent.id, AgentUpdate(config={"cli_runtime": " "}))

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_agent_update_cli_rejects_config_replacement_that_removes_runtime(db_session: AsyncSession):
    svc = AgentService()
    agent = await svc.create(db_session, _agent_create_data(
        adapter_type="cli", cli_runtime=None, config={"cli_runtime": "pi"}
    ))

    with pytest.raises(HTTPException) as exc_info:
        await svc.update(db_session, agent.id, AgentUpdate(config={"temperature": 0.2}))

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_agent_update_legacy_blank_config_runtime_allows_unrelated_change(db_session: AsyncSession):
    svc = AgentService()
    agent = Agent(
        name=f"agent-{uuid.uuid4()}", role="developer", provider="openai", model="gpt-4o-mini",
        adapter_type="cli", cli_runtime="codex", capabilities=[], config={"cli_runtime": ""},
    )
    db_session.add(agent)
    await db_session.flush()

    updated = await svc.update(db_session, agent.id, AgentUpdate(role="reviewer"))

    assert updated.role == "reviewer"
    assert updated.config["cli_runtime"] == ""


@pytest.mark.asyncio
async def test_agent_update_switch_to_cli_uses_config_runtime_override(db_session: AsyncSession):
    svc = AgentService()
    agent = await svc.create(db_session, _agent_create_data(adapter_type="api"))

    updated = await svc.update(
        db_session, agent.id, AgentUpdate(adapter_type="cli", config={"cli_runtime": "pi"})
    )

    assert updated.adapter_type == "cli"
    assert updated.config["cli_runtime"] == "pi"


@pytest.mark.asyncio
@pytest.mark.parametrize("cli_runtime", ["", "custom-legacy-runtime"], ids=["blank_legacy", "custom_legacy"])
async def test_agent_update_legacy_cli_allows_unrelated_change(
    db_session: AsyncSession, cli_runtime: str
):
    svc = AgentService()
    agent = Agent(
        name=f"agent-{uuid.uuid4()}", role="developer", provider="openai", model="gpt-4o-mini",
        adapter_type="cli", cli_runtime=cli_runtime, capabilities=[], config={},
    )
    db_session.add(agent)
    await db_session.flush()

    updated = await svc.update(db_session, agent.id, AgentUpdate(role="reviewer"))

    assert updated.role == "reviewer"
    assert updated.cli_runtime == cli_runtime


@pytest.mark.asyncio
async def test_agent_delete_deactivates(db_session: AsyncSession):
    """delete sets is_active=False instead of removing the row."""
    svc = AgentService()
    agent = await svc.create(db_session, _agent_create_data())
    await svc.delete(db_session, agent.id)
    fetched = await svc.get(db_session, agent.id)
    assert fetched is not None
    assert fetched.is_active is False


@pytest.mark.asyncio
async def test_agent_list_filter_by_role(db_session: AsyncSession):
    """list with role filter returns only matching agents."""
    svc = AgentService()
    await svc.create(db_session, _agent_create_data(role="qa"))
    await svc.create(db_session, _agent_create_data(role="ops"))
    qa_items, _ = await svc.list(db_session, role="qa")
    roles = {a.role for a in qa_items}
    assert "qa" in roles
    assert "ops" not in roles


@pytest.mark.asyncio
async def test_agent_build_context(db_session: AsyncSession, test_project):
    """build_context returns AgentContextResponse with correct agent data."""
    svc = AgentService()
    agent = await svc.create(db_session, _agent_create_data())
    ctx = await svc.build_context(db_session, agent.id)
    assert ctx.agent.id == agent.id
    assert isinstance(ctx.current_tasks, list)


# ===========================================================================
# 7. SessionService — create, get, cancel, update_status
# ===========================================================================

@pytest.mark.asyncio
async def test_session_create(db_session: AsyncSession, runnable_project, test_agent):
    """create stores a session row and dispatches (mocked)."""
    svc = SessionService()
    data = SessionCreate(
        agent_id=test_agent.id,
        project_id=runnable_project.id,
    )
    with patch("huddleroom.workers.task_runner.dispatch_session", new=AsyncMock(return_value="fake-task-id")):
        session = await svc.create(db_session, data)

    assert session.id is not None
    assert session.agent_id == test_agent.id
    assert session.project_id == runnable_project.id
    assert session.runner_task_id is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_session_create_dispatches_only_after_commit(test_engine, tmp_path):
    """Dispatch must wait for commit so background workers can read the session row."""
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as db_session:
        workspace = tmp_path / "dispatch-workspace"
        workspace.mkdir()
        project = Project(
            name=f"Project-{uuid.uuid4()}",
            description="dispatch test",
            workspace_path=str(workspace.resolve()),
            config={},
        )
        agent = Agent(
            name=f"agent-{uuid.uuid4()}",
            role="developer",
            provider="openai",
            model="gpt-4o-mini",
            adapter_type="api",
            capabilities=[],
            config={},
        )
        db_session.add_all([project, agent])
        await db_session.flush()

        svc = SessionService()
        data = SessionCreate(
            agent_id=agent.id,
            project_id=project.id,
        )
        register_session = Mock(return_value="deferred-task-id")

        with patch("huddleroom.workers.task_runner.register_session", new=register_session):
            session = await svc.create(db_session, data)

            assert session.runner_task_id is not None
            register_session.assert_not_called()

            await db_session.commit()

        register_session.assert_called_once_with(
            str(session.id), "api", session.project_id, task_id=session.runner_task_id
        )


@pytest.mark.asyncio
async def test_session_create_unknown_agent(db_session: AsyncSession, runnable_project):
    """create with an unknown agent_id raises HTTPException 404."""
    svc = SessionService()
    data = SessionCreate(agent_id=uuid.uuid4(), project_id=runnable_project.id)
    with pytest.raises(HTTPException) as exc_info:
        await svc.create(db_session, data)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_session_create_rejects_unknown_graph_run(
    db_session: AsyncSession, runnable_project, test_agent
):
    """graph_run_id must reference an existing run in the same project."""
    svc = SessionService()
    data = SessionCreate(
        agent_id=test_agent.id,
        project_id=runnable_project.id,
        graph_run_id=uuid.uuid4(),
    )

    with pytest.raises(HTTPException) as exc_info:
        await svc.create(db_session, data)

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_session_create_rejects_mismatched_graph_task(
    db_session: AsyncSession, runnable_project, test_agent
):
    """graph_run_id must match the requested task when both are supplied."""
    graph = Graph(project_id=None, name="session-validation", version="1.0", definition={}, triggers=[])
    linked_task = Task(project_id=runnable_project.id, title="Linked task")
    other_task = Task(project_id=runnable_project.id, title="Other task")
    db_session.add_all([graph, linked_task, other_task])
    await db_session.flush()

    run = GraphRun(
        graph_id=graph.id,
        project_id=runnable_project.id,
        linked_task_id=linked_task.id,
        current_node="waiting",
        status="active",
        actor_assignments={},
        context={},
    )
    db_session.add(run)
    await db_session.flush()

    data = SessionCreate(
        agent_id=test_agent.id,
        task_id=other_task.id,
        project_id=runnable_project.id,
        graph_run_id=run.id,
    )

    with pytest.raises(HTTPException) as exc_info:
        await SessionService().create(db_session, data)

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_session_create_rejects_task_from_different_graph_run(
    db_session: AsyncSession, runnable_project, test_agent
):
    """an unlinked graph run can only create sessions for its own child tasks."""
    graph = Graph(project_id=None, name="session-validation-child", version="1.0", definition={}, triggers=[])
    db_session.add(graph)
    await db_session.flush()

    target_run = GraphRun(
        graph_id=graph.id,
        project_id=runnable_project.id,
        current_node="waiting",
        status="active",
        actor_assignments={},
        context={},
    )
    other_run = GraphRun(
        graph_id=graph.id,
        project_id=runnable_project.id,
        current_node="waiting",
        status="active",
        actor_assignments={},
        context={},
    )
    db_session.add_all([target_run, other_run])
    await db_session.flush()

    other_task = Task(
        project_id=runnable_project.id,
        title="Other graph child task",
        graph_run_id=other_run.id,
    )
    db_session.add(other_task)
    await db_session.flush()

    data = SessionCreate(
        agent_id=test_agent.id,
        task_id=other_task.id,
        project_id=runnable_project.id,
        graph_run_id=target_run.id,
    )

    with pytest.raises(HTTPException) as exc_info:
        await SessionService().create(db_session, data)

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_session_create_accepts_matching_graph_task(
    db_session: AsyncSession, runnable_project, test_agent
):
    """valid graph_run_id/task_id pairs are persisted on the session."""
    graph = Graph(project_id=None, name="session-validation-ok", version="1.0", definition={}, triggers=[])
    task = Task(project_id=runnable_project.id, title="Linked task")
    db_session.add_all([graph, task])
    await db_session.flush()

    run = GraphRun(
        graph_id=graph.id,
        project_id=runnable_project.id,
        linked_task_id=task.id,
        current_node="waiting",
        status="active",
        actor_assignments={},
        context={},
    )
    db_session.add(run)
    await db_session.flush()

    data = SessionCreate(
        agent_id=test_agent.id,
        task_id=task.id,
        project_id=runnable_project.id,
        graph_run_id=run.id,
    )
    with patch("huddleroom.workers.task_runner.dispatch_session", new=AsyncMock(return_value="fake-task-id")):
        session = await SessionService().create(db_session, data)

    assert session.graph_run_id == run.id


@pytest.mark.asyncio
async def test_session_response_includes_graph_run_id(db_session: AsyncSession, test_project, test_agent):
    """SessionResponse exposes graph run linkage to API callers."""
    graph = Graph(project_id=None, name="session-response", version="1.0", definition={}, triggers=[])
    db_session.add(graph)
    await db_session.flush()
    run = GraphRun(
        graph_id=graph.id,
        project_id=test_project.id,
        current_node="waiting",
        status="active",
        actor_assignments={},
        context={},
    )
    db_session.add(run)
    await db_session.flush()
    raw_session = Session(
        agent_id=test_agent.id,
        project_id=test_project.id,
        graph_run_id=run.id,
        adapter_type="api",
        status="pending",
        input_context={},
        metadata_={},
    )
    db_session.add(raw_session)
    await db_session.flush()

    response = SessionResponse.model_validate(raw_session)

    assert response.graph_run_id == run.id


@pytest.mark.asyncio
async def test_session_get_or_404_missing(db_session: AsyncSession):
    """get_or_404 raises 404 for unknown session id."""
    svc = SessionService()
    with pytest.raises(HTTPException) as exc_info:
        await svc.get_or_404(db_session, uuid.uuid4())
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_session_cancel(db_session: AsyncSession, test_project, test_agent):
    """cancel sets status to 'cancelled'."""
    svc = SessionService()
    # Create a pending session directly (bypass dispatch)
    from huddleroom.models.session import Session as SessionModel
    raw_session = SessionModel(
        agent_id=test_agent.id,
        project_id=test_project.id,
        adapter_type="api",
        status="pending",
        input_context={},
        metadata_={},
    )
    db_session.add(raw_session)
    await db_session.flush()

    cancelled = await svc.cancel(db_session, raw_session.id)
    assert cancelled.status == "cancelled"


@pytest.mark.asyncio
async def test_session_cancel_already_done(db_session: AsyncSession, test_project, test_agent):
    """cancel raises 409 if session is already completed."""
    svc = SessionService()
    from huddleroom.models.session import Session as SessionModel
    raw_session = SessionModel(
        agent_id=test_agent.id,
        project_id=test_project.id,
        adapter_type="api",
        status="completed",
        input_context={},
        metadata_={},
    )
    db_session.add(raw_session)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await svc.cancel(db_session, raw_session.id)
    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_session_update_status_running(db_session: AsyncSession, test_project, test_agent):
    """update_status to 'running' sets started_at."""
    svc = SessionService()
    from huddleroom.models.session import Session as SessionModel
    raw_session = SessionModel(
        agent_id=test_agent.id,
        project_id=test_project.id,
        adapter_type="api",
        status="pending",
        input_context={},
        metadata_={},
    )
    db_session.add(raw_session)
    await db_session.flush()

    updated = await svc.update_status(db_session, raw_session.id, "running")
    assert updated.status == "running"
    assert updated.started_at is not None


@pytest.mark.asyncio
async def test_session_update_status_completed(db_session: AsyncSession, test_project, test_agent):
    """update_status to 'completed' sets ended_at and stores output."""
    svc = SessionService()
    from huddleroom.models.session import Session as SessionModel
    raw_session = SessionModel(
        agent_id=test_agent.id,
        project_id=test_project.id,
        adapter_type="api",
        status="running",
        input_context={},
        metadata_={},
    )
    db_session.add(raw_session)
    await db_session.flush()

    updated = await svc.update_status(
        db_session, raw_session.id, "completed", output="result text"
    )
    assert updated.status == "completed"
    assert updated.output == "result text"
    assert updated.ended_at is not None


@pytest.mark.asyncio
async def test_session_get_output(db_session: AsyncSession, test_project, test_agent):
    """get_output returns SessionOutputResponse with correct fields."""
    svc = SessionService()
    from huddleroom.models.session import Session as SessionModel
    raw_session = SessionModel(
        agent_id=test_agent.id,
        project_id=test_project.id,
        adapter_type="api",
        status="completed",
        input_context={},
        output="final output",
        metadata_={},
    )
    db_session.add(raw_session)
    await db_session.flush()

    out = await svc.get_output(db_session, raw_session.id)
    assert out.session_id == raw_session.id
    assert out.output == "final output"
    assert out.status == "completed"


# ===========================================================================
# 8. ChannelService — CRUD + get_or_create_for_task
# ===========================================================================

@pytest.mark.asyncio
async def test_channel_create_and_get(db_session: AsyncSession, test_project):
    """create stores a channel; get retrieves it."""
    svc = ChannelService()
    channel = await svc.create(
        db_session, test_project.id, ChannelCreate(name="general", channel_type="general")
    )
    assert channel.id is not None
    assert channel.name == "general"

    fetched = await svc.get(db_session, channel.id)
    assert fetched is not None
    assert fetched.id == channel.id


@pytest.mark.asyncio
async def test_channel_get_or_404_missing(db_session: AsyncSession):
    """get_or_404 raises 404 for unknown channel id."""
    svc = ChannelService()
    with pytest.raises(HTTPException) as exc_info:
        await svc.get_or_404(db_session, uuid.uuid4())
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_channel_list(db_session: AsyncSession, test_project):
    """list returns channels belonging to the given project."""
    svc = ChannelService()
    await svc.create(db_session, test_project.id, ChannelCreate(name="ch-1", channel_type="general"))
    await svc.create(db_session, test_project.id, ChannelCreate(name="ch-2", channel_type="general"))
    channels = await svc.list(db_session, test_project.id)
    names = [c.name for c in channels]
    assert "ch-1" in names
    assert "ch-2" in names


@pytest.mark.asyncio
async def test_channel_get_or_create_for_task_creates(db_session: AsyncSession, test_project, test_agent):
    """get_or_create_for_task creates a new task channel if none exists."""
    svc = ChannelService()
    from huddleroom.schemas.task import TaskCreate as TC
    from huddleroom.services.task_service import TaskService as TS
    task = await TS().create(db_session, test_project.id, TC(title="Channel Task"))
    channel = await svc.get_or_create_for_task(db_session, test_project.id, task.id)
    assert channel.task_id == task.id
    assert channel.channel_type == "task"


@pytest.mark.asyncio
async def test_channel_get_or_create_for_task_idempotent(db_session: AsyncSession, test_project):
    """get_or_create_for_task returns the same channel on second call."""
    svc = ChannelService()
    from huddleroom.schemas.task import TaskCreate as TC
    from huddleroom.services.task_service import TaskService as TS
    task = await TS().create(db_session, test_project.id, TC(title="Idempotent Task"))
    ch1 = await svc.get_or_create_for_task(db_session, test_project.id, task.id)
    ch2 = await svc.get_or_create_for_task(db_session, test_project.id, task.id)
    assert ch1.id == ch2.id


# ===========================================================================
# 9. MessageService — create + list
# ===========================================================================

@pytest.mark.asyncio
async def test_message_create_and_list(db_session: AsyncSession, test_project, test_user):
    """create stores a message; list retrieves it in the channel."""
    channel_svc = ChannelService()
    msg_svc = MessageService()

    channel = await channel_svc.create(
        db_session, test_project.id, ChannelCreate(name="msg-ch", channel_type="general")
    )
    msg = await msg_svc.create(
        db_session,
        channel_id=channel.id,
        content="Hello world",
        sender_user_id=test_user.id,
    )
    assert msg.id is not None
    assert msg.content == "Hello world"
    assert msg.sender_user_id == test_user.id

    messages, cursor = await msg_svc.list(db_session, channel.id)
    assert any(m.id == msg.id for m in messages)
    assert cursor is None  # only one message, no pagination


@pytest.mark.asyncio
async def test_message_list_pagination(db_session: AsyncSession, test_project, test_agent):
    """list returns a next_cursor when there are more messages than the limit."""
    channel_svc = ChannelService()
    msg_svc = MessageService()

    channel = await channel_svc.create(
        db_session, test_project.id, ChannelCreate(name="page-ch", channel_type="general")
    )
    for i in range(3):
        await msg_svc.create(
            db_session, channel_id=channel.id, content=f"msg {i}",
            sender_agent_id=test_agent.id,
        )

    items, next_cursor = await msg_svc.list(db_session, channel.id, limit=2)
    assert len(items) == 2
    assert next_cursor is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_recover_orphaned_sessions(db_session: AsyncSession, test_project):
    """Recovery fails stale sessions with explicit reasons and leaves recent sessions alone."""
    from datetime import datetime, timezone, timedelta
    from huddleroom.models.session import Session
    from huddleroom.services.session_service import SessionService

    agent_svc = AgentService()
    agent = await agent_svc.create(db_session, _agent_create_data(
        name="OrphanAgent", adapter_type="api"
    ))

    old_time = datetime.now(timezone.utc) - timedelta(hours=2)
    session = Session(
        agent_id=agent.id,
        project_id=test_project.id,
        adapter_type="api",
        status="running",
        metadata_={},
    )
    session.started_at = old_time
    db_session.add(session)
    await db_session.flush()

    recent_session = Session(
        agent_id=agent.id,
        project_id=test_project.id,
        adapter_type="api",
        status="running",
        metadata_={},
    )
    recent_session.started_at = datetime.now(timezone.utc)
    db_session.add(recent_session)
    await db_session.flush()

    # pending session with NULL started_at (crash before it was dispatched)
    pending_session = Session(
        agent_id=agent.id,
        project_id=test_project.id,
        adapter_type="api",
        status="pending",
        metadata_={},
    )
    # started_at stays None; use created_at to simulate old session
    db_session.add(pending_session)
    await db_session.flush()
    # Force created_at to be old
    pending_session.created_at = datetime.now(timezone.utc) - timedelta(hours=2)
    await db_session.flush()

    svc = SessionService()
    count = await svc.recover_orphaned_sessions(db_session, timeout_seconds=3600)

    assert count == 2
    await db_session.refresh(session)
    await db_session.refresh(recent_session)
    await db_session.refresh(pending_session)
    assert session.status == "failed"
    assert session.error == "stale_running_session: exceeded recovery timeout before watchdog pass"
    assert recent_session.status == "running"
    assert pending_session.status == "failed"
    assert pending_session.error == "stale_pending_session: never started before recovery timeout"


@pytest.mark.asyncio(loop_scope="session")
async def test_recover_orphaned_sessions_watchdog_does_not_redispatch_recent_pending(
    db_session: AsyncSession,
    test_project,
    monkeypatch,
):
    """Periodic watchdog should avoid redispatching recent healthy pending sessions."""
    from huddleroom.models.session import Session
    from huddleroom.services.session_service import SessionService

    agent_svc = AgentService()
    agent = await agent_svc.create(db_session, _agent_create_data(
        name="WatchdogAgent", adapter_type="api"
    ))

    recent_pending_session = Session(
        agent_id=agent.id,
        project_id=test_project.id,
        adapter_type="api",
        status="pending",
        metadata_={},
    )
    db_session.add(recent_pending_session)
    await db_session.flush()

    dispatch_calls: list[tuple[str, str, uuid.UUID]] = []

    async def fake_dispatch_session(session_id: str, adapter_type: str, project_id: uuid.UUID) -> str:
        dispatch_calls.append((session_id, adapter_type, project_id))
        return "runner-task-id"

    monkeypatch.setattr("huddleroom.workers.task_runner.dispatch_session", fake_dispatch_session)

    count = await SessionService().recover_orphaned_sessions(
        db_session,
        timeout_seconds=3600,
        redispatch_pending=False,
    )

    assert count == 0
    assert dispatch_calls == []
    await db_session.refresh(recent_pending_session)
    assert recent_pending_session.status == "pending"
    assert recent_pending_session.runner_task_id is None


def test_require_project_access_removed():
    """require_project_access must not exist — it was a misleading no-op."""
    import huddleroom.dependencies as deps
    assert not hasattr(deps, "require_project_access"), (
        "require_project_access is a no-op that creates a false security signal. "
        "It should be deleted, not kept as dead code."
    )


def test_cli_adapter_env_does_not_expose_secrets(monkeypatch):
    """_build_env must not pass Rally internals to subprocess."""
    from unittest.mock import MagicMock
    from huddleroom.adapters.cli_adapter import CliAdapter
    import uuid as _uuid

    monkeypatch.setenv("RALLY_JWT_SECRET", "super-secret-jwt")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host/db")
    monkeypatch.setenv("SECRET_KEY", "very-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-be-passed")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    adapter = CliAdapter()
    session_id = _uuid.uuid4()

    agent = MagicMock()
    agent.id = _uuid.uuid4()
    agent.config = {"cli_env_extras": {"RALLY_JWT_SECRET": "injected-via-config", "CUSTOM_VAR": "ok"}}

    project = MagicMock()
    project.id = _uuid.uuid4()

    task = MagicMock()
    task.id = _uuid.uuid4()

    env = adapter._build_env(session_id, task, agent, project)

    assert "RALLY_JWT_SECRET" not in env
    assert "DATABASE_URL" not in env
    assert "SECRET_KEY" not in env
    assert env.get("ANTHROPIC_API_KEY") == "sk-ant-should-be-passed"
    assert env["RALLY_SESSION_ID"] == str(session_id)
    assert env["RALLY_AGENT_ID"] == str(agent.id)
    assert env["RALLY_PROJECT_ID"] == str(project.id)
    assert "PATH" in env
    # cli_env_extras secrets must be stripped
    assert "RALLY_JWT_SECRET" not in env
    assert env.get("CUSTOM_VAR") == "ok"  # safe extras are still passed


@pytest.mark.asyncio
async def test_api_adapter_commits_started_state_before_llm_call(test_engine):
    from huddleroom.adapters.api_adapter import ApiAdapter
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as db:
        project = Project(name=f"Project-{uuid.uuid4()}", description="adapter test", config={})
        agent = Agent(
            name=f"agent-{uuid.uuid4()}",
            role="reviewer",
            provider="openai",
            model="gpt-4o-mini",
            adapter_type="api",
            capabilities=[],
            config={},
        )
        db.add_all([project, agent])
        await db.flush()
        session = Session(
            agent_id=agent.id,
            project_id=project.id,
            adapter_type="api",
            status="pending",
            metadata_={},
        )
        db.add(session)
        await db.commit()

        call_order: list[str] = []
        original_commit = db.commit

        async def tracked_commit():
            call_order.append("commit")
            await original_commit()

        db.commit = AsyncMock(side_effect=tracked_commit)

        adapter = ApiAdapter()

        async def fake_build_messages(_agent, _task, _db):
            return ([{"role": "user", "content": "review this"}], "task context", 0, 0)

        async def fake_run_with_retry(_agent, _session, _messages, _task_context, _db):
            call_order.append("run_with_retry")

        with patch.object(adapter, "_build_messages", new=fake_build_messages):
            with patch.object(adapter, "_run_with_retry", new=fake_run_with_retry):
                await adapter.run(session.id, db)

        assert call_order == ["commit", "run_with_retry"]


@pytest.mark.parametrize(
    "provider,model,model_override,env_api_base,expected_model,expected_api_base",
    [
        # Default Ollama (no env override)
        ("ollama", "gemma3:27b", None, None, "ollama/gemma3:27b", "http://localhost:11434"),
        # Ollama with env override
        ("ollama", "gemma3:27b", None, "http://host.docker.internal:11434", "ollama/gemma3:27b", "http://host.docker.internal:11434"),
        # Ollama with model override (skips api_base)
        ("ollama", "gemma3:27b", "openai/gpt-4o-mini", None, "openai/gpt-4o-mini", None),
        # OpenRouter with model override (adds provider prefix)
        ("openrouter", "openai/gpt-4o-mini", "nvidia/nemotron-3-super-120b-a12b:free", None, "openrouter/nvidia/nemotron-3-super-120b-a12b:free", None),
    ],
    ids=[
        "ollama_defaults_api_base",
        "ollama_uses_env_api_base",
        "ollama_model_override_skips_api_base",
        "openrouter_model_override_adds_provider",
    ],
)
@pytest.mark.asyncio
async def test_api_adapter_model_and_api_base_resolution(
    monkeypatch, provider, model, model_override, env_api_base, expected_model, expected_api_base
):
    from huddleroom.adapters.api_adapter import ApiAdapter

    if env_api_base:
        monkeypatch.setenv("OLLAMA_API_BASE", env_api_base)
    else:
        monkeypatch.delenv("OLLAMA_API_BASE", raising=False)

    agent = Agent(
        id=uuid.uuid4(),
        name=f"agent-{uuid.uuid4()}",
        role="developer",
        provider=provider,
        model=model,
        adapter_type="api",
        capabilities=[],
        config={},
    )

    metadata = {}
    if model_override:
        metadata["_run_config"] = {"model_override": model_override}

    session = Session(
        id=uuid.uuid4(),
        agent_id=agent.id,
        project_id=uuid.uuid4(),
        adapter_type="api",
        status="running",
        metadata_=metadata,
    )
    captured: dict = {}

    async def fake_run_tool_loop(**kwargs):
        captured.update(kwargs)
        return "done"

    with patch("huddleroom.adapters.api_adapter.run_tool_loop", new=fake_run_tool_loop):
        with patch("huddleroom.adapters.api_adapter.emit_event", new=AsyncMock()):
            with patch("huddleroom.services.session_sync.sync_task_from_session", new=AsyncMock()):
                await ApiAdapter()._run_with_retry(
                    agent,
                    session,
                    [{"role": "user", "content": "hello"}],
                    "",
                    AsyncMock(spec=AsyncSession),
                )

    assert captured["model"] == expected_model
    if expected_api_base is not None:
        assert captured["api_base"] == expected_api_base
    else:
        assert "api_base" not in captured


@pytest.mark.asyncio
async def test_api_delayed_retry_refreshes_agent_config(test_engine, monkeypatch):
    """A retry must use the agent configuration saved during its backoff."""
    import litellm
    from huddleroom.adapters.api_adapter import ApiAdapter

    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as db:
        project = Project(name=f"Project-{uuid.uuid4()}", description="adapter test", config={})
        agent = Agent(
            name=f"agent-{uuid.uuid4()}", role="developer", provider="openai", model="original-model",
            adapter_type="api", capabilities=[], config={"temperature": 0.7},
        )
        db.add_all([project, agent])
        await db.flush()
        session = Session(
            agent_id=agent.id, project_id=project.id, adapter_type="api", status="running",
            metadata_={"_run_config": {"model_override": "override"}, "keep": "value"},
        )
        db.add(session)
        await db.commit()

        calls: list[dict] = []

        async def fake_run_tool_loop(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise litellm.RateLimitError("retry", "openai", "gpt-4o-mini")
            return "done"

        async def update_during_backoff(_delay):
            async with session_factory.begin() as update_db:
                changed = await update_db.get(Agent, agent.id)
                changed.provider = "changed-provider"
                changed.model = "changed-model"
                changed.config = {"temperature": 0.1, "provider_extras": {"x": "new"}}

        with patch("huddleroom.adapters.api_adapter.run_tool_loop", new=fake_run_tool_loop):
            with patch("huddleroom.adapters.api_adapter.asyncio.sleep", new=update_during_backoff):
                with patch("huddleroom.adapters.api_adapter.emit_event", new=AsyncMock()):
                    with patch("huddleroom.services.session_sync.sync_task_from_session", new=AsyncMock()):
                        await ApiAdapter()._run_with_retry(
                            agent, session, [{"role": "user", "content": "hello"}], "", db
                        )

        assert calls[0]["model"] == "openai/override"
        assert calls[1]["model"] == "changed-provider/override"
        assert calls[1]["temperature"] == 0.1
        assert calls[1]["x"] == "new"
        assert session.metadata_["_run_config"]["model_override"] == "override"
        assert session.metadata_["keep"] == "value"
        assert session.metadata_["model_used"] == "override"


@pytest.mark.asyncio
async def test_api_tool_fallback_refreshes_agent_config(test_engine):
    """A tools-free fallback must resolve settings after a tool failure."""
    from huddleroom.adapters.api_adapter import ApiAdapter

    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as db:
        project = Project(name=f"Project-{uuid.uuid4()}", description="adapter test", config={})
        agent = Agent(
            name=f"agent-{uuid.uuid4()}", role="developer", provider="openai", model="original-model",
            adapter_type="api", capabilities=[], config={"memory_enabled": True, "temperature": 0.7},
        )
        db.add_all([project, agent])
        await db.flush()
        session = Session(
            agent_id=agent.id, project_id=project.id, adapter_type="api", status="running", metadata_={},
        )
        db.add(session)
        await db.commit()

        calls: list[dict] = []

        async def fake_run_tool_loop(**kwargs):
            calls.append(kwargs)
            if kwargs["tools"] is not None:
                async with session_factory.begin() as update_db:
                    changed = await update_db.get(Agent, agent.id)
                    changed.provider = "changed-provider"
                    changed.model = "changed-model"
                    changed.config = {"temperature": 0.1, "provider_extras": {"x": "new"}}
                raise RuntimeError("tools unsupported")
            return "done"

        with patch("huddleroom.adapters.api_adapter.run_tool_loop", new=fake_run_tool_loop):
            with patch("huddleroom.adapters.api_adapter.emit_event", new=AsyncMock()):
                with patch("huddleroom.services.session_sync.sync_task_from_session", new=AsyncMock()):
                    await ApiAdapter()._run_with_retry(
                        agent, session, [{"role": "user", "content": "hello"}], "", db
                    )

        assert calls[1]["tools"] is None
        assert calls[1]["model"] == "changed-provider/changed-model"
        assert calls[1]["temperature"] == 0.1
        assert calls[1]["x"] == "new"


@pytest.mark.asyncio
async def test_api_tool_loop_completion_refreshes_agent_config(test_engine):
    """Each completion in a tool loop must use the latest agent settings."""
    from types import SimpleNamespace

    from huddleroom.adapters.api_adapter import ApiAdapter

    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as db:
        project = Project(name=f"Project-{uuid.uuid4()}", description="adapter test", config={})
        agent = Agent(
            name=f"agent-{uuid.uuid4()}", role="developer", provider="ollama", model="original-model",
            adapter_type="api", capabilities=[],
            config={"memory_enabled": True, "temperature": 0.7, "provider_extras": {"old": "value"}},
        )
        db.add_all([project, agent])
        await db.flush()
        session = Session(
            agent_id=agent.id, project_id=project.id, adapter_type="api", status="running",
            metadata_={"_run_config": {"model_override": "override"}, "keep": "value"},
        )
        db.add(session)
        await db.commit()

        calls: list[dict] = []

        async def fake_completion(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                async with session_factory.begin() as update_db:
                    changed = await update_db.get(Agent, agent.id)
                    changed.provider = "changed-provider"
                    changed.model = "changed-model"
                    changed.config = {"temperature": 0.1, "provider_extras": {"x": "new"}}
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(
                        id="tool-1",
                        function=SimpleNamespace(name="memory_search", arguments="{}"),
                    )],
                    model_dump=lambda: {"role": "assistant", "content": None},
                ))])
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content="done", tool_calls=None,
            ))])

        async def fake_execute_memory_tool(**_kwargs):
            return "{}"

        with patch("litellm.acompletion", new=fake_completion):
            with patch("huddleroom.services.tool_executor.execute_memory_tool", new=fake_execute_memory_tool):
                with patch("huddleroom.adapters.api_adapter.emit_event", new=AsyncMock()):
                    with patch("huddleroom.services.session_sync.sync_task_from_session", new=AsyncMock()):
                        await ApiAdapter()._run_with_retry(
                            agent, session, [{"role": "user", "content": "hello"}], "", db
                        )

        assert calls[0]["model"] == "ollama/override"
        assert calls[0]["api_base"] == "http://localhost:11434"
        assert calls[0]["old"] == "value"
        assert calls[1]["model"] == "changed-provider/override"
        assert calls[1]["temperature"] == 0.1
        assert calls[1]["x"] == "new"
        assert "api_base" not in calls[1]
        assert "old" not in calls[1]
        assert session.metadata_["_run_config"]["model_override"] == "override"
        assert session.metadata_["keep"] == "value"
        assert session.metadata_["model_used"] == "override"


@pytest.mark.asyncio
async def test_cli_adapter_commits_started_state_before_waiting_on_subprocess(
    test_engine, tmp_path
):
    from huddleroom.adapters.cli_adapter import CliAdapter
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as db:
        project = Project(
            name=f"Project-{uuid.uuid4()}", description="adapter test",
            workspace_path=str(tmp_path), config={},
        )
        agent = Agent(
            name=f"agent-{uuid.uuid4()}",
            role="reviewer",
            provider="openai",
            model="gpt-4o-mini",
            adapter_type="cli",
            capabilities=[],
            config={},
        )
        db.add_all([project, agent])
        await db.flush()
        session = Session(
            agent_id=agent.id,
            project_id=project.id,
            adapter_type="cli",
            status="pending",
            metadata_={},
        )
        db.add(session)
        await db.commit()

        call_order: list[str] = []
        original_commit = db.commit

        async def tracked_commit():
            call_order.append("commit")
            await original_commit()

        db.commit = AsyncMock(side_effect=tracked_commit)

        class FakeProc:
            returncode = 0

            async def communicate(self):
                call_order.append("communicate")
                return b'{\"result\":\"APPROVE\",\"session_id\":\"provider-12345678\"}', b""

        adapter = CliAdapter()
        exchange = Mock()

        with patch("huddleroom.adapters.cli_adapter.log_cli_exchange", exchange):
            with patch.object(adapter, "_setup_sandbox", return_value=None):
                with patch.object(adapter, "_build_env", return_value={}):
                    with patch.object(adapter, "_build_command", return_value=["claude", "--print"]):
                        with patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", new=AsyncMock(return_value=FakeProc())):
                            await adapter.run(session.id, db)

        assert call_order == ["commit", "commit", "communicate"]
        exchange.assert_called_once_with(
            prompt="",
            session_id="provider-12345678",
            response="APPROVE",
            runtime="claude_code",
            rally_session_id=str(session.id),
        )


@pytest.mark.asyncio
async def test_knowledge_search_excludes_superseded(db_session: AsyncSession, test_project):
    """Superseded knowledge items must not appear in search results."""
    from unittest.mock import AsyncMock, patch

    svc = KnowledgeService()
    fake_embedding = [0.1] * 1536

    with patch('huddleroom.services.knowledge_service.embedding_service') as mock_emb:
        mock_emb.generate_embedding = AsyncMock(return_value=fake_embedding)

        active_item = await svc.create(
            db_session,
            test_project.id,
            KnowledgeCreate(title="Active", content="python async patterns", content_type="text"),
        )

        superseded_item = await svc.create(
            db_session,
            test_project.id,
            KnowledgeCreate(title="Old", content="python async patterns outdated", content_type="text"),
        )
        superseded_item.is_superseded = True
        await db_session.flush()

        results = await svc.search(
            db_session,
            test_project.id,
            query="python async patterns",
            min_relevance_score=0.0,
        )

    result_ids = [r.id for r in results]
    assert active_item.id in result_ids
    assert superseded_item.id not in result_ids, "Superseded items must be excluded from search"


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_get_caller_raises_on_unexpected_jwt_error(db_session: AsyncSession):
    """get_caller must not silently swallow unexpected errors from JWT decode."""
    from unittest.mock import patch
    from starlette.requests import Request
    from huddleroom.dependencies import get_caller
    from huddleroom.config import Settings

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(b"authorization", b"Bearer not-an-api-key-prefix-xyzzy")],
        "query_string": b"",
    }
    request = Request(scope)

    with patch("huddleroom.dependencies.settings", Settings(auth_enabled=True)):
        with patch("huddleroom.dependencies.decode_access_token", side_effect=ValueError("unexpected")):
            with pytest.raises(ValueError, match="unexpected"):
                await get_caller(request, db_session)


@pytest.mark.asyncio
async def test_agent_update_is_active_false(db_session: AsyncSession, test_project):
    """AgentService.update must be able to set is_active=False."""
    svc = AgentService()
    agent = await svc.create(db_session, _agent_create_data(name="DeactivateMe"))
    assert agent.is_active is True

    updated = await svc.update(db_session, agent.id, AgentUpdate(is_active=False))
    assert updated.is_active is False, "is_active=False must be applied, not silently ignored"


@pytest.mark.asyncio
async def test_list_agents_limit_validation(client, auth_headers):
    """limit=0 and limit=99999 must return 422."""
    resp = await client.get("/api/v1/agents?limit=0", headers=auth_headers)
    assert resp.status_code == 422, f"limit=0 should be 422, got {resp.status_code}"
    resp2 = await client.get("/api/v1/agents?limit=99999", headers=auth_headers)
    assert resp2.status_code == 422, f"limit=99999 should be 422, got {resp2.status_code}"


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_delete_api_key_bad_uuid_returns_422(client, auth_headers):
    """DELETE /api/v1/auth/api-keys/{key_id} with bad UUID must return 422, not 500."""
    resp = await client.delete("/api/v1/auth/api-keys/not-a-uuid", headers=auth_headers)
    assert resp.status_code == 422, f"Expected 422 on bad UUID, got {resp.status_code}"


@pytest.mark.asyncio
async def test_pagination_no_skip_on_same_timestamp(db_session: AsyncSession, test_project):
    """Cursor pagination must not skip records when created_at values are identical."""
    from datetime import datetime, timezone
    from huddleroom.services.task_service import TaskService

    svc = TaskService()
    same_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    created_ids = []
    for i in range(3):
        t = await svc.create(db_session, test_project.id, TaskCreate(title=f"SameTs Task {i}"))
        t.created_at = same_time
        created_ids.append(t.id)
    await db_session.flush()

    page1, cursor = await svc.list(db_session, test_project.id, limit=2)
    assert cursor is not None, "Should have a next cursor"
    assert len(page1) == 2

    page2, _ = await svc.list(db_session, test_project.id, cursor=cursor, limit=2)
    assert len(page2) >= 1, "Must not skip the third task with identical timestamp"

    all_ids = {t.id for t in page1} | {t.id for t in page2}
    for task_id in created_ids:
        assert task_id in all_ids, f"Task {task_id} was skipped by cursor pagination"


def test_knowledge_search_request_has_no_tags_field():
    """KnowledgeSearchRequest must not have a 'tags' field that is silently ignored."""
    from huddleroom.schemas.knowledge import KnowledgeSearchRequest
    req = KnowledgeSearchRequest(query="test")
    assert not hasattr(req, "tags"), (
        "KnowledgeSearchRequest.tags is silently ignored by the search service. "
        "Remove it until tag filtering is implemented."
    )
