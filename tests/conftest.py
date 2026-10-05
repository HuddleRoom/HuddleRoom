import asyncio
import os
import tempfile
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import pytest
import pytest_asyncio
from fastapi import Request
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# Keep collection independent of a developer's ~/.huddleroom/config.toml.
os.environ["HOME"] = tempfile.mkdtemp(prefix="huddleroom-test-home-")

from huddleroom.config import settings
from huddleroom.database import get_db

# Prevent pytest from writing to the production log file.
# Set log_file to a per-pid temp path before any code path calls
# register_litellm_debug_logger() or configure_debug_file_logging().
settings.log_file = os.path.join(tempfile.gettempdir(), f"rally-test-debug-{os.getpid()}.log")
from huddleroom.dependencies import get_current_user, _ANON_USER, _ANON_AGENT
from huddleroom.main import create_app
from huddleroom.models.base import Base
from huddleroom.models.agent import Agent
from huddleroom.models.project import Project
from huddleroom.models.user import User
from huddleroom.security import hash_password, create_access_token
import huddleroom.models  # noqa: F401 — ensure all models are imported

# Test DB URL: SQLite gets a separate file; PostgreSQL replaces the db name
_db_url = settings.database_url
if _db_url.startswith("sqlite"):
    _test_db_dir = tempfile.gettempdir()
    _test_db_path = os.path.join(_test_db_dir, f"rally_test_{os.getpid()}.db")
    TEST_DATABASE_URL = f"sqlite+aiosqlite:///{_test_db_path}"
else:
    base_url, _ = _db_url.rsplit("/", 1)
    TEST_DATABASE_URL = f"{base_url}/rally_test"

@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def safe_goal_analysis(monkeypatch):
    """Keep downstream process tests independent of the external analyzer."""
    from huddleroom.services.orchestration_goal_analyzer import (
        GoalAnalysis,
        GoalClarificationAnalyzer,
    )

    async def analyze(self, request, *, project_id=None):
        return GoalAnalysis((), (), False)

    monkeypatch.setattr(GoalClarificationAnalyzer, "analyze_request", analyze)


@pytest.fixture
def safe_agent_definition_review(monkeypatch):
    """Keep tick()'s internal agent_definition_review pass (which constructs
    AgentDefinitionReviewProcess() with its default analyzer) independent of
    the external semantic analyzer (litellm/OpenRouter)."""
    from huddleroom.services.orchestration_agent_definition_analyzer import (
        AgentDefinitionSemanticAnalyzer,
        SemanticAgentAssessment,
    )

    async def review(self, agent_snapshot, goal_snapshot, candidate_work_functions, *, project_id=None):
        return SemanticAgentAssessment("approved", (), "test fixture", tuple(candidate_work_functions))

    async def review_request(self, request, candidate_work_functions, *, project_id=None):
        return SemanticAgentAssessment("approved", (), "test fixture", tuple(candidate_work_functions))

    monkeypatch.setattr(AgentDefinitionSemanticAnalyzer, "review", review)
    monkeypatch.setattr(AgentDefinitionSemanticAnalyzer, "review_request", review_request)


@pytest.fixture
def safe_effectiveness_review(monkeypatch):
    """Keep tick()'s/debug-service's effectiveness_review pass (which constructs
    EffectivenessReviewProcess() with its default analyzer) independent of the
    external analyzer (litellm/OpenRouter). Non-"continue" disposition so the
    stepped process parks behind a decision (waiting_decision) like the
    happy-path tests expect, rather than completing immediately."""
    from huddleroom.services.orchestration_effectiveness_analyzer import (
        EffectivenessAnalysis,
        EffectivenessAnalyzer,
    )

    async def review_request(self, request, *, project_id=None):
        return EffectivenessAnalysis("pause", (), "test fixture")

    monkeypatch.setattr(EffectivenessAnalyzer, "review_request", review_request)


@pytest.fixture
def safe_effectiveness_review_continue(monkeypatch):
    """Like safe_effectiveness_review, but "continue" disposition so the
    stepped process completes immediately (process "completed") instead of
    parking in waiting_decision — both of which gate can_finish, but tests
    that drive a goal all the way to goal_closeout need the non-blocking
    terminal outcome."""
    from huddleroom.services.orchestration_effectiveness_analyzer import (
        EffectivenessAnalysis,
        EffectivenessAnalyzer,
    )

    async def review_request(self, request, *, project_id=None):
        return EffectivenessAnalysis("continue", (), "test fixture")

    monkeypatch.setattr(EffectivenessAnalyzer, "review_request", review_request)


async def _seed_anon_actors(conn) -> None:
    """Seed the anon user/agent rows that huddleroom.dependencies falls back to
    when auth_enabled=False (the test default). Under FK enforcement, any
    row that records this actor as created_by_user_id / producer_agent_id
    etc. needs a real users/agents row to point at — previously these ids
    were never persisted anywhere. Called both at initial schema setup and
    after clean_test_database's per-test full-table wipe, which would
    otherwise delete these rows along with everything else.
    """
    seed_session = async_sessionmaker(bind=conn, expire_on_commit=False)()
    seed_session.add(User(
        id=_ANON_USER.id, email=_ANON_USER.email, hashed_password="",
        is_active=True, role="admin",
    ))
    seed_session.add(Agent(
        # is_active=False: this row exists only to satisfy FK columns that
        # reference the anon actor (e.g. producer_agent_id); tests that
        # assert "no active agents exist" as a precondition would otherwise
        # see this placeholder and miscount.
        id=_ANON_AGENT.id, name=_ANON_AGENT.name, role="agent",
        provider="local", model="local", adapter_type="api",
        capabilities=[], config={}, is_active=False,
    ))
    await seed_session.flush()
    await seed_session.close()


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    """Session-scoped engine that creates/drops all tables once."""
    if TEST_DATABASE_URL.startswith("sqlite"):
        sqlite_path = TEST_DATABASE_URL.removeprefix("sqlite+aiosqlite:///")
        for suffix in ("", "-shm", "-wal"):
            try:
                os.remove(f"{sqlite_path}{suffix}")
            except FileNotFoundError:
                pass

    engine_kwargs: dict = {"echo": False}
    if TEST_DATABASE_URL.startswith("sqlite"):
        engine_kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    engine = create_async_engine(TEST_DATABASE_URL, **engine_kwargs)
    if TEST_DATABASE_URL.startswith("sqlite"):
        from sqlalchemy import event as _sa_event

        @_sa_event.listens_for(engine.sync_engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    async with engine.begin() as conn:
        dialect = engine.dialect.name
        if dialect == "postgresql":
            await conn.execute(__import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        await _seed_anon_actors(conn)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()
    if TEST_DATABASE_URL.startswith("sqlite"):
        sqlite_path = TEST_DATABASE_URL.removeprefix("sqlite+aiosqlite:///")
        for suffix in ("", "-shm", "-wal"):
            try:
                os.remove(f"{sqlite_path}{suffix}")
            except FileNotFoundError:
                pass


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Per-test transaction that rolls back after each test."""
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        async with session.begin():
            yield session
            await session.rollback()


@pytest_asyncio.fixture(autouse=True)
async def clean_test_database(test_engine) -> AsyncGenerator[None, None]:
    yield
    async with test_engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            await conn.execute(table.delete())
        await _seed_anon_actors(conn)


@pytest_asyncio.fixture
async def concurrent_sessions(test_engine) -> AsyncGenerator[tuple[AsyncSession, AsyncSession], None]:
    """Two independent sessions for testing concurrent access.

    Unlike db_session, these are not wrapped in a transaction, allowing true
    concurrent access to the same DB file (for WAL mode testing).
    """
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    session1 = session_factory()
    session2 = session_factory()
    try:
        yield session1, session2
    finally:
        await session1.close()
        await session2.close()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Test HTTP client with DB override."""
    from sqlalchemy import select
    from huddleroom.security import decode_access_token

    app = create_app()

    async def override_get_db():
        yield db_session

    async def override_get_current_user(request: Request):
        from fastapi import HTTPException, status
        import huddleroom.dependencies as dep_module

        authorization = request.headers.get("Authorization", "")

        # If auth is enabled and no token provided, raise 401
        if dep_module.settings.auth_enabled and not authorization.startswith("Bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

        # If a Bearer token is provided, decode it and fetch the user
        if authorization.startswith("Bearer "):
            token = authorization[7:]
            try:
                payload = decode_access_token(token)
                user_id_str = payload.get("sub")
                if user_id_str:
                    user_id = uuid.UUID(user_id_str)
                    result = await db_session.execute(select(User).where(User.id == user_id))
                    user = result.scalar_one_or_none()
                    if user:
                        return user
            except Exception:
                if dep_module.settings.auth_enabled:
                    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

        # Default to anonymous admin user (for backward compatibility with existing tests when auth is disabled)
        return dep_module._ANON_USER

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession) -> User:
    user = User(
        email=f"test-{uuid.uuid4()}@example.com",
        hashed_password=hash_password("testpassword"),
        display_name="Test User",
        role="member",
    )
    db_session.add(user)
    await db_session.flush()
    return user


@pytest_asyncio.fixture
async def auth_headers(test_user: User) -> dict:
    token = create_access_token({"sub": str(test_user.id)})
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def test_project(db_session: AsyncSession, tmp_path) -> Project:
    workspace = tmp_path / "project-workspace"
    workspace.mkdir()
    project = Project(
        name="Test Project",
        description="A test project",
        workspace_path=str(workspace.resolve()),
        config={},
    )
    db_session.add(project)
    await db_session.flush()
    return project


@pytest_asyncio.fixture
async def legacy_project(db_session: AsyncSession) -> Project:
    project = Project(name="Legacy Project", description="No workspace configured", config={})
    db_session.add(project)
    await db_session.flush()
    return project


@pytest_asyncio.fixture
async def runnable_project(test_project: Project) -> Project:
    return test_project


@pytest_asyncio.fixture
async def conversation_goal_run(db_session: AsyncSession, test_project: Project, test_user: User):
    from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun

    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Conversation fixture goal",
        success_criteria=[],
        constraints={},
        budget={},
        created_by_user_id=test_user.id,
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(
        goal_id=goal.id,
        event_cursor=None,
        plan_state={},
        active_blockers=[],
        budget_state={},
        retry_state={},
    )
    db_session.add(run)
    await db_session.flush()
    return goal, run


@pytest_asyncio.fixture
async def test_agent(db_session: AsyncSession) -> Agent:
    agent = Agent(
        name=f"test-agent-{uuid.uuid4()}",
        role="developer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=[],
        config={},
    )
    db_session.add(agent)
    await db_session.flush()
    return agent


@pytest_asyncio.fixture
async def memory_agent(db_session: AsyncSession) -> Agent:
    """Agent with memory enabled."""
    agent = Agent(
        name=f"memory-agent-{uuid.uuid4()}",
        role="developer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=[],
        config={"memory_enabled": True},
    )
    db_session.add(agent)
    await db_session.flush()
    return agent


async def complete_baseline_processes(db_session, goal, run):
    """Complete all baseline processes so tests can exercise their own subject."""
    import uuid as _uuid

    from huddleroom.models.user import User
    from huddleroom.security import hash_password
    from huddleroom.services.orchestration_authority_service import (
        OrchestrationAuthorityDecisionService,
    )
    from huddleroom.services.orchestration_goal_analyzer import GoalAnalysis
    from huddleroom.services.orchestration_goal_definition import GoalDefinitionProcess
    from huddleroom.services.orchestration_manager_selection import (
        HUMAN_AS_MANAGER_OPTION,
        REVIEW_OVERRIDE_DECISION_KEY,
        SELECT_MANAGER_DECISION_KEY,
        ManagerSelectionProcess,
    )
    from huddleroom.services.orchestration_manager_analyzer import ManagerAssessment
    from huddleroom.services.orchestration_agent_definition_review import (
        AgentDefinitionReviewProcess,
    )
    from huddleroom.services.orchestration_agent_definition_analyzer import (
        SemanticAgentAssessment,
    )
    from huddleroom.services.orchestration_team_hierarchy import TeamHierarchyProcess
    from huddleroom.services.orchestration_team_hierarchy_analyzer import TeamHierarchyAnalysis

    class SafeGoalAnalyzer:
        async def analyze(self, goal_snapshot, *, project_id=None):
            return GoalAnalysis((), (), False)

        def build_request(self, goal_snapshot, project=None):
            return {"goal_snapshot": goal_snapshot}

        async def analyze_request(self, request, *, project_id=None):
            return await self.analyze(request["goal_snapshot"], project_id=project_id)

    class SafeAgentDefinitionAnalyzer:
        async def review(self, agent_snapshot, goal_snapshot, candidate_work_functions, project=None, *, project_id=None):
            return SemanticAgentAssessment("approved", (), "test fixture", tuple(candidate_work_functions))

        def build_request(self, agent_snapshot, goal_snapshot, candidate_work_functions, project=None):
            return {
                "agent": agent_snapshot,
                "goal": goal_snapshot,
                "candidate_work_functions": candidate_work_functions,
            }

        async def review_request(self, request, candidate_work_functions, *, project_id=None):
            return await self.review(
                request["agent"], request["goal"], candidate_work_functions, project_id=project_id
            )

    class SafeManagerSelectionAnalyzer:
        async def review(self, payload, project=None, *, project_id=None):
            return ManagerAssessment(
                "override",
                HUMAN_AS_MANAGER_OPTION,
                "test fixture",
            )

        async def review_request(self, request, *, project_id=None):
            return await self.review({}, project_id=project_id)

    class SafeTeamHierarchyAnalyzer:
        async def review(self, payload, project=None, *, project_id=None):
            producer_refs: set[str] = set()
            reviewer_refs: set[str] = set()
            assignments = []
            for work_function in payload["required_work_functions"]:
                verifier = work_function in {"review", "validation"}
                candidates = (
                    agent for agent in payload["agents"]
                    if work_function in agent["capabilities"]
                    and agent["id"] not in (producer_refs if verifier else reviewer_refs)
                )
                if agent := next(candidates, None):
                    assignments.append({"work_function": work_function, "agent_ref": agent["id"]})
                    (reviewer_refs if verifier else producer_refs).add(agent["id"])
            assigned_refs = dict.fromkeys(item["agent_ref"] for item in assignments)
            assigned_functions = {item["work_function"] for item in assignments}
            return TeamHierarchyAnalysis(
                proposed_agents=(),
                assignments=tuple(assignments),
                reporting_lines=tuple(
                    {"agent_ref": agent_ref, "reports_to": "manager"}
                    for agent_ref in assigned_refs
                ),
                documented_gaps=tuple(
                    work_function
                    for work_function in payload["required_work_functions"]
                    if work_function not in assigned_functions
                ),
                rationale="Maps reviewed roster capabilities without proposing agents.",
                self_review="Checked exact coverage, reporting lines, and verifier independence.",
            )

        async def review_request(self, request, *, project_id=None):
            return await self.review(request, project_id=project_id)

    decision_svc = OrchestrationAuthorityDecisionService()
    answering_user_id = goal.created_by_user_id
    if answering_user_id is None:
        user = User(
            email=f"baseline-fixture-{_uuid.uuid4()}@example.com",
            hashed_password=hash_password("testpassword"),
            display_name="Baseline Fixture User",
            role="member",
        )
        db_session.add(user)
        await db_session.flush()
        answering_user_id = user.id
        goal.created_by_user_id = user.id
    for process, prefix in (
        (GoalDefinitionProcess(SafeGoalAnalyzer()), "goal_definition:"),
        (ManagerSelectionProcess(SafeManagerSelectionAnalyzer()), "manager_selection:"),
    ):
        summary = await process.advance(db_session, goal, run)
        for _retry in range(10):
            if summary["status"] != "waiting_decision":
                break
            pending = await decision_svc.list_decisions(db_session, goal.id, status="pending")
            found_matching = False
            for decision in pending:
                if not decision.decision_key.startswith(prefix):
                    continue
                found_matching = True
                if decision.decision_key == SELECT_MANAGER_DECISION_KEY:
                    option = HUMAN_AS_MANAGER_OPTION
                elif decision.decision_key == REVIEW_OVERRIDE_DECISION_KEY:
                    option = "approve"
                elif decision.options:
                    option = "approved"
                else:
                    option = "fixture default"
                await decision_svc.answer_decision(
                    db_session, decision,
                    selected_option=option,
                    reason="test fixture auto-answer",
                    decided_by_user_id=answering_user_id,
                )
            summary = await process.advance(db_session, goal, run)
            if summary["status"] != "waiting_decision":
                break
            if not found_matching:
                raise AssertionError(
                    f"Process {prefix.rstrip(':')} status is waiting_decision but no matching pending decisions found"
                )
        else:
            raise AssertionError(
                f"Baseline process {prefix.rstrip(':')} did not converge within 10 iterations"
            )
    agent_review = await AgentDefinitionReviewProcess(SafeAgentDefinitionAnalyzer()).advance(
        db_session, goal, run
    )
    assert agent_review["status"] in {"completed", "skipped"}
    hierarchy = TeamHierarchyProcess(SafeTeamHierarchyAnalyzer())
    summary = await hierarchy.advance(db_session, goal, run)
    for _retry in range(10):
        if summary["status"] != "waiting_decision":
            break
        pending = await decision_svc.list_decisions(db_session, goal.id, status="pending")
        matching = [
            decision
            for decision in pending
            if decision.decision_key == "team_hierarchy:approval"
        ]
        if not matching:
            raise AssertionError(
                "Process team_hierarchy is waiting_decision but no matching pending decision was found"
            )
        for decision in matching:
            option = next(
                item["key"]
                for item in decision.options
                if item["key"] in {"approve", "approve_with_documented_gaps"}
            )
            principal = (
                {"decided_by_user_id": answering_user_id}
                if decision.authority == "human"
                else {"decided_by_agent_id": decision.authority_agent_id}
            )
            await decision_svc.answer_decision(
                db_session,
                decision,
                selected_option=option,
                reason="test fixture auto-answer",
                **principal,
            )
        summary = await hierarchy.advance(db_session, goal, run)
        if summary["status"] != "waiting_decision":
            break
    else:
        raise AssertionError(
            "Baseline process team_hierarchy did not converge within 10 iterations"
        )
    await db_session.flush()


async def dismiss_stale_baseline_suggestions(db_session, goal_id):
    """Item 3 (fix-baseline-gating-stale-inputs-cleanup): a completed
    baseline process whose inputs drifted after acceptance now raises a
    one-time suggestion instead of silently auto-rerunning -- the
    stale-readiness gate (OrchestrationService._baseline_readiness_reason)
    legitimately 409s until a human approves a rerun or dismisses it.

    Tests that mutate goal state (weight, roster, ...) purely for unrelated
    downstream setup convenience -- not to exercise staleness itself -- can
    call this to fast-forward through that gate exactly like a human
    clicking Dismiss on every outstanding suggestion for the goal.
    """
    from huddleroom.services.orchestration_warning_service import (
        STALE_INPUTS_DISMISSED_REASON,
        OrchestrationWarningService,
    )

    warning_service = OrchestrationWarningService()
    for warning in await warning_service.list_warnings(db_session, goal_id, active_only=True):
        if warning.warning_type.endswith("_stale_inputs"):
            await warning_service.resolve_warning(
                db_session, warning, resolved_by="human:test-fixture",
                reason=STALE_INPUTS_DISMISSED_REASON,
            )


async def heal_baseline_drift_for_test(db_session, goal, run):
    """Fully re-derive the baseline (not just silence the gate) after a test
    mutates goal/roster state (e.g. `goal.weight = "trivial"`) purely for
    unrelated downstream setup, not to exercise item-3 staleness itself.

    Item 3 converts completed-baseline-process staleness into a one-time
    suggestion instead of silently auto-rerunning; that's correct for real
    usage (a human must approve), but a test that just wants "baseline
    reflects current goal state" needs the equivalent of a human clicking
    Approve on every outstanding suggestion. Superseding each stale
    completed row (mirroring the real rerun endpoint) and re-running
    complete_baseline_processes -- which drives every process through with
    deterministic Safe* analyzers and answers any resulting decisions --
    gets a fully-consistent baseline, not just a readiness-gate bypass
    (dismiss alone leaves stale content behind, e.g. team_hierarchy gates
    still shaped for the old weight tier).
    """
    from huddleroom.models.orchestration_process import OrchestrationWarning
    from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_team_hierarchy import TeamHierarchyProcess
    from sqlalchemy import select as _select

    process_service = OrchestrationProcessService()
    any_superseded = False
    # manager_selection deliberately excluded: its own stale-readiness check
    # (orchestration_service._baseline_readiness_reason) only covers a
    # removed/inactive manager, not weight-tier drift (unchanged since
    # before item 3) -- forcing a rerun here risks picking a different
    # manager than what earlier test assertions already pinned.
    for process_type, process in (
        ("agent_definition_review", AgentDefinitionReviewProcess()),
        ("team_hierarchy", TeamHierarchyProcess()),
    ):
        current = await process_service.get_current(db_session, goal.id, process_type)
        if current is None or current.status != "completed":
            continue
        # Only actually re-derive processes item 3 flagged as genuinely
        # stale (advance() raises the suggestion as a side effect without
        # mutating the row) -- unconditionally superseding an unchanged,
        # already-correct row on every call is wasteful and risks
        # non-deterministic diffs for content that never needed to change.
        # Fix A (post-baseline auto-heal) scopes the suggestion-only
        # behavior to `run.phase == "baseline"`; this probe must stay a
        # side-effect-free suggestion (except for PROCESS_VERSION migration re-stamping) regardless of the run's real phase
        # (many callers here are already past baseline), so it temporarily
        # presents as baseline-phase for the duration of the probe call --
        # never triggering the real (LLM-calling) auto-rerun this helper
        # explicitly avoids.
        original_phase = run.phase
        run.phase = "baseline"
        try:
            await process.advance(db_session, goal, run, manual=True)
        finally:
            run.phase = original_phase
        is_stale = (
            await db_session.execute(
                _select(OrchestrationWarning.id).where(
                    OrchestrationWarning.goal_id == goal.id,
                    OrchestrationWarning.warning_type == f"{process_type}_stale_inputs",
                    OrchestrationWarning.source_process_run_id == current.id,
                    OrchestrationWarning.active.is_(True),
                )
            )
        ).scalars().first() is not None
        if not is_stale:
            continue
        current.superseded_by_id = current.id
        any_superseded = True
    if not any_superseded:
        return
    await db_session.flush()
    await complete_baseline_processes(db_session, goal, run)
    await dismiss_stale_baseline_suggestions(db_session, goal.id)
    # A fresh agent_definition_review row raises its own findings as brand
    # new (unacknowledged) warnings, even when the underlying agent
    # definitions are unchanged -- acknowledge them so an unrelated
    # downstream closeout-completion check isn't newly blocked by content
    # that was already accepted (implicitly, by the test never touching it)
    # under the row this just superseded.
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    warning_service = OrchestrationWarningService()
    for warning in await warning_service.list_warnings(db_session, goal.id, active_only=True):
        if not warning.warning_type.endswith("_stale_inputs") and warning.acknowledged_by is None:
            await warning_service.acknowledge_warning(
                db_session, warning, acknowledged_by="human:test-fixture"
            )


@pytest.fixture
def stub_decision(monkeypatch):
    """Monkeypatches OrchestrationDecisionAdapter.decide to return a deterministic
    parsed_decision -- no real LLM. Returns a setter: call it with a callable
    `decision_fn(context) -> dict` (the parsed decision); tests can compute agent
    ids from the context's roster. input_snapshot mirrors the context so the
    service's dedup logic (input_snapshot == context) behaves correctly.
    """
    from copy import deepcopy

    from huddleroom.services.orchestration_llm_decision_adapter import (
        OrchestrationDecisionAdapter,
        OrchestrationDecisionAdapterResult,
    )

    def _install(decision_fn):
        async def fake_decide(self, context, *, project=None, goal=None):
            parsed_decision = decision_fn(context)
            return OrchestrationDecisionAdapterResult(
                input_snapshot=deepcopy(dict(context)),
                llm_output={"raw_content": None},
                parsed_decision=parsed_decision,
            )

        monkeypatch.setattr(OrchestrationDecisionAdapter, "decide", fake_decide)

    return _install


@pytest.fixture
def stream_monitors(monkeypatch):
    """Mock project reset monitors for invocation context tests."""
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    ack = SimpleNamespace(generation=None)
    monkeypatch.setattr(
        "huddleroom.services.agent_response_relay.register_project_reset_monitor",
        AsyncMock(return_value=ack),
    )
    monkeypatch.setattr(
        "huddleroom.services.agent_response_relay.unregister_project_reset_monitor",
        AsyncMock(),
    )

    async def _never_reset(_project_id):
        await asyncio.Event().wait()

    monkeypatch.setattr(
        "huddleroom.services.agent_response_stream.wait_for_project_reset", _never_reset
    )
