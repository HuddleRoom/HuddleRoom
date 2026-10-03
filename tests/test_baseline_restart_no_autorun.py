"""Scenario/regression test for the ORIGINAL bug: a fresh (unauthorized)
goal must never auto-run baseline via the scheduler's reconcile sweep, and
once a baseline step is complete, cosmetic input drift (provider/model/
config) must not re-trigger it while semantic drift (system_prompt) should
surface exactly one dismissable suggestion instead of auto-rerunning.

Exercises the real `reconcile_orchestration_runs_async` scheduler path
(not the process-level advance() calls the orchestration unit suites use)."""

import pytest
from sqlalchemy import select

pytestmark = pytest.mark.usefixtures("safe_goal_analysis")

from huddleroom.models.agent import Agent
from huddleroom.models.orchestration_process import OrchestrationProcessRun, OrchestrationWarning
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.services.orchestration_manager_analyzer import ManagerAssessment, ManagerSelectionAnalyzer
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService
from huddleroom.workers.scheduler import reconcile_orchestration_runs_async


@pytest.fixture(autouse=True)
def safe_manager_selection(monkeypatch):
    """Keep tick()'s manager_selection pass (which constructs
    ManagerSelectionProcess() with its default analyzer) independent of the
    external LLM -- mirrors the pattern in tests/conftest.py's
    safe_agent_definition_review, applied to manager selection."""

    async def review(self, payload, project=None, *, project_id=None):
        return ManagerAssessment("confirm", "no_manager", "test fixture")

    monkeypatch.setattr(ManagerSelectionAnalyzer, "review", review)


@pytest.fixture(autouse=True)
def approve_semantic_definitions_by_default(monkeypatch):
    """Keep tick()'s agent_definition_review pass independent of the
    external semantic analyzer -- same pattern used in
    test_orchestration_agent_definition_review.py."""
    from huddleroom.services.orchestration_agent_definition_analyzer import (
        AgentDefinitionSemanticAnalyzer,
        SemanticAgentAssessment,
    )

    async def review_request(self, request, candidate_work_functions, *, project_id=None):
        return SemanticAgentAssessment(
            "approved", (), "Definition is actionable.", tuple(candidate_work_functions)
        )

    monkeypatch.setattr(
        AgentDefinitionSemanticAnalyzer, "review_request", review_request
    )


@pytest.fixture(autouse=True)
def safe_team_hierarchy(monkeypatch):
    """Keep tick()'s team_hierarchy pass independent of the external LLM.
    Assigns the single test agent to every required work function so the
    process reaches a deterministic terminal state without a human decision
    -- same pattern used in test_orchestration_team_hierarchy.py."""
    from huddleroom.services.orchestration_team_hierarchy_analyzer import (
        TeamHierarchyAnalysis,
        TeamHierarchyAnalyzer,
    )

    async def review(self, payload, project=None, *, project_id=None):
        agent_id = payload["agents"][0]["id"]
        work_functions = payload["required_work_functions"]
        covered = "implementation" if "implementation" in work_functions else work_functions[0]
        return TeamHierarchyAnalysis(
            proposed_agents=(),
            assignments=({"work_function": covered, "agent_ref": agent_id},),
            reporting_lines=({"agent_ref": agent_id, "reports_to": "manager"},),
            documented_gaps=tuple(wf for wf in work_functions if wf != covered),
            rationale="test fixture",
            self_review="test fixture",
        )

    monkeypatch.setattr(TeamHierarchyAnalyzer, "review", review)


async def _process_rows(db_session):
    return (await db_session.execute(select(OrchestrationProcessRun))).scalars().all()


async def _create_goal(db_session, test_project):
    service = OrchestrationService()
    return await service.create_goal(
        db_session,
        test_project.id,
        OrchestrationGoalCreate(
            objective="Fix the widget",
            success_criteria=[{"key": "works", "description": "widget works", "evidence": "test"}],
            # Non-empty constraints keep classify_goal_weight() off "trivial"
            # (score==0), which would otherwise compress agent_definition_review
            # to a no-op with no targets -- the second test needs an actual
            # per-agent fingerprint to drift.
            constraints={"scope": "project"},
            budget={},
        ),
        created_by_user_id=None,
    )


@pytest.mark.asyncio
async def test_fresh_unauthorized_goal_reconcile_is_a_pure_noop(db_session, test_project, monkeypatch):
    """3a: reconcile must not create any process row or call the LLM analyzer
    for a goal whose baseline was never authorized."""
    from huddleroom.services.orchestration_goal_analyzer import GoalClarificationAnalyzer

    async def boom(self, request, *, project_id=None):
        raise AssertionError("goal analyzer must not be called for an unauthorized baseline")

    monkeypatch.setattr(GoalClarificationAnalyzer, "analyze_request", boom)

    goal, run = await _create_goal(db_session, test_project)
    assert run.baseline_authorized is False

    for _ in range(3):
        reconciled = await reconcile_orchestration_runs_async(db_session)
        assert reconciled == 1

    assert await _process_rows(db_session) == []
    await db_session.refresh(run)
    assert run.phase == "baseline"
    assert run.baseline_authorized is False


@pytest.mark.asyncio
async def test_authorized_goal_reconcile_advances_chain_then_stale_input_handling(
    db_session, test_project
):
    """3c then 3b: authorize -> reconcile advances the full baseline chain;
    then cosmetic drift is silent, semantic drift is a one-time dismissable
    suggestion (not an auto-rerun), and dismissal sticks."""
    goal, run = await _create_goal(db_session, test_project)

    agent = Agent(
        name="implementation-agent",
        role="developer",
        description="Implements scoped product changes.",
        system_prompt="Implement the requested change and verify it.",
        provider="example",
        model="unranked-model",
        adapter_type="api",
        config={"temperature": 0.2, "reasoning_effort": "medium", "tools": ["editor"]},
        capabilities=["implementation"],
        is_active=True,
    )
    db_session.add(agent)
    await db_session.flush()
    db_session.add(Task(
        project_id=test_project.id,
        title="do the work",
        assigned_to=agent.id,
        metadata_={"orchestration": {"run_id": str(run.id), "work_function": "implementation"}},
    ))
    await db_session.flush()

    # --- 3c: authorize, then reconcile advances the chain ---
    goal, run = await OrchestrationService().authorize_baseline(
        db_session, test_project.id, goal.id, actor="human:test"
    )
    assert run.baseline_authorized is True

    reconciled = await reconcile_orchestration_runs_async(db_session)
    assert reconciled == 1
    await db_session.refresh(run)

    rows_by_type = {row.process_type: row for row in await _process_rows(db_session)}
    assert rows_by_type["goal_definition"].status == "completed"
    assert rows_by_type["manager_selection"].status == "completed"
    review = rows_by_type["agent_definition_review"]
    assert review.status == "completed"
    # team_hierarchy has a documented gap (3 of the 4 required work functions
    # have no agent) so it parks waiting for a human decision -- baseline
    # isn't terminal yet, so phase correctly stays "baseline". This is
    # exactly the state 3b targets: "goal with completed
    # agent_definition_review in phase baseline".
    assert rows_by_type["team_hierarchy"].status == "waiting_decision"
    assert run.phase == "baseline"
    review_process_run_id = review.id

    from huddleroom.services.orchestration_authority_service import (
        OrchestrationAuthorityDecisionService,
    )

    pending_before = await OrchestrationAuthorityDecisionService().list_decisions(
        db_session, goal.id, status="pending"
    )
    assert pending_before  # team_hierarchy's documented-gap decision is parked

    # --- 3b: cosmetic drift (provider/model/config) must not re-trigger ---
    agent.provider = "other-provider"
    agent.model = "other-model"
    agent.config = {"temperature": 0.9}
    await db_session.flush()

    await reconcile_orchestration_runs_async(db_session)

    assert len(await _process_rows(db_session)) == 4  # no new row
    warnings = (await db_session.execute(select(OrchestrationWarning))).scalars().all()
    # manager_selection_no_manager is an unrelated, expected warning (this
    # goal is running with authority_model "no_manager" per the test's
    # manager-selection stub); only agent_definition_review_stale_inputs is
    # under test here.
    stale_input_warnings = [
        w for w in warnings if w.warning_type == "agent_definition_review_stale_inputs"
    ]
    assert stale_input_warnings == []

    # --- 3b: semantic drift (system_prompt) -> exactly one suggestion, no rerun ---
    agent.system_prompt = "Implement the requested change differently and verify it."
    await db_session.flush()

    await reconcile_orchestration_runs_async(db_session)

    assert len(await _process_rows(db_session)) == 4  # still no new row
    stale_warnings = [
        w for w in (await db_session.execute(select(OrchestrationWarning))).scalars().all()
        if w.warning_type == "agent_definition_review_stale_inputs"
    ]
    assert len(stale_warnings) == 1
    assert stale_warnings[0].source_process_run_id == review_process_run_id

    # pending decisions (team_hierarchy's) must not have been cancelled by
    # the agent_definition_review suggestion path.
    pending_after = await OrchestrationAuthorityDecisionService().list_decisions(
        db_session, goal.id, status="pending"
    )
    assert {d.id for d in pending_before} <= {d.id for d in pending_after}

    # reconcile again with the same drift: still exactly one warning (idempotent)
    await reconcile_orchestration_runs_async(db_session)
    stale_warnings_again = [
        w for w in (await db_session.execute(select(OrchestrationWarning))).scalars().all()
        if w.warning_type == "agent_definition_review_stale_inputs"
    ]
    assert len(stale_warnings_again) == 1
    assert stale_warnings_again[0].id == stale_warnings[0].id
    assert len(await _process_rows(db_session)) == 4

    # --- resolve as dismissed; further drift never resurfaces a new suggestion ---
    await OrchestrationWarningService().resolve_warning(
        db_session, stale_warnings[0], resolved_by="human:test", reason="dismissed by human"
    )

    agent.system_prompt = "A third, further-changed persona."
    await db_session.flush()

    await reconcile_orchestration_runs_async(db_session)

    all_stale_warnings = [
        w for w in (await db_session.execute(select(OrchestrationWarning))).scalars().all()
        if w.warning_type == "agent_definition_review_stale_inputs"
    ]
    assert len(all_stale_warnings) == 1
    assert all_stale_warnings[0].active is False
    assert len(await _process_rows(db_session)) == 4
