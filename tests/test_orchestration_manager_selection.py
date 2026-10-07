import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from fastapi import HTTPException

FIXED_TS = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)


async def _make_agent(db_session, *, name, role, capabilities=None, is_active=True):
    from huddleroom.models.agent import Agent

    agent = Agent(
        name=name,
        role=role,
        provider="anthropic",
        model="claude-sonnet-5-5",
        adapter_type="api",
        capabilities=capabilities or [],
        is_active=is_active,
    )
    db_session.add(agent)
    await db_session.flush()
    return agent


@pytest.mark.asyncio
async def test_management_profile_ranks_manager_roles_strong(db_session, test_project):
    from huddleroom.services.orchestration_roster_mapper import OrchestrationRosterMapper

    lead = await _make_agent(
        db_session, name="Tessa Team-Lead", role="team lead",
        capabilities=["management", "planning", "coordination"],
    )
    dev = await _make_agent(db_session, name="Devin Developer", role="developer",
                            capabilities=["coding"])

    fits = await OrchestrationRosterMapper().rank_agents(db_session, test_project.id, "management")
    by_id = {fit.agent_id: fit for fit in fits}
    assert by_id[lead.id].weak is False
    assert by_id[dev.id].weak is True
    assert fits[0].agent_id == lead.id  # manager candidate ranks first


@pytest.mark.asyncio
async def test_management_profile_matches_owner_and_pm_roles(db_session, test_project):
    from huddleroom.services.orchestration_roster_mapper import OrchestrationRosterMapper

    owner = await _make_agent(
        db_session, name="Petra Product", role="product owner",
        capabilities=["ownership", "planning"],
    )
    fits = await OrchestrationRosterMapper().rank_agents(db_session, test_project.id, "management")
    assert fits[0].agent_id == owner.id
    assert fits[0].weak is False


@pytest_asyncio.fixture
async def trivial_goal(db_session, test_project):
    from huddleroom.models.orchestration import OrchestrationGoal

    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Ship the widget",
        success_criteria=[{"key": "works", "description": "widget works", "evidence": "demo"}],
        weight="trivial",
    )
    db_session.add(goal)
    await db_session.flush()
    return goal


@pytest_asyncio.fixture
async def standard_goal(db_session, test_project):
    from huddleroom.models.orchestration import OrchestrationGoal

    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Ship the widget with strict style rules",
        success_criteria=[{"key": "works", "description": "widget works", "evidence": "demo"}],
        constraints={"style": "strict"},
        weight="standard",
    )
    db_session.add(goal)
    await db_session.flush()
    return goal


@pytest_asyncio.fixture
async def orch_run(db_session, trivial_goal):
    from huddleroom.models.orchestration import OrchestrationRun

    run = OrchestrationRun(goal_id=trivial_goal.id)
    db_session.add(run)
    await db_session.flush()
    return run


@pytest_asyncio.fixture
async def standard_run(db_session, standard_goal):
    from huddleroom.models.orchestration import OrchestrationRun

    run = OrchestrationRun(goal_id=standard_goal.id)
    db_session.add(run)
    await db_session.flush()
    return run


@pytest.mark.asyncio
async def test_goal_manager_columns_default_null_and_round_trip(db_session, trivial_goal, test_user):
    assert trivial_goal.manager_agent_id is None
    assert trivial_goal.manager_user_id is None
    assert trivial_goal.authority_model is None

    trivial_goal.authority_model = "human_manager"
    trivial_goal.manager_user_id = test_user.id
    await db_session.flush()
    await db_session.refresh(trivial_goal)
    assert trivial_goal.authority_model == "human_manager"
    assert trivial_goal.manager_user_id == test_user.id


@pytest.mark.asyncio
async def test_goal_response_schema_exposes_manager_fields(db_session, trivial_goal):
    from huddleroom.schemas.orchestration import OrchestrationGoalResponse

    payload = OrchestrationGoalResponse.model_validate(trivial_goal)
    assert payload.manager_agent_id is None
    assert payload.manager_user_id is None
    assert payload.authority_model is None


# ---------------------------------------------------------------------------
# ManagerSelectionProcess
# ---------------------------------------------------------------------------


async def _advance(db_session, goal, run):
    from huddleroom.services.orchestration_manager_selection import ManagerSelectionProcess

    return await ManagerSelectionProcess(_ManagerAnalyzerStub("legacy")).advance(db_session, goal, run)


async def _pending_select_manager(db_session, goal):
    from huddleroom.services.orchestration_authority_service import (
        OrchestrationAuthorityDecisionService,
    )
    from huddleroom.services.orchestration_manager_selection import REVIEW_OVERRIDE_DECISION_KEY

    pending = await OrchestrationAuthorityDecisionService().list_decisions(
        db_session, goal.id, status="pending"
    )
    return [d for d in pending if d.decision_key == REVIEW_OVERRIDE_DECISION_KEY]


async def _manager_candidates(db_session, goal):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    current = await OrchestrationProcessService().get_current(
        db_session, goal.id, "manager_selection"
    )
    return current.outputs["manager_candidates"]


class _ManagerAnalyzerStub:
    def __init__(self, verdict, selected_key=None):
        self.verdict = verdict
        self.selected_key = selected_key

    async def review(self, payload, project=None, *, project_id=None):
        from huddleroom.services.orchestration_manager_analyzer import ManagerAssessment

        if self.verdict == "legacy":
            recommendation = payload["deterministic_recommendation"]
            alternatives = [candidate["key"] for candidate in payload["candidates"]
                            if candidate["key"] != recommendation]
            if recommendation != "human_as_manager" or not alternatives:
                return ManagerAssessment("confirm", recommendation, "ranking confirmed")
            return ManagerAssessment("override", alternatives[0], "exercise frozen override")
        selected = self.selected_key or payload["deterministic_recommendation"]
        return ManagerAssessment(self.verdict, selected, "best fit for this goal")


def test_manager_assessment_requires_integer_schema_version():
    from huddleroom.services.orchestration_manager_analyzer import parse_manager_assessment

    with pytest.raises(ValueError, match="schema_version"):
        parse_manager_assessment(
            {"schema_version": True, "verdict": "confirm", "selected_key": "human_as_manager",
             "rationale": "fit"},
            {"human_as_manager"}, "human_as_manager",
        )


@pytest.mark.asyncio
async def test_llm_confirmation_auto_applies_frozen_recommendation(
    db_session, standard_goal, standard_run
):
    from huddleroom.services.orchestration_manager_selection import ManagerSelectionProcess

    agent = await _make_agent(
        db_session, name="Manager", role="manager", capabilities=["management", "planning"]
    )
    result = await ManagerSelectionProcess(_ManagerAnalyzerStub("confirm")).advance(
        db_session, standard_goal, standard_run
    )

    assert result["status"] == "completed"
    await db_session.refresh(standard_goal)
    assert standard_goal.manager_agent_id == agent.id
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    current = await OrchestrationProcessService().get_current(
        db_session, standard_goal.id, "manager_selection"
    )
    assert "system_prompt" not in str(current.outputs)
    assert "description" not in str(current.outputs)


def test_manager_retry_checkpoint_uses_existing_request_envelope():
    from huddleroom.schemas.orchestration import valid_lm_retry_checkpoint
    from huddleroom.services.orchestration_manager_analyzer import ManagerSelectionAnalyzer

    request = ManagerSelectionAnalyzer.build_request({
        "schema_version": 1, "goal": {},
        "candidates": [{"key": "human_as_manager"}],
        "deterministic_recommendation": "human_as_manager",
    })
    checkpoint = {
        "kind": "manager_selection", "version": 1,
        "request": request,
    }
    assert valid_lm_retry_checkpoint(checkpoint, "manager_selection") is True
    checkpoint["request"]["messages"] = []
    assert valid_lm_retry_checkpoint(checkpoint, "manager_selection") is False


@pytest.mark.asyncio
async def test_failed_manager_selection_retry_that_fails_again_reparks_without_raising(
    db_session, standard_goal, standard_run
):
    from huddleroom.schemas.orchestration import valid_lm_retry_checkpoint
    from huddleroom.services.orchestration_manager_analyzer import ManagerSelectionAnalyzer
    from huddleroom.services.orchestration_manager_selection import ManagerSelectionProcess
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    await _make_agent(
        db_session, name="Manager", role="manager", capabilities=["management", "planning"]
    )
    requests = []

    async def always_fails(**request):
        requests.append(request)
        raise RuntimeError("provider unavailable")

    process = ManagerSelectionProcess(ManagerSelectionAnalyzer(always_fails))
    failed = await process.advance(db_session, standard_goal, standard_run)
    assert failed["retryable"] is True
    current = await OrchestrationProcessService().get_current(
        db_session, standard_goal.id, "manager_selection"
    )
    assert "_lm_retry" in current.outputs

    retried = await process.retry_failed(db_session, standard_goal, standard_run, current)

    assert len(requests) == 2
    assert retried["retryable"] is True
    assert "error" in retried
    checkpoint = current.outputs["_lm_retry"]
    assert valid_lm_retry_checkpoint(checkpoint, "manager_selection") is True


@pytest.mark.asyncio
async def test_llm_override_does_not_mutate_manager_before_human_approval(
    db_session, standard_goal, standard_run, test_user
):
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
    from huddleroom.services.orchestration_manager_selection import (
        ManagerSelectionProcess, REVIEW_OVERRIDE_DECISION_KEY,
    )

    first = await _make_agent(
        db_session, name="Manager A", role="manager", capabilities=["management", "planning"]
    )
    second = await _make_agent(
        db_session, name="Manager B", role="manager", capabilities=["management"]
    )
    process = ManagerSelectionProcess(_ManagerAnalyzerStub("override", f"agent:{second.id}"))
    result = await process.advance(db_session, standard_goal, standard_run)

    assert result["status"] == "waiting_decision"
    await db_session.refresh(standard_goal)
    assert standard_goal.manager_agent_id is None
    assert standard_goal.manager_user_id is None
    assert standard_goal.authority_model is None
    decisions = await OrchestrationAuthorityDecisionService().list_decisions(
        db_session, standard_goal.id, status="pending"
    )
    decision = next(d for d in decisions if d.decision_key == REVIEW_OVERRIDE_DECISION_KEY)
    assert decision.options == [{"key": "approve", "label": "Approve"},
                                {"key": "reject", "label": "Reject"}]
    assert first.id != second.id
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option="approve", reason="approved",
        decided_by_user_id=test_user.id,
    )
    result = await process.advance(db_session, standard_goal, standard_run)
    assert result["status"] == "completed"
    await db_session.refresh(standard_goal)
    assert standard_goal.manager_agent_id == second.id
    assert standard_goal.manager_user_id is None
    assert standard_goal.authority_model == "agent_manager"


@pytest.mark.asyncio
async def test_rejected_override_applies_frozen_deterministic_recommendation(
    db_session, standard_goal, standard_run, test_user
):
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
    from huddleroom.services.orchestration_manager_selection import ManagerSelectionProcess, REVIEW_OVERRIDE_DECISION_KEY

    first = await _make_agent(
        db_session, name="Manager A", role="manager", capabilities=["management", "planning"]
    )
    second = await _make_agent(
        db_session, name="Manager B", role="manager", capabilities=["management"]
    )
    process = ManagerSelectionProcess(_ManagerAnalyzerStub("override", f"agent:{second.id}"))
    await process.advance(db_session, standard_goal, standard_run)
    decisions = await OrchestrationAuthorityDecisionService().list_decisions(
        db_session, standard_goal.id, status="pending"
    )
    decision = next(d for d in decisions if d.decision_key == REVIEW_OVERRIDE_DECISION_KEY)
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option="reject", reason="keep ranking",
        decided_by_user_id=test_user.id,
    )

    result = await process.advance(db_session, standard_goal, standard_run)
    assert result["status"] == "completed"
    await db_session.refresh(standard_goal)
    assert standard_goal.manager_agent_id == first.id


@pytest.mark.asyncio
async def test_trivial_goal_auto_selects_human_manager(db_session, trivial_goal, orch_run):
    from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    summary = await _advance(db_session, trivial_goal, orch_run)
    assert summary == {
        "process_type": "manager_selection",
        "status": "completed",
        "questions_created": 0,
    }
    await db_session.refresh(trivial_goal)
    assert trivial_goal.authority_model == "human_manager"
    assert trivial_goal.manager_agent_id is None
    assert trivial_goal.manager_user_id == trivial_goal.created_by_user_id  # None here

    assert await _pending_select_manager(db_session, trivial_goal) == []
    warnings = await OrchestrationWarningService().list_warnings(
        db_session, trivial_goal.id, active_only=True
    )
    assert warnings == []  # informational pass, not a warning (spec 6.2)

    current = await OrchestrationProcessService().get_current(
        db_session, trivial_goal.id, "manager_selection"
    )
    assert current.status == "completed"
    assert current.outputs["compressed"] is True
    assert current.outputs["gates"] == {
        "manager_selected": True,
        "authority_model_confirmed": True,
    }

    section = await OrchestrationMemoryService().get_section(
        db_session, trivial_goal.project_id, trivial_goal.id, "manager_authority"
    )
    assert section is not None
    assert "human" in section.body.lower()


@pytest.mark.asyncio
async def test_completed_process_is_terminal_and_idempotent(db_session, trivial_goal, orch_run):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    await _advance(db_session, trivial_goal, orch_run)
    summary = await _advance(db_session, trivial_goal, orch_run)
    assert summary["status"] == "completed"
    runs = await OrchestrationProcessService().list_process_runs(
        db_session, trivial_goal.id, process_type="manager_selection"
    )
    assert len(runs) == 1


@pytest.mark.asyncio
async def test_trivial_goal_implicit_human_manager_does_not_auto_rerun(
    db_session, trivial_goal, orch_run
):
    """Review finding: manager_user_id is NULL by design for the trivial
    implicit-human fallback (nobody was 'removed'). Without the `compressed`
    guard in `_stale_manager_reason`, this NULL reads exactly like 'the
    human manager was deleted' and every tick would auto-rerun and re-park
    the process forever."""
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    for _ in range(3):
        summary = await _advance(db_session, trivial_goal, orch_run)
        assert summary["status"] == "completed"
        assert summary["questions_created"] == 0

    runs = await OrchestrationProcessService().list_process_runs(
        db_session, trivial_goal.id, process_type="manager_selection"
    )
    assert len(runs) == 1
    assert await _pending_select_manager(db_session, trivial_goal) == []


@pytest.mark.asyncio
async def test_trivial_goal_creator_removed_auto_reruns(db_session, test_project, test_user):
    """Review finding, HIGH: a trivial goal persists the CONCRETE creator id
    as manager_user_id (not the generic-"human" NULL case tested above), so
    deleting/deactivating that creator must still trigger the spec 8.1
    auto-rerun instead of being silently swallowed by the `compressed`
    bypass in `_stale_manager_reason`."""
    from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun

    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Ship the widget",
        success_criteria=[{"key": "works", "description": "widget works", "evidence": "demo"}],
        weight="trivial",
        created_by_user_id=test_user.id,
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    db_session.add(run)
    await db_session.flush()

    summary = await _advance(db_session, goal, run)
    assert summary["status"] == "completed"
    await db_session.refresh(goal)
    assert goal.manager_user_id == test_user.id

    test_user.is_active = False
    await db_session.flush()

    # Auto-rerun-on-stale is now a one-time suggestion (item 3, orchestrator
    # override): the completed selection stands, the creator id is NOT
    # cleared, and no new process row is created -- a warning is raised
    # instead for a human to approve a rerun.
    summary = await _advance(db_session, goal, run)
    assert summary["status"] == "completed"
    await db_session.refresh(goal)
    assert goal.manager_user_id == test_user.id

    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    runs = await OrchestrationProcessService().list_process_runs(
        db_session, goal.id, process_type="manager_selection"
    )
    assert len(runs) == 1

    warnings = await OrchestrationWarningService().list_warnings(db_session, goal.id, active_only=True)
    stale_warnings = [w for w in warnings if w.warning_type == "manager_selection_stale_inputs"]
    assert len(stale_warnings) == 1
    assert "removed or inactive" in stale_warnings[0].message


@pytest.mark.asyncio
async def test_standard_goal_with_strong_candidate_auto_selects(
    db_session, test_project, standard_goal, standard_run
):
    lead = await _make_agent(
        db_session, name="Tessa Team-Lead", role="team lead",
        capabilities=["management", "planning", "coordination"],
    )
    summary = await _advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "completed"
    assert summary["questions_created"] == 0
    await db_session.refresh(standard_goal)
    assert standard_goal.manager_agent_id == lead.id
    assert standard_goal.authority_model == "agent_manager"
    assert await _pending_select_manager(db_session, standard_goal) == []


@pytest.mark.asyncio
async def test_standard_goal_without_candidate_recommends_human(
    db_session, standard_goal, standard_run
):
    from huddleroom.services.orchestration_manager_selection import ManagerSelectionProcess

    summary = await ManagerSelectionProcess(_ManagerAnalyzerStub("confirm")).advance(
        db_session, standard_goal, standard_run
    )
    assert summary["status"] == "completed"
    assert await _pending_select_manager(db_session, standard_goal) == []
    await db_session.refresh(standard_goal)
    assert standard_goal.authority_model == "human_manager"
    assert standard_goal.manager_user_id == standard_goal.created_by_user_id
    assert standard_goal.manager_agent_id is None


@pytest.mark.asyncio
async def test_approved_human_override_uses_goal_creator_not_approver(
    db_session, standard_goal, standard_run, test_user
):
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
    from huddleroom.services.orchestration_manager_selection import ManagerSelectionProcess
    from huddleroom.models.user import User

    standard_goal.created_by_user_id = test_user.id
    approver = User(
        email=f"approver-{uuid.uuid4()}@example.com", hashed_password="unused",
        display_name="Approver", role="member",
    )
    db_session.add(approver)
    await _make_agent(
        db_session, name="Manager", role="manager", capabilities=["management", "planning"]
    )
    await db_session.flush()
    process = ManagerSelectionProcess(_ManagerAnalyzerStub("override", "human_as_manager"))
    assert (await process.advance(db_session, standard_goal, standard_run))["status"] == "waiting_decision"
    decision = (await _pending_select_manager(db_session, standard_goal))[0]
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option="approve", reason="use creator",
        decided_by_user_id=approver.id,
    )

    assert (await process.advance(db_session, standard_goal, standard_run))["status"] == "completed"
    await db_session.refresh(standard_goal)
    assert standard_goal.manager_user_id == standard_goal.created_by_user_id == test_user.id
    assert standard_goal.manager_agent_id is None
    assert standard_goal.authority_model == "human_manager"


@pytest.mark.asyncio
async def test_approved_frozen_override_agent_inactive_before_apply(
    db_session, standard_goal, standard_run, test_user
):
    from huddleroom.services.orchestration_authority_service import (
        OrchestrationAuthorityDecisionService,
    )

    first = await _make_agent(
        db_session, name="Manager A", role="manager", capabilities=["management", "planning"]
    )
    selected = await _make_agent(
        db_session, name="Manager B", role="manager", capabilities=["management"]
    )
    from huddleroom.services.orchestration_manager_selection import ManagerSelectionProcess
    process = ManagerSelectionProcess(_ManagerAnalyzerStub("override", f"agent:{selected.id}"))
    await process.advance(db_session, standard_goal, standard_run)
    decision = (await _pending_select_manager(db_session, standard_goal))[0]
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision,
        selected_option="approve",
        reason="best fit",
        decided_by_user_id=test_user.id,
    )
    selected.is_active = False
    await db_session.flush()

    summary = await process.advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "running"
    assert summary["error"] == "frozen candidate is no longer active"
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    current = await OrchestrationProcessService().get_current(
        db_session, standard_goal.id, "manager_selection"
    )
    assert current.outputs["manager_review"]["selected_key"] == f"agent:{selected.id}"
    await db_session.refresh(standard_goal)
    assert standard_goal.manager_agent_id is None
    assert standard_goal.manager_user_id is None
    assert standard_goal.authority_model is None
    assert await _pending_select_manager(db_session, standard_goal) == []
    assert first.id != selected.id


@pytest.mark.asyncio
async def test_weight_drop_to_trivial_cancels_pending_and_completes(
    db_session, standard_goal, standard_run
):
    await _advance(db_session, standard_goal, standard_run)
    assert len(await _pending_select_manager(db_session, standard_goal)) == 1

    standard_goal.weight = "trivial"
    await db_session.flush()

    summary = await _advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "completed"
    await db_session.refresh(standard_goal)
    assert standard_goal.authority_model == "human_manager"


@pytest.mark.asyncio
async def test_skip_cancels_pending_and_creates_skip_warning(
    db_session, standard_goal, standard_run
):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    await _advance(db_session, standard_goal, standard_run)
    assert len(await _pending_select_manager(db_session, standard_goal)) == 1

    skipped = await OrchestrationProcessService().skip_process(
        db_session, standard_goal.id,
        process_type="manager_selection",
        skipped_by=f"human:{uuid.uuid4()}",
        reason="not needed for this goal",
    )
    assert skipped.status == "skipped"
    assert await _pending_select_manager(db_session, standard_goal) == []
    warnings = await OrchestrationWarningService().list_warnings(
        db_session, standard_goal.id, active_only=True
    )
    assert "manager_selection_skipped" in [w.warning_type for w in warnings]
    assert "manager_selection_no_manager" in [w.warning_type for w in warnings]
    await db_session.refresh(standard_goal)
    assert standard_goal.authority_model == "no_manager"

    summary = await _advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "skipped"  # terminal for advance()


@pytest.mark.asyncio
async def test_rerun_selecting_manager_resolves_prior_no_manager_warning(
    db_session, standard_goal, standard_run, test_user
):
    """Review finding: warnings must be re-evaluated on rerun -- a manager
    selected after a prior no_manager completion must resolve the spec-8.7
    warning, not leave it active alongside a now-selected manager."""
    from huddleroom.services.orchestration_authority_service import (
        OrchestrationAuthorityDecisionService,
    )
    from huddleroom.services.orchestration_manager_selection import NO_MANAGER_WARNING_TYPE
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    await _advance(db_session, standard_goal, standard_run)
    decision = (await _pending_select_manager(db_session, standard_goal))[0]
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision,
        selected_option="approve", reason="solo project",
        decided_by_user_id=test_user.id,
    )
    await _advance(db_session, standard_goal, standard_run)
    warnings = await OrchestrationWarningService().list_warnings(
        db_session, standard_goal.id, active_only=True
    )
    assert NO_MANAGER_WARNING_TYPE in [w.warning_type for w in warnings]

    lead = await _make_agent(
        db_session, name="Tessa Team-Lead", role="developer",
        capabilities=["coding"],
    )
    await OrchestrationProcessService().start_process(
        db_session, standard_goal.id,
        process_type="manager_selection",
        trigger_reason="human requested: re-pick manager",
    )
    await _advance(db_session, standard_goal, standard_run)
    reask = (await _pending_select_manager(db_session, standard_goal))[0]
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, reask,
        selected_option="approve", reason="fit after all",
        decided_by_user_id=test_user.id,
    )
    await _advance(db_session, standard_goal, standard_run)

    warnings = await OrchestrationWarningService().list_warnings(
        db_session, standard_goal.id, active_only=True
    )
    assert NO_MANAGER_WARNING_TYPE not in [w.warning_type for w in warnings]


@pytest.mark.asyncio
async def test_force_start_rerun_auto_selects_strong_candidate(
    db_session, standard_goal, standard_run, test_user
):
    from huddleroom.services.orchestration_authority_service import (
        OrchestrationAuthorityDecisionService,
    )
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    lead = await _make_agent(
        db_session, name="Tessa Team-Lead", role="team lead",
        capabilities=["management", "planning", "coordination"],
    )
    assert (await _advance(db_session, standard_goal, standard_run))["status"] == "completed"

    rerun = await OrchestrationProcessService().start_process(
        db_session, standard_goal.id,
        process_type="manager_selection",
        trigger_reason="human requested: re-pick manager",
    )
    assert rerun.status == "running"

    summary = await _advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "completed"
    assert summary["questions_created"] == 0
    assert standard_goal.manager_agent_id == lead.id
    assert await _pending_select_manager(db_session, standard_goal) == []


@pytest.mark.asyncio
async def test_force_rerun_replaces_pending_decision_from_prior_process_run(
    db_session, test_project, standard_goal, standard_run, monkeypatch
):
    from huddleroom.services.orchestration_authority_service import (
        OrchestrationAuthorityDecisionService,
    )
    from huddleroom.services.orchestration_debug_service import OrchestrationDebugService
    from huddleroom.services.orchestration_manager_selection import (
        ManagerSelectionProcess, REVIEW_OVERRIDE_DECISION_KEY,
    )
    from huddleroom.services.orchestration_manager_analyzer import ManagerAssessment, ManagerSelectionAnalyzer
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    process_service = OrchestrationProcessService()
    goal_definition = await process_service.start_process(
        db_session, standard_goal.id, process_type="goal_definition",
        trigger_reason="test prerequisite", run_id=standard_run.id,
    )
    await process_service.complete_process(db_session, goal_definition)

    process = ManagerSelectionProcess(_ManagerAnalyzerStub("override", "no_manager"))
    await process.advance(db_session, standard_goal, standard_run)
    old_pending = (await _pending_select_manager(db_session, standard_goal))[0]

    async def override_no_manager(self, payload, project=None, *, project_id=None):
        return ManagerAssessment("override", "no_manager", "exercise frozen override")

    monkeypatch.setattr(ManagerSelectionAnalyzer, "review", override_no_manager)
    summary = await OrchestrationDebugService().rerun_last(
        db_session, test_project.id, standard_goal.id, "manager_selection"
    )
    current = await process_service.get_current(
        db_session, standard_goal.id, "manager_selection"
    )
    assert current.id != old_pending.source_process_run_id
    assert summary["process"]["status"] == "waiting_decision"

    decisions = [
        decision
        for decision in await OrchestrationAuthorityDecisionService().list_decisions(
            db_session, standard_goal.id
        )
        if decision.decision_key == REVIEW_OVERRIDE_DECISION_KEY
    ]
    await db_session.refresh(old_pending)
    assert old_pending.status == "cancelled"
    new_pending = [decision for decision in decisions if decision.status == "pending"]
    assert len(new_pending) == 1
    assert new_pending[0].id != old_pending.id
    assert new_pending[0].source_process_run_id == current.id


@pytest.mark.asyncio
async def test_deactivated_manager_suggests_rerun_without_auto_restart(
    db_session, standard_goal, standard_run, test_user
):
    """Spec 8.1 rerun trigger: 'current manager is removed or inactive' --
    converted to a one-time suggestion (item 3, orchestrator override): the
    completed selection stands, no new process row, no pending decision."""
    from huddleroom.services.orchestration_authority_service import (
        OrchestrationAuthorityDecisionService,
    )
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    lead = await _make_agent(
        db_session, name="Tessa Team-Lead", role="developer",
        capabilities=["coding"],
    )
    await _advance(db_session, standard_goal, standard_run)
    decision = (await _pending_select_manager(db_session, standard_goal))[0]
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision,
        selected_option="approve", reason="fit",
        decided_by_user_id=test_user.id,
    )
    completed = await _advance(db_session, standard_goal, standard_run)
    assert completed["status"] == "completed"

    lead.is_active = False
    await db_session.flush()

    # No force-start call here -- the tick-driven advance() alone must
    # notice the manager went stale and raise the suggestion.
    summary = await _advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "completed"
    assert summary["questions_created"] == 0

    runs = await OrchestrationProcessService().list_process_runs(
        db_session, standard_goal.id, process_type="manager_selection"
    )
    assert len(runs) == 1

    await db_session.refresh(standard_goal)
    assert standard_goal.manager_agent_id == lead.id  # not cleared

    warnings = await OrchestrationWarningService().list_warnings(
        db_session, standard_goal.id, active_only=True
    )
    stale_warnings = [w for w in warnings if w.warning_type == "manager_selection_stale_inputs"]
    assert len(stale_warnings) == 1
    assert "removed or inactive" in stale_warnings[0].message

    # Repeated ticks with the same stale condition don't pile up warnings.
    await _advance(db_session, standard_goal, standard_run)
    warnings_again = await OrchestrationWarningService().list_warnings(
        db_session, standard_goal.id, active_only=True
    )
    assert len([w for w in warnings_again if w.warning_type == "manager_selection_stale_inputs"]) == 1


@pytest.mark.asyncio
async def test_stale_manager_leaves_memory_section_and_goal_columns_untouched(
    db_session, standard_goal, standard_run, test_user
):
    """Since the rerun trigger is now a one-time suggestion, not an
    auto-rerun, the manager_authority memory section and goal columns must
    NOT be rewritten just because the manager went stale."""
    from huddleroom.services.orchestration_authority_service import (
        OrchestrationAuthorityDecisionService,
    )
    from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService

    lead = await _make_agent(
        db_session, name="Tessa Team-Lead", role="developer",
        capabilities=["coding"],
    )
    await _advance(db_session, standard_goal, standard_run)
    decision = (await _pending_select_manager(db_session, standard_goal))[0]
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision,
        selected_option="approve", reason="fit",
        decided_by_user_id=test_user.id,
    )
    await _advance(db_session, standard_goal, standard_run)

    lead.is_active = False
    await db_session.flush()
    await _advance(db_session, standard_goal, standard_run)  # suggestion only, no rerun

    section = await OrchestrationMemoryService().get_section(
        db_session, standard_goal.project_id, standard_goal.id, "manager_authority"
    )
    assert "Tessa Team-Lead" in section.body
    assert "reselection in progress" not in section.body


@pytest.mark.asyncio
async def test_deleted_manager_agent_suggests_rerun(db_session, standard_goal, standard_run, test_user):
    """Same trigger via FK SET NULL (agent row deleted) instead of
    deactivation -- also converts to a suggestion, no auto-rerun."""
    from huddleroom.services.orchestration_authority_service import (
        OrchestrationAuthorityDecisionService,
    )
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    lead = await _make_agent(
        db_session, name="Tessa Team-Lead", role="developer",
        capabilities=["coding"],
    )
    await _advance(db_session, standard_goal, standard_run)
    decision = (await _pending_select_manager(db_session, standard_goal))[0]
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision,
        selected_option="approve", reason="fit",
        decided_by_user_id=test_user.id,
    )
    await _advance(db_session, standard_goal, standard_run)

    await db_session.delete(lead)
    await db_session.flush()
    await db_session.refresh(standard_goal)
    assert standard_goal.manager_agent_id is None  # FK SET NULL fired

    summary = await _advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "completed"
    assert summary["questions_created"] == 0

    warnings = await OrchestrationWarningService().list_warnings(
        db_session, standard_goal.id, active_only=True
    )
    assert any(w.warning_type == "manager_selection_stale_inputs" for w in warnings)


@pytest.mark.asyncio
async def test_candidate_definition_inspection_breaks_role_score_tie(
    db_session, test_project, standard_goal, standard_run
):
    """Spec 8.3 step 3: description/instructions/model/config are examined, not just role
    and capability terms. Two agents with an identical 'team lead' role/
    capability profile must be distinguishable by system_prompt content.
    Names are deliberately reversed-alphabetical from what the roster
    mapper's own tie-break (fit.name.lower()) would pick, so this only
    passes if the description-inspection bonus -- not alphabetical luck --
    is what promotes the decisive candidate."""
    silent = await _make_agent(
        db_session, name="Aaron Silent-Lead", role="team lead",
        capabilities=["management", "planning", "coordination"],
    )
    silent.description = "Writes code for the widget subsystem."
    decisive = await _make_agent(
        db_session, name="Zoe Decisive-Lead", role="team lead",
        capabilities=["management", "planning", "coordination"],
    )
    decisive.description = (
        "Owner-minded coordinator. Own the roadmap, coordinate the team, and decide priority "
        "tradeoffs when conflicts come up."
    )
    await db_session.flush()

    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    summary = await _advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "completed"
    current = await OrchestrationProcessService().get_current(
        db_session, standard_goal.id, "manager_selection"
    )
    keys = [option["key"] for option in current.outputs["candidates"]]
    # Both tie on role/capability score; the instruction-inspection bonus
    # must rank the agent whose description actually describes owning
    # decisions ahead of the one whose prompt describes coding work.
    assert keys.index(f"agent:{decisive.id}") < keys.index(f"agent:{silent.id}")
    assert standard_goal.manager_agent_id == decisive.id


@pytest.mark.asyncio
async def test_definition_inspection_does_not_reward_negated_management_terms(
    db_session, standard_goal, standard_run
):
    """Review finding: keyword matches inside negated instructions such as
    'do not manage decisions' must not increase manager fit."""
    limited = await _make_agent(
        db_session, name="Aaron Limited", role="developer",
        capabilities=["coding"],
    )
    limited.system_prompt = "Do not manage decisions. Only implement assigned tickets."
    affirmative = await _make_agent(
        db_session, name="Zoe Owner", role="developer",
        capabilities=["coding"],
    )
    affirmative.system_prompt = "Own decisions, coordinate the team, and prioritize tradeoffs."
    await db_session.flush()

    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    summary = await _advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "completed"
    current = await OrchestrationProcessService().get_current(
        db_session, standard_goal.id, "manager_selection"
    )
    options = {option["key"]: option for option in current.outputs["candidates"]}
    assert options[f"agent:{limited.id}"]["definition_bonus"] == 0
    assert options[f"agent:{affirmative.id}"]["definition_bonus"] > 0
    assert standard_goal.manager_agent_id == affirmative.id


@pytest.mark.asyncio
async def test_definition_negation_only_cancels_its_own_category(
    db_session, standard_goal, standard_run
):
    """Review finding (MEDIUM): a negation phrase must only cancel the
    category it actually qualifies, not the whole definition. A mixed-
    responsibility agent that is explicitly told NOT to make decisions, but
    whose instructions separately establish ownership and coordination,
    must still be scored (and surfaced) for the categories that were never
    negated -- the old global 'any negation phrase anywhere cancels
    everything' flag would have wiped out all three."""
    from huddleroom.services.orchestration_manager_selection import DEFINITION_CATEGORY_BONUS

    mixed = await _make_agent(
        db_session, name="Mira Mixed", role="developer",
        capabilities=["coding"],
    )
    mixed.system_prompt = (
        "Own the roadmap and coordinate the team's day-to-day work. "
        "Do not decide priority tradeoffs yourself -- escalate those."
    )
    await db_session.flush()

    summary = await _advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "waiting_decision"
    option = next(o for o in await _manager_candidates(db_session, standard_goal)
                  if o["key"] == f"agent:{mixed.id}")

    # Ownership and coordination are affirmed and unrelated to the negated
    # clause -- both must still score. Only decision-making, the negated
    # category, contributes nothing.
    assert option["definition_bonus"] == 2 * DEFINITION_CATEGORY_BONUS
    assert "instructions describe ownership" in option["signals"]
    assert "instructions describe coordination" in option["signals"]
    assert "instructions explicitly limit decision-making" in option["signals"]
    assert not any(s.startswith("instructions describe decision-making") for s in option["signals"])


@pytest.mark.asyncio
async def test_definition_negation_catches_no_and_without_phrasing(
    db_session, standard_goal, standard_run
):
    """Review finding (MEDIUM): negation cues were limited to "not "/"n't
    "/"never ", missing common phrasings like "no ownership" and "without
    ... authority" -- letting an agent explicitly denied all three
    categories still collect every bonus."""
    denied = await _make_agent(
        db_session, name="Deo Denied", role="developer",
        capabilities=["coding"],
    )
    denied.system_prompt = (
        "This agent has no ownership of the roadmap, does not coordinate "
        "other agents, and operates without any decision authority."
    )
    await db_session.flush()

    summary = await _advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "waiting_decision"
    option = next(o for o in await _manager_candidates(db_session, standard_goal)
                  if o["key"] == f"agent:{denied.id}")
    assert option["definition_bonus"] == 0
    assert not any(s.startswith("instructions describe") for s in option["signals"])


@pytest.mark.asyncio
async def test_definition_negation_not_only_is_not_negation(
    db_session, standard_goal, standard_run
):
    """Review finding (MEDIUM): "not only" is affirmative emphasis ("not
    only owns the roadmap but also...") -- it must not be read as a
    negation of the "own" term that follows it."""
    from huddleroom.services.orchestration_manager_selection import DEFINITION_CATEGORY_BONUS

    emphatic = await _make_agent(
        db_session, name="Emma Emphatic", role="developer",
        capabilities=["coding"],
    )
    emphatic.system_prompt = "This agent not only owns the roadmap but also drives it forward."
    await db_session.flush()

    summary = await _advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "waiting_decision"
    option = next(o for o in await _manager_candidates(db_session, standard_goal)
                  if o["key"] == f"agent:{emphatic.id}")
    assert option["definition_bonus"] == DEFINITION_CATEGORY_BONUS
    assert "instructions describe ownership" in option["signals"]


@pytest.mark.asyncio
async def test_definition_negation_stops_at_contrastive_segment(
    db_session, standard_goal, standard_run
):
    """A negation before "but" does not cancel the following ownership."""
    from huddleroom.services.orchestration_manager_selection import DEFINITION_CATEGORY_BONUS

    owner = await _make_agent(
        db_session, name="Contrast Owner", role="developer", capabilities=["coding"]
    )
    owner.system_prompt = "Do not manage budgets, but own the roadmap."
    await db_session.flush()

    await _advance(db_session, standard_goal, standard_run)
    option = next(
        o for o in await _manager_candidates(db_session, standard_goal)
        if o["key"] == f"agent:{owner.id}"
    )
    assert option["definition_bonus"] == DEFINITION_CATEGORY_BONUS
    assert "instructions describe ownership" in option["signals"]


@pytest.mark.asyncio
async def test_definition_negation_cues_match_whole_words_and_cannot(
    db_session, standard_goal, standard_run
):
    """Embedded cue text is affirmative; cannot still limits authority."""
    from huddleroom.services.orchestration_manager_selection import DEFINITION_CATEGORY_BONUS

    bruno = await _make_agent(
        db_session, name="Bruno Owner", role="developer", capabilities=["coding"]
    )
    bruno.system_prompt = "Bruno owns the roadmap."
    limited = await _make_agent(
        db_session, name="Cannot Decide", role="developer", capabilities=["coding"]
    )
    limited.system_prompt = "This agent cannot decide priority tradeoffs."
    await db_session.flush()

    await _advance(db_session, standard_goal, standard_run)
    options = {o["key"]: o for o in await _manager_candidates(db_session, standard_goal)}
    assert options[f"agent:{bruno.id}"]["definition_bonus"] == DEFINITION_CATEGORY_BONUS
    assert options[f"agent:{limited.id}"]["definition_bonus"] == 0


@pytest.mark.asyncio
async def test_definition_support_context_suppresses_authority_keywords(
    db_session, standard_goal, standard_run
):
    """Review finding #7: lexical-mention scoring treats mentions of
    authority-related words as exercised authority even in subordinate/support
    contexts. An agent whose system_prompt reads 'supports the owner,
    coordinates logistics, prepares decision support' currently scores bonus
    credit for owner/coordination/decision categories and can even hit
    'definition_strong' despite clearly describing a subordinate/support role,
    not someone holding authority. Support-context cues like 'supports the',
    'assists the', 'helps the', 'advises the', 'reports to' must suppress the
    bonus even when authority keywords appear in the same clause."""
    support_agent = await _make_agent(
        db_session, name="Sam Support", role="developer",
        capabilities=["coding"],
    )
    support_agent.system_prompt = (
        "supports the owner, coordinates logistics, prepares decision support"
    )
    await db_session.flush()

    summary = await _advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "waiting_decision"
    option = next(o for o in await _manager_candidates(db_session, standard_goal)
                  if o["key"] == f"agent:{support_agent.id}")

    # Support context suppresses all bonuses: no ownership, no coordination,
    # no decision-making bonus despite those keywords appearing in the text
    assert option["definition_bonus"] == 0
    # definition_strong flag must be False (not all three categories)
    assert option["weak"] is True  # weak because definition_bonus=0 and base score is low
    assert not any(s.startswith("instructions describe") for s in option["signals"])


@pytest.mark.asyncio
async def test_definition_inspection_surfaces_config_without_scoring_it(
    db_session, standard_goal, standard_run
):
    """Review finding: config boundaries are inspected and shown, but
    memory_enabled is already scored by the roster mapper and global access
    is a permission boundary, not a manager-fit bonus."""
    candidate = await _make_agent(
        db_session, name="Casey Config", role="developer",
        capabilities=["coding"],
    )
    candidate.config = {
        "memory_enabled": True,
        "allow_global_scope": True,
        "temperature": 0.2,
    }
    await db_session.flush()

    summary = await _advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "waiting_decision"
    option = next(o for o in await _manager_candidates(db_session, standard_goal)
                  if o["key"] == f"agent:{candidate.id}")
    assert option["definition_bonus"] == 0
    assert "config: memory_enabled=True (context persists across the goal)" in option["signals"]
    assert "config: allow_global_scope=True (broad permission boundary)" in option["signals"]
    assert "config: temperature=0.2" in option["signals"]
    assert "tools: api adapter (scoped to configured tools)" in option["signals"]


@pytest.mark.asyncio
async def test_definition_inspection_surfaces_cli_tools_and_personality(
    db_session, standard_goal, standard_run
):
    """Review finding: spec 8.3 step 3 names 'tools' and 'personality' as
    required inspection inputs. adapter_type/cli_runtime is the field that
    actually determines a candidate's tool/permission surface; description/
    system_prompt is the personality signal (Phase 3 precedent). Both must
    be surfaced -- not skipped -- even for a candidate with no management
    keyword bonus at all."""
    candidate = await _make_agent(
        db_session, name="Casey CLI", role="developer",
        capabilities=["coding"],
    )
    candidate.adapter_type = "cli"
    candidate.cli_runtime = "claude_code"
    candidate.description = "Calm, methodical, and communicates status clearly."
    await db_session.flush()

    summary = await _advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "waiting_decision"
    option = next(o for o in await _manager_candidates(db_session, standard_goal)
                  if o["key"] == f"agent:{candidate.id}")
    assert option["definition_bonus"] == 0  # tools/personality are surfaced, not scored
    assert any(s.startswith("tools: cli adapter (claude_code)") for s in option["signals"])


@pytest.mark.asyncio
async def test_definition_bonus_promotes_candidate_outside_raw_score_top_slice(
    db_session, test_project, standard_goal, standard_run
):
    """Review finding: candidate evaluation must not be limited to a
    pre-inspection top slice. Six 'developer' agents tie on raw roster
    score (no management role/capability terms); one of them --
    deliberately named to sort LAST alphabetically, so a pre-inspection
    slice of any size < 6 taken by (score, name) would drop it -- has a
    system_prompt describing ownership/coordination. Only a fix that inspects every
    candidate (not a pre-inspection top-N) can let this one out-rank its
    tied peers and reach the final offered options."""
    lead = await _make_agent(
        db_session, name="Tessa Team-Lead", role="team lead",
        capabilities=["management", "planning", "coordination"],
    )
    fillers = [
        await _make_agent(
            db_session, name=f"Aaron Filler-{i}", role="developer",
            capabilities=["coding"],
        )
        for i in range(1, 6)
    ]
    hidden_gem = await _make_agent(
        db_session, name="Zora Hidden-Gem", role="developer",
        capabilities=["coding"],
    )
    hidden_gem.system_prompt = (
        "You own the roadmap, coordinate the team, and decide priority "
        "tradeoffs when conflicts come up."
    )
    await db_session.flush()

    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    summary = await _advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "completed"
    current = await OrchestrationProcessService().get_current(
        db_session, standard_goal.id, "manager_selection"
    )
    keys = [option["key"] for option in current.outputs["candidates"]]
    # hidden_gem's combined score (base "developer" score + affirmative
    # instruction bonus only) must beat every tied, zero-bonus filler -- proving it was
    # actually inspected and ranked on the combined score, not silently
    # dropped by a pre-inspection slice. lead still wins the recommendation
    # on raw role/capability fit; hidden_gem must place ahead of every
    # filler it tied with pre-inspection.
    assert f"agent:{hidden_gem.id}" in keys
    for filler in fillers:
        assert f"agent:{filler.id}" not in keys or keys.index(
            f"agent:{hidden_gem.id}"
        ) < keys.index(f"agent:{filler.id}")
    assert standard_goal.manager_agent_id == lead.id


@pytest.mark.asyncio
async def test_definition_only_management_promotes_low_score_candidate_to_strong(
    db_session, test_project, standard_goal, standard_run
):
    """Review finding (HIGH): an agent whose instructions clearly establish
    ownership, coordination, AND decision-making must not stay 'weak' --
    and must be able to WIN the recommendation over a role/capability-only
    candidate -- just because its role/capabilities carry no management
    term for the roster mapper to score. Without the strong-override this
    agent's combined score (low base 'developer' score + capped bonus)
    could still land under MANAGEMENT_STRONG_THRESHOLD, silently producing
    a human-as-manager recommendation despite the instructions."""
    from huddleroom.services.orchestration_manager_selection import MANAGEMENT_STRONG_THRESHOLD

    definition_only = await _make_agent(
        db_session, name="Devin Definition-Only", role="developer",
        capabilities=["coding"],
    )
    definition_only.system_prompt = (
        "You own this project end to end: you coordinate the team's work "
        "and decide priority tradeoffs whenever conflicts come up. Point "
        "of contact for all status questions."
    )
    await db_session.flush()

    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    summary = await _advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "completed"
    current = await OrchestrationProcessService().get_current(
        db_session, standard_goal.id, "manager_selection"
    )
    option = next(
        o for o in current.outputs["candidates"] if o["key"] == f"agent:{definition_only.id}"
    )

    # The role/capability score alone would be well under threshold; the
    # instruction-only strong-override must still mark it not-weak.
    assert option["score"] < MANAGEMENT_STRONG_THRESHOLD
    assert option["weak"] is False
    assert standard_goal.manager_agent_id == definition_only.id


@pytest.mark.asyncio
async def test_strong_instruction_override_survives_options_truncation(
    db_session, test_project, standard_goal, standard_run
):
    """Review finding (HIGH): sorting candidates by combined score BEFORE
    applying the instruction-only strong override, then slicing to
    MAX_AGENT_OPTIONS, let a definition-strong candidate with a low raw
    score get pushed out of the offered options entirely by several
    role-only candidates that score higher on the roster mapper alone but
    are still 'weak' by the same MANAGEMENT_STRONG_THRESHOLD. There are
    more decoys than MAX_AGENT_OPTIONS (3) specifically so a pure
    score-only sort would drop the override candidate from both the
    options list and the recommendation -- the fix must rank every
    strong candidate (including override-only ones) ahead of every weak
    one regardless of raw score."""
    decoys = [
        await _make_agent(
            db_session, name=f"Decoy Coordinator-{i}", role="developer",
            # "product" and "lead" are both management *keywords* (+5 each)
            # but neither is a management capability_term, and role is
            # "developer" (not a management role_term), so the +30
            # capability / +35 role bonuses never fire. Base 20 + keyword
            # "product" (+5) + keyword "lead" (+5) + api adapter (+3) = 33
            # -- weak by MANAGEMENT_STRONG_THRESHOLD (45), but still higher
            # than override_only's load-floored combined score of 30 (see
            # below) -- verified via OrchestrationRosterMapper._fit_agent
            # at implementation time.
            capabilities=["product", "lead"],
        )
        for i in range(1, 5)  # 4 decoys > MAX_AGENT_OPTIONS
    ]
    override_only = await _make_agent(
        db_session, name="Aaron Override-Only", role="developer",
        capabilities=["coding"],
    )
    override_only.system_prompt = (
        "You own this project end to end: you coordinate the team's work "
        "and decide priority tradeoffs whenever conflicts come up."
    )

    # Load penalty floors override_only's roster score at 0 (max(0, 23 - 28)),
    # so its COMBINED score is 0 + DEFINITION_CATEGORY_BONUS*3 = 30 -- BELOW the
    # weak decoys' 33. Only the strong-first sort tier (not raw combined score)
    # can then rescue it into the offered options; a pure combined-score sort
    # ranks the four 33-point decoys ahead and truncates override_only out.
    from huddleroom.models.task import Task

    for _ in range(4):
        db_session.add(
            Task(
                project_id=test_project.id,
                title="active work",
                status="in_progress",
                assigned_to=override_only.id,
            )
        )

    await db_session.flush()

    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    summary = await _advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "completed"
    current = await OrchestrationProcessService().get_current(
        db_session, standard_goal.id, "manager_selection"
    )
    options = current.outputs["candidates"]
    keys = [option["key"] for option in options]

    # Review finding #8: verify the precondition claimed in the docstring.
    # The decoys must have higher combined_score than override_only, and
    # the override_only must be strong (weak=False) while decoys are weak.
    override_option = next(
        (o for o in options if o["key"] == f"agent:{override_only.id}"),
        None
    )
    assert override_option is not None, "override_only must be in decision options"
    assert override_option["weak"] is False, "override_only must be strong (weak=False)"

    # Collect decoy options from the decision to verify score relationship
    decoy_ids = {d.id for d in decoys}
    decoy_options = [
        o for o in options
        if o["key"].startswith("agent:") and uuid.UUID(o["key"].removeprefix("agent:")) in decoy_ids
    ]

    assert len(decoy_options) > 0, "decoys should be in decision options"
    for decoy_option in decoy_options:
        assert decoy_option["weak"] is True, f"decoy {decoy_option['key']} must be weak"
        assert (
            decoy_option["combined_score"] > override_option["combined_score"]
        ), (
            f"decoy {decoy_option['key']} (combined={decoy_option['combined_score']}) "
            f"must score higher than override_only (combined={override_option['combined_score']})"
        )

    # The override candidate must still be offered and recommended even
    # though several weak decoys score higher on combined roster+definition
    # score (33 vs override_only's 30) -- verify at implementation time
    # (Step 2, run-to-fail) that the chosen decoy capabilities actually
    # produce a higher combined score than override_only's, so this test
    # genuinely exercises the truncation path pre-fix; tune decoy count or
    # capabilities if the roster mapper scores them differently.
    assert f"agent:{override_only.id}" in keys
    assert standard_goal.manager_agent_id == override_only.id


@pytest.mark.asyncio
async def test_definition_inspection_reports_runtime_specific_cli_permissions(
    db_session, standard_goal, standard_run
):
    """Review finding (MEDIUM): only the `claude_code` CLI runtime actually
    runs under a non-interactive approval bypass (huddleroom/adapters/
    cli_adapter.py `_build_command`); `codex` and `aider` do not. The
    signal text must not claim every CLI agent is unrestricted, and it must
    reflect `Agent.config['cli_runtime']` when it overrides the
    `agent.cli_runtime` column -- the same resolution CliAdapter itself
    uses -- since that's what actually executes and is what the human is
    being asked to approve."""
    codex_agent = await _make_agent(
        db_session, name="Cody Codex", role="developer", capabilities=["coding"],
    )
    codex_agent.adapter_type = "cli"
    codex_agent.cli_runtime = "codex"
    overridden_agent = await _make_agent(
        db_session, name="Olive Override", role="developer", capabilities=["coding"],
    )
    overridden_agent.adapter_type = "cli"
    overridden_agent.cli_runtime = "claude_code"
    overridden_agent.config = {"cli_runtime": "aider"}
    unsafe_agents = {}
    for runtime in ("copilot", "opencode", "pi"):
        agent = await _make_agent(
            db_session, name=f"{runtime.title()} Manager", role="developer", capabilities=["coding"],
        )
        agent.adapter_type = "cli"
        agent.cli_runtime = runtime
        unsafe_agents[runtime] = agent
    await db_session.flush()

    summary = await _advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "waiting_decision"
    options = {o["key"]: o for o in await _manager_candidates(db_session, standard_goal)}

    codex_option = options[f"agent:{codex_agent.id}"]
    assert any("cli adapter (codex)" in s for s in codex_option["signals"])
    assert not any("--dangerously-skip-permissions" in s for s in codex_option["signals"])

    overridden_option = options[f"agent:{overridden_agent.id}"]
    # config["cli_runtime"] ("aider") wins over the agent.cli_runtime
    # column ("claude_code") -- must not report the unused column value
    # or claim unrestricted permissions for a runtime that doesn't use them.
    assert any("cli adapter (aider)" in s for s in overridden_option["signals"])
    assert not any("claude_code" in s for s in overridden_option["signals"])
    assert not any("--dangerously-skip-permissions" in s for s in overridden_option["signals"])
    from huddleroom.services.orchestration_manager_selection import ManagerSelectionProcess

    process = ManagerSelectionProcess()
    for runtime, agent in unsafe_agents.items():
        signals = process._inspect_candidate_definition(agent)[1]
        assert any(
            f"cli adapter ({runtime}), runs with unrestricted local tool permissions" in signal
            for signal in signals
        )


@pytest.mark.asyncio
async def test_skip_creates_spec_87_warning_and_records_no_manager(
    db_session, standard_goal, standard_run
):
    """§8.7: 'If skipped, create an active warning: <verbatim text>' --
    the shared skip service path must deliver this for every direct or
    REST caller of skip_process, not only one call site."""
    from huddleroom.services.orchestration_manager_selection import (
        NO_MANAGER_WARNING_MESSAGE,
        NO_MANAGER_WARNING_TYPE,
    )
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    await _advance(db_session, standard_goal, standard_run)
    skipped = await OrchestrationProcessService().skip_process(
        db_session, standard_goal.id,
        process_type="manager_selection",
        skipped_by=f"human:{uuid.uuid4()}",
        reason="not needed for this goal",
    )

    await db_session.refresh(standard_goal)
    assert standard_goal.authority_model == "no_manager"
    assert standard_goal.manager_agent_id is None
    assert standard_goal.manager_user_id is None

    warnings = await OrchestrationWarningService().list_warnings(
        db_session, standard_goal.id, active_only=True
    )
    warning_types = {w.warning_type for w in warnings}
    assert "manager_selection_skipped" in warning_types  # generic Phase 2 warning
    assert NO_MANAGER_WARNING_TYPE in warning_types       # spec 8.7 verbatim warning
    spec_warning = next(w for w in warnings if w.warning_type == NO_MANAGER_WARNING_TYPE)
    assert spec_warning.message == NO_MANAGER_WARNING_MESSAGE
    assert spec_warning.severity == "warning"


@pytest.mark.asyncio
async def test_skip_retry_across_run_contexts_creates_one_no_manager_warning(
    db_session, standard_goal, standard_run
):
    """Idempotent skip retry arriving with a DIFFERENT run_id than the first
    skip must not spawn a second active spec-8.7 no_manager warning. The
    warning's run linkage follows the skipped row's own run_id (fixed by the
    first skip), not the caller's run_id -- otherwise a first skip with
    run_id=None followed by a REST retry carrying the active run would create
    a duplicate active manager_selection_no_manager warning (review finding)."""
    from huddleroom.services.orchestration_manager_selection import NO_MANAGER_WARNING_TYPE
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    # No prior advance: the first skip creates the skipped row FRESH with
    # run_id=None, so the two retries genuinely diverge on run context.
    svc = OrchestrationProcessService()
    skipped_by = f"human:{uuid.uuid4()}"
    kwargs = dict(
        process_type="manager_selection",
        skipped_by=skipped_by,
        reason="not needed for this goal",
    )
    # First skip: no run context (e.g. direct/service call).
    first = await svc.skip_process(db_session, standard_goal.id, run_id=None, **kwargs)
    assert first.run_id is None
    # Retry via REST route carrying the active run -- returns the same
    # already-skipped row idempotently.
    again = await svc.skip_process(
        db_session, standard_goal.id, run_id=standard_run.id, **kwargs
    )
    assert again.id == first.id

    warnings = await OrchestrationWarningService().list_warnings(
        db_session, standard_goal.id, active_only=True
    )
    no_manager = [w for w in warnings if w.warning_type == NO_MANAGER_WARNING_TYPE]
    assert len(no_manager) == 1  # exactly one, not one per run context
    assert no_manager[0].run_id is None  # follows the skipped row's run_id


# ---------------------------------------------------------------------------
# Tick wiring and forward-progress gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tick_advances_manager_selection_after_goal_definition(
    db_session, test_project, safe_goal_analysis
):
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_service import OrchestrationService

    service = OrchestrationService()
    # The goal analyzer is fixed at this test boundary; this test covers
    # downstream process sequencing, not adaptive clarification.
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="fix typo in README",
            success_criteria=[
                {"key": "fixed", "description": "typo gone", "evidence": "diff"}
            ],
        ),
        created_by_user_id=None,
    )
    run.baseline_authorized = True
    result = await service.tick(db_session, run.id)
    assert result["baseline_process"]["status"] == "completed"
    assert result["manager_selection_process"]["status"] == "completed"
    assert result["agent_definition_review_process"]["status"] == "completed"
    assert result["team_hierarchy_process"]["status"] == "completed"

    current = await OrchestrationProcessService().get_current(
        db_session, goal.id, "manager_selection"
    )
    assert current.status == "completed"
    await db_session.refresh(goal)
    assert goal.authority_model == "human_manager"


@pytest.mark.asyncio
async def test_tick_does_not_start_manager_selection_before_goal_definition_terminal(
    db_session, test_project, monkeypatch
):
    from huddleroom.services.orchestration_goal_analyzer import (
        GoalAnalysis,
        GoalClarificationAnalyzer,
        GoalQuestion,
    )
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_service import OrchestrationService

    async def analyze_request(self, request, *, project_id=None):
        return GoalAnalysis(
            (),
            (GoalQuestion(
                "Who approves this?", "Changes execution", "orchestrator_context.execution_details"
            ),),
            False,
        )

    monkeypatch.setattr(GoalClarificationAnalyzer, "analyze_request", analyze_request)
    service = OrchestrationService()
    # A material clarification question parks goal definition, so the next
    # baseline process must not start.
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="fix typo in README",
            success_criteria=[{"key": "fixed", "description": "typo gone"}],
        ),
        created_by_user_id=None,
    )
    run.baseline_authorized = True
    result = await service.tick(db_session, run.id)
    assert result["baseline_process"]["status"] == "waiting_decision"
    assert result["manager_selection_process"] is None
    assert result["agent_definition_review_process"] is None
    assert result["team_hierarchy_process"] is None
    assert result["tick_emitted"] is True  # parked processes never block the tick

    current = await OrchestrationProcessService().get_current(
        db_session, goal.id, "manager_selection"
    )
    assert current is None  # spec 6.3: B does not start until A is terminal
    goal_definition = await OrchestrationProcessService().get_current(
        db_session, goal.id, "goal_definition"
    )
    assert goal_definition.status == "waiting_decision"


@pytest.mark.asyncio
async def test_forward_progress_gate_requires_manager_selection(
    db_session, standard_goal, standard_run
):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_service import OrchestrationService

    # Drive goal_definition to terminal via skip (human), leaving
    # manager_selection untouched.

    await OrchestrationProcessService().skip_process(
        db_session, standard_goal.id,
        process_type="goal_definition",
        skipped_by=f"human:{uuid.uuid4()}",
        reason="test: skip A",
    )
    service = OrchestrationService()
    with pytest.raises(HTTPException) as exc_info:
        await service._ensure_baseline_processes_ready(db_session, standard_goal.id)
    assert exc_info.value.status_code == 409

    # Now make manager_selection terminal too (skip) -- Phase 7 still gates progress.
    await OrchestrationProcessService().skip_process(
        db_session, standard_goal.id,
        process_type="manager_selection",
        skipped_by=f"human:{uuid.uuid4()}",
        reason="test: skip B",
    )
    with pytest.raises(HTTPException) as exc_info:
        await service._ensure_baseline_processes_ready(db_session, standard_goal.id)
    assert exc_info.value.detail == (
        "Goal definition, manager selection, agent definition review, and "
        "team hierarchy must complete before planning or delegation"
    )


@pytest.mark.asyncio
async def test_manager_selection_is_force_startable_via_rest_set():
    from huddleroom.routers.orchestration_processes import STARTABLE_PROCESS_TYPES

    assert "manager_selection" in STARTABLE_PROCESS_TYPES


# ---------------------------------------------------------------------------
# HTTP-level tests for the manager_selection start/skip routes (review
# finding #14): the constant-membership check above only proves the route
# *accepts* the process type, not that the route actually works end to end
# (project scoping, run linkage, human attribution, response shape, and the
# spec-8.7 skip side effects).
# ---------------------------------------------------------------------------


def _processes_url(project_id, goal_id, process_type, action):
    return (
        f"/api/v1/projects/{project_id}/orchestration/goals/{goal_id}"
        f"/processes/{process_type}/{action}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["start", "skip"])
async def test_manager_selection_rejects_goal_from_other_project(
    client, auth_headers, test_project, standard_goal, db_session, action
):
    """Cross-project scoping: both start and skip actions must reject
    goals from other projects with 404."""
    from huddleroom.models.project import Project

    other_project = Project(name="Other Project", description="", config={})
    db_session.add(other_project)
    await db_session.flush()

    json_body = {"reason": "human requested" if action == "start" else "not needed"}
    resp = await client.post(
        _processes_url(other_project.id, standard_goal.id, "manager_selection", action),
        json=json_body,
        headers=auth_headers,
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_start_manager_selection_links_active_run_and_response_shape(
    client, auth_headers, test_project, standard_goal, standard_run
):
    resp = await client.post(
        _processes_url(test_project.id, standard_goal.id, "manager_selection", "start"),
        json={"reason": "human wants to pick a manager now"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    body = resp.json()

    # Response serialization matches OrchestrationProcessRunResponse.
    assert set(body.keys()) == {
        "id", "goal_id", "run_id", "process_type", "process_version", "status",
        "trigger_reason", "input_snapshot", "outputs", "skipped_by",
        "override_reason", "superseded_by_id", "started_at", "completed_at",
        "created_at", "updated_at",
    }
    assert body["goal_id"] == str(standard_goal.id)
    assert body["process_type"] == "manager_selection"
    assert body["status"] == "running"
    assert "human requested: human wants to pick a manager now" == body["trigger_reason"]
    assert body["skipped_by"] is None
    assert body["superseded_by_id"] is None
    for field in ("started_at", "created_at", "updated_at"):
        assert datetime.fromisoformat(body[field]).utcoffset() == timedelta(0)

    # Run linkage: the created row must point at the goal's active run.
    assert body["run_id"] == str(standard_run.id)


@pytest.mark.asyncio
async def test_skip_manager_selection_response_shape_run_linkage_and_attribution(
    client, auth_headers, test_project, test_user, standard_goal, standard_run, db_session
):
    resp = await client.post(
        _processes_url(test_project.id, standard_goal.id, "manager_selection", "skip"),
        json={"reason": "solo project, no manager needed"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    body = resp.json()

    assert set(body.keys()) == {
        "id", "goal_id", "run_id", "process_type", "process_version", "status",
        "trigger_reason", "input_snapshot", "outputs", "skipped_by",
        "override_reason", "superseded_by_id", "started_at", "completed_at",
        "created_at", "updated_at",
    }
    assert body["goal_id"] == str(standard_goal.id)
    assert body["process_type"] == "manager_selection"
    assert body["status"] == "skipped"
    assert body["override_reason"] == "solo project, no manager needed"
    for field in ("started_at", "completed_at", "created_at", "updated_at"):
        assert datetime.fromisoformat(body[field]).utcoffset() == timedelta(0)

    # Run linkage: skipped row belongs to the goal's active orchestration run.
    assert body["run_id"] == str(standard_run.id)

    # Attribution: skipped_by reflects the calling HUMAN user (test_user via
    # auth_headers), not an agent/system actor.
    assert body["skipped_by"] == f"human:{test_user.id}"


@pytest.mark.asyncio
async def test_skip_manager_selection_applies_spec_8_7_side_effects(
    client, auth_headers, test_project, test_user, standard_goal, standard_run, db_session
):
    from huddleroom.services.orchestration_manager_selection import NO_MANAGER_WARNING_TYPE
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    # Give the goal a manager first so clearing is observable, not just
    # already-NULL.
    standard_goal.manager_user_id = test_user.id
    standard_goal.authority_model = "human_manager"
    await db_session.flush()

    resp = await client.post(
        _processes_url(test_project.id, standard_goal.id, "manager_selection", "skip"),
        json={"reason": "solo project, no manager needed"},
        headers=auth_headers,
    )
    assert resp.status_code == 200

    await db_session.refresh(standard_goal)
    assert standard_goal.authority_model == "no_manager"
    assert standard_goal.manager_agent_id is None
    assert standard_goal.manager_user_id is None

    warnings = await OrchestrationWarningService().list_warnings(
        db_session, standard_goal.id, active_only=True
    )
    warning_types = [w.warning_type for w in warnings]
    assert NO_MANAGER_WARNING_TYPE in warning_types
    assert "manager_selection_skipped" in warning_types


@pytest.mark.asyncio
async def test_answer_preserves_decision_time_roster_snapshot(
    db_session, test_project, standard_goal, standard_run, test_user
):
    """Review finding #6: when the human answers a manager-selection decision,
    the persisted candidates output and memory-section body must reflect the
    roster state at decision-time (what was shown), not a mutated roster after
    the human answered. This regression test asks with roster state X (a strong
    candidate with specific definition bonuses), mutates the roster before
    answering (e.g., change an agent's system_prompt), answers, and verifies
    the completed run's outputs["candidates"] and memory section still describe
    the original snapshot, not the mutated roster."""
    from huddleroom.services.orchestration_authority_service import (
        OrchestrationAuthorityDecisionService,
    )
    from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    # Create two candidate agents: one strong (with management definition),
    # one weak (no management signals)
    strong_candidate = await _make_agent(
        db_session, name="Tessa Candidate", role="developer",
        capabilities=["coding"],
    )
    strong_candidate.system_prompt = (
        "You own the technical roadmap and coordinate the team."
    )
    await db_session.flush()

    weak_candidate = await _make_agent(
        db_session, name="Devin Developer", role="developer",
        capabilities=["coding"],
    )

    # First advance: ask with original roster state
    summary = await _advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "waiting_decision"

    decision = (await _pending_select_manager(db_session, standard_goal))[0]
    # Save the decision-time snapshot of the strong candidate's option
    strong_option_at_ask_time = next(
        (o for o in await _manager_candidates(db_session, standard_goal)
         if o["key"] == f"agent:{strong_candidate.id}"),
        None
    )
    assert strong_option_at_ask_time is not None
    original_definition_bonus = strong_option_at_ask_time["definition_bonus"]
    original_signals = strong_option_at_ask_time["signals"]
    assert original_definition_bonus > 0  # Should have ownership/coordination bonus
    assert any("instructions describe" in s for s in original_signals)

    # NOW mutate the roster: remove the management definition from the strong candidate
    strong_candidate.system_prompt = "You write code."
    await db_session.flush()

    # Answer the decision with the mutated roster in effect
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision,
        selected_option="approve",
        reason="best management fit",
        decided_by_user_id=test_user.id,
    )

    # Advance to complete the process
    summary = await _advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "completed"

    # Verify the completed run's outputs["candidates"] reflect the ORIGINAL
    # snapshot (with the management definition), not the mutated roster
    current = await OrchestrationProcessService().get_current(
        db_session, standard_goal.id, "manager_selection"
    )
    completed_candidates = current.outputs["candidates"]
    assert len(completed_candidates) > 0

    # Find the strong candidate in the completed outputs
    strong_in_outputs = next(
        (c for c in completed_candidates if c["key"] == f"agent:{strong_candidate.id}"),
        None
    )
    assert strong_in_outputs is not None
    # The completed outputs must preserve the original decision-time bonus and signals,
    # NOT the mutated definition
    assert strong_in_outputs["definition_bonus"] == original_definition_bonus
    assert strong_in_outputs["signals"] == original_signals

    # Also verify the memory section reflects the original snapshot
    section = await OrchestrationMemoryService().get_section(
        db_session, standard_goal.project_id, standard_goal.id, "manager_authority"
    )
    assert section is not None
    # The memory body should mention the original bonus (the snapshot), not 0
    assert f"definition bonus {original_definition_bonus}" in section.body
    # And should reference the original signals


# ---------------------------------------------------------------------------
# Finding #1: Weight tier changes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_weight_promotion_trivial_to_standard_suggests_rerun(
    db_session, trivial_goal, orch_run, test_user
):
    """Finding #1: when a completed process's goal weight changes from trivial
    to standard, advance() must detect the tier change and raise a one-time
    suggestion (item 3, orchestrator override) -- not auto-rerun."""
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    # Complete the process with goal at trivial weight
    summary = await _advance(db_session, trivial_goal, orch_run)
    assert summary["status"] == "completed"
    await db_session.refresh(trivial_goal)
    assert trivial_goal.authority_model == "human_manager"

    # Promote the goal to standard weight
    trivial_goal.weight = "standard"
    await db_session.flush()

    # Advance must detect the tier change and suggest a rerun, not restart
    summary = await _advance(db_session, trivial_goal, orch_run)
    assert summary["status"] == "completed"

    runs = await OrchestrationProcessService().list_process_runs(
        db_session, trivial_goal.id, process_type="manager_selection"
    )
    assert len(runs) == 1

    warnings = await OrchestrationWarningService().list_warnings(
        db_session, trivial_goal.id, active_only=True
    )
    stale_warnings = [w for w in warnings if w.warning_type == "manager_selection_stale_inputs"]
    assert len(stale_warnings) == 1
    assert "weight tier changed" in stale_warnings[0].message


@pytest.mark.asyncio
async def test_weight_demotion_standard_to_trivial_suggests_rerun(
    db_session, standard_goal, standard_run, test_user
):
    """Finding #1: when a completed process's goal weight changes from standard
    to trivial, advance() must detect the tier change and raise a one-time
    suggestion -- the completed agent-manager selection stands unchanged."""
    from huddleroom.services.orchestration_authority_service import (
        OrchestrationAuthorityDecisionService,
    )
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    # Create agent first, then ask/answer
    agent = await _make_agent(
        db_session, name="Tessa Team-Lead", role="developer",
        capabilities=["coding"],
    )

    # Complete the process with goal at standard weight and agent manager
    await _advance(db_session, standard_goal, standard_run)
    decision = (await _pending_select_manager(db_session, standard_goal))[0]
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision,
        selected_option="approve",
        reason="fit",
        decided_by_user_id=test_user.id,
    )
    summary = await _advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "completed"
    await db_session.refresh(standard_goal)
    assert standard_goal.authority_model == "agent_manager"
    assert standard_goal.manager_agent_id == agent.id

    # Demote the goal to trivial weight
    standard_goal.weight = "trivial"
    await db_session.flush()

    # Advance must detect the tier change and suggest a rerun -- the
    # completed agent-manager selection is left as-is, not overwritten with
    # the implicit human manager.
    summary = await _advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "completed"
    await db_session.refresh(standard_goal)
    assert standard_goal.authority_model == "agent_manager"
    assert standard_goal.manager_agent_id == agent.id

    runs = await OrchestrationProcessService().list_process_runs(
        db_session, standard_goal.id, process_type="manager_selection"
    )
    assert len(runs) == 1

    warnings = await OrchestrationWarningService().list_warnings(
        db_session, standard_goal.id, active_only=True
    )
    stale_warnings = [w for w in warnings if w.warning_type == "manager_selection_stale_inputs"]
    assert len(stale_warnings) == 1
    assert "weight tier changed" in stale_warnings[0].message


# ---------------------------------------------------------------------------
# Finding #8: Position-aware support context cue matching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trailing_support_cue_does_not_suppress_earlier_terms(
    db_session, standard_goal, standard_run
):
    """Finding #8: support context cues must be position-aware. A trailing
    support cue like 'provides decision support' should not suppress earlier
    affirmative ownership/coordination terms in the same segment."""
    from huddleroom.services.orchestration_manager_selection import DEFINITION_CATEGORY_BONUS

    agent = await _make_agent(
        db_session, name="Sam Support", role="developer",
        capabilities=["coding"],
    )
    # Ownership and coordination terms BEFORE the trailing support cue
    agent.system_prompt = (
        "You own the roadmap and coordinate team execution, providing "
        "decision support to the product manager."
    )
    await db_session.flush()

    summary = await _advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "waiting_decision"
    option = next(o for o in await _manager_candidates(db_session, standard_goal)
                  if o["key"] == f"agent:{agent.id}")

    # The trailing "decision support" cue should NOT suppress the earlier
    # "own" and "coordinate" terms that precede it in the segment
    assert option["definition_bonus"] >= 2 * DEFINITION_CATEGORY_BONUS
    assert "instructions describe ownership" in option["signals"]
    assert "instructions describe coordination" in option["signals"]


@pytest.mark.asyncio
async def test_support_cue_phrase_decision_support_to_is_advisory(
    db_session, standard_goal, standard_run
):
    """Finding #8: 'decision support to X' phrasing should be recognized as
    advisory, not authoritative, even when paired with management keywords."""
    from huddleroom.services.orchestration_manager_selection import DEFINITION_CATEGORY_BONUS

    agent = await _make_agent(
        db_session, name="Advisor Agent", role="developer",
        capabilities=["coding"],
    )
    agent.system_prompt = (
        "This agent provides decision support to the owner and decision makers. "
        "Never decides independently."
    )
    await db_session.flush()

    summary = await _advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "waiting_decision"
    option = next(o for o in await _manager_candidates(db_session, standard_goal)
                  if o["key"] == f"agent:{agent.id}")

    # "decision support to" is advisory, not authoritative; it should not earn
    # decision-making bonus despite the "decision" keyword appearing
    assert option["definition_bonus"] == 0
    assert not any(s.startswith("instructions describe") for s in option["signals"])


# ---------------------------------------------------------------------------
# Finding #13: Candidates filtering to agent-only entries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_completed_candidates_uniform_shape_agent_only(
    db_session, standard_goal, standard_run, test_user
):
    """Finding #13: when storing completed candidates, only agent entries
    (with uniform score/bonus fields) should be persisted, not control entries
    like human_as_manager/no_manager (which lack those fields)."""
    from huddleroom.services.orchestration_authority_service import (
        OrchestrationAuthorityDecisionService,
    )
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    # Create some agent candidates
    for i in range(2):
        agent = await _make_agent(
            db_session, name=f"Agent-{i}", role="developer",
            capabilities=["coding"],
        )

    # Ask and answer
    await _advance(db_session, standard_goal, standard_run)
    decision = (await _pending_select_manager(db_session, standard_goal))[0]
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision,
        selected_option="reject",
        reason="I'll manage",
        decided_by_user_id=test_user.id,
    )

    # Complete
    await _advance(db_session, standard_goal, standard_run)

    # Verify completed candidates only contain agent entries with uniform shape
    current = await OrchestrationProcessService().get_current(
        db_session, standard_goal.id, "manager_selection"
    )
    candidates = current.outputs.get("candidates", [])

    # All candidates must be agent entries
    assert all(c.get("key", "").startswith("agent:") for c in candidates)

    # All must have the required score fields
    for candidate in candidates:
        assert "key" in candidate
        assert "label" in candidate
        assert "score" in candidate
        assert "definition_bonus" in candidate
        assert "combined_score" in candidate
        assert "weak" in candidate
        assert "signals" in candidate

    # Control entries (human_as_manager, no_manager) must NOT be present
    keys = [c.get("key") for c in candidates]
    assert "human_as_manager" not in keys
    assert "no_manager" not in keys


@pytest.mark.asyncio
async def test_empty_roster_manager_selection_suggests_rerun_when_agent_added(
    db_session, test_project, test_user
):
    """Bug #89: a completed run that selected human_as_manager because the
    roster had zero agent candidates must be flagged for re-evaluation once
    the roster gains candidates (item 3, orchestrator override: one-time
    suggestion, not an auto-rerun). The human's previous choice
    (human_as_manager with outputs["candidates"]==[]) stands until a human
    approves a rerun."""
    from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
    from huddleroom.services.orchestration_manager_selection import ManagerSelectionProcess
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    # Create a standard goal with a creator (so manager_user_id has a concrete value)
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Ship the widget with strict style rules",
        success_criteria=[{"key": "works", "description": "widget works", "evidence": "demo"}],
        constraints={"style": "strict"},
        weight="standard",
        created_by_user_id=test_user.id,
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    db_session.add(run)
    await db_session.flush()

    # First advance with empty roster: auto-completes with human_as_manager
    # (the "confirm" stub confirms the recommendation when no candidates)
    summary = await ManagerSelectionProcess(_ManagerAnalyzerStub("confirm")).advance(
        db_session, goal, run
    )
    assert summary["status"] == "completed"
    assert summary["questions_created"] == 0

    await db_session.refresh(goal)
    assert goal.authority_model == "human_manager"
    assert goal.manager_user_id == test_user.id

    # Verify the first run completed with empty candidates
    current = await OrchestrationProcessService().get_current(
        db_session, goal.id, "manager_selection"
    )
    assert current.outputs.get("candidates") == []

    # Now add a strong-fit manager agent
    lead = await _make_agent(
        db_session, name="Tessa Team-Lead", role="team lead",
        capabilities=["management", "planning", "coordination"],
    )

    # Advance again: the roster now has candidates, so a suggestion should be raised
    summary = await ManagerSelectionProcess(_ManagerAnalyzerStub("confirm")).advance(
        db_session, goal, run
    )
    assert summary["status"] == "completed"
    assert summary["questions_created"] == 0

    # No new run is created -- the prior human_as_manager choice stands.
    runs = await OrchestrationProcessService().list_process_runs(
        db_session, goal.id, process_type="manager_selection"
    )
    assert len(runs) == 1

    warnings = await OrchestrationWarningService().list_warnings(db_session, goal.id, active_only=True)
    stale_warnings = [w for w in warnings if w.warning_type == "manager_selection_stale_inputs"]
    assert len(stale_warnings) == 1
    assert "roster now has candidates" in stale_warnings[0].message


@pytest.mark.asyncio
async def test_explicit_human_manager_with_candidates_does_not_rerun_on_new_agent(
    db_session, standard_goal, standard_run, test_user
):
    """Bug #89 boundary condition: if candidates were offered and the human
    explicitly chose human_as_manager (outputs["candidates"] non-empty), that
    choice should not be revisited when a new agent is added. The human made
    an explicit decision among available options."""
    from huddleroom.services.orchestration_manager_selection import ManagerSelectionProcess
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    # Create an initial candidate so outputs["candidates"] is non-empty
    initial_lead = await _make_agent(
        db_session, name="Initial Lead", role="team lead",
        capabilities=["management", "planning"],
    )

    # Use override stub to force an override decision (not auto-completion)
    summary = await ManagerSelectionProcess(
        _ManagerAnalyzerStub("override", "human_as_manager")
    ).advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "waiting_decision"

    # Complete the override decision (reject the override so it uses deterministic which is human_as_manager)
    decision = (await _pending_select_manager(db_session, standard_goal))[0]
    from huddleroom.services.orchestration_authority_service import (
        OrchestrationAuthorityDecisionService,
    )
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision,
        selected_option="reject", reason="want direct involvement",
        decided_by_user_id=test_user.id,
    )

    # Complete (with human_as_manager because we rejected the override)
    summary = await ManagerSelectionProcess(
        _ManagerAnalyzerStub("override", "human_as_manager")
    ).advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "completed"

    # Verify candidates were recorded (not empty) - frozen candidates list
    current = await OrchestrationProcessService().get_current(
        db_session, standard_goal.id, "manager_selection"
    )
    assert current.outputs.get("candidates")

    # Add another agent
    new_lead = await _make_agent(
        db_session, name="New Lead", role="team lead",
        capabilities=["management", "coordination"],
    )

    # Advance again: should NOT rerun because candidates were explicitly offered
    summary = await ManagerSelectionProcess(
        _ManagerAnalyzerStub("override", "human_as_manager")
    ).advance(db_session, standard_goal, standard_run)
    assert summary["status"] == "completed"
    assert summary["questions_created"] == 0

    # Verify no second run was created
    runs = await OrchestrationProcessService().list_process_runs(
        db_session, standard_goal.id, process_type="manager_selection"
    )
    assert len(runs) == 1


@pytest.mark.asyncio
async def test_trivial_goal_empty_roster_does_not_rerun_on_new_agent(
    db_session, test_project, test_user
):
    """Bug #89 boundary condition: trivial goals auto-select human_as_manager
    regardless of roster state. Adding an agent should not trigger a rerun
    for a trivial goal (the human is always the manager)."""
    from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    # Create a trivial goal with empty roster
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Trivial task",
        success_criteria=[{"key": "done", "description": "completed", "evidence": "done"}],
        weight="trivial",
        created_by_user_id=test_user.id,
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    db_session.add(run)
    await db_session.flush()

    # First advance: should auto-complete with human_as_manager
    summary = await _advance(db_session, goal, run)
    assert summary["status"] == "completed"
    assert summary["questions_created"] == 0

    await db_session.refresh(goal)
    assert goal.authority_model == "human_manager"

    # Verify the first run completed
    current = await OrchestrationProcessService().get_current(
        db_session, goal.id, "manager_selection"
    )
    assert current.outputs.get("compressed")  # trivial uses compressed path

    # Add an agent
    lead = await _make_agent(
        db_session, name="Manager Lead", role="team lead",
        capabilities=["management", "planning"],
    )

    # Advance again: should NOT rerun for trivial goal
    summary = await _advance(db_session, goal, run)
    assert summary["status"] == "completed"
    assert summary["questions_created"] == 0

    # Verify no second run was created
    runs = await OrchestrationProcessService().list_process_runs(
        db_session, goal.id, process_type="manager_selection"
    )
    assert len(runs) == 1
