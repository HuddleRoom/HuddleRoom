"""Combined acceptance for the zero-assignment candidate review, closeout diagnostic and warning
retirement fixes.

The chain is split into two tests at the one natural seam: the closeout fixtures build their own
goal (accepted plan, done work, accepted gates) directly, which is exactly the state a goal reaches
after planning and execution. Part 1 therefore proves everything *before* task assignment on a goal
with zero assignments; part 2 proves everything from the accepted plan through completion.
Only isolated per-test databases are used; no provider is called."""
import json
import uuid

import pytest
from sqlalchemy import func, select

from huddleroom.models.orchestration import OrchestrationGate, OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.task import Task
from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess
from huddleroom.services.orchestration_agent_review_service import OrchestrationAgentReviewService
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_team_hierarchy import TeamHierarchyProcess
from huddleroom.services.orchestration_team_hierarchy_analyzer import (
    TeamHierarchyAnalysis,
    parse_team_hierarchy_analysis,
)
from tests.test_orchestration_goal_closeout import (  # noqa: F401  (fixtures + helpers reused, not copied)
    _active_warnings,
    _add_warning,
    _all_warnings,
    _kinds,
    _no_live_native_dispatch,
    _recovery_lineage,
    _tick,
    blocked_closeout,
    clean_test_database,
    completion_ready_goal,
    db_session,
    no_provider,
    test_engine,
)
from tests.test_orchestration_team_hierarchy import _agent, _CandidateReviewAnalyzer

FUNCTIONS = ("planning", "implementation", "review", "summarization")


async def _count(db, model):
    return (await db.execute(select(func.count()).select_from(model))).scalar_one()


class _CoverageHierarchyAnalyzer:
    """Stub model: assigns each required function to its candidate, covered by the accepted review id."""

    def __init__(self):
        self.payloads = []

    async def review(self, payload, project=None, *, project_id=None):
        self.payloads.append(payload)
        reviews = {r["agent_id"]: r["review_id"] for r in payload["hierarchy_context"]["agent_reviews"]}
        by_function = {cap: a["id"] for a in payload["agents"] for cap in a["capabilities"]}
        required = payload["required_work_functions"]
        analysis = TeamHierarchyAnalysis(
            proposed_agents=(),
            assignments=tuple({"work_function": fn, "agent_ref": by_function[fn]} for fn in required),
            reporting_lines=tuple(
                {"agent_ref": ref, "reports_to": "manager"} for ref in dict.fromkeys(by_function[fn] for fn in required)
            ),
            documented_gaps=(),
            rationale="Every required function is owned by a reviewed candidate.",
            self_review="Checked references and evidence.",
            responsibility_coverage=tuple(
                {"responsibility_id": f"{fn}-owner", "responsibility": f"Own {fn}.", "source_ref": "objective",
                 "work_function": fn, "agent_ref": by_function[fn], "status": "covered",
                 "reason": "Accepted review approves it.",
                 "evidence_ref": f"agent_review:{reviews[by_function[fn]]}"}
                for fn in required
            ),
        )
        parse_team_hierarchy_analysis(analysis.to_dict(), payload)  # same validation the real analyzer applies
        return analysis


@pytest.mark.asyncio
async def test_part1_zero_assignment_candidates_reviewed_then_complete_hierarchy(db_session, test_project, test_user):
    goal = OrchestrationGoal(
        project_id=test_project.id, objective="Implement the API and review the change", success_criteria=[],
        constraints={}, weight="standard", authority_model="human_manager", manager_user_id=test_user.id,
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    planner = _agent("planner-candidate", "planner", ["planning"])
    planner.description, planner.system_prompt = "Plans the work.", "Break objectives into verifiable plan items."
    impl = _agent("implementer-candidate", "developer", ["implementation"])
    impl.description, impl.system_prompt = "Helps with code.", "Be useful."
    impl.adapter_type, impl.cli_runtime = "cli", "codex"
    impl.config = {"temperature": 0.2, "reasoning_effort": "medium", "tools": ["editor"]}
    reviewer = _agent("code-reviewer", "reviewer", ["review"])
    reviewer.description, reviewer.system_prompt = "Reviews changes for defects.", "Review diffs and report findings."
    writer = _agent("summarizer-candidate", "writer", ["summarization"])
    writer.description, writer.system_prompt = "Writes summaries.", "Summarize results with evidence."
    agents = [planner, impl, reviewer, writer]
    db_session.add_all([run, *agents])
    await db_session.flush()
    assert await _count(db_session, Task) == 0

    # 1. Genuine candidate reviews with zero assignments; the finding needs a supported decision.
    review_process = AgentDefinitionReviewProcess(analyzer=_CandidateReviewAnalyzer())
    assert (await review_process.advance(db_session, goal, run))["status"] == "waiting_decision"
    reviews = {r.agent_id: r for r in await OrchestrationAgentReviewService().list_reviews(db_session, goal.id)}
    assert set(reviews) == {a.id for a in agents}
    assert await _count(db_session, Task) == 0  # nothing fabricated while reviewing
    hierarchy = TeamHierarchyProcess(analyzer=_CoverageHierarchyAnalyzer())
    request = await hierarchy._hierarchy_prompt_payload(db_session, goal, await hierarchy._analysis_input(db_session, goal), None)
    assert request["hierarchy_context"]["agent_reviews"] == []  # unresolved finding is not evidence yet

    (decision,) = await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id, status="pending")
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option="approve", reason="ok", decided_by_user_id=test_user.id)
    assert (await review_process.advance(db_session, goal, run))["status"] == "completed"
    request = await hierarchy._hierarchy_prompt_payload(db_session, goal, await hierarchy._analysis_input(db_session, goal), None)
    accepted = {item["agent_id"]: item for item in request["hierarchy_context"]["agent_reviews"]}
    assert set(accepted) == {str(a.id) for a in agents}
    assert accepted[str(impl.id)]["acceptance_basis"] == "human_decision"
    assert accepted[str(planner.id)]["acceptance_basis"] == "review_passed_unchanged"

    # 2. Complete hierarchy built from the accepted review ids.
    assert (await hierarchy.advance(db_session, goal, run))["status"] == "waiting_decision"
    service = OrchestrationAuthorityDecisionService()
    decisions = await service.list_decisions(db_session, goal.id, status="pending")
    approval = next(d for d in decisions if d.decision_key == "team_hierarchy:approval")
    await service.answer_decision(db_session, approval, selected_option="approve", decided_by_user_id=test_user.id)
    assert (await hierarchy.advance(db_session, goal, run))["status"] == "completed"
    current = await OrchestrationProcessService().get_current(db_session, goal.id, "team_hierarchy")
    assert current.outputs["approval"] == "approve"  # not approve_with_documented_gaps
    assert not current.outputs.get("documented_gaps")
    covered = (await hierarchy.analyzer.review(request, None)).responsibility_coverage  # validated by the stub
    assert {c["evidence_ref"] for c in covered} == {f"agent_review:{r.id}" for r in reviews.values()}

    # No bypasses: no tasks created by review/hierarchy, weight untouched, no override.
    assert await _count(db_session, Task) == 0
    await db_session.refresh(goal)
    assert goal.weight == "standard" and goal.weight_overridden_by is None
    assert await _count(db_session, OrchestrationGate) == 0


@pytest.mark.asyncio
async def test_part2_accepted_plan_to_blocked_closeout_retirement_and_authorized_completion(
    db_session, blocked_closeout, client, auth_headers, test_project, test_user, no_provider
):
    service, goal, run = blocked_closeout
    gates_before = {g.id: g.status for g in (await db_session.execute(select(OrchestrationGate))).scalars()}
    assert set(gates_before.values()) == {"accepted"}  # accepted work + final summary evidence, no pending gate
    cleared = await _recovery_lineage(db_session, goal, run, task_status="done")  # failed source, since fixed
    ambiguous = await _add_warning(db_session, goal, run, "summary_gate_reconciliation_unavailable")

    # Blocked closeout: the proven-cleared warning retires automatically; only the ambiguous one blocks.
    result = await _tick(db_session, service, run)
    assert result["authorized_execution"]["outcome"] == "closeout_ready"
    assert result["run_completed"] is False and result["goal_closeout_process"] is None
    await db_session.refresh(cleared)
    assert not cleared.active and cleared.resolved_by == "orchestrator:recovery"
    (diag,) = await _active_warnings(db_session, goal.id, "closeout_blocked")
    assert run.supervision_state["closeout_attention"]["source_warning_ids"] == [str(ambiguous.id)]
    assert "summary_gate_reconciliation_unavailable" in diag.message and "recovery_blocked" not in diag.message
    await db_session.refresh(ambiguous)
    assert ambiguous.active  # ambiguous provenance is never auto-retired

    # Resolve the ambiguous warning through the supported API.
    response = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/warnings/{ambiguous.id}/resolve",
        json={"reason": "Checked: the final summary gate is accepted."}, headers=auth_headers)
    assert response.status_code == 200 and response.json()["active"] is False

    # Ordinary closeout still needs its approval decision.
    result = await _tick(db_session, service, run)
    assert result["run_completed"] is False and goal.status == "active"
    assert result["goal_closeout_process"]["status"] == "waiting_decision"
    await db_session.refresh(diag)
    assert diag.active is False and diag.resolved_reason == "precondition cleared"
    decision = await db_session.get(
        OrchestrationAuthorityDecision, uuid.UUID(result["goal_closeout_process"]["decision_id"]))
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option="approve_completion", reason="ok", decided_by_user_id=test_user.id)
    done = await _tick(db_session, service, run)
    assert done["run_completed"] is True

    # No bypasses and no stale diagnostic.
    assert decision.selected_option == "approve_completion"  # not a documented-gaps workaround
    assert goal.weight == "standard" and goal.weight_overridden_by is None
    gates_after = {g.id: g.status for g in (await db_session.execute(select(OrchestrationGate))).scalars()}
    assert gates_after == gates_before  # no gate was force-accepted or created to unblock closeout
    assert await _active_warnings(db_session, goal.id, "closeout_blocked") == []
    assert "closeout_attention" not in (run.supervision_state or {})
    assert not any(str(k).startswith("warning:") for k in _kinds(run))
    assert len(await _all_warnings(db_session, goal.id, "closeout_blocked")) == 1
    assert no_provider == []
