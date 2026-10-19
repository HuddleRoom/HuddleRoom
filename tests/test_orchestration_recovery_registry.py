"""Acceptance tests for the static, approval-gated recovery registry."""

from dataclasses import replace
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from pydantic import TypeAdapter

from sqlalchemy import event

from huddleroom.models.agent import Agent
from huddleroom.models.artifact import Artifact
from huddleroom.models.graph import Graph, GraphRun
from huddleroom.models.meeting import Meeting
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationEvidence, OrchestrationGate, OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import (
    OrchestrationAgentReview,
    OrchestrationAuthorityDecision,
    OrchestrationProcessRun,
    OrchestrationWarning,
)
from huddleroom.models.project import Project
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration_recovery import (
    ChangeV1,
    EvidenceV1,
    ProposalDraftV1,
    RecoveryCallV1,
    RecoveryProposalV1,
    RecoveryTriggerV1,
    TeamConfigurationApplyV1,
    ToolReceiptV1,
    VerificationCheckV1,
)
from huddleroom.services.orchestration_recovery_configuration import RecoveryConfigurationService
from huddleroom.services.orchestration_recovery_registry import RecoveryToolRegistry
from huddleroom.services.orchestration_recovery_analyzer import RecoveryAnalyzer
from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess
from huddleroom.services.orchestration_roster_mutation import (
    CanonicalAgentState,
    CanonicalRosterState,
    GoalRosterFingerprintInput,
    ValidatedRosterDelta,
    derive_roster_mutation_plan,
    preview_roster_delta,
)
from huddleroom.services.orchestration_team_hierarchy import TeamHierarchyProcess
from huddleroom.services.orchestration_service import OrchestrationService


PLANNER_ID = UUID("11111111-1111-1111-1111-111111111111")
REVIEWER_ID = UUID("22222222-2222-2222-2222-222222222222")
GOAL_A_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
GOAL_B_ID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
RUN_A_ID = UUID("aaaaaaaa-0000-0000-0000-000000000001")
RUN_B_ID = UUID("bbbbbbbb-0000-0000-0000-000000000001")
PROJECT_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
SHA256 = "a" * 64


@pytest.fixture
def seal_without_post_state_validation(monkeypatch):
    """(carry-in 5): ``prepare`` now runs the full post-state validation at seal time. These 
    fixtures pin canonicalization, redaction and disclosure with partial team configurations (no gap
    coverage, legacy gap codes, managers the goal lacks), so they bypass only that post-state check; the
    seal-time rejection itself is pinned in test_orchestration_recovery_activation.py. Only the seal call site
    is bypassed; apply-time validation (``_validate_post_state``) always runs."""
    from huddleroom.services.orchestration_recovery_configuration import RecoveryConfigurationService

    monkeypatch.setattr(RecoveryConfigurationService, "_validate_at_seal", lambda *_args: None)


def _agent(agent_id: UUID, name: str, role: str) -> CanonicalAgentState:
    return CanonicalAgentState(
        id=agent_id,
        name=name,
        role=role,
        description=f"{role} for the project.",
        provider="local",
        model="local",
        system_prompt=f"Act as the {role}.",
        adapter_type="api",
        cli_runtime=None,
        capabilities=(role,),
        config={},
        is_active=True,
    )


def _goal_input(
    goal_id: UUID,
    run_id: UUID,
    assignments: tuple[tuple[UUID, UUID, str, bool], ...],
    *,
    objective: str | None = None,
    authority_model: str | None = "human",
) -> GoalRosterFingerprintInput:
    return GoalRosterFingerprintInput(
        goal_id=goal_id,
        run_id=run_id,
        project_id=PROJECT_ID,
        weight="standard",
        objective=objective or f"Goal {goal_id}",
        success_criteria=[],
        constraints={},
        authority_model=authority_model,
        manager_agent_id=None,
        manager_user_id=None,
        assignments=assignments,
        hierarchy_analysis=None,
        resolved_proposal_exclusions=(),
        trivial_hierarchy_proposal=None,
    )


def _evidence() -> EvidenceV1:
    return EvidenceV1(
        label="The planning gate has no eligible author.",
        value={"status": "blocked"},
        source_kind="gate",
        source_id=GOAL_A_ID,
    )


def _trigger(goal_id: UUID, run_id: UUID) -> RecoveryTriggerV1:
    return RecoveryTriggerV1.model_validate({
        "schema_version": 1,
        "goal_id": str(goal_id),
        "run_id": str(run_id),
        "source_kind": "planning_preflight",
        "source_id": str(GOAL_A_ID),
        "blocker_code": "planning_author_missing",
        "summary": "No accepted active planning author exists.",
        "source_fingerprint": SHA256,
        "retry_target": {"kind": "planning"},
    })


def _draft(tool: str) -> ProposalDraftV1:
    return ProposalDraftV1(
        kind="proposal",
        diagnosis="The planning roster has no accepted author.",
        evidence=[_evidence()],
        problem="Planning cannot start.",
        general_updates=["Repair the planning roster."],
        expected_effect="Planning is ready for its one approved retry.",
        calls=[{"tool": tool, "arguments": {}, "rationale": "Repair the blocked configuration."}],
    )


def _tool_draft(tool: str, arguments: dict[str, object]) -> ProposalDraftV1:
    """Use approval prose that cannot hide a malformed registered request."""
    payload = _draft(tool).model_dump(mode="json")
    payload.update({
        "general_updates": ["The orchestration request is fully disclosed for approval."],
        "calls": [{"tool": tool, "arguments": arguments, "rationale": "Repair the blocked configuration."}],
    })
    return ProposalDraftV1.model_validate(payload)


def _configuration_arguments(*, agents: list[dict[str, object]]) -> dict[str, object]:
    return {
        "agents": agents,
        "eligibility": [],
        "assignments": [],
        "hierarchy": {
            "manager": {"kind": "none"}, "team_leads": [], "contributors": [],
            "reviewers": [], "validators": [], "specialists": [],
        },
        "documented_gaps": [],
    }


def _proposal() -> RecoveryProposalV1:
    return RecoveryProposalV1.model_validate({
        "schema_version": 1,
        "proposal_id": str(GOAL_A_ID),
        "lineage_id": str(GOAL_B_ID),
        "attempt": 1,
        "source": {"kind": "planning_preflight", "id": str(GOAL_A_ID), "blocker_code": "planning_author_missing", "fingerprint": SHA256},
        "resource_fingerprint": SHA256,
        "registry_version": "1",
        "problem": "Planning cannot start.",
        "general_updates": ["Repair the planning roster."],
        "expected_effect": "Planning is ready for its one approved retry.",
        "evidence": [_evidence().model_dump(mode="json")],
        "calls": [{
            "call_id": "call-1", "tool": "orchestration.request_plan", "tool_version": "1", "arguments": {},
            "changes": [{"operation": "request", "scope": "goal", "target_type": "goal", "target_id": str(GOAL_A_ID), "after": {"request": "plan"}}],
        }],
        "verification": [{"check_id": "planning-ready", "verifier": "planning.ready", "verifier_version": "1", "arguments": {"goal_id": str(GOAL_A_ID)}}],
        "retry": {"target": {"kind": "planning"}, "max_attempts": 1},
        "authority_required": "human",
        "global_effect": False,
    })


async def _seed_goal_run(db_session, test_project, *, objective: str) -> tuple[OrchestrationGoal, OrchestrationRun]:
    goal = OrchestrationGoal(project_id=test_project.id, objective=objective)
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running")
    db_session.add(run)
    await db_session.flush()
    return goal, run


def test_manifest_exposes_only_the_ten_code_owned_v1_mutation_tools():
    """The analyzer may select only the bounded server-owned tool surface."""
    manifest = RecoveryToolRegistry().manifest()

    assert tuple((entry.name, entry.version) for entry in manifest) == (
        ("team_configuration.apply", "1"),
        ("orchestration.request_plan", "1"),
        ("orchestration.request_plan_revision", "1"),
        ("orchestration.request_roadmap_replan", "1"),
        ("orchestration.retry_task", "1"),
        ("orchestration.reassign_task", "1"),
        ("orchestration.request_verification", "1"),
        ("orchestration.schedule_meeting", "1"),
        ("orchestration.start_graph", "1"),
        ("orchestration.delegate_remediation", "1"),
        ("team_configuration.accepted", "1"),
        ("planning.ready", "1"),
        ("action.completed", "1"),
        ("task.completed", "1"),
        ("gate.satisfied", "1"),
        ("process.completed", "1"),
        ("warning.resolved", "1"),
        ("artifact.exists", "1"),
    )


def test_manifest_entries_expose_nonempty_strict_argument_schemas():
    for entry in RecoveryToolRegistry().manifest():
        schema = entry.arguments_schema
        if "oneOf" in schema:
            assert schema["discriminator"]["propertyName"] == "source"
            assert len(schema["oneOf"]) == 2
            for branch in schema["oneOf"]:
                definition = schema["$defs"][branch["$ref"].rsplit("/", 1)[-1]]
                assert definition["additionalProperties"] is False
                assert definition["properties"]
                assert "source" in definition["required"]
        else:
            assert schema["additionalProperties"] is False
            assert schema["properties"]
            assert schema.get("required")


@pytest.mark.asyncio
async def test_analyzer_manifest_distinguishes_calls_from_server_owned_verification(db_session, test_project):
    planner = _agent(PLANNER_ID, "Recovery Planner", "planning")
    db_session.add(Agent(**planner.__dict__))
    await db_session.flush()
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    registry = RecoveryToolRegistry()
    manifest = [entry.model_dump(mode="json") for entry in registry.manifest()]
    draft = _tool_draft("orchestration.request_plan@1", {
        "agent_id": str(planner.id), "work_function": "planning", "scope": "goal",
    }).model_copy(update={"general_updates": ["orchestration request plan"]})
    completion = AsyncMock(return_value={"choices": [{"message": {
        "content": json.dumps(draft.model_dump(mode="json")),
    }}]})
    assessed = await RecoveryAnalyzer(completion_fn=completion).analyze(
        {"registry_manifest": manifest}, project_id=test_project.id, evidence_query=AsyncMock(),
    )
    request = completion.await_args.kwargs
    captured_manifest = json.loads(request["messages"][1]["content"])["snapshot"]["registry_manifest"]
    for entry in captured_manifest[:10]:
        assert "Executable recovery tool" in entry["description"]
        assert "Allowed in proposal calls" in entry["description"]
    for entry in captured_manifest[10:]:
        assert "Read-only verification predicate" in entry["description"]
        assert "server-owned verification checks; never in proposal calls" in entry["description"]
    prompt = request["messages"][0]["content"]
    assert "Proposal calls may use only executable recovery tools" in prompt
    assert "Verification predicates are used only in server-owned verification checks" in prompt
    assert "do not author a verification field" in prompt

    proposal = await registry.canonicalize(db_session, goal, run, _trigger(goal.id, run.id), assessed)
    assert proposal.calls[0].tool == "orchestration.request_plan"
    assert proposal.verification[0].verifier == "action.completed"
    assert proposal.verification[0].arguments == {"call_id": proposal.calls[0].call_id}


@pytest.mark.asyncio
@pytest.mark.parametrize("predicate", ("planning.ready", "artifact.exists"))
async def test_verification_predicates_remain_rejected_in_proposal_calls(db_session, test_project, predicate):
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    with pytest.raises(ValueError, match="unknown_or_unversioned_tool"):
        await RecoveryToolRegistry().canonicalize(
            db_session, goal, run, _trigger(goal.id, run.id), _draft(f"{predicate}@1"),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "arguments"),
    (
        ("orchestration.request_plan@1", {"agent_id": "not-a-uuid", "work_function": "planning", "scope": "goal", "extra": True}),
        ("orchestration.request_plan_revision@1", {"plan_task_id": "not-a-uuid", "revision_request": "", "extra": True}),
        ("orchestration.request_roadmap_replan@1", {"agent_id": "not-a-uuid", "scope": "", "reason": "", "extra": True}),
        ("orchestration.retry_task@1", {"task_id": "not-a-uuid", "timeout": 0, "extra": True}),
        ("orchestration.reassign_task@1", {"task_id": "not-a-uuid", "agent_id": "not-a-uuid", "extra": True}),
        ("orchestration.request_verification@1", {"gate_id": "not-a-uuid", "work_function": "not-a-function", "extra": True}),
        ("orchestration.schedule_meeting@1", {"topic": "", "participant_agent_ids": [], "extra": True}),
        ("orchestration.start_graph@1", {"graph_id": "not-a-uuid", "subject_type": "run", "subject_id": "not-a-uuid", "extra": True}),
        ("orchestration.delegate_remediation@1", {"agent_id": "not-a-uuid", "objective": "", "desired_result": "", "extra": True}),
        ("team_configuration.apply@1", {**_configuration_arguments(agents=[]), "extra": True}),
    ),
)
async def test_registered_tools_reject_unknown_missing_and_wrong_typed_arguments_before_sealing(
    db_session, test_project, tool: str, arguments: dict[str, object],
):
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")

    with pytest.raises(ValueError):
        await RecoveryToolRegistry().canonicalize(
            db_session, goal, run, _trigger(goal.id, run.id), _tool_draft(tool, arguments),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "arguments"),
    (
        ("orchestration.request_plan@1", {"agent_id": str(REVIEWER_ID), "work_function": "planning", "scope": "goal"}),
        ("orchestration.request_plan_revision@1", {"plan_task_id": str(REVIEWER_ID), "revision_request": "Correct the plan."}),
        ("orchestration.request_roadmap_replan@1", {"agent_id": str(REVIEWER_ID), "scope": "goal", "reason": "Changed objective."}),
        ("orchestration.retry_task@1", {"task_id": str(REVIEWER_ID)}),
        ("orchestration.reassign_task@1", {"task_id": str(REVIEWER_ID), "agent_id": str(PLANNER_ID)}),
        ("orchestration.request_verification@1", {"gate_id": str(REVIEWER_ID), "work_function": "validation"}),
        ("orchestration.schedule_meeting@1", {"topic": "Resolve blocker", "participant_agent_ids": [str(REVIEWER_ID)]}),
        ("orchestration.start_graph@1", {"graph_id": str(PLANNER_ID), "subject_type": "task", "subject_id": str(REVIEWER_ID)}),
        ("orchestration.delegate_remediation@1", {"agent_id": str(REVIEWER_ID), "objective": "Repair", "desired_result": "Verified repair"}),
    ),
)
async def test_registered_tools_reject_nonexistent_or_out_of_scope_targets_before_sealing(
    db_session, test_project, tool: str, arguments: dict[str, object],
):
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")

    with pytest.raises(ValueError):
        await RecoveryToolRegistry().canonicalize(
            db_session, goal, run, _trigger(goal.id, run.id), _tool_draft(tool, arguments),
        )


@pytest.mark.asyncio
async def test_roster_preview_before_fingerprints_match_standard_production_oracles(db_session, test_project):
    planner = _agent(PLANNER_ID, "Planner", "planning")
    db_session.add(Agent(**planner.__dict__))
    await db_session.flush()
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    goal.authority_model = "agent_manager"
    goal.manager_agent_id = planner.id
    task = Task(
        project_id=test_project.id,
        title="Write the plan",
        assigned_to=planner.id,
        metadata_={"orchestration": {"run_id": str(run.id), "work_function": "planning"}},
    )
    db_session.add(task)
    await db_session.flush()

    plan = await preview_roster_delta(
        db_session,
        ValidatedRosterDelta(updates=(replace(planner, role="lead planning"),)),
        origin_goal_id=goal.id,
    )
    change = next(item for item in plan.goal_fingerprints if item.goal_id == goal.id)

    assert change.before_coverage_fingerprint == await AgentDefinitionReviewProcess().current_coverage_fingerprint(
        db_session, goal, run,
    )
    assert change.before_hierarchy_fingerprint == await TeamHierarchyProcess().current_fingerprint(
        db_session, goal, run,
    )


@pytest.mark.parametrize(
    "delta",
    (
        ValidatedRosterDelta(),
        ValidatedRosterDelta(creates=(_agent(PLANNER_ID, "Planner", "planning"),)),
        ValidatedRosterDelta(deletes=(REVIEWER_ID,)),
        ValidatedRosterDelta(
            updates=(_agent(PLANNER_ID, "Planner", "planning"),),
            deletes=(PLANNER_ID,),
        ),
    ),
)
def test_roster_plan_rejects_noop_and_invalid_delta_identity(delta: ValidatedRosterDelta):
    before = CanonicalRosterState(agents=(_agent(PLANNER_ID, "Planner", "planning"),))
    goal = _goal_input(GOAL_A_ID, RUN_A_ID, ())

    with pytest.raises(ValueError):
        derive_roster_mutation_plan(before, delta, (goal,), origin_goal_id=GOAL_A_ID)


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ("team_configuration.apply", "not.registered@1", "team_configuration.apply@2"))
async def test_registry_rejects_unversioned_or_unknown_tool_before_any_mutation(db_session, test_project, tool: str):
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")

    with pytest.raises(ValueError):
        await RecoveryToolRegistry().canonicalize(db_session, goal, run, _trigger(goal.id, run.id), _draft(tool))


@pytest.mark.asyncio
async def test_registry_rejects_unknown_verifier_and_schema_rejects_expression_arguments(db_session, test_project):
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    registry = RecoveryToolRegistry()
    proposal = _proposal()

    unknown = VerificationCheckV1(
        check_id="unknown", verifier="not.registered", verifier_version="1", arguments={"goal_id": str(goal.id)},
    )
    with pytest.raises(ValueError):
        await registry.verify(db_session, proposal, unknown)
    with pytest.raises(ValueError):
        VerificationCheckV1(
            check_id="expression",
            verifier="planning.ready",
            verifier_version="1",
            arguments={"goal_id": str(goal.id), "expression": "true"},
        )


@pytest.mark.asyncio
async def test_roster_preview_matches_production_oracles_without_writes(db_session, test_engine, test_project):
    planner = _agent(PLANNER_ID, "Planner", "planning")
    reviewer = _agent(REVIEWER_ID, "Reviewer", "review")
    db_session.add_all([
        Agent(**planner.__dict__),
        Agent(**reviewer.__dict__),
    ])
    goal_a, run_a = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    goal_b, run_b = await _seed_goal_run(db_session, test_project, objective="Review the release.")
    delta = ValidatedRosterDelta(updates=(replace(reviewer, role="validation"),))
    writes: list[str] = []

    def record_write(_conn, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
            writes.append(statement)

    event.listen(test_engine.sync_engine, "before_cursor_execute", record_write)
    try:
        preview = await preview_roster_delta(db_session, delta, origin_goal_id=goal_a.id)
    finally:
        event.remove(test_engine.sync_engine, "before_cursor_execute", record_write)

    for goal, run in ((goal_a, run_a), (goal_b, run_b)):
        change = next(item for item in preview.goal_fingerprints if item.goal_id == goal.id)
        assert change.before_coverage_fingerprint == await AgentDefinitionReviewProcess().current_coverage_fingerprint(
            db_session, goal, run,
        )
        assert change.before_hierarchy_fingerprint == await TeamHierarchyProcess().current_fingerprint(
            db_session, goal, run,
        )
    assert writes == []


@pytest.mark.asyncio
async def test_roster_preview_uses_each_goal_current_run_not_origin_goal_as_run_id(db_session, test_project):
    planner = _agent(PLANNER_ID, "Planner", "planning")
    db_session.add(Agent(**planner.__dict__))
    await db_session.flush()
    goal_a, run_a = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    goal_b, run_b = await _seed_goal_run(db_session, test_project, objective="Review the release.")
    db_session.add(Task(
        project_id=test_project.id,
        title="Only run A owns this planning task",
        assigned_to=planner.id,
        metadata_={"orchestration": {"run_id": str(run_a.id), "work_function": "planning"}},
    ))
    await db_session.flush()

    plan = await preview_roster_delta(
        db_session,
        ValidatedRosterDelta(updates=(replace(planner, role="lead planning"),)),
        origin_goal_id=goal_a.id,
    )
    change_b = next(item for item in plan.goal_fingerprints if item.goal_id == goal_b.id)

    assert change_b.before_coverage_fingerprint == await AgentDefinitionReviewProcess().current_coverage_fingerprint(
        db_session, goal_b, run_b,
    )
    assert change_b.before_hierarchy_fingerprint == await TeamHierarchyProcess().current_fingerprint(
        db_session, goal_b, run_b,
    )


@pytest.mark.asyncio
async def test_roster_preview_substantial_coverage_preserves_scored_inferred_functions(db_session, test_project):
    planner = _agent(PLANNER_ID, "Product Planner", "product strategist")
    builder = _agent(REVIEWER_ID, "Builder", "software engineer")
    db_session.add_all([Agent(**planner.__dict__), Agent(**builder.__dict__)])
    await db_session.flush()
    goal, run = await _seed_goal_run(db_session, test_project, objective="Build and validate the release.")
    goal.weight = "substantial"
    await db_session.flush()

    plan = await preview_roster_delta(
        db_session,
        ValidatedRosterDelta(updates=(replace(builder, role="lead software engineer"),)),
        origin_goal_id=goal.id,
    )
    change = next(item for item in plan.goal_fingerprints if item.goal_id == goal.id)

    assert change.before_coverage_fingerprint == await AgentDefinitionReviewProcess().current_coverage_fingerprint(
        db_session, goal, run,
    )


@pytest.mark.asyncio
async def test_roster_preview_trivial_hierarchy_uses_exact_production_proposal_payload(db_session, test_project):
    planner = _agent(PLANNER_ID, "Planner", "planning")
    db_session.add(Agent(**planner.__dict__))
    await db_session.flush()
    goal, run = await _seed_goal_run(db_session, test_project, objective="Write a brief.")
    goal.weight = "trivial"
    await db_session.flush()

    plan = await preview_roster_delta(
        db_session,
        ValidatedRosterDelta(updates=(replace(planner, role="lead planner"),)),
        origin_goal_id=goal.id,
    )
    change = next(item for item in plan.goal_fingerprints if item.goal_id == goal.id)

    assert change.before_coverage_fingerprint == await AgentDefinitionReviewProcess().current_coverage_fingerprint(
        db_session, goal, run,
    )
    assert change.before_hierarchy_fingerprint == await TeamHierarchyProcess().current_fingerprint(
        db_session, goal, run,
    )


@pytest.mark.asyncio
async def test_roster_preview_hierarchy_preserves_manager_order_and_runtime_field_semantics(db_session, test_project):
    manager = _agent(PLANNER_ID, "Zulu Manager", "manager")
    unrelated = _agent(REVIEWER_ID, "Alpha Reviewer", "review")
    db_session.add_all([Agent(**manager.__dict__), Agent(**unrelated.__dict__)])
    await db_session.flush()
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan and review the release.")
    goal.authority_model = "agent_manager"
    goal.manager_agent_id = manager.id
    await db_session.flush()

    runtime_only = await preview_roster_delta(
        db_session,
        ValidatedRosterDelta(updates=(replace(unrelated, provider="other", config={"endpoint": "local"}),)),
        origin_goal_id=goal.id,
    )
    runtime_change = next(item for item in runtime_only.goal_fingerprints if item.goal_id == goal.id)
    semantic_change = await preview_roster_delta(
        db_session,
        ValidatedRosterDelta(updates=(replace(unrelated, role="validation"),)),
        origin_goal_id=goal.id,
    )
    semantic_fingerprint = next(item for item in semantic_change.goal_fingerprints if item.goal_id == goal.id)

    assert runtime_change.before_hierarchy_fingerprint == await TeamHierarchyProcess().current_fingerprint(
        db_session, goal, run,
    )
    assert runtime_change.before_hierarchy_fingerprint == runtime_change.after_hierarchy_fingerprint
    assert semantic_fingerprint.before_hierarchy_fingerprint != semantic_fingerprint.after_hierarchy_fingerprint


@pytest.mark.asyncio
async def test_roster_preview_coverage_excludes_action_and_authority_deliveries(db_session, test_project):
    planner = _agent(PLANNER_ID, "Planner", "planning")
    db_session.add(Agent(**planner.__dict__))
    await db_session.flush()
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    db_session.add_all([
        Task(
            project_id=test_project.id,
            title="Action-created planning delivery",
            assigned_to=planner.id,
            metadata_={"orchestration": {
                "run_id": str(run.id),
                "work_function": "planning",
                "action_id": str(UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")),
            }},
        ),
        Task(
            project_id=test_project.id,
            title="Authority delivery",
            assigned_to=planner.id,
            metadata_={"orchestration": {
                "run_id": str(run.id),
                "work_function": "planning",
                "authority_decision_id": str(UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee")),
            }},
        ),
    ])
    await db_session.flush()

    plan = await preview_roster_delta(
        db_session,
        ValidatedRosterDelta(updates=(replace(planner, role="lead planner"),)),
        origin_goal_id=goal.id,
    )
    change = next(item for item in plan.goal_fingerprints if item.goal_id == goal.id)

    assert change.before_coverage_fingerprint == await AgentDefinitionReviewProcess().current_coverage_fingerprint(
        db_session, goal, run,
    )


@pytest.mark.asyncio
async def test_coverage_keeps_action_work_by_default_and_sanitizes_only_covered_targets(db_session, test_project):
    planner = _agent(PLANNER_ID, "Planner", "planning")
    db_session.add(Agent(**planner.__dict__))
    await db_session.flush()
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    action_task = Task(
        project_id=test_project.id,
        title="Action-created planning delivery",
        assigned_to=planner.id,
        metadata_={"orchestration": {
            "run_id": str(run.id),
            "work_function": "planning",
            "action_id": str(UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")),
        }},
    )
    db_session.add(action_task)
    await db_session.flush()
    process = AgentDefinitionReviewProcess()
    assignments, targets, _ = await process._review_inputs(db_session, goal, run)
    expected_default = process._coverage_fingerprint(goal, targets)
    baseline = [item for item in assignments if not process._assignment_action_id(item[0])]
    baseline_targets = await process._targets(db_session, goal, baseline)
    sanitized = [
        (task, agent, "" if process._assignment_action_id(task) else work_function)
        for task, agent, work_function in assignments
    ]
    sanitized_targets = await process._targets(db_session, goal, sanitized)
    expected_covered = process._coverage_fingerprint(
        goal,
        process._merge_coverage_targets(
            baseline_targets,
            [target for target in sanitized_targets if str(target[0].id) == str(planner.id)],
        ),
    )

    assert await process.current_coverage_fingerprint(db_session, goal, run) == expected_default
    assert await process.current_coverage_fingerprint(
        db_session, goal, run, covered_target_ids={str(planner.id)},
    ) == expected_covered


@pytest.mark.asyncio
async def test_preview_uses_completed_review_target_ids_for_accepted_coverage(db_session, test_project):
    planner = _agent(PLANNER_ID, "Planner", "planning")
    reviewer = _agent(REVIEWER_ID, "Reviewer", "review")
    db_session.add_all([Agent(**planner.__dict__), Agent(**reviewer.__dict__)])
    await db_session.flush()
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    db_session.add_all([
        Task(
            project_id=test_project.id,
            title="Baseline planning assignment",
            assigned_to=planner.id,
            metadata_={"orchestration": {"run_id": str(run.id), "work_function": "planning"}},
        ),
        Task(
            project_id=test_project.id,
            title="Action-only reviewed target",
            assigned_to=reviewer.id,
            metadata_={"orchestration": {
                "run_id": str(run.id),
                "work_function": "review",
                "action_id": str(UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")),
            }},
        ),
        OrchestrationProcessRun(
            goal_id=goal.id,
            run_id=run.id,
            process_type="agent_definition_review",
            status="completed",
            trigger_reason="accepted review",
            outputs={"target_ids": [str(reviewer.id)]},
        ),
    ])
    await db_session.flush()
    expected = await AgentDefinitionReviewProcess().current_coverage_fingerprint(
        db_session, goal, run, covered_target_ids={str(reviewer.id)},
    )

    plan = await preview_roster_delta(
        db_session,
        ValidatedRosterDelta(updates=(replace(reviewer, role="lead reviewer"),)),
        origin_goal_id=goal.id,
    )
    change = next(item for item in plan.goal_fingerprints if item.goal_id == goal.id)

    assert change.before_coverage_fingerprint == expected


@pytest.mark.asyncio
async def test_roster_preview_hierarchy_keeps_objective_notes_from_goal_context(db_session, test_project):
    planner = _agent(PLANNER_ID, "Planner", "planning")
    investigator = _agent(REVIEWER_ID, "Investigator", "researcher")
    db_session.add_all([Agent(**planner.__dict__), Agent(**investigator.__dict__)])
    await db_session.flush()
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan release")
    goal.orchestrator_context = {"objective_notes": ["Research root cause"]}
    await db_session.flush()

    plan = await preview_roster_delta(
        db_session,
        ValidatedRosterDelta(updates=(replace(investigator, role="lead researcher"),)),
        origin_goal_id=goal.id,
    )
    change = next(item for item in plan.goal_fingerprints if item.goal_id == goal.id)
    coverage_process = AgentDefinitionReviewProcess()
    _assignments, raw_targets, _ = await coverage_process._review_inputs(db_session, goal, run)

    assert change.before_coverage_fingerprint == coverage_process._coverage_fingerprint(goal, raw_targets)
    assert change.before_hierarchy_fingerprint == await TeamHierarchyProcess().current_fingerprint(
        db_session, goal, run,
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("seal_without_post_state_validation")
async def test_configuration_preparation_seals_complete_resolved_change_without_writes(db_session, test_engine, test_project):
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    draft = TeamConfigurationApplyV1.model_validate({
        "agents": [{
            "operation": "create",
            "ref": "new:planner",
            "definition": {
                "name": "Recovery Planner",
                "role": "planning",
                "description": "Plans the project deliverables.",
                "provider": "local",
                "model": "local",
                "adapter_type": "api",
                "capabilities": ["planning"],
                "config": {},
            },
        }],
        "eligibility": [{"agent_ref": "new:planner", "work_functions": ["planning"]}],
        "assignments": [{"work_function": "planning", "agent_ref": "new:planner"}],
        "hierarchy": {
            "manager": {"kind": "none"}, "team_leads": ["new:planner"], "contributors": [],
            "reviewers": [], "validators": [], "specialists": [],
        },
        "documented_gaps": [],
    })
    writes: list[str] = []

    def record_write(_conn, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
            writes.append(statement)

    event.listen(test_engine.sync_engine, "before_cursor_execute", record_write)
    try:
        prepared = await RecoveryConfigurationService().prepare(db_session, goal, run, draft)
    finally:
        event.remove(test_engine.sync_engine, "before_cursor_execute", record_write)

    assert prepared.roster_plan == await preview_roster_delta(
        db_session, prepared.roster_delta, origin_goal_id=goal.id,
    )
    assert prepared.changes[0].operation == "create"
    assert prepared.changes[0].scope == "global"
    assert len(prepared.roster_delta.creates) == 1
    assert "new:" not in json.dumps(prepared.arguments, sort_keys=True)
    assert writes == []


@pytest.mark.asyncio
@pytest.mark.parametrize("inactive", (False, True))
async def test_configuration_preparation_rejects_missing_and_inactive_update_targets(db_session, test_project, inactive: bool):
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    agent_id = PLANNER_ID
    if inactive:
        agent = _agent(agent_id, "Inactive Planner", "planning")
        db_session.add(Agent(**{**agent.__dict__, "is_active": False}))
        await db_session.flush()
    draft = TeamConfigurationApplyV1.model_validate(_configuration_arguments(agents=[{
        "operation": "update", "agent_id": str(agent_id), "set": {"role": "lead planning"},
    }]))

    with pytest.raises(ValueError):
        await RecoveryConfigurationService().prepare(db_session, goal, run, draft)


@pytest.mark.asyncio
async def test_configuration_preparation_rejects_duplicate_local_refs_and_unresolved_poststate_refs(db_session, test_project):
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    definition = {
        "name": "Recovery Planner", "role": "planning", "provider": "local", "model": "local",
        "adapter_type": "api", "capabilities": ["planning"], "config": {},
    }
    duplicate = TeamConfigurationApplyV1.model_validate(_configuration_arguments(agents=[
        {"operation": "create", "ref": "new:planner", "definition": definition},
        {"operation": "create", "ref": "new:planner", "definition": {**definition, "name": "Recovery Planner 2"}},
    ]))
    unresolved_payload = _configuration_arguments(agents=[])
    unresolved_payload["eligibility"] = [{"agent_ref": "new:missing", "work_functions": ["planning"]}]
    unresolved = TeamConfigurationApplyV1.model_validate(unresolved_payload)

    with pytest.raises(ValueError):
        await RecoveryConfigurationService().prepare(db_session, goal, run, duplicate)
    with pytest.raises(ValueError):
        await RecoveryConfigurationService().prepare(db_session, goal, run, unresolved)


@pytest.mark.asyncio
async def test_configuration_preparation_rejects_duplicate_updates_and_a_noop_complete_poststate(db_session, test_project):
    planner = _agent(PLANNER_ID, "Planner", "planning")
    db_session.add(Agent(**planner.__dict__))
    await db_session.flush()
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    duplicate_updates = TeamConfigurationApplyV1.model_validate(_configuration_arguments(agents=[
        {"operation": "update", "agent_id": str(planner.id), "set": {"role": "lead planning"}},
        {"operation": "update", "agent_id": str(planner.id), "set": {"description": "Own the recovery."}},
    ]))
    noop = TeamConfigurationApplyV1.model_validate(_configuration_arguments(agents=[{
        "operation": "update", "agent_id": str(planner.id), "set": {"role": planner.role},
    }]))

    with pytest.raises(ValueError):
        await RecoveryConfigurationService().prepare(db_session, goal, run, duplicate_updates)
    with pytest.raises(ValueError):
        await RecoveryConfigurationService().prepare(db_session, goal, run, noop)


@pytest.mark.asyncio
@pytest.mark.usefixtures("seal_without_post_state_validation")
async def test_c1_configuration_prepare_mirrors_agent_service_cli_cleanup_and_sparse_patch(db_session, test_project):
    cli_agent = replace(
        _agent(PLANNER_ID, "CLI Planner", "planning"),
        adapter_type="cli", cli_runtime="column-runtime",
        capabilities=("planning", "review"), config={"cli_runtime": "config-runtime", "mode": "safe"},
    )
    db_session.add(Agent(**cli_agent.__dict__))
    await db_session.flush()
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    payload = _configuration_arguments(agents=[{
        "operation": "update", "agent_id": str(cli_agent.id), "set": {"adapter_type": "api"},
    }])
    prepared = await RecoveryConfigurationService().prepare(
        db_session, goal, run, TeamConfigurationApplyV1.model_validate(payload),
    )
    sealed_update = prepared.arguments["agents"][0]
    revalidated = TeamConfigurationApplyV1.model_validate(prepared.arguments)
    state = prepared.roster_delta.updates[0]

    assert sealed_update["set"] == {"adapter_type": "api"}
    assert revalidated.agents[0].set.model_fields_set == {"adapter_type"}
    assert state.adapter_type == "api"
    assert state.cli_runtime is None
    assert "cli_runtime" not in state.config
    assert state.capabilities == cli_agent.capabilities
    assert state.config == {"mode": "safe"}


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ("capabilities", "config"))
@pytest.mark.usefixtures("seal_without_post_state_validation")
async def test_c1_configuration_prepare_preserves_explicit_none_json_updates_like_agent_service(
    db_session, test_project, field: str,
):
    planner = replace(_agent(PLANNER_ID, "Planner", "planning"), capabilities=("planning", "review"), config={"mode": "safe"})
    db_session.add(Agent(**planner.__dict__))
    await db_session.flush()
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    prepared = await RecoveryConfigurationService().prepare(
        db_session,
        goal,
        run,
        TeamConfigurationApplyV1.model_validate(_configuration_arguments(agents=[{
            "operation": "update", "agent_id": str(planner.id), "set": {"role": "lead planning", field: None},
        }])),
    )
    state = prepared.roster_delta.updates[0]

    assert state.capabilities == planner.capabilities
    assert state.config == planner.config
    assert prepared.arguments["agents"][0]["set"] == {"role": "lead planning", field: None}


@pytest.mark.asyncio
@pytest.mark.usefixtures("seal_without_post_state_validation")
async def test_c1_configuration_prepare_round_trips_sparse_poststate_without_writes(db_session, test_engine, test_project):
    planner = replace(
        _agent(PLANNER_ID, "Planner", "planning"),
        description="Preserve this description.", capabilities=("planning", "review"), config={"mode": "safe"},
    )
    db_session.add(Agent(**planner.__dict__))
    await db_session.flush()
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    draft = TeamConfigurationApplyV1.model_validate(_configuration_arguments(agents=[{
        "operation": "update", "agent_id": str(planner.id), "set": {"role": "lead planning"},
    }]))
    writes: list[str] = []

    def record_write(_conn, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
            writes.append(statement)

    event.listen(test_engine.sync_engine, "before_cursor_execute", record_write)
    try:
        first = await RecoveryConfigurationService().prepare(db_session, goal, run, draft)
        second = await RecoveryConfigurationService().prepare(
            db_session, goal, run, TeamConfigurationApplyV1.model_validate(first.arguments),
        )
    finally:
        event.remove(test_engine.sync_engine, "before_cursor_execute", record_write)

    assert first.roster_delta == second.roster_delta
    assert first.roster_plan == second.roster_plan
    assert first.arguments == second.arguments
    assert writes == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("seal_without_post_state_validation")
async def test_c1_configuration_rejects_within_surface_duplicates_but_allows_cross_surface_reuse(db_session, test_project):
    planner = _agent(PLANNER_ID, "Planner", "planning")
    db_session.add(Agent(**planner.__dict__))
    await db_session.flush()
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    base = _configuration_arguments(agents=[{
        "operation": "update", "agent_id": str(planner.id), "set": {"role": "lead planning"},
    }])
    base["eligibility"] = [{"agent_ref": str(planner.id), "work_functions": ["planning"]}]
    base["assignments"] = [{"agent_ref": str(planner.id), "work_function": "planning"}]
    base["hierarchy"]["team_leads"] = [str(planner.id)]
    positive = TeamConfigurationApplyV1.model_validate(base)
    duplicate_eligibility = TeamConfigurationApplyV1.model_validate({**base, "eligibility": base["eligibility"] * 2})
    duplicate_assignment = TeamConfigurationApplyV1.model_validate({**base, "assignments": base["assignments"] * 2})
    duplicate_members = tuple(
        TeamConfigurationApplyV1.model_validate({
            **base,
            "hierarchy": {**base["hierarchy"], key: [str(planner.id), str(planner.id)]},
        })
        for key in ("team_leads", "contributors", "reviewers", "validators", "specialists")
    )

    prepared = await RecoveryConfigurationService().prepare(db_session, goal, run, positive)
    assert prepared.arguments["eligibility"][0]["agent_id"] == str(planner.id)
    assert prepared.arguments["assignments"][0]["agent_id"] == str(planner.id)
    assert prepared.arguments["hierarchy"]["team_leads"] == [str(planner.id)]
    for duplicate in (duplicate_eligibility, duplicate_assignment, *duplicate_members):
        with pytest.raises(ValueError):
            await RecoveryConfigurationService().prepare(db_session, goal, run, duplicate)


@pytest.mark.asyncio
@pytest.mark.usefixtures("seal_without_post_state_validation")
async def test_c1_configuration_changes_redact_legacy_secrets_and_disclose_all_aggregate_surfaces(db_session, test_project):
    secret = "sk-live-secret"
    planner = replace(
        _agent(PLANNER_ID, "Planner", "planning"),
        system_prompt=f"Use token={secret} only.", config={"mode": f"token={secret}"},
    )
    db_session.add(Agent(**planner.__dict__))
    await db_session.flush()
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    collateral_goal, _collateral_run = await _seed_goal_run(db_session, test_project, objective="Review the release.")
    payload = _configuration_arguments(agents=[{
        "operation": "update", "agent_id": str(planner.id), "set": {"role": "lead planning"},
    }])
    payload.update({
        "eligibility": [{"agent_ref": str(planner.id), "work_functions": ["planning"]}],
        "assignments": [{"agent_ref": str(planner.id), "work_function": "planning"}],
        "hierarchy": {
            "manager": {"kind": "agent", "agent_ref": str(planner.id)},
            "team_leads": [str(planner.id)], "contributors": [], "reviewers": [str(planner.id)],
            "validators": [], "specialists": [],
        },
        "documented_gaps": [{"code": "review_capacity", "rationale": "Record the accepted coverage gap."}],
    })
    prepared = await RecoveryConfigurationService().prepare(
        db_session, goal, run, TeamConfigurationApplyV1.model_validate(payload),
    )
    changes = [change.model_dump(mode="json") for change in prepared.changes]
    targets = {(change["target_type"], str(change["target_id"])) for change in changes}
    serialized = json.dumps(changes, sort_keys=True)

    assert secret not in serialized
    assert "[REDACTED]" in serialized
    assert {kind for kind, _identifier in targets} >= {
        "agent", "eligibility", "assignments", "manager", "reporting_hierarchy", "documented_gaps", "configuration_fingerprint",
    }
    expected_fingerprints = {
        str(item.goal_id): {
            "before": {"coverage": item.before_coverage_fingerprint, "hierarchy": item.before_hierarchy_fingerprint},
            "after": {"coverage": item.after_coverage_fingerprint, "hierarchy": item.after_hierarchy_fingerprint},
        }
        for item in prepared.roster_plan.goal_fingerprints
        if item.goal_id in {goal.id, collateral_goal.id}
    }
    observed_fingerprints: dict[str, dict[str, object]] = {}
    for change in changes:
        if change["target_type"] != "configuration_fingerprint":
            continue
        if change["target_id"] == "aggregate":
            observed_fingerprints.update(change["after"])
        else:
            observed_fingerprints[str(change["target_id"])] = {
                "before": change["before"], "after": change["after"],
            }
    assert observed_fingerprints == expected_fingerprints


@pytest.mark.asyncio
@pytest.mark.usefixtures("seal_without_post_state_validation")
async def test_c1_correction_seals_new_refs_into_server_only_canonical_configuration(db_session, test_project):
    from huddleroom.services.orchestration_recovery_configuration import CanonicalTeamConfigurationApplyV1

    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    payload = _configuration_arguments(agents=[{
        "operation": "create", "ref": "new:planner", "definition": {
            "name": "Recovery Planner", "role": "planning", "provider": "local", "model": "local",
            "adapter_type": "api", "capabilities": ["planning"], "config": {},
        },
    }])
    payload.update({
        "eligibility": [{"agent_ref": "new:planner", "work_functions": ["planning"]}],
        "assignments": [{"agent_ref": "new:planner", "work_function": "planning"}],
        "hierarchy": {
            "manager": {"kind": "agent", "agent_ref": "new:planner"}, "team_leads": ["new:planner"],
            "contributors": [], "reviewers": [], "validators": [], "specialists": [],
        },
    })
    prepared = await RecoveryConfigurationService().prepare(
        db_session, goal, run, TeamConfigurationApplyV1.model_validate(payload),
    )
    canonical = CanonicalTeamConfigurationApplyV1.model_validate(prepared.arguments)
    encoded = canonical.model_dump(mode="json")
    created_id = encoded["agents"][0]["agent_id"]

    assert encoded == CanonicalTeamConfigurationApplyV1.model_validate(encoded).model_dump(mode="json")
    assert "new:" not in json.dumps(encoded, sort_keys=True)
    assert "agent_ref" not in json.dumps(encoded, sort_keys=True)
    assert encoded["eligibility"][0]["agent_id"] == created_id
    assert encoded["assignments"][0]["agent_id"] == created_id
    assert encoded["hierarchy"]["manager"]["agent_id"] == created_id
    assert encoded["hierarchy"]["team_leads"] == [created_id]
    with pytest.raises(ValueError):
        TeamConfigurationApplyV1.model_validate(encoded)


@pytest.mark.asyncio
@pytest.mark.usefixtures("seal_without_post_state_validation")
async def test_c1_correction_uses_accepted_before_surfaces_and_omits_unchanged_ones(db_session, test_project):
    planner = _agent(PLANNER_ID, "Planner", "planning")
    reviewer = _agent(REVIEWER_ID, "Reviewer", "review")
    db_session.add_all([Agent(**planner.__dict__), Agent(**reviewer.__dict__)])
    await db_session.flush()
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    goal.authority_model = "agent_manager"
    goal.manager_agent_id = planner.id
    db_session.add_all([
        OrchestrationProcessRun(
            goal_id=goal.id, run_id=run.id, process_type="agent_definition_review", status="completed", trigger_reason="accepted",
            outputs={"eligibility": {str(planner.id): ["planning"], str(reviewer.id): ["review"]}},
        ),
        OrchestrationProcessRun(
            goal_id=goal.id, run_id=run.id, process_type="team_hierarchy", status="completed", trigger_reason="accepted",
            outputs={
                "assignments": [
                    {"work_function": "planning", "agent_ref": str(planner.id)},
                    {"work_function": "review", "agent_ref": str(reviewer.id)},
                ],
                "reporting_lines": [{"agent_ref": str(reviewer.id), "reports_to": str(planner.id)}],
                "documented_gaps": [{"code": "review_capacity", "rationale": "Existing accepted gap."}],
            },
        ),
    ])
    await db_session.flush()
    payload = _configuration_arguments(agents=[{
        "operation": "update", "agent_id": str(planner.id), "set": {"role": "lead planning"},
    }])
    payload.update({
        "eligibility": [
            {"agent_ref": str(planner.id), "work_functions": ["planning"]},
            {"agent_ref": str(reviewer.id), "work_functions": ["review"]},
        ],
        "assignments": [
            {"agent_ref": str(planner.id), "work_function": "planning"},
            {"agent_ref": str(reviewer.id), "work_function": "review"},
        ],
        "hierarchy": {"manager": {"kind": "agent", "agent_ref": str(planner.id)}, "team_leads": [str(planner.id)],
                      "contributors": [], "reviewers": [str(reviewer.id)], "validators": [], "specialists": []},
        "documented_gaps": [{"code": "review_capacity", "rationale": "Existing accepted gap."}],
    })
    prepared = await RecoveryConfigurationService().prepare(
        db_session, goal, run, TeamConfigurationApplyV1.model_validate(payload),
    )
    by_type = {change.target_type: change for change in prepared.changes}

    assert "eligibility" not in by_type
    assert "assignments" not in by_type
    assert "manager" not in by_type
    assert "documented_gaps" not in by_type
    reporting = by_type["reporting_hierarchy"]
    edge = [{"agent_ref": str(reviewer.id), "reports_to": str(planner.id)}]
    assert reporting.before == {
        "role_membership": {
            "contributors": [str(planner.id)], "reviewers": [str(reviewer.id)], "specialists": [],
            "team_leads": [], "validators": [],
        },
        "reporting_lines": edge,
    }
    assert reporting.after == {
        "role_membership": {
            "contributors": [], "reviewers": [str(reviewer.id)], "specialists": [],
            "team_leads": [str(planner.id)], "validators": [],
        },
        "reporting_lines": edge,
    }


@pytest.mark.asyncio
@pytest.mark.usefixtures("seal_without_post_state_validation")
async def test_c1_correction_semantic_redaction_keeps_benign_secretariat_text(db_session, test_project):
    planner = replace(
        _agent(PLANNER_ID, "Planner", "planning"),
        system_prompt="secretariat notes token=sk-live-secret stay private.",
    )
    db_session.add(Agent(**planner.__dict__))
    await db_session.flush()
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    prepared = await RecoveryConfigurationService().prepare(
        db_session, goal, run, TeamConfigurationApplyV1.model_validate(_configuration_arguments(agents=[{
            "operation": "update", "agent_id": str(planner.id), "set": {"role": "lead planning"},
        }])),
    )
    serialized = json.dumps([change.model_dump(mode="json") for change in prepared.changes], sort_keys=True)

    assert "sk-live-secret" not in serialized
    assert "[REDACTED]" in serialized
    assert "secretariat" in serialized


@pytest.mark.asyncio
@pytest.mark.usefixtures("seal_without_post_state_validation")
async def test_c1_correction_aggregates_large_complete_change_sets_within_changev1_bound(db_session, test_project):
    agents = tuple(_agent(UUID(int=index), f"Planner {index}", "planning") for index in range(10_000, 10_028))
    db_session.add_all(Agent(**agent.__dict__) for agent in agents)
    await db_session.flush()
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    prepared = await RecoveryConfigurationService().prepare(
        db_session,
        goal,
        run,
        TeamConfigurationApplyV1.model_validate(_configuration_arguments(agents=[
            {"operation": "update", "agent_id": str(agent.id), "set": {"role": "lead planning"}}
            for agent in agents
        ])),
    )
    encoded = [change.model_dump(mode="json") for change in prepared.changes]
    serialized = json.dumps(encoded, sort_keys=True)

    assert len(prepared.changes) <= 32
    assert all(str(agent.id) in serialized for agent in agents)
    for change in prepared.changes:
        ChangeV1.model_validate(change.model_dump(mode="json"))


@pytest.mark.asyncio
@pytest.mark.usefixtures("seal_without_post_state_validation")
async def test_c1_final_empty_accepted_reporting_lines_match_empty_member_groups(db_session, test_project):
    planner = _agent(PLANNER_ID, "Planner", "planning")
    db_session.add(Agent(**planner.__dict__))
    await db_session.flush()
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    db_session.add(OrchestrationProcessRun(
        goal_id=goal.id, run_id=run.id, process_type="team_hierarchy", status="completed", trigger_reason="accepted",
        outputs={"assignments": [], "reporting_lines": [], "documented_gaps": []},
    ))
    await db_session.flush()
    prepared = await RecoveryConfigurationService().prepare(
        db_session, goal, run, TeamConfigurationApplyV1.model_validate(_configuration_arguments(agents=[{
            "operation": "update", "agent_id": str(planner.id), "set": {"role": "lead planning"},
        }])),
    )

    assert "reporting_hierarchy" not in {change.target_type for change in prepared.changes}


@pytest.mark.asyncio
@pytest.mark.usefixtures("seal_without_post_state_validation")
async def test_c1_final_semantic_redaction_covers_nested_secret_key_values(db_session, test_project):
    secret = "sk-nested-secret"
    planner = replace(
        _agent(PLANNER_ID, "Planner", "planning"),
        config={"nested": [{"authorization": f"Bearer {secret}"}, {"token": secret}]},
    )
    db_session.add(Agent(**planner.__dict__))
    await db_session.flush()
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    prepared = await RecoveryConfigurationService().prepare(
        db_session, goal, run, TeamConfigurationApplyV1.model_validate(_configuration_arguments(agents=[{
            "operation": "update", "agent_id": str(planner.id), "set": {"role": "lead planning"},
        }])),
    )
    serialized = json.dumps([change.model_dump(mode="json") for change in prepared.changes], sort_keys=True)

    assert secret not in serialized
    assert "[REDACTED]" in serialized


def test_c1_final_canonical_configuration_rejects_empty_or_unknown_trusted_fields():
    from huddleroom.services.orchestration_recovery_configuration import CanonicalTeamConfigurationApplyV1

    agent_id = str(PLANNER_ID)
    base = {
        "agents": [],
        "eligibility": [{"agent_id": agent_id, "work_functions": ["planning"]}],
        "assignments": [{"agent_id": agent_id, "work_function": "planning"}],
        "hierarchy": {"manager": {"kind": "none"}, "team_leads": [], "contributors": [], "reviewers": [], "validators": [], "specialists": []},
        "documented_gaps": [{"code": "review_capacity", "rationale": "Documented gap."}],
    }
    invalid = (
        {**base, "eligibility": [{"agent_id": agent_id, "work_functions": []}]},
        {**base, "eligibility": [{"agent_id": agent_id, "work_functions": ["unknown"]}]},
        {**base, "assignments": [{"agent_id": agent_id, "work_function": "unknown"}]},
        {**base, "documented_gaps": [{"code": "", "rationale": ""}]},
    )

    for payload in invalid:
        with pytest.raises(ValueError):
            CanonicalTeamConfigurationApplyV1.model_validate(payload)


def test_c1_final_single_oversize_change_is_normalized_to_code_owned_failure():
    from huddleroom.services.orchestration_recovery_configuration import RecoveryConfigurationService

    with pytest.raises(ValueError, match="change_set_too_large"):
        RecoveryConfigurationService._chunked_changes(
            "agent", "global", [(PLANNER_ID, {}, {"description": "x" * 20_000})],
        )


def test_c1_final_later_oversize_change_is_normalized_to_code_owned_failure():
    from huddleroom.services.orchestration_recovery_configuration import RecoveryConfigurationService

    with pytest.raises(ValueError, match="change_set_too_large"):
        RecoveryConfigurationService._chunked_changes("agent", "global", [
            (UUID(int=1), {}, {"description": "ok"}),
            (UUID(int=2), {}, {"description": "x" * 20_000}),
        ])


@pytest.mark.asyncio
@pytest.mark.parametrize("subject_type", ("task", "artifact"))
async def test_c2_start_graph_requires_active_project_or_global_graph_before_sealing(db_session, test_project, subject_type):
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    task = Task(
        project_id=test_project.id, title="Graph subject", assigned_to=None,
        metadata_={"orchestration": {"run_id": str(run.id), "work_function": "implementation"}},
    )
    foreign = Graph(project_id=UUID(int=99_999), name="foreign", version="1", definition={}, triggers=[], is_active=True)
    inactive = Graph(project_id=test_project.id, name="inactive", version="1", definition={}, triggers=[], is_active=False)
    artifact = Artifact(project_id=test_project.id, name="plan.md", artifact_type="plan", status="ready", content_hash=SHA256)
    db_session.add_all([task, artifact, foreign, inactive])
    await db_session.flush()

    for graph_id in (UUID(int=99_998), foreign.id, inactive.id):
        with pytest.raises(ValueError, match="out_of_scope_target"):
            await RecoveryToolRegistry().canonicalize(
                db_session,
                goal,
                run,
                _trigger(goal.id, run.id),
                _tool_draft("orchestration.start_graph@1", {
                    "graph_id": str(graph_id), "subject_type": subject_type,
                    "subject_id": str(task.id if subject_type == "task" else artifact.id),
                }),
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("subject_type", ("task", "artifact"))
@pytest.mark.parametrize("global_graph", (False, True))
async def test_c2_start_graph_accepts_active_project_and_global_graph_for_project_subject(
    db_session, test_project, subject_type, global_graph,
):
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    graph = Graph(project_id=None if global_graph else test_project.id, name="active", version="1",
                  definition={}, triggers=[], is_active=True)
    gate = OrchestrationGate(run_id=run.id, success_criterion_key="plan", gate_type="plan", status="open")
    db_session.add_all([graph, gate])
    await db_session.flush()
    task = Task(project_id=test_project.id, title="Graph subject",
                metadata_={"orchestration": {"run_id": str(run.id), "gate_id": str(gate.id)}})
    db_session.add(task)
    await db_session.flush()
    subject = (task if subject_type == "task" else
               Artifact(project_id=test_project.id, name="plan.md", artifact_type="plan", status="ready",
                        content_hash=SHA256, linked_task_id=task.id))
    db_session.add(subject)
    await db_session.flush()
    draft = _tool_draft("orchestration.start_graph@1", {
        "graph_id": str(graph.id), "subject_type": subject_type, "subject_id": str(subject.id),
    }).model_copy(update={"general_updates": ["start graph"]})
    proposal = await RecoveryToolRegistry().canonicalize(db_session, goal, run, _trigger(goal.id, run.id), draft)
    assert proposal.calls[0].arguments["graph_id"] == str(graph.id)


@pytest.mark.asyncio
async def test_c2_summary_requires_action_identity_then_broker_derives_public_effect_bullets(db_session, test_project):
    planner = _agent(PLANNER_ID, "Recovery Planner", "planning")
    db_session.add(Agent(**planner.__dict__))
    await db_session.flush()
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    arguments = {"agent_id": str(planner.id), "work_function": "planning", "scope": "goal"}
    with pytest.raises(ValueError, match="summary_missing_material_effect"):
        await RecoveryToolRegistry().canonicalize(
            db_session, goal, run, _trigger(goal.id, run.id), _tool_draft("orchestration.request_plan@1", arguments),
        )


@pytest.mark.asyncio
async def test_c2_non_global_plan_captures_referenced_agent_and_run_state_drift(db_session, test_project):
    planner = _agent(PLANNER_ID, "Recovery Planner", "planning")
    stored_planner = Agent(**planner.__dict__)
    db_session.add(stored_planner)
    await db_session.flush()
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    run.plan_state = {"stage": "waiting"}
    run.retry_state = {"attempt": 0}
    draft = _tool_draft("orchestration.request_plan@1", {
        "agent_id": str(planner.id), "work_function": "planning", "scope": "goal",
    }).model_copy(update={"general_updates": ["orchestration request plan"]})
    registry = RecoveryToolRegistry()
    proposal = await registry.canonicalize(db_session, goal, run, _trigger(goal.id, run.id), draft)
    before = await registry.capture_resources(db_session, goal, run, proposal)

    assert f"agent:{planner.id}" in before.values
    assert before.values[f"run:{run.id}"]["plan_state"] == {"stage": "waiting"}
    stored_planner.role = "lead planning"
    run.retry_state = {"attempt": 1}
    await db_session.flush()
    after = await registry.capture_resources(db_session, goal, run, proposal)
    assert after.fingerprint != before.fingerprint


@pytest.mark.asyncio
async def test_c2_global_capture_redacts_retained_process_and_task_secrets_before_hashing(db_session, test_project):
    secret = "sk-c2-retained-secret"
    planner = replace(
        _agent(PLANNER_ID, "Planner", "planning"), system_prompt=f"Keep token={secret} private.",
        config={"mode": f"Keep token={secret} private."},
    )
    db_session.add(Agent(**planner.__dict__))
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    task = Task(
        project_id=test_project.id, title="Secret task", assigned_to=planner.id,
        metadata_={"orchestration": {"run_id": str(run.id), "work_function": f"token={secret}"},
                   "ordinary": f"token={secret}"},
    )
    process = OrchestrationProcessRun(
        goal_id=goal.id, run_id=run.id, process_type="team_hierarchy", status="completed", trigger_reason="accepted",
        input_snapshot={"fingerprint": SHA256, "ordinary": f"token={secret}"},
        outputs={"analysis": {"note": f"Keep token={secret} private."}, "ordinary": f"token={secret}"},
    )
    db_session.add_all([task, process])
    await db_session.flush()
    proposal = _proposal().model_copy(update={"global_effect": True})
    snapshot = await RecoveryToolRegistry().capture_resources(db_session, goal, run, proposal)
    published = json.dumps(snapshot.values, sort_keys=True)

    assert secret not in published
    assert "[REDACTED]" in published
    assert "input_snapshot" not in published
    assert "ordinary" not in published
    assert "metadata" not in snapshot.values[f"task:{task.id}"]
    assert snapshot.values[f"agent:{planner.id}"]["config"]["mode"] == "Keep token=[REDACTED] private."
    assert snapshot.values[f"agent:{planner.id}"]["system_prompt"] == "Keep token=[REDACTED] private."
    assert snapshot.values[f"process:team_hierarchy:{process.id}"]["analysis_fingerprint"] == hashlib.sha256(
        json.dumps({"note": "Keep token=[REDACTED] private."}, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()
    assert "[REDACTED]" in json.dumps(snapshot.values[f"task:{task.id}"])
    assert snapshot.fingerprint == hashlib.sha256(
        json.dumps(snapshot.values, sort_keys=True, separators=(",", ":"), default=str).encode(),
    ).hexdigest()


async def _c2_complete_configuration_draft(db_session, test_project):
    db_session.add(Agent(**_agent(PLANNER_ID, "Planner", "planning").__dict__))
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    collateral_goal, _ = await _seed_goal_run(db_session, test_project, objective="Review the release.")
    arguments = _configuration_arguments(agents=[
        {"operation": "update", "agent_id": str(PLANNER_ID), "set": {"role": "lead planning"}},
        {"operation": "create", "ref": "new:reviewer", "definition": {
            "name": "Recovery Reviewer", "role": "review", "provider": "local", "model": "local",
            "adapter_type": "api", "capabilities": ["review"], "config": {},
        }},
    ])
    arguments.update({
        "eligibility": [{"agent_ref": str(PLANNER_ID), "work_functions": ["planning"]},
                        {"agent_ref": "new:reviewer", "work_functions": ["review"]}],
        "assignments": [{"agent_ref": str(PLANNER_ID), "work_function": "planning"},
                        {"agent_ref": "new:reviewer", "work_function": "review"}],
        "hierarchy": {"manager": {"kind": "agent", "agent_ref": str(PLANNER_ID)},
                      "team_leads": [str(PLANNER_ID)], "contributors": [], "reviewers": ["new:reviewer"],
                      "validators": [], "specialists": []},
        "documented_gaps": [{"code": "review_capacity", "rationale": "Disclose the accepted review gap."},
                            {"code": "validation_override", "rationale": "Disclose the accepted validation override."}],
    })
    bullets = ["agent definitions", "eligibility", "assignments", "manager", "reporting hierarchy",
               "documented gaps", "Planner", "new:reviewer", "review_capacity", "validation_override",
               "apply team configuration", "request plan"]
    payload = _draft("team_configuration.apply@1").model_dump(mode="json")
    payload.update({
        "general_updates": bullets,
        "calls": [
            {"tool": "team_configuration.apply@1", "arguments": arguments, "rationale": "Repair the global roster."},
            {"tool": "orchestration.request_plan@1", "arguments": {
                "agent_id": str(PLANNER_ID), "work_function": "planning", "scope": "goal",
            }, "rationale": "Request the repaired plan."},
        ],
    })
    return goal, run, collateral_goal, ProposalDraftV1.model_validate(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("omitted", (
    "agent definitions", "eligibility", "assignments", "manager", "reporting hierarchy", "documented gaps",
    "Planner", "new:reviewer", "review_capacity", "validation_override", "apply team configuration", "request plan",
    "generic_agent", "generic_orchestration",
))
@pytest.mark.usefixtures("seal_without_post_state_validation")
async def test_c2_complete_configuration_summary_rejects_each_missing_surface_identity_and_action(
    db_session, test_project, omitted,
):
    goal, run, _, draft = await _c2_complete_configuration_draft(db_session, test_project)
    bullets = (["agent configuration_fingerprint"] if omitted == "generic_agent" else
               ["orchestration"] if omitted == "generic_orchestration" else
               [bullet for bullet in draft.general_updates if bullet != omitted])
    with pytest.raises(ValueError, match="summary_missing_material_effect"):
        await RecoveryToolRegistry().canonicalize(
            db_session, goal, run, _trigger(goal.id, run.id), draft.model_copy(update={"general_updates": bullets}),
        )


@pytest.mark.asyncio
@pytest.mark.usefixtures("seal_without_post_state_validation")
async def test_c2_complete_configuration_derives_bounded_public_global_verification_and_retry_bullets(
    db_session, test_project,
):
    goal, run, collateral, draft = await _c2_complete_configuration_draft(db_session, test_project)
    proposal = await RecoveryToolRegistry().canonicalize(db_session, goal, run, _trigger(goal.id, run.id), draft)
    bullets = [bullet.lower() for bullet in proposal.general_updates]
    public = " ".join(bullets)

    assert proposal.global_effect
    assert proposal.general_updates != draft.general_updates
    assert len(bullets) <= 16
    assert any("global" in bullet and "roster" in bullet for bullet in bullets)
    assert any("2" in bullet and "goal" in bullet for bullet in bullets)
    assert "review_capacity" in public and "validation_override" in public
    assert any("team configuration accepted" in bullet and "action completed" in bullet for bullet in bullets)
    assert len([bullet for bullet in bullets if "retry" in bullet]) == 1
    assert "planning" in next(bullet for bullet in bullets if "retry" in bullet)
    assert proposal.retry.max_attempts == 1
    generated = [proposal.proposal_id, proposal.lineage_id, collateral.id]
    generated.extend(UUID(item["agent_id"]) for item in proposal.calls[0].arguments["agents"] if item["operation"] == "create")
    assert all(str(identifier) not in public for identifier in generated)


def _c2_resource_proposal(tool, arguments, *, global_effect=False):
    data = _proposal().model_dump(mode="json")
    data.update({"global_effect": global_effect, "calls": [{
        "call_id": "call-1", "tool": tool, "tool_version": "1", "arguments": arguments,
        "changes": [{"operation": "request", "scope": "run", "target_type": tool,
                     "target_id": str(GOAL_A_ID), "after": arguments}],
    }]})
    return RecoveryProposalV1.model_validate(data)


@pytest.mark.parametrize("sealed", (False, True))
def test_c2_team_configuration_envelopes_allow_only_typed_documented_gap_code(sealed):
    arguments = _configuration_arguments(agents=[])
    arguments["documented_gaps"] = [{"code": "review_capacity", "rationale": "Disclose the accepted gap."}]
    if sealed:
        call = RecoveryCallV1.model_validate({
            "call_id": "call-1", "tool": "team_configuration.apply", "tool_version": "1", "arguments": arguments,
            "changes": [{"operation": "update", "scope": "goal", "target_type": "documented_gaps",
                         "target_id": str(GOAL_A_ID), "after": {"gaps": ["review_capacity"]}}],
        })
        assert call.arguments["documented_gaps"][0]["code"] == "review_capacity"
    else:
        draft = _tool_draft("team_configuration.apply@1", arguments)
        assert draft.calls[0].arguments["documented_gaps"][0]["code"] == "review_capacity"


@pytest.mark.parametrize("sealed", (False, True))
@pytest.mark.parametrize("unsafe_location", ("root", "nested", "agent_config", "other_tool", "gap_command", "wrong_gap_shape"))
def test_c2_documented_gap_exception_keeps_other_executable_code_paths_rejected(sealed, unsafe_location):
    arguments = _configuration_arguments(agents=[])
    arguments["documented_gaps"] = [{"code": "review_capacity", "rationale": "Disclose the accepted gap."}]
    tool = "team_configuration.apply"
    if unsafe_location == "root":
        arguments["code"] = "print('unsafe')"
    elif unsafe_location == "nested":
        arguments["nested"] = {"code": "print('unsafe')"}
    elif unsafe_location == "agent_config":
        arguments["agents"] = [{"operation": "create", "ref": "new:unsafe", "definition": {
            "name": "Unsafe", "role": "planning", "provider": "local", "model": "local",
            "adapter_type": "api", "config": {"code": "print('unsafe')"},
        }}]
    elif unsafe_location == "other_tool":
        tool = "orchestration.request_plan"
    elif unsafe_location == "gap_command":
        arguments["documented_gaps"][0]["command"] = "execute unsafe"
    else:
        arguments["documented_gaps"] = {"code": "print('unsafe')"}
    with pytest.raises(ValueError, match="predicate/query or executable"):
        if sealed:
            RecoveryCallV1.model_validate({
                "call_id": "call-1", "tool": tool, "tool_version": "1", "arguments": arguments,
                "changes": [{"operation": "request", "scope": "goal", "target_type": tool,
                             "target_id": str(GOAL_A_ID), "after": {"request": "repair"}}],
            })
        else:
            _tool_draft(f"{tool}@1", arguments)


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool", "target"), (
    ("orchestration.request_plan", "agent"), ("orchestration.request_roadmap_replan", "agent"),
    ("orchestration.request_plan_revision", "task"), ("orchestration.retry_task", "task"),
    ("orchestration.reassign_task", "agent"), ("orchestration.reassign_task", "task"),
    ("orchestration.request_verification", "gate"),
    ("orchestration.schedule_meeting", "participant"), ("orchestration.schedule_meeting", "organizer"),
    ("orchestration.schedule_meeting", "task"), ("orchestration.schedule_meeting", "gate"),
    ("orchestration.start_graph", "graph"), ("orchestration.start_graph", "task"),
    ("orchestration.start_graph", "artifact"),
))
async def test_c2_each_tool_captures_exact_referenced_rows_and_independent_drift(db_session, test_project, tool, target):
    participant = Agent(**_agent(PLANNER_ID, "Planner", "planning").__dict__)
    organizer = Agent(**_agent(REVIEWER_ID, "Organizer", "review").__dict__)
    db_session.add_all([participant, organizer])
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    task = Task(project_id=test_project.id, title="Referenced task", assigned_to=participant.id,
                metadata_={"orchestration": {"run_id": str(run.id), "work_function": "planning"}})
    gate = OrchestrationGate(run_id=run.id, success_criterion_key="plan", gate_type="plan", status="open")
    graph = Graph(project_id=test_project.id, name="approval graph", version="1", definition={"nodes": []},
                  triggers=[], is_active=True)
    artifact = Artifact(project_id=test_project.id, name="plan.md", artifact_type="plan", status="ready", content_hash=SHA256)
    db_session.add_all([task, gate, graph, artifact])
    await db_session.flush()
    if tool in {"orchestration.schedule_meeting", "orchestration.start_graph"}:
        task.metadata_ = {"orchestration": {"run_id": str(run.id), "work_function": "planning", "gate_id": str(gate.id)}}
        artifact.linked_task_id = task.id
        await db_session.flush()
    arguments = {
        "orchestration.request_plan": {"agent_id": str(participant.id), "work_function": "planning", "scope": "goal"},
        "orchestration.request_roadmap_replan": {"agent_id": str(participant.id), "scope": "goal", "reason": "Repair roadmap."},
        "orchestration.request_plan_revision": {"plan_task_id": str(task.id), "revision_request": "Repair plan."},
        "orchestration.retry_task": {"task_id": str(task.id)},
        "orchestration.reassign_task": {"task_id": str(task.id), "agent_id": str(participant.id)},
        "orchestration.request_verification": {"gate_id": str(gate.id), "work_function": "planning"},
        "orchestration.schedule_meeting": {"topic": "Repair", "participant_agent_ids": [str(participant.id), str(organizer.id)],
                                           "organizer_agent_id": str(organizer.id), "task_id": str(task.id), "gate_id": str(gate.id)},
        "orchestration.start_graph": {"graph_id": str(graph.id), "subject_type": "artifact" if target == "artifact" else "task",
                                      "subject_id": str(artifact.id) if target == "artifact" else str(task.id)},
    }[tool]
    proposal = _c2_resource_proposal(tool, arguments)
    registry = RecoveryToolRegistry()
    before = await registry.capture_resources(db_session, goal, run, proposal)
    rows = {"agent": participant, "participant": participant, "organizer": organizer,
            "task": task, "gate": gate, "graph": graph, "artifact": artifact}
    row = rows[target]
    prefix = "agent" if target in {"participant", "organizer"} else target
    key = f"{prefix}:{row.id}"
    expected = {"registry", f"run:{run.id}", f"goal_contract:{goal.id}"}
    for name in ("agent_id", "organizer_agent_id"):
        if name in arguments:
            expected.add(f"agent:{arguments[name]}")
    expected.update(f"agent:{item}" for item in arguments.get("participant_agent_ids", []))
    expected.update(f"task:{arguments[name]}" for name in ("task_id", "plan_task_id") if name in arguments)
    if "gate_id" in arguments:
        expected.add(f"gate:{arguments['gate_id']}")
    if "graph_id" in arguments:
        expected.update((f"graph:{graph.id}", f"{arguments['subject_type']}:{arguments['subject_id']}",
                         f"task:{task.id}", f"gate:{gate.id}"))
    assert set(before.values) == expected
    assert key in before.values
    if prefix == "agent":
        row.role = "changed role"
    elif target in {"task", "gate"}:
        row.status = "done" if target == "task" else "accepted"
    elif target == "graph":
        row.definition = {"nodes": [{"name": "repair"}]}
    else:
        row.content_hash = "b" * 64
    await db_session.flush()
    after = await registry.capture_resources(db_session, goal, run, proposal)
    assert after.values[key] != before.values[key]
    assert after.fingerprint != before.fingerprint


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ("orchestration.request_plan", "orchestration.request_roadmap_replan"))
@pytest.mark.parametrize("field", ("plan_state", "retry_state"))
async def test_c2_non_global_planning_captures_plan_and_retry_drift_independently(db_session, test_project, tool, field):
    db_session.add(Agent(**_agent(PLANNER_ID, "Planner", "planning").__dict__))
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    run.plan_state = {"stage": "waiting"}
    run.retry_state = {"attempt": 0}
    await db_session.flush()
    arguments = {"agent_id": str(PLANNER_ID), "scope": "goal"}
    arguments.update({"work_function": "planning"} if tool.endswith("request_plan") else {"reason": "Repair roadmap."})
    proposal = _c2_resource_proposal(tool, arguments)
    registry = RecoveryToolRegistry()
    before = await registry.capture_resources(db_session, goal, run, proposal)
    assert before.values[f"run:{run.id}"][field] == getattr(run, field)
    setattr(run, field, {"stage": "requested"} if field == "plan_state" else {"attempt": 1})
    await db_session.flush()
    after = await registry.capture_resources(db_session, goal, run, proposal)
    assert after.values[f"run:{run.id}"][field] != before.values[f"run:{run.id}"][field]
    assert after.fingerprint != before.fingerprint


@pytest.mark.asyncio
@pytest.mark.parametrize("category", (
    "goal_contract", "active_run", "definition_fingerprint", "definition_targets", "hierarchy_fingerprint",
    "hierarchy_analysis", "answered_exclusion", "exclusion_definition", "assignment", "task_linkage",
    "task_work_function", "roster", "authority_delivery", "active_goal_membership",
))
async def test_c2_global_configuration_captures_cross_project_preview_categories_and_independent_drift(
    db_session, test_project, category,
):
    planner = Agent(**_agent(PLANNER_ID, "Planner", "planning").__dict__)
    reviewer = Agent(**_agent(REVIEWER_ID, "Reviewer", "review").__dict__)
    foreign_project = Project(name="Collateral project")
    db_session.add_all([planner, reviewer, foreign_project])
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    other_goal, other_run = await _seed_goal_run(db_session, foreign_project, objective="Review another project.")
    other_run.plan_state = {"stage": "waiting"}
    other_run.retry_state = {"attempt": 0}
    definition = OrchestrationProcessRun(
        goal_id=other_goal.id, run_id=other_run.id, process_type="agent_definition_review", status="completed",
        trigger_reason="accepted", input_snapshot={"fingerprint": SHA256}, outputs={"target_ids": [str(planner.id)]},
    )
    hierarchy = OrchestrationProcessRun(
        goal_id=other_goal.id, run_id=other_run.id, process_type="team_hierarchy", status="completed",
        trigger_reason="accepted", input_snapshot={"fingerprint": SHA256}, outputs={"analysis": {"manager": "human"}},
    )
    assignment = Task(
        project_id=foreign_project.id, title="Collateral planning assignment", assigned_to=planner.id,
        metadata_={"orchestration": {"run_id": str(other_run.id), "work_function": "planning"}},
    )
    delivery = OrchestrationAction(
        run_id=other_run.id, action_type="request_plan", status="completed", request={},
        idempotency_key="authority_decision_delegate:test-coverage",
    )
    db_session.add_all([definition, hierarchy, assignment, delivery])
    await db_session.flush()
    delivery_task = Task(
        project_id=foreign_project.id, title="Authority delivery assignment", assigned_to=reviewer.id,
        metadata_={"orchestration": {"run_id": str(other_run.id), "work_function": "review", "action_id": str(delivery.id)}},
    )
    exclusion_definition = {
        "name": "Rejected Reviewer", "role": "review", "provider": "local", "model": "local", "adapter_type": "api",
    }
    authority = OrchestrationAuthorityDecision(
        goal_id=other_goal.id, run_id=other_run.id, source_process_run_id=hierarchy.id,
        decision_key="team_hierarchy:agent:rejected-reviewer", title="Exclude the proposed reviewer.",
        question="Accept this reviewer?", status="answered", authority="human", selected_option="reject",
        options=[{"key": option, "proposed_agent": {"definition": exclusion_definition}} for option in ("edit", "reject")],
    )
    db_session.add_all([authority, delivery_task])
    await db_session.flush()
    proposal = _c2_resource_proposal("team_configuration.apply", _configuration_arguments(agents=[]), global_effect=True)
    registry = RecoveryToolRegistry()
    before = await registry.capture_resources(db_session, goal, run, proposal)
    required = {
        f"goal_contract:{other_goal.id}", f"run:{other_run.id}", f"agent:{planner.id}", f"agent:{reviewer.id}",
        f"task:{assignment.id}", f"task:{delivery_task.id}", f"process:agent_definition_review:{definition.id}",
        f"process:team_hierarchy:{hierarchy.id}", f"authority_decision:{authority.id}", f"action:{delivery.id}",
    }
    assert required <= set(before.values)
    assert before.values[f"process:agent_definition_review:{definition.id}"]["fingerprint"] == SHA256
    assert before.values[f"process:team_hierarchy:{hierarchy.id}"]["fingerprint"] == SHA256
    assert before.values[f"authority_decision:{authority.id}"]["selected_option"] == "reject"
    assert "input_snapshot" not in before.values[f"process:team_hierarchy:{hierarchy.id}"]
    assert "outputs" not in before.values[f"process:team_hierarchy:{hierarchy.id}"]
    assert "options" not in before.values[f"authority_decision:{authority.id}"]
    assert "metadata" not in before.values[f"task:{assignment.id}"]
    if category == "goal_contract":
        other_goal.constraints = {"deadline": "tomorrow"}
    elif category == "active_run":
        other_run.retry_state = {"attempt": 1}
    elif category == "definition_fingerprint":
        definition.input_snapshot = {"fingerprint": "b" * 64}
    elif category == "definition_targets":
        definition.outputs = {"target_ids": [str(reviewer.id)]}
    elif category == "hierarchy_fingerprint":
        hierarchy.input_snapshot = {"fingerprint": "b" * 64}
    elif category == "hierarchy_analysis":
        hierarchy.outputs = {"analysis": {"manager": str(reviewer.id)}}
    elif category == "answered_exclusion":
        authority.selected_option = "edit"
    elif category == "exclusion_definition":
        authority.options = [{"key": option, "proposed_agent": {"definition": {
            **exclusion_definition, "name": "Rejected Specialist",
        }}} for option in ("edit", "reject")]
    elif category == "assignment":
        assignment.assigned_to = reviewer.id
    elif category == "task_linkage":
        assignment.metadata_ = {"orchestration": {"run_id": str(run.id), "work_function": "planning"}}
    elif category == "task_work_function":
        assignment.metadata_ = {"orchestration": {"run_id": str(other_run.id), "work_function": "review"}}
    elif category == "roster":
        reviewer.description = "Changed global roster definition."
    elif category == "authority_delivery":
        delivery.idempotency_key = "ordinary:test-coverage"
    else:
        added_goal, added_run = await _seed_goal_run(db_session, foreign_project, objective="New active collateral goal.")
    await db_session.flush()
    after = await registry.capture_resources(db_session, goal, run, proposal)
    assert after.fingerprint != before.fingerprint
    if category == "active_goal_membership":
        assert f"goal_contract:{added_goal.id}" in after.values
        assert f"run:{added_run.id}" in after.values


@pytest.mark.asyncio
async def test_c2_global_configuration_captures_accepted_review_functions_consumed_by_trivial_hierarchy(
    db_session, test_project,
):
    planner = Agent(**_agent(PLANNER_ID, "Planner", "planning").__dict__)
    collateral_project = Project(name="Trivial collateral project")
    db_session.add_all([planner, collateral_project])
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    collateral_goal, collateral_run = await _seed_goal_run(
        db_session, collateral_project, objective="Deliver one small result.",
    )
    collateral_goal.weight = "trivial"
    definition = OrchestrationProcessRun(
        goal_id=collateral_goal.id, run_id=collateral_run.id, process_type="agent_definition_review",
        status="completed", trigger_reason="accepted", input_snapshot={"fingerprint": SHA256}, outputs={},
    )
    db_session.add(definition)
    await db_session.flush()
    review = OrchestrationAgentReview(
        goal_id=collateral_goal.id, run_id=collateral_run.id, agent_id=planner.id,
        source_process_run_id=definition.id, approved_for_work_functions=["planning"],
        proposed_work_functions=["planning"], fit_summary="Approved planning coverage.",
        definition_snapshot={"name": "Planner", "system_prompt": "token=sk-private-review-snapshot"},
    )
    db_session.add(review)
    await db_session.flush()
    definition.outputs = {"target_ids": [str(planner.id)], "review_ids": [str(review.id)]}
    await db_session.flush()
    registry = RecoveryToolRegistry()
    proposal = _c2_resource_proposal("team_configuration.apply", _configuration_arguments(agents=[]), global_effect=True)
    before = await registry.capture_resources(db_session, goal, run, proposal)
    key = f"review:{review.id}"
    assert before.values[key]["approved_for_work_functions"] == ["planning"]
    assert before.values[key]["source_process_run_id"] == str(definition.id)
    assert "definition_snapshot" not in json.dumps(before.values, sort_keys=True)
    assert "sk-private-review-snapshot" not in json.dumps(before.values, sort_keys=True)
    review.approved_for_work_functions = ["planning", "review"]
    await db_session.flush()
    after = await registry.capture_resources(db_session, goal, run, proposal)
    assert after.values[key]["approved_for_work_functions"] == ["planning", "review"]
    assert after.fingerprint != before.fingerprint


async def _c2_coordination_rows(db_session, test_project):
    planner = Agent(**_agent(PLANNER_ID, "Planner", "planning").__dict__)
    organizer = Agent(**_agent(REVIEWER_ID, "Organizer", "review").__dict__)
    db_session.add_all([planner, organizer])
    goal, run = await _seed_goal_run(db_session, test_project, objective="Coordinate the release.")
    _, foreign_run = await _seed_goal_run(db_session, test_project, objective="Another run.")
    gate = OrchestrationGate(run_id=run.id, success_criterion_key="plan", gate_type="plan", status="open")
    alternate_gate = OrchestrationGate(run_id=run.id, success_criterion_key="review", gate_type="review", status="open")
    foreign_gate = OrchestrationGate(run_id=foreign_run.id, success_criterion_key="other", gate_type="plan", status="open")
    graph = Graph(project_id=test_project.id, name="coordination", version="1", definition={}, triggers=[], is_active=True)
    db_session.add_all([gate, alternate_gate, foreign_gate, graph])
    await db_session.flush()
    task = Task(project_id=test_project.id, title="Coordination task", assigned_to=planner.id,
                metadata_={"orchestration": {"run_id": str(run.id), "gate_id": str(gate.id), "work_function": "planning"}})
    db_session.add(task)
    await db_session.flush()
    artifact = Artifact(project_id=test_project.id, name="coordination.md", artifact_type="plan", status="ready",
                        content_hash=SHA256, linked_task_id=task.id)
    db_session.add(artifact)
    await db_session.flush()
    return goal, run, {"planner": planner, "organizer": organizer, "task": task, "gate": gate,
                       "alternate_gate": alternate_gate, "foreign_run": foreign_run, "foreign_gate": foreign_gate,
                       "graph": graph, "artifact": artifact}


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool_kind", "invalid"), (
    ("meeting", "organizer"), ("meeting", "task_run"), ("meeting", "task_gate"),
    ("graph_task", "task_run"), ("graph_task", "no_gate"), ("graph_task", "foreign_gate"),
    ("graph_artifact", "no_link"), ("graph_artifact", "missing_task"), ("graph_artifact", "task_run"),
    ("graph_artifact", "no_gate"), ("graph_artifact", "foreign_gate"),
))
async def test_c2_coordination_scope_rejects_executor_invalid_meeting_and_graph_relationships(
    db_session, test_project, tool_kind, invalid,
):
    goal, run, rows = await _c2_coordination_rows(db_session, test_project)
    task, gate, artifact = rows["task"], rows["gate"], rows["artifact"]
    if invalid == "task_run":
        task.metadata_ = {"orchestration": {"run_id": str(rows["foreign_run"].id), "gate_id": str(rows["foreign_gate"].id)}}
    elif invalid == "no_gate":
        task.metadata_ = {"orchestration": {"run_id": str(run.id)}}
    elif invalid == "foreign_gate":
        task.metadata_ = {"orchestration": {"run_id": str(run.id), "gate_id": str(rows["foreign_gate"].id)}}
    elif invalid in {"no_link", "missing_task"}:
        artifact.linked_task_id = None if invalid == "no_link" else UUID(int=99_999)
    await db_session.flush()
    if tool_kind == "meeting":
        tool, label = "orchestration.schedule_meeting@1", "schedule meeting"
        arguments = {"topic": "Repair", "participant_agent_ids": [str(rows["planner"].id)], "task_id": str(task.id),
                     "gate_id": str(rows["alternate_gate"].id if invalid == "task_gate" else gate.id),
                     "organizer_agent_id": str(rows["organizer"].id if invalid == "organizer" else rows["planner"].id)}
    else:
        tool, label = "orchestration.start_graph@1", "start graph"
        arguments = {"graph_id": str(rows["graph"].id), "subject_type": "task" if tool_kind == "graph_task" else "artifact",
                     "subject_id": str(task.id if tool_kind == "graph_task" else artifact.id)}
    draft = _tool_draft(tool, arguments).model_copy(update={"general_updates": [label]})
    with pytest.raises(ValueError, match="out_of_scope_target"):
        await RecoveryToolRegistry().canonicalize(db_session, goal, run, _trigger(goal.id, run.id), draft)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ("gate_id", "plan_item_gate_id", "orchestration_plan_item"))
async def test_c2_graph_task_captures_each_runtime_task_gate_source_and_linkage_drift(db_session, test_project, source):
    goal, run, rows = await _c2_coordination_rows(db_session, test_project)
    task, gate, alternate = rows["task"], rows["gate"], rows["alternate_gate"]

    def linked_metadata(gate_id):
        if source == "orchestration_plan_item":
            return {"orchestration": {"run_id": str(run.id)},
                    "orchestration_plan_item": {"plan_item_gate_id": str(gate_id)}}
        return {"orchestration": {"run_id": str(run.id), source: str(gate_id)}}

    task.metadata_ = linked_metadata(gate.id)
    await db_session.flush()
    draft = _tool_draft("orchestration.start_graph@1", {
        "graph_id": str(rows["graph"].id), "subject_type": "task", "subject_id": str(task.id),
    }).model_copy(update={"general_updates": ["start graph"]})
    registry = RecoveryToolRegistry()
    proposal = await registry.canonicalize(db_session, goal, run, _trigger(goal.id, run.id), draft)
    before = await registry.capture_resources(db_session, goal, run, proposal)
    assert f"gate:{gate.id}" in before.values
    task.metadata_ = linked_metadata(alternate.id)
    await db_session.flush()
    after = await registry.capture_resources(db_session, goal, run, proposal)
    assert f"gate:{alternate.id}" in after.values
    assert after.values[f"task:{task.id}"] != before.values[f"task:{task.id}"]
    assert after.fingerprint != before.fingerprint


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ("blockers", "task", "gate"))
async def test_c2_implicit_meeting_captures_blockers_and_derived_task_gate_drift(db_session, test_project, changed):
    goal, run, rows = await _c2_coordination_rows(db_session, test_project)
    task, gate = rows["task"], rows["gate"]
    run.active_blockers = [{"kind": "task_blocked", "task_id": str(task.id)}]
    await db_session.flush()
    draft = _tool_draft("orchestration.schedule_meeting@1", {
        "topic": "Repair", "participant_agent_ids": [str(rows["planner"].id)],
        "organizer_agent_id": str(rows["planner"].id),
    }).model_copy(update={"general_updates": ["schedule meeting"]})
    registry = RecoveryToolRegistry()
    proposal = await registry.canonicalize(db_session, goal, run, _trigger(goal.id, run.id), draft)
    before = await registry.capture_resources(db_session, goal, run, proposal)
    assert before.values[f"run:{run.id}"]["active_blockers"] == run.active_blockers
    assert f"task:{task.id}" in before.values and f"gate:{gate.id}" in before.values
    if changed == "blockers":
        run.active_blockers = [{"kind": "task_escalated", "task_id": str(task.id)}]
    elif changed == "task":
        task.status = "done"
    else:
        gate.status = "accepted"
    await db_session.flush()
    after = await registry.capture_resources(db_session, goal, run, proposal)
    assert after.fingerprint != before.fingerprint


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ("linked_task", "task", "gate"))
async def test_c2_artifact_graph_captures_linked_task_and_gate_with_independent_drift(db_session, test_project, changed):
    goal, run, rows = await _c2_coordination_rows(db_session, test_project)
    task, gate, artifact = rows["task"], rows["gate"], rows["artifact"]
    draft = _tool_draft("orchestration.start_graph@1", {
        "graph_id": str(rows["graph"].id), "subject_type": "artifact", "subject_id": str(artifact.id),
    }).model_copy(update={"general_updates": ["start graph"]})
    registry = RecoveryToolRegistry()
    proposal = await registry.canonicalize(db_session, goal, run, _trigger(goal.id, run.id), draft)
    before = await registry.capture_resources(db_session, goal, run, proposal)
    assert before.values[f"artifact:{artifact.id}"]["linked_task_id"] == str(task.id)
    assert f"task:{task.id}" in before.values and f"gate:{gate.id}" in before.values
    if changed == "linked_task":
        replacement = Task(project_id=test_project.id, title="Replacement linked task",
                           metadata_={"orchestration": {"run_id": str(run.id), "gate_id": str(gate.id)}})
        db_session.add(replacement)
        await db_session.flush()
        artifact.linked_task_id = replacement.id
    elif changed == "task":
        task.status = "done"
    else:
        gate.status = "accepted"
    await db_session.flush()
    after = await registry.capture_resources(db_session, goal, run, proposal)
    assert after.fingerprint != before.fingerprint
    if changed == "linked_task":
        assert f"task:{replacement.id}" in after.values


@pytest.mark.asyncio
@pytest.mark.parametrize("consumed_by", ("assignment", "manager"))
async def test_c2_global_capture_retains_only_consumed_inactive_agent_definitions_and_detects_drift(
    db_session, test_project, consumed_by,
):
    inactive = Agent(**replace(_agent(PLANNER_ID, "Inactive Planner", "planning"), is_active=False).__dict__)
    unused = Agent(**replace(_agent(REVIEWER_ID, "Unused Inactive Reviewer", "review"), is_active=False).__dict__)
    db_session.add_all([inactive, unused])
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    if consumed_by == "assignment":
        db_session.add(Task(project_id=test_project.id, title="Existing inactive assignment", assigned_to=inactive.id,
                            metadata_={"orchestration": {"run_id": str(run.id), "work_function": "planning"}}))
    else:
        goal.authority_model = "agent_manager"
        goal.manager_agent_id = inactive.id
    await db_session.flush()
    registry = RecoveryToolRegistry()
    proposal = _c2_resource_proposal("team_configuration.apply", _configuration_arguments(agents=[]), global_effect=True)
    before = await registry.capture_resources(db_session, goal, run, proposal)
    assert before.values[f"agent:{inactive.id}"]["is_active"] is False
    assert f"agent:{unused.id}" not in before.values
    inactive.description = "Changed consumed inactive definition."
    await db_session.flush()
    after = await registry.capture_resources(db_session, goal, run, proposal)
    assert after.values[f"agent:{inactive.id}"] != before.values[f"agent:{inactive.id}"]
    assert after.fingerprint != before.fingerprint


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent_change",
    (
        {"operation": "create", "ref": "new:cli", "definition": {
            "name": "CLI Recovery", "role": "planning", "provider": "local", "model": "local",
            "adapter_type": "cli", "capabilities": ["planning"], "config": {},
        }},
        {"operation": "update", "agent_id": str(PLANNER_ID), "set": {"adapter_type": "cli", "config": {}}},
        {"operation": "update", "agent_id": str(PLANNER_ID), "set": {"adapter_type": "cli", "config": {"cli_runtime": ""}}},
    ),
)
async def test_configuration_preparation_enforces_effective_cli_runtime_before_sealing(
    db_session, test_project, agent_change: dict[str, object],
):
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    if agent_change["operation"] == "update":
        db_session.add(Agent(**_agent(PLANNER_ID, "Planner", "planning").__dict__))
        await db_session.flush()
    draft = TeamConfigurationApplyV1.model_validate(_configuration_arguments(agents=[agent_change]))

    with pytest.raises(ValueError):
        await RecoveryConfigurationService().prepare(db_session, goal, run, draft)


@pytest.mark.asyncio
@pytest.mark.usefixtures("seal_without_post_state_validation")
async def test_configuration_changes_disclose_all_redacted_agent_fields_and_collateral_fingerprints(db_session, test_project):
    planner = _agent(PLANNER_ID, "Planner", "planning")
    db_session.add(Agent(**planner.__dict__))
    await db_session.flush()
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    collateral_goal, _collateral_run = await _seed_goal_run(db_session, test_project, objective="Review the release.")
    draft = TeamConfigurationApplyV1.model_validate(_configuration_arguments(agents=[{
        "operation": "update", "agent_id": str(planner.id), "set": {
            "name": "Lead Planner", "role": "lead planning", "description": "Own the remediation plan.",
            "provider": "local", "model": "recovery-model", "system_prompt": "Plan safely.",
            "adapter_type": "api", "capabilities": ["planning", "review"], "config": {"mode": "safe"},
        },
    }]))

    prepared = await RecoveryConfigurationService().prepare(db_session, goal, run, draft)
    agent_change = next(change for change in prepared.changes if change.target_type == "agent")
    fingerprint_changes = [change for change in prepared.changes if change.target_type == "configuration_fingerprint"]

    assert set(agent_change.before) == {
        "name", "role", "description", "provider", "model", "system_prompt", "adapter_type",
        "cli_runtime", "capabilities", "config", "is_active",
    }
    assert set(agent_change.after) == set(agent_change.before)
    assert len(fingerprint_changes) == 1
    assert fingerprint_changes[0].target_id == "aggregate"
    assert set(fingerprint_changes[0].after) == {str(goal.id), str(collateral_goal.id)}
    assert fingerprint_changes[0].after == {
        str(item.goal_id): {
            "before": {"coverage": item.before_coverage_fingerprint, "hierarchy": item.before_hierarchy_fingerprint},
            "after": {"coverage": item.after_coverage_fingerprint, "hierarchy": item.after_hierarchy_fingerprint},
        }
        for item in prepared.roster_plan.goal_fingerprints
        if item.goal_id in {goal.id, collateral_goal.id}
    }


@pytest.mark.asyncio
@pytest.mark.usefixtures("seal_without_post_state_validation")
async def test_configuration_summary_cannot_bypass_material_effect_disclosure_with_target_type_words(db_session, test_project):
    planner = _agent(PLANNER_ID, "Planner", "planning")
    db_session.add(Agent(**planner.__dict__))
    await db_session.flush()
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    await _seed_goal_run(db_session, test_project, objective="Review the release.")
    arguments = _configuration_arguments(agents=[{
        "operation": "update", "agent_id": str(planner.id), "set": {"role": "lead planning"},
    }])
    draft = _tool_draft("team_configuration.apply@1", arguments).model_copy(update={
        "general_updates": ["The agent configuration fingerprint goal retry has changes."],
    })

    with pytest.raises(ValueError, match="summary_missing_material_effect"):
        await RecoveryToolRegistry().canonicalize(db_session, goal, run, _trigger(goal.id, run.id), draft)


@pytest.mark.asyncio
async def test_resource_capture_contains_full_redacted_roster_and_all_preview_inputs(db_session, test_project):
    planner = replace(
        _agent(PLANNER_ID, "Planner", "planning"),
        description="Own the sensitive remediation.", model="recovery-model", system_prompt="Repair safely.",
        config={"api_key": "must-not-appear", "mode": "safe"},
    )
    db_planner = Agent(**planner.__dict__)
    db_session.add(db_planner)
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    collateral_goal, collateral_run = await _seed_goal_run(db_session, test_project, objective="Review the release.")
    task = Task(
        project_id=test_project.id, title="Run-owned planning task", assigned_to=planner.id,
        metadata_={"orchestration": {"run_id": str(run.id), "work_function": "planning"}},
    )
    definition = OrchestrationProcessRun(
        goal_id=goal.id, run_id=run.id, process_type="agent_definition_review", status="completed",
        trigger_reason="accepted", input_snapshot={"fingerprint": SHA256}, outputs={"target_ids": [str(planner.id)]},
    )
    hierarchy = OrchestrationProcessRun(
        goal_id=goal.id, run_id=run.id, process_type="team_hierarchy", status="completed",
        trigger_reason="accepted", input_snapshot={"fingerprint": SHA256}, outputs={
            "analysis": {"manager": str(planner.id)},
            "responsibility_coverage": [{
                "responsibility_id": "ux-zero-eligible",
                "status": "gap",
                "reason": "Sensitive note token=sk-coverage-secret must be redacted.",
                "evidence_ref": "gap:zero-eligible-ux",
            }],
            "accepted_responsibility_gap_ids": ["ux-zero-eligible"],
        },
    )
    db_session.add_all([task, definition, hierarchy])
    await db_session.flush()
    proposal_data = _proposal().model_dump(mode="json")
    proposal_data.update({
        "global_effect": True,
        "calls": [{
            "call_id": "call-1", "tool": "team_configuration.apply", "tool_version": "1",
            "arguments": _configuration_arguments(agents=[]),
            "changes": [{"operation": "update", "scope": "global", "target_type": "agent", "target_id": str(planner.id), "after": {"role": "lead planning"}}],
        }],
    })
    proposal = RecoveryProposalV1.model_validate(proposal_data)
    registry = RecoveryToolRegistry()

    before = await registry.capture_resources(db_session, goal, run, proposal)
    roster = before.values[f"agent:{planner.id}"]

    assert set(roster) == {
        "id", "name", "role", "description", "provider", "model", "system_prompt", "adapter_type",
        "cli_runtime", "capabilities", "config", "is_active",
    }
    assert f"goal_contract:{collateral_goal.id}" in before.values
    assert f"run:{collateral_run.id}" in before.values
    assert f"task:{task.id}" in before.values
    assert f"process:agent_definition_review:{definition.id}" in before.values
    assert f"process:team_hierarchy:{hierarchy.id}" in before.values
    captured_hierarchy = before.values[f"process:team_hierarchy:{hierarchy.id}"]
    assert captured_hierarchy["accepted_responsibility_coverage"][0]["responsibility_id"] == "ux-zero-eligible"
    assert captured_hierarchy["accepted_accepted_responsibility_gap_ids"] == ["ux-zero-eligible"]
    captured_hierarchy["accepted_responsibility_coverage"][0]["reason"] = "mutated snapshot"
    assert hierarchy.outputs["responsibility_coverage"][0]["reason"] != "mutated snapshot"
    serialized = json.dumps(before.values, sort_keys=True)
    assert "api_key" not in serialized
    assert "must-not-appear" not in serialized
    assert "sk-coverage-secret" not in serialized

    db_planner.description = "Changed full roster field."
    collateral_goal.constraints = {"deadline": "tomorrow"}
    definition.input_snapshot = {"fingerprint": "b" * 64}
    task.title = "Changed run-owned assignment"
    await db_session.flush()
    after = await registry.capture_resources(db_session, goal, run, proposal)

    assert after.fingerprint != before.fingerprint


def test_configuration_gates_require_exact_substantive_gap_acknowledgement_but_keep_legacy_behavior():
    definition = SimpleNamespace(
        status="completed",
        outputs={"gates": {"agent_definitions_reviewed": True}},
    )
    gap = {
        "responsibility_id": "ux-zero-eligible",
        "status": "gap",
        "reason": "No accepted UX capability.",
    }
    hierarchy = SimpleNamespace(
        status="completed",
        outputs={
            "approval": "approve_with_documented_gaps",
            "approval_decision_id": "bound-human-decision",
            "responsibility_coverage": [gap],
            "accepted_responsibility_gap_ids": [gap["responsibility_id"]],
            "documented_gaps": [],
            "gates": {
                "team_structure_reviewed": True,
                "agent_definitions_reviewed": True,
                "required_work_functions_mapped": False,
                "missing_capabilities_handled": True,
                "independent_verification_possible_or_overridden": True,
            },
        },
    )

    assert RecoveryToolRegistry._configuration_gates(definition, hierarchy) is True
    for invalid in (
        {**hierarchy.outputs, "accepted_responsibility_gap_ids": []},
        {**hierarchy.outputs, "accepted_responsibility_gap_ids": ["wrong-gap"]},
        {**hierarchy.outputs, "accepted_responsibility_gap_ids": [gap["responsibility_id"], gap["responsibility_id"]]},
        {**hierarchy.outputs, "accepted_responsibility_gap_ids": [gap["responsibility_id"], 1]},
        {**hierarchy.outputs, "accepted_responsibility_gap_ids": [""]},
        {**hierarchy.outputs, "accepted_responsibility_gap_ids": ["   "]},
        {**hierarchy.outputs, "approval": "approve"},
        {**hierarchy.outputs, "approval_decision_id": None},
        {**hierarchy.outputs, "responsibility_coverage": []},
        {**hierarchy.outputs, "responsibility_coverage": [{}]},
        {**hierarchy.outputs, "responsibility_coverage": [{**gap, "status": "proposed"}]},
        {**hierarchy.outputs, "responsibility_coverage": [{**gap, "status": "partial"}]},
        {**hierarchy.outputs, "responsibility_coverage": [{key: value for key, value in gap.items() if key != "responsibility_id"}]},
        {**hierarchy.outputs, "responsibility_coverage": [{**gap, "responsibility_id": ""}]},
        {**hierarchy.outputs, "responsibility_coverage": [{**gap, "responsibility_id": "   "}]},
        {**hierarchy.outputs, "responsibility_coverage": [{**gap, "responsibility_id": 1}]},
        {**hierarchy.outputs, "responsibility_coverage": [gap, deepcopy(gap)]},
    ):
        hierarchy.outputs = invalid
        assert RecoveryToolRegistry._configuration_gates(definition, hierarchy) is False

    generic_gaps = ["legacy-generic-gap"]
    hierarchy.outputs = {
        "approval": "approve_with_documented_gaps",
        "approval_decision_id": "bound-human-decision",
        "responsibility_coverage": [],
        "accepted_responsibility_gap_ids": [],
        "documented_gaps": generic_gaps,
        "gates": {
            "team_structure_reviewed": True,
            "agent_definitions_reviewed": True,
            "required_work_functions_mapped": False,
            "missing_capabilities_handled": True,
            "independent_verification_possible_or_overridden": True,
        },
    }
    assert RecoveryToolRegistry._configuration_gates(definition, hierarchy) is True
    for invalid in (
        {**hierarchy.outputs, "approval": "approve"},
        {**hierarchy.outputs, "approval_decision_id": ""},
        {**hierarchy.outputs, "accepted_responsibility_gap_ids": ["stale-gap"]},
    ):
        hierarchy.outputs = invalid
        assert RecoveryToolRegistry._configuration_gates(definition, hierarchy) is False

    hierarchy.outputs = {
        **{
            key: value for key, value in hierarchy.outputs.items()
            if key not in {"approval", "approval_decision_id", "responsibility_coverage", "accepted_responsibility_gap_ids"}
        },
        "approval": "approve_with_documented_gaps",
        "approval_decision_id": "bound-human-decision",
        "responsibility_coverage": [gap],
        "accepted_responsibility_gap_ids": [gap["responsibility_id"]],
    }
    assert RecoveryToolRegistry._configuration_gates(definition, hierarchy) is True

    hierarchy.outputs = {
        "approval": "approve",
        "approval_decision_id": "clean-bound-decision",
        "responsibility_coverage": [],
        "accepted_responsibility_gap_ids": [],
        "documented_gaps": [],
        "gates": {
            "team_structure_reviewed": True,
            "agent_definitions_reviewed": True,
            "required_work_functions_mapped": True,
            "missing_capabilities_handled": True,
            "independent_verification_possible_or_overridden": True,
        },
    }
    assert RecoveryToolRegistry._configuration_gates(definition, hierarchy) is True
    hierarchy.outputs = {**hierarchy.outputs, "accepted_responsibility_gap_ids": ["stale-gap"]}
    assert RecoveryToolRegistry._configuration_gates(definition, hierarchy) is False

    hierarchy.outputs = {
        "documented_gaps": ["legacy-generic-gap"],
        "gates": {
            "team_structure_reviewed": True,
            "agent_definitions_reviewed": True,
            "required_work_functions_mapped": False,
            "missing_capabilities_handled": True,
            "independent_verification_possible_or_overridden": True,
        },
    }
    assert RecoveryToolRegistry._configuration_gates(definition, hierarchy) is True


@pytest.mark.asyncio
async def test_canonicalization_seals_server_owned_ids_retry_checks_and_resource_fingerprint(db_session, test_project):
    db_session.add(Agent(**_agent(PLANNER_ID, "Planner", "planning").__dict__))
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    draft = _tool_draft("orchestration.request_plan@1", {
        "agent_id": str(PLANNER_ID), "work_function": "planning", "scope": "goal",
    }).model_copy(update={
        "general_updates": ["Request plan."],
    })
    registry = RecoveryToolRegistry()

    proposal = await registry.canonicalize(db_session, goal, run, _trigger(goal.id, run.id), draft)

    assert proposal.registry_version == "1"
    assert proposal.calls[0].call_id == "call-1"
    assert proposal.calls[0].tool == "orchestration.request_plan"
    assert proposal.calls[0].tool_version == "1"
    assert proposal.verification[0].verifier == "action.completed"
    assert proposal.retry.max_attempts == 1
    assert proposal.resource_fingerprint != "0" * 64
    assert registry.last_proposal_hash is not None
    assert len(registry.last_proposal_hash) == 64


@pytest.mark.asyncio
async def test_registry_rejects_summary_narrower_than_canonical_changes(db_session, test_project):
    db_session.add(Agent(**_agent(PLANNER_ID, "Planner", "planning").__dict__))
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")

    with pytest.raises(ValueError, match="summary_missing_material_effect"):
        await RecoveryToolRegistry().canonicalize(
            db_session,
            goal,
            run,
            _trigger(goal.id, run.id),
            _tool_draft("orchestration.request_plan@1", {
                "agent_id": str(PLANNER_ID), "work_function": "planning", "scope": "goal",
            }).model_copy(update={"general_updates": ["Repair the planning roster."]}),
        )


@pytest.mark.asyncio
async def test_resource_fingerprint_changes_with_any_global_roster_member(db_session, test_project):
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    proposal = _proposal().model_copy(update={"global_effect": True})
    registry = RecoveryToolRegistry()
    before = await registry.capture_resources(db_session, goal, run, proposal)
    db_session.add(Agent(**_agent(PLANNER_ID, "Planner", "planning").__dict__))
    await db_session.flush()

    after = await registry.capture_resources(db_session, goal, run, proposal)

    assert before.fingerprint != after.fingerprint
    assert f"run:{run.id}" in after.values
    assert f"goal_contract:{goal.id}" in after.values
    assert f"agent:{PLANNER_ID}" in after.values


@pytest.mark.asyncio
async def test_resource_capture_includes_each_registered_call_target(db_session, test_project):
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the release.")
    task = Task(
        project_id=test_project.id, title="Retry this task", status="failed",
        metadata_={"orchestration": {"run_id": str(run.id), "work_function": "implementation"}},
    )
    db_session.add(task)
    await db_session.flush()
    draft_data = _draft("orchestration.retry_task@1").model_dump(mode="json")
    draft_data["general_updates"] = ["Request the orchestration retry task repair."]
    draft_data["calls"][0]["arguments"] = {"task_id": str(task.id)}
    draft = ProposalDraftV1.model_validate(draft_data)
    registry = RecoveryToolRegistry()

    proposal = await registry.canonicalize(db_session, goal, run, _trigger(goal.id, run.id), draft)
    resources = await registry.capture_resources(db_session, goal, run, proposal)

    assert f"task:{task.id}" in resources.values


@pytest.mark.asyncio
async def test_registered_verifiers_pass_only_for_matching_real_linked_state(db_session, test_project):
    rows = await _c3_context(db_session, test_project)
    await _c3_accepted_configuration(db_session, rows)
    await _c3_satisfied_gate(db_session, rows)
    await _c3_bound_receipt(db_session, rows, artifact=True)
    goal, run, task, gate, process, artifact = (rows[key] for key in ("goal", "run", "task", "gate", "hierarchy", "artifact"))
    warning = OrchestrationWarning(
        goal_id=goal.id,
        run_id=run.id,
        warning_type="stale_configuration",
        severity="warning",
        message="The old configuration was replaced.",
        active=False,
        resolved_by="orchestrator:test",
        resolved_reason="The approved repair completed.",
    )
    db_session.add(warning)
    await db_session.flush()
    checks = (
        VerificationCheckV1(check_id="task", verifier="task.completed", verifier_version="1", arguments={
            "source": "resource", "goal_id": str(goal.id), "run_id": str(run.id), "task_id": str(task.id),
        }),
        VerificationCheckV1(check_id="gate", verifier="gate.satisfied", verifier_version="1", arguments={
            "goal_id": str(goal.id), "run_id": str(run.id), "gate_id": str(gate.id),
        }),
        VerificationCheckV1(check_id="process", verifier="process.completed", verifier_version="1", arguments={
            "goal_id": str(goal.id), "process_id": str(process.id), "expected_fingerprint": process.input_snapshot["fingerprint"],
        }),
        VerificationCheckV1(check_id="warning", verifier="warning.resolved", verifier_version="1", arguments={
            "goal_id": str(goal.id), "warning_id": str(warning.id),
        }),
        VerificationCheckV1(check_id="artifact", verifier="artifact.exists", verifier_version="1", arguments={
            "source": "resource", "project_id": str(test_project.id), "goal_id": str(goal.id),
            "artifact_id": str(artifact.id), "artifact_type": "plan", "content_hash": SHA256,
        }),
    )

    results = [await RecoveryToolRegistry().verify(db_session, rows["proposal"], check) for check in checks]

    assert [result.status for result in results] == ["passed"] * len(checks)


@pytest.mark.asyncio
@pytest.mark.usefixtures("seal_without_post_state_validation")
async def test_configuration_and_action_verifiers_reject_unlinked_completed_records(db_session, test_project):
    rows = await _c3_context(db_session, test_project, configuration=True)
    goal, run = rows["goal"], rows["run"]
    definition = OrchestrationProcessRun(
        goal_id=goal.id,
        run_id=run.id,
        process_type="agent_definition_review",
        status="completed",
        trigger_reason="ordinary review",
        outputs={},
    )
    hierarchy = OrchestrationProcessRun(
        goal_id=goal.id,
        run_id=run.id,
        process_type="team_hierarchy",
        status="completed",
        trigger_reason="ordinary review",
        outputs={},
    )
    unrelated_action = OrchestrationAction(
        run_id=run.id,
        idempotency_key="ordinary-action",
        action_type="request_plan",
        request={},
        dispatch_contract={},
        status="completed",
    )
    db_session.add_all([definition, hierarchy, unrelated_action])
    await db_session.flush()
    proposal = rows["proposal"]
    configuration = VerificationCheckV1(
        check_id="configuration",
        verifier="team_configuration.accepted",
        verifier_version="1",
        arguments={"goal_id": str(goal.id), "call_id": proposal.calls[0].call_id},
    )
    action = VerificationCheckV1(
        check_id="action",
        verifier="action.completed",
        verifier_version="1",
        arguments={"call_id": proposal.calls[0].call_id},
    )

    configuration_result = await RecoveryToolRegistry().verify(db_session, proposal, configuration)
    action_result = await RecoveryToolRegistry().verify(db_session, proposal, action)

    assert configuration_result.status == "failed"
    assert action_result.status == "failed"


@pytest.mark.asyncio
async def test_planning_ready_verifier_passes_only_when_the_named_run_is_ready(db_session, test_project):
    rows = await _c3_context(db_session, test_project)
    await _c3_accepted_configuration(db_session, rows)
    goal, run = rows["goal"], rows["run"]
    run.phase = "ready"
    await db_session.flush()
    check = VerificationCheckV1(
        check_id="planning",
        verifier="planning.ready",
        verifier_version="1",
        arguments={"goal_id": str(goal.id), "run_id": str(run.id)},
    )

    result = await RecoveryToolRegistry().verify(db_session, rows["proposal"], check)

    assert result.status == "passed"


@pytest.mark.asyncio
async def test_action_completed_rejects_mismatched_immutable_effect_identity(db_session, test_project):
    rows = await _c3_context(db_session, test_project)
    await _c3_bound_receipt(db_session, rows)
    rows["effect"].idempotency_key = "different-effect-key"
    await db_session.flush()
    check = VerificationCheckV1(
        check_id="action",
        verifier="action.completed",
        verifier_version="1",
        arguments={"call_id": rows["proposal"].calls[0].call_id},
    )

    result = await RecoveryToolRegistry().verify(db_session, rows["proposal"], check)

    assert result.status == "failed"


def _c3_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


async def _c3_context(db_session, test_project, *, configuration=False):
    planner = Agent(**_agent(PLANNER_ID, "Planner", "planning").__dict__)
    db_session.add(planner)
    goal, run = await _seed_goal_run(db_session, test_project, objective="Plan the verified release.")
    goal.authority_model = "no_manager"
    task = Task(project_id=test_project.id, title="Accepted planning assignment", assigned_to=planner.id, status="done",
                metadata_={"orchestration": {"run_id": str(run.id), "work_function": "planning"}})
    db_session.add(task)
    await db_session.flush()
    if configuration:
        arguments = _configuration_arguments(agents=[{
            "operation": "update", "agent_id": str(planner.id), "set": {"role": "lead planning"},
        }])
        arguments["eligibility"] = [{"agent_ref": str(planner.id), "work_functions": ["planning"]}]
        arguments["assignments"] = [{"agent_ref": str(planner.id), "work_function": "planning"}]
        arguments["hierarchy"]["team_leads"] = [str(planner.id)]
        draft = _tool_draft("team_configuration.apply@1", arguments).model_copy(update={
            "general_updates": ["apply team configuration", "agent definitions", "eligibility", "assignments",
                                "reporting hierarchy", "Planner"],
        })
    else:
        draft = _tool_draft("orchestration.request_plan@1", {
            "agent_id": str(planner.id), "work_function": "planning", "scope": "goal",
        }).model_copy(update={"general_updates": ["request plan"]})
    trigger = _trigger(goal.id, run.id).model_copy(update={"source_id": run.id})
    proposal = await RecoveryToolRegistry().canonicalize(db_session, goal, run, trigger, draft)
    proposal_action = OrchestrationAction(
        run_id=run.id, action_type="recovery_proposal", status="completed",
        idempotency_key=f"recovery:{proposal.proposal_id}:proposal", request=proposal.model_dump(mode="json"),
        dispatch_contract={"proposal_hash": _c3_hash(proposal.model_dump(mode="json"))},
    )
    db_session.add(proposal_action)
    await db_session.flush()
    authority = OrchestrationAuthorityDecision(
        goal_id=goal.id, run_id=run.id, related_action_id=proposal_action.id,
        decision_key=f"recovery:{proposal.proposal_id}:approval", title="Approve the recovery.", question="Apply the repair?",
        status="answered", authority="human", selected_option="approve_fix", options=[{"key": "approve_fix"}],
    )
    db_session.add(authority)
    await db_session.flush()
    return {"goal": goal, "run": run, "planner": planner, "task": task, "proposal": proposal,
            "proposal_action": proposal_action, "authority": authority}


async def _c3_accepted_configuration(db_session, rows):
    goal, run, planner, proposal = (rows[key] for key in ("goal", "run", "planner", "proposal"))
    if proposal.calls[0].tool == "team_configuration.apply":
        planner.role = proposal.calls[0].arguments["agents"][0]["set"]["role"]
    definition = OrchestrationProcessRun(
        goal_id=goal.id, run_id=run.id, process_type="agent_definition_review", status="completed", trigger_reason="accepted",
        input_snapshot={}, outputs={"target_ids": [str(planner.id)], "eligibility": {str(planner.id): ["planning"]},
                                   "gates": {"agent_definitions_reviewed": True}},
    )
    hierarchy = OrchestrationProcessRun(
        goal_id=goal.id, run_id=run.id, process_type="team_hierarchy", status="completed", trigger_reason="accepted",
        input_snapshot={}, outputs={
            "assignments": [{"agent_ref": str(planner.id), "work_function": "planning"}],
            "hierarchy": {"team_leads": [str(planner.id)], "contributors": [], "reviewers": [], "validators": [], "specialists": []},
            "reporting_lines": [], "documented_gaps": [],
            "gates": {"team_structure_reviewed": True, "required_work_functions_mapped": True,
                      "agent_definitions_reviewed": True, "missing_capabilities_handled": True,
                      "independent_verification_possible_or_overridden": True},
        },
    )
    prerequisites = [OrchestrationProcessRun(
        goal_id=goal.id, run_id=run.id, process_type=kind, status="completed", trigger_reason="accepted", outputs={},
    ) for kind in ("goal_definition", "manager_selection")]
    db_session.add_all([definition, hierarchy, *prerequisites])
    await db_session.flush()
    definition_fingerprint = await AgentDefinitionReviewProcess().current_fingerprint(db_session, goal, run)
    coverage_fingerprint = await AgentDefinitionReviewProcess().current_coverage_fingerprint(
        db_session, goal, run, covered_target_ids={str(planner.id)},
    )
    hierarchy_fingerprint = await TeamHierarchyProcess().current_fingerprint(db_session, goal, run)
    provenance = {"proposal_id": str(proposal.proposal_id), "proposal_hash": _c3_hash(proposal.model_dump(mode="json")),
                  "call_id": proposal.calls[0].call_id, "canonical_request_hash": _c3_hash(proposal.calls[0].arguments)}
    definition.input_snapshot = {"fingerprint": definition_fingerprint}
    definition.outputs = {**definition.outputs, "fingerprint": definition_fingerprint,
                          "coverage_fingerprint": coverage_fingerprint, "recovery_provenance": provenance}
    hierarchy.input_snapshot = {"fingerprint": hierarchy_fingerprint}
    hierarchy.outputs = {**hierarchy.outputs, "fingerprint": hierarchy_fingerprint, "recovery_provenance": provenance}
    await db_session.flush()
    assert await OrchestrationService()._baseline_processes_ready_for_goal(db_session, goal.id, allow_heal=False)
    from huddleroom.services.orchestration_team_hierarchy import select_plan_author
    assert await select_plan_author(db_session, goal, run) == planner.id
    rows.update(definition=definition, hierarchy=hierarchy, prerequisites=prerequisites)


async def _c3_satisfied_gate(db_session, rows):
    gate = OrchestrationGate(
        run_id=rows["run"].id, success_criterion_key="planning", gate_type="evidence", status="accepted",
        required_evidence={"required_source_types": ["task"], "min_count": 1},
    )
    db_session.add(gate)
    await db_session.flush()
    evidence = OrchestrationEvidence(
        run_id=rows["run"].id, gate_id=gate.id, source_type="task", source_id=rows["task"].id,
        producer_agent_id=rows["planner"].id, verdict="accepted", evidence_metadata={},
    )
    db_session.add(evidence)
    await db_session.flush()
    assert await OrchestrationService()._validate_gate(db_session, gate)
    assert gate.status == "accepted"
    rows.update(gate=gate, gate_evidence=evidence)


def _c3_check(verifier, arguments):
    return VerificationCheckV1(check_id="checked", verifier=verifier, verifier_version="1", arguments=arguments)


@pytest.mark.asyncio
@pytest.mark.parametrize("verifier", ("task.completed", "gate.satisfied", "process.completed", "warning.resolved"))
@pytest.mark.parametrize("lineage", ("matching", "foreign_run", "foreign_goal", "foreign_project"))
async def test_c3_resource_verifiers_require_real_current_state_and_exact_lineage(db_session, test_project, verifier, lineage):
    rows = await _c3_context(db_session, test_project)
    await _c3_satisfied_gate(db_session, rows)
    goal, run = rows["goal"], rows["run"]
    if lineage == "foreign_project":
        project = Project(name="Foreign verifier project")
        db_session.add(project)
        await db_session.flush()
        foreign_goal, foreign_run = await _seed_goal_run(db_session, project, objective="Foreign project goal.")
    else:
        foreign_goal, foreign_run = await _seed_goal_run(db_session, test_project, objective="Foreign verifier goal.")
    if lineage == "foreign_run":
        foreign_run.status = "completed"
        foreign_run.goal_id = goal.id
    selected_goal = goal if lineage in {"matching", "foreign_run"} else foreign_goal
    selected_run = run if lineage == "matching" else foreign_run
    if verifier == "task.completed":
        row = rows["task"]
        if lineage != "matching":
            row.project_id = selected_goal.project_id
            row.metadata_ = {"orchestration": {"run_id": str(selected_run.id), "work_function": "planning"}}
        arguments = {"source": "resource", "goal_id": str(goal.id), "run_id": str(run.id), "task_id": str(row.id)}
        source_kind = "task"
    elif verifier == "gate.satisfied":
        row = rows["gate"]
        row.run_id = selected_run.id
        rows["gate_evidence"].run_id = selected_run.id
        arguments = {"goal_id": str(goal.id), "run_id": str(run.id), "gate_id": str(row.id)}
        source_kind = "gate"
    elif verifier == "process.completed":
        row = OrchestrationProcessRun(goal_id=selected_goal.id, run_id=selected_run.id, process_type="team_hierarchy",
                                      status="completed", trigger_reason="accepted", input_snapshot={"fingerprint": SHA256})
        db_session.add(row)
        await db_session.flush()
        arguments = {"goal_id": str(goal.id), "process_id": str(row.id), "expected_fingerprint": SHA256}
        source_kind = "process"
    else:
        row = OrchestrationWarning(goal_id=selected_goal.id, run_id=selected_run.id, warning_type="stale_configuration",
                                   severity="warning", message="A resolved warning token=sk-verifier-secret.", active=False,
                                   resolved_by="human:test", resolved_reason="Accepted repair.")
        db_session.add(row)
        await db_session.flush()
        arguments = {"goal_id": str(goal.id), "warning_id": str(row.id)}
        source_kind = "warning"
    await db_session.flush()
    result = await RecoveryToolRegistry().verify(db_session, rows["proposal"], _c3_check(verifier, arguments))
    assert result.status == ("passed" if lineage == "matching" or (verifier == "warning.resolved" and lineage == "foreign_run") else "failed")
    assert any(item.source_kind == source_kind and item.source_id == row.id for item in result.evidence)
    assert "sk-verifier-secret" not in json.dumps(result.model_dump(mode="json"), sort_keys=True)
    assert len(json.dumps(result.model_dump(mode="json"), sort_keys=True).encode()) < 20_000


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ("historical", "fingerprint", "not_completed", "optional_fingerprint"))
async def test_c3_process_requires_current_named_process_and_optional_exact_fingerprint(db_session, test_project, invalid):
    rows = await _c3_context(db_session, test_project)
    process = OrchestrationProcessRun(goal_id=rows["goal"].id, run_id=rows["run"].id, process_type="team_hierarchy",
                                      status="completed", trigger_reason="accepted", input_snapshot={"fingerprint": SHA256})
    db_session.add(process)
    await db_session.flush()
    if invalid == "historical":
        from huddleroom.services.orchestration_process_service import OrchestrationProcessService
        service = OrchestrationProcessService()
        replacement = await service.start_process(
            db_session, rows["goal"].id, process_type="team_hierarchy", trigger_reason="replacement",
            run_id=rows["run"].id, input_snapshot={"fingerprint": SHA256},
        )
        await service.complete_process(db_session, replacement, outputs={"fingerprint": SHA256})
        assert process.superseded_by_id == replacement.id
    elif invalid == "not_completed":
        process.status = "running"
    arguments = {"goal_id": str(rows["goal"].id), "process_id": str(process.id)}
    if invalid != "optional_fingerprint":
        arguments["expected_fingerprint"] = "b" * 64 if invalid == "fingerprint" else SHA256
    await db_session.flush()
    result = await RecoveryToolRegistry().verify(db_session, rows["proposal"], _c3_check("process.completed", arguments))
    assert result.status == ("passed" if invalid == "optional_fingerprint" else "failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ("accepted_status_only", "latest_rejected", "missing_source", "stale_evidence"))
async def test_c3_gate_accepted_text_cannot_replace_current_evidence_predicate(db_session, test_project, invalid):
    rows = await _c3_context(db_session, test_project)
    await _c3_satisfied_gate(db_session, rows)
    gate, evidence = rows["gate"], rows["gate_evidence"]
    if invalid == "accepted_status_only":
        await db_session.delete(evidence)
    elif invalid == "latest_rejected":
        evidence.verdict = "rejected"
    elif invalid == "missing_source":
        gate.required_evidence = {"required_source_types": ["task", "verification"], "min_count": 2}
    else:
        gate.required_evidence = {"required_source_types": ["task", "verification"], "min_count": 2}
        verification = OrchestrationEvidence(run_id=rows["run"].id, gate_id=gate.id, source_type="verification",
                                             source_id=UUID(int=99_990), verdict="accepted",
                                             evidence_metadata={"event_seq": 1})
        evidence.evidence_metadata = {"event_seq": 2}
        db_session.add(verification)
    await db_session.flush()
    assert gate.status == "accepted"
    result = await RecoveryToolRegistry().verify(db_session, rows["proposal"], _c3_check("gate.satisfied", {
        "goal_id": str(rows["goal"].id), "run_id": str(rows["run"].id), "gate_id": str(gate.id),
    }))
    assert result.status == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ("resolved_by", "resolved_reason", "inactive"))
async def test_c3_warning_requires_inactive_state_and_resolution_provenance(db_session, test_project, missing):
    rows = await _c3_context(db_session, test_project)
    warning = OrchestrationWarning(goal_id=rows["goal"].id, run_id=rows["run"].id, warning_type="stale_configuration",
                                   severity="warning", message="Resolved.", active=missing == "inactive",
                                   resolved_by=None if missing == "resolved_by" else "human:test",
                                   resolved_reason=None if missing == "resolved_reason" else "Accepted repair.")
    db_session.add(warning)
    await db_session.flush()
    result = await RecoveryToolRegistry().verify(db_session, rows["proposal"], _c3_check("warning.resolved", {
        "goal_id": str(rows["goal"].id), "warning_id": str(warning.id),
    }))
    assert result.status == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", (
    "matching", "agents", "eligibility", "assignments", "definition_fingerprint", "coverage_fingerprint",
    "hierarchy_fingerprint", "definition_gate", "team_structure_reviewed", "required_work_functions_mapped",
    "agent_definitions_reviewed", "missing_capabilities_handled", "independent_verification_possible_or_overridden",
    "proposal_provenance", "hash_provenance", "foreign_goal", "foreign_run", "foreign_project",
))
@pytest.mark.usefixtures("seal_without_post_state_validation")
async def test_c3_team_acceptance_requires_exact_sealed_state_fingerprints_gates_and_provenance(
    db_session, test_project, changed,
):
    rows = await _c3_context(db_session, test_project, configuration=True)
    await _c3_accepted_configuration(db_session, rows)
    goal, run, definition, hierarchy = (rows[key] for key in ("goal", "run", "definition", "hierarchy"))
    expected = next(change.after for change in rows["proposal"].calls[0].changes
                    if change.target_type == "configuration_fingerprint")
    assert expected["coverage"] == definition.outputs["coverage_fingerprint"]
    assert expected["hierarchy"] == hierarchy.outputs["fingerprint"]
    if changed == "agents":
        rows["planner"].description = "Different accepted agent definition."
    elif changed in {"eligibility", "definition_fingerprint", "coverage_fingerprint", "definition_gate"}:
        outputs = deepcopy(definition.outputs)
        if changed == "eligibility":
            outputs["eligibility"] = {str(rows["planner"].id): ["review"]}
        elif changed == "definition_gate":
            outputs["gates"]["agent_definitions_reviewed"] = False
        else:
            outputs["fingerprint" if changed == "definition_fingerprint" else "coverage_fingerprint"] = "b" * 64
        definition.outputs = outputs
    elif changed in {"assignments", "hierarchy_fingerprint", "team_structure_reviewed", "required_work_functions_mapped",
                     "agent_definitions_reviewed", "missing_capabilities_handled", "independent_verification_possible_or_overridden"}:
        outputs = deepcopy(hierarchy.outputs)
        if changed == "assignments":
            outputs["assignments"] = [{"agent_ref": str(rows["planner"].id), "work_function": "review"}]
        elif changed == "hierarchy_fingerprint":
            outputs["fingerprint"] = "b" * 64
        else:
            outputs["gates"][changed] = False
        hierarchy.outputs = outputs
    elif changed in {"proposal_provenance", "hash_provenance"}:
        outputs = deepcopy(definition.outputs)
        outputs["recovery_provenance"]["proposal_id" if changed == "proposal_provenance" else "proposal_hash"] = (
            str(UUID(int=98_001)) if changed == "proposal_provenance" else "b" * 64
        )
        definition.outputs = outputs
    elif changed.startswith("foreign"):
        if changed == "foreign_project":
            project = Project(name="Foreign accepted configuration")
            db_session.add(project)
            await db_session.flush()
        else:
            project = test_project
        foreign_goal, foreign_run = await _seed_goal_run(db_session, project, objective="Foreign configuration.")
        for process in (definition, hierarchy):
            process.run_id = foreign_run.id
            if changed != "foreign_run":
                process.goal_id = foreign_goal.id
    await db_session.flush()
    result = await RecoveryToolRegistry().verify(db_session, rows["proposal"], _c3_check("team_configuration.accepted", {
        "goal_id": str(goal.id), "call_id": rows["proposal"].calls[0].call_id,
    }))
    assert result.status == ("passed" if changed == "matching" else "failed")
    if changed == "matching":
        assert any(item.source_kind == "process" and item.source_id in {definition.id, hierarchy.id} for item in result.evidence)


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", (
    "matching", "phase_only", "inactive_author", "unaccepted_author", "baseline_not_complete", "stale_definition",
    "stale_hierarchy", "terminal_plan", "terminal_planning_gate", "foreign_goal", "foreign_run", "foreign_project",
))
async def test_c3_planning_ready_requires_pure_preflight_active_accepted_author_and_baseline_admission(
    db_session, test_engine, test_project, monkeypatch, changed,
):
    rows = await _c3_context(db_session, test_project)
    goal, run = rows["goal"], rows["run"]
    run.phase = "ready"
    if changed != "phase_only":
        await _c3_accepted_configuration(db_session, rows)
    if changed == "inactive_author":
        rows["planner"].is_active = False
    elif changed == "unaccepted_author":
        rows["definition"].outputs = {**rows["definition"].outputs, "eligibility": {str(rows["planner"].id): []}}
    elif changed == "baseline_not_complete":
        rows["prerequisites"][0].status = "running"
    elif changed in {"stale_definition", "stale_hierarchy"}:
        name, key = ("definition", "coverage_fingerprint") if changed == "stale_definition" else ("hierarchy", "fingerprint")
        rows[name].outputs = {**rows[name].outputs, key: "b" * 64}
    elif changed == "terminal_plan":
        run.plan_state = {"status": "accepted"}
    elif changed == "terminal_planning_gate":
        from huddleroom.services.orchestration_service import PLAN_GATE_KEY, PLAN_GATE_TYPE
        db_session.add(OrchestrationGate(run_id=run.id, gate_type=PLAN_GATE_TYPE,
                                        success_criterion_key=PLAN_GATE_KEY, status="failed"))
    arguments = {"goal_id": str(goal.id), "run_id": str(run.id)}
    if changed.startswith("foreign"):
        if changed == "foreign_project":
            project = Project(name="Foreign ready project")
            db_session.add(project)
            await db_session.flush()
        else:
            project = test_project
        foreign_goal, foreign_run = await _seed_goal_run(db_session, project, objective="Foreign ready goal.")
        foreign_run.phase = "ready"
        arguments["goal_id" if changed == "foreign_goal" else "run_id"] = str(
            foreign_goal.id if changed == "foreign_goal" else foreign_run.id,
        )
    await db_session.flush()
    from unittest.mock import AsyncMock
    flush_spy = AsyncMock(wraps=db_session.flush)
    commit_spy = AsyncMock(wraps=db_session.commit)
    reserve_spy = AsyncMock(side_effect=AssertionError("verification must not reserve actions"))
    monkeypatch.setattr(db_session, "flush", flush_spy)
    monkeypatch.setattr(db_session, "commit", commit_spy)
    monkeypatch.setattr(OrchestrationService, "reserve_action", reserve_spy)
    writes = []

    def record_write(_conn, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
            writes.append(statement)

    event.listen(test_engine.sync_engine, "before_cursor_execute", record_write)
    try:
        result = await RecoveryToolRegistry().verify(db_session, rows["proposal"], _c3_check("planning.ready", arguments))
    finally:
        event.remove(test_engine.sync_engine, "before_cursor_execute", record_write)
    assert result.status == ("passed" if changed == "matching" else "failed")
    assert any(item.source_kind == "run" and item.source_id == UUID(arguments["run_id"]) for item in result.evidence)
    flush_spy.assert_not_awaited()
    commit_spy.assert_not_awaited()
    reserve_spy.assert_not_awaited()
    assert writes == []


async def _c3_bound_receipt(db_session, rows, *, aggregate=False, artifact=False):
    proposal, run, task = (rows[key] for key in ("proposal", "run", "task"))
    call = proposal.calls[0]
    wrapper_key = f"recovery:{proposal.proposal_id}:apply:{call.call_id}"
    wrapper = OrchestrationAction(
        run_id=run.id, action_type="recovery_apply", status="completed", idempotency_key=wrapper_key,
        request={"proposal_action_id": str(rows["proposal_action"].id),
                 "proposal_hash": _c3_hash(proposal.model_dump(mode="json")), "call_id": call.call_id},
        target_type="agent" if aggregate else "task", target_id=rows["planner"].id if aggregate else task.id,
    )
    db_session.add(wrapper)
    await db_session.flush()
    effect = wrapper if aggregate else OrchestrationAction(
        run_id=run.id, action_type="request_plan", status="completed", idempotency_key=f"{wrapper_key}:effect",
        request=call.arguments, target_type="task", target_id=task.id,
    )
    if not aggregate:
        db_session.add(effect)
        await db_session.flush()
    targets = [{"kind": effect.target_type, "id": str(effect.target_id)}]
    observed = {"agent_id": str(rows["planner"].id), "role": rows["planner"].role} if aggregate else {
        "task_id": str(task.id), "status": "done",
    }
    if artifact:
        artifact_row = Artifact(project_id=rows["goal"].project_id, name="verified-plan.md", artifact_type="plan",
                                content_hash=SHA256, status="ready", linked_task_id=task.id)
        db_session.add(artifact_row)
        await db_session.flush()
        targets.append({"kind": "artifact", "id": str(artifact_row.id)})
        observed = {"artifact_id": str(artifact_row.id), "artifact_type": artifact_row.artifact_type,
                    "content_hash": artifact_row.content_hash}
        rows["artifact"] = artifact_row
    receipt = ToolReceiptV1(
        proposal_id=proposal.proposal_id, call_id=call.call_id, tool=f"{call.tool}@{call.tool_version}",
        wrapper_action_id=wrapper.id, action_id=effect.id, idempotency_key=effect.idempotency_key,
        canonical_request_hash=_c3_hash(call.arguments), execution_identity={
            "proposal_id": str(proposal.proposal_id), "call_id": call.call_id,
            "authority_decision_id": str(rows["authority"].id), "purpose": "apply",
        }, status="applied", target_refs=targets, observed_post_state=observed,
        evidence=[EvidenceV1(label="The completed effect is bound to the approved call.",
                             value={"status": "completed"}, source_kind="action", source_id=effect.id)],
    )
    wrapper.dispatch_contract = {
        "authority_decision_id": str(rows["authority"].id), "canonical_request_hash": _c3_hash(call.arguments),
        "effect_action_id": str(effect.id), "effect_idempotency_key": effect.idempotency_key,
        "tool_receipt_v1": receipt.model_dump(mode="json"),
    }
    await db_session.flush()
    rows.update(wrapper=wrapper, effect=effect, receipt=receipt)


@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "changed"), (
    ("split", "matching"), ("aggregate", "matching"), ("aggregate", "receipt_wrapper"), ("aggregate", "final_effect_key"),
    *(("split", item) for item in (
        "authority", "contract_authority", "receipt_wrapper", "proposal", "call", "proposal_hash", "final_effect_key",
        "tool_version", "request_hash", "effect_request", "target_ids", "effect_link", "effect_status", "missing_receipt",
        "failed_receipt", "unknown_effect", "waiting_without_live_producer", "foreign_goal", "foreign_run", "foreign_project",
    )),
))
@pytest.mark.usefixtures("seal_without_post_state_validation")
async def test_c3_action_completion_requires_immutable_receipt_wrapper_effect_and_approval_identity(
    db_session, test_project, mode, changed,
):
    rows = await _c3_context(db_session, test_project, configuration=mode == "aggregate")
    if mode == "aggregate":
        await _c3_accepted_configuration(db_session, rows)
    await _c3_bound_receipt(db_session, rows, aggregate=mode == "aggregate")
    wrapper, effect = rows["wrapper"], rows["effect"]
    contract = deepcopy(wrapper.dispatch_contract)
    receipt = contract["tool_receipt_v1"]
    if changed == "authority":
        receipt["execution_identity"]["authority_decision_id"] = str(UUID(int=97_001))
    elif changed == "contract_authority":
        contract["authority_decision_id"] = str(UUID(int=97_001))
    elif changed == "receipt_wrapper":
        receipt["wrapper_action_id"] = str(UUID(int=97_002))
    elif changed in {"proposal", "call", "tool_version", "request_hash", "target_ids"}:
        if changed == "proposal":
            receipt["proposal_id"] = str(UUID(int=97_003))
        elif changed == "call":
            receipt["call_id"] = "another-call"
        elif changed == "tool_version":
            receipt["tool"] = f"{rows['proposal'].calls[0].tool}@2"
        elif changed == "request_hash":
            receipt["canonical_request_hash"] = "b" * 64
        else:
            receipt["target_refs"] = [{"kind": effect.target_type, "id": str(UUID(int=97_004))}]
    elif changed == "proposal_hash":
        wrapper.request = {**wrapper.request, "proposal_hash": "b" * 64}
    elif changed == "final_effect_key":
        effect.idempotency_key = "unapproved-effect-key"
    elif changed == "effect_request":
        effect.request = {**effect.request, "scope": "changed scope"}
    elif changed == "effect_link":
        contract["effect_action_id"] = str(UUID(int=97_005))
    elif changed == "effect_status":
        effect.status = "failed"
    elif changed == "missing_receipt":
        contract.pop("tool_receipt_v1")
    elif changed in {"failed_receipt", "unknown_effect", "waiting_without_live_producer"}:
        receipt["status"] = {"failed_receipt": "failed", "unknown_effect": "effect_unknown",
                             "waiting_without_live_producer": "waiting"}[changed]
        if changed == "waiting_without_live_producer":
            receipt["wait"] = {"events": [{"event_type": "session.completed", "matcher": {"task_id": str(rows["task"].id)}}],
                               "recheck_after_seconds": 30, "expected_result": "The linked task finishes."}
    elif changed.startswith("foreign"):
        if changed == "foreign_project":
            project = Project(name="Foreign completed effect")
            db_session.add(project)
            await db_session.flush()
        else:
            project = test_project
        foreign_goal, foreign_run = await _seed_goal_run(db_session, project, objective="Foreign action lineage.")
        if changed == "foreign_run":
            foreign_run.goal_id = rows["goal"].id
            foreign_run.status = "completed"
        effect.run_id = foreign_run.id
    wrapper.dispatch_contract = contract
    await db_session.flush()
    result = await RecoveryToolRegistry().verify(db_session, rows["proposal"], _c3_check("action.completed", {
        "call_id": rows["proposal"].calls[0].call_id,
    }))
    assert result.status == ("passed" if changed == "matching" else "failed")
    assert any(item.source_kind == "action" and item.source_id in {wrapper.id, effect.id} for item in result.evidence)


@pytest.mark.asyncio
@pytest.mark.parametrize(("source", "changed"), (
    ("resource", "matching"), ("call", "matching"),
    *(("resource", item) for item in ("unbound", "content", "type", "receipt_content", "receipt_target",
                                      "foreign_goal", "foreign_run", "foreign_project", "deleted")),
    ("call", "unbound"), ("call", "content"), ("call", "deleted"),
))
async def test_c3_artifact_requires_exact_receipt_content_and_linked_task_goal_project_run(
    db_session, test_project, source, changed,
):
    rows = await _c3_context(db_session, test_project)
    await _c3_bound_receipt(db_session, rows, artifact=True)
    artifact, wrapper = rows["artifact"], rows["wrapper"]
    if changed == "unbound":
        contract = deepcopy(wrapper.dispatch_contract)
        contract.pop("tool_receipt_v1")
        wrapper.dispatch_contract = contract
    elif changed == "content":
        artifact.content_hash = "b" * 64
    elif changed == "type":
        artifact.artifact_type = "report"
    elif changed == "deleted":
        artifact.status = "deleted"
    elif changed in {"receipt_content", "receipt_target"}:
        contract = deepcopy(wrapper.dispatch_contract)
        if changed == "receipt_content":
            contract["tool_receipt_v1"]["observed_post_state"]["content_hash"] = "b" * 64
        else:
            contract["tool_receipt_v1"]["target_refs"] = [ref for ref in contract["tool_receipt_v1"]["target_refs"]
                                                         if ref["kind"] != "artifact"]
        wrapper.dispatch_contract = contract
    elif changed.startswith("foreign"):
        if changed == "foreign_project":
            project = Project(name="Foreign artifact project")
            db_session.add(project)
            await db_session.flush()
        else:
            project = test_project
        foreign_goal, foreign_run = await _seed_goal_run(db_session, project, objective="Foreign artifact lineage.")
        if changed == "foreign_run":
            foreign_run.goal_id = rows["goal"].id
            foreign_run.status = "completed"
        rows["task"].project_id = project.id
        rows["task"].metadata_ = {"orchestration": {"run_id": str(foreign_run.id), "work_function": "planning"}}
        if changed == "foreign_project":
            artifact.project_id = project.id
    arguments = {"source": source, "project_id": str(test_project.id), "goal_id": str(rows["goal"].id),
                 "artifact_type": "plan", "content_hash": SHA256}
    arguments["artifact_id" if source == "resource" else "call_id"] = str(artifact.id) if source == "resource" else rows["proposal"].calls[0].call_id
    await db_session.flush()
    result = await RecoveryToolRegistry().verify(db_session, rows["proposal"], _c3_check("artifact.exists", arguments))
    assert result.status == ("passed" if changed == "matching" else "failed")
    if changed == "matching":
        assert any(item.source_kind == "artifact" and item.source_id == artifact.id for item in result.evidence)
    else:
        assert any((item.source_kind == "action" and item.source_id == wrapper.id)
                   or (item.source_kind == "run" and item.source_id == rows["run"].id) for item in result.evidence)


@pytest.mark.asyncio
@pytest.mark.parametrize("verifier", ("action.completed", "task.completed"))
@pytest.mark.parametrize("producer", ("live", "absent", "missing_named_session", "completed_named_session",
                                      "already_consumable_event"))
async def test_c3_waiting_requires_normalized_task_wake_and_real_live_session(
    db_session, test_project, verifier, producer,
):
    rows = await _c3_context(db_session, test_project)
    await _c3_bound_receipt(db_session, rows)
    task = rows["task"]
    task.status = "in_progress"
    if producer != "absent":
        db_session.add(Session(task_id=task.id, agent_id=rows["planner"].id, project_id=test_project.id,
                               adapter_type="api", status="running", input_context={}))
    wake = {"events": [{"event_type": "session.completed", "matcher": {"task_id": str(task.id)}}],
            "recheck_after_seconds": 30, "expected_result": "The linked task finishes."}
    if producer == "missing_named_session":
        wake["events"][0]["matcher"]["session_id"] = str(UUID(int=98_001))
    elif producer == "completed_named_session":
        completed = Session(task_id=task.id, agent_id=rows["planner"].id, project_id=test_project.id,
                            adapter_type="api", status="completed", input_context={})
        db_session.add(completed)
        await db_session.flush()
        wake["events"][0]["matcher"]["session_id"] = str(completed.id)
    elif producer == "already_consumable_event":
        wake["events"][0]["event_type"] = "session.created"
    contract = deepcopy(rows["wrapper"].dispatch_contract)
    receipt = contract["tool_receipt_v1"]
    receipt.update(status="waiting", wait=wake)
    assert ToolReceiptV1.model_validate(receipt).wait.model_dump(mode="json") == wake
    rows["wrapper"].dispatch_contract = contract
    await db_session.flush()
    arguments = {"call_id": rows["proposal"].calls[0].call_id}
    if verifier == "task.completed":
        arguments.update(source="call", goal_id=str(rows["goal"].id), run_id=str(rows["run"].id))
    result = await RecoveryToolRegistry().verify(db_session, rows["proposal"], _c3_check(verifier, arguments))
    assert result.status == ("pending" if producer == "live" else "failed")
    assert (result.wake.model_dump(mode="json") if result.wake else None) == (wake if producer == "live" else None)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ("missing_origin", "hash_mismatch"))
async def test_c3_invalid_origin_evidence_does_not_relabel_reported_warning_as_action(db_session, test_project, invalid):
    rows = await _c3_context(db_session, test_project)
    warning = OrchestrationWarning(goal_id=rows["goal"].id, run_id=rows["run"].id,
                                   warning_type="reported_blocker", severity="warning", message="Reported blocker.", active=True)
    db_session.add(warning)
    await db_session.flush()
    proposal = rows["proposal"].model_copy(update={"source": rows["proposal"].source.model_copy(update={
        "kind": "reported_blocker", "id": warning.id,
    })})
    rows["proposal_action"].request = proposal.model_dump(mode="json")
    rows["proposal_action"].dispatch_contract = {"proposal_hash": "b" * 64}
    if invalid == "missing_origin":
        await db_session.delete(rows["proposal_action"])
    await db_session.flush()
    result = await RecoveryToolRegistry().verify(db_session, proposal, _c3_check("action.completed", {
        "call_id": proposal.calls[0].call_id,
    }))
    assert result.status == "failed"
    assert result.evidence
    assert all(not (item.source_kind == "action" and item.source_id == warning.id) for item in result.evidence)
    assert all((item.source_kind == "warning" and item.source_id == warning.id)
               or (invalid == "hash_mismatch" and item.source_kind == "action"
                   and item.source_id == rows["proposal_action"].id) for item in result.evidence)


@pytest.mark.asyncio
@pytest.mark.parametrize("target_kind", ("meeting", "graph_run"))
@pytest.mark.parametrize("relationship", ("matching", "foreign_run", "wrong_sealed_task"))
async def test_c3_indirect_effect_task_must_match_sealed_call_and_origin_run(
    db_session, test_project, target_kind, relationship,
):
    rows = await _c3_context(db_session, test_project)
    task, goal, run, planner = (rows[key] for key in ("task", "goal", "run", "planner"))
    gate = OrchestrationGate(run_id=run.id, success_criterion_key="coordination", gate_type="review", status="open")
    graph = Graph(project_id=test_project.id, name="receipt-coordination", version="1", definition={}, triggers=[], is_active=True)
    db_session.add_all([gate, graph])
    await db_session.flush()
    task.metadata_ = {"orchestration": {"run_id": str(run.id), "gate_id": str(gate.id), "work_function": "planning"}}
    await db_session.flush()
    if target_kind == "meeting":
        tool, label = "orchestration.schedule_meeting@1", "schedule meeting"
        arguments = {"topic": "Verify coordination", "participant_agent_ids": [str(planner.id)],
                     "organizer_agent_id": str(planner.id), "task_id": str(task.id), "gate_id": str(gate.id)}
    else:
        tool, label = "orchestration.start_graph@1", "start graph"
        arguments = {"graph_id": str(graph.id), "subject_type": "task", "subject_id": str(task.id)}
    proposal = await RecoveryToolRegistry().canonicalize(db_session, goal, run,
        _trigger(goal.id, run.id).model_copy(update={"source_id": run.id}),
        _tool_draft(tool, arguments).model_copy(update={"general_updates": [label]}))
    rows["proposal"] = proposal
    rows["proposal_action"].request = proposal.model_dump(mode="json")
    rows["proposal_action"].dispatch_contract = {"proposal_hash": _c3_hash(proposal.model_dump(mode="json"))}
    await _c3_bound_receipt(db_session, rows)
    linked_task = task
    if relationship != "matching":
        linked_run = run
        if relationship == "foreign_run":
            _, linked_run = await _seed_goal_run(db_session, test_project, objective="Foreign linked task run.")
        linked_task = Task(project_id=test_project.id, title="Unapproved indirect target", status="done",
                           metadata_={"orchestration": {"run_id": str(linked_run.id)}})
        db_session.add(linked_task)
        await db_session.flush()
    if target_kind == "meeting":
        target = Meeting(project_id=test_project.id, title="Receipt meeting", meeting_type="coordination",
                         source_task_id=linked_task.id, status="completed", participant_agent_ids=[str(planner.id)])
    else:
        target = GraphRun(graph_id=graph.id, project_id=test_project.id, linked_task_id=linked_task.id,
                          current_node="done", status="completed")
    db_session.add(target)
    await db_session.flush()
    rows["effect"].action_type = "schedule_meeting" if target_kind == "meeting" else "start_graph"
    rows["effect"].target_type, rows["effect"].target_id = target_kind, target.id
    contract = deepcopy(rows["wrapper"].dispatch_contract)
    contract["tool_receipt_v1"]["target_refs"] = [{"kind": target_kind, "id": str(target.id)}]
    rows["wrapper"].dispatch_contract = contract
    await db_session.flush()
    result = await RecoveryToolRegistry().verify(db_session, proposal, _c3_check("action.completed", {
        "call_id": proposal.calls[0].call_id,
    }))
    assert result.status == ("passed" if relationship == "matching" else "failed")
    assert any(item.source_kind == "action" and item.source_id == rows["wrapper"].id for item in result.evidence)


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ("retry_task", "reassign_task", "request_plan_revision"))
@pytest.mark.parametrize("wrong_task", (False, True))
async def test_c3_explicit_task_calls_reject_coordinated_effect_and_receipt_target_substitution(
    db_session, test_project, tool, wrong_task,
):
    rows = await _c3_context(db_session, test_project)
    task, run = rows["task"], rows["run"]
    arguments = ({"plan_task_id": str(task.id), "revision_request": "Repair the approved plan."}
                 if tool == "request_plan_revision" else {"task_id": str(task.id)})
    if tool == "reassign_task":
        arguments["agent_id"] = str(rows["planner"].id)
    proposal = await RecoveryToolRegistry().canonicalize(db_session, rows["goal"], run,
        _trigger(rows["goal"].id, run.id).model_copy(update={"source_id": run.id}),
        _tool_draft(f"orchestration.{tool}@1", arguments).model_copy(update={
            "general_updates": [tool.replace("_", " ")],
        }))
    rows["proposal"] = proposal
    rows["proposal_action"].request = proposal.model_dump(mode="json")
    rows["proposal_action"].dispatch_contract = {"proposal_hash": _c3_hash(proposal.model_dump(mode="json"))}
    await _c3_bound_receipt(db_session, rows)
    linked_task = task
    if wrong_task:
        linked_task = Task(project_id=test_project.id, title="Wrong same-run task", status="done",
                           metadata_={"orchestration": {"run_id": str(run.id)}})
        db_session.add(linked_task)
        await db_session.flush()
    effect = rows["effect"]
    effect.action_type = tool
    effect.target_id = linked_task.id
    if tool == "retry_task":
        session = Session(task_id=linked_task.id, agent_id=rows["planner"].id, project_id=test_project.id,
                          adapter_type="api", status="completed", input_context={})
        db_session.add(session)
        await db_session.flush()
        effect.target_type, effect.target_id = "session", session.id
    contract = deepcopy(rows["wrapper"].dispatch_contract)
    contract["tool_receipt_v1"]["target_refs"] = [{"kind": effect.target_type, "id": str(effect.target_id)}]
    rows["wrapper"].dispatch_contract = contract
    await db_session.flush()
    result = await RecoveryToolRegistry().verify(db_session, proposal, _c3_check("action.completed", {
        "call_id": proposal.calls[0].call_id,
    }))
    assert result.status == ("failed" if wrong_task else "passed")
    assert any(item.source_kind == "action" and item.source_id == rows["wrapper"].id for item in result.evidence)


_C3_VERIFIER_ARGUMENTS = {
    "team_configuration.accepted": {"goal_id": str(GOAL_A_ID), "call_id": "call-1"},
    "planning.ready": {"goal_id": str(GOAL_A_ID), "run_id": str(RUN_A_ID)},
    "action.completed": {"call_id": "call-1"},
    "task.completed": {"source": "resource", "goal_id": str(GOAL_A_ID), "run_id": str(RUN_A_ID), "task_id": str(PLANNER_ID)},
    "gate.satisfied": {"goal_id": str(GOAL_A_ID), "run_id": str(RUN_A_ID), "gate_id": str(PLANNER_ID)},
    "process.completed": {"goal_id": str(GOAL_A_ID), "process_id": str(PLANNER_ID), "expected_fingerprint": SHA256},
    "warning.resolved": {"goal_id": str(GOAL_A_ID), "warning_id": str(PLANNER_ID)},
    "artifact.exists": {"source": "resource", "project_id": str(PROJECT_ID), "goal_id": str(GOAL_A_ID),
                        "artifact_id": str(PLANNER_ID), "artifact_type": "plan", "content_hash": SHA256},
}


@pytest.mark.parametrize("verifier", tuple(_C3_VERIFIER_ARGUMENTS))
@pytest.mark.parametrize("invalid", ("predicate", "expression", "code", "sql", "unknown", "missing_id", "missing_expected"))
def test_c3_all_eight_static_verifier_argument_models_reject_arbitrary_predicates_and_missing_expectations(verifier, invalid):
    from huddleroom.services.orchestration_recovery_registry import _VERIFIER_MODELS
    arguments = deepcopy(_C3_VERIFIER_ARGUMENTS[verifier])
    if invalid == "missing_id":
        primary = {"team_configuration.accepted": "call_id", "planning.ready": "run_id", "action.completed": "call_id",
                   "task.completed": "task_id", "gate.satisfied": "gate_id", "process.completed": "process_id",
                   "warning.resolved": "warning_id", "artifact.exists": "artifact_id"}[verifier]
        arguments.pop(primary)
    elif invalid == "missing_expected":
        if verifier == "artifact.exists":
            arguments.pop("content_hash")
        elif verifier == "process.completed":
            arguments["expected_fingerprint"] = "not-a-fingerprint"
        else:
            arguments["call_id" if verifier in {"action.completed", "team_configuration.accepted"} else "goal_id"] = ""
    else:
        arguments[invalid] = "status == completed"
    with pytest.raises(ValueError):
        TypeAdapter(_VERIFIER_MODELS[verifier]).validate_python(arguments)


@pytest.mark.asyncio
@pytest.mark.parametrize("expected,passed", [("b" * 64, True), (SHA256, False)])
async def test_g4_recovery_verifies_effective_comparison_without_rewriting_capture(
    db_session, test_project, expected, passed
):
    rows = await _c3_context(db_session, test_project)
    process = OrchestrationProcessRun(
        goal_id=rows["goal"].id, run_id=rows["run"].id, process_type="team_hierarchy",
        status="completed", trigger_reason="accepted",
        input_snapshot={"fingerprint": SHA256, "effective_comparison": {"fingerprint": "b" * 64}},
    )
    db_session.add(process)
    await db_session.flush()
    arguments = {"goal_id": str(rows["goal"].id), "process_id": str(process.id),
                 "expected_fingerprint": expected}
    result = await RecoveryToolRegistry().verify(
        db_session, rows["proposal"], _c3_check("process.completed", arguments)
    )
    assert (result.status == "passed") is passed
    assert process.input_snapshot["fingerprint"] == SHA256
