"""SPR 180/184: atomic direct team-configuration repair.

``team_configuration.apply@1`` turns one approved recovery proposal into agent rows plus
recovery-sourced accepted ``agent_definition_review`` / ``team_hierarchy`` snapshots, a
receipt and a completed apply wrapper -- all in one roster-first transaction, with no
LM analysis, no ``waiting_decision`` and no second approval.

This module also hosts the shared recovery reused by the cases in
test_orchestration_agent_definition_review.py and test_orchestration_team_hierarchy.py.

Interface the tests rely on (recovery, verbatim signatures):
- ``RecoveryConfigurationService().apply(db, proposal, call, wrapper, authority_decision)``
  -> ``ToolReceiptV1``; it owns (and commits) its roster-first transaction, so the caller
  passes an AUTOBEGIN session that holds no goal lock.
- ``RecoveryConfigurationService().reconcile(db, proposal, call, wrapper)`` -> receipt | None.
- ``AgentDefinitionReviewProcess().persist_recovery_acceptance(db, goal, run, *, accepted_payload,
  proposal_id, proposal_hash, authority_decision_id, evidence)`` and the same method on
  ``TeamHierarchyProcess()``; both return the written ``OrchestrationProcessRun``.
- A failed apply either raises or returns a receipt whose status is not ``applied``; in both
  cases nothing it wrote survives and the wrapper is left ``reserved`` (the executor of a
  later task records the lifecycle failure).
"""
from __future__ import annotations

import hashlib
import json
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from huddleroom.models.agent import Agent
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_memory import OrchestrationMemorySection
from huddleroom.models.orchestration_process import (
    OrchestrationAgentReview,
    OrchestrationAuthorityDecision,
    OrchestrationProcessRun,
    OrchestrationWarning,
)
from huddleroom.models.project import Project
from huddleroom.schemas.orchestration_recovery import (
    ProposalDraftV1,
    RecoveryProposalV1,
    RecoveryTriggerV1,
    ToolReceiptV1,
    VerificationCheckV1,
)
from huddleroom.services.orchestration_agent_definition_analyzer import AgentDefinitionSemanticAnalyzer
from huddleroom.services.orchestration_agent_definition_review import (
    AgentDefinitionReviewProcess,
    assess_agent_definition,
)
from huddleroom.services.orchestration_blocker_recovery import OrchestrationBlockerRecoveryService
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_recovery_configuration import RecoveryConfigurationService
from huddleroom.services.orchestration_recovery_registry import RecoveryToolRegistry
from huddleroom.services.orchestration_roster_mapper import WEAK_FIT_THRESHOLD, OrchestrationRosterMapper
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_team_hierarchy import (
    TeamHierarchyProcess,
    accepted_role_map,
    select_plan_author,
)
from huddleroom.services.orchestration_team_hierarchy_analyzer import TeamHierarchyAnalyzer
from tests.conftest import complete_baseline_processes
from tests.test_orchestration_roster_mutation import (  # noqa: F401 -- fixtures re-exported
    assert_roster_first_publication,
    assign,
    lock_tracer,
    make_agent,
    seed_goal_run,
    seed_project_user,
    sessions,
    roster_api,
)

TRIGGER_REASON = "approved_recovery"
PROVENANCE_KEYS = ("proposal_id", "proposal_hash", "call_id", "canonical_request_hash", "authority_decision_id")
SURFACE_BULLETS = ["apply team configuration", "agent definitions", "eligibility", "assignments", "manager",
                   "reporting hierarchy", "documented gaps"]
PLANNER_DESCRIPTION = "Plans scoped release work into ordered milestones and summarizes the plan outcome."
PLANNER_PROMPT = "Produce the delivery plan, then summarize progress for the team."
PLANNER_DEFINITION = {
    "name": "Recovery Planner", "role": "planner", "description": PLANNER_DESCRIPTION,
    "provider": "openai", "model": "gpt-4o-mini", "system_prompt": PLANNER_PROMPT,
    "adapter_type": "api", "capabilities": ["planning", "summarization"], "config": {},
}
PLANNER_UPDATE = {"description": PLANNER_DESCRIPTION, "system_prompt": PLANNER_PROMPT,
                  "capabilities": ["planning", "summarization"]}
PLANNER_FUNCTIONS = ["planning", "summarization"]
REVIEW_KEY_PREFIXES = ("agent_definition_review:", "team_hierarchy:")


def sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


# --------------------------------------------------------------------------- seeding


async def seed_spr180(sessions, tmp_path, *, mode="create", independent=False, extra_inactive=False):
    """SPR 180 state: an accepted baseline whose delivery plan has no eligible author.

    ``create``: no planning agent exists at all.  ``update``: the hierarchy assigns planning to
    a vague planner whose accepted eligibility is empty (select_plan_author raises 409).
    """
    async with sessions() as s:
        project, user = await seed_project_user(s, tmp_path)
        implementer = make_agent(
            name="release-implementer", role="developer", adapter_type="cli", cli_runtime="codex",
            description="Implements scoped product changes and summarizes the shipped result.",
            system_prompt="Implement the requested change, verify it, then summarize the result.",
            capabilities=["implementation", "summarization"],
        )
        agents = [implementer]
        planner = None
        if mode == "update":
            # Valid at hierarchy time (the eligibility guard rejects ineligible assignments);
            # degraded to the legacy stale/vague state after the baseline completes below.
            planner = make_agent(name="release-planner", role="planner", description=PLANNER_DESCRIPTION,
                                 system_prompt=PLANNER_PROMPT, capabilities=["planning"])
            agents.append(planner)
        inactive = None
        if extra_inactive:
            inactive = make_agent(name="retired-planner", role="planner", is_active=False,
                                  description=PLANNER_DESCRIPTION, system_prompt=PLANNER_PROMPT,
                                  capabilities=["planning"])
            agents.append(inactive)
        s.add_all(agents)
        await s.flush()
        goal, run = await seed_goal_run(s, project, user=user, objective="Implement the release API")
        goal.constraints = {"deadline": "end of sprint"}
        goal.planning_mode = "delivery"
        if independent:
            goal.success_criteria = [{"key": "done", "description": "The change is independently reviewed."}]
        await assign(s, project.id, run.id, implementer, "implementation")
        if planner is not None:
            await assign(s, project.id, run.id, planner, "planning")
        await complete_baseline_processes(s, goal, run)
        definition = await OrchestrationProcessService().get_current(s, goal.id, "agent_definition_review")
        if planner is not None:  # legacy stale eligibility (SPR 180): hierarchy predates the guard
            planner.description = planner.system_prompt = ""
            definition.outputs = {**definition.outputs,
                                  "eligibility": {**definition.outputs["eligibility"], str(planner.id): []}}
        await s.commit()
        reviews = {review.agent_id: review.id for review in (await s.scalars(select(OrchestrationAgentReview).where(
            OrchestrationAgentReview.goal_id == goal.id,
            OrchestrationAgentReview.source_process_run_id == definition.id))).all()}
        return SimpleNamespace(
            mode=mode, project_id=project.id, user_id=user.id, goal_id=goal.id, run_id=run.id,
            implementer_id=implementer.id, planner_id=planner.id if planner else None,
            inactive_id=inactive.id if inactive else None,
            manager_user_id=goal.manager_user_id, implementer_review_id=reviews.get(implementer.id),
            implementer_functions=list((definition.outputs or {}).get("eligibility", {}).get(str(implementer.id), [])),
            baseline_ids={kind: (await OrchestrationProcessService().get_current(s, goal.id, kind)).id
                          for kind in ("agent_definition_review", "team_hierarchy")},
        )


async def plan_author_or_none(s, goal_id, run_id):
    goal = await s.get(OrchestrationGoal, goal_id, populate_existing=True)
    run = await s.get(OrchestrationRun, run_id, populate_existing=True)
    try:
        return await select_plan_author(s, goal, run)
    except HTTPException:
        return None


def planner_ref(seeded) -> str:
    return "new:planner" if seeded.mode == "create" else str(seeded.planner_id)


def configuration_arguments(seeded, **overrides) -> dict:
    """The valid SPR 180 repair: an accepted planner, the unchanged implementer, full coverage."""
    planner, implementer = planner_ref(seeded), str(seeded.implementer_id)
    agents = ([{"operation": "create", "ref": "new:planner", "definition": dict(PLANNER_DEFINITION)}]
              if seeded.mode == "create" else
              [{"operation": "update", "agent_id": str(seeded.planner_id), "set": dict(PLANNER_UPDATE)}])
    arguments = {
        "agents": agents,
        "eligibility": [{"agent_ref": planner, "work_functions": list(PLANNER_FUNCTIONS)},
                        {"agent_ref": implementer, "work_functions": list(seeded.implementer_functions)}],
        "assignments": [{"work_function": "planning", "agent_ref": planner},
                        {"work_function": "implementation", "agent_ref": implementer},
                        {"work_function": "summarization", "agent_ref": planner}],
        "hierarchy": {"manager": {"kind": "human", "user_id": str(seeded.manager_user_id)},
                      "team_leads": [], "contributors": [planner, implementer], "reviewers": [],
                      "validators": [], "specialists": []},
        "documented_gaps": [],
    }
    arguments.update(overrides)
    return arguments


def proposal_draft(seeded, arguments) -> ProposalDraftV1:
    names = ["new:planner", "Recovery Planner"] if seeded.mode == "create" else ["release-planner"]
    names += ["release-implementer", "retired-planner"]
    gaps = [gap["code"] for gap in arguments.get("documented_gaps", [])]
    return ProposalDraftV1.model_validate({
        "kind": "proposal",
        "diagnosis": "No accepted agent can author the delivery plan.",
        "evidence": [{"label": "Planning preflight found no eligible plan author.",
                      "value": {"code": "plan_author_unavailable"}, "source_kind": "run",
                      "source_id": str(seeded.run_id)}],
        "problem": "Planning cannot start because no accepted agent can author the plan.",
        "general_updates": SURFACE_BULLETS + names + gaps,
        "expected_effect": "An accepted planner can author the plan; planning is retried once.",
        "calls": [{"tool": "team_configuration.apply@1", "arguments": arguments,
                   "rationale": "Accept a planning author for this goal."}],
    })


class DraftAnalyzer:
    """Deterministic stand-in for the recovery LM: returns one sealable draft."""

    def __init__(self, draft: ProposalDraftV1):
        self.draft = draft

    async def analyze(self, snapshot, *, project_id, evidence_query, on_raw=None):
        if on_raw is not None:
            on_raw(self.draft.model_dump_json())
        return self.draft


def trigger_for(seeded) -> RecoveryTriggerV1:
    return RecoveryTriggerV1.model_validate({
        "schema_version": 1, "goal_id": str(seeded.goal_id), "run_id": str(seeded.run_id),
        "source_kind": "planning_preflight", "source_id": str(seeded.run_id),
        "blocker_code": "plan_author_unavailable", "summary": "No accepted planning author exists.",
        "source_fingerprint": "a" * 64, "retry_target": {"kind": "planning"},
    })


async def propose(sessions, seeded, arguments):
    """Diagnose (fake analyzer) -> sealed proposal + pending approve_fix decision, committed."""
    service = OrchestrationBlockerRecoveryService(analyzer=DraftAnalyzer(proposal_draft(seeded, arguments)))
    async with sessions() as s:
        trigger = await service.create_trigger(s, trigger_for(seeded))
        await s.commit()
        proposal_action = await service.diagnose(s, trigger.id)
        decision = (await s.scalars(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.related_action_id == proposal_action.id))).one()
        await s.commit()
        return SimpleNamespace(service=service, proposal_action_id=proposal_action.id, decision_id=decision.id)


async def approve(sessions, seeded, proposed):
    async with sessions() as s:
        decision = await s.get(OrchestrationAuthorityDecision, proposed.decision_id)
        reservation = await proposed.service.reserve_answer(
            s, decision, selected_option="approve_fix", reason=None, user_id=seeded.user_id)
        await s.commit()
    proposed.wrapper_id = reservation.first_wrapper_action_id
    return proposed


async def approved_flow(sessions, seeded, arguments=None):
    proposed = await propose(sessions, seeded, arguments or configuration_arguments(seeded))
    return await approve(sessions, seeded, proposed)


async def load_flow(s, flow):
    proposal_action = await s.get(OrchestrationAction, flow.proposal_action_id, populate_existing=True)
    proposal = RecoveryProposalV1.model_validate(proposal_action.request)
    wrapper = await s.get(OrchestrationAction, flow.wrapper_id, populate_existing=True)
    authority = await s.get(OrchestrationAuthorityDecision, flow.decision_id, populate_existing=True)
    return proposal, proposal.calls[0], wrapper, authority


async def apply_flow(sessions, flow, *, call=None, authority_id=None):
    """Run apply in a fresh AUTOBEGIN session; the outcome is a receipt or the raised exception.

    The session is closed without a caller commit: whatever apply did not commit itself is lost.
    """
    async with sessions() as s:
        proposal, sealed_call, wrapper, authority = await load_flow(s, flow)
        if authority_id is not None:
            authority = await s.get(OrchestrationAuthorityDecision, authority_id)
        try:
            return await RecoveryConfigurationService().apply(s, proposal, call or sealed_call, wrapper, authority)
        except Exception as exc:  # noqa: BLE001 -- failure contract allows raise-or-return
            return exc


def assert_not_applied(outcome):
    assert isinstance(outcome, Exception) or (isinstance(outcome, ToolReceiptV1) and outcome.status != "applied"), (
        f"invalid configuration was applied: {outcome!r}")
    # A missing entry point is not a valid rejection.
    assert not isinstance(outcome, AttributeError), outcome


async def snapshot_configuration_state(db, goal_id) -> dict:
    """Agent/goal/process/review/decision/action/warning/memory state relevant to an apply."""
    goal_ids = [goal_id] if isinstance(goal_id, uuid.UUID) else list(goal_id)

    def rows(result, *fields):
        return sorted((tuple(json.dumps(getattr(row, field), sort_keys=True, default=str) for field in fields)
                       for row in result), key=str)

    state = {"agents": rows((await db.scalars(select(Agent).execution_options(populate_existing=True))).all(),
                            "id", "name", "role", "description", "provider", "model", "system_prompt",
                            "adapter_type", "cli_runtime", "capabilities", "config", "is_active")}
    for gid in goal_ids:
        goal = await db.get(OrchestrationGoal, gid, populate_existing=True)
        run_ids = (await db.scalars(select(OrchestrationRun.id).where(OrchestrationRun.goal_id == gid))).all()
        state[str(gid)] = {
            "goal": (str(goal.manager_agent_id), str(goal.manager_user_id), goal.authority_model, goal.status),
            "processes": rows((await db.scalars(select(OrchestrationProcessRun).where(
                OrchestrationProcessRun.goal_id == gid).execution_options(populate_existing=True))).all(),
                "id", "process_type", "status", "trigger_reason", "superseded_by_id", "outputs", "input_snapshot"),
            "reviews": rows((await db.scalars(select(OrchestrationAgentReview).where(
                OrchestrationAgentReview.goal_id == gid))).all(), "id", "agent_id", "approved_for_work_functions"),
            "decisions": rows((await db.scalars(select(OrchestrationAuthorityDecision).where(
                OrchestrationAuthorityDecision.goal_id == gid).execution_options(populate_existing=True))).all(),
                "id", "decision_key", "status", "selected_option"),
            "actions": rows((await db.scalars(select(OrchestrationAction).where(
                OrchestrationAction.run_id.in_(run_ids)).execution_options(populate_existing=True))).all(),
                "id", "action_type", "status", "idempotency_key", "dispatch_contract", "error"),
            "warnings": rows((await db.scalars(select(OrchestrationWarning).where(
                OrchestrationWarning.goal_id == gid).execution_options(populate_existing=True))).all(),
                "id", "warning_type", "active"),
            "memory": rows((await db.scalars(select(OrchestrationMemorySection).where(
                OrchestrationMemorySection.goal_id == gid).execution_options(populate_existing=True))).all(),
                "section_key", "body"),
        }
    return state


async def fresh_snapshot(sessions, goal_id):
    async with sessions() as s:
        return await snapshot_configuration_state(s, goal_id)


async def current(s, goal_id, kind):
    return await OrchestrationProcessService().get_current(s, goal_id, kind)


def created_planner_id(proposal: RecoveryProposalV1, seeded) -> uuid.UUID:
    if seeded.mode == "update":
        return seeded.planner_id
    return uuid.UUID(next(item["agent_id"] for item in proposal.calls[0].arguments["agents"]
                          if item["operation"] == "create"))


async def pending_review_decisions(s, goal_id):
    return [row.decision_key for row in (await s.scalars(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.goal_id == goal_id,
        OrchestrationAuthorityDecision.status == "pending").execution_options(populate_existing=True))).all()
        if row.decision_key.startswith(REVIEW_KEY_PREFIXES)]


async def verify(s, proposal, verifier, arguments):
    check = VerificationCheckV1(check_id="check-t8", verifier=verifier, verifier_version="1", arguments=arguments)
    return await RecoveryToolRegistry().verify(s, proposal, check)


async def applied_flow(sessions, seeded, arguments=None):
    """Seeded -> approved -> applied; asserts the apply succeeded."""
    flow = await approved_flow(sessions, seeded, arguments)
    receipt = await apply_flow(sessions, flow)
    assert isinstance(receipt, ToolReceiptV1) and receipt.status == "applied", receipt
    flow.receipt = receipt
    return flow


def arm_semantic_guard(monkeypatch):
    """Recovery snapshots must never call the definition/hierarchy LM analyzers.

    Armed after seeding, because seeding itself runs the baseline with Safe* analyzers.
    """
    async def boom(*_args, **_kwargs):
        raise AssertionError("approved recovery invoked a semantic review")

    for cls in (AgentDefinitionSemanticAnalyzer, TeamHierarchyAnalyzer):
        for method in ("review", "review_request"):
            if hasattr(cls, method):
                monkeypatch.setattr(cls, method, boom)


# --------------------------------------------------------------------------- harness sanity


def test_harness_planner_definition_passes_deterministic_review_and_scores_strong():
    """Guards the oracle: the valid repair is not secretly rejected by hard disqualifiers or weak fit."""
    planner = make_agent(**PLANNER_DEFINITION)
    assert assess_agent_definition(planner, PLANNER_FUNCTIONS)["approved_for_work_functions"] == PLANNER_FUNCTIONS
    mapper = OrchestrationRosterMapper()
    assert all(mapper.score_agent_definition(planner, fn).score >= WEAK_FIT_THRESHOLD for fn in PLANNER_FUNCTIONS)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["create", "update"])
async def test_harness_seeds_spr180_state_and_seals_an_approvable_proposal(sessions, tmp_path, mode):
    """Guards the oracle: Tasks 5/6 already seal and approve the valid repair (no code involved)."""
    seeded = await seed_spr180(sessions, tmp_path, mode=mode)
    async with sessions() as s:
        assert await plan_author_or_none(s, seeded.goal_id, seeded.run_id) is None
    assert seeded.implementer_functions == ["implementation"]
    flow = await approved_flow(sessions, seeded)
    async with sessions() as s:
        _proposal, call, wrapper, authority = await load_flow(s, flow)
    assert call.tool == "team_configuration.apply" and wrapper.status == "reserved"
    assert authority.status == "answered" and authority.selected_option == "approve_fix"


@pytest.mark.asyncio
async def test_global_resource_fingerprint_is_not_staled_by_its_own_recovery_approval(sessions, tmp_path):
    """The sealed roster-wide fingerprint must survive the proposal's own approval card and answer.

    Otherwise every global configuration proposal is cancelled as ``state_changed`` on approval
    and the apply-time revalidation can never match.
    """
    seeded = await seed_spr180(sessions, tmp_path)
    proposed = await propose(sessions, seeded, configuration_arguments(seeded))

    async def recaptured():
        async with sessions() as s:
            action = await s.get(OrchestrationAction, proposed.proposal_action_id)
            proposal = RecoveryProposalV1.model_validate(action.request)
            goal = await s.get(OrchestrationGoal, seeded.goal_id)
            run = await s.get(OrchestrationRun, seeded.run_id)
            snapshot = await RecoveryToolRegistry().capture_resources(s, goal, run, proposal)
            return snapshot.fingerprint, proposal.resource_fingerprint

    pending_fingerprint, sealed = await recaptured()
    assert pending_fingerprint == sealed, "the pending approve_fix card changed the sealed resource fingerprint"
    await approve(sessions, seeded, proposed)
    answered_fingerprint, sealed = await recaptured()
    assert answered_fingerprint == sealed, "answering approve_fix changed the sealed resource fingerprint"


# --------------------------------------------------------------------------- success


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["create", "update"])
async def test_plan_author_repair_atomically_persists_recovery_snapshots(sessions, tmp_path, monkeypatch, mode):
    seeded = await seed_spr180(sessions, tmp_path, mode=mode)
    async with sessions() as s:
        assert await plan_author_or_none(s, seeded.goal_id, seeded.run_id) is None, "precondition: SPR 180 state"
    flow = await approved_flow(sessions, seeded)
    arm_semantic_guard(monkeypatch)

    receipt = await apply_flow(sessions, flow)

    assert isinstance(receipt, ToolReceiptV1), receipt
    assert receipt.status == "applied"
    async with sessions() as db_session:
        proposal, call, wrapper, _authority = await load_flow(db_session, flow)
        goal = await db_session.get(OrchestrationGoal, seeded.goal_id)
        run = await db_session.get(OrchestrationRun, seeded.run_id)
        planner = await db_session.get(Agent, created_planner_id(proposal, seeded))
        assert planner is not None and planner.is_active
        assert planner.description == PLANNER_DESCRIPTION and planner.system_prompt == PLANNER_PROMPT
        assert await select_plan_author(db_session, goal, run) == planner.id
        definition_run = await current(db_session, goal.id, "agent_definition_review")
        hierarchy_run = await current(db_session, goal.id, "team_hierarchy")
        assert definition_run.trigger_reason == hierarchy_run.trigger_reason == "approved_recovery"
        assert definition_run.status == hierarchy_run.status == "completed"
        for kind, row in (("agent_definition_review", definition_run), ("team_hierarchy", hierarchy_run)):
            superseded = await db_session.get(OrchestrationProcessRun, seeded.baseline_ids[kind])
            assert superseded.superseded_by_id is not None, f"{kind} accepted snapshot was not superseded"
            assert row.run_id == run.id
        assert definition_run.outputs["eligibility"][str(planner.id)] == PLANNER_FUNCTIONS
        assert accepted_role_map(hierarchy_run.outputs)["planning"] == str(planner.id)
        pending = await pending_review_decisions(db_session, goal.id)
        assert pending == []
        never_parked = (await db_session.scalars(select(OrchestrationProcessRun).where(
            OrchestrationProcessRun.goal_id == goal.id, OrchestrationProcessRun.trigger_reason == TRIGGER_REASON,
            OrchestrationProcessRun.status != "completed"))).all()
        assert never_parked == [], "recovery snapshots must not enter waiting_decision/running"
        assert await OrchestrationService()._baseline_processes_ready_for_goal(db_session, goal.id, allow_heal=False)
        assert wrapper.status == "completed"
        accepted = await verify(db_session, proposal, "team_configuration.accepted",
                                {"goal_id": str(goal.id), "call_id": call.call_id})
        ready = await verify(db_session, proposal, "planning.ready", {"goal_id": str(goal.id), "run_id": str(run.id)})
        assert accepted.status == "passed", accepted
        assert ready.status == "passed", ready


@pytest.mark.asyncio
async def test_database_aggregate_uses_wrapper_as_effect_identity(sessions, tmp_path, monkeypatch):
    seeded = await seed_spr180(sessions, tmp_path)
    flow = await approved_flow(sessions, seeded)
    arm_semantic_guard(monkeypatch)
    async with sessions() as s:
        actions_before = set((await s.scalars(select(OrchestrationAction.id))).all())

    receipt = await apply_flow(sessions, flow)

    assert isinstance(receipt, ToolReceiptV1), receipt
    async with sessions() as s:
        proposal, call, wrapper, authority = await load_flow(s, flow)
        key = f"recovery:{proposal.proposal_id}:apply:{call.call_id}"
        assert receipt.action_id == receipt.wrapper_action_id
        assert receipt.wrapper_action_id == wrapper.id
        assert receipt.idempotency_key == wrapper.idempotency_key == key
        assert receipt.proposal_id == proposal.proposal_id and receipt.call_id == call.call_id
        assert receipt.tool == "team_configuration.apply@1"
        assert receipt.canonical_request_hash == sha(call.arguments)
        identity = receipt.execution_identity
        assert (identity.proposal_id, identity.call_id, identity.authority_decision_id, identity.purpose) == (
            proposal.proposal_id, call.call_id, authority.id, "apply")
        assert {"kind": "agent", "id": str(created_planner_id(proposal, seeded))} in receipt.target_refs
        assert wrapper.status == "completed"
        contract = wrapper.dispatch_contract
        assert ToolReceiptV1.model_validate(contract["tool_receipt_v1"]) == receipt
        assert contract["effect_action_id"] == str(wrapper.id)
        assert contract["effect_idempotency_key"] == key
        assert contract["authority_decision_id"] == str(authority.id)
        assert contract["canonical_request_hash"] == sha(call.arguments)
        domain_effect_rows = (await s.scalars(select(OrchestrationAction).where(
            OrchestrationAction.id.not_in(actions_before)))).all()
        assert domain_effect_rows == [], "a database-only aggregate must not reserve a child effect action"
        completed = await verify(s, proposal, "action.completed", {"call_id": call.call_id})
        assert completed.status == "passed", completed


@pytest.mark.asyncio
async def test_apply_replay_and_reconcile_reconstruct_the_same_receipt(sessions, tmp_path, monkeypatch):
    seeded = await seed_spr180(sessions, tmp_path)
    flow = await approved_flow(sessions, seeded)
    arm_semantic_guard(monkeypatch)
    async with sessions() as s:
        proposal, call, wrapper, _authority = await load_flow(s, flow)
        assert await RecoveryConfigurationService().reconcile(s, proposal, call, wrapper) is None

    first = await apply_flow(sessions, flow)
    assert isinstance(first, ToolReceiptV1), first
    after_first = await fresh_snapshot(sessions, seeded.goal_id)

    replay = await apply_flow(sessions, flow)
    async with sessions() as s:
        proposal, call, wrapper, _authority = await load_flow(s, flow)
        reconciled = await RecoveryConfigurationService().reconcile(s, proposal, call, wrapper)

    assert replay == first
    assert reconciled == first
    assert await fresh_snapshot(sessions, seeded.goal_id) == after_first, "replay applied the effect twice"


# --------------------------------------------------------------------------- rollback


def _inject(monkeypatch, cls, point):
    """Wrap a public recovery helper so the aggregate fails between its steps."""
    original = cls.persist_recovery_acceptance  # AttributeError (RED) until adds the helper

    async def failing(self, *args, **kwargs):
        if point == "after":
            await original(self, *args, **kwargs)
        raise RuntimeError(f"injected failure ({cls.__name__}:{point})")

    monkeypatch.setattr(cls, "persist_recovery_acceptance", failing)


@pytest.mark.asyncio
@pytest.mark.parametrize("point", ["definition_before", "definition_after", "hierarchy_before", "hierarchy_after"])
async def test_configuration_failure_rolls_back_every_effect(sessions, tmp_path, monkeypatch, point):
    seeded = await seed_spr180(sessions, tmp_path)
    flow = await approved_flow(sessions, seeded)
    before = await fresh_snapshot(sessions, seeded.goal_id)
    process, step = point.split("_")
    _inject(monkeypatch, AgentDefinitionReviewProcess if process == "definition" else TeamHierarchyProcess, step)
    calls = []
    original = TeamHierarchyProcess.persist_recovery_acceptance
    monkeypatch.setattr(TeamHierarchyProcess, "persist_recovery_acceptance",
                        lambda self, *a, **k: calls.append(1) or original(self, *a, **k))

    outcome = await apply_flow(sessions, flow)

    assert_not_applied(outcome)
    if process == "hierarchy":
        assert calls, "the hierarchy helper was never reached, so the injection proved nothing"
    async with sessions() as db_session:
        assert await snapshot_configuration_state(db_session, seeded.goal_id) == before
        wrapper = await db_session.get(OrchestrationAction, flow.wrapper_id)
        assert wrapper.status == "reserved"
        assert "tool_receipt_v1" not in (wrapper.dispatch_contract or {})


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ["edited_roster_member", "new_roster_member", "deactivated_assignee"])
async def test_stale_roster_after_approval_fails_without_effects(sessions, tmp_path, monkeypatch, drift):
    seeded = await seed_spr180(sessions, tmp_path)
    flow = await approved_flow(sessions, seeded)
    arm_semantic_guard(monkeypatch)
    async with sessions() as s:  # a concurrent global roster change lands after approval
        if drift == "new_roster_member":
            s.add(make_agent(name="unrelated-newcomer"))
        else:
            implementer = await s.get(Agent, seeded.implementer_id)
            if drift == "edited_roster_member":
                implementer.description = "Changed after the fix was approved."
            else:
                implementer.is_active = False
        await s.commit()
    before = await fresh_snapshot(sessions, seeded.goal_id)
    probe = []
    original = RecoveryConfigurationService.apply
    monkeypatch.setattr(RecoveryConfigurationService, "apply",
                        lambda self, *a, **k: probe.append(1) or original(self, *a, **k))

    outcome = await apply_flow(sessions, flow)

    assert probe, "apply was not reached"
    assert_not_applied(outcome)
    assert await fresh_snapshot(sessions, seeded.goal_id) == before


INVALID_CASES = (
    "cli_without_runtime", "hard_disqualification", "api_implementation", "unapproved_assignment",
    "inactive_agent", "missing_agent", "manager_mismatch", "unregistered_gap", "uncovered_function_without_gap",
)


def invalid_arguments(seeded, case) -> dict:
    arguments = configuration_arguments(seeded)
    planner = planner_ref(seeded)
    definition = arguments["agents"][0]["definition"]
    if case == "cli_without_runtime":
        definition.update(adapter_type="cli", cli_runtime=None)
    elif case == "hard_disqualification":
        definition["config"] = {"allow_global_scope": True}
    elif case == "api_implementation":
        arguments["eligibility"][0]["work_functions"] = ["planning", "implementation", "summarization"]
        arguments["assignments"][1]["agent_ref"] = planner
    elif case == "unapproved_assignment":
        arguments["assignments"][1]["agent_ref"] = planner  # planner is not accepted for implementation
    elif case in {"inactive_agent", "missing_agent"}:
        other = str(seeded.inactive_id) if case == "inactive_agent" else str(uuid.uuid4())
        arguments["eligibility"].append({"agent_ref": other, "work_functions": ["planning"]})
        arguments["assignments"][0]["agent_ref"] = other
        arguments["hierarchy"]["contributors"].append(other)
    elif case == "manager_mismatch":
        arguments["hierarchy"]["manager"] = {"kind": "human", "user_id": str(uuid.uuid4())}
    elif case == "unregistered_gap":
        arguments["documented_gaps"] = [{"code": "anything_goes", "rationale": "Skip every invariant."}]
        arguments["assignments"] = [item for item in arguments["assignments"] if item["work_function"] != "summarization"]
    elif case == "uncovered_function_without_gap":
        arguments["assignments"] = [item for item in arguments["assignments"] if item["work_function"] != "summarization"]
    return arguments


async def assert_rejected_or_rolled_back(sessions, seeded, arguments):
    """Rejected before approval (no approve_fix card) or at apply (full rollback) -- never applied."""
    pre_proposal = await fresh_snapshot(sessions, seeded.goal_id)
    try:
        flow = await approved_flow(sessions, seeded, arguments)
    except Exception:  # noqa: BLE001 -- rejected while sealing/approving
        after = await fresh_snapshot(sessions, seeded.goal_id)
        for key in ("goal", "processes", "reviews"):
            assert after[str(seeded.goal_id)][key] == pre_proposal[str(seeded.goal_id)][key]
        assert after["agents"] == pre_proposal["agents"]
        async with sessions() as s:
            cards = (await s.scalars(select(OrchestrationAuthorityDecision).where(
                OrchestrationAuthorityDecision.goal_id == seeded.goal_id,
                OrchestrationAuthorityDecision.decision_key.startswith("recovery:"),
                OrchestrationAuthorityDecision.status.in_(("pending", "answered"))))).all()
        assert cards == [], "an invalid configuration reached an approve_fix card"
        return "rejected_before_approval"
    before = await fresh_snapshot(sessions, seeded.goal_id)
    outcome = await apply_flow(sessions, flow)
    assert_not_applied(outcome)
    assert await fresh_snapshot(sessions, seeded.goal_id) == before
    return "rolled_back_at_apply"


@pytest.mark.asyncio
@pytest.mark.parametrize("case", INVALID_CASES)
async def test_invalid_configuration_never_takes_effect(sessions, tmp_path, monkeypatch, case):
    seeded = await seed_spr180(sessions, tmp_path, extra_inactive=case == "inactive_agent")
    arm_semantic_guard(monkeypatch)
    await assert_rejected_or_rolled_back(sessions, seeded, invalid_arguments(seeded, case))


@pytest.mark.asyncio
async def test_producer_verifier_conflict_without_override_never_takes_effect(sessions, tmp_path, monkeypatch):
    seeded = await seed_spr180(sessions, tmp_path, independent=True)
    arm_semantic_guard(monkeypatch)
    implementer = str(seeded.implementer_id)
    arguments = configuration_arguments(seeded)
    arguments["eligibility"][1]["work_functions"] = ["implementation", "review", "validation"]
    arguments["assignments"] += [{"work_function": "review", "agent_ref": implementer},
                                 {"work_function": "validation", "agent_ref": implementer}]
    arguments["hierarchy"].update(reviewers=[implementer], validators=[implementer])
    # The implementation producer is also its own reviewer and validator, with no documented override.
    await assert_rejected_or_rolled_back(sessions, seeded, arguments)


@pytest.mark.asyncio
@pytest.mark.parametrize("binding", ["tampered_arguments", "unknown_call", "unapproved_authority"])
async def test_apply_rejects_a_call_or_authority_not_bound_to_the_approved_proposal(
    sessions, tmp_path, monkeypatch, binding
):
    seeded = await seed_spr180(sessions, tmp_path)
    flow = await approved_flow(sessions, seeded)
    arm_semantic_guard(monkeypatch)
    before = await fresh_snapshot(sessions, seeded.goal_id)
    async with sessions() as s:
        _proposal, call, _wrapper, _authority = await load_flow(s, flow)
        other_authority = (await s.scalars(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.goal_id == seeded.goal_id,
            OrchestrationAuthorityDecision.decision_key == "team_hierarchy:approval"))).first()
    kwargs = {}
    if binding == "tampered_arguments":
        arguments = json.loads(json.dumps(call.arguments))
        arguments["eligibility"][0]["work_functions"] = ["planning"]
        kwargs["call"] = call.model_copy(update={"arguments": arguments})
    elif binding == "unknown_call":
        kwargs["call"] = call.model_copy(update={"call_id": "call-9"})
    else:
        kwargs["authority_id"] = other_authority.id
    probe = []
    original = RecoveryConfigurationService.apply
    monkeypatch.setattr(RecoveryConfigurationService, "apply",
                        lambda self, *a, **k: probe.append(1) or original(self, *a, **k))

    outcome = await apply_flow(sessions, flow, **kwargs)

    assert probe, "apply was not reached"
    assert_not_applied(outcome)
    assert await fresh_snapshot(sessions, seeded.goal_id) == before


# --------------------------------------------------------------------------- locks / collateral


@pytest.mark.asyncio
async def test_apply_publishes_roster_first_and_never_runs_under_a_goal_lock(
    sessions, tmp_path, monkeypatch, lock_tracer
):
    seeded = await seed_spr180(sessions, tmp_path)
    flow = await approved_flow(sessions, seeded)
    arm_semantic_guard(monkeypatch)
    before = await fresh_snapshot(sessions, seeded.goal_id)
    async with sessions() as s:  # a caller still holding the origin goal lock must fail before effects
        proposal, call, wrapper, authority = await load_flow(s, flow)
        async with OrchestrationService()._lock_goal_for_baseline_transition(s, seeded.goal_id):
            try:
                held = await RecoveryConfigurationService().apply(s, proposal, call, wrapper, authority)
            except Exception as exc:  # noqa: BLE001
                held = exc
    assert_not_applied(held)
    assert await fresh_snapshot(sessions, seeded.goal_id) == before
    lock_tracer.start()

    receipt = await apply_flow(sessions, flow)

    assert isinstance(receipt, ToolReceiptV1) and receipt.status == "applied", receipt
    assert not [event for event in lock_tracer.of("roster:acquire") if event.goals]
    assert_roster_first_publication(lock_tracer, [seeded.goal_id])
    async with sessions() as s:
        assert (await s.get(OrchestrationAction, flow.wrapper_id)).status == "completed"


@pytest.mark.asyncio
async def test_two_goal_repair_replaces_origin_and_invalidates_collateral_goal_atomically(
    sessions, tmp_path, monkeypatch
):
    rm = roster_api()
    seeded = await seed_spr180(sessions, tmp_path)
    async with sessions() as s:
        project = await s.get(Project, seeded.project_id)
        implementer = await s.get(Agent, seeded.implementer_id)
        other_goal, other_run = await seed_goal_run(s, project, objective="Implement the billing API")
        other_goal.constraints = {"deadline": "next sprint"}
        await assign(s, project.id, other_run.id, implementer, "implementation")
        await complete_baseline_processes(s, other_goal, other_run)
        await s.commit()
        other_goal_id = other_goal.id
        other_baseline = {kind: (await current(s, other_goal_id, kind)).id
                          for kind in ("agent_definition_review", "team_hierarchy")}
    flow = await approved_flow(sessions, seeded)
    arm_semantic_guard(monkeypatch)
    async with sessions() as s:
        proposal, call, _wrapper, _authority = await load_flow(s, flow)
    disclosed = set()
    for change in call.changes:
        if change.target_type == "configuration_fingerprint":
            disclosed.update(change.after if change.target_id == "aggregate" else [str(change.target_id)])
    assert str(other_goal_id) in disclosed, "precondition: the proposal discloses the collateral goal"

    receipt = await apply_flow(sessions, flow)

    assert isinstance(receipt, ToolReceiptV1) and receipt.status == "applied", receipt
    async with sessions() as s:
        origin = await current(s, seeded.goal_id, "team_hierarchy")
        assert origin.status == "completed" and origin.trigger_reason == TRIGGER_REASON
        for kind in ("agent_definition_review", "team_hierarchy"):
            row = await current(s, other_goal_id, kind)
            assert row.id != other_baseline[kind]
            assert row.status == rm.INVALIDATED_PROCESS_STATUS
        warnings = (await s.scalars(select(OrchestrationWarning).where(
            OrchestrationWarning.goal_id == other_goal_id, OrchestrationWarning.active.is_(True)))).all()
        assert any(row.related_action_id == flow.proposal_action_id for row in warnings)
        assert not await OrchestrationService()._baseline_processes_ready_for_goal(s, other_goal_id, allow_heal=False)
        other_definition = await current(s, other_goal_id, "agent_definition_review")
        # The origin's approval never leaks into the other goal's eligibility.
        planner = str(created_planner_id(proposal, seeded))
        assert planner not in ((other_definition.outputs or {}).get("eligibility") or {})


@pytest.mark.asyncio
async def test_process_history_never_claims_a_standard_review_ran(sessions, tmp_path, monkeypatch):
    seeded = await seed_spr180(sessions, tmp_path)
    arm_semantic_guard(monkeypatch)
    flow = await applied_flow(sessions, seeded)
    async with sessions() as s:
        _proposal, _call, _wrapper, authority = await load_flow(s, flow)
        history = await OrchestrationProcessService().list_process_runs(s, seeded.goal_id)
        recovery_rows = [row for row in history if row.trigger_reason == TRIGGER_REASON]
        assert sorted(row.process_type for row in recovery_rows) == ["agent_definition_review", "team_hierarchy"]
        for row in recovery_rows:
            assert row.status == "completed"
            outputs = row.outputs or {}
            assert not outputs.get("semantic_assessments"), "no semantic review ran for a recovery snapshot"
            assert outputs.get("approval") not in {"approve", "approve_with_documented_gaps", "edit"}
            assert outputs.get("approval_decision_id") in (None, str(authority.id))
            assert outputs["recovery_provenance"]["authority_decision_id"] == str(authority.id)
        sections = {row.section_key: row for row in (await s.scalars(select(OrchestrationMemorySection).where(
            OrchestrationMemorySection.goal_id == seeded.goal_id))).all()}
        for key in ("agent_definition_review", "team_hierarchy"):
            body = sections[key].body
            assert "approved recovery" in body.lower(), f"{key} memory does not disclose the recovery source"
            assert not body.startswith("Reviewed agent definitions"), f"{key} memory claims a standard review"


# --------------------------------------------------------------------------- seal-time rejection (follow-up)


async def recovery_cards(sessions, goal_id):
    async with sessions() as s:
        return (await s.scalars(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.goal_id == goal_id,
            OrchestrationAuthorityDecision.decision_key.startswith("recovery:")))).all()


@pytest.mark.asyncio
async def test_trivial_goal_configuration_is_rejected_while_sealing(sessions, tmp_path):
    """apply can never accept a trivial goal, so sealing must not create an approve_fix card."""
    seeded = await seed_spr180(sessions, tmp_path)
    async with sessions() as s:
        (await s.get(OrchestrationGoal, seeded.goal_id)).weight = "trivial"
        await s.commit()
    goal = SimpleNamespace(id=seeded.goal_id, weight="trivial")
    with pytest.raises(ValueError, match="unsupported_recovery_goal"):
        async with sessions() as s:
            from huddleroom.schemas.orchestration_recovery import TeamConfigurationApplyV1
            await RecoveryConfigurationService().prepare(
                s, goal, None, TeamConfigurationApplyV1.model_validate(configuration_arguments(seeded)))
    try:
        await propose(sessions, seeded, configuration_arguments(seeded))
    except Exception:  # noqa: BLE001 -- rejected while sealing
        pass
    assert await recovery_cards(sessions, seeded.goal_id) == []
