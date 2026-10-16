"""durable recovery trigger/proposal persistence (fake completions only)."""

import copy
import hashlib
import json
import uuid
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationDecision
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.schemas.orchestration_recovery import RecoveryProposalResponse, RecoveryProposalV1, RecoveryTriggerV1
from huddleroom.services.orchestration_blocker_recovery import (
    OrchestrationBlockerRecoveryService,
    RecoveryLifecycleStatus,
)
from huddleroom.services.orchestration_recovery_analyzer import RecoveryAnalyzer

SHA = "a" * 64


@pytest_asyncio.fixture
async def db_session(test_engine):
    """SPR230/SPR231 this persistence suite owns its preparation session."""
    async with AsyncSession(test_engine, expire_on_commit=False) as owned:
        yield owned
        await owned.rollback()


@pytest_asyncio.fixture
async def owned_recovery_setup(test_engine, tmp_path, monkeypatch):
    """Own preparation only for reset tests that need persisted recovery history."""
    from tests.test_orchestration_roster_mutation import seed_project_user
    async with AsyncSession(test_engine, expire_on_commit=False) as owned:
        project, user = await seed_project_user(owned, tmp_path)
        await owned.commit()
        owned.expunge(project)
        owned.expunge(user)
        original = OrchestrationBlockerRecoveryService.diagnose
        async def diagnose(service, db, trigger_id, **kwargs):
            if db is owned:
                await db.flush()
                await db.commit()
            return await original(service, db, trigger_id, **kwargs)
        monkeypatch.setattr(OrchestrationBlockerRecoveryService, "diagnose", diagnose)
        yield owned, project, user


@pytest.fixture(autouse=True)
def _owned_diagnosis_setup(monkeypatch, db_session):
    original = OrchestrationBlockerRecoveryService.diagnose
    async def diagnose(service, db, trigger_id, **kwargs):
        if db is db_session:
            # This test fixture owns staged setup, then enters actual diagnosis
            # with a fresh owned transaction; caller-contract tests live separately.
            await db.flush()
            await db.commit()
        return await original(service, db, trigger_id, **kwargs)
    monkeypatch.setattr(OrchestrationBlockerRecoveryService, "diagnose", diagnose)


def trigger_for(goal, run, *, fingerprint=SHA, source_id=None):
    return RecoveryTriggerV1.model_validate({
        "schema_version": 1, "goal_id": str(goal.id), "run_id": str(run.id),
        "source_kind": "planning_preflight", "source_id": str(source_id or uuid.uuid4()),
        "blocker_code": "planning_author_missing", "summary": "No accepted planning author exists.",
        "source_fingerprint": fingerprint, "retry_target": {"kind": "planning"},
    })


def completion_returning(payload):
    return AsyncMock(return_value={"choices": [{"message": {"content": json.dumps(payload)}}]})


def proposal_payload(agent):
    return {
        "kind": "proposal",
        "diagnosis": "The planning author is missing.",
        "evidence": [{
            "label": "Preflight failed.", "value": {"code": "planning_author_missing"},
            "source_kind": "agent", "source_id": str(agent.id),
        }],
        "problem": "Planning cannot start.",
        "general_updates": ["orchestration request plan"],
        "expected_effect": "Planning is ready for its one approved retry.",
        "calls": [{
            "tool": "orchestration.request_plan@1",
            "arguments": {"agent_id": str(agent.id), "work_function": "planning", "scope": "goal"},
            "rationale": "Ask the active agent to plan.",
        }],
    }


def gap_payload(agent):
    return {
        "kind": "capability_gap",
        "diagnosis": "No tool can create the missing credential.",
        "evidence": proposal_payload(agent)["evidence"],
        "missing_capability": "Provider credential provisioning",
        "required_human_action": "Provision the provider credential.",
        "wake_after": {"kind": "human_decision", "decision": "Credential provisioned"},
    }


def service_for(payload):
    # Interface assumption: constructor accepts an injected analyzer.
    return OrchestrationBlockerRecoveryService(analyzer=RecoveryAnalyzer(completion_fn=completion_returning(payload)))


async def actions_of(db, run, action_type):
    return (await db.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == action_type,
    ))).all()


async def approval_for(db, proposal):
    return (await db.scalars(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.related_action_id == proposal.id,
    ))).all()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@pytest.fixture
def goal_run(conversation_goal_run):
    goal, run = conversation_goal_run
    run.status = "running"
    return goal, run


def option_keys(options):
    return [o if isinstance(o, str) else o["key"] for o in options]


def test_lifecycle_status_enum_has_the_eight_spec_values():
    assert issubclass(RecoveryLifecycleStatus, str)
    assert {s.value for s in RecoveryLifecycleStatus} == {
        "pending_approval", "applying", "verifying", "retrying", "resolved", "rejected", "failed", "unsupported",
    }
    assert len(list(RecoveryLifecycleStatus)) == 8


@pytest.mark.asyncio
async def test_create_trigger_is_idempotent_on_run_source_and_fingerprint(db_session, goal_run):
    goal, run = goal_run
    svc = service_for({})
    trig = trigger_for(goal, run)
    first = await svc.create_trigger(db_session, trig)
    again = await svc.create_trigger(db_session, trig.model_copy())
    assert again.id == first.id
    assert len(await actions_of(db_session, run, "recovery_trigger")) == 1
    assert first.status == "reserved"


@pytest.mark.asyncio
async def test_new_fingerprint_or_source_creates_distinct_trigger(db_session, goal_run):
    goal, run = goal_run
    svc = service_for({})
    trig = trigger_for(goal, run)
    first = await svc.create_trigger(db_session, trig)
    changed = await svc.create_trigger(db_session, trig.model_copy(update={"source_fingerprint": "b" * 64}))
    other_source = await svc.create_trigger(db_session, trig.model_copy(update={"source_id": uuid.uuid4()}))
    assert len({first.id, changed.id, other_source.id}) == 3
    assert len(await actions_of(db_session, run, "recovery_trigger")) == 3


@pytest.mark.asyncio
async def test_diagnose_persists_recovery_exchange_proposal_and_one_human_approval(db_session, goal_run, test_agent):
    goal, run = goal_run
    svc = service_for(proposal_payload(test_agent))
    trigger = await svc.create_trigger(db_session, trigger_for(goal, run))
    proposal = await svc.diagnose(db_session, trigger.id)

    assert proposal.action_type == "recovery_proposal"
    assert proposal.status == "reserved"
    exchanges = (await db_session.scalars(select(OrchestrationDecision).where(
        OrchestrationDecision.run_id == run.id, OrchestrationDecision.decision_type == "recovery",
    ))).all()
    assert len(exchanges) == 1

    parsed = RecoveryProposalV1.model_validate(proposal.request)
    assert parsed.authority_required == "human"
    assert proposal.dispatch_contract["proposal_hash"] == canonical_hash(proposal.request)
    assert proposal.dispatch_contract["resource_fingerprint"] == proposal.request["resource_fingerprint"]
    assert proposal.dispatch_contract["lineage_id"] == proposal.request["lineage_id"]
    assert proposal.dispatch_contract["attempt"] == proposal.request["attempt"]

    approvals = await approval_for(db_session, proposal)
    assert len(approvals) == 1
    approval = approvals[0]
    assert approval.authority == "human" and approval.status == "pending"
    assert approval.authority_agent_id is None
    assert approval.related_action_id == proposal.id
    assert option_keys(approval.options) == ["approve_fix", "reject"]
    assert approval.decision_key.startswith("recovery:")
    assert str(parsed.proposal_id) in approval.decision_key
    assert proposal.dispatch_contract["proposal_hash"] in approval.decision_key


@pytest.mark.asyncio
async def test_approval_context_is_validated_recovery_proposal_v1_not_runtime_continuation(db_session, goal_run, test_agent):
    goal, run = goal_run
    svc = service_for(proposal_payload(test_agent))
    trigger = await svc.create_trigger(db_session, trigger_for(goal, run))
    proposal = await svc.diagnose(db_session, trigger.id)
    (approval,) = await approval_for(db_session, proposal)

    response = RecoveryProposalResponse.model_validate_json(approval.context)
    assert response.kind == "recovery_proposal_v1"
    assert str(response.proposal_id) == proposal.request["proposal_id"]
    assert response.status == "pending_approval"
    assert response.global_effect is False
    assert response.problem == proposal.request["problem"]
    assert approval.continuation is None
    assert approval.runtime_identity is None
    assert approval.contract_version is None


@pytest.mark.asyncio
async def test_approval_context_is_redacted(db_session, goal_run, test_agent):
    goal, run = goal_run
    payload = proposal_payload(test_agent)
    svc = service_for(payload)
    trigger = await svc.create_trigger(db_session, trigger_for(goal, run))
    proposal = await svc.diagnose(db_session, trigger.id)
    (approval,) = await approval_for(db_session, proposal)
    # Public payload must not smuggle credential-shaped text.
    assert "sk-" not in approval.context
    assert "api_key" not in approval.context.lower()


@pytest.mark.asyncio
async def test_repeat_diagnosis_returns_same_proposal_with_one_pending_approval_and_immutable_request(db_session, goal_run, test_agent):
    goal, run = goal_run
    svc = service_for(proposal_payload(test_agent))
    trigger = await svc.create_trigger(db_session, trigger_for(goal, run))
    first = await svc.diagnose(db_session, trigger.id)
    snapshot = copy.deepcopy(first.request)
    contract = copy.deepcopy(first.dispatch_contract)

    second = await svc.diagnose(db_session, trigger.id)

    assert second.id == first.id
    assert len(await actions_of(db_session, run, "recovery_proposal")) == 1
    pending = [a for a in await approval_for(db_session, first) if a.status == "pending"]
    assert len(pending) == 1
    assert second.request == snapshot
    assert second.dispatch_contract == contract


@pytest.mark.asyncio
async def test_identical_pending_trigger_after_diagnosis_reuses_the_same_rows(db_session, goal_run, test_agent):
    goal, run = goal_run
    svc = service_for(proposal_payload(test_agent))
    trig = trigger_for(goal, run)
    trigger = await svc.create_trigger(db_session, trig)
    proposal = await svc.diagnose(db_session, trigger.id)
    trigger_again = await svc.create_trigger(db_session, trig.model_copy())
    assert trigger_again.id == trigger.id
    assert (await svc.diagnose(db_session, trigger_again.id)).id == proposal.id


@pytest.mark.asyncio
async def test_diagnose_executes_no_effect_and_creates_no_wrapper(db_session, goal_run, test_agent):
    goal, run = goal_run
    svc = service_for(proposal_payload(test_agent))
    trigger = await svc.create_trigger(db_session, trigger_for(goal, run))
    await svc.diagnose(db_session, trigger.id)
    for kind in ("recovery_apply", "recovery_verify", "recovery_retry"):
        assert await actions_of(db_session, run, kind) == []


@pytest.mark.asyncio
async def test_project_status_of_fresh_proposal_is_pending_approval(db_session, goal_run, test_agent):
    goal, run = goal_run
    svc = service_for(proposal_payload(test_agent))
    trigger = await svc.create_trigger(db_session, trigger_for(goal, run))
    proposal = await svc.diagnose(db_session, trigger.id)
    assert await svc.project_status(db_session, proposal) == "pending_approval"


@pytest.mark.asyncio
async def test_capability_gap_has_no_approve_fix_and_projects_unsupported(db_session, goal_run, test_agent):
    # Assumption: a capability-gap diagnosis still returns a recovery_proposal action whose
    # projection is "unsupported" and which never offers approve_fix.
    goal, run = goal_run
    svc = service_for(gap_payload(test_agent))
    trigger = await svc.create_trigger(db_session, trigger_for(goal, run))
    result = await svc.diagnose(db_session, trigger.id)
    assert await svc.project_status(db_session, result) == "unsupported"
    decisions = (await db_session.scalars(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.goal_id == goal.id,
    ))).all()
    assert all("approve_fix" not in option_keys(d.options) for d in decisions)


# ---- follow-up --------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_diagnosis_persists_one_proposal_and_one_approval(concurrent_sessions, tmp_path):
    import asyncio

    from huddleroom.models.agent import Agent
    from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
    from huddleroom.models.project import Project

    s1, s2 = concurrent_sessions
    project = Project(name="Concurrent diagnosis", description="", workspace_path=str(tmp_path), config={})
    agent = Agent(name=f"a-{uuid.uuid4()}", role="planning", provider="openai", model="gpt-4o-mini",
                  adapter_type="api", capabilities=[], config={})
    s1.add_all([project, agent])
    await s1.flush()
    goal = OrchestrationGoal(project_id=project.id, objective="Plan the release.")
    s1.add(goal)
    await s1.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running")
    s1.add(run)
    await s1.flush()
    trigger = await service_for(proposal_payload(agent)).create_trigger(s1, trigger_for(goal, run))
    await s1.commit()

    async def diagnose(session):
        try:
            result = await service_for(proposal_payload(agent)).diagnose(session, trigger.id)
            await session.commit()
            return result.id
        except Exception as exc:  # the loser may hit a write conflict
            await session.rollback()
            return exc

    results = await asyncio.gather(diagnose(s1), diagnose(s2))
    winners = [r for r in results if isinstance(r, uuid.UUID)]
    assert winners, results
    assert len(set(winners)) == 1
    assert len(await actions_of(s1, run, "recovery_proposal")) == 1
    approvals = (await s1.scalars(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.goal_id == goal.id,
    ))).all()
    assert len(approvals) == 1


@pytest.mark.asyncio
async def test_diagnose_records_evidence_queries_and_redacted_raw_output(db_session, goal_run, test_agent):
    goal, run = goal_run
    query = {"kind": "evidence_query", "query": {"kind": "run_state", "run_id": str(run.id)}}
    completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": json.dumps(query)}}]},
        {"choices": [{"message": {"content": json.dumps(proposal_payload(test_agent))}}]},
    ])
    svc = OrchestrationBlockerRecoveryService(analyzer=RecoveryAnalyzer(completion_fn=completion))
    trigger = await svc.create_trigger(db_session, trigger_for(goal, run))
    await svc.diagnose(db_session, trigger.id)

    (exchange,) = (await db_session.scalars(select(OrchestrationDecision).where(
        OrchestrationDecision.run_id == run.id, OrchestrationDecision.decision_type == "recovery",
    ))).all()
    assert exchange.validator_status == "accepted"
    assert "context" in exchange.input_snapshot
    (record,) = exchange.input_snapshot["evidence_queries"]
    assert record["query"]["kind"] == "run_state"
    assert record["result"]["source_kind"] == "run" and record["result"]["source_id"] == str(run.id)
    assert isinstance(exchange.llm_output, list) and len(exchange.llm_output) == 2
    assert json.loads(exchange.llm_output[1])["kind"] == "proposal"


@pytest.mark.asyncio
async def test_analyzer_failure_is_recorded_on_a_rejected_exchange(db_session, goal_run, test_agent):
    from huddleroom.services.orchestration_blocker_recovery import RecoveryDiagnosisFailed

    goal, run = goal_run
    completion = AsyncMock(side_effect=RuntimeError("provider down api_key=sk-abcdefghijklmnopqrstuvwxyz123456"))
    svc = OrchestrationBlockerRecoveryService(analyzer=RecoveryAnalyzer(completion_fn=completion))
    trigger = await svc.create_trigger(db_session, trigger_for(goal, run))
    with pytest.raises(RecoveryDiagnosisFailed) as caught:
        await svc.diagnose(db_session, trigger.id)

    exchange = await db_session.get(OrchestrationDecision, caught.value.exchange_id)
    assert exchange.decision_type == "recovery" and exchange.validator_status == "rejected"
    assert exchange.rejection_reason.startswith("analysis_provider_failed")
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in exchange.rejection_reason
    await db_session.refresh(trigger)
    assert trigger.status == "reserved"
    assert await actions_of(db_session, run, "recovery_proposal") == []


@pytest.mark.asyncio
async def test_repeated_analyzer_outages_never_escalate_to_a_capability_gap(db_session, goal_run, test_agent):
    """F3: provider errors are not refusals of the diagnosis; they stamp the trigger for backoff instead."""
    from huddleroom.services.orchestration_blocker_recovery import (
        MAX_RECOVERY_ATTEMPTS_WITHOUT_PROGRESS, RecoveryDiagnosisFailed,
    )

    goal, run = goal_run
    completion = AsyncMock(side_effect=RuntimeError("provider down"))
    svc = OrchestrationBlockerRecoveryService(analyzer=RecoveryAnalyzer(completion_fn=completion))
    trigger = await svc.create_trigger(db_session, trigger_for(goal, run))
    for _ in range(MAX_RECOVERY_ATTEMPTS_WITHOUT_PROGRESS + 1):
        with pytest.raises(RecoveryDiagnosisFailed):
            await svc.diagnose(db_session, trigger.id)

    assert completion.await_count == MAX_RECOVERY_ATTEMPTS_WITHOUT_PROGRESS + 1, "an outage escalated without a call"
    await db_session.refresh(trigger)
    assert trigger.status == "reserved" and trigger.dispatch_contract.get("last_diagnosis_error_at")
    assert await svc._escalation(db_session, run, RecoveryTriggerV1.model_validate(trigger.request), trigger,
                                 trigger.dispatch_contract) is None


@pytest.mark.asyncio
async def test_admission_rejected_diagnoses_are_not_counted_as_refusals(db_session, goal_run, test_agent):
    goal, run = goal_run
    svc = service_for(gap_payload(test_agent))
    trigger = await svc.create_trigger(db_session, trigger_for(goal, run))
    for _ in range(3):
        db_session.add(OrchestrationDecision(
            run_id=run.id, decision_type="recovery", validator_status="rejected", parsed_decision={},
            input_snapshot={"trigger_action_id": str(trigger.id)},
            rejection_reason="baseline_repair_admission_rejected: the baseline changed since approval"))
    await db_session.flush()

    assert await svc._escalation(db_session, run, RecoveryTriggerV1.model_validate(trigger.request), trigger,
                                 trigger.dispatch_contract or {}) is None


@pytest.mark.asyncio
async def test_cancelling_a_capability_gap_decision_fails_its_gap_action(db_session, goal_run, test_agent):
    """F4: a gap decision cannot be bare-cancelled (stranding its reserved gap action); the closeout bulk
    cancel routes it through cancel_recovery_decision, which fails the gap action."""
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
    from huddleroom.services.orchestration_goal_closeout import GoalCloseoutProcess

    goal, run = goal_run
    svc = service_for(gap_payload(test_agent))
    trigger = await svc.create_trigger(db_session, trigger_for(goal, run))
    gap = await svc.diagnose(db_session, trigger.id)
    (decision,) = (await db_session.scalars(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.related_action_id == gap.id))).all()
    assert decision.decision_key.startswith("recovery_gap:") and gap.status == "reserved"

    with pytest.raises(ValueError):
        await OrchestrationAuthorityDecisionService().cancel_decision(db_session, decision, reason="bare")

    assert await GoalCloseoutProcess()._cancel_pending_decisions(db_session, goal.id) == [str(decision.id)]
    await db_session.refresh(decision)
    await db_session.refresh(gap)
    assert decision.status == "cancelled"
    assert gap.status == "failed" and gap.error == "goal closeout completed"
