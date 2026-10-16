"""recovery approval answer, reservation, drift and projection (fake completions only)."""

import asyncio
import copy
import dataclasses
import json
import uuid
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from huddleroom.models.agent import Agent
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.project import Project
from huddleroom.models.task import Task
from huddleroom.models.user import User
from huddleroom.schemas.orchestration_recovery import RecoveryTriggerV1
from huddleroom.services.orchestration_blocker_recovery import (
    OrchestrationBlockerRecoveryService,
    RecoveryAnswerReservation,
    RecoveryLifecycleStatus,
    RecoveryProposalStale,
)
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_recovery_analyzer import RecoveryAnalyzer
from huddleroom.services.orchestration_service import OrchestrationService

SHA = "a" * 64


def trigger_for(goal, run):
    return RecoveryTriggerV1.model_validate({
        "schema_version": 1, "goal_id": str(goal.id), "run_id": str(run.id),
        "source_kind": "planning_preflight", "source_id": str(uuid.uuid4()),
        "blocker_code": "planning_author_missing", "summary": "No accepted planning author exists.",
        "source_fingerprint": SHA, "retry_target": {"kind": "planning"},
    })


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


def service_for(agent):
    completion = AsyncMock(return_value={"choices": [{"message": {"content": json.dumps(proposal_payload(agent))}}]})
    # Interface assumption: constructor accepts an injected analyzer.
    return OrchestrationBlockerRecoveryService(analyzer=RecoveryAnalyzer(completion_fn=completion))


async def actions_of(db, run, action_type):
    return (await db.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == action_type,
    ))).all()


async def diagnosed(db, goal, run, agent, *, owns_setup=False):
    svc = service_for(agent)
    trigger = await svc.create_trigger(db, trigger_for(goal, run))
    if owns_setup:
        await db.flush()
        await db.commit()
    proposal = await svc.diagnose(db, trigger.id)
    decision = (await db.scalars(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.related_action_id == proposal.id,
    ))).one()
    return svc, trigger, proposal, decision


@pytest.fixture
def goal_run(conversation_goal_run):
    goal, run = conversation_goal_run
    run.status = "running"
    return goal, run


@pytest.fixture
def fail_on_effect(monkeypatch):
    """Any attempt to execute a registered call or ordinary dispatch inside the answer path fails the test."""
    from huddleroom.services import orchestration_decision_dispatcher as dispatcher
    from huddleroom.services.orchestration_recovery_registry import RecoveryToolRegistry

    async def boom(*args, **kwargs):
        raise AssertionError("effect executed during answer reservation")

    monkeypatch.setattr(RecoveryToolRegistry, "execute_call", boom)
    monkeypatch.setattr(dispatcher.OrchestrationDecisionDispatcher, "dispatch", boom, raising=False)


async def assert_no_effect_rows(db, run):
    rows = (await db.scalars(select(OrchestrationAction).where(OrchestrationAction.run_id == run.id))).all()
    assert [r for r in rows if r.idempotency_key.endswith(":effect")] == []
    assert (await db.scalars(select(Task).where(Task.project_id.is_not(None)))).all() == []


# ---- approval -----------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approve_records_answer_and_reserves_first_apply_wrapper(db_session, goal_run, test_agent, test_user, fail_on_effect):
    goal, run = goal_run
    svc, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)

    reservation = await svc.reserve_answer(
        db_session, decision, selected_option="approve_fix", reason=None, user_id=test_user.id,
    )

    assert isinstance(reservation, RecoveryAnswerReservation)
    assert dataclasses.is_dataclass(reservation) and len(dataclasses.fields(reservation)) == 3
    # Field order per brief: answered decision id, proposal action id, first wrapper action id.
    decision_id, proposal_id, wrapper_id = dataclasses.astuple(reservation)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(reservation, dataclasses.fields(reservation)[0].name, uuid.uuid4())

    await db_session.refresh(decision)
    assert decision.status == "answered" and decision.selected_option == "approve_fix"
    assert decision.decided_by_user_id == test_user.id
    assert decision_id == decision.id and proposal_id == proposal.id

    (wrapper,) = await actions_of(db_session, run, "recovery_apply")
    assert wrapper.id == wrapper_id and wrapper.status == "reserved"
    assert wrapper.idempotency_key == f"recovery:{proposal.request['proposal_id']}:apply:call-1"
    assert wrapper.request["proposal_action_id"] == str(proposal.id)
    assert wrapper.request["proposal_hash"] == proposal.dispatch_contract["proposal_hash"]
    assert wrapper.request["call_id"] == "call-1"
    assert await svc.project_status(db_session, proposal) == "applying"


@pytest.mark.asyncio
async def test_duplicate_approval_reserves_one_apply_wrapper_without_running_effect(db_session, goal_run, test_agent, test_user, fail_on_effect):
    goal, run = goal_run
    svc, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)

    first = await svc.reserve_answer(db_session, decision, selected_option="approve_fix", reason=None, user_id=test_user.id)
    second = await svc.reserve_answer(db_session, decision, selected_option="approve_fix", reason=None, user_id=test_user.id)

    apply_wrappers = await actions_of(db_session, run, "recovery_apply")
    effect_rows = [a for a in apply_wrappers if a.idempotency_key.endswith(":effect")]
    assert len(apply_wrappers) == 1
    assert effect_rows == []
    assert first == second
    await assert_no_effect_rows(db_session, run)


@pytest.mark.asyncio
async def test_conflicting_duplicate_answer_is_rejected_and_changes_nothing(db_session, goal_run, test_agent, test_user, fail_on_effect):
    goal, run = goal_run
    svc, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)
    await svc.reserve_answer(db_session, decision, selected_option="approve_fix", reason=None, user_id=test_user.id)

    with pytest.raises(ValueError):
        await svc.reserve_answer(db_session, decision, selected_option="reject", reason="changed my mind", user_id=test_user.id)

    await db_session.refresh(decision)
    assert decision.selected_option == "approve_fix"
    assert len(await actions_of(db_session, run, "recovery_apply")) == 1


@pytest.mark.asyncio
async def test_answer_rejects_options_outside_approve_fix_and_reject(db_session, goal_run, test_agent, test_user):
    goal, run = goal_run
    svc, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)
    for bad in ("edit", "approve", "", "APPROVE_FIX"):
        with pytest.raises(ValueError):
            await svc.reserve_answer(db_session, decision, selected_option=bad, reason="x", user_id=test_user.id)
    await db_session.refresh(decision)
    assert decision.status == "pending"
    assert await actions_of(db_session, run, "recovery_apply") == []


@pytest.mark.asyncio
async def test_answer_runs_under_the_goal_lock(db_session, goal_run, test_agent, test_user, monkeypatch, fail_on_effect):
    goal, run = goal_run
    svc, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)
    locked = []
    original = OrchestrationService._lock_goal_for_baseline_transition

    def spy(self, db, goal_id, **kwargs):
        locked.append(goal_id)
        return original(self, db, goal_id, **kwargs)

    monkeypatch.setattr(OrchestrationService, "_lock_goal_for_baseline_transition", spy)
    await svc.reserve_answer(db_session, decision, selected_option="approve_fix", reason=None, user_id=test_user.id)
    assert goal.id in locked


@pytest.mark.asyncio
async def test_approval_leaves_proposal_request_and_hash_immutable(db_session, goal_run, test_agent, test_user, fail_on_effect):
    goal, run = goal_run
    svc, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)
    request, contract = copy.deepcopy(proposal.request), copy.deepcopy(proposal.dispatch_contract)
    await svc.reserve_answer(db_session, decision, selected_option="approve_fix", reason=None, user_id=test_user.id)
    await db_session.refresh(proposal)
    assert proposal.request == request
    for key in ("proposal_hash", "resource_fingerprint", "lineage_id", "attempt"):
        assert proposal.dispatch_contract[key] == contract[key]


# ---- rejection ----------------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [None, "", "   "])
async def test_reject_requires_a_reason(db_session, goal_run, test_agent, test_user, reason):
    goal, run = goal_run
    svc, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)
    with pytest.raises(ValueError):
        await svc.reserve_answer(db_session, decision, selected_option="reject", reason=reason, user_id=test_user.id)
    await db_session.refresh(decision)
    await db_session.refresh(proposal)
    assert decision.status == "pending"
    assert proposal.status == "reserved"


@pytest.mark.asyncio
async def test_reject_with_reason_applies_nothing_and_projects_rejected(db_session, goal_run, test_agent, test_user, fail_on_effect):
    goal, run = goal_run
    svc, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)

    await svc.reserve_answer(db_session, decision, selected_option="reject", reason="Wrong agent for this.", user_id=test_user.id)

    await db_session.refresh(decision)
    await db_session.refresh(proposal)
    assert decision.status == "answered" and decision.selected_option == "reject"
    assert decision.reason == "Wrong agent for this."
    assert proposal.status == "failed" and proposal.error == "rejected_by_human"
    for kind in ("recovery_apply", "recovery_verify", "recovery_retry"):
        assert await actions_of(db_session, run, kind) == []
    assert await svc.project_status(db_session, proposal) == "rejected"
    await assert_no_effect_rows(db_session, run)


# ---- drift --------------------------------------------------------------------------------------

async def _assert_stale(db, svc, goal, run, proposal, decision, user, mutate):
    await mutate()
    await db.flush()
    # Domain exception; 's route catches it, commits the flushed writes, then returns 409.
    with pytest.raises(RecoveryProposalStale) as caught:
        await svc.reserve_answer(db, decision, selected_option="approve_fix", reason=None, user_id=user.id)
    assert caught.value.proposal_action_id == proposal.id and caught.value.decision_id == decision.id
    await db.refresh(decision)
    await db.refresh(proposal)
    assert decision.status == "cancelled"
    assert proposal.status == "failed"
    assert proposal.error == "state_changed"
    for kind in ("recovery_apply", "recovery_verify", "recovery_retry"):
        assert await actions_of(db, run, kind) == []
    await assert_no_effect_rows(db, run)
    assert await svc.project_status(db, proposal) == "failed"
    # Fresh diagnosis is scheduled: a reserved recovery trigger exists for the run again.
    assert [t for t in await actions_of(db, run, "recovery_trigger") if t.status == "reserved"]


@pytest.mark.asyncio
async def test_stale_approval_applies_nothing_and_schedules_diagnosis(db_session, goal_run, test_agent, test_user, fail_on_effect):
    goal, run = goal_run
    svc, trigger, proposal, decision = await diagnosed(db_session, goal, run, test_agent)
    # Complete the trigger so "scheduled" must be an explicit new/reopened reservation.
    trigger.status = "completed"

    async def drift():
        test_agent.role = "something else entirely"  # resource fingerprint covers the referenced agent

    await _assert_stale(db_session, svc, goal, run, proposal, decision, test_user, drift)


@pytest.mark.asyncio
async def test_run_state_drift_is_stale(db_session, goal_run, test_agent, test_user, fail_on_effect):
    goal, run = goal_run
    svc, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)

    async def drift():
        run.retry_state = {"attempt": 99}

    await _assert_stale(db_session, svc, goal, run, proposal, decision, test_user, drift)


@pytest.mark.asyncio
async def test_inactive_run_is_stale(db_session, goal_run, test_agent, test_user, fail_on_effect):
    goal, run = goal_run
    svc, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)

    async def drift():
        run.status = "cancelled"

    with pytest.raises(RecoveryProposalStale):
        await drift()
        await db_session.flush()
        await svc.reserve_answer(db_session, decision, selected_option="approve_fix", reason=None, user_id=test_user.id)
    await db_session.refresh(proposal)
    assert proposal.error == "state_changed"
    assert await actions_of(db_session, run, "recovery_apply") == []


@pytest.mark.asyncio
async def test_tampered_proposal_hash_is_stale(db_session, goal_run, test_agent, test_user, fail_on_effect):
    goal, run = goal_run
    svc, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)

    async def drift():
        tampered = copy.deepcopy(proposal.request)
        tampered["problem"] = "A different problem than the one approved."
        proposal.request = tampered

    await _assert_stale(db_session, svc, goal, run, proposal, decision, test_user, drift)


@pytest.mark.asyncio
async def test_registry_version_change_is_stale(db_session, goal_run, test_agent, test_user, monkeypatch, fail_on_effect):
    goal, run = goal_run
    svc, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)
    from huddleroom.services import orchestration_recovery_registry as registry_module

    async def drift():
        monkeypatch.setattr(registry_module, "REGISTRY_VERSION", "2")

    await _assert_stale(db_session, svc, goal, run, proposal, decision, test_user, drift)


# ---- concurrency --------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_identical_answers_reserve_exactly_one_wrapper(concurrent_sessions, tmp_path, fail_on_effect):
    s1, s2 = concurrent_sessions
    user = User(email=f"c-{uuid.uuid4()}@example.com", hashed_password="x", display_name="C", role="member")
    project = Project(name="Concurrent recovery", description="", workspace_path=str(tmp_path), config={})
    agent = Agent(name=f"a-{uuid.uuid4()}", role="planning", provider="openai", model="gpt-4o-mini",
                  adapter_type="api", capabilities=[], config={})
    s1.add_all([user, project, agent])
    await s1.flush()
    goal = OrchestrationGoal(project_id=project.id, objective="Plan the release.")
    s1.add(goal)
    await s1.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running")
    s1.add(run)
    await s1.flush()
    svc, _, proposal, decision = await diagnosed(s1, goal, run, agent)
    await s1.commit()

    async def answer(session):
        row = await session.get(OrchestrationAuthorityDecision, decision.id)
        try:
            result = await service_for(agent).reserve_answer(
                session, row, selected_option="approve_fix", reason=None, user_id=user.id,
            )
            await session.commit()
            return result
        except Exception as exc:  # the loser may be rejected as a conflicting/duplicate answer
            await session.rollback()
            return exc

    results = await asyncio.gather(answer(s1), answer(s2))
    winners = [r for r in results if isinstance(r, RecoveryAnswerReservation)]
    assert winners, results
    assert len({dataclasses.astuple(w) for w in winners}) == 1

    rows = (await s1.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "recovery_apply",
    ))).all()
    assert len(rows) == 1
    await s1.refresh(decision)
    assert decision.status == "answered" and decision.selected_option == "approve_fix"


# ---- projection ---------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_projection_failed_when_proposal_failed_with_effect_unknown(db_session, goal_run, test_agent, test_user, fail_on_effect):
    goal, run = goal_run
    svc, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)
    await svc.reserve_answer(db_session, decision, selected_option="approve_fix", reason=None, user_id=test_user.id)
    (wrapper,) = await actions_of(db_session, run, "recovery_apply")
    wrapper.status, wrapper.error = "failed", "effect_unknown"
    proposal.status, proposal.error = "failed", "effect_unknown"
    await db_session.flush()
    assert await svc.project_status(db_session, proposal) == "failed"


@pytest.mark.asyncio
async def test_projection_progresses_from_wrapper_rows(db_session, goal_run, test_agent, test_user, fail_on_effect):
    # Assumption: apply/verify/retry progress is read from lifecycle-wrapper statuses (receipt details
    # are exercised by + executor tests).
    goal, run = goal_run
    svc, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)
    await svc.reserve_answer(db_session, decision, selected_option="approve_fix", reason=None, user_id=test_user.id)
    pid = proposal.request["proposal_id"]
    (apply_wrapper,) = await actions_of(db_session, run, "recovery_apply")
    assert await svc.project_status(db_session, proposal) == "applying"

    apply_wrapper.status = "completed"
    verify = await OrchestrationService().reserve_action(
        db_session, run.id, f"recovery:{pid}:verify:check-1", "recovery_verify",
        {"proposal_action_id": str(proposal.id), "proposal_hash": proposal.dispatch_contract["proposal_hash"], "check_id": "check-1"},
    )
    assert await svc.project_status(db_session, proposal) == "verifying"

    verify.status = "completed"
    await OrchestrationService().reserve_action(
        db_session, run.id, f"recovery:{pid}:retry:1", "recovery_retry",
        {"proposal_action_id": str(proposal.id), "proposal_hash": proposal.dispatch_contract["proposal_hash"], "approved_retry_ordinal": 1},
    )
    assert await svc.project_status(db_session, proposal) == "retrying"

    proposal.status = "completed"
    assert await svc.project_status(db_session, proposal) == "resolved"


@pytest.mark.asyncio
async def test_projection_failed_when_apply_wrapper_failed(db_session, goal_run, test_agent, test_user, fail_on_effect):
    goal, run = goal_run
    svc, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)
    await svc.reserve_answer(db_session, decision, selected_option="approve_fix", reason=None, user_id=test_user.id)
    (wrapper,) = await actions_of(db_session, run, "recovery_apply")
    wrapper.status, wrapper.error = "failed", "apply_failed"
    proposal.status, proposal.error = "failed", "apply_failed"
    assert await svc.project_status(db_session, proposal) == "failed"


@pytest.mark.asyncio
async def test_lifecycle_status_values_are_exactly_the_spec_set():
    assert {s.value for s in RecoveryLifecycleStatus} == {
        "pending_approval", "applying", "verifying", "retrying", "resolved", "rejected", "failed", "unsupported",
    }


# ---- follow-up --------------------------------------------------------------------------------

async def _seed_committed(session, tmp_path):
    user = User(email=f"c-{uuid.uuid4()}@example.com", hashed_password="x", display_name="C", role="member")
    project = Project(name="Committed recovery", description="", workspace_path=str(tmp_path), config={})
    agent = Agent(name=f"a-{uuid.uuid4()}", role="planning", provider="openai", model="gpt-4o-mini",
                  adapter_type="api", capabilities=[], config={})
    session.add_all([user, project, agent])
    await session.flush()
    goal = OrchestrationGoal(project_id=project.id, objective="Plan the release.")
    session.add(goal)
    await session.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running")
    session.add(run)
    await session.flush()
    return user, agent, goal, run


@pytest.mark.asyncio
async def test_stale_writes_survive_when_caller_commits_after_catching(concurrent_sessions, tmp_path, fail_on_effect):
    s1, s2 = concurrent_sessions
    user, agent, goal, run = await _seed_committed(s1, tmp_path)
    svc, _, proposal, decision = await diagnosed(s1, goal, run, agent)
    await s1.commit()

    agent.role = "drifted role"
    await s1.flush()
    with pytest.raises(RecoveryProposalStale):
        await svc.reserve_answer(s1, decision, selected_option="approve_fix", reason=None, user_id=user.id)
    await s1.commit()  # what the route does before returning 409

    stored_decision = await s2.get(OrchestrationAuthorityDecision, decision.id)
    stored_proposal = await s2.get(OrchestrationAction, proposal.id)
    assert stored_decision.status == "cancelled"
    assert (stored_proposal.status, stored_proposal.error) == ("failed", "state_changed")
    triggers = (await s2.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "recovery_trigger",
        OrchestrationAction.status == "reserved",
    ))).all()
    assert len(triggers) == 1


@pytest.mark.asyncio
async def test_three_consecutive_stale_cycles_keep_trigger_keys_bounded(db_session, goal_run, test_agent, test_user, fail_on_effect):
    goal, run = goal_run
    svc = service_for(test_agent)
    trigger = await svc.create_trigger(db_session, trigger_for(goal, run))
    attempts = []
    for cycle in range(3):
        proposal = await svc.diagnose(db_session, trigger.id)
        attempts.append(proposal.request["attempt"])
        decision = (await db_session.scalars(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.related_action_id == proposal.id,
        ))).one()
        test_agent.role = f"drifted role {cycle}"
        await db_session.flush()
        with pytest.raises(RecoveryProposalStale):
            await svc.reserve_answer(db_session, decision, selected_option="approve_fix", reason=None, user_id=test_user.id)
        (trigger,) = [t for t in await actions_of(db_session, run, "recovery_trigger") if t.status == "reserved"]

    assert attempts == [1, 2, 3]
    keys = [t.idempotency_key for t in await actions_of(db_session, run, "recovery_trigger")]
    assert len(keys) == 4 and len(set(keys)) == 4
    assert all(len(key) <= 255 for key in keys)
    assert trigger.dispatch_contract["attempt"] == 4
    # Fourth diagnosis still works: the key did not overflow.
    assert (await svc.diagnose(db_session, trigger.id)).request["attempt"] == 4


@pytest.mark.asyncio
async def test_generic_cancel_cannot_cancel_a_recovery_approval(db_session, goal_run, test_agent):
    goal, run = goal_run
    _, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)
    authority = OrchestrationAuthorityDecisionService()
    with pytest.raises(ValueError):
        await authority.cancel_decision(db_session, decision, reason="user cancel")
    with pytest.raises(ValueError):
        await authority.answer_decision(db_session, decision, selected_option="approve_fix", decided_by_user_id=uuid.uuid4())
    await db_session.refresh(decision)
    await db_session.refresh(proposal)
    assert decision.status == "pending" and proposal.status == "reserved"

    await authority.cancel_recovery_decision(db_session, decision, reason="goal cancelled")
    await db_session.refresh(proposal)
    assert decision.status == "cancelled"
    assert (proposal.status, proposal.error) == ("failed", "goal cancelled")


@pytest.mark.asyncio
async def test_invalid_approval_context_fails_closed_as_stale(db_session, goal_run, test_agent, test_user, fail_on_effect):
    goal, run = goal_run
    svc, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)

    async def corrupt():
        decision.context = json.dumps({"kind": "something_else"})

    await _assert_stale(db_session, svc, goal, run, proposal, decision, test_user, corrupt)


@pytest.mark.asyncio
async def test_deleted_decision_fails_closed_with_domain_error(db_session, goal_run, test_agent, test_user, fail_on_effect):
    goal, run = goal_run
    svc, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)
    await db_session.delete(decision)
    await db_session.flush()
    with pytest.raises(ValueError, match="no longer exists"):
        await svc.reserve_answer(db_session, decision, selected_option="approve_fix", reason=None, user_id=test_user.id)
    assert await actions_of(db_session, run, "recovery_apply") == []


@pytest.mark.asyncio
async def test_reject_requires_proposal_still_reserved(db_session, goal_run, test_agent, test_user, fail_on_effect):
    goal, run = goal_run
    svc, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)
    proposal.status = "completed"
    await db_session.flush()
    with pytest.raises(ValueError, match="no longer awaiting approval"):
        await svc.reserve_answer(db_session, decision, selected_option="reject", reason="Not needed.", user_id=test_user.id)
    await db_session.refresh(decision)
    assert decision.status == "pending"


@pytest.mark.asyncio
async def test_goal_closeout_cancels_pending_recovery_approval(db_session, goal_run, test_agent):
    from huddleroom.services.orchestration_goal_closeout import GoalCloseoutProcess

    goal, run = goal_run
    _, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)
    ids = await GoalCloseoutProcess()._cancel_pending_decisions(db_session, goal.id, reason="goal cancelled")
    await db_session.refresh(decision)
    await db_session.refresh(proposal)
    assert str(decision.id) in ids
    assert decision.status == "cancelled"
    assert (proposal.status, proposal.error) == ("failed", "goal cancelled")
