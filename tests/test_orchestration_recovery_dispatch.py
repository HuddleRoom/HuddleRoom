"""Trusted recovery dispatch identity and wrapper/effect persistence tests."""

import hashlib
import json
import uuid

import pytest
from sqlalchemy import select

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationDecision
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.schemas.orchestration_recovery import (
    RecoveryApplyDispatchContext,
    RecoveryRetryDispatchContext,
)
from huddleroom.services import orchestration_decision_dispatcher as dispatcher_module
from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher
from huddleroom.services.orchestration_service import OrchestrationService


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _proposal(proposal_id):
    source_id = uuid.uuid4()
    return {
        "schema_version": 1,
        "proposal_id": str(proposal_id),
        "lineage_id": str(uuid.uuid4()),
        "attempt": 1,
        "source": {
            "kind": "reported_blocker", "id": str(source_id), "blocker_code": "plan_author_unavailable",
            "fingerprint": "a" * 64,
        },
        "resource_fingerprint": "b" * 64,
        "registry_version": "1",
        "problem": "No eligible planning author is available.",
        "general_updates": ["Assign an eligible planning author."],
        "expected_effect": "Planning can resume.",
        "evidence": [{
            "label": "Preflight failed.", "value": {"code": "plan_author_unavailable"},
            "source_kind": "run", "source_id": str(source_id),
        }],
        "calls": [{
            "call_id": "call-1", "tool": "existing_action.dispatch", "tool_version": "1",
            "arguments": {"kind": "request_plan"},
            "changes": [{
                "operation": "request", "scope": "run", "target_type": "plan", "target_id": "plan",
                "after": {"status": "requested"},
            }],
        }],
        "verification": [{
            "check_id": "planning-ready", "verifier": "planning.ready", "verifier_version": "1",
            "arguments": {"kind": "planning_ready"},
        }],
        "retry": {"target": {"kind": "decision_loop"}, "max_attempts": 1},
        "authority_required": "human",
        "global_effect": False,
    }


def _request():
    return {
        "action_type": "request_plan",
        "agent_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "work_function": "planning",
        "scope": "Create the delivery plan.",
    }


def _request_hash():
    return _hash(_request())


async def _rows(db, run_id, action_type):
    return (await db.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run_id,
        OrchestrationAction.action_type == action_type,
    ))).all()


async def _effect_rows(db, run_id):
    return [action for action in (await db.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run_id,
    ))).all() if action.idempotency_key.endswith(":effect")]


async def _recovery_fixture(db, conversation_goal_run, *, purpose="apply", authority_status="answered",
                            selected_option="approve_fix", proposal_id=None, call_id="call-1", retry_ordinal=1):
    goal, run = conversation_goal_run
    service = OrchestrationService()
    proposal_id = proposal_id or uuid.uuid4()
    proposal = _proposal(proposal_id)
    proposal_hash = _hash(proposal)
    proposal_action = await service.reserve_action(
        db, run.id, f"recovery:{proposal_id}:proposal", "recovery_proposal", proposal,
    )
    proposal_action.dispatch_contract = {
        "proposal_hash": proposal_hash,
        "resource_fingerprint": proposal["resource_fingerprint"],
        "lineage_id": proposal["lineage_id"],
        "attempt": 1,
    }
    authority = OrchestrationAuthorityDecision(
        goal_id=goal.id, run_id=run.id, decision_key=f"recovery:{proposal_id}:approval",
        title="Approve recovery fix", authority="human", question="Apply the approved recovery fix?",
        options=["approve_fix", "reject"], status=authority_status,
        selected_option=selected_option if authority_status == "answered" else None,
        related_action_id=proposal_action.id,
    )
    db.add(authority)
    await db.flush()
    if purpose == "apply":
        context = RecoveryApplyDispatchContext(
            purpose="apply", proposal_id=proposal_id, call_id=call_id, authority_decision_id=authority.id,
        )
    else:
        context = RecoveryRetryDispatchContext(
            purpose="retry", proposal_id=proposal_id, approved_retry_ordinal=retry_ordinal,
            authority_decision_id=authority.id,
        )
    wrapper_key = (
        f"recovery:{proposal_id}:apply:{call_id}"
        if purpose == "apply" else f"recovery:{proposal_id}:retry:{retry_ordinal}"
    )
    wrapper = await service.reserve_action(
        db, run.id, wrapper_key, f"recovery_{purpose}",
        {
            "proposal_action_id": str(proposal_action.id), "proposal_hash": proposal_hash,
            **({"call_id": call_id} if purpose == "apply" else {"approved_retry_ordinal": retry_ordinal}),
        },
    )
    wrapper.dispatch_contract = {"canonical_request_hash": _request_hash()}
    decision = OrchestrationDecision(
        run_id=run.id, decision_type="execution", validator_status="accepted",
        input_snapshot={"run": {"id": str(run.id), "plan_state": run.plan_state}},
        parsed_decision={**_request(), "applies_decision_id": str(authority.id)},
    )
    db.add(decision)
    await db.flush()
    return service, run, decision, context, wrapper, authority


async def _reserve_only_plan_effect(service, db, *, run_id, request, idempotency_key, decision_id, **_extra):
    """Keep dispatch persistence real while replacing plan execution and its external work."""
    return await service.reserve_action(
        db, run_id, idempotency_key, "request_plan", request, decision_id,
    )


def test_recovery_keys_are_exact_and_effects_never_share_wrapper_identity():
    proposal_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
    authority_id = uuid.UUID("22222222-2222-2222-2222-222222222222")
    apply = RecoveryApplyDispatchContext(
        purpose="apply", proposal_id=proposal_id, call_id="call-1", authority_decision_id=authority_id,
    )
    retry = RecoveryRetryDispatchContext(
        purpose="retry", proposal_id=proposal_id, approved_retry_ordinal=2, authority_decision_id=authority_id,
    )

    assert dispatcher_module.recovery_wrapper_key(apply) == "recovery:11111111-1111-1111-1111-111111111111:apply:call-1"
    assert dispatcher_module.recovery_effect_key(apply) == "recovery:11111111-1111-1111-1111-111111111111:apply:call-1:effect"
    assert dispatcher_module.recovery_wrapper_key(retry) == "recovery:11111111-1111-1111-1111-111111111111:retry:2"
    assert dispatcher_module.recovery_effect_key(retry) == "recovery:11111111-1111-1111-1111-111111111111:retry:2:effect"


@pytest.mark.asyncio
async def test_recovery_backed_dispatch_rejects_missing_context_before_effect_reservation(
    db_session, conversation_goal_run,
):
    service, run, decision, _context, _wrapper, _authority = await _recovery_fixture(
        db_session, conversation_goal_run,
    )

    assert await OrchestrationDecisionDispatcher(service).dispatch(db_session, run, decision) is None
    assert decision.validator_status == "rejected"
    assert await _rows(db_session, run.id, "request_plan") == []


@pytest.mark.asyncio
async def test_recovery_backed_dispatch_rejects_wrong_purpose_before_effect_reservation(
    db_session, conversation_goal_run,
):
    service, run, decision, _context, _wrapper, authority = await _recovery_fixture(
        db_session, conversation_goal_run,
    )
    wrong_purpose = RecoveryRetryDispatchContext(
        purpose="retry", proposal_id=uuid.UUID(str(authority.decision_key.split(":")[1])),
        approved_retry_ordinal=1, authority_decision_id=authority.id,
    )

    assert await OrchestrationDecisionDispatcher(service).dispatch(
        db_session, run, decision, recovery_context=wrong_purpose,
    ) is None
    assert decision.validator_status == "rejected"
    assert await _rows(db_session, run.id, "request_plan") == []


@pytest.mark.asyncio
async def test_recovery_backed_dispatch_rejects_context_for_a_different_authority_decision(
    db_session, conversation_goal_run,
):
    service, run, decision, context, _wrapper, _authority = await _recovery_fixture(
        db_session, conversation_goal_run,
    )
    wrong_authority = RecoveryApplyDispatchContext(
        purpose="apply", proposal_id=context.proposal_id, call_id=context.call_id,
        authority_decision_id=uuid.uuid4(),
    )

    assert await OrchestrationDecisionDispatcher(service).dispatch(
        db_session, run, decision, recovery_context=wrong_authority,
    ) is None
    assert decision.validator_status == "rejected"
    assert await _rows(db_session, run.id, "request_plan") == []


@pytest.mark.asyncio
async def test_recovery_dispatch_rejects_wrapper_for_call_absent_from_approved_proposal(
    db_session, conversation_goal_run,
):
    service, run, decision, context, wrapper, _authority = await _recovery_fixture(
        db_session, conversation_goal_run, call_id="call-2",
    )

    assert await OrchestrationDecisionDispatcher(service).dispatch(
        db_session, run, decision, recovery_context=context,
    ) is None
    assert decision.validator_status == "rejected"
    assert await _rows(db_session, run.id, "request_plan") == []
    assert await _effect_rows(db_session, run.id) == []
    assert "effect_action_id" not in wrapper.dispatch_contract


@pytest.mark.asyncio
async def test_recovery_dispatch_rejects_wrapper_for_retry_beyond_approved_proposal_limit(
    db_session, conversation_goal_run,
):
    service, run, decision, context, wrapper, _authority = await _recovery_fixture(
        db_session, conversation_goal_run, purpose="retry", retry_ordinal=2,
    )

    assert await OrchestrationDecisionDispatcher(service).dispatch(
        db_session, run, decision, recovery_context=context,
    ) is None
    assert decision.validator_status == "rejected"
    assert await _rows(db_session, run.id, "request_plan") == []
    assert await _effect_rows(db_session, run.id) == []
    assert "effect_action_id" not in wrapper.dispatch_contract


@pytest.mark.asyncio
async def test_recovery_dispatch_rejects_unanswered_approval_before_effect_reservation(
    db_session, conversation_goal_run,
):
    service, run, decision, context, _wrapper, _authority = await _recovery_fixture(
        db_session, conversation_goal_run, authority_status="pending", selected_option=None,
    )

    assert await OrchestrationDecisionDispatcher(service).dispatch(
        db_session, run, decision, recovery_context=context,
    ) is None
    assert decision.validator_status == "rejected"
    assert await _rows(db_session, run.id, "request_plan") == []


@pytest.mark.asyncio
async def test_recovery_dispatch_rejects_answer_without_approve_fix_before_effect_reservation(
    db_session, conversation_goal_run,
):
    service, run, decision, context, _wrapper, _authority = await _recovery_fixture(
        db_session, conversation_goal_run, selected_option="reject",
    )

    assert await OrchestrationDecisionDispatcher(service).dispatch(
        db_session, run, decision, recovery_context=context,
    ) is None
    assert decision.validator_status == "rejected"
    assert await _rows(db_session, run.id, "request_plan") == []


@pytest.mark.asyncio
async def test_recovery_dispatch_rejects_wrapper_with_wrong_canonical_request_hash(
    db_session, conversation_goal_run,
):
    service, run, decision, context, wrapper, _authority = await _recovery_fixture(
        db_session, conversation_goal_run,
    )
    wrapper.dispatch_contract = {"canonical_request_hash": "f" * 64}

    assert await OrchestrationDecisionDispatcher(service).dispatch(
        db_session, run, decision, recovery_context=context,
    ) is None
    assert decision.validator_status == "rejected"
    assert await _rows(db_session, run.id, "request_plan") == []


@pytest.mark.asyncio
async def test_recovery_dispatch_replay_keeps_one_wrapper_and_one_effect(db_session, conversation_goal_run):
    service, run, decision, context, wrapper, authority = await _recovery_fixture(db_session, conversation_goal_run)
    service.execute_request_plan_action = _reserve_only_plan_effect.__get__(service, OrchestrationService)
    dispatcher = OrchestrationDecisionDispatcher(service)

    effect = await dispatcher.dispatch(db_session, run, decision, recovery_context=context)
    replay = await dispatcher.dispatch(db_session, run, decision, recovery_context=context)

    assert replay.id == effect.id
    assert len(await _rows(db_session, run.id, "request_plan")) == 1
    assert effect.idempotency_key == f"recovery:{context.proposal_id}:apply:{context.call_id}:effect"
    assert wrapper.dispatch_contract == {
        "canonical_request_hash": _request_hash(),
        "authority_decision_id": str(authority.id),
        "effect_action_id": str(effect.id),
        "effect_idempotency_key": effect.idempotency_key,
    }


@pytest.mark.asyncio
async def test_later_approved_proposal_can_reserve_same_domain_request_with_new_effect_identity(
    db_session, conversation_goal_run,
):
    first_service, run, first_decision, first_context, _first_wrapper, _first_authority = await _recovery_fixture(
        db_session, conversation_goal_run,
    )
    first_service.execute_request_plan_action = _reserve_only_plan_effect.__get__(first_service, OrchestrationService)
    first = await OrchestrationDecisionDispatcher(first_service).dispatch(
        db_session, run, first_decision, recovery_context=first_context,
    )

    second_service, same_run, second_decision, second_context, _second_wrapper, _second_authority = (
        await _recovery_fixture(db_session, conversation_goal_run)
    )
    assert same_run.id == run.id
    second_service.execute_request_plan_action = _reserve_only_plan_effect.__get__(second_service, OrchestrationService)
    second = await OrchestrationDecisionDispatcher(second_service).dispatch(
        db_session, same_run, second_decision, recovery_context=second_context,
    )

    assert first.request == second.request
    assert first.id != second.id
    assert first.idempotency_key != second.idempotency_key
