"""SPR 180/184: recovery lineage, semantic dedupe, delegation-failure rules and the attempt cap.

Shares the recovery in ``test_orchestration_recovery_execution`` (scripted analyzer, recorder
executors, real diagnose/answer path). Interface assumptions are listed there; in addition:

- ``rediagnose(db, proposal_action, evidence)`` re-enters diagnosis for a failed/rejected attempt in the
  SAME lineage (``dispatch_contract["lineage_id"]`` kept, ``attempt`` + 1). It may return the reserved
  follow-up ``recovery_trigger`` (then ``diagnose`` runs it) or the diagnosed outcome directly.
- A refused re-diagnosis (duplicate semantic proposal on the same state, or a repeat of a failed
  delegation to the same agent/scope) surfaces as ``RecoveryDiagnosisFailed`` whose ``code`` contains
  ``"duplicate"`` / ``"repeat"`` respectively, and creates no approve_fix card.
- After ``MAX_RECOVERY_ATTEMPTS_WITHOUT_PROGRESS`` (3) materially different failed attempts, the next
  re-diagnosis produces exactly one pending human decision without an ``approve_fix`` option.
"""
from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import func, select

from huddleroom.models.orchestration import OrchestrationAction
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision, OrchestrationWarning
from huddleroom.schemas.orchestration_recovery import EvidenceV1
from huddleroom.services.orchestration_blocker_recovery import RecoveryDiagnosisFailed
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from tests.conftest import adopt_session
from tests.test_orchestration_recovery_execution import (
    Executors,
    ScriptedAnalyzer,
    approve,
    approved,
    attempt,
    bound_task,
    decision_for,
    delegation_flow,
    deliver,
    draft,
    lifecycle,
    propose,
    reassign_call,
    retry_task_call,
    seed as _execution_seed,
    status,
    trigger,
    verified,
    verify_call,
)


@pytest_asyncio.fixture
async def db_session(owned_session_factory):
    """Autobegin session (no session.begin()) so the test task can own it."""
    async with owned_session_factory() as db:
        yield db
        await db.rollback()


async def seed(db, project, **kwargs):
    adopt_session(db)  # the test task owns this autobegin session, as get_db does
    return await _execution_seed(db, project, **kwargs)


def evidence(label: str, source_id, kind="action", **value) -> EvidenceV1:
    return EvidenceV1(label=label, value=value or {"note": label}, source_kind=kind, source_id=source_id)


async def rediagnosed(svc, db, proposal, item: EvidenceV1):
    """Follow-up outcome of a re-diagnosis, or the RecoveryDiagnosisFailed that refused it."""
    try:
        action = await svc.rediagnose(db, proposal, item)
        if action.action_type == "recovery_trigger":
            action = await svc.diagnose(db, action.id)
        return action
    except RecoveryDiagnosisFailed as exc:
        return exc


async def recovery_decisions(db, goal_id, *, status_=None):
    rows = (await db.scalars(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.goal_id == goal_id,
    ).execution_options(populate_existing=True))).all()
    return [row for row in rows if (status_ is None or row.status == status_)
            and row.decision_key.startswith(("recovery:", "recovery_gap:", "recovery_escalation:", "recovery"))]


def offers_approve_fix(decision) -> bool:
    return "approve_fix" in json.dumps(decision.options, default=str)


async def proposals(db, run_id):
    rows = (await db.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run_id, OrchestrationAction.action_type == "recovery_proposal",
    ).execution_options(populate_existing=True))).all()
    return sorted(rows, key=lambda row: (row.created_at, row.id))


async def failed_attempt(db, rows, user, monkeypatch, *calls, analyzer=None):
    flow = await approved(db, rows, user, *calls, analyzer=analyzer)
    Executors(monkeypatch, rows, plan={"request_verification": ["reject"] * 8, "retry_task": ["reject"] * 8,
                                       "reassign_task": ["reject"] * 8})
    await attempt(flow.svc.execute_next_call(db, flow.proposal))
    await attempt(flow.svc.resume(db, flow.proposal.id))
    assert await status(db, flow.svc, flow.proposal) == "failed"
    return flow


# --------------------------------------------------------------------------- same lineage


@pytest.mark.asyncio
async def test_apply_failure_rediagnoses_in_the_same_lineage_with_its_evidence(db_session, test_project, test_user, monkeypatch):
    rows = await seed(db_session, test_project)
    analyzer = ScriptedAnalyzer(draft(rows, verify_call(rows)), draft(rows, retry_task_call(rows)))
    flow = await failed_attempt(db_session, rows, test_user, monkeypatch, analyzer=analyzer)
    failure = evidence("The approved verification request was rejected by its executor.", flow.proposal.id)

    outcome = await rediagnosed(flow.svc, db_session, flow.proposal, failure)

    assert isinstance(outcome, OrchestrationAction) and outcome.action_type == "recovery_proposal", outcome
    assert outcome.id != flow.proposal.id
    assert outcome.dispatch_contract["lineage_id"] == flow.proposal.dispatch_contract["lineage_id"]
    assert outcome.dispatch_contract["attempt"] == flow.proposal.dispatch_contract["attempt"] + 1
    assert outcome.request["lineage_id"] == flow.proposal.request["lineage_id"]
    snapshot = json.dumps(analyzer.snapshots[-1], default=str)
    assert failure.label in snapshot, "the failure evidence reaches the next diagnosis"
    assert str(flow.proposal.id) in snapshot, "the prior attempt in this run's lineage is visible"
    assert (await decision_for(db_session, outcome)).status == "pending"


@pytest.mark.asyncio
async def test_partial_failure_rediagnosis_starts_from_the_applied_state(db_session, test_project, test_user, monkeypatch):
    rows = await seed(db_session, test_project)
    analyzer = ScriptedAnalyzer(draft(rows, retry_task_call(rows), verify_call(rows)), draft(rows, reassign_call(rows)))
    flow = await approved(db_session, rows, test_user, analyzer=analyzer)
    Executors(monkeypatch, rows, plan={"request_verification": ["reject"]})
    first = await flow.svc.execute_next_call(db_session, flow.proposal)
    await attempt(flow.svc.execute_next_call(db_session, flow.proposal))
    assert await status(db_session, flow.svc, flow.proposal) == "failed"

    await rediagnosed(flow.svc, db_session, flow.proposal,
                      evidence("Call 2 failed after call 1 applied.", flow.proposal.id))

    snapshot = json.dumps(analyzer.snapshots[-1], default=str)
    assert str(first.action_id) in snapshot or str(first.wrapper_action_id) in snapshot, (
        "the next diagnosis must see call 1's applied effect, not a rolled-back state")


@pytest.mark.asyncio
async def test_retry_exposing_the_same_blocker_continues_the_lineage(db_session, test_project, test_user, monkeypatch):
    rows = await seed(db_session, test_project)
    analyzer = ScriptedAnalyzer(draft(rows, verify_call(rows)), draft(rows, retry_task_call(rows)))
    flow = await approved(db_session, rows, test_user, analyzer=analyzer)
    Executors(monkeypatch, rows, plan={"request_verification": ["ok", "reject"]})
    await verified(db_session, flow)
    await attempt(flow.svc.retry_blocked_work(db_session, flow.proposal))
    assert await status(db_session, flow.svc, flow.proposal) == "failed"

    outcome = await rediagnosed(flow.svc, db_session, flow.proposal, evidence(
        "The retried verification failed with the same blocker.", flow.proposal.id,
        blocker_code="verification_unavailable"))

    assert isinstance(outcome, OrchestrationAction) and outcome.action_type == "recovery_proposal", outcome
    assert outcome.dispatch_contract["lineage_id"] == flow.proposal.dispatch_contract["lineage_id"]
    assert outcome.dispatch_contract["attempt"] == 2


@pytest.mark.asyncio
async def test_a_different_blocker_after_retry_starts_a_new_lineage(db_session, test_project, test_user, monkeypatch):
    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, verify_call(rows))
    Executors(monkeypatch, rows)
    await verified(db_session, flow)
    await flow.svc.retry_blocked_work(db_session, flow.proposal)
    assert await status(db_session, flow.svc, flow.proposal) == "resolved"
    other = OrchestrationAction(run_id=rows.run.id, idempotency_key=f"seed:{uuid.uuid4()}:retry", action_type="retry_task",
                                status="failed", request={"task_id": str(rows.task.id)}, error="task_unrunnable")
    db_session.add(other)
    await db_session.flush()

    _svc, outcome = await propose(db_session, rows, ScriptedAnalyzer(draft(rows, reassign_call(rows))), trigger(
        rows, source_id=other.id, blocker_code="task_unrunnable", fingerprint="d" * 64,
        retry_target={"kind": "action", "source_action_id": str(other.id)}))

    assert outcome.dispatch_contract["lineage_id"] != flow.proposal.dispatch_contract["lineage_id"]
    assert outcome.dispatch_contract["attempt"] == 1


# --------------------------------------------------------------------------- semantic dedupe


@pytest.mark.asyncio
async def test_duplicate_semantic_proposal_on_the_same_state_is_rejected(db_session, test_project, test_user):
    rows = await seed(db_session, test_project)
    analyzer = ScriptedAnalyzer(draft(rows, verify_call(rows)), draft(rows, verify_call(rows), problem="Reworded."))
    svc, proposal = await propose(db_session, rows, analyzer)
    decision = await decision_for(db_session, proposal)
    await svc.reserve_answer(db_session, decision, selected_option="reject", reason="Use a different gate.",
                             user_id=test_user.id)

    outcome = await rediagnosed(svc, db_session, proposal, evidence("Human rejected: use a different gate.",
                                                                    decision.id, kind="decision"))

    assert isinstance(outcome, RecoveryDiagnosisFailed), outcome
    assert "duplicate" in outcome.code
    assert [row for row in await recovery_decisions(db_session, rows.goal.id, status_="pending") if offers_approve_fix(row)] == []
    assert len(await proposals(db_session, rows.run.id)) == 1, "the rejected proposal is never silently re-offered"


@pytest.mark.asyncio
async def test_rejection_reason_feeds_the_next_diagnosis(db_session, test_project, test_user):
    rows = await seed(db_session, test_project)
    analyzer = ScriptedAnalyzer(draft(rows, verify_call(rows)), draft(rows, retry_task_call(rows)))
    svc, proposal = await propose(db_session, rows, analyzer)
    decision = await decision_for(db_session, proposal)
    await svc.reserve_answer(db_session, decision, selected_option="reject", reason="Retry the task instead.",
                             user_id=test_user.id)

    await rediagnosed(svc, db_session, proposal, evidence("Human rejected the fix.", decision.id, kind="decision"))

    assert "Retry the task instead." in json.dumps(analyzer.snapshots[-1], default=str)


@pytest.mark.asyncio
async def test_materially_different_proposal_permits_another_approval(db_session, test_project, test_user):
    rows = await seed(db_session, test_project)
    analyzer = ScriptedAnalyzer(draft(rows, verify_call(rows)), draft(rows, retry_task_call(rows)))
    svc, proposal = await propose(db_session, rows, analyzer)
    decision = await decision_for(db_session, proposal)
    await svc.reserve_answer(db_session, decision, selected_option="reject", reason="Retry the task instead.",
                             user_id=test_user.id)

    outcome = await rediagnosed(svc, db_session, proposal, evidence("Human rejected the fix.", decision.id, kind="decision"))

    assert isinstance(outcome, OrchestrationAction) and outcome.action_type == "recovery_proposal", outcome
    assert [call["tool"] for call in outcome.request["calls"]] == ["orchestration.retry_task"]
    assert outcome.dispatch_contract["lineage_id"] == proposal.dispatch_contract["lineage_id"]
    new_decision = await decision_for(db_session, outcome)
    assert new_decision.status == "pending" and offers_approve_fix(new_decision)


@pytest.mark.asyncio
async def test_same_proposal_after_a_real_state_change_is_permitted(db_session, test_project, test_user):
    rows = await seed(db_session, test_project)
    analyzer = ScriptedAnalyzer(draft(rows, verify_call(rows)))
    svc, proposal = await propose(db_session, rows, analyzer)
    decision = await decision_for(db_session, proposal)
    await svc.reserve_answer(db_session, decision, selected_option="reject", reason="Wait for the gate update.",
                             user_id=test_user.id)
    rows.gate.failure_reason = "updated evidence attached"  # new external state: a new fingerprint
    await db_session.flush()

    outcome = await rediagnosed(svc, db_session, proposal, evidence("The gate gained new evidence.", rows.gate.id, kind="gate"))

    assert isinstance(outcome, OrchestrationAction) and outcome.action_type == "recovery_proposal", outcome
    assert outcome.request["resource_fingerprint"] != proposal.request["resource_fingerprint"]


# --------------------------------------------------------------------------- delegation failure


async def failed_delegation(db, project, user, monkeypatch):
    flow = await delegation_flow(db, project, user, monkeypatch)
    receipt = await flow.svc.execute_next_call(db, flow.proposal)
    task = await bound_task(db, receipt)
    task.status = "failed"
    await db.flush()
    await deliver(db, flow.rows.run, "task.status_changed", task.id)
    assert await flow.svc.resume(db, flow.proposal.id) == "failed"
    return flow, task


async def second_agent(db, rows):
    from tests.test_orchestration_recovery_delegation import cli_agent

    other = cli_agent("other-fixer")
    db.add(other)
    await db.flush()
    review = await OrchestrationProcessService().get_current(db, rows.goal.id, "agent_definition_review")
    eligibility = dict(review.outputs.get("eligibility", {}))
    eligibility[str(other.id)] = ["implementation", "investigation"]
    review.outputs = {**review.outputs, "eligibility": eligibility}
    await db.flush()
    return other


async def delegation_rows(db, run_id):
    return (await db.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run_id, OrchestrationAction.action_type == "create_delegation_task",
    ))).all()


@pytest.mark.asyncio
async def test_failed_delegation_to_the_same_agent_and_scope_cannot_recur(db_session, test_project, test_user, monkeypatch):
    adopt_session(db_session)
    from tests.test_orchestration_recovery_delegation import draft as delegation_draft

    flow, task = await failed_delegation(db_session, test_project, test_user, monkeypatch)
    flow.analyzer.drafts = [delegation_draft(flow.rows)]  # the very same agent, objective and result

    outcome = await rediagnosed(flow.svc, db_session, flow.proposal,
                                evidence("The remediation task failed.", task.id, kind="task"))

    assert isinstance(outcome, RecoveryDiagnosisFailed), outcome
    assert "repeat" in outcome.code or "duplicate" in outcome.code
    assert len(await delegation_rows(db_session, flow.rows.run.id)) == 1
    pending = await recovery_decisions(db_session, flow.rows.goal.id, status_="pending")
    assert [row for row in pending if offers_approve_fix(row)] == []


@pytest.mark.asyncio
async def test_failed_delegation_is_evidence_for_a_different_qualified_agent(db_session, test_project, test_user, monkeypatch):
    adopt_session(db_session)
    from tests.test_orchestration_recovery_delegation import draft as delegation_draft

    flow, task = await failed_delegation(db_session, test_project, test_user, monkeypatch)
    other = await second_agent(db_session, flow.rows)
    flow.analyzer.drafts = [delegation_draft(SimpleNamespace(**{**vars(flow.rows), "agent": other}))]

    outcome = await rediagnosed(flow.svc, db_session, flow.proposal,
                                evidence("The remediation task failed.", task.id, kind="task"))

    assert isinstance(outcome, OrchestrationAction) and outcome.action_type == "recovery_proposal", outcome
    assert outcome.request["calls"][0]["arguments"]["agent_id"] == str(other.id)
    assert outcome.dispatch_contract["lineage_id"] == flow.proposal.dispatch_contract["lineage_id"]
    assert offers_approve_fix(await decision_for(db_session, outcome))


# --------------------------------------------------------------------------- attempt cap


@pytest.mark.asyncio
async def test_three_failed_attempts_without_progress_escalate_once_without_a_loop(
    db_session, test_project, test_user, monkeypatch,
):
    from huddleroom.services import orchestration_blocker_recovery as module
    from huddleroom.services import orchestration_recovery_analyzer as analyzer_module

    cap = getattr(module, "MAX_RECOVERY_ATTEMPTS_WITHOUT_PROGRESS",
                  getattr(analyzer_module, "MAX_RECOVERY_ATTEMPTS_WITHOUT_PROGRESS", None))
    assert cap == 3, "code-owned constant, not a model field or setting"
    rows = await seed(db_session, test_project)
    analyzer = ScriptedAnalyzer(
        draft(rows, verify_call(rows)), draft(rows, verify_call(rows, "review")), draft(rows, retry_task_call(rows)),
        draft(rows, reassign_call(rows)),
    )
    flow = await failed_attempt(db_session, rows, test_user, monkeypatch, analyzer=analyzer)
    current = flow.proposal
    for _ in range(2):
        nxt = await rediagnosed(flow.svc, db_session, current, evidence("The approved fix failed.", current.id))
        assert isinstance(nxt, OrchestrationAction) and nxt.action_type == "recovery_proposal", nxt
        await approve(db_session, flow.svc, nxt, test_user)
        await attempt(flow.svc.execute_next_call(db_session, nxt))
        await attempt(flow.svc.resume(db_session, nxt.id))
        assert await status(db_session, flow.svc, nxt) == "failed"
        current = nxt
    assert len(await proposals(db_session, rows.run.id)) == 3
    warnings_before = await db_session.scalar(select(func.count(OrchestrationWarning.id)))

    await rediagnosed(flow.svc, db_session, current, evidence("The third approved fix failed.", current.id))
    await rediagnosed(flow.svc, db_session, current, evidence("The third approved fix failed.", current.id))

    executable = [row for row in await proposals(db_session, rows.run.id) if (row.request or {}).get("calls")]
    assert len(executable) == 3, "no fourth executable proposal after three failed attempts"
    pending = await recovery_decisions(db_session, rows.goal.id, status_="pending")
    assert [row for row in pending if offers_approve_fix(row)] == []
    escalations = [row for row in pending if not offers_approve_fix(row)]
    assert len(escalations) == 1, "exactly one actionable human/capability decision"
    assert escalations[0].authority == "human"
    warnings_after = await db_session.scalar(select(func.count(OrchestrationWarning.id)))
    assert warnings_after - warnings_before <= 1, "no repeating warnings"
    noops = (await db_session.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == rows.run.id, OrchestrationAction.action_type == "noop"))).all()
    assert noops == [], "no fake timer noop"


# --------------------------------------------------------------------------- no recursion


@pytest.mark.asyncio
async def test_recovery_wrapper_failures_never_start_a_recursive_lineage(db_session, test_project, test_user, monkeypatch):
    rows = await seed(db_session, test_project)
    flow = await failed_attempt(db_session, rows, test_user, monkeypatch, verify_call(rows))
    (wrapper,) = await lifecycle(db_session, flow.proposal, "recovery_apply")

    await attempt(flow.svc.create_trigger(db_session, trigger(
        rows, source_id=wrapper.id, blocker_code="recovery_apply_failed", fingerprint="e" * 64)))

    recursive = [row for row in (await db_session.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == rows.run.id, OrchestrationAction.action_type == "recovery_trigger"))).all()
        if (row.request or {}).get("source_id") == str(wrapper.id)]
    assert recursive == [], "a failed recovery_* action is handled inside its lineage, never as a new trigger"


# --------------------------------------------------------------------------- human input (carry-in 10)


async def _escalated(db, rows, user, monkeypatch, analyzer):
    flow = await failed_attempt(db, rows, user, monkeypatch, analyzer=analyzer)
    current = flow.proposal
    for _ in range(2):
        nxt = await rediagnosed(flow.svc, db, current, evidence("The approved fix failed.", current.id))
        await approve(db, flow.svc, nxt, user)
        await attempt(flow.svc.execute_next_call(db, nxt))
        current = nxt
    escalation = await rediagnosed(flow.svc, db, current, evidence("The third approved fix failed.", current.id))
    assert escalation.request.get("kind") == "capability_gap"
    return flow, escalation


@pytest.mark.asyncio
async def test_capability_provided_answer_rediagnoses_with_a_reset_budget(db_session, test_project, test_user, monkeypatch):
    rows = await seed(db_session, test_project)
    analyzer = ScriptedAnalyzer(
        draft(rows, verify_call(rows)), draft(rows, verify_call(rows, "review")), draft(rows, retry_task_call(rows)),
        draft(rows, reassign_call(rows)),
    )
    flow, escalation = await _escalated(db_session, rows, test_user, monkeypatch, analyzer)
    calls_before = analyzer.calls
    (gap_decision,) = [row for row in await recovery_decisions(db_session, rows.goal.id, status_="pending")
                       if row.related_action_id == escalation.id]

    follow_up = await flow.svc.answer_capability_gap(db_session, gap_decision, user_id=test_user.id)

    assert follow_up.action_type == "recovery_trigger"
    assert follow_up.dispatch_contract["lineage_id"] == escalation.dispatch_contract["lineage_id"]
    outcome = await flow.svc.diagnose(db_session, follow_up.id)
    assert analyzer.calls == calls_before + 1, "the human answer re-enters diagnosis"
    assert "The human reported" in json.dumps(analyzer.snapshots[-1], default=str)
    assert outcome.request.get("calls") and outcome.dispatch_contract["lineage_id"] == escalation.dispatch_contract["lineage_id"]
    assert offers_approve_fix(await decision_for(db_session, outcome)), "the reset budget permits a new fix"
    await db_session.refresh(gap_decision)
    assert gap_decision.status == "answered" and gap_decision.selected_option == "capability_provided"
    assert await flow.svc.project_status(db_session, escalation) == "unsupported"


@pytest.mark.asyncio
async def test_unanswered_capability_gap_cannot_be_rediagnosed(db_session, test_project):
    from huddleroom.schemas.orchestration_recovery import CapabilityGapAssessmentV1

    rows = await seed(db_session, test_project)
    gap = CapabilityGapAssessmentV1.model_validate({
        "kind": "capability_gap", "diagnosis": "No registered tool can grant network access.",
        "evidence": [{"label": "The source action failed.", "value": {"status": "failed"}, "source_kind": "action",
                      "source_id": str(rows.source.id)}],
        "missing_capability": "Network access for the verifier.",
        "required_human_action": "Grant the verifier network access.",
        "wake_after": {"kind": "human_decision", "decision": "Network access granted"},
    })
    svc, outcome = await propose(db_session, rows, ScriptedAnalyzer(gap))

    with pytest.raises(ValueError):
        await svc.rediagnose(db_session, outcome, evidence("Nothing changed.", outcome.id))


# --------------------------------------------------------------------------- follow-up


@pytest.mark.asyncio
async def test_repeatedly_refused_diagnosis_escalates_instead_of_calling_the_analyzer_forever(db_session, test_project):
    from huddleroom.services.orchestration_blocker_recovery import (
        MAX_RECOVERY_ATTEMPTS_WITHOUT_PROGRESS,
        OrchestrationBlockerRecoveryService,
    )

    rows = await seed(db_session, test_project)
    analyzer = ScriptedAnalyzer(draft(rows, ("orchestration.unregistered_tool", {"gate_id": str(rows.gate.id)})))
    svc = OrchestrationBlockerRecoveryService(analyzer=analyzer)
    trigger_action = await svc.create_trigger(db_session, trigger(rows))
    for _ in range(MAX_RECOVERY_ATTEMPTS_WITHOUT_PROGRESS):
        with pytest.raises(RecoveryDiagnosisFailed):
            await svc.diagnose(db_session, trigger_action.id)

    outcome = await svc.diagnose(db_session, trigger_action.id)
    again = await svc.diagnose(db_session, trigger_action.id)

    assert analyzer.calls == MAX_RECOVERY_ATTEMPTS_WITHOUT_PROGRESS, "no LLM call after the refusal budget"
    assert outcome.request.get("kind") == "capability_gap" and again.id == outcome.id
    pending = await recovery_decisions(db_session, rows.goal.id, status_="pending")
    assert len(pending) == 1 and not offers_approve_fix(pending[0])


@pytest.mark.asyncio
async def test_recovery_effect_rows_never_start_a_recursive_lineage(db_session, test_project, test_user, monkeypatch):
    rows = await seed(db_session, test_project)
    flow = await failed_attempt(db_session, rows, test_user, monkeypatch, verify_call(rows))
    (wrapper,) = await lifecycle(db_session, flow.proposal, "recovery_apply")
    effect = (await db_session.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == rows.run.id,
        OrchestrationAction.idempotency_key == f"{wrapper.idempotency_key}:effect"))).one()
    assert effect.action_type == "request_verification" and effect.status == "failed"

    with pytest.raises(ValueError):
        await flow.svc.create_trigger(db_session, trigger(rows, source_id=effect.id, blocker_code="verification_conflict",
                                                          fingerprint="f" * 64))


@pytest.mark.asyncio
async def test_one_lineage_warning_does_not_stale_another_lineage_proposal(db_session, test_project, test_user):
    rows = await seed(db_session, test_project)
    _svc_a, proposal_a = await propose(db_session, rows, ScriptedAnalyzer(draft(rows, verify_call(rows))))
    other = OrchestrationAction(run_id=rows.run.id, idempotency_key=f"seed:{uuid.uuid4()}:retry", action_type="retry_task",
                                status="failed", request={"task_id": str(rows.task.id)}, error="task_unrunnable")
    db_session.add(other)
    await db_session.flush()
    blockers_before = list(rows.run.active_blockers or [])
    svc_b, proposal_b = await propose(db_session, rows, ScriptedAnalyzer(draft(rows, reassign_call(rows))), trigger(
        rows, source_id=other.id, blocker_code="task_unrunnable", fingerprint="d" * 64,
        retry_target={"kind": "action", "source_action_id": str(other.id)}))
    await db_session.refresh(rows.run)
    assert rows.run.active_blockers != blockers_before, "precondition: lineage B mirrored its blocker warning"
    assert proposal_b.dispatch_contract["lineage_id"] != proposal_a.dispatch_contract["lineage_id"]

    decision = await decision_for(db_session, proposal_a)
    reservation = await svc_b.reserve_answer(db_session, decision, selected_option="approve_fix", reason=None,
                                             user_id=test_user.id)

    assert reservation.first_wrapper_action_id is not None, "lineage A's approval is not staled by lineage B"


@pytest.mark.asyncio
async def test_reworded_objective_cannot_repeat_a_failed_delegation(db_session, test_project, test_user, monkeypatch):
    adopt_session(db_session)
    from tests.test_orchestration_recovery_delegation import draft as delegation_draft

    flow, task = await failed_delegation(db_session, test_project, test_user, monkeypatch)
    flow.analyzer.drafts = [delegation_draft(flow.rows, objective="Inspect the hierarchy inputs and repair them carefully.")]

    outcome = await rediagnosed(flow.svc, db_session, flow.proposal,
                                evidence("The remediation task failed.", task.id, kind="task"))

    assert isinstance(outcome, RecoveryDiagnosisFailed), outcome
    assert "repeat" in outcome.code


@pytest.mark.asyncio
async def test_capability_answer_is_fenced_by_a_reset_reservation(db_session, test_project):
    from fastapi import HTTPException

    from huddleroom.schemas.orchestration_recovery import CapabilityGapAssessmentV1
    from huddleroom.services.orchestration_baseline_runtime import reset_baseline

    rows = await seed(db_session, test_project)
    gap = CapabilityGapAssessmentV1.model_validate({
        "kind": "capability_gap", "diagnosis": "No registered tool can grant network access.",
        "evidence": [{"label": "The source action failed.", "value": {"status": "failed"}, "source_kind": "action",
                      "source_id": str(rows.source.id)}],
        "missing_capability": "Network access for the verifier.",
        "required_human_action": "Grant the verifier network access.",
        "wake_after": {"kind": "human_decision", "decision": "Network access granted"},
    })
    svc, outcome = await propose(db_session, rows, ScriptedAnalyzer(gap))
    (decision,) = [row for row in await recovery_decisions(db_session, rows.goal.id, status_="pending")
                   if row.related_action_id == outcome.id]

    async with reset_baseline(rows.goal.id):
        with pytest.raises(HTTPException) as exc:
            await svc.answer_capability_gap(db_session, decision, user_id=uuid.uuid4())

    assert exc.value.status_code == 409
    await db_session.refresh(decision)
    assert decision.status == "pending"


# --------------------------------------------------------------------------- reported-blocker lineage


@pytest.mark.asyncio
async def test_reported_blockers_with_different_evidence_get_separate_lineages_and_repeats_join(
    db_session, test_project,
):
    adopt_session(db_session)
    from huddleroom.models.task import Task
    from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher
    from huddleroom.services.orchestration_service import OrchestrationService
    from tests.test_orchestration_recovery_activation import _actions, _authorized_goal, _decision

    _goal, run = await _authorized_goal(db_session, test_project)
    tasks = [Task(project_id=test_project.id, title=f"Blocked {i}", status="blocked",
                  metadata_={"orchestration": {"run_id": str(run.id)}}) for i in range(2)]
    db_session.add_all(tasks)
    await db_session.flush()
    dispatcher = OrchestrationDecisionDispatcher(OrchestrationService())

    async def report(task, reason):
        decision = await _decision(db_session, run, {
            "action_type": "recover_blocker", "reason": reason,
            "evidence_refs": [{"kind": "task", "id": str(task.id)}],
        })
        await dispatcher.dispatch(db_session, run, decision)

    await report(tasks[0], "First gate is stuck.")
    await report(tasks[1], "Second, unrelated gate is stuck.")
    await report(tasks[0], "First gate is still stuck.")

    triggers = await _actions(db_session, run.id, "recovery_trigger")
    assert len(triggers) == 2
    assert len({t.request["source_fingerprint"] for t in triggers}) == 2


@pytest.mark.asyncio
async def test_reported_blocker_evidence_variants_join_the_same_lineage(db_session, test_project):
    adopt_session(db_session)
    from huddleroom.models.task import Task
    from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher
    from huddleroom.services.orchestration_service import OrchestrationService
    from tests.test_orchestration_recovery_activation import _actions, _authorized_goal, _decision

    _goal, run = await _authorized_goal(db_session, test_project)
    task = Task(project_id=test_project.id, title="Blocked", status="blocked",
                metadata_={"orchestration": {"run_id": str(run.id)}})
    db_session.add(task)
    await db_session.flush()
    ref = {"kind": "task", "id": str(task.id)}
    variants = [
        [ref],
        [ref, ref],
        [{**ref, "note": "extra"}],
        [{"kind": " Task ", "id": f" {str(task.id).upper()} "}],
    ]
    dispatcher = OrchestrationDecisionDispatcher(OrchestrationService())
    for refs in variants:
        decision = await _decision(db_session, run, {
            "action_type": "recover_blocker", "reason": "Stuck.", "evidence_refs": refs})
        await dispatcher.dispatch(db_session, run, decision)

    assert len(await _actions(db_session, run.id, "recovery_trigger")) == 1


@pytest.mark.asyncio
async def test_malformed_reported_blocker_evidence_refs_are_ignored_not_fatal(db_session, test_project):
    adopt_session(db_session)
    from huddleroom.models.task import Task
    from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher
    from huddleroom.services.orchestration_service import OrchestrationService
    from tests.test_orchestration_recovery_activation import _actions, _authorized_goal, _decision

    _goal, run = await _authorized_goal(db_session, test_project)
    task = Task(project_id=test_project.id, title="Blocked", status="blocked",
                metadata_={"orchestration": {"run_id": str(run.id)}})
    db_session.add(task)
    await db_session.flush()
    ref = {"kind": "task", "id": str(task.id)}
    dispatcher = OrchestrationDecisionDispatcher(OrchestrationService())
    reports = []
    for refs in ([ref], [ref, f"task:{task.id}", None, 7, ["task", str(task.id)]]):
        decision = await _decision(db_session, run, {
            "action_type": "recover_blocker", "reason": "Stuck.", "evidence_refs": refs})
        reports.append(await dispatcher.dispatch(db_session, run, decision))

    # non-object refs neither crash the report nor split its lineage
    [trigger] = await _actions(db_session, run.id, "recovery_trigger")
    assert [(item.status, item.target_id) for item in reports] == [("completed", trigger.id)] * 2
