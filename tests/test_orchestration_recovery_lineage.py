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


async def _resolve_escalated_lineage_warning(db, rows, user, monkeypatch, resolved_by):
    from huddleroom.services.orchestration_recovery_registry import RECOVERY_WARNING_TYPE
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    analyzer = ScriptedAnalyzer(
        draft(rows, verify_call(rows)), draft(rows, verify_call(rows, "review")), draft(rows, retry_task_call(rows)),
        draft(rows, reassign_call(rows)),
    )
    flow, escalation = await _escalated(db, rows, user, monkeypatch, analyzer)
    (gap_decision,) = [row for row in await recovery_decisions(db, rows.goal.id, status_="pending")
                       if row.related_action_id == escalation.id]
    (warning,) = (await db.scalars(select(OrchestrationWarning).where(
        OrchestrationWarning.goal_id == rows.goal.id, OrchestrationWarning.warning_type == RECOVERY_WARNING_TYPE,
        OrchestrationWarning.active.is_(True)))).all()
    await OrchestrationWarningService().resolve_warning(
        db, warning, resolved_by=resolved_by, reason="The blocker was cleared outside recovery.")
    calls_before = analyzer.calls
    follow_up = await flow.svc.answer_capability_gap(db, gap_decision, user_id=user.id)
    outcome = await flow.svc.diagnose(db, follow_up.id)
    return analyzer, calls_before, outcome


@pytest.mark.asyncio
async def test_human_resolved_lineage_warning_closes_the_capability_answer_followup(
    db_session, test_project, test_user, monkeypatch,
):
    from huddleroom.services.orchestration_recovery_registry import RECOVERY_WARNING_TYPE

    rows = await seed(db_session, test_project)
    analyzer, calls_before, outcome = await _resolve_escalated_lineage_warning(
        db_session, rows, test_user, monkeypatch, resolved_by=f"human:{test_user.id}")

    assert outcome.dispatch_contract["closed"] == "blocker_gone"
    assert analyzer.calls == calls_before, "a human-resolved lineage warning retires the follow-up without analysis"
    active = await db_session.scalars(select(OrchestrationWarning).where(
        OrchestrationWarning.goal_id == rows.goal.id, OrchestrationWarning.warning_type == RECOVERY_WARNING_TYPE,
        OrchestrationWarning.active.is_(True)))
    assert active.all() == []
    assert await recovery_decisions(db_session, rows.goal.id, status_="pending") == []


@pytest.mark.asyncio
async def test_system_resolved_lineage_warning_still_rediagnoses(db_session, test_project, test_user, monkeypatch):
    rows = await seed(db_session, test_project)
    analyzer, calls_before, outcome = await _resolve_escalated_lineage_warning(
        db_session, rows, test_user, monkeypatch, resolved_by="orchestrator:recovery")

    assert analyzer.calls == calls_before + 1, "a system-resolved warning does not retire the follow-up"
    assert outcome.request.get("calls"), outcome.request


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


# --------------------------------------------------------------------------- SPR 284: resolved-warning reconciliation

from huddleroom.models.orchestration import OrchestrationGate, OrchestrationGoal, OrchestrationRun  # noqa: E402
from huddleroom.models.task import Task  # noqa: E402
from huddleroom.services.orchestration_blocker_recovery import OrchestrationBlockerRecoveryService  # noqa: E402
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService  # noqa: E402

_MIRROR = "warning:{}"


async def _reconcile(db, rows) -> int:
    return await OrchestrationBlockerRecoveryService().reconcile_resolved_warnings(db, rows.goal, rows.run)


async def _other_run(db, rows) -> OrchestrationRun:
    goal = OrchestrationGoal(project_id=rows.goal.project_id, objective="Another goal", constraints={}, budget={},
                             success_criteria=[{"key": "done", "description": "Done"}])
    db.add(goal)
    await db.flush()
    other = OrchestrationRun(goal_id=goal.id)
    db.add(other)
    await db.flush()
    return other


async def _task(db, rows, status="failed", run_id="same") -> Task:
    run_id = str(rows.run.id) if run_id == "same" else str(run_id)
    task = Task(project_id=rows.task.project_id, title=f"work {uuid.uuid4().hex[:6]}", assigned_to=rows.planner.id,
                status=status, metadata_={"orchestration": {"run_id": run_id}})
    db.add(task)
    await db.flush()
    return task


async def _gate(db, rows, status="failed", run_id=None) -> OrchestrationGate:
    gate = OrchestrationGate(run_id=run_id or rows.run.id, success_criterion_key=f"k-{uuid.uuid4().hex[:8]}",
                             gate_type="plan", status=status)
    db.add(gate)
    await db.flush()
    return gate


async def lineage(db, rows, *, target="task", subject=None, source_kind="failed_action", source_type="failure_observed",
                  closes=True, source_run=None, trigger_run=None, warning_type="recovery_blocked", session=True,
                  request_id=None):
    """A real failed_action lineage: trigger -> failure_observed source (-> task/gate) -> blocker warning on the root."""
    subject = subject or (await _task(db, rows) if target == "task" else await _gate(db, rows))
    # producer shape (orchestration_service._task_recovery_trigger): task_id + session_id (None when no failed session)
    request = ({"task_id": request_id or str(subject.id), "session_id": str(uuid.uuid4()) if session else None}
               if target == "task" else {"gate_id": request_id or str(subject.id)})
    if closes is not None:
        request["closes_when_task_healthy"] = closes
    source = OrchestrationAction(
        run_id=source_run or rows.run.id, idempotency_key=f"seed:{uuid.uuid4()}:failure", action_type=source_type,
        status="failed", request=request, error="boom", target_type=target, target_id=subject.id,
    )
    db.add(source)
    await db.flush()
    payload = trigger(rows, source_id=source.id, fingerprint=uuid.uuid4().hex + uuid.uuid4().hex).model_dump(mode="json")
    payload["source_kind"] = source_kind
    if trigger_run is not None:
        payload["run_id"] = str(trigger_run)
    root = OrchestrationAction(
        run_id=rows.run.id, idempotency_key=f"seed:{uuid.uuid4()}:trigger", action_type="recovery_trigger",
        status="completed", request=payload,
    )
    db.add(root)
    await db.flush()
    warning = await OrchestrationWarningService().create_warning(
        db, rows.goal.id, warning_type=warning_type, severity="blocker", message="The blocker is still active.",
        run_id=rows.run.id, related_action_id=root.id,
    )
    return SimpleNamespace(subject=subject, source=source, root=root, warning=warning)


async def _fresh(db, warning):
    await db.refresh(warning)
    return warning


def _mirrored(rows, warning) -> bool:
    return _MIRROR.format(warning.id) in [b.get("kind") for b in rows.run.active_blockers or [] if isinstance(b, dict)]


@pytest.mark.asyncio
async def test_failed_task_later_done_retires_its_warning_with_provenance(db_session, test_project):
    rows = await seed(db_session, test_project)
    item = await lineage(db_session, rows)
    assert _mirrored(rows, item.warning)
    assert await _reconcile(db_session, rows) == 0  # still failed

    item.subject.status = "done"
    assert await _reconcile(db_session, rows) == 1

    warning = await _fresh(db_session, item.warning)
    assert not warning.active and warning.resolved_at is not None
    assert warning.resolved_by == "orchestrator:recovery"
    assert str(item.source.id) in warning.resolved_reason
    assert not _mirrored(rows, warning)
    assert (await db_session.get(type(warning), warning.id)) is not None  # row kept for audit

    # Idempotent: a repeated pass changes neither attribution nor row count.
    resolved_at, reason = warning.resolved_at, warning.resolved_reason
    total = await db_session.scalar(select(func.count()).select_from(OrchestrationWarning))
    assert await _reconcile(db_session, rows) == 0
    await db_session.refresh(warning)
    assert (warning.resolved_at, warning.resolved_reason, warning.resolved_by) == (
        resolved_at, reason, "orchestrator:recovery")
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationWarning)) == total


@pytest.mark.asyncio
async def test_failed_gate_later_accepted_retires_its_warning(db_session, test_project):
    rows = await seed(db_session, test_project)
    item = await lineage(db_session, rows, target="gate")
    assert await _reconcile(db_session, rows) == 0
    item.subject.status = "accepted"
    assert await _reconcile(db_session, rows) == 1
    warning = await _fresh(db_session, item.warning)
    assert not warning.active and warning.resolved_by == "orchestrator:recovery"
    assert str(item.source.id) in warning.resolved_reason and not _mirrored(rows, warning)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["in_progress", "cancelled", "failed", "backlog", "blocked", "ready"])
async def test_task_not_done_keeps_warning(db_session, test_project, status):
    rows = await seed(db_session, test_project)
    item = await lineage(db_session, rows, subject=await _task(db_session, rows, status=status))
    assert await _reconcile(db_session, rows) == 0
    assert (await _fresh(db_session, item.warning)).active and _mirrored(rows, item.warning)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["open", "failed"])
async def test_gate_not_accepted_keeps_warning(db_session, test_project, status):
    rows = await seed(db_session, test_project)
    try:
        item = await lineage(db_session, rows, subject=await _gate(db_session, rows, status=status), target="gate")
    except Exception as exc:  # a status outside the gate CHECK is not a representable state
        pytest.skip(f"gate status {status!r} not representable: {exc}")
    assert await _reconcile(db_session, rows) == 0
    assert (await _fresh(db_session, item.warning)).active


@pytest.mark.asyncio
async def test_other_run_task_and_gate_do_not_retire(db_session, test_project):
    rows = await seed(db_session, test_project)
    other = await _other_run(db_session, rows)
    foreign_task = await lineage(db_session, rows, subject=await _task(db_session, rows, "done", run_id=other.id))
    foreign_gate = await lineage(db_session, rows, target="gate",
                                 subject=await _gate(db_session, rows, "accepted", run_id=other.id))
    foreign_source = await lineage(db_session, rows, subject=await _task(db_session, rows, "done"), source_run=other.id)
    foreign_trigger = await lineage(db_session, rows, subject=await _task(db_session, rows, "done"), trigger_run=other.id)
    assert await _reconcile(db_session, rows) == 0
    for item in (foreign_task, foreign_gate, foreign_source, foreign_trigger):
        assert (await _fresh(db_session, item.warning)).active


@pytest.mark.asyncio
async def test_unrelated_accepted_gate_or_different_done_task_does_not_retire(db_session, test_project):
    rows = await seed(db_session, test_project)
    await _gate(db_session, rows, "accepted")        # unrelated accepted gate in the same run
    await _task(db_session, rows, "done")            # unrelated done task in the same run
    on_task = await lineage(db_session, rows)        # its own task is still failed
    on_gate = await lineage(db_session, rows, target="gate")
    assert await _reconcile(db_session, rows) == 0
    assert (await _fresh(db_session, on_task.warning)).active and (await _fresh(db_session, on_gate.warning)).active


@pytest.mark.asyncio
@pytest.mark.parametrize("closes", [False, 0])
async def test_closes_when_task_healthy_false_keeps_warning(db_session, test_project, closes):
    rows = await seed(db_session, test_project)
    item = await lineage(db_session, rows, subject=await _task(db_session, rows, "done"), closes=closes)
    assert await _reconcile(db_session, rows) == 0
    assert (await _fresh(db_session, item.warning)).active


@pytest.mark.asyncio
@pytest.mark.parametrize("source_kind", ["reported_blocker", "eventless_noop", "planning_preflight"])
async def test_non_failed_action_roots_stay_for_a_human(db_session, test_project, source_kind):
    rows = await seed(db_session, test_project)
    # Even with a healthy-looking done task behind the source id, only failed_action roots are provable.
    item = await lineage(db_session, rows, subject=await _task(db_session, rows, "done"), source_kind=source_kind)
    assert await _reconcile(db_session, rows) == 0
    assert (await _fresh(db_session, item.warning)).active


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["session", "action"])
async def test_session_and_action_targets_stay_active(db_session, test_project, target):
    rows = await seed(db_session, test_project)
    done = await _task(db_session, rows, "done")
    item = await lineage(db_session, rows, subject=done)
    item.source.target_type = target  # a failure whose subject is not a task or gate has no health predicate
    await db_session.flush()
    assert await _reconcile(db_session, rows) == 0
    assert (await _fresh(db_session, item.warning)).active


@pytest.mark.asyncio
async def test_non_failure_observed_source_stays_active(db_session, test_project):
    rows = await seed(db_session, test_project)
    item = await lineage(db_session, rows, subject=await _task(db_session, rows, "done"), source_type="request_verification")
    assert await _reconcile(db_session, rows) == 0
    assert (await _fresh(db_session, item.warning)).active


@pytest.mark.asyncio
async def test_legacy_model_chosen_gate_warning_without_condition_stays(db_session, test_project):
    rows = await seed(db_session, test_project)
    gate = await _gate(db_session, rows, "accepted")
    legacy = await OrchestrationWarningService().create_warning(
        db_session, rows.goal.id, warning_type="summary_gate_reconciliation_unavailable", severity="blocker",
        message="No supported route reconciles the gate.", run_id=rows.run.id, related_gate_id=gate.id,
    )
    assert await _reconcile(db_session, rows) == 0
    assert (await _fresh(db_session, legacy)).active and _mirrored(rows, legacy)


@pytest.mark.asyncio
async def test_non_recovery_warning_types_are_never_touched(db_session, test_project):
    rows = await seed(db_session, test_project)
    # Same structural proof (done task) but not a recovery_blocked lineage: owner decisions/budget must stay.
    kept = [await lineage(db_session, rows, subject=await _task(db_session, rows, "done"), warning_type=kind)
            for kind in ("budget_exhausted", "owner_decision", "permission_required")]
    assert await _reconcile(db_session, rows) == 0
    for item in kept:
        assert (await _fresh(db_session, item.warning)).active


@pytest.mark.asyncio
async def test_missing_deleted_and_malformed_linkage_never_aborts_the_pass(db_session, test_project):
    rows = await seed(db_session, test_project)
    good = await lineage(db_session, rows, subject=await _task(db_session, rows, "done"))
    malformed = await lineage(db_session, rows, subject=await _task(db_session, rows, "done"))
    malformed.root.request = {"junk": True}
    no_source = await lineage(db_session, rows, subject=await _task(db_session, rows, "done"))
    await db_session.delete(no_source.source)
    no_task_subject = await _task(db_session, rows, "done")
    no_task = await lineage(db_session, rows, subject=no_task_subject)
    await db_session.delete(no_task_subject)
    deleted_root = await lineage(db_session, rows, subject=await _task(db_session, rows, "done"))
    await db_session.delete(deleted_root.root)
    unlinked = await OrchestrationWarningService().create_warning(
        db_session, rows.goal.id, warning_type="recovery_blocked", severity="blocker", message="No linkage.",
        run_id=rows.run.id)
    last = await lineage(db_session, rows, subject=await _task(db_session, rows, "done"))
    await db_session.flush()

    assert await _reconcile(db_session, rows) == 2  # good and last, whatever the order

    for kept in (malformed, no_source, no_task, deleted_root):
        assert (await _fresh(db_session, kept.warning)).active
    assert (await _fresh(db_session, unlinked)).active
    for done in (good, last):
        assert not (await _fresh(db_session, done.warning)).active


@pytest.mark.asyncio
async def test_many_recovery_warnings_all_reconciled_in_one_pass(db_session, test_project):
    rows = await seed(db_session, test_project)
    cleared = [await lineage(db_session, rows, subject=await _task(db_session, rows, "done")) for _ in range(60)]
    kept = [await lineage(db_session, rows, subject=await _task(db_session, rows, "failed")) for _ in range(5)]
    assert await _reconcile(db_session, rows) == 60
    for item in cleared:
        assert not (await _fresh(db_session, item.warning)).active
    for item in kept:
        assert (await _fresh(db_session, item.warning)).active


@pytest.mark.asyncio
async def test_manual_resolution_before_or_during_reconcile_is_not_overwritten(db_session, test_project, monkeypatch):
    rows = await seed(db_session, test_project)
    manual = await lineage(db_session, rows, subject=await _task(db_session, rows, "done"))
    # racing: a person resolves it after the scan, before the savepoint re-read (re-read skips it)
    racing = await lineage(db_session, rows, subject=await _task(db_session, rows, "done"))
    # racing_at_retire: a person resolves it after the re-read, inside resolve_warning (ValueError path skips it)
    racing_at_retire = await lineage(db_session, rows, subject=await _task(db_session, rows, "done"))
    clean = await lineage(db_session, rows, subject=await _task(db_session, rows, "done"))
    human = f"human:{uuid.uuid4()}"
    await OrchestrationWarningService().resolve_warning(db_session, manual.warning, resolved_by=human, reason="by hand")

    original_resolve = OrchestrationWarningService.resolve_warning
    original_cleared = OrchestrationBlockerRecoveryService._cleared_source

    async def human_resolves_after_scan(db, run, warning):
        cleared = await original_cleared(db, run, warning)
        if warning.id == racing.warning.id:
            await original_resolve(OrchestrationWarningService(), db, warning, resolved_by=human, reason="by hand")
        return cleared

    async def human_wins_at_retire(self, db, warning, *, resolved_by, reason):
        if warning.id == racing_at_retire.warning.id:
            await original_resolve(self, db, warning, resolved_by=human, reason="by hand")
        return await original_resolve(self, db, warning, resolved_by=resolved_by, reason=reason)

    monkeypatch.setattr(OrchestrationBlockerRecoveryService, "_cleared_source", staticmethod(human_resolves_after_scan))
    monkeypatch.setattr(OrchestrationWarningService, "resolve_warning", human_wins_at_retire)
    # a human-won race is a benign skip: no exception, only the unraced lineage is retired
    assert await _reconcile(db_session, rows) == 1
    warning = await _fresh(db_session, manual.warning)
    assert not warning.active and warning.resolved_by == human and warning.resolved_reason == "by hand"
    for raced in (racing, racing_at_retire):
        raced_warning = await _fresh(db_session, raced.warning)
        assert not raced_warning.active and raced_warning.resolved_by == human
        assert raced_warning.resolved_reason == "by hand"
    assert not (await _fresh(db_session, clean.warning)).active
    assert (await _fresh(db_session, clean.warning)).resolved_by == "orchestrator:recovery"


@pytest.mark.asyncio
async def test_no_session_task_failure_is_never_retired_but_session_task_is(db_session, test_project):
    rows = await seed(db_session, test_project)
    # the no-session episode key (status:failed) is reused, so a later re-failure would be hidden
    nosession = await lineage(db_session, rows, subject=await _task(db_session, rows, "done"), session=False)
    withsession = await lineage(db_session, rows, subject=await _task(db_session, rows, "done"))
    assert await _reconcile(db_session, rows) == 1
    assert (await _fresh(db_session, nosession.warning)).active and _mirrored(rows, nosession.warning)
    assert not (await _fresh(db_session, withsession.warning)).active


@pytest.mark.asyncio
async def test_mismatched_request_target_id_is_not_retired(db_session, test_project):
    rows = await seed(db_session, test_project)
    bad_task = await lineage(db_session, rows, subject=await _task(db_session, rows, "done"), request_id=str(uuid.uuid4()))
    bad_gate = await lineage(db_session, rows, target="gate", subject=await _gate(db_session, rows, "accepted"),
                             request_id=str(uuid.uuid4()))
    assert await _reconcile(db_session, rows) == 0
    for item in (bad_task, bad_gate):
        assert (await _fresh(db_session, item.warning)).active and _mirrored(rows, item.warning)


@pytest.mark.asyncio
async def test_two_passes_second_returns_zero_and_attribution_unchanged(db_session, test_project):
    rows = await seed(db_session, test_project)
    item = await lineage(db_session, rows, subject=await _task(db_session, rows, "done"))
    assert await _reconcile(db_session, rows) == 1
    first = await _fresh(db_session, item.warning)
    snap = (first.resolved_at, first.resolved_by, first.resolved_reason)
    assert await _reconcile(db_session, rows) == 0
    second = await _fresh(db_session, item.warning)
    assert (second.resolved_at, second.resolved_by, second.resolved_reason) == snap
    assert not _mirrored(rows, second)


@pytest.mark.asyncio
async def test_pending_recovery_decision_in_same_lineage_is_left_open(db_session, test_project):
    rows = await seed(db_session, test_project)
    _svc, proposal = await propose(db_session, rows, ScriptedAnalyzer(draft(rows, verify_call(rows))))
    decision = await decision_for(db_session, proposal)
    assert decision.status == "pending"
    # a separate cleared task-failure lineage in the same run
    item = await lineage(db_session, rows, subject=await _task(db_session, rows, "done"))
    assert await _reconcile(db_session, rows) >= 1
    assert not (await _fresh(db_session, item.warning)).active
    await db_session.refresh(decision)
    assert decision.status == "pending", "retiring a warning never answers or closes a recovery decision"


@pytest.mark.asyncio
async def test_resolve_error_propagates_and_savepoint_rolls_back(db_session, test_project, monkeypatch):
    rows = await seed(db_session, test_project)
    item = await lineage(db_session, rows, subject=await _task(db_session, rows, "done"))
    original = OrchestrationWarningService.resolve_warning

    async def partial_then_boom(self, db, warning, *, resolved_by, reason):
        await original(self, db, warning, resolved_by=resolved_by, reason=reason)  # partial work inside the savepoint
        raise RuntimeError("boom")

    monkeypatch.setattr(OrchestrationWarningService, "resolve_warning", partial_then_boom)
    with pytest.raises(RuntimeError, match="boom"):
        await _reconcile(db_session, rows)
    monkeypatch.undo()

    await db_session.refresh(item.warning)
    await db_session.refresh(rows.run)
    assert item.warning.active and item.warning.resolved_at is None
    assert _mirrored(rows, item.warning)


# --------------------------------------------------------------------------- M1: gate source already accepted


class _NoAnalysis:
    calls = 0

    async def analyze(self, *args, **kwargs):
        raise AssertionError("a trigger whose gate is already accepted must not be analyzed")


async def _reserved_trigger(db, item, *, follow_up: bool) -> OrchestrationAction:
    if not follow_up:
        item.root.status = "reserved"
        await db.flush()
        return item.root
    follow = OrchestrationAction(
        run_id=item.root.run_id, idempotency_key=f"seed:{uuid.uuid4()}:followup", action_type="recovery_trigger",
        status="reserved", request=item.root.request,
        dispatch_contract={"root_trigger_action_id": str(item.root.id), "attempt": 2,
                           "supersedes_proposal_action_id": str(uuid.uuid4())},
    )
    db.add(follow)
    await db.flush()
    return follow


async def _active_recovery_warnings(db, goal_id):
    return (await db.scalars(select(OrchestrationWarning).where(
        OrchestrationWarning.goal_id == goal_id, OrchestrationWarning.warning_type == "recovery_blocked",
        OrchestrationWarning.active.is_(True)).execution_options(populate_existing=True))).all()


@pytest.mark.asyncio
@pytest.mark.parametrize("follow_up", [False, True], ids=["root_trigger", "follow_up_trigger"])
async def test_accepted_gate_source_retires_warning_and_closes_trigger_without_analysis(
    db_session, test_project, follow_up,
):
    rows = await seed(db_session, test_project)
    item = await lineage(db_session, rows, target="gate")
    reserved = await _reserved_trigger(db_session, item, follow_up=follow_up)
    item.subject.status = "accepted"

    assert await _reconcile(db_session, rows) == 1  # tick order: reconcile, then diagnose reserved triggers
    outcome = await OrchestrationBlockerRecoveryService(analyzer=_NoAnalysis()).diagnose(db_session, reserved.id)

    assert outcome.status == "completed" and outcome.dispatch_contract["closed"] == "blocker_gone"
    assert await _active_recovery_warnings(db_session, rows.goal.id) == []
    assert not (await _fresh(db_session, item.warning)).active


@pytest.mark.asyncio
@pytest.mark.parametrize("follow_up", [False, True], ids=["root_trigger", "follow_up_trigger"])
async def test_gate_not_accepted_still_diagnoses_and_keeps_warning(db_session, test_project, follow_up):
    rows = await seed(db_session, test_project)
    item = await lineage(db_session, rows, target="gate")  # gate stays failed
    reserved = await _reserved_trigger(db_session, item, follow_up=follow_up)
    analyzer = ScriptedAnalyzer(draft(rows, verify_call(rows)))

    assert await _reconcile(db_session, rows) == 0
    outcome = await rediagnosed_trigger(db_session, analyzer, reserved)

    assert analyzer.calls >= 1, outcome
    assert (outcome.dispatch_contract or {}).get("closed") != "blocker_gone"
    assert (await _fresh(db_session, item.warning)).active


async def rediagnosed_trigger(db, analyzer, trigger_action):
    try:
        return await OrchestrationBlockerRecoveryService(analyzer=analyzer).diagnose(db, trigger_action.id)
    except RecoveryDiagnosisFailed:
        return trigger_action
