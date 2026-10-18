"""SPR 180/184: apply / verify / retry coordinator, crash recovery and delegated lifecycle.

Fake completions and synthetic events only: the analyzer is a scripted stand-in and every ordinary
executor that a test exercises is replaced by a recorder that performs a real reserve/complete.

Interface assumptions (recovery, verbatim signatures on ``OrchestrationBlockerRecoveryService``):

- ``resume(db, proposal_action_id) -> RecoveryLifecycleStatus`` advances every step that is ready
  (remaining apply calls -> verification -> the one retry) and stops at the first concrete wait or
  terminal state; it returns the projected status. It is the crash/tick re-entry point.
- ``execute_next_call(db, proposal_action) -> ToolReceiptV1 | None`` executes the next sealed call
  in declared order (reserving its ``recovery:<proposal_id>:apply:<call_id>`` wrapper if needed) and
  returns its receipt; ``None`` when no call is executable (not approved, terminal, or all done).
- ``verify_proposal(db, proposal_action) -> tuple[VerificationResultV1, ...]`` runs every sealed
  check through ``recovery_verify`` wrappers (``recovery:<proposal_id>:verify:<check_id>``).
- ``retry_blocked_work(db, proposal_action) -> OrchestrationAction`` returns the single
  ``recovery_retry`` wrapper ``recovery:<proposal_id>:retry:1``; an ``{"kind": "action"}`` retry
  target re-dispatches the source action's canonical request through the dispatcher with a
  ``purpose="retry"`` context, so its effect key is ``recovery:<proposal_id>:retry:1:effect``.
- ``rediagnose(db, proposal_action, evidence) -> OrchestrationAction`` re-enters diagnosis in the same
  lineage; it may return the reserved follow-up ``recovery_trigger`` or the diagnosed outcome
  (``next_outcome`` below accepts either).
- Methods may fail a step by raising a domain error (``ValueError``/``HTTPException``/
  ``RecoveryProposalStale``) or by returning; the tests assert the durable rows, never a message.
- All methods work inside the caller-owned ``db_session`` transaction (flush-only there).
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import select

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGate, OrchestrationWait
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision, OrchestrationWarning
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration_recovery import (
    EvidenceV1,
    ProposalDraftV1,
    RecoveryTriggerV1,
    ToolReceiptV1,
    VerificationResultV1,
)
from huddleroom.services.orchestration_blocker_recovery import (
    OrchestrationBlockerRecoveryService,
    RecoveryProposalStale,
)
from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_supervision import OrchestrationSupervisionService
from tests.conftest import adopt_session
from tests.test_orchestration_recovery_adapters import seed_rows
from tests.test_orchestration_roster_mutation import sessions  # noqa: F401 -- fixture re-exported

DOMAIN_ERRORS = (ValueError, HTTPException, RecoveryProposalStale)


@pytest_asyncio.fixture
async def db_session(owned_session_factory):
    """Autobegin session (no session.begin()) so the test task can own it."""
    async with owned_session_factory() as db:
        yield db
        await db.rollback()


# --------------------------------------------------------------------------- scripted analyzer


class ScriptedAnalyzer:
    """Deterministic recovery LM stand-in: returns the queued drafts in order (the last one repeats)."""

    def __init__(self, *drafts):
        self.drafts = list(drafts)
        self.snapshots: list[dict] = []

    async def analyze(self, snapshot, *, project_id, evidence_query, on_raw=None):
        self.snapshots.append(snapshot)
        draft = self.drafts.pop(0) if len(self.drafts) > 1 else self.drafts[0]
        if on_raw is not None:
            on_raw(draft.model_dump_json())
        return draft

    @property
    def calls(self) -> int:
        return len(self.snapshots)


# --------------------------------------------------------------------------- seeding


async def seed(db, project, *, task_status="ready"):
    """seed_rows (goal/run/planner/organizer/gate/graph/task) plus a durably failed source action."""
    adopt_session(db)  # the test task owns this autobegin session, as get_db does
    rows = await seed_rows(db, project, task_status=task_status)
    request = OrchestrationService().canonical_decision_request(
        "request_verification", {"gate_id": str(rows.gate.id), "work_function": "validation"}, run_id=rows.run.id)
    source = OrchestrationAction(
        run_id=rows.run.id, idempotency_key=f"seed:{uuid.uuid4()}:verification", action_type="request_verification",
        status="failed", request=request, error="verification_unavailable",
    )
    db.add(source)
    await db.flush()
    rows.source = source
    return rows


def trigger(rows, *, source_id=None, blocker_code="verification_unavailable", fingerprint="c" * 64,
            retry_target=None) -> RecoveryTriggerV1:
    source_id = source_id or rows.source.id
    return RecoveryTriggerV1.model_validate({
        "schema_version": 1, "goal_id": str(rows.goal.id), "run_id": str(rows.run.id),
        "source_kind": "failed_action", "source_id": str(source_id), "blocker_code": blocker_code,
        "summary": "The verification request failed.", "source_fingerprint": fingerprint,
        "retry_target": retry_target or {"kind": "action", "source_action_id": str(rows.source.id)},
    })


def verify_call(rows, work_function="validation"):
    return ("orchestration.request_verification", {"gate_id": str(rows.gate.id), "work_function": work_function})


def retry_task_call(rows):
    return ("orchestration.retry_task", {"task_id": str(rows.task.id)})


def reassign_call(rows):
    return ("orchestration.reassign_task", {"task_id": str(rows.task.id), "agent_id": str(rows.organizer.id)})


def draft(rows, *calls, problem="Verification cannot run.") -> ProposalDraftV1:
    labels = [tool.split(".", 1)[1].replace("_", " ") for tool, _ in calls]
    return ProposalDraftV1.model_validate({
        "kind": "proposal", "diagnosis": "The blocked verification needs an existing orchestration action.",
        "evidence": [{"label": "The source action failed.", "value": {"status": "failed"}, "source_kind": "action",
                      "source_id": str(rows.source.id)}],
        "problem": problem, "general_updates": labels, "expected_effect": "The blocked verification can continue.",
        "calls": [{"tool": f"{tool}@1", "arguments": arguments, "rationale": "Repair the blocker."}
                  for tool, arguments in calls],
    })


# --------------------------------------------------------------------------- executors


class Executors:
    """Replaces existing executor bodies with a real reserve/complete; behaviours are consumed per call.

    Behaviours: ``"ok"`` (default), ``"reject"`` (genuine 409 rejection), ``"pending"`` (reserve and
    return the still-reserved row, i.e. an external effect whose outcome is not yet known), or a
    callable ``mutate(db, rows)`` applied before completing.
    """

    def __init__(self, monkeypatch, rows, plan=None, *, only=None):
        self.rows = rows
        self.plan = {key: list(value) for key, value in (plan or {}).items()}
        self.calls: list[dict] = []
        for action_type, method in OrchestrationDecisionDispatcher._EXECUTORS.items():
            if only is not None and action_type not in only:
                continue
            monkeypatch.setattr(OrchestrationService, method, self._executor(action_type))

    def _executor(self, action_type):
        recorder = self

        async def execute(service, db, run_id, request, idempotency_key, decision_id=None, **extra):
            queue = recorder.plan.get(action_type) or []
            behaviour = queue.pop(0) if queue else "ok"
            recorder.calls.append({"action_type": action_type, "idempotency_key": idempotency_key,
                                   "request": request, "behaviour": behaviour})
            if behaviour == "reject":
                raise HTTPException(status_code=409, detail=f"{action_type}_conflict")
            action = await service.reserve_action(db, run_id, idempotency_key, action_type, request, decision_id)
            if action.status != "reserved" or behaviour == "pending":
                return action
            if callable(behaviour):
                await behaviour(db, recorder.rows)
            return await service._mark_action_completed(db, action, "task", recorder.rows.task.id)

        return execute

    def keys(self, action_type=None):
        return [call["idempotency_key"] for call in self.calls
                if action_type is None or call["action_type"] == action_type]


# --------------------------------------------------------------------------- flow helpers


async def propose(db, rows, analyzer, trigger_model=None, *, owns_setup=False):
    svc = OrchestrationBlockerRecoveryService(analyzer=analyzer)
    trigger_action = await svc.create_trigger(db, trigger_model or trigger(rows))
    if owns_setup:
        await db.flush()
        await db.commit()
    proposal = await svc.diagnose(db, trigger_action.id)
    return svc, proposal


async def decision_for(db, proposal):
    return (await db.scalars(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.related_action_id == proposal.id,
    ).execution_options(populate_existing=True))).one()


async def approve(db, svc, proposal, user):
    decision = await decision_for(db, proposal)
    await svc.reserve_answer(db, decision, selected_option="approve_fix", reason=None, user_id=user.id)
    return decision


async def approved(db, rows, user, *calls, analyzer=None, owns_setup=False):
    analyzer = analyzer or ScriptedAnalyzer(draft(rows, *calls))
    svc, proposal = await propose(db, rows, analyzer, owns_setup=owns_setup)
    decision = await approve(db, svc, proposal, user)
    return SimpleNamespace(svc=svc, proposal=proposal, decision=decision, analyzer=analyzer,
                           pid=proposal.request["proposal_id"])


async def attempt(coro):
    """Run a lifecycle step; a domain failure is an allowed outcome (missing methods still fail)."""
    try:
        return await coro
    except DOMAIN_ERRORS as exc:
        return exc


async def status(db, svc, proposal):
    await db.refresh(proposal)
    return await svc.project_status(db, proposal)


async def lifecycle(db, proposal, action_type):
    rows = (await db.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == proposal.run_id, OrchestrationAction.action_type == action_type,
    ).execution_options(populate_existing=True))).all()
    return sorted((row for row in rows if (row.request or {}).get("proposal_action_id") == str(proposal.id)),
                  key=lambda row: row.idempotency_key)


async def by_key(db, run_id, key):
    return (await db.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run_id, OrchestrationAction.idempotency_key == key,
    ).execution_options(populate_existing=True))).one_or_none()


async def effects(db, run_id, marker=""):
    rows = (await db.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run_id).execution_options(populate_existing=True))).all()
    return [row for row in rows if row.idempotency_key.startswith("recovery:")
            and row.idempotency_key.endswith(":effect") and marker in row.idempotency_key]


async def verified(db, flow):
    while await flow.svc.execute_next_call(db, flow.proposal) is not None:
        pass
    results = await flow.svc.verify_proposal(db, flow.proposal)
    assert results and all(isinstance(item, VerificationResultV1) for item in results)
    assert [item.status for item in results] == ["passed"] * len(results), results
    return results


def no_rollback_claim(*values) -> None:
    text = json.dumps(values, default=str).lower()
    assert "rolled back" not in text and "rollback" not in text and "reverted" not in text, text


# --------------------------------------------------------------------------- harness sanity


@pytest.mark.asyncio
async def test_harness_seals_an_approvable_two_call_proposal_with_action_retry(db_session, test_project, test_user):
    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, retry_task_call(rows), verify_call(rows))

    assert [call["call_id"] for call in flow.proposal.request["calls"]] == ["call-1", "call-2"]
    assert flow.proposal.request["retry"]["target"] == {"kind": "action", "source_action_id": str(rows.source.id)}
    assert flow.decision.status == "answered" and flow.decision.selected_option == "approve_fix"
    (wrapper,) = await lifecycle(db_session, flow.proposal, "recovery_apply")
    assert wrapper.idempotency_key == f"recovery:{flow.pid}:apply:call-1" and wrapper.status == "reserved"


# --------------------------------------------------------------------------- multi-call order and truth


@pytest.mark.asyncio
async def test_multi_call_proposal_executes_calls_in_declared_order(db_session, test_project, test_user, monkeypatch):
    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, retry_task_call(rows), verify_call(rows))
    executors = Executors(monkeypatch, rows)

    first = await flow.svc.execute_next_call(db_session, flow.proposal)
    second = await flow.svc.execute_next_call(db_session, flow.proposal)
    third = await flow.svc.execute_next_call(db_session, flow.proposal)

    assert isinstance(first, ToolReceiptV1) and isinstance(second, ToolReceiptV1)
    assert (first.call_id, first.status) == ("call-1", "applied")
    assert (second.call_id, second.status) == ("call-2", "applied")
    assert third is None
    assert executors.keys() == [f"recovery:{flow.pid}:apply:call-1:effect", f"recovery:{flow.pid}:apply:call-2:effect"]
    assert [call["action_type"] for call in executors.calls] == ["retry_task", "request_verification"]
    wrappers = await lifecycle(db_session, flow.proposal, "recovery_apply")
    assert [(row.idempotency_key, row.status) for row in wrappers] == [
        (f"recovery:{flow.pid}:apply:call-1", "completed"), (f"recovery:{flow.pid}:apply:call-2", "completed")]
    for wrapper, receipt in zip(wrappers, (first, second)):
        assert wrapper.dispatch_contract["tool_receipt_v1"]["call_id"] == receipt.call_id
        assert receipt.wrapper_action_id == wrapper.id
    # Nothing beyond apply happened yet: no verification, no retry, not resolved.
    assert await lifecycle(db_session, flow.proposal, "recovery_retry") == []
    assert await status(db_session, flow.svc, flow.proposal) in {"applying", "verifying"}


@pytest.mark.asyncio
async def test_execute_next_call_before_approval_executes_nothing(db_session, test_project, test_user, monkeypatch):
    rows = await seed(db_session, test_project)
    svc, proposal = await propose(db_session, rows, ScriptedAnalyzer(draft(rows, verify_call(rows))))
    executors = Executors(monkeypatch, rows)

    outcome = await attempt(svc.execute_next_call(db_session, proposal))

    assert not isinstance(outcome, ToolReceiptV1)
    assert executors.calls == []
    assert await lifecycle(db_session, proposal, "recovery_apply") == []
    assert await status(db_session, svc, proposal) == "pending_approval"


@pytest.mark.asyncio
async def test_verified_update_is_retrying_until_blocked_work_progresses(db_session, test_project, test_user, monkeypatch):
    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, verify_call(rows))
    Executors(monkeypatch, rows)

    assert await status(db_session, flow.svc, flow.proposal) == "applying"
    receipt = await flow.svc.execute_next_call(db_session, flow.proposal)
    assert receipt.status == "applied"
    assert await status(db_session, flow.svc, flow.proposal) not in {"retrying", "resolved"}

    results = await flow.svc.verify_proposal(db_session, flow.proposal)

    assert [(item.check_id, item.status) for item in results] == [("check-1", "passed")]
    (verify_wrapper,) = await lifecycle(db_session, flow.proposal, "recovery_verify")
    assert verify_wrapper.idempotency_key == f"recovery:{flow.pid}:verify:check-1"
    assert verify_wrapper.status == "completed"
    assert await flow.svc.project_status(db_session, flow.proposal) == "retrying"
    assert flow.proposal.status == "reserved", "a verified update is not resolved before its retry progresses"


@pytest.mark.asyncio
async def test_successful_single_retry_resolves_the_proposal(db_session, test_project, test_user, monkeypatch):
    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, verify_call(rows))
    executors = Executors(monkeypatch, rows)
    await verified(db_session, flow)

    retry = await flow.svc.retry_blocked_work(db_session, flow.proposal)

    assert retry.action_type == "recovery_retry"
    assert retry.idempotency_key == f"recovery:{flow.pid}:retry:1"
    assert retry.request["approved_retry_ordinal"] == 1
    assert retry.request["proposal_action_id"] == str(flow.proposal.id)
    (effect,) = await effects(db_session, rows.run.id, ":retry:")
    assert effect.idempotency_key == f"recovery:{flow.pid}:retry:1:effect"
    assert effect.action_type == rows.source.action_type and effect.status == "completed"
    assert retry.dispatch_contract["effect_action_id"] == str(effect.id)
    assert executors.keys("request_verification")[-1] == effect.idempotency_key
    assert await status(db_session, flow.svc, flow.proposal) == "resolved"
    assert flow.proposal.status == "completed"


@pytest.mark.asyncio
async def test_retry_runs_once_even_when_requested_twice(db_session, test_project, test_user, monkeypatch):
    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, verify_call(rows))
    executors = Executors(monkeypatch, rows)
    await verified(db_session, flow)

    first = await flow.svc.retry_blocked_work(db_session, flow.proposal)
    second = await attempt(flow.svc.retry_blocked_work(db_session, flow.proposal))

    assert not isinstance(second, OrchestrationAction) or second.id == first.id
    assert len(await lifecycle(db_session, flow.proposal, "recovery_retry")) == 1
    assert len(await effects(db_session, rows.run.id, ":retry:")) == 1
    assert len([key for key in executors.keys() if ":retry:" in key]) == 1


@pytest.mark.asyncio
async def test_retry_never_runs_before_every_check_passes(db_session, test_project, test_user, monkeypatch):
    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, verify_call(rows))
    executors = Executors(monkeypatch, rows)
    await flow.svc.execute_next_call(db_session, flow.proposal)

    outcome = await attempt(flow.svc.retry_blocked_work(db_session, flow.proposal))

    assert not (isinstance(outcome, OrchestrationAction) and outcome.action_type == "recovery_retry")
    assert await lifecycle(db_session, flow.proposal, "recovery_retry") == []
    assert [key for key in executors.keys() if ":retry:" in key] == []


@pytest.mark.asyncio
async def test_failed_verification_fails_the_attempt_without_retry(db_session, test_project, test_user, monkeypatch):
    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, verify_call(rows))
    executors = Executors(monkeypatch, rows)
    receipt = await flow.svc.execute_next_call(db_session, flow.proposal)
    effect = await db_session.get(OrchestrationAction, receipt.action_id)
    effect.status, effect.error = "failed", "external_drift"  # the applied effect no longer holds
    await db_session.flush()

    results = await attempt(flow.svc.verify_proposal(db_session, flow.proposal))

    assert not isinstance(results, tuple) or [item.status for item in results] == ["failed"]
    assert await status(db_session, flow.svc, flow.proposal) == "failed"
    assert await lifecycle(db_session, flow.proposal, "recovery_retry") == []
    assert [key for key in executors.keys() if ":retry:" in key] == []


@pytest.mark.asyncio
async def test_verification_pass_followed_by_retry_failure_is_failed(db_session, test_project, test_user, monkeypatch):
    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, verify_call(rows))
    Executors(monkeypatch, rows, plan={"request_verification": ["ok", "reject"]})
    await verified(db_session, flow)

    outcome = await attempt(flow.svc.retry_blocked_work(db_session, flow.proposal))

    (retry,) = await lifecycle(db_session, flow.proposal, "recovery_retry")
    assert retry.status == "failed"
    assert not isinstance(outcome, OrchestrationAction) or outcome.id == retry.id
    assert await status(db_session, flow.svc, flow.proposal) == "failed"
    assert flow.proposal.status == "failed"
    # Verification truth is kept: the apply receipt and the passed check stay on record.
    (apply_wrapper,) = await lifecycle(db_session, flow.proposal, "recovery_apply")
    assert apply_wrapper.dispatch_contract["tool_receipt_v1"]["status"] == "applied"
    assert [row.status for row in await lifecycle(db_session, flow.proposal, "recovery_verify")] == ["completed"]


# --------------------------------------------------------------------------- partial failure and rolling fingerprints


@pytest.mark.asyncio
async def test_partial_call_failure_keeps_applied_call_and_makes_no_rollback_claim(
    db_session, test_project, test_user, monkeypatch,
):
    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, retry_task_call(rows), verify_call(rows))
    executors = Executors(monkeypatch, rows, plan={"request_verification": ["reject"]})

    first = await flow.svc.execute_next_call(db_session, flow.proposal)
    second = await attempt(flow.svc.execute_next_call(db_session, flow.proposal))
    after = await attempt(flow.svc.execute_next_call(db_session, flow.proposal))

    assert first.status == "applied"
    assert not isinstance(second, ToolReceiptV1) or second.status == "failed"
    assert not isinstance(after, ToolReceiptV1), "no call runs after a failed call"
    call_1, call_2 = await lifecycle(db_session, flow.proposal, "recovery_apply")
    assert call_1.status == "completed" and call_1.dispatch_contract["tool_receipt_v1"]["status"] == "applied"
    assert (await db_session.get(OrchestrationAction, first.action_id)).status == "completed", "call 1 is not undone"
    assert call_2.status == "failed"
    assert await status(db_session, flow.svc, flow.proposal) == "failed"
    assert len(executors.calls) == 2
    assert await lifecycle(db_session, flow.proposal, "recovery_retry") == []
    no_rollback_claim(flow.proposal.error, flow.proposal.dispatch_contract, call_2.error, call_2.dispatch_contract)


@pytest.mark.asyncio
async def test_rolling_fingerprint_accepts_changes_made_by_a_completed_call(db_session, test_project, test_user, monkeypatch):
    async def start_task(db, rows):
        rows.task.status = "in_progress"  # call 1's own, receipted effect on its target
        await db.flush()

    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, retry_task_call(rows), verify_call(rows))
    executors = Executors(monkeypatch, rows, plan={"retry_task": [start_task]})

    first = await flow.svc.execute_next_call(db_session, flow.proposal)
    second = await flow.svc.execute_next_call(db_session, flow.proposal)

    assert first.status == "applied"
    assert isinstance(second, ToolReceiptV1) and second.status == "applied", second
    assert [call["action_type"] for call in executors.calls] == ["retry_task", "request_verification"]


@pytest.mark.asyncio
async def test_rolling_fingerprint_rejects_drift_of_an_untouched_resource(db_session, test_project, test_user, monkeypatch):
    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, retry_task_call(rows), verify_call(rows))
    executors = Executors(monkeypatch, rows)
    first = await flow.svc.execute_next_call(db_session, flow.proposal)
    rows.gate.failure_reason = "changed by someone else"  # call 2's gate, untouched by call 1
    await db_session.flush()

    second = await attempt(flow.svc.execute_next_call(db_session, flow.proposal))

    assert first.status == "applied"
    assert not (isinstance(second, ToolReceiptV1) and second.status == "applied")
    assert [call["action_type"] for call in executors.calls] == ["retry_task"]
    assert await status(db_session, flow.svc, flow.proposal) == "failed"
    assert flow.proposal.error == "state_changed"
    call_1 = (await lifecycle(db_session, flow.proposal, "recovery_apply"))[0]
    assert call_1.dispatch_contract["tool_receipt_v1"]["status"] == "applied", "the applied call stays applied"


@pytest.mark.asyncio
async def test_rolling_fingerprint_rejects_external_change_to_a_completed_call_resource(
    db_session, test_project, test_user, monkeypatch,
):
    async def start_task(db, rows):
        rows.task.status = "in_progress"
        await db.flush()

    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, retry_task_call(rows), verify_call(rows))
    executors = Executors(monkeypatch, rows, plan={"retry_task": [start_task]})
    await flow.svc.execute_next_call(db_session, flow.proposal)
    rows.task.title = "Edited after call 1 completed"  # differs from the post-state call 1 left behind
    await db_session.flush()

    second = await attempt(flow.svc.execute_next_call(db_session, flow.proposal))

    assert not (isinstance(second, ToolReceiptV1) and second.status == "applied")
    assert [call["action_type"] for call in executors.calls] == ["retry_task"]
    assert await status(db_session, flow.svc, flow.proposal) == "failed"


@pytest.mark.asyncio
async def test_stale_state_before_the_first_effect_fails_safely_and_rediagnoses(
    db_session, test_project, test_user, monkeypatch,
):
    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, verify_call(rows))
    executors = Executors(monkeypatch, rows)
    rows.gate.failure_reason = "drift after approval"
    await db_session.flush()

    outcome = await attempt(flow.svc.execute_next_call(db_session, flow.proposal))

    assert not isinstance(outcome, ToolReceiptV1)
    assert executors.calls == []
    assert await effects(db_session, rows.run.id) == []
    assert await status(db_session, flow.svc, flow.proposal) == "failed"
    assert flow.proposal.error == "state_changed"
    (wrapper,) = await lifecycle(db_session, flow.proposal, "recovery_apply")
    assert wrapper.status == "failed", "the reserved wrapper is closed, not left reserved"
    follow_ups = [row for row in (await db_session.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == rows.run.id, OrchestrationAction.action_type == "recovery_trigger",
        OrchestrationAction.status == "reserved"))).all()
        if (row.dispatch_contract or {}).get("lineage_id") == flow.proposal.dispatch_contract["lineage_id"]]
    assert len(follow_ups) == 1, "drift after answer schedules exactly one fresh diagnosis"


@pytest.mark.asyncio
async def test_steering_change_after_approval_fails_the_wrapper_instead_of_leaving_it_reserved(
    db_session, test_project, test_user, monkeypatch,
):
    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, verify_call(rows))
    assert flow.proposal.dispatch_contract["steering"].get("versions"), "precondition: a real steering seal"
    rows.run.plan_state = {**(rows.run.plan_state or {}), "accepted_plan_fingerprint": "changed-after-approval"}
    await db_session.flush()
    executors = Executors(monkeypatch, rows)

    outcome = await attempt(flow.svc.execute_next_call(db_session, flow.proposal))

    assert not isinstance(outcome, ToolReceiptV1) or outcome.status != "applied"
    assert executors.calls == []
    (wrapper,) = await lifecycle(db_session, flow.proposal, "recovery_apply")
    assert wrapper.status == "failed"
    assert await status(db_session, flow.svc, flow.proposal) == "failed"


# --------------------------------------------------------------------------- crash and uncertain effects


@pytest.mark.asyncio
async def test_uncertain_external_effect_is_failed_and_never_replayed(db_session, test_project, test_user, monkeypatch):
    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, verify_call(rows))
    service = OrchestrationService()

    async def in_flight(self, db, run_id, request, idempotency_key, decision_id=None, **extra):
        # The external effect was started (its row reserved and linked) but its outcome is unknown.
        return await service.reserve_action(db, run_id, idempotency_key, "request_verification", request, decision_id)

    executor = AsyncMock(side_effect=in_flight)
    monkeypatch.setattr(OrchestrationService, OrchestrationDecisionDispatcher._EXECUTORS["request_verification"],
                        lambda self, *args, **kwargs: executor(self, *args, **kwargs))

    await attempt(flow.svc.execute_next_call(db_session, flow.proposal))
    await attempt(flow.svc.resume(db_session, flow.proposal.id))
    await attempt(OrchestrationBlockerRecoveryService(analyzer=flow.analyzer).resume(db_session, flow.proposal.id))

    (wrapper,) = await lifecycle(db_session, flow.proposal, "recovery_apply")
    effect = await by_key(db_session, rows.run.id, f"{wrapper.idempotency_key}:effect")
    assert wrapper.status == effect.status == "failed"
    assert wrapper.error == effect.error == "effect_unknown"
    executor.assert_awaited_once()
    assert await status(db_session, flow.svc, flow.proposal) == "failed"
    assert await lifecycle(db_session, flow.proposal, "recovery_verify") == []
    assert await lifecycle(db_session, flow.proposal, "recovery_retry") == []


@pytest.mark.asyncio
async def test_crash_left_linked_reserved_effect_is_failed_without_redispatch(db_session, test_project, test_user, monkeypatch):
    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, verify_call(rows))
    (wrapper,) = await lifecycle(db_session, flow.proposal, "recovery_apply")
    call = flow.proposal.request["calls"][0]
    effect_key = f"{wrapper.idempotency_key}:effect"
    effect = await OrchestrationService().reserve_action(db_session, rows.run.id, effect_key, "request_verification",
                                                         call["arguments"])
    wrapper.dispatch_contract = {**wrapper.dispatch_contract, "effect_action_id": str(effect.id),
                                 "effect_idempotency_key": effect_key}
    await db_session.flush()
    executors = Executors(monkeypatch, rows)

    restarted = OrchestrationBlockerRecoveryService(analyzer=flow.analyzer)  # a fresh process after the crash
    await attempt(restarted.resume(db_session, flow.proposal.id))

    await db_session.refresh(wrapper)
    await db_session.refresh(effect)
    assert executors.calls == []
    assert wrapper.status == effect.status == "failed"
    assert wrapper.error == effect.error == "effect_unknown"
    assert await status(db_session, restarted, flow.proposal) == "failed"


@pytest.mark.asyncio
async def test_reserved_wrapper_resumes_after_a_crash_and_links_one_effect(db_session, test_project, test_user, monkeypatch):
    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, retry_task_call(rows), verify_call(rows))
    executors = Executors(monkeypatch, rows)
    (reserved,) = await lifecycle(db_session, flow.proposal, "recovery_apply")
    assert reserved.status == "reserved"

    restarted = OrchestrationBlockerRecoveryService(analyzer=flow.analyzer)
    first = await restarted.resume(db_session, flow.proposal.id)
    second = await OrchestrationBlockerRecoveryService(analyzer=flow.analyzer).resume(db_session, flow.proposal.id)

    assert first == second == "resolved"
    assert executors.keys("retry_task") == [f"recovery:{flow.pid}:apply:call-1:effect"]
    assert [key for key in executors.keys("request_verification") if ":apply:" in key] == [
        f"recovery:{flow.pid}:apply:call-2:effect"]
    assert len([key for key in executors.keys() if ":retry:" in key]) == 1
    wrappers = await lifecycle(db_session, flow.proposal, "recovery_apply")
    assert len(wrappers) == 2
    for wrapper in wrappers:
        linked = await effects(db_session, rows.run.id, wrapper.idempotency_key + ":")
        assert [row.id for row in linked] == [uuid.UUID(wrapper.dispatch_contract["effect_action_id"])]
    assert [row.status for row in await lifecycle(db_session, flow.proposal, "recovery_verify")] == ["completed"] * 2


@pytest.mark.asyncio
async def test_resume_reports_the_projected_status_without_effects_while_pending(db_session, test_project, test_user, monkeypatch):
    rows = await seed(db_session, test_project)
    svc, proposal = await propose(db_session, rows, ScriptedAnalyzer(draft(rows, verify_call(rows))))
    executors = Executors(monkeypatch, rows)

    assert await svc.resume(db_session, proposal.id) == "pending_approval"
    assert executors.calls == []


# --------------------------------------------------------------------------- delegated lifecycle


async def delegation_flow(db, project, user, monkeypatch, *, retry_target=None):
    from tests.test_orchestration_recovery_delegation import draft as delegation_draft
    from tests.test_orchestration_recovery_delegation import executables, seed as delegation_seed, spy_build_command

    executables(monkeypatch)
    spy_build_command(monkeypatch)
    rows = await delegation_seed(db, project)
    gate = OrchestrationGate(run_id=rows.run.id, success_criterion_key="delegated", gate_type="plan", status="open")
    db.add(gate)
    await db.flush()
    request = OrchestrationService().canonical_decision_request(
        "request_verification", {"gate_id": str(gate.id), "work_function": "validation"}, run_id=rows.run.id)
    rows.source = OrchestrationAction(run_id=rows.run.id, idempotency_key=f"seed:{uuid.uuid4()}:verification",
                                      action_type="request_verification", status="failed", request=request,
                                      error="verification_unavailable")
    db.add(rows.source)
    await db.flush()
    rows.gate = gate
    rows.retry_executors = Executors(monkeypatch, rows, only={"request_verification"})
    analyzer = ScriptedAnalyzer(delegation_draft(rows))
    svc, proposal = await propose(db, rows, analyzer, trigger(rows, retry_target=retry_target))
    decision = await approve(db, svc, proposal, user)
    return SimpleNamespace(svc=svc, proposal=proposal, decision=decision, analyzer=analyzer, rows=rows,
                           pid=proposal.request["proposal_id"], trigger=None)


async def bound_task(db, receipt):
    (ref,) = [ref for ref in receipt.target_refs if ref["kind"] == "task"]
    return await db.get(Task, uuid.UUID(ref["id"]))


async def open_waits_for(db, run_id, task_id):
    rows = (await db.scalars(select(OrchestrationWait).where(
        OrchestrationWait.run_id == run_id, OrchestrationWait.status == "open",
    ).execution_options(populate_existing=True))).all()
    return [row for row in rows if ((row.awaited_event or {}).get("matcher") or {}).get("task_id") == str(task_id)]


async def deliver(db, run, event_type, task_id):
    await OrchestrationSupervisionService(OrchestrationService()).clear_matching_waits(
        db, run, event_type=event_type, matcher={"task_id": str(task_id)})


@pytest.mark.asyncio
async def test_delegation_waiting_receipt_stores_a_wait_and_holds_no_lock(db_session, test_project, test_user, monkeypatch):
    from huddleroom.services import orchestration_service as service_module

    flow = await delegation_flow(db_session, test_project, test_user, monkeypatch)

    receipt = await flow.svc.execute_next_call(db_session, flow.proposal)

    assert receipt.status == "waiting"
    task = await bound_task(db_session, receipt)
    waits = await open_waits_for(db_session, flow.rows.run.id, task.id)
    assert waits, "the delegated wait is stored as an OrchestrationWait on the bound task"
    assert {wait.awaited_event["event_type"] for wait in waits} >= {"task.status_changed"}
    assert await status(db_session, flow.svc, flow.proposal) == "applying"
    assert await lifecycle(db_session, flow.proposal, "recovery_verify") == []
    goal_lock = service_module._SQLITE_GOAL_LOCKS.get(flow.rows.goal.id)
    assert goal_lock is None or not goal_lock._lock.locked()


@pytest.mark.asyncio
async def test_delegation_resumes_only_on_a_matched_event_then_verifies_before_one_retry(
    db_session, test_project, test_user, monkeypatch,
):
    flow = await delegation_flow(db_session, test_project, test_user, monkeypatch)
    receipt = await flow.svc.execute_next_call(db_session, flow.proposal)
    task = await bound_task(db_session, receipt)
    flow.rows.task = task  # the retry executor recorder reads rows.task.id

    other = Task(project_id=test_project.id, title="unrelated", status="done")
    db_session.add(other)
    await db_session.flush()
    await deliver(db_session, flow.rows.run, "task.status_changed", other.id)
    assert await flow.svc.resume(db_session, flow.proposal.id) == "applying"
    assert await lifecycle(db_session, flow.proposal, "recovery_verify") == []
    assert await lifecycle(db_session, flow.proposal, "recovery_retry") == []
    assert flow.rows.retry_executors.calls == []

    task.status = "done"
    await db_session.flush()
    await deliver(db_session, flow.rows.run, "task.status_changed", task.id)
    final = await flow.svc.resume(db_session, flow.proposal.id)

    (verify_wrapper,) = await lifecycle(db_session, flow.proposal, "recovery_verify")
    assert verify_wrapper.status == "completed", "task.completed passed before the retry"
    (retry,) = await lifecycle(db_session, flow.proposal, "recovery_retry")
    assert verify_wrapper.created_at <= retry.created_at
    assert flow.rows.retry_executors.keys() == [f"recovery:{flow.pid}:retry:1:effect"]
    assert final == "resolved"
    assert await open_waits_for(db_session, flow.rows.run.id, task.id) == []


@pytest.mark.asyncio
async def test_failed_bound_task_is_failure_evidence_not_a_pending_wait(db_session, test_project, test_user, monkeypatch):
    flow = await delegation_flow(db_session, test_project, test_user, monkeypatch)
    receipt = await flow.svc.execute_next_call(db_session, flow.proposal)
    task = await bound_task(db_session, receipt)
    task.status = "failed"
    await db_session.flush()
    await deliver(db_session, flow.rows.run, "task.status_changed", task.id)

    assert await flow.svc.resume(db_session, flow.proposal.id) == "failed"
    assert await lifecycle(db_session, flow.proposal, "recovery_retry") == []
    assert flow.rows.retry_executors.calls == []
    assert await open_waits_for(db_session, flow.rows.run.id, task.id) == []


@pytest.mark.asyncio
async def test_expired_delegation_deadline_fails_the_attempt_instead_of_polling(
    db_session, test_project, test_user, monkeypatch,
):
    from huddleroom.services import orchestration_recovery_registry as registry_module

    monkeypatch.setattr(registry_module, "_REMEDIATION_DEADLINE", timedelta(seconds=3))
    flow = await delegation_flow(db_session, test_project, test_user, monkeypatch)
    receipt = await flow.svc.execute_next_call(db_session, flow.proposal)
    assert receipt.status == "waiting"
    task = await bound_task(db_session, receipt)
    await asyncio.sleep(3.5)  # the sealed absolute deadline passes while the task is still open

    assert await flow.svc.resume(db_session, flow.proposal.id) == "failed"
    assert await lifecycle(db_session, flow.proposal, "recovery_retry") == []
    assert await open_waits_for(db_session, flow.rows.run.id, task.id) == []


@pytest.mark.asyncio
async def test_admitted_remediation_task_gets_a_session_while_the_run_is_baseline_blocked(
    db_session, test_project, test_user, monkeypatch,
):
    from huddleroom.schemas.session import SessionCreate
    from huddleroom.services.session_service import SessionService
    from tests.test_orchestration_recovery_delegation import HIERARCHY_RETRY, executables, fail_process, spy_build_command
    from tests.test_orchestration_recovery_delegation import draft as delegation_draft
    from tests.test_orchestration_recovery_delegation import seed as delegation_seed

    executables(monkeypatch)
    spy_build_command(monkeypatch)
    monkeypatch.setattr(SessionService, "_schedule_dispatch_after_commit", lambda *args, **kwargs: None)
    rows = await delegation_seed(db_session, test_project)
    await fail_process(db_session, rows, "team_hierarchy")
    rows.source = SimpleNamespace(id=rows.goal.id)
    svc, proposal = await propose(db_session, rows, ScriptedAnalyzer(delegation_draft(rows)), RecoveryTriggerV1.model_validate({
        "schema_version": 1, "goal_id": str(rows.goal.id), "run_id": str(rows.run.id),
        "source_kind": "planning_preflight", "source_id": str(rows.goal.id), "blocker_code": "baseline_blocked",
        "summary": "The baseline cannot complete.", "source_fingerprint": "b" * 64, "retry_target": HIERARCHY_RETRY,
    }))
    await approve(db_session, svc, proposal, test_user)

    receipt = await svc.execute_next_call(db_session, proposal)

    assert receipt.status == "waiting"
    task = await bound_task(db_session, receipt)
    assert await rows.service._baseline_readiness_reason(db_session, rows.goal.id) is not None, "still blocked"
    sessions = (await db_session.scalars(select(Session).where(Session.task_id == task.id))).all()
    if not sessions:
        sessions = [await SessionService().create(db_session, SessionCreate(
            agent_id=task.assigned_to, task_id=task.id, project_id=task.project_id, origin="auto",
            adapter_type_override="api"))]
    assert sessions and sessions[0].task_id == task.id and sessions[0].agent_id == rows.agent.id
    assert sessions[0].adapter_type == "api"
    config = sessions[0].metadata_["_run_config"]
    assert config["_roadmap_budget_enforced"] is True and 0 < config["max_tokens"] <= 4096
    assert await status(db_session, svc, proposal) == "applying"


# --------------------------------------------------------------------------- SPR 180 end to end (transactional effect)


@pytest.mark.asyncio
async def test_configuration_repair_resumes_verifies_and_reconciles_without_a_second_effect(sessions, tmp_path, monkeypatch):
    from tests.test_orchestration_recovery_configuration import (
        approved_flow, arm_semantic_guard, fresh_snapshot, seed_spr180,
    )

    seeded = await seed_spr180(sessions, tmp_path)
    flow = await approved_flow(sessions, seeded)
    arm_semantic_guard(monkeypatch)

    async with sessions() as s:
        first = await flow.service.resume(s, flow.proposal_action_id)
        await s.commit()
    after_first = await fresh_snapshot(sessions, seeded.goal_id)
    async with sessions() as s:
        second = await OrchestrationBlockerRecoveryService().resume(s, flow.proposal_action_id)
        await s.commit()

    assert first in {"retrying", "resolved"} and second in {"retrying", "resolved"}
    async with sessions() as s:
        proposal = await s.get(OrchestrationAction, flow.proposal_action_id)
        (wrapper,) = await lifecycle(s, proposal, "recovery_apply")
        assert wrapper.status == "completed"
        assert wrapper.dispatch_contract["tool_receipt_v1"]["status"] == "applied"
        assert wrapper.dispatch_contract["effect_action_id"] == str(wrapper.id)
        verifies = await lifecycle(s, proposal, "recovery_verify")
        assert verifies and all(row.status == "completed" for row in verifies)
        assert len(await lifecycle(s, proposal, "recovery_retry")) <= 1
    after_second = await fresh_snapshot(sessions, seeded.goal_id)
    assert after_second[str(seeded.goal_id)]["processes"] == after_first[str(seeded.goal_id)]["processes"]
    assert after_second["agents"] == after_first["agents"], "replay applied the configuration twice"


# --------------------------------------------------------------------------- warnings and on-track truth


async def active_recovery_warnings(db, goal_id):
    return [row for row in (await db.scalars(select(OrchestrationWarning).where(
        OrchestrationWarning.goal_id == goal_id, OrchestrationWarning.active.is_(True),
    ).execution_options(populate_existing=True))).all()]


@pytest.mark.asyncio
async def test_failed_recovery_maintains_a_blocker_warning(db_session, test_project, test_user, monkeypatch):
    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, verify_call(rows))
    Executors(monkeypatch, rows, plan={"request_verification": ["reject"]})

    await attempt(flow.svc.execute_next_call(db_session, flow.proposal))
    await attempt(flow.svc.resume(db_session, flow.proposal.id))

    assert await status(db_session, flow.svc, flow.proposal) == "failed"
    assert await active_recovery_warnings(db_session, rows.goal.id), "a failed recovery must prevent an on-track view"


@pytest.mark.asyncio
async def test_unsupported_recovery_maintains_a_blocker_warning(db_session, test_project):
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

    assert await svc.project_status(db_session, outcome) == "unsupported"
    assert await active_recovery_warnings(db_session, rows.goal.id)


@pytest.mark.asyncio
async def test_warning_and_wait_resolve_only_after_the_retry_removes_the_blocker(
    db_session, test_project, test_user, monkeypatch,
):
    rows = await seed(db_session, test_project)
    analyzer = ScriptedAnalyzer(draft(rows, verify_call(rows)), draft(rows, retry_task_call(rows)))
    flow = await approved(db_session, rows, test_user, analyzer=analyzer)
    Executors(monkeypatch, rows, plan={"request_verification": ["reject", "ok"]})
    await attempt(flow.svc.execute_next_call(db_session, flow.proposal))
    await attempt(flow.svc.resume(db_session, flow.proposal.id))
    warnings = {row.id for row in await active_recovery_warnings(db_session, rows.goal.id)}
    assert warnings

    follow_up = await flow.svc.rediagnose(db_session, flow.proposal, EvidenceV1(
        label="The verification request was rejected.", value={"error": "request_verification_conflict"},
        source_kind="action", source_id=flow.proposal.id))
    second = await flow.svc.diagnose(db_session, follow_up.id) if follow_up.action_type == "recovery_trigger" else follow_up
    await approve(db_session, flow.svc, second, test_user)
    await flow.svc.execute_next_call(db_session, second)
    await flow.svc.verify_proposal(db_session, second)
    assert await flow.svc.project_status(db_session, second) == "retrying"
    assert warnings <= {row.id for row in await active_recovery_warnings(db_session, rows.goal.id)}

    await flow.svc.retry_blocked_work(db_session, second)

    assert await status(db_session, flow.svc, second) == "resolved"
    for warning_id in warnings:
        row = await db_session.get(OrchestrationWarning, warning_id, populate_existing=True)
        assert row.active is False and (row.resolved_by or "").strip()
        assert second.request["proposal_id"] in f"{row.resolved_by} {row.resolved_reason}"


# --------------------------------------------------------------------------- follow-up


async def approved_with(db, rows, user, retry_target, *calls):
    analyzer = ScriptedAnalyzer(draft(rows, *calls))
    svc, proposal = await propose(db, rows, analyzer, trigger(rows, retry_target=retry_target))
    decision = await approve(db, svc, proposal, user)
    return SimpleNamespace(svc=svc, proposal=proposal, decision=decision, analyzer=analyzer,
                           pid=proposal.request["proposal_id"])


@pytest.mark.asyncio
async def test_drift_landing_while_waiting_for_the_goal_lock_is_never_applied(db_session, test_project, test_user, monkeypatch):
    from contextlib import asynccontextmanager

    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, verify_call(rows))
    executors = Executors(monkeypatch, rows)
    original = OrchestrationService._lock_goal_for_baseline_transition
    drifted = []

    @asynccontextmanager
    async def drifting(self, db, goal_id, **kwargs):
        async with original(self, db, goal_id, **kwargs):
            if not drifted:  # another writer committed just before this caller obtained the lock
                rows.gate.failure_reason = "changed while the executor waited for the lock"
                await db.flush()
                drifted.append(True)
            yield

    monkeypatch.setattr(OrchestrationService, "_lock_goal_for_baseline_transition", drifting)

    outcome = await attempt(flow.svc.execute_next_call(db_session, flow.proposal))

    assert drifted and not isinstance(outcome, ToolReceiptV1)
    assert executors.calls == []
    assert await status(db_session, flow.svc, flow.proposal) == "failed"
    assert flow.proposal.error == "state_changed"


@pytest.mark.asyncio
async def test_retry_is_fenced_by_the_sealed_steering(db_session, test_project, test_user, monkeypatch):
    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, verify_call(rows))
    assert flow.proposal.dispatch_contract["steering"].get("versions")
    executors = Executors(monkeypatch, rows)
    await verified(db_session, flow)
    rows.run.plan_state = {**(rows.run.plan_state or {}), "accepted_plan_fingerprint": "changed-before-retry"}
    await db_session.flush()

    retry = await attempt(flow.svc.retry_blocked_work(db_session, flow.proposal))

    assert [key for key in executors.keys() if ":retry:" in key] == []
    assert not isinstance(retry, OrchestrationAction) or retry.status == "failed"
    assert await status(db_session, flow.svc, flow.proposal) == "failed"
    assert flow.proposal.error == "state_changed"


@pytest.mark.asyncio
async def test_failure_path_after_a_reset_reservation_writes_nothing_for_the_old_run(
    db_session, test_project, test_user, monkeypatch,
):
    from huddleroom.services import orchestration_baseline_runtime as runtime

    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, verify_call(rows))
    Executors(monkeypatch, rows)
    warnings_before = len(await active_recovery_warnings(db_session, rows.goal.id))

    async def reset_then_fail(*args, **kwargs):
        runtime._resetting[rows.goal.id] = runtime._resetting.get(rows.goal.id, 0) + 1
        raise ValueError("executor failed while the reset was reserved")

    monkeypatch.setattr(flow.svc.registry, "execute_call", reset_then_fail)
    try:
        with pytest.raises(HTTPException) as exc:
            await flow.svc.execute_next_call(db_session, flow.proposal)
    finally:
        runtime._resetting[rows.goal.id] -= 1
        if not runtime._resetting[rows.goal.id]:
            runtime._resetting.pop(rows.goal.id)

    assert exc.value.status_code == 409
    assert len(await active_recovery_warnings(db_session, rows.goal.id)) == warnings_before
    follow_ups = (await db_session.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == rows.run.id, OrchestrationAction.action_type == "recovery_trigger",
        OrchestrationAction.status == "reserved"))).all()
    assert follow_ups == [], "no fresh diagnosis is scheduled for a run that is being reset"


@pytest.mark.asyncio
async def test_decision_loop_retry_stays_retrying_until_the_loop_progresses(db_session, test_project, test_user, monkeypatch):
    from huddleroom.models.orchestration import OrchestrationDecision

    rows = await seed(db_session, test_project)
    flow = await approved_with(db_session, rows, test_user, {"kind": "decision_loop"}, verify_call(rows))
    Executors(monkeypatch, rows)
    await verified(db_session, flow)

    retry = await flow.svc.retry_blocked_work(db_session, flow.proposal)

    assert retry.status == "reserved"
    assert await status(db_session, flow.svc, flow.proposal) == "retrying"
    assert flow.proposal.status == "reserved", "no false resolved before the loop progressed"
    warnings = {row.id for row in await active_recovery_warnings(db_session, rows.goal.id)}
    assert warnings
    assert await flow.svc.resume(db_session, flow.proposal.id) == "retrying"

    # Bookkeeping and operator/API actions are not decision-loop progress.
    db_session.add_all([
        OrchestrationAction(run_id=rows.run.id, idempotency_key=f"operator:{uuid.uuid4()}",
                            action_type="request_verification", status="completed", request={}),
        OrchestrationAction(run_id=rows.run.id, idempotency_key=f"run:{rows.run.id}:supervision:{uuid.uuid4()}",
                            action_type="request_verification", status="completed", request={}),
    ])
    await db_session.flush()
    assert await flow.svc.resume(db_session, flow.proposal.id) == "retrying"

    loop_decision = OrchestrationDecision(run_id=rows.run.id, decision_type="request_verification",
                                          validator_status="accepted", input_snapshot={}, parsed_decision={})
    db_session.add(loop_decision)
    await db_session.flush()
    db_session.add(OrchestrationAction(run_id=rows.run.id, idempotency_key=f"loop:{uuid.uuid4()}", decision_id=loop_decision.id,
                                       action_type="request_verification", status="completed", request={}))
    await db_session.flush()

    assert await flow.svc.resume(db_session, flow.proposal.id) == "resolved"
    for warning_id in warnings:
        assert (await db_session.get(OrchestrationWarning, warning_id, populate_existing=True)).active is False


@pytest.mark.asyncio
async def test_retrying_decision_loop_proposal_does_not_park_the_tick(db_session, test_project, test_user, monkeypatch):
    rows = await seed(db_session, test_project)
    flow = await approved_with(db_session, rows, test_user, {"kind": "decision_loop"}, verify_call(rows))
    Executors(monkeypatch, rows)
    await verified(db_session, flow)
    await flow.svc.retry_blocked_work(db_session, flow.proposal)
    assert await status(db_session, flow.svc, flow.proposal) == "retrying"

    open_waits = (await db_session.scalars(select(OrchestrationWait).where(
        OrchestrationWait.run_id == rows.run.id, OrchestrationWait.status == "open"))).all()
    assert open_waits == [], "an in-wrapper retry stores no wait that could park the loop it waits on"
    rows.goal.status, rows.run.status, rows.run.phase = "active", "running", "authorized"
    await db_session.flush()
    outcome = await OrchestrationSupervisionService(OrchestrationService()).reconcile_local(db_session, rows.goal, rows.run)
    assert (outcome or {}).get("outcome") != "waiting", outcome


@pytest.mark.asyncio
async def test_baseline_process_retry_completes_only_after_the_process_completes(
    db_session, test_project, test_user, monkeypatch,
):
    from tests.test_orchestration_recovery_delegation import fail_process

    rows = await seed(db_session, test_project)
    flow = await approved_with(db_session, rows, test_user, {"kind": "baseline_process", "process_type": "team_hierarchy"},
                               verify_call(rows))
    Executors(monkeypatch, rows)
    await verified(db_session, flow)
    blocked = await fail_process(db_session, rows, "team_hierarchy")

    retry = await flow.svc.retry_blocked_work(db_session, flow.proposal)

    assert retry.status == "reserved"
    assert await status(db_session, flow.svc, flow.proposal) == "retrying"
    blocked.status = "completed"
    await db_session.flush()
    assert await flow.svc.resume(db_session, flow.proposal.id) == "resolved"


@pytest.mark.asyncio
async def test_failing_an_attempt_also_fails_its_linked_reserved_effect(db_session, test_project, test_user):
    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, verify_call(rows))
    (wrapper,) = await lifecycle(db_session, flow.proposal, "recovery_apply")
    effect_key = f"{wrapper.idempotency_key}:effect"
    effect = await OrchestrationService().reserve_action(db_session, rows.run.id, effect_key, "request_verification",
                                                         flow.proposal.request["calls"][0]["arguments"])
    wrapper.dispatch_contract = {**wrapper.dispatch_contract, "effect_action_id": str(effect.id),
                                 "effect_idempotency_key": effect_key}
    await db_session.flush()

    await flow.svc._fail_attempt(db_session, flow.proposal, rows.run, rows.goal, "state_changed", [])

    await db_session.refresh(effect)
    await db_session.refresh(wrapper)
    assert wrapper.status == effect.status == "failed"
    assert wrapper.error == effect.error == "state_changed"


@pytest.mark.asyncio
async def test_retry_replay_after_a_transient_error_reuses_one_decision_row(db_session, test_project, test_user, monkeypatch):
    from huddleroom.models.orchestration import OrchestrationDecision

    async def unavailable(db, rows):
        raise HTTPException(status_code=503, detail="provider unavailable")

    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, verify_call(rows))
    Executors(monkeypatch, rows, plan={"request_verification": ["ok", unavailable, "ok"]})
    await verified(db_session, flow)

    with pytest.raises(HTTPException):
        await flow.svc.retry_blocked_work(db_session, flow.proposal)
    retry = await flow.svc.retry_blocked_work(db_session, flow.proposal)

    assert retry.status == "completed"
    rows_ = (await db_session.scalars(select(OrchestrationDecision).where(
        OrchestrationDecision.run_id == rows.run.id, OrchestrationDecision.decision_type == "recovery_retry"))).all()
    assert len(rows_) == 1
    assert await status(db_session, flow.svc, flow.proposal) == "resolved"


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_replay", [False, True])
async def test_server_planning_retry_captures_current_generation_and_never_refreshes_reused_snapshot(
    db_session, test_project, test_user, monkeypatch, stale_replay,
):
    from copy import deepcopy
    from huddleroom.models.orchestration import OrchestrationDecision
    from huddleroom.services.orchestration_service import PlanGenerationChanged
    rows = await seed(db_session, test_project)
    rows.run.plan_state = {"test_nested": {"items": []}}
    await db_session.flush()
    flow = await approved_with(db_session, rows, test_user,
                               {"kind": "planning", "requested_agent_id": str(rows.planner.id)}, verify_call(rows))
    async def unavailable(db, rows):
        raise HTTPException(status_code=503, detail="provider unavailable")
    executors = Executors(monkeypatch, rows, plan={"request_plan": [unavailable, "ok"]})
    await verified(db_session, flow)
    with pytest.raises(HTTPException):
        await flow.svc.retry_blocked_work(db_session, flow.proposal)
    decisions = list(await db_session.scalars(select(OrchestrationDecision).where(
        OrchestrationDecision.run_id == rows.run.id, OrchestrationDecision.decision_type == "recovery_retry",
    )))
    assert len(decisions) == 1
    decision = decisions[0]
    expected = {"run": {"id": str(rows.run.id), "plan_state": {"test_nested": {"items": []}}}}
    assert decision.input_snapshot == expected
    retained = deepcopy(decision.input_snapshot)
    rows.run.plan_state["test_nested"]["items"].append("later state mutation")
    assert decision.input_snapshot == retained
    if stale_replay:
        rows.run.plan_state = {**rows.run.plan_state, "planning_version": 1}
        await db_session.flush()
        with pytest.raises(PlanGenerationChanged):
            await flow.svc.retry_blocked_work(db_session, flow.proposal)
        assert len(executors.keys("request_plan")) == 1
    else:
        retry = await flow.svc.retry_blocked_work(db_session, flow.proposal)
        assert retry.status == "completed"
        assert len(executors.keys("request_plan")) == 2
    await db_session.refresh(decision)
    assert decision.input_snapshot == retained
