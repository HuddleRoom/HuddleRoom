import pytest

from huddleroom.services.orchestration_condition import (
    CONDITIONS,
    ConditionError,
    is_reconcilable,
    validate_condition,
)


@pytest.mark.parametrize("condition", sorted(CONDITIONS))
def test_valid_condition_is_normalized(condition):
    assert validate_condition({"condition": condition, "subject": " t1 "}) == {
        "condition": condition,
        "subject": "t1",
    }


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "session_running",
        {"condition": "whenever", "subject": "t1"},
        {"condition": "session_running"},
        {"condition": "plan_missing", "subject": "  "},
        {"condition": "gate_unverified", "subject": 3},
    ],
)
def test_invalid_condition_rejected(raw):
    with pytest.raises(ConditionError):
        validate_condition(raw)
    assert not is_reconcilable(raw)


@pytest.mark.parametrize("kind", ["budget", "permission", "provider", "Owner_Decision"])
def test_owner_decisions_never_reconcilable(kind):
    raw = {"condition": "session_running", "subject": "t1", "kind": kind}
    assert not is_reconcilable(raw)


@pytest.mark.parametrize("subject", ["budget", " Permission ", "provider"])
def test_owner_decision_subject_rejected(subject):
    with pytest.raises(ConditionError):
        validate_condition({"condition": "session_running", "subject": subject})


@pytest.mark.parametrize("kw", [{"action_kind": "budget"}, {"category": "permission_required"}])
def test_reconcile_time_action_context_blocks_owner_decisions(kw):
    stored = validate_condition({"condition": "valid_result_missing", "subject": "t1"})
    assert is_reconcilable(stored)
    assert not is_reconcilable(stored, **kw)
    assert is_reconcilable(stored, action_kind="record_warning", category="other")


# --- Reconciliation through the real record/ask actions, ingestion and ticks ---

import json  # noqa: E402
import uuid  # noqa: E402
from datetime import timedelta  # noqa: E402

from sqlalchemy import select  # noqa: E402

from huddleroom.models.base import _utcnow  # noqa: E402
from huddleroom.models.event_log import EventLog  # noqa: E402
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGate, OrchestrationWait  # noqa: E402
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision, OrchestrationWarning  # noqa: E402
from huddleroom.models.session import Session  # noqa: E402
from huddleroom.services.event_bus import emit_event_once  # noqa: E402
from huddleroom.services.orchestration_decision_validator import validate_orchestration_decision  # noqa: E402
from huddleroom.services.session_sync import sync_task_from_session  # noqa: E402
from tests.test_orchestration_final_summary_request import runnable_workspace  # noqa: E402,F401  (autouse)
from tests.test_orchestration_plan_ingestion import _db_output, _planning_run  # noqa: E402
from tests.test_orchestration_result_correction import (  # noqa: E402
    MISSING_FIELDS, VALID, _complete, _corrections, _setup,
)


@pytest.fixture(autouse=True)
def no_live_decisions(stub_decision):
    stub_decision(lambda _context: {"action_type": "noop"})


async def _warn(service, db, run, message, condition=None, warning_type="model_attention"):
    request = {"action_type": "record_warning", "warning_type": warning_type, "severity": "warning", "message": message}
    if condition is not None:
        request["condition"] = condition
    action = await service.execute_record_warning_action(db, run.id, request, f"run:{run.id}:test-warning:{uuid.uuid4()}")
    return await db.get(OrchestrationWarning, action.target_id)


async def _ask(service, db, run, question, condition=None, reason=None):
    request = {"action_type": "ask_human", "question": question, "subject": question}
    if condition is not None:
        request["condition"] = condition
    if reason is not None:
        request["reason"] = reason
    action = await service.execute_ask_human_action(db, run.id, request, f"run:{run.id}:test-ask:{uuid.uuid4()}")
    return await db.get(OrchestrationAuthorityDecision, action.target_id)


def test_decision_validator_accepts_only_declared_conditions():
    base = {"action_type": "record_warning", "warning_type": "x", "severity": "warning", "message": "m"}
    assert validate_orchestration_decision({**base, "condition": {"condition": "plan_missing", "subject": "r"}}).accepted
    assert not validate_orchestration_decision({**base, "condition": {"condition": "soon", "subject": "r"}}).accepted
    ask = {"action_type": "ask_human", "question": "q"}
    assert validate_orchestration_decision({**ask, "condition": {"condition": "gate_unverified", "subject": "g"}}).accepted
    assert not validate_orchestration_decision({**ask, "condition": {"condition": "gate_unverified", "subject": "budget"}}).accepted


async def test_late_plan_clears_its_warning_and_question_but_never_owner_or_undeclared_items(db_session, test_project):
    from huddleroom.services.orchestration_plan_ingestion import ingest

    service, goal, run, task, session = await _planning_run(db_session, test_project.id)
    version = run.plan_state["planning_version"]
    plan_missing = {"condition": "plan_missing", "subject": f"{run.id}:{version}"}
    warning = await _warn(service, db_session, run, "No plan yet.", plan_missing, "plan_missing")
    question = await _ask(service, db_session, run, "Should planning continue?", plan_missing)
    owner_budget = await _ask(service, db_session, run, "Raise the budget?", plan_missing, reason="budget_exhausted")
    escalation = await service._escalate_owner(
        db_session, goal, run, "budget_exhausted", "plan", ["spent"], "Raise the budget for planning?",
    )
    escalated = await db_session.get(OrchestrationAuthorityDecision, escalation.target_id)
    undeclared = await _warn(service, db_session, run, "The plan looks slow.")

    await service._reconcile_conditions(db_session, run)
    assert warning.active and question.status == "pending"

    artifact = await ingest(db_session, goal, run, task, session.id, version, _db_output())
    await service.execute_accept_plan_action(
        db_session, run_id=run.id, request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
        idempotency_key=f"run:{run.id}:kind:accept_plan",
    )
    await service._reconcile_conditions(db_session, run)
    await service._reconcile_conditions(db_session, run)  # idempotent

    for row in (warning, question, owner_budget, escalated, undeclared):
        await db_session.refresh(row)
    assert not warning.active and warning.resolved_by == "system"
    assert question.status == "cancelled"
    assert owner_budget.status == "pending" and escalated.status == "pending"
    assert undeclared.active


async def test_rejected_output_keeps_result_missing_until_the_corrected_result_is_accepted(db_session, test_project):
    service, goal, run, agent, task = await _setup(db_session, test_project)
    task.status = "in_progress"
    running = Session(
        project_id=task.project_id, task_id=task.id, agent_id=agent.id, adapter_type="api", status="running",
        input_context={}, origin="orchestrator",
        metadata_={"orchestration": {"action_id": task.metadata_["orchestration"]["action_id"]}},
    )
    db_session.add(running)
    await db_session.flush()
    supervision = service.supervision
    await supervision.reconcile_local(db_session, goal, run)
    wait = await db_session.scalar(select(OrchestrationWait).where(
        OrchestrationWait.run_id == run.id, OrchestrationWait.wait_key.like(f"%session:{running.id}%"),
    ))
    wait.due_recheck_at = _utcnow() - timedelta(seconds=1)
    fired = await supervision.reconcile_local(db_session, goal, run)
    assert fired["outcome"] == "due_fallback"
    fallback = await db_session.get(OrchestrationAction, uuid.UUID(fired["action_id"]))
    assert fallback.request["condition"] == {"condition": "session_running", "subject": str(task.id)}
    session_warning = await db_session.get(OrchestrationWarning, fallback.target_id)
    result_warning = await _warn(
        service, db_session, run, "No valid result yet.", {"condition": "valid_result_missing", "subject": str(task.id)},
    )

    # The session ends with output that fails validation: the session condition holds, the result one does not.
    running.status, running.output, running.ended_at = "completed", MISSING_FIELDS, _utcnow()
    await db_session.flush()
    await sync_task_from_session(db_session, running)
    key = f"condition-test:{running.id}"
    await emit_event_once(db_session, task.project_id, "session.completed", {
        "session_id": str(running.id), "task_id": str(task.id),
    }, dedup_key=key)
    await service._ingest_evidence_from_event(
        db_session, run, await db_session.scalar(select(EventLog).where(EventLog.dedup_key == key)),
    )
    await service.tick(db_session, run.id)
    await db_session.refresh(session_warning)
    await db_session.refresh(result_warning)
    assert not session_warning.active and session_warning.resolved_by == "system"
    assert result_warning.active

    [correction] = await _corrections(db_session, task)
    await _complete(db_session, service, run, correction, json.dumps(VALID))
    await service.tick(db_session, run.id)
    await db_session.refresh(result_warning)
    assert not result_warning.active and result_warning.resolved_by == "system"


async def test_gate_condition_clears_only_on_acceptance_and_shared_undeclared_item_stays(db_session, test_project):
    service, goal, run, _agent, _task = await _setup(db_session, test_project)
    gate = OrchestrationGate(run_id=run.id, success_criterion_key="done", gate_type="work_completed")
    db_session.add(gate)
    await db_session.flush()
    unverified = {"condition": "gate_unverified", "subject": str(gate.id)}
    cleared = await _warn(service, db_session, run, "Gate not verified.", unverified, "gate_unverified")
    shared = await _warn(service, db_session, run, "Gate still open.", unverified)
    # Warnings of one type dedupe into one row: an undeclared origin keeps the shared row open.
    assert (await _warn(service, db_session, run, "Gate still open, says the model.")).id == shared.id

    gate.status = "failed"
    await service._reconcile_conditions(db_session, run)
    assert cleared.active

    gate.status = "accepted"
    await service._reconcile_conditions(db_session, run)
    await db_session.refresh(cleared)
    await db_session.refresh(shared)
    assert not cleared.active and shared.active
