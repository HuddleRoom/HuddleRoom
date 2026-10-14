"""Bounded result correction: one output-only correction task per failed result, driven through real
delegation creation, session completion and evidence ingestion."""
import json
import uuid

import pytest
from sqlalchemy import select

from huddleroom.models.agent import Agent
from huddleroom.models.base import _utcnow
from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGate, OrchestrationRun
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.event_bus import emit_event_once
from huddleroom.services.orchestration_result_contract import (
    CORRECTION_INSTRUCTION,
    build_correction_brief,
    failure_fingerprint,
    next_step,
)
from huddleroom.services.orchestration_work_report import validate_work_report
from huddleroom.services.session_sync import sync_task_from_session
from huddleroom.services.tool_executor import TOOL_EXECUTED_NO_CONTENT
from tests.test_orchestration_final_summary_request import runnable_workspace  # noqa: F401  (autouse)
from tests.test_orchestration_runtime_e2e import _authorized_run


VALID = {
    "status": "completed", "changes": ["Audited the API."], "evidence": ["artifact:audit-notes"],
    "criterion_progress": {"done": "candidate"}, "decisions": [], "risks": [], "open_questions": [],
    "next_step": "Request verification.", "collaboration_need": None,
}
MISSING_FIELDS = json.dumps({"status": "completed", "changes": ["Audited the API."]})


@pytest.fixture(autouse=True)
def no_live_decisions(stub_decision):
    stub_decision(lambda _context: {"action_type": "noop"})


async def _setup(db, project, *, caps=None):
    agent = Agent(name=f"auditor-{uuid.uuid4()}", role="developer", provider="openai", model="gpt-4o-mini",
                  adapter_type="api", capabilities=["investigation"], config={}, is_active=True)
    db.add(agent)
    await db.flush()
    service, goal, run = await _authorized_run(db, project)
    if caps:
        goal.budget = {"caps": caps}
    request = {
        "action_type": "create_delegation_task", "agent_id": str(agent.id), "work_function": "investigation",
        "scope": "Audit the API surface.", "deliverable": "Audit findings.",
    }
    if caps:
        request["budget"] = {"max_tokens": 1}
    action = await service.execute_create_delegation_task_action(
        db, run.id, request, f"run:{run.id}:kind:create_delegation_task:audit", ad_hoc=True,
    )
    return service, goal, run, agent, await db.get(Task, action.target_id)


async def _complete(db, service, run, task, output, *, tokens=0):
    """Finish one attempt the way an adapter does, then ingest its completion event."""
    task.status = "in_progress"
    session = Session(
        project_id=task.project_id, task_id=task.id, agent_id=task.assigned_to, adapter_type="api",
        status="completed", input_context={}, output=output, origin="orchestrator", ended_at=_utcnow(),
        metadata_={"token_count_in": tokens, "token_count_out": 0,
                   "orchestration": {"action_id": task.metadata_["orchestration"]["action_id"]}},
    )
    db.add(session)
    await db.flush()
    await sync_task_from_session(db, session)
    key = f"result-correction-test:{session.id}"
    await emit_event_once(db, task.project_id, "session.completed", {
        "session_id": str(session.id), "task_id": str(task.id),
    }, dedup_key=key)
    event = await db.scalar(select(EventLog).where(EventLog.dedup_key == key))
    await service._ingest_evidence_from_event(db, run, event)
    return session


async def _ledger(service, db, run, task):
    marker = await service._existing_action_for_key(db, run.id, f"run:{run.id}:kind:report_consumed:task:{task.id}")
    return marker.request.get("result_correction", {})


async def _corrections(db, task):
    rows = await db.scalars(select(Task).where(Task.parent_id == task.id))
    return [row for row in rows if (row.metadata_ or {}).get("result_correction_parent_task_id") == str(task.id)]


def _escalations(run, category):
    return [b for b in run.active_blockers if b.get("kind") == "owner_escalation" and b.get("category") == category]


def test_next_step_allows_one_correction_and_a_second_only_on_progress():
    first = {"fingerprint": "a", "errors": ["x", "y"]}
    assert next_step([], "a", 2) == "correct"
    assert next_step([first], "b", 1) == "correct"
    assert next_step([first], "b", 2) == "stop"  # no fewer errors
    assert next_step([first], "a", 1) == "stop"  # repeated failure
    assert next_step([first, {"fingerprint": "b", "errors": ["x"]}], "c", 0) == "stop"
    assert failure_fingerprint("work_report", "{}", ["b", "a"]) == failure_fingerprint("work_report", "{}", ["a", "b"])
    assert failure_fingerprint("work_report", "{}", ["a"]) != failure_fingerprint("final_summary", "{}", ["a"])


def test_brief_carries_instruction_schema_errors_version_and_whole_output():
    output = "x" * 40000 + "TAIL"
    brief = build_correction_brief("Do the audit.", output, {"status": "str"}, ["missing key: evidence"], {"v": 1})
    assert CORRECTION_INSTRUCTION in brief and "Do the audit." in brief and '"status": "str"' in brief
    assert "- missing key: evidence" in brief and '{"v": 1}' in brief and output in brief


async def test_uncapped_default_delegation_report_is_corrected_and_consumed(db_session, test_project):
    service, _goal, run, agent, task = await _setup(db_session, test_project)
    original = await _complete(db_session, service, run, task, MISSING_FIELDS)

    [correction] = await _corrections(db_session, task)
    errors = list(validate_work_report(MISSING_FIELDS).errors)
    contract = correction.metadata_["orchestration_contract"]
    assert correction.assigned_to == agent.id and contract["work_function"] == "report_clarification"
    assert contract["expected_result"] == "work_report" and contract["parent_task_id"] == str(task.id)
    assert MISSING_FIELDS in correction.description and CORRECTION_INSTRUCTION in correction.description
    assert all(error in correction.description for error in errors) and task.description in correction.description
    ledger = await _ledger(service, db_session, run, task)
    assert ledger["kind"] == "work_report" and ledger["status"] == "correcting"
    assert ledger["correction_task_id"] == str(correction.id)
    assert [(a["task_id"], a["session_id"], a["errors"]) for a in ledger["attempts"]] == [
        (str(task.id), str(original.id), errors)
    ]

    fixed = await _complete(db_session, service, run, correction, json.dumps(VALID))

    ledger = await _ledger(service, db_session, run, task)
    assert ledger["status"] == "corrected" and ledger["correction_session_id"] == str(fixed.id)
    corrected = await service._existing_action_for_key(
        db_session, run.id, f"run:{run.id}:kind:report_consumed:task:{correction.id}",
    )
    assert corrected.request["report"]["evidence"] == ["artifact:audit-notes"]
    assert len(await _corrections(db_session, task)) == 1


async def test_capped_goal_funds_correction_with_protected_allocation(db_session, test_project):
    service, _goal, run, _agent, task = await _setup(db_session, test_project, caps={"max_tokens": 10})
    await _complete(db_session, service, run, task, MISSING_FIELDS, tokens=2)

    [correction] = await _corrections(db_session, task)
    assert correction.metadata_["orchestration_contract"]["budget"] == {"max_tokens": "1"}
    assert not _escalations(run, "budget_exhausted")


async def test_exhausted_budget_escalates_and_dispatches_nothing(db_session, test_project):
    service, _goal, run, _agent, task = await _setup(db_session, test_project, caps={"max_tokens": 2})
    await _complete(db_session, service, run, task, MISSING_FIELDS, tokens=2)

    assert await _corrections(db_session, task) == []
    [blocker] = _escalations(run, "budget_exhausted")
    assert blocker["subject"] == f"task:{task.id}:result_correction"
    assert blocker["reason"] == f"budget_exhausted: task:{task.id}:result_correction"
    assert (await _ledger(service, db_session, run, task))["status"] == "pending"


async def test_identical_failure_stops_with_both_attempts_and_ticks_do_not_rearm(db_session, test_project):
    service, _goal, run, _agent, task = await _setup(db_session, test_project)
    first = await _complete(db_session, service, run, task, MISSING_FIELDS)
    [correction] = await _corrections(db_session, task)
    second = await _complete(db_session, service, run, correction, MISSING_FIELDS)

    ledger = await _ledger(service, db_session, run, task)
    assert ledger["status"] == "exhausted"
    assert [a["session_id"] for a in ledger["attempts"]] == [str(first.id), str(second.id)]
    [blocker] = _escalations(run, "result_correction_exhausted")
    assert len(blocker["history"]) == 2 and str(first.id) in blocker["history"][0]
    assert str(second.id) in blocker["history"][1]
    asks = lambda: db_session.scalars(select(OrchestrationAction).where(  # noqa: E731
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "ask_human"))

    before = len(list(await asks()))
    for _ in range(2):
        await service.tick(db_session, run.id)
        await service.consume_canonical_report(db_session, run, task)
    assert len(await _corrections(db_session, task)) == 1
    assert len(list(await asks())) == before
    assert (await _ledger(service, db_session, run, task))["attempts"] == ledger["attempts"]


async def test_second_correction_only_when_failure_changes_with_fewer_errors(db_session, test_project):
    service, _goal, run, _agent, task = await _setup(db_session, test_project)
    three_errors = json.dumps({**VALID, "status": 1, "next_step": 2, "risks": [3]})
    one_error = json.dumps({**VALID, "status": 1})
    assert len(validate_work_report(three_errors).errors) == 3 and len(validate_work_report(one_error).errors) == 1
    await _complete(db_session, service, run, task, three_errors)
    [first] = await _corrections(db_session, task)
    await _complete(db_session, service, run, first, one_error)

    corrections = await _corrections(db_session, task)
    assert len(corrections) == 2
    second = next(item for item in corrections if item.id != first.id)
    assert one_error in second.description
    assert (await _ledger(service, db_session, run, task))["correction_task_id"] == str(second.id)

    # The superseded first correction can no longer steer the ledger.
    await service._request_result_correction(db_session, run, first, None, "work_report", ["late"])
    assert len((await _ledger(service, db_session, run, task))["attempts"]) == 2

    await _complete(db_session, service, run, second, json.dumps({**VALID, "next_step": 1}))
    ledger = await _ledger(service, db_session, run, task)
    assert ledger["status"] == "exhausted" and len(ledger["attempts"]) == 3
    assert len(await _corrections(db_session, task)) == 2


async def test_large_failed_output_is_carried_whole_with_its_tail_error(db_session, test_project):
    service, _goal, run, _agent, task = await _setup(db_session, test_project)
    output = json.dumps({**VALID, "changes": ["a" * 32000], "unexpected": True})
    assert len(output) > 30000 and validate_work_report(output).errors
    await _complete(db_session, service, run, task, output)

    [correction] = await _corrections(db_session, task)
    assert output in correction.description
    assert correction.description.rstrip().endswith(output + "\nFAILED_OUTPUT>>>")


@pytest.mark.parametrize("stored", ["", TOOL_EXECUTED_NO_CONTENT])
async def test_unavailable_failed_output_blocks_subject_without_dispatch(db_session, test_project, stored):
    service, _goal, run, _agent, task = await _setup(db_session, test_project)
    await _complete(db_session, service, run, task, stored)

    assert await _corrections(db_session, task) == []
    ledger = await _ledger(service, db_session, run, task)
    assert ledger["status"] == "unavailable" and ledger["reason"] == "correction_input_unavailable"
    [blocker] = _escalations(run, "capability_unavailable")
    assert blocker["subject"] == f"task:{task.id}:correction_input_unavailable"


async def test_paused_run_dispatches_no_correction(db_session, test_project):
    service, _goal, run, _agent, task = await _setup(db_session, test_project)
    run.status = "paused"
    await _complete(db_session, service, run, task, MISSING_FIELDS)

    assert await _corrections(db_session, task) == []
    assert (await _ledger(service, db_session, run, task))["stale"] == "run_paused"


@pytest.mark.parametrize("change", ["paused", "superseded"])
async def test_pending_correction_is_not_consumed_after_run_pause_or_supersede(db_session, test_project, change):
    service, goal, run, _agent, task = await _setup(db_session, test_project)
    await _complete(db_session, service, run, task, MISSING_FIELDS)
    [correction] = await _corrections(db_session, task)
    if change == "paused":
        run.status = "paused"
    else:
        run.status = "cancelled"  # one active run per goal: a successor replaces a closed run
        await db_session.flush()
        db_session.add(OrchestrationRun(goal_id=goal.id, status="running", supersedes_run_id=run.id))
    await db_session.flush()

    await _complete(db_session, service, run, correction, json.dumps(VALID))

    ledger = await _ledger(service, db_session, run, task)
    assert ledger.get("stale") == {"paused": "run_paused", "superseded": None}[change]
    assert ledger["status"] == "correcting" and ledger["correction_session_id"] is None
    assert await service._existing_action_for_key(
        db_session, run.id, f"run:{run.id}:kind:report_consumed:task:{correction.id}",
    ) is None


async def test_correction_finished_while_paused_is_consumed_after_resume(db_session, test_project):
    service, _goal, run, _agent, task = await _setup(db_session, test_project)
    await _complete(db_session, service, run, task, MISSING_FIELDS)
    [correction] = await _corrections(db_session, task)
    run.status = "paused"
    fixed = await _complete(db_session, service, run, correction, json.dumps(VALID))
    assert (await _ledger(service, db_session, run, task))["stale"] == "run_paused"

    run.status = "running"
    await service.tick(db_session, run.id)

    ledger = await _ledger(service, db_session, run, task)
    assert ledger["status"] == "corrected" and ledger["correction_session_id"] == str(fixed.id)
    assert "stale" not in ledger


async def test_owner_contract_change_starts_a_fresh_cycle_and_drops_the_old_correction(db_session, test_project):
    service, goal, run, _agent, task = await _setup(db_session, test_project)
    original = await _complete(db_session, service, run, task, MISSING_FIELDS)
    [old] = await _corrections(db_session, task)
    goal.constraints = {**(goal.constraints or {}), "no_network": True}
    await db_session.flush()

    await _complete(db_session, service, run, old, json.dumps(VALID))

    ledger = await _ledger(service, db_session, run, task)
    assert ledger["cycle"] == 2 and ledger["status"] == "correcting"
    assert [a["session_id"] for a in ledger["attempts"]] == [str(original.id)]
    new = next(item for item in await _corrections(db_session, task) if item.id != old.id)
    assert ledger["correction_task_id"] == str(new.id)
    assert await service._existing_action_for_key(
        db_session, run.id, f"run:{run.id}:kind:report_consumed:task:{old.id}",
    ) is None
    for _ in range(2):  # ticks do not start yet another cycle
        await service.tick(db_session, run.id)
    assert len(await _corrections(db_session, task)) == 2


async def _answer(db, run, ledger, option, user):
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    decision = await db.get(OrchestrationAuthorityDecision, uuid.UUID(ledger["decision_id"]))
    await OrchestrationAuthorityDecisionService().answer_runtime_question(
        db, decision, option, actor_user_id=user.id, contract_version=decision.contract_version,
    )
    await db.refresh(run)


async def test_owner_retry_answer_starts_new_cycle_with_cycle_subject(db_session, test_project, test_user):
    service, _goal, run, _agent, task = await _setup(db_session, test_project)
    await _complete(db_session, service, run, task, MISSING_FIELDS)
    [first] = await _corrections(db_session, task)
    await _complete(db_session, service, run, first, MISSING_FIELDS)
    await _answer(db_session, run, await _ledger(service, db_session, run, task), "retry", test_user)

    await service.tick(db_session, run.id)

    ledger = await _ledger(service, db_session, run, task)
    assert ledger["cycle"] == 2 and ledger["status"] == "correcting" and len(ledger["previous_attempts"]) == 2
    second = next(item for item in await _corrections(db_session, task) if item.id != first.id)
    await _complete(db_session, service, run, second, MISSING_FIELDS)
    [blocker] = _escalations(run, "result_correction_exhausted")
    assert blocker["subject"] == f"task:{task.id}:result_correction:cycle:2"


async def test_owner_dismiss_answer_closes_the_ledger(db_session, test_project, test_user):
    service, _goal, run, _agent, task = await _setup(db_session, test_project)
    await _complete(db_session, service, run, task, MISSING_FIELDS)
    [first] = await _corrections(db_session, task)
    await _complete(db_session, service, run, first, MISSING_FIELDS)
    await _answer(db_session, run, await _ledger(service, db_session, run, task), "dismiss", test_user)

    for _ in range(2):
        await service.tick(db_session, run.id)

    assert (await _ledger(service, db_session, run, task))["status"] == "owner_declined"
    assert len(await _corrections(db_session, task)) == 1


async def test_unavailable_retry_uses_cycle_subject(db_session, test_project, test_user):
    service, _goal, run, _agent, task = await _setup(db_session, test_project)
    await _complete(db_session, service, run, task, "")
    await _answer(db_session, run, await _ledger(service, db_session, run, task), "retry", test_user)

    await service.tick(db_session, run.id)

    [blocker] = _escalations(run, "capability_unavailable")
    assert blocker["subject"] == f"task:{task.id}:correction_input_unavailable:cycle:2"
    assert await _corrections(db_session, task) == []


async def _fail(db, task):
    task.status = "in_progress"
    session = Session(project_id=task.project_id, task_id=task.id, agent_id=task.assigned_to, adapter_type="api",
                      status="failed", input_context={}, error="timeout", origin="orchestrator", ended_at=_utcnow())
    db.add(session)
    await db.flush()
    await sync_task_from_session(db, session)


async def test_failed_correction_task_counts_as_an_attempt_and_gate_follows_the_ledger(db_session, test_project):
    service, _goal, run, _agent, task = await _setup(db_session, test_project)
    gate = await db_session.get(OrchestrationGate, uuid.UUID(task.metadata_["orchestration"]["gate_id"]))
    await _complete(db_session, service, run, task, MISSING_FIELDS)
    [first] = await _corrections(db_session, task)
    await _fail(db_session, first)

    await service.tick(db_session, run.id)

    [second] = [item for item in await _corrections(db_session, task) if item.id != first.id]
    assert (await service._adhoc_report_failure(db_session, gate))[1] is False
    await _fail(db_session, second)
    await service.tick(db_session, run.id)

    ledger = await _ledger(service, db_session, run, task)
    assert ledger["status"] == "exhausted" and [a.get("task_failed") for a in ledger["attempts"]] == [None, True, True]
    assert len(_escalations(run, "result_correction_exhausted")) == 1
    assert await service._adhoc_report_failure(db_session, gate) == ("Task report is invalid", True)
    assert len(await _corrections(db_session, task)) == 2


async def test_adhoc_gate_waits_through_second_correction_and_passes_on_its_report(db_session, test_project):
    service, _goal, run, _agent, task = await _setup(db_session, test_project)
    gate = await db_session.get(OrchestrationGate, uuid.UUID(task.metadata_["orchestration"]["gate_id"]))
    await _complete(db_session, service, run, task, json.dumps({**VALID, "status": 1, "next_step": 2, "risks": [3]}))
    [first] = await _corrections(db_session, task)
    await _complete(db_session, service, run, first, json.dumps({**VALID, "status": 1}))
    [second] = [item for item in await _corrections(db_session, task) if item.id != first.id]

    await service.validate_open_gates(db_session, run.id)
    assert gate.status == "open"
    assert await service._adhoc_report_failure(db_session, gate) == ("Task report is invalid", False)

    await _complete(db_session, service, run, second, json.dumps(VALID))
    assert await service._adhoc_report_failure(db_session, gate) is None


async def test_planning_envelope_failure_dispatches_correction_with_its_own_schema(db_session, test_project):
    from tests.test_orchestration_plan_ingestion import _completed, _env, _finish, _item, _planning_run

    service, _goal, run, task, session = await _planning_run(db_session, test_project.id)
    output = json.dumps(_env(items=[_item(success_criterion_keys=["protocol_fallback"])]))
    await _finish(db_session, task, session, output)
    await service._ingest_session_evidence(db_session, run, _completed(session.id))
    marker = await service._existing_action_for_key(db_session, run.id, f"run:{run.id}:kind:report_consumed:task:{task.id}")
    error = marker.request["plan_ingestion_error"]

    await service._request_result_correction(db_session, run, task, session.id, "planning_envelope", [error])

    [correction] = await _corrections(db_session, task)
    assert correction.metadata_["orchestration_contract"]["expected_result"] == "planning_envelope"
    assert output in correction.description and error in correction.description
    assert '"plan_items"' in correction.description and correction.assigned_to == task.assigned_to
    ledger = (await service._existing_action_for_key(
        db_session, run.id, f"run:{run.id}:kind:report_consumed:task:{task.id}",
    )).request["result_correction"]
    assert ledger["kind"] == "planning_envelope" and ledger["correction_task_id"] == str(correction.id)
    assert ledger["attempts"][0]["session_id"] == str(session.id)
    # A work-report validator failure is not this task's contract and never starts a correction.
    other = await service._request_result_correction(db_session, run, task, session.id, "work_report", ["x"])
    assert other is None and len(await _corrections(db_session, task)) == 1


@pytest.mark.usefixtures("safe_goal_analysis", "safe_agent_definition_review", "safe_effectiveness_review_continue")
async def test_final_summary_failure_dispatches_correction_with_summary_schema(db_session, test_project):
    from tests.test_orchestration_final_summary_request import _request_summary

    service, _goal, run, writer, task, _gate = await _request_summary(db_session, test_project)
    output = "Everything shipped; see the gates."
    session = await _complete(db_session, service, run, task, output)
    assert await _corrections(db_session, task) == []  # a work-report validator failure is not this contract

    await service._request_result_correction(
        db_session, run, task, session.id, "final_summary", ["final summary must be one JSON object"],
    )

    [correction] = await _corrections(db_session, task)
    assert correction.assigned_to == task.assigned_to
    assert correction.metadata_["orchestration_contract"]["expected_result"] == "final_summary"
    assert output in correction.description and "final summary must be one JSON object" in correction.description
    assert '"unresolved_gaps"' in correction.description
    assert (await _ledger(service, db_session, run, task))["correction_task_id"] == str(correction.id)
