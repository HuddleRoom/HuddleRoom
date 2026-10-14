import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationEvidence, OrchestrationGate
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.event_bus import emit_event_once
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_work_report import validate_work_report
from tests.conftest import heal_baseline_drift_for_test
from tests.test_orchestration_delegation_contracts import _agent, _delegation_request, _make_run
from tests.test_orchestration_final_summary_request import (  # noqa: F401  (autouse fixtures are reused)
    _manual_task_start, _real_planning_expansion_run, _request_summary, _valid_summary_output,
    no_live_decisions, runnable_workspace,
)
from tests.test_orchestration_plan_ingestion import (
    _completed, _db_output, _finish, _plan_artifacts, _planning_run,
)

REPORT = {
    "status": "done", "changes": ["x"], "evidence": [], "criterion_progress": {}, "decisions": [],
    "risks": [], "open_questions": [], "next_step": "verify", "collaboration_need": None,
}


async def _marker(db_session, run, task):
    return await db_session.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.idempotency_key == f"run:{run.id}:kind:report_consumed:task:{task.id}",
    ))


@pytest.mark.asyncio
async def test_framed_plan_envelope_registers_plan_and_keeps_raw_output(db_session, test_project):
    service, _, run, task, session = await _planning_run(db_session, test_project.id)
    raw = "Here is the plan:\n" + _db_output() + "\nLet me know."
    await _finish(db_session, task, session, raw)

    await service._ingest_session_evidence(db_session, run, _completed(session.id))

    assert "plan_artifact_id" in run.plan_state and await _plan_artifacts(db_session, task.id) == 1
    assert (await _marker(db_session, run, task)).request["normalized"] == "framing"
    assert session.output == raw


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["\n" + json.dumps({"other": 1}), ""])  # second object, truncated
async def test_ambiguous_plan_output_still_rejected(db_session, test_project, suffix):
    service, _, run, task, session = await _planning_run(db_session, test_project.id)
    out = _db_output() + suffix if suffix else _db_output()[:-5]
    await _finish(db_session, task, session, "Plan:\n" + out)

    await service._ingest_session_evidence(db_session, run, _completed(session.id))

    assert "plan_artifact_id" not in run.plan_state and await _plan_artifacts(db_session, task.id) == 0


@pytest.mark.asyncio
async def test_duplicate_key_plan_output_rejected(db_session, test_project):
    service, _, run, task, session = await _planning_run(db_session, test_project.id)
    out = _db_output()[:-1] + ', "summary": "again"}'
    await _finish(db_session, task, session, "Plan:\n" + out)

    await service._ingest_session_evidence(db_session, run, _completed(session.id))

    assert await _plan_artifacts(db_session, task.id) == 0
    assert "exactly one JSON object" in (await _marker(db_session, run, task)).request["plan_ingestion_error"]


@pytest.mark.asyncio
async def test_fenced_work_report_accepted_via_real_delegation(db_session, test_project):
    agent = _agent("delegate", capabilities=["implementation"])
    db_session.add(agent)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    action = await service.execute_create_delegation_task_action(
        db_session, run_id=run.id, request={**_delegation_request(agent.id), "work_function": "implementation"},
        idempotency_key=f"run:{run.id}:kind:create_delegation_task:fenced",
    )
    task = await db_session.get(Task, action.id)
    task.status, task.completed_at = "done", datetime.now(timezone.utc)
    raw = "```json\n" + json.dumps(REPORT) + "\n```"
    session = Session(project_id=test_project.id, task_id=task.id, agent_id=agent.id, adapter_type="api",
                      status="completed", input_context={}, output=raw, metadata_={}, origin="auto")
    db_session.add(session)
    await db_session.flush()

    _, report = await service.consume_canonical_report(db_session, run, task, session.id, None)

    assert report is not None and report.status == "done"
    assert (await _marker(db_session, run, task)).request["normalized"] == "fence"
    assert session.output == raw


@pytest.mark.asyncio
async def test_clean_work_report_marker_has_no_normalized_key(db_session, test_project):
    agent = _agent("delegate", capabilities=["implementation"])
    db_session.add(agent)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    action = await service.execute_create_delegation_task_action(
        db_session, run_id=run.id, request={**_delegation_request(agent.id), "work_function": "implementation"},
        idempotency_key=f"run:{run.id}:kind:create_delegation_task:clean",
    )
    task = await db_session.get(Task, action.id)
    task.status, task.completed_at = "done", datetime.now(timezone.utc)
    session = Session(project_id=test_project.id, task_id=task.id, agent_id=agent.id, adapter_type="api",
                      status="completed", input_context={}, output=json.dumps(REPORT), metadata_={}, origin="auto")
    db_session.add(session)
    await db_session.flush()
    await service.consume_canonical_report(db_session, run, task, session.id, None)
    assert "normalized" not in (await _marker(db_session, run, task)).request


@pytest.mark.asyncio
async def test_plan_hash_identical_for_clean_and_framed_output(db_session, test_project):
    hashes = []
    for wrap in (lambda o: o, lambda o: "Plan follows.\n" + o + "\nDone."):
        service, _, run, task, session = await _planning_run(db_session, test_project.id)
        await _finish(db_session, task, session, wrap(_db_output()))
        await service._ingest_session_evidence(db_session, run, _completed(session.id))
        hashes.append(run.plan_state["plan_output_hash"])
    assert hashes[0] == hashes[1]


@pytest.mark.asyncio
async def test_fenced_plan_envelope_registers_plan(db_session, test_project):
    service, _, run, task, session = await _planning_run(db_session, test_project.id)
    await _finish(db_session, task, session, "```json\n" + _db_output() + "\n```")
    await service._ingest_session_evidence(db_session, run, _completed(session.id))
    assert await _plan_artifacts(db_session, task.id) == 1
    assert (await _marker(db_session, run, task)).request["normalized"] == "fence"


async def _summary_run(db_session, test_project, wrap):
    service, goal, run, writer, task, gate = await _request_summary(db_session, test_project)
    goal.weight = "trivial"
    await heal_baseline_drift_for_test(db_session, goal, run)
    session = Session(agent_id=writer.id, task_id=task.id, project_id=test_project.id, adapter_type="api",
                      status="completed", output=wrap(await _valid_summary_output(db_session, goal, run)),
                      metadata_={}, origin="auto")
    db_session.add(session)
    await db_session.flush()
    await emit_event_once(db_session, test_project.id, "session.completed",
                          {"session_id": str(session.id), "task_id": str(task.id)},
                          dedup_key=f"norm-summary:{session.id}")
    result = await service.tick(db_session, run.id)
    evidence = (await db_session.execute(select(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == gate.id, OrchestrationEvidence.source_type == "session",
    ))).scalar_one()
    return result, gate, evidence, session


@pytest.mark.asyncio
@pytest.mark.usefixtures("safe_goal_analysis", "safe_agent_definition_review", "safe_effectiveness_review_continue")
@pytest.mark.parametrize("wrap,how", [
    (lambda o: "Summary:\n" + o, "framing"),
    (lambda o: "```json\n" + o + "\n```", "fence"),
])
async def test_wrapped_final_summary_accepted_and_evidence_records_normalization(db_session, test_project, wrap, how):
    result, gate, evidence, session = await _summary_run(db_session, test_project, wrap)
    assert result["run_completed"] is True and gate.status == "accepted"
    assert evidence.evidence_metadata["normalized"] == how
    assert session.output.startswith(("Summary:", "```"))


@pytest.mark.asyncio
@pytest.mark.usefixtures("safe_goal_analysis", "safe_agent_definition_review", "safe_effectiveness_review_continue")
async def test_ambiguous_final_summary_rejected(db_session, test_project):
    result, gate, evidence, _ = await _summary_run(db_session, test_project, lambda o: o + "\n" + o)
    assert result["run_completed"] is False and gate.status == "failed"
    assert gate.failure_reason == "Final summary output must be valid JSON"
    assert "normalized" not in (evidence.evidence_metadata or {})


@pytest.mark.asyncio
async def test_framed_outcome_verification_evidence_records_normalization(db_session, test_project):
    planner = _agent("planner", role="planner", capabilities=["planning", "validation"])
    developer = _agent("developer", role="developer", capabilities=["implementation"])
    db_session.add_all([planner, developer])
    await db_session.flush()
    service, goal, run = await _real_planning_expansion_run(db_session, test_project.id, planner, developer)
    row = (await db_session.execute(
        select(OrchestrationEvidence, Session).join(Session, Session.id == OrchestrationEvidence.source_id)
        .where(OrchestrationEvidence.run_id == run.id, OrchestrationEvidence.source_type == "verification")
    )).first()
    old, session = row
    gate = await db_session.get(OrchestrationGate, old.gate_id)
    task = await db_session.get(Task, session.task_id)
    session.output = "Verified:\n" + session.output
    await emit_event_once(db_session, test_project.id, "session.completed",
                          {"session_id": str(session.id), "task_id": str(task.id)},
                          dedup_key=f"norm-verif:{session.id}")
    event = await db_session.scalar(select(EventLog).where(EventLog.dedup_key == f"norm-verif:{session.id}"))
    assert await service._ingest_outcome_verification_evidence(db_session, run, event, gate, task, session) == 1
    new = await db_session.scalar(select(OrchestrationEvidence).where(
        OrchestrationEvidence.observed_event_id == event.id, OrchestrationEvidence.source_type == "verification"))
    assert new.verdict == "candidate" and new.evidence_metadata["normalized"] == "framing"


def test_validate_work_report_rejects_ambiguous_output():
    text = json.dumps(REPORT)
    assert validate_work_report("note\n" + text).normalized == "framing"
    assert validate_work_report(text).normalized is None
    for bad in (text + text, text[:-3], text[:-1] + ', "status": "x"}'):
        assert validate_work_report(bad).report is None


def test_verification_report_verdict_and_ambiguity():
    verdict = json.dumps({"status": "done", "verdict": "accepted", "evidence": ["e"]})
    findings = {"evidence": ["e"], "notes": None}
    assert OrchestrationService._verification_report("```json\n" + verdict + "\n```") == (
        "accepted", [], {**findings, "normalized": "fence"},
    )
    assert OrchestrationService._verification_report(verdict) == ("accepted", [], findings)
    assert OrchestrationService._verification_report(verdict + verdict) == (
        None, ["result contains more than one JSON object"], None,
    )
