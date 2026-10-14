import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from huddleroom.models.orchestration import OrchestrationAction
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_work_report import validate_work_report
from tests.test_orchestration_plan_ingestion import (
    _completed, _db_output, _finish, _plan_artifacts, _planning_run,
)
from tests.test_orchestration_work_report import (
    _agent, _make_goal_and_run, _make_gate, _make_orchestrated_task,
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


@pytest.mark.asyncio
async def test_fenced_work_report_accepted_and_marker_records_fence(db_session, test_project):
    service, _, run = await _make_goal_and_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    dev = _agent("developer", ["implementation"])
    db_session.add(dev)
    await db_session.flush()
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, dev.id)
    task.status, task.completed_at = "done", datetime.now(timezone.utc)
    from huddleroom.models.session import Session
    raw = "```json\n" + json.dumps(REPORT) + "\n```"
    session = Session(project_id=test_project.id, task_id=task.id, agent_id=dev.id, adapter_type="api",
                      status="completed", input_context={}, output=raw, metadata_={}, origin="auto")
    db_session.add(session)
    await db_session.flush()

    _, report = await service.consume_canonical_report(db_session, run, task, session.id, None)

    assert report is not None and report.status == "done"
    assert (await _marker(db_session, run, task)).request["normalized"] == "fence"
    assert session.output == raw


def test_validate_work_report_rejects_ambiguous_output():
    text = json.dumps(REPORT)
    assert validate_work_report("note\n" + text).normalized == "framing"
    assert validate_work_report(text).normalized is None
    for bad in (text + text, text[:-3], text[:-1] + ', "status": "x"}'):
        assert validate_work_report(bad).report is None


def test_verification_and_summary_text_normalized():
    verdict = json.dumps({"status": "done", "verdict": "accepted", "evidence": ["e"]})
    assert OrchestrationService._verification_report("```json\n" + verdict + "\n```") == "accepted"
    assert OrchestrationService._verification_report(verdict + verdict) is None
