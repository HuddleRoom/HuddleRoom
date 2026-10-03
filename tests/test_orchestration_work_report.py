import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from huddleroom.models.agent import Agent
from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import OrchestrationGate
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.services.event_bus import emit_event_once
from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_work_report import (
    WorkReport,
    curate_report_memory,
    parse_work_report,
)

pytestmark = pytest.mark.asyncio


def test_parse_valid_report():
    output = ('{"status": "done", "changes": ["added X"], "evidence": ["artifact:abc"], '
              '"criterion_progress": {"c1": "claimed"}, "decisions": [], "risks": [], '
              '"open_questions": [], "next_step": "verify c1", "collaboration_need": null}')
    report = parse_work_report(output, {"status": "str", "changes": "list[str]"})
    assert report is not None
    assert report.changes == ["added X"]
    assert report.candidate_evidence == ["artifact:abc"]
    assert report.recommended_next_step == "verify c1"


def test_parse_malformed_returns_none():
    assert parse_work_report("not json at all", {}) is None
    assert parse_work_report("", {}) is None
    assert parse_work_report(None, {}) is None


def _agent(name_prefix: str, capabilities: list[str]) -> Agent:
    return Agent(
        name=f"{name_prefix}-{uuid.uuid4()}",
        role="agent",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=capabilities,
        config={},
        is_active=True,
    )


async def _make_goal_and_run(db_session, project_id):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=project_id,
        data=OrchestrationGoalCreate(
            objective="Ship work-report curation",
            success_criteria=[{"key": "work", "description": "Reports are curated."}],
            constraints={},
            budget={},
        ),
        created_by_user_id=None,
    )
    return service, goal, run


async def test_curate_report_memory_writes_section_with_provenance(db_session, test_project):
    _service, goal, run = await _make_goal_and_run(db_session, test_project.id)
    memory_service = OrchestrationMemoryService()
    developer_id = uuid.uuid4()
    report = WorkReport(
        status="done",
        changes=["added the parser"],
        decisions=["used regex over grammar"],
        risks=["edge case in unicode input"],
    )

    count = await curate_report_memory(
        db_session,
        memory_service,
        goal,
        run,
        report,
        report_id="report-abc",
        producer_agent_id=developer_id,
    )
    await db_session.flush()

    assert count == 1
    section = await memory_service.get_section(db_session, goal.project_id, goal.id, "report_report-abc")
    assert section is not None
    assert "report-abc" in section.body
    assert str(developer_id) in section.body
    assert "added the parser" in section.body
    assert section.run_id == run.id
    assert section.created_by == "orchestrator"


async def test_curate_report_memory_writes_nothing_for_ineligible_report(db_session, test_project):
    _service, goal, run = await _make_goal_and_run(db_session, test_project.id)
    memory_service = OrchestrationMemoryService()
    empty_report = WorkReport(status="done")

    count = await curate_report_memory(
        db_session,
        memory_service,
        goal,
        run,
        empty_report,
        report_id="report-empty",
        producer_agent_id=None,
    )

    assert count == 0
    section = await memory_service.get_section(db_session, goal.project_id, goal.id, "report_report-empty")
    assert section is None


async def _make_gate(db_session, run_id) -> OrchestrationGate:
    gate = OrchestrationGate(
        run_id=run_id,
        success_criterion_key="plan_item:implement-report",
        gate_type="work_completed",
        required_evidence={
            "required_source_types": ["task", "session"],
            "min_count": 1,
            "plan_item_id": "implement-report",
        },
    )
    db_session.add(gate)
    await db_session.flush()
    return gate


async def _make_orchestrated_task(db_session, project_id, run_id, gate_id, agent_id, report_schema=None) -> Task:
    task = Task(
        project_id=project_id,
        title="Implementation work",
        description="Work created from an accepted orchestration plan item.",
        status="in_progress",
        assigned_to=agent_id,
        metadata_={
            "orchestration": {
                "goal_id": "unused",
                "run_id": str(run_id),
                "work_function": "implementation",
                "plan_item_id": "implement-report",
                "plan_item_gate_id": str(gate_id),
                "expand_action_id": str(uuid.uuid4()),
            },
            "orchestration_plan_item": {
                "id": "implement-report",
                "work_function": "implementation",
                "accepted_plan_artifact_id": str(uuid.uuid4()),
            },
            "orchestration_contract": {
                "report_schema": report_schema or {},
            },
        },
    )
    db_session.add(task)
    await db_session.flush()
    return task


async def test_session_completed_event_replay_curates_memory_once(db_session, test_project):
    """Replaying the same session.completed event does not curate memory twice.

    This exercises the pre-existing evidence-level replay guard: on replay,
    _record_evidence_once returns created=0 for the already-recorded evidence
    row, so _ingest_session_evidence's report-consumption block (which is
    itself gated on `created`) never re-runs and never reaches the
    report_consumed marker. The report_consumed OrchestrationAction marker is
    a separate, belt-and-suspenders guard at the task-terminalization level
    (relevant if a *second, distinct* completed session arrived for an
    already-"done" task; task_service separately prevents that scenario from
    occurring in practice for a single task)."""
    service, goal, run = await _make_goal_and_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    developer = _agent("developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, developer.id)
    task.status = "done"
    task.completed_at = datetime.now(timezone.utc)
    report_output = (
        '{"status": "done", "changes": ["implemented the report parser"], '
        '"evidence": [], "criterion_progress": {}, '
        '"decisions": ["kept it dependency-free"], "risks": [], '
        '"open_questions": [], "next_step": "verify", "collaboration_need": null}'
    )
    session = Session(
        project_id=test_project.id,
        task_id=task.id,
        agent_id=developer.id,
        adapter_type="api",
        status="completed",
        input_context={},
        output=report_output,
        metadata_={},
        origin="auto",
    )
    db_session.add(session)
    await db_session.flush()
    event, _ = await emit_event_once(
        db_session,
        test_project.id,
        "session.completed",
        {
            "session_id": str(session.id),
            "task_id": str(task.id),
            "project_id": str(test_project.id),
        },
        dedup_key=f"report-session-completed:{session.id}",
    )

    first = await service.tick(db_session, run.id)
    event_log = (await db_session.execute(
        select(EventLog).where(EventLog.id == event.id)
    )).scalar_one()
    run.event_cursor = event_log.seq - 1
    await db_session.flush()
    second = await service.tick(db_session, run.id)

    assert first["evidence_created"] == 1
    assert second["evidence_created"] == 0

    memory_service = OrchestrationMemoryService()
    section_key = f"report_{session.id}"
    section = await memory_service.get_section(db_session, goal.project_id, goal.id, section_key)
    assert section is not None
    assert "implemented the report parser" in section.body
    assert str(session.id) in section.body

    sections = await memory_service.list_sections(db_session, goal.project_id, goal.id)
    report_sections = [s for s in sections if s.section_key == section_key]
    assert len(report_sections) == 1  # event replay did not curate twice
