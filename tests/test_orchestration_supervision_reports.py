import json
import uuid

import pytest
from sqlalchemy import select

from huddleroom.models.base import _utcnow
from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGate, OrchestrationGoal, OrchestrationRun
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.event_bus import emit_event_once
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_work_report import validate_work_report
from tests.test_orchestration_roadmap_integration import child_item, release
from tests.test_orchestration_runtime_e2e import _authorized_run


pytestmark = pytest.mark.asyncio


def canonical_report() -> str:
    return json.dumps({
        "status": "completed",
        "changes": ["Implemented bounded wake-up coalescing."],
        "evidence": ["artifact:coalescing-test"],
        "criterion_progress": {"event_latency": "candidate"},
        "decisions": ["Reuse the existing event bus."],
        "risks": [],
        "open_questions": [],
        "next_step": "Request independent verification.",
        "collaboration_need": None,
    })


@pytest.mark.parametrize("output", [
    "not json",
    "[]",
    json.dumps({"status": "completed"}),
    json.dumps({**json.loads(canonical_report()), "changes": [1]}),
    json.dumps({**json.loads(canonical_report()), "criterion_progress": {"x": 1}}),
])
async def test_canonical_report_rejects_malformed_nested_values(output):
    validation = validate_work_report(output)
    assert validation.report is None
    assert validation.errors


async def _terminal_task(db_session, test_project, test_agent, output, *, sessionless=False, contract=None):
    service, _goal, run = await _authorized_run(db_session, test_project)
    task = Task(
        project_id=test_project.id,
        title="Supervised work",
        status="done",
        assigned_to=test_agent.id,
        completed_at=_utcnow(),
        metadata_={
            "orchestration": {"terminal_output": output} if sessionless else {},
            "orchestration_contract": contract or {},
        },
    )
    db_session.add(task)
    await db_session.flush()
    session = None
    if not sessionless:
        session = Session(
            task_id=task.id, agent_id=test_agent.id, project_id=test_project.id,
            adapter_type="api", status="completed", output=output,
            origin="orchestrator", ended_at=_utcnow(),
        )
        db_session.add(session)
    db_session.add(OrchestrationAction(
        run_id=run.id, idempotency_key=f"test-task:{task.id}",
        action_type="create_delegation_task", request={}, target_type="task",
        target_id=task.id, status="completed", dispatch_contract={}, budget_ledger={},
    ))
    await db_session.flush()
    return service, run, task, session


async def test_winning_attempt_is_consumed_once_and_claim_only_requests_verification(
    db_session, test_project, test_agent,
):
    service, run, task, session = await _terminal_task(
        db_session, test_project, test_agent, canonical_report()
    )
    first = await service.consume_canonical_report(db_session, run, task)
    second = await service.consume_canonical_report(db_session, run, task)
    assert first == second
    assert first[0] == session.id
    assert run.supervision_state["verified_progress"] == []
    assert run.supervision_state["verification_candidates"] == ["artifact:coalescing-test"]
    marker = await db_session.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "report_consumed",
    ))
    assert marker.status == "completed"


async def test_completed_marker_replays_its_persisted_report_not_mutated_output(
    db_session, test_project, test_agent,
):
    service, run, task, session = await _terminal_task(
        db_session, test_project, test_agent, canonical_report()
    )
    await service.consume_canonical_report(db_session, run, task)
    session.output = "{}"
    session_id, report = await service.consume_canonical_report(db_session, run, task)
    assert session_id == session.id
    assert report.candidate_evidence == ["artifact:coalescing-test"]


async def test_reserved_marker_is_completed_with_persisted_result(db_session, test_project, test_agent):
    service, run, task, session = await _terminal_task(
        db_session, test_project, test_agent, canonical_report()
    )
    marker = OrchestrationAction(
        run_id=run.id, idempotency_key=f"run:{run.id}:kind:report_consumed:task:{task.id}",
        action_type="report_consumed", request={"task_id": str(task.id), "session_id": str(session.id)},
    )
    db_session.add(marker)
    await db_session.flush()
    _session_id, report = await service.consume_canonical_report(db_session, run, task)
    assert report is not None and marker.status == "completed"
    assert marker.request["report"]["evidence"] == ["artifact:coalescing-test"]


async def test_sessionless_invalid_report_records_validation_errors(db_session, test_project, test_agent):
    service, run, task, _session = await _terminal_task(
        db_session, test_project, test_agent, "{}", sessionless=True
    )
    session_id, report = await service.consume_canonical_report(db_session, run, task)
    assert session_id is None and report is None
    marker = await db_session.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "report_consumed",
    ))
    assert marker.request["report_validation_errors"]
    assert marker.request["clarification_count"] == 0


async def test_malformed_report_gets_one_funded_same_owner_clarification(db_session, test_project, test_agent):
    service, run, task, session = await _terminal_task(
        db_session, test_project, test_agent, "not json",
        contract={"expected_result": "canonical_work_report"},
    )
    goal = await db_session.get(OrchestrationGoal, run.goal_id)
    goal.budget = {"caps": {"max_tokens": 2}}
    session.metadata_ = {"token_count_in": 0, "token_count_out": 0}

    await service.consume_canonical_report(db_session, run, task, session.id)

    clarification = await db_session.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.idempotency_key == f"run:{run.id}:kind:report_clarification:task:{task.id}",
    ))
    assert clarification is not None and clarification.status == "completed"
    follow_up = await db_session.get(Task, clarification.target_id)
    assert follow_up is not None and follow_up.assigned_to == test_agent.id
    assert follow_up.metadata_["orchestration_contract"]["budget"]


async def test_legacy_malformed_report_does_not_request_clarification(db_session, test_project, test_agent):
    service, run, task, session = await _terminal_task(db_session, test_project, test_agent, "not json")
    goal = await db_session.get(OrchestrationGoal, run.goal_id)
    goal.budget = {"caps": {"max_tokens": 2}}
    session.metadata_ = {"token_count_in": 0, "token_count_out": 0}

    await service.consume_canonical_report(db_session, run, task, session.id)

    assert await db_session.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.idempotency_key == f"run:{run.id}:kind:report_clarification:task:{task.id}",
    )) is None


async def test_malformed_report_in_exhausted_outcome_records_attention_without_follow_up(
    db_session, test_project, test_agent,
):
    _roadmap, _parent, _parent_run, _result, row = await release(
        db_session, test_project,
        child_item(allocation={"max_tokens": 0, "max_turns": 0, "max_hours": 0}),
    )
    goal = await db_session.get(OrchestrationGoal, row.child_goal_id)
    run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == goal.id))
    gate = OrchestrationGate(run_id=run.id, success_criterion_key="report", gate_type="work_completed")
    db_session.add(gate)
    await db_session.flush()
    task = Task(
        project_id=test_project.id, title="Exhausted outcome", status="done", assigned_to=test_agent.id,
        metadata_={
            "orchestration": {"run_id": str(run.id), "gate_id": str(gate.id)},
            "orchestration_contract": {"expected_result": "canonical_work_report"},
        },
    )
    db_session.add(task)
    await db_session.flush()
    db_session.add_all([
        OrchestrationAction(
            run_id=run.id, idempotency_key=f"test:outcome-lineage:{task.id}",
            action_type="create_delegation_task", request={}, target_type="task", target_id=task.id,
            status="completed", dispatch_contract={}, budget_ledger={},
        ),
        Session(
            task_id=task.id, agent_id=test_agent.id, project_id=test_project.id, adapter_type="api",
            status="completed", output="not json", origin="orchestrator",
            metadata_={"token_count_in": 0, "token_count_out": 0, "_roadmap_turn_count": 0,
                       "_roadmap_elapsed_seconds": 0},
        ),
    ])
    await db_session.flush()
    session = await db_session.scalar(select(Session).where(Session.task_id == task.id))

    _event, _ = await emit_event_once(db_session, test_project.id, "session.completed", {
        "session_id": str(session.id),
    }, dedup_key=f"exhausted-malformed-report:{session.id}")
    event = await db_session.scalar(select(EventLog).where(
        EventLog.dedup_key == f"exhausted-malformed-report:{session.id}"
    ))
    service = OrchestrationService()
    await service._ingest_session_evidence(db_session, run, event)
    await service._ingest_session_evidence(db_session, run, event)

    assert any(item.get("kind") == "budget_exhausted" for item in run.active_blockers)
    assert any(item.get("kind") == "report_clarification_required" for item in run.active_blockers)
    assert await db_session.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.idempotency_key == f"run:{run.id}:kind:report_clarification:task:{task.id}",
    )) is None
    assert await db_session.scalar(select(Task).where(
        Task.metadata_["orchestration_contract"]["work_function"].as_string() == "report_clarification"
    )) is None


async def test_second_malformed_report_requests_human_attention(db_session, test_project, test_agent):
    service, run, task, session = await _terminal_task(
        db_session, test_project, test_agent, "not json",
        contract={"expected_result": "canonical_work_report"},
    )
    goal = await db_session.get(OrchestrationGoal, run.goal_id)
    goal.budget = {"caps": {"max_tokens": 2}}
    session.metadata_ = {"token_count_in": 0, "token_count_out": 0}
    await service.consume_canonical_report(db_session, run, task, session.id)
    clarification = await db_session.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.idempotency_key == f"run:{run.id}:kind:report_clarification:task:{task.id}",
    ))
    follow_up = await db_session.get(Task, clarification.target_id)
    follow_up.status = "done"
    retry = Session(task_id=follow_up.id, agent_id=test_agent.id, project_id=test_project.id,
                    adapter_type="api", status="completed", output="still not json", origin="orchestrator")
    db_session.add(retry)
    await db_session.flush()

    await service.consume_canonical_report(db_session, run, follow_up, retry.id)

    assert any(item.get("kind") == "report_clarification_required" for item in run.active_blockers)


async def test_done_task_event_consumes_sessionless_metadata_report(db_session, test_project, test_agent):
    service, run, task, _session = await _terminal_task(
        db_session, test_project, test_agent, canonical_report(), sessionless=True
    )
    gate = OrchestrationGate(run_id=run.id, success_criterion_key="done", gate_type="work_completed")
    db_session.add(gate)
    await db_session.flush()
    task.metadata_ = {"orchestration": {"run_id": str(run.id), "gate_id": str(gate.id),
                                           "terminal_output": canonical_report()}}
    await db_session.flush()
    event, _ = await emit_event_once(db_session, test_project.id, "task.status_changed", {
        "task_id": str(task.id), "status": "done", "previous_status": "in_progress",
    }, dedup_key=f"sessionless-report:{task.id}")
    event_log = await db_session.scalar(select(EventLog).where(
        EventLog.dedup_key == f"sessionless-report:{task.id}"
    ))
    await service._ingest_task_evidence(db_session, run, event_log)
    marker = await db_session.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "report_consumed",
    ))
    assert marker.status == "completed"


def test_strict_validator_reports_field_specific_missing_extra_and_duplicate_keys():
    missing = validate_work_report('{"status":"completed"}')
    extra = validate_work_report(canonical_report()[:-1] + ',"extra":true}')
    duplicate = validate_work_report(canonical_report()[:-1] + ',"status":"again"}')
    assert "changes" in " ".join(missing.errors)
    assert "extra" in " ".join(extra.errors)
    assert "duplicate" in " ".join(duplicate.errors)


def test_strict_validator_names_each_invalid_scalar_field():
    for field, value in (("status", 1), ("next_step", 1), ("collaboration_need", 1)):
        payload = json.loads(canonical_report())
        payload[field] = value
        assert field in " ".join(validate_work_report(json.dumps(payload)).errors)


async def test_competing_late_session_cannot_replace_consumed_lineage(
    db_session, test_project, test_agent,
):
    service, run, task, winner = await _terminal_task(
        db_session, test_project, test_agent, canonical_report()
    )
    await service.consume_canonical_report(db_session, run, task, winner.id)
    late = Session(
        task_id=task.id, agent_id=test_agent.id, project_id=test_project.id,
        adapter_type="api", status="completed", output=canonical_report(),
        origin="orchestrator", ended_at=_utcnow(),
    )
    db_session.add(late)
    await db_session.flush()
    with pytest.raises(Exception, match="lineage"):
        await service.consume_canonical_report(db_session, run, task, late.id)
