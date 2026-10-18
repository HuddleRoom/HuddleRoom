import json
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import select

from huddleroom.models.base import _utcnow
from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGate, OrchestrationGoal, OrchestrationRun
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.event_bus import emit_event_once
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_work_report import validate_work_report
from tests.conftest import adopt_session
from tests.test_orchestration_roadmap_integration import child_item, release
from tests.test_orchestration_runtime_e2e import _authorized_run


pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def test_engine(tmp_path):
    """Private engine per test: committed orchestration rows are discarded with the file."""
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import create_async_engine
    from huddleroom.models.base import Base
    from tests.conftest import _seed_anon_actors

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'owned-reports.db'}")

    @event.listens_for(engine.sync_engine, "connect")
    def pragmas(connection, _record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await _seed_anon_actors(connection)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(autouse=True)
async def clean_test_database():
    yield


@pytest_asyncio.fixture
async def db_session(owned_session_factory):
    """Autobegin session (no session.begin()) so the test task can own it."""
    async with owned_session_factory() as db:
        yield db
        await db.rollback()


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
    adopt_session(db_session)
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
    adopt_session(db_session)
    service, run, task, session = await _terminal_task(
        db_session, test_project, test_agent, canonical_report()
    )
    await service.consume_canonical_report(db_session, run, task)
    session.output = "{}"
    session_id, report = await service.consume_canonical_report(db_session, run, task)
    assert session_id == session.id
    assert report.candidate_evidence == ["artifact:coalescing-test"]


async def test_reserved_marker_is_completed_with_persisted_result(db_session, test_project, test_agent):
    adopt_session(db_session)
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
    adopt_session(db_session)
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
    adopt_session(db_session)
    # A legacy contract without a stored result kind is a work report and gets the same bounded correction.
    service, run, task, session = await _terminal_task(db_session, test_project, test_agent, "not json")
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
    assert follow_up.metadata_["orchestration_contract"]["budget"] == {"max_tokens": "1"}
    assert follow_up.metadata_["result_correction_parent_task_id"] == str(task.id)
    marker = await service._existing_action_for_key(
        db_session, run.id, f"run:{run.id}:kind:report_consumed:task:{task.id}"
    )
    assert marker.request["result_correction"]["correction_task_id"] == str(follow_up.id)


async def test_malformed_report_in_exhausted_outcome_records_attention_without_follow_up(
    db_session, test_project, test_agent,
):
    adopt_session(db_session)
    _roadmap, _parent, _parent_run, _result, row = await release(
        db_session, test_project,
        child_item(allocation={"max_tokens": 2, "max_turns": 1, "max_hours": 1}),
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
            metadata_={"token_count_in": 2, "token_count_out": 0, "_roadmap_turn_count": 1,
                       "_roadmap_elapsed_seconds": 3600},
        ),
    ])
    await db_session.flush()
    session = await db_session.scalar(select(Session).where(Session.task_id == task.id))
    from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService

    summary = await OrchestrationBudgetService().supervision_summary(db_session, goal, run)
    assert summary["remaining"] == {"max_tokens": "0", "max_turns": "0", "max_hours": "0"}

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
    assert [item["category"] for item in run.active_blockers if item.get("kind") == "owner_escalation"] == [
        "budget_exhausted"
    ]
    assert await db_session.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.idempotency_key == f"run:{run.id}:kind:report_clarification:task:{task.id}",
    )) is None
    assert await db_session.scalar(select(Task).where(
        Task.metadata_["orchestration_contract"]["work_function"].as_string() == "report_clarification"
    )) is None


async def test_second_malformed_report_requests_human_attention(db_session, test_project, test_agent):
    adopt_session(db_session)
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

    # Same error count as the first failure: no second correction, the owner is asked.
    assert [item["category"] for item in run.active_blockers if item.get("kind") == "owner_escalation"] == [
        "result_correction_exhausted"
    ]


async def test_done_task_event_consumes_sessionless_metadata_report(db_session, test_project, test_agent):
    adopt_session(db_session)
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
    adopt_session(db_session)
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


async def test_cached_invalid_report_retries_after_funding_returns(db_session, test_project, test_agent, stub_decision):
    adopt_session(db_session)
    stub_decision(lambda _context: {"action_type": "noop"})
    service, run, task, session = await _terminal_task(
        db_session, test_project, test_agent, "not json", contract={"expected_result": "canonical_work_report"},
    )
    goal = await db_session.get(OrchestrationGoal, run.goal_id)
    goal.budget = {"caps": {"max_tokens": 2}}
    session.metadata_ = {"token_count_in": 2, "token_count_out": 0}
    await service.consume_canonical_report(db_session, run, task, session.id)
    marker = await service._existing_action_for_key(db_session, run.id, f"run:{run.id}:kind:report_consumed:task:{task.id}")
    assert marker.request["clarification_count"] == 0
    escalated = lambda: [b for b in run.active_blockers if b.get("kind") == "owner_escalation"]  # noqa: E731
    assert [b["category"] for b in escalated()] == ["budget_exhausted"]
    session.metadata_ = {"token_count_in": 0, "token_count_out": 0}
    for _ in range(2):  # the scheduler alone re-enters the pending correction once funding returns
        await service.tick(db_session, run.id)
    assert escalated() == []
    clarification = await service._existing_action_for_key(db_session, run.id, f"run:{run.id}:kind:report_clarification:task:{task.id}")
    assert clarification is not None and clarification.status == "completed"
    ledger = marker.request["result_correction"]
    assert ledger["status"] == "correcting" and ledger["correction_task_id"] == str(clarification.target_id)
    assert len(ledger["attempts"]) == 1  # replays never add attempts
    await service.consume_canonical_report(db_session, run, task, session.id)
    assert marker.request["result_correction"] == ledger
