import uuid
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.sql.functions import count

pytestmark = pytest.mark.usefixtures("safe_goal_analysis")

from huddleroom.models.agent import Agent
from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationEvidence, OrchestrationGate
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.services.orchestration_service import OrchestrationService


def _agent(
    name_prefix: str,
    role: str,
    capabilities: list[str],
    *,
    is_active: bool = True,
) -> Agent:
    return Agent(
        name=f"{name_prefix}-{uuid.uuid4()}",
        role=role,
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=capabilities,
        config={},
        is_active=is_active,
    )


async def _complete_goal_definition(db_session, goal, run):
    from tests.conftest import complete_baseline_processes

    await complete_baseline_processes(db_session, goal, run)


async def _refresh_baseline_processes(db_session, goal, run):
    from tests.conftest import complete_baseline_processes

    await complete_baseline_processes(db_session, goal, run)


async def _make_run(db_session, project_id):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=project_id,
        data=OrchestrationGoalCreate(
            objective="Ship basic recovery",
            success_criteria=[{"key": "recovered", "description": "Basic recovery keeps work moving."}],
            constraints={},
            budget={},
        ),
        created_by_user_id=None,
    )
    run.baseline_authorized = True
    await _complete_goal_definition(db_session, goal, run)
    return service, goal, run


async def _make_gate(db_session, run_id, *, status: str = "open", failure_reason: str | None = None):
    gate = OrchestrationGate(
        run_id=run_id,
        success_criterion_key="plan_item:implement-api",
        gate_type="work_completed",
        required_evidence={
            "required_source_types": ["task"],
            "min_count": 1,
            "plan_item_id": "implement-api",
        },
        status=status,
        failure_reason=failure_reason,
        failed_at=datetime.now(timezone.utc) if status == "failed" else None,
    )
    db_session.add(gate)
    await db_session.flush()
    return gate


async def _make_orchestrated_task(
    db_session,
    project_id,
    run_id,
    gate_id,
    agent_id,
    *,
    status: str = "failed",
    work_function: str = "implementation",
) -> Task:
    task = Task(
        project_id=project_id,
        title="Implementation work",
        description="Work created from an accepted orchestration plan item.",
        status=status,
        assigned_to=agent_id,
        started_at=datetime.now(timezone.utc),
        metadata_={
            "orchestration": {
                "goal_id": "unused-by-basic-recovery",
                "run_id": str(run_id),
                "action_id": str(uuid.uuid4()),
                "work_function": work_function,
                "plan_item_id": "implement-api",
                "plan_item_gate_id": str(gate_id),
                "expand_action_id": str(uuid.uuid4()),
            },
            "orchestration_plan_item": {
                "id": "implement-api",
                "work_function": work_function,
                "accepted_plan_artifact_id": str(uuid.uuid4()),
            },
        },
    )
    db_session.add(task)
    await db_session.flush()
    return task


async def _add_failed_session(db_session, task: Task, agent_id) -> Session:
    session = Session(
        agent_id=agent_id,
        task_id=task.id,
        project_id=task.project_id,
        adapter_type="api",
        status="failed",
        input_context={},
        output=None,
        error="model_error",
        metadata_={},
        origin="auto",
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
    )
    db_session.add(session)
    await db_session.flush()
    return session


async def _add_gate_evidence(
    db_session,
    run_id,
    gate_id,
    *,
    source_type: str,
    source_id,
    producer_agent_id,
    verdict: str,
    event_seq: int,
) -> OrchestrationEvidence:
    evidence = OrchestrationEvidence(
        run_id=run_id,
        gate_id=gate_id,
        source_type=source_type,
        source_id=source_id,
        observed_event_id=None,
        producer_agent_id=producer_agent_id,
        verdict=verdict,
        evidence_metadata={"event_seq": event_seq, "source_type": source_type},
    )
    db_session.add(evidence)
    await db_session.flush()
    return evidence


async def _actions(db_session, run_id, action_type: str):
    result = await db_session.execute(
        select(OrchestrationAction)
        .where(
            OrchestrationAction.run_id == run_id,
            OrchestrationAction.action_type == action_type,
        )
        .order_by(OrchestrationAction.created_at.asc(), OrchestrationAction.id.asc())
    )
    return list(result.scalars().all())


async def _events(db_session, project_id, event_type: str):
    result = await db_session.execute(
        select(EventLog)
        .where(EventLog.project_id == project_id, EventLog.event_type == event_type)
        .order_by(EventLog.seq.asc())
    )
    return list(result.scalars().all())


@pytest.mark.asyncio
async def test_recover_run_retries_failed_task_once(db_session, test_project):
    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, developer.id)
    failed_session = await _add_failed_session(db_session, task, developer.id)

    created = await service.recover_run(db_session, run.id, baseline_ready=True)
    retry_actions = await _actions(db_session, run.id, "retry_task")

    assert created == 1
    assert task.status == "in_progress"
    assert retry_actions[0].status == "completed"
    assert retry_actions[0].target_type == "session"
    assert retry_actions[0].target_id != failed_session.id
    retried_session = await db_session.get(Session, retry_actions[0].target_id)
    assert retried_session is not None
    assert retried_session.task_id == task.id
    assert retried_session.agent_id == developer.id
    assert retried_session.status == "pending"
    assert run.retry_state["tasks"][str(task.id)]["last_recovery"] == "retry_task"

    replay_created = await service.recover_run(db_session, run.id, baseline_ready=True)
    assert replay_created == 0
    assert await db_session.scalar(select(count(Session.id)).where(Session.task_id == task.id)) == 2


@pytest.mark.asyncio
async def test_recover_run_reassigns_after_retry_limit(db_session, test_project):
    original = _agent("developer", "developer", ["implementation"])
    alternate = _agent("alternate", "developer", ["implementation"])
    db_session.add_all([original, alternate])
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, original.id)
    await _add_failed_session(db_session, task, original.id)
    await _add_failed_session(db_session, task, original.id)

    created = await service.recover_run(db_session, run.id, baseline_ready=True)
    reassign_actions = await _actions(db_session, run.id, "reassign_task")

    assert created == 1
    assert task.assigned_to == alternate.id
    assert task.status == "in_progress"
    assert reassign_actions[0].status == "completed"
    assert reassign_actions[0].target_type == "session"
    session = await db_session.get(Session, reassign_actions[0].target_id)
    assert session is not None
    assert session.agent_id == alternate.id
    assert run.retry_state["tasks"][str(task.id)]["last_recovery"] == "reassign_task"


@pytest.mark.asyncio
async def test_recover_run_asks_human_when_reassign_has_no_alternate_agent(db_session, test_project):
    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, developer.id)
    await _add_failed_session(db_session, task, developer.id)
    await _add_failed_session(db_session, task, developer.id)

    created = await service.recover_run(db_session, run.id, baseline_ready=True)
    ask_actions = await _actions(db_session, run.id, "ask_human")
    events = await _events(db_session, test_project.id, "orchestration.human_input_required")

    assert created == 1
    assert task.status == "failed"
    assert goal.status == "blocked"
    assert run.status == "blocked"
    assert run.active_blockers[0]["kind"] == "reassign_required"
    assert run.active_blockers[0]["task_id"] == str(task.id)
    assert ask_actions[0].status == "completed"
    assert ask_actions[0].request["gate_id"] == str(gate.id)
    assert events[0].payload["question"].startswith("Task failed after retry limit")


@pytest.mark.asyncio
async def test_recover_run_escalates_after_repeated_failures(db_session, test_project):
    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, developer.id)
    await _add_failed_session(db_session, task, developer.id)
    await _add_failed_session(db_session, task, developer.id)
    await _add_failed_session(db_session, task, developer.id)

    created = await service.recover_run(db_session, run.id, baseline_ready=True)
    ask_actions = await _actions(db_session, run.id, "ask_human")
    events = await _events(db_session, test_project.id, "orchestration.human_input_required")

    assert created == 1
    assert goal.status == run.status == "blocked"
    assert ask_actions[0].status == "completed"
    assert ask_actions[0].target_type == "authority_decision"
    assert events[0].payload["reason"] == "Task failed repeatedly after retry and reassignment attempts."
    assert run.active_blockers == [{
        "kind": "repeated_failure",
        "task_id": str(task.id),
        "gate_id": str(gate.id),
        "attempt_count": 3,
        "failed_session_ids": [str(session.id) for session in await db_session.scalars(
            select(Session).where(Session.task_id == task.id, Session.status == "failed").order_by(Session.id)
        )],
        "owner": "human",
        "reason": "Task failed repeatedly after retry and reassignment attempts.",
        "recommended_action": "Review the failed attempts and choose how to proceed.",
        "decision_id": str(ask_actions[0].target_id),
    }]


@pytest.mark.asyncio
async def test_recover_run_stops_creating_work_after_repeated_failure_escalation(db_session, test_project):
    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task_to_pause = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, developer.id)
    sibling_failed_task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, developer.id)
    task_to_pause.updated_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
    sibling_failed_task.updated_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    await db_session.flush()
    await _add_failed_session(db_session, task_to_pause, developer.id)
    await _add_failed_session(db_session, task_to_pause, developer.id)
    await _add_failed_session(db_session, task_to_pause, developer.id)
    await _add_failed_session(db_session, sibling_failed_task, developer.id)

    created = await service.recover_run(db_session, run.id, baseline_ready=True)
    ask_actions = await _actions(db_session, run.id, "ask_human")
    retry_actions = await _actions(db_session, run.id, "retry_task")

    assert created == 1
    assert goal.status == "blocked"
    assert run.status == "blocked"
    assert len(ask_actions) == 1
    assert retry_actions == []
    assert await db_session.scalar(select(count(Session.id)).where(Session.task_id == sibling_failed_task.id)) == 1


@pytest.mark.asyncio
async def test_recover_run_marks_blocked_task_and_asks_human_once(db_session, test_project):
    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(
        db_session,
        test_project.id,
        run.id,
        gate.id,
        developer.id,
        status="blocked",
    )

    created = await service.recover_run(db_session, run.id, baseline_ready=True)
    replay_created = await service.recover_run(db_session, run.id, baseline_ready=True)
    ask_actions = await _actions(db_session, run.id, "ask_human")

    assert created == 1
    assert replay_created == 0
    assert goal.status == "blocked"
    assert run.status == "blocked"
    assert run.active_blockers[0]["kind"] == "task_blocked"
    assert run.active_blockers[0]["task_id"] == str(task.id)
    assert len(ask_actions) == 1
    assert ask_actions[0].request["question"].startswith("Task is blocked")


@pytest.mark.asyncio
async def test_recover_run_does_not_mutate_blocked_work_before_agent_review(
    db_session, test_project
):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Recover only after agent review",
            success_criteria=[{"key": "recovered", "description": "Work resumes."}],
        ),
        created_by_user_id=None,
    )
    process_service = OrchestrationProcessService()
    for process_type in ("goal_definition", "manager_selection"):
        process = await process_service.start_process(
            db_session,
            goal.id,
            process_type=process_type,
            trigger_reason="test setup",
            run_id=run.id,
        )
        await process_service.complete_process(db_session, process)
        assert process.status == "completed"
    assert await process_service.get_current(
        db_session, goal.id, "agent_definition_review"
    ) is None
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(
        db_session,
        test_project.id,
        run.id,
        gate.id,
        developer.id,
        status="blocked",
    )

    created = await service.recover_run(db_session, run.id, baseline_ready=False)

    assert created == 0
    assert await _actions(db_session, run.id, "ask_human") == []
    assert goal.status == "active"
    assert run.status == "running"
    assert run.active_blockers == []
    assert run.retry_state == {}
    assert task.status == "blocked"


@pytest.mark.asyncio
async def test_recover_run_requests_verification_for_stale_gate(db_session, test_project):
    developer = _agent("developer", "developer", ["implementation"])
    validator = _agent("validator", "validator", ["validation", "testing"])
    db_session.add_all([developer, validator])
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id, status="failed", failure_reason="Evidence is stale")
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, developer.id, status="done")
    await _add_gate_evidence(
        db_session,
        run.id,
        gate.id,
        source_type="task",
        source_id=task.id,
        producer_agent_id=developer.id,
        verdict="accepted",
        event_seq=20,
    )
    await _refresh_baseline_processes(db_session, goal, run)

    created = await service.recover_run(db_session, run.id, baseline_ready=True)
    verification_actions = await _actions(db_session, run.id, "request_verification")

    assert created == 1
    assert verification_actions[0].status == "completed"
    assert verification_actions[0].target_type == "task"
    verification_task = await db_session.get(Task, verification_actions[0].target_id)
    assert verification_task is not None
    assert verification_task.assigned_to == validator.id
    assert verification_task.parent_id == task.id
    assert verification_task.metadata_["orchestration"]["work_function"] == "validation"
    assert verification_task.metadata_["orchestration"]["recovery_kind"] == "stale_gate_verification"


@pytest.mark.asyncio
async def test_recover_run_creates_fix_task_for_rejected_review(db_session, test_project):
    developer = _agent("developer", "developer", ["implementation"])
    reviewer = _agent("reviewer", "reviewer", ["review"])
    db_session.add_all([developer, reviewer])
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id, status="failed", failure_reason="Rejected evidence from review")
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, developer.id, status="done")
    review_session = Session(
        agent_id=reviewer.id,
        task_id=task.id,
        project_id=test_project.id,
        adapter_type="api",
        status="completed",
        input_context={},
        output="changes requested",
        metadata_={},
        origin="auto",
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
    )
    db_session.add(review_session)
    await db_session.flush()
    await _add_gate_evidence(
        db_session,
        run.id,
        gate.id,
        source_type="review",
        source_id=review_session.id,
        producer_agent_id=reviewer.id,
        verdict="rejected",
        event_seq=30,
    )
    await _refresh_baseline_processes(db_session, goal, run)

    created = await service.recover_run(db_session, run.id, baseline_ready=True)
    fix_actions = [
        action
        for action in await _actions(db_session, run.id, "create_delegation_task")
        if action.request["work_function"] == "implementation"
    ]

    assert created == 1
    assert fix_actions[0].status == "completed"
    fix_task = await db_session.get(Task, fix_actions[0].target_id)
    assert fix_task is not None
    assert fix_task.assigned_to == developer.id
    assert fix_task.parent_id == task.id
    assert fix_task.metadata_["orchestration"]["recovery_kind"] == "review_rejected_fix"
    assert "Fix the rejected review findings" in fix_actions[0].request["scope"]


@pytest.mark.asyncio
async def test_tick_reports_recoveries_created(db_session, test_project):
    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    await _make_orchestrated_task(
        db_session,
        test_project.id,
        run.id,
        gate.id,
        developer.id,
        status="blocked",
    )

    result = await service.tick(db_session, run.id)

    assert result["recoveries_created"] == 1
    assert result["status"] == "blocked"
    tick_events = await _events(db_session, test_project.id, "orchestration.tick")
    assert tick_events[0].payload["recoveries_created"] == 1


@pytest.mark.asyncio
async def test_recover_run_escalates_review_fix_when_source_unresolvable(db_session, test_project):
    developer = _agent("developer", "developer", ["implementation"])
    reviewer = _agent("reviewer", "reviewer", ["review"])
    db_session.add_all([developer, reviewer])
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id, status="failed", failure_reason="Rejected evidence from review")
    await _add_gate_evidence(
        db_session,
        run.id,
        gate.id,
        source_type="review",
        source_id=uuid.uuid4(),
        producer_agent_id=reviewer.id,
        verdict="rejected",
        event_seq=30,
    )

    created = await service.recover_run(db_session, run.id, baseline_ready=True)
    ask_actions = await _actions(db_session, run.id, "ask_human")

    assert created == 1
    assert run.status == "blocked"
    assert goal.status == "blocked"
    assert run.active_blockers[0]["kind"] == "review_fix_required"
    assert len(ask_actions) == 1
    assert ask_actions[0].request["question"].startswith("Rejected review needs a fix")

    replay_created = await service.recover_run(db_session, run.id, baseline_ready=True)
    assert replay_created == 0


@pytest.mark.asyncio
async def test_execute_pause_run_action_rejects_non_tickable_run(db_session, test_project):
    service, _goal, run = await _make_run(db_session, test_project.id)
    # "paused" is still an active run status (passes reserve_action's guard) but is
    # not a tickable/decision status, so execute_pause_run_action's own check must
    # reject it and leave the reserved action failed rather than completed.
    run.status = "paused"
    await db_session.flush()

    with pytest.raises(HTTPException) as exc:
        await service.execute_pause_run_action(
            db_session,
            run_id=run.id,
            request={"action_type": "pause_run", "reason": "test"},
            idempotency_key=f"run:{run.id}:kind:pause_run:test",
        )
    assert exc.value.status_code == 409

    pause_actions = await _actions(db_session, run.id, "pause_run")
    assert pause_actions[0].status == "failed"
    assert run.status == "paused"


@pytest.mark.asyncio
async def test_execute_retry_task_action_fails_action_when_run_raises(db_session, test_project, monkeypatch):
    async def _raise(self, *a, **k):
        raise HTTPException(status_code=409, detail="boom")

    monkeypatch.setattr("huddleroom.services.orchestration_service.TaskService.run", _raise)

    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(
        db_session, test_project.id, run.id, gate.id, developer.id, status="failed"
    )

    with pytest.raises(HTTPException):
        await service.execute_retry_task_action(
            db_session,
            run_id=run.id,
            request={"action_type": "retry_task", "task_id": str(task.id)},
            idempotency_key=f"run:{run.id}:kind:retry_task:task:{task.id}",
        )

    retry_actions = await _actions(db_session, run.id, "retry_task")
    assert len(retry_actions) == 1
    assert retry_actions[0].status == "failed"


@pytest.mark.asyncio
async def test_recover_run_escalates_when_retry_execution_fails(db_session, test_project, monkeypatch):
    async def _raise(self, *a, **k):
        raise HTTPException(status_code=409, detail="retry boom")

    monkeypatch.setattr("huddleroom.services.orchestration_service.OrchestrationService.execute_retry_task_action", _raise)

    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, developer.id)
    await _add_failed_session(db_session, task, developer.id)

    created = await service.recover_run(db_session, run.id, baseline_ready=True)
    ask_actions = await _actions(db_session, run.id, "ask_human")

    assert created == 1
    assert goal.status == "blocked"
    assert run.status == "blocked"
    assert len(ask_actions) == 1
    assert ask_actions[0].request["question"].startswith("Task retry could not start")
    assert run.active_blockers[0]["kind"] == "retry_required"


@pytest.mark.asyncio
async def test_execute_reassign_task_action_restores_assignee_when_run_raises(
    db_session,
    test_project,
    monkeypatch,
):
    async def _raise(self, *a, **k):
        raise HTTPException(status_code=409, detail="reassign boom")

    monkeypatch.setattr("huddleroom.services.orchestration_service.TaskService.run", _raise)

    original = _agent("developer", "developer", ["implementation"])
    alternate = _agent("alternate", "developer", ["implementation"])
    db_session.add_all([original, alternate])
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, original.id)

    with pytest.raises(HTTPException) as exc:
        await service.execute_reassign_task_action(
            db_session,
            run_id=run.id,
            request={
                "action_type": "reassign_task",
                "task_id": str(task.id),
                "agent_id": str(alternate.id),
            },
            idempotency_key=f"run:{run.id}:kind:reassign_task:task:{task.id}:agent:{alternate.id}",
        )
    assert exc.value.status_code == 409

    reassign_actions = await _actions(db_session, run.id, "reassign_task")
    await db_session.refresh(task)
    assert len(reassign_actions) == 1
    assert reassign_actions[0].status == "failed"
    assert task.assigned_to == original.id
    assert task.status == "failed"
    assert await db_session.scalar(select(count(Session.id)).where(Session.task_id == task.id)) == 0


@pytest.mark.asyncio
async def test_execute_request_verification_action_replay_delegation_failure_fails_reserved_action(
    db_session,
    test_project,
    monkeypatch,
):
    async def _raise(self, *a, **k):
        raise HTTPException(status_code=409, detail="delegation boom")

    monkeypatch.setattr(
        "huddleroom.services.orchestration_service.OrchestrationService.execute_create_delegation_task_action",
        _raise,
    )

    developer = _agent("developer", "developer", ["implementation"])
    validator = _agent("validator", "validator", ["validation"])
    db_session.add_all([developer, validator])
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id, status="failed", failure_reason="Evidence is stale")
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, developer.id, status="done")
    await _add_gate_evidence(
        db_session,
        run.id,
        gate.id,
        source_type="task",
        source_id=task.id,
        producer_agent_id=developer.id,
        verdict="accepted",
        event_seq=20,
    )
    action = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key=f"run:{run.id}:kind:request_verification:stale_gate:{gate.id}",
        action_type="request_verification",
        request={
            "action_type": "request_verification",
            "gate_id": str(gate.id),
            "work_function": "validation",
        },
    )

    with pytest.raises(HTTPException) as exc:
        await service.execute_request_verification_action(
            db_session,
            run_id=run.id,
            request={
                "action_type": "request_verification",
                "gate_id": str(gate.id),
                "work_function": "validation",
            },
            idempotency_key=f"run:{run.id}:kind:request_verification:stale_gate:{gate.id}",
        )
    assert exc.value.status_code == 409
    assert exc.value.detail == "delegation boom"

    await db_session.refresh(action)
    assert action.status == "failed"
    assert action.error == "delegation boom"


@pytest.mark.asyncio
async def test_recover_run_does_not_persist_blocked_state_before_ask_human_reservation_for_blocked_task(
    db_session,
    test_project,
    monkeypatch,
):
    original_reserve_action = OrchestrationService.reserve_action

    async def _fail_ask_human_reservation(self, db, run_id, idempotency_key, action_type, request, decision_id=None):
        if action_type == "ask_human":
            raise HTTPException(status_code=409, detail="ask_human reserve boom")
        return await original_reserve_action(self, db, run_id, idempotency_key, action_type, request, decision_id)

    monkeypatch.setattr(OrchestrationService, "reserve_action", _fail_ask_human_reservation)

    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    await _make_orchestrated_task(
        db_session,
        test_project.id,
        run.id,
        gate.id,
        developer.id,
        status="blocked",
    )

    with pytest.raises(HTTPException) as exc:
        await service.recover_run(db_session, run.id, baseline_ready=True)
    assert exc.value.status_code == 409
    assert exc.value.detail == "ask_human reserve boom"

    await db_session.refresh(goal)
    await db_session.refresh(run)
    assert goal.status == "active"
    assert run.status == "running"
    assert run.active_blockers == []


@pytest.mark.asyncio
async def test_recover_run_does_not_persist_escalation_before_ask_human_reservation(
    db_session,
    test_project,
    monkeypatch,
):
    async def _raise_reassign(self, *a, **k):
        raise HTTPException(status_code=409, detail="reassign boom")

    original_reserve_action = OrchestrationService.reserve_action

    async def _fail_ask_human_reservation(self, db, run_id, idempotency_key, action_type, request, decision_id=None):
        if action_type == "ask_human":
            raise HTTPException(status_code=409, detail="ask_human reserve boom")
        return await original_reserve_action(self, db, run_id, idempotency_key, action_type, request, decision_id)

    monkeypatch.setattr(
        "huddleroom.services.orchestration_service.OrchestrationService.execute_reassign_task_action",
        _raise_reassign,
    )
    monkeypatch.setattr(OrchestrationService, "reserve_action", _fail_ask_human_reservation)

    original = _agent("developer", "developer", ["implementation"])
    alternate = _agent("alternate", "developer", ["implementation"])
    db_session.add_all([original, alternate])
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, original.id)
    await _add_failed_session(db_session, task, original.id)
    await _add_failed_session(db_session, task, original.id)

    with pytest.raises(HTTPException) as exc:
        await service.recover_run(db_session, run.id, baseline_ready=True)
    assert exc.value.status_code == 409
    assert exc.value.detail == "ask_human reserve boom"

    await db_session.refresh(goal)
    await db_session.refresh(run)
    await db_session.refresh(task)
    assert goal.status == "active"
    assert run.status == "running"
    assert run.active_blockers == []
    assert task.assigned_to == original.id


@pytest.mark.asyncio
async def test_recover_run_does_not_persist_verification_escalation_before_ask_human_reservation(
    db_session,
    test_project,
    monkeypatch,
):
    async def _raise_verification(self, *a, **k):
        # Must match the exact no-fit detail: Finding 3 restricts stale-gate
        # recovery's ask_human fallback to this specific 409, so any other
        # verification failure propagates instead of being misclassified.
        raise HTTPException(status_code=409, detail="No strong verification agent fit")

    original_reserve_action = OrchestrationService.reserve_action

    async def _fail_ask_human_reservation(self, db, run_id, idempotency_key, action_type, request, decision_id=None):
        if action_type == "ask_human":
            raise HTTPException(status_code=409, detail="ask_human reserve boom")
        return await original_reserve_action(self, db, run_id, idempotency_key, action_type, request, decision_id)

    monkeypatch.setattr(
        "huddleroom.services.orchestration_service.OrchestrationService.execute_request_verification_action",
        _raise_verification,
    )
    monkeypatch.setattr(OrchestrationService, "reserve_action", _fail_ask_human_reservation)

    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id, status="failed", failure_reason="Evidence is stale")

    with pytest.raises(HTTPException) as exc:
        await service.recover_run(db_session, run.id, baseline_ready=True)
    assert exc.value.status_code == 409
    assert exc.value.detail == "ask_human reserve boom"

    await db_session.refresh(goal)
    await db_session.refresh(run)
    await db_session.refresh(gate)
    assert goal.status == "active"
    assert run.status == "running"
    assert run.active_blockers == []


@pytest.mark.asyncio
async def test_recover_run_escalates_when_reassign_execution_fails(db_session, test_project, monkeypatch):
    async def _raise(self, *a, **k):
        raise HTTPException(status_code=409, detail="reassign boom")

    monkeypatch.setattr(
        "huddleroom.services.orchestration_service.OrchestrationService.execute_reassign_task_action",
        _raise,
    )

    original = _agent("developer", "developer", ["implementation"])
    alternate = _agent("developer", "developer", ["implementation"])
    db_session.add_all([original, alternate])
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, original.id)
    await _add_failed_session(db_session, task, original.id)
    await _add_failed_session(db_session, task, original.id)

    created = await service.recover_run(db_session, run.id, baseline_ready=True)
    ask_actions = await _actions(db_session, run.id, "ask_human")

    assert created == 1
    assert run.status == "blocked"
    assert len(ask_actions) == 1
    assert ask_actions[0].request["question"].startswith("Task failed after retry limit")
    assert run.active_blockers[0]["kind"] == "reassign_required"
