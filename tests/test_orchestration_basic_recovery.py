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
from huddleroom.services.orchestration_dependencies import task_gate_id
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
        metadata_={"attempt": {"effect_state": "started"}},
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


async def _blocked_setup(db_session, project, *, agent=None, reason=None, authorize=True):
    if agent is None:
        agent = _agent("developer", "developer", ["implementation"])
        db_session.add(agent)
        await db_session.flush()
    service, goal, run = await _make_run(db_session, project.id)
    if authorize:
        run.phase = "authorized"
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(
        db_session, project.id, run.id, gate.id, agent.id, status="blocked",
    )
    if reason:
        task.metadata_ = {**task.metadata_, "blocked": {"reason": reason, "at": datetime.now(timezone.utc).isoformat()}}
        await db_session.flush()
    return service, goal, run, task, agent


async def _task_waits(db_session, run_id):
    from huddleroom.models.orchestration import OrchestrationWait

    return list((await db_session.scalars(
        select(OrchestrationWait).where(OrchestrationWait.run_id == run_id).order_by(OrchestrationWait.created_at)
    )).all())


async def _assert_waiting(db_session, service, goal, run):
    created = await service.recover_run(db_session, run.id, baseline_ready=True)
    replay = await service.recover_run(db_session, run.id, baseline_ready=True)
    waits = await _task_waits(db_session, run.id)
    assert (created, replay) == (1, 0)
    assert len(waits) == 1 and waits[0].awaited_event["event_type"] == "task.status_changed"
    assert await _actions(db_session, run.id, "ask_human") == []
    assert goal.status == "active" and run.status == "running"
    return waits[0]


@pytest.mark.asyncio
async def test_blocked_task_owner_only_asks_human_with_stored_reason_once(db_session, test_project):
    service, goal, run, task, _ = await _blocked_setup(db_session, test_project, reason="Need the prod API key")

    created = await service.recover_run(db_session, run.id, baseline_ready=True)
    replay_created = await service.recover_run(db_session, run.id, baseline_ready=True)
    ask_actions = await _actions(db_session, run.id, "ask_human")

    assert (created, replay_created) == (1, 0)
    assert goal.status == "blocked" and run.status == "blocked"
    assert run.active_blockers[0]["kind"] == "task_blocked"
    assert run.active_blockers[0]["task_id"] == str(task.id)
    assert len(ask_actions) == 1
    assert ask_actions[0].request["question"] == (
        'Task "Implementation work" is blocked: Need the prod API key. '
        "What input or decision do you need to supply to unblock it?"
    )
    assert await _task_waits(db_session, run.id) == []


@pytest.mark.asyncio
async def test_blocked_task_without_reason_says_so(db_session, test_project):
    service, _goal, run, _task, _ = await _blocked_setup(db_session, test_project)

    await service.recover_run(db_session, run.id, baseline_ready=True)

    question = (await _actions(db_session, run.id, "ask_human"))[0].request["question"]
    assert "is blocked: no reason was recorded." in question


@pytest.mark.asyncio
async def test_blocked_task_with_unfinished_dependency_waits(db_session, test_project):
    service, goal, run, task, agent = await _blocked_setup(db_session, test_project, reason="needs upstream")
    dep = await _make_orchestrated_task(
        db_session, test_project.id, run.id, (await _make_gate(db_session, run.id)).id, agent.id, status="in_progress",
    )
    task.depends_on = [str(dep.id)]
    await db_session.flush()

    wait = await _assert_waiting(db_session, service, goal, run)

    assert str(dep.id) in wait.fallback["expected_result"]
    assert wait.owner["task_id"] == str(task.id)


@pytest.mark.asyncio
async def test_blocked_task_with_done_dependency_is_not_a_wait(db_session, test_project):
    service, _goal, run, task, agent = await _blocked_setup(db_session, test_project, reason="why")
    dep = await _make_orchestrated_task(
        db_session, test_project.id, run.id, (await _make_gate(db_session, run.id, status="accepted")).id, agent.id,
        status="done",
    )
    task.depends_on = [str(dep.id)]
    await db_session.flush()

    await service.recover_run(db_session, run.id, baseline_ready=True)

    assert await _task_waits(db_session, run.id) == []
    assert len(await _actions(db_session, run.id, "ask_human")) == 1


@pytest.mark.asyncio
async def test_blocked_task_with_done_dependency_and_open_gate_waits(db_session, test_project):
    service, _goal, run, task, agent = await _blocked_setup(db_session, test_project, reason="why")
    await _with_dependency(db_session, test_project, run, task, agent, status="done")

    await service.recover_run(db_session, run.id, baseline_ready=True)

    assert len(await _task_waits(db_session, run.id)) == 1
    assert await _actions(db_session, run.id, "ask_human") == []


@pytest.mark.asyncio
async def test_blocked_task_with_live_session_waits(db_session, test_project):
    service, goal, run, task, agent = await _blocked_setup(db_session, test_project, reason="provider busy")
    db_session.add(Session(
        agent_id=agent.id, task_id=task.id, project_id=task.project_id, adapter_type="api",
        status="running", input_context={}, metadata_={}, origin="auto",
    ))
    await db_session.flush()

    await _assert_waiting(db_session, service, goal, run)


@pytest.mark.asyncio
async def test_blocked_task_with_active_meeting_source_waits(db_session, test_project):
    from huddleroom.models.meeting import Meeting

    service, goal, run, task, _ = await _blocked_setup(db_session, test_project, reason="discussing")
    db_session.add(Meeting(
        project_id=test_project.id, title="Unblock", meeting_type="decision", status="active", source_task_id=task.id,
    ))
    await db_session.flush()

    await _assert_waiting(db_session, service, goal, run)


@pytest.mark.asyncio
async def test_blocked_task_with_no_alternate_agent_asks_human(db_session, test_project):
    original = _agent("original", "developer", ["implementation"], is_active=False)
    db_session.add(original)
    await db_session.flush()
    service, _goal, run, _task, _ = await _blocked_setup(db_session, test_project, agent=original, reason="no owner")

    await service.recover_run(db_session, run.id, baseline_ready=True)

    assert len(await _actions(db_session, run.id, "ask_human")) == 1
    assert await _actions(db_session, run.id, "reassign_task") == []


async def _with_dependency(db_session, project, run, task, agent, *, status="in_progress"):
    dep = await _make_orchestrated_task(
        db_session, project.id, run.id, (await _make_gate(db_session, run.id)).id, agent.id, status=status,
    )
    task.depends_on = [str(dep.id)]
    await db_session.flush()
    return dep


@pytest.mark.asyncio
async def test_blocked_task_with_live_dependency_rewaits_after_each_expiry_and_never_asks(db_session, test_project):
    service, goal, run, task, agent = await _blocked_setup(db_session, test_project, reason="stuck dependency")
    dep = await _with_dependency(db_session, test_project, run, task, agent)

    for expected_waits in (1, 2, 3):
        await service.recover_run(db_session, run.id, baseline_ready=True)
        waits = await _task_waits(db_session, run.id)
        assert len(waits) == expected_waits and goal.status == "active"
        assert waits[-1].awaited_event["matcher"] == {"task_id": str(dep.id)}
        waits[-1].status = "cleared"
        await db_session.flush()

    assert await _actions(db_session, run.id, "ask_human") == []
    assert run.status == "running"


@pytest.mark.asyncio
async def test_blocked_task_asks_owner_once_wait_bound_is_exceeded(db_session, test_project):
    from datetime import timedelta
    from huddleroom.config import settings

    service, _goal, run, task, agent = await _blocked_setup(db_session, test_project)
    await _with_dependency(db_session, test_project, run, task, agent)
    await service.recover_run(db_session, run.id, baseline_ready=True)
    wait = (await _task_waits(db_session, run.id))[0]
    wait.status = "cleared"
    wait.created_at = datetime.now(timezone.utc) - timedelta(seconds=settings.orchestration_wake_max_seconds * 24 + 60)
    await db_session.flush()

    await service.recover_run(db_session, run.id, baseline_ready=True)

    assert len(await _task_waits(db_session, run.id)) == 1
    assert len(await _actions(db_session, run.id, "ask_human")) == 1
    assert run.status == "blocked"


@pytest.mark.asyncio
async def test_blocked_task_with_cancelled_dependency_is_not_awaiting(db_session, test_project):
    service, _goal, run, task, agent = await _blocked_setup(db_session, test_project, reason="dep failed")
    await _with_dependency(db_session, test_project, run, task, agent, status="cancelled")

    await service.recover_run(db_session, run.id, baseline_ready=True)

    assert await _task_waits(db_session, run.id) == []
    assert len(await _actions(db_session, run.id, "ask_human")) == 1


@pytest.mark.asyncio
async def test_blocked_task_with_done_dependency_and_final_failed_gate_asks_owner_at_once(db_session, test_project):
    service, _goal, run, task, agent = await _blocked_setup(db_session, test_project, reason="dep gate dead")
    dep = await _with_dependency(db_session, test_project, run, task, agent, status="done")
    gate = await db_session.get(OrchestrationGate, task_gate_id(dep, run.id))
    gate.status = "failed"
    gate.failure_reason = "Task report has no evidence"
    await db_session.flush()

    await service.recover_run(db_session, run.id, baseline_ready=True)

    assert await _task_waits(db_session, run.id) == []
    assert len(await _actions(db_session, run.id, "ask_human")) == 1


@pytest.mark.asyncio
async def test_blocked_task_with_done_dependency_and_attempt_rejected_gate_still_waits(db_session, test_project):
    service, _goal, run, task, agent = await _blocked_setup(db_session, test_project, reason="dep retrying")
    dep = await _with_dependency(db_session, test_project, run, task, agent, status="done")
    gate = await db_session.get(OrchestrationGate, task_gate_id(dep, run.id))
    gate.status = "failed"
    gate.failure_reason = "Rejected evidence from task"
    await db_session.flush()

    await service.recover_run(db_session, run.id, baseline_ready=True)

    assert len(await _task_waits(db_session, run.id)) == 1
    assert await _actions(db_session, run.id, "ask_human") == []


@pytest.mark.asyncio
async def test_blocked_task_with_missing_dependency_is_not_awaiting(db_session, test_project):
    service, _goal, run, task, _agent = await _blocked_setup(db_session, test_project, reason="dep gone")
    task.depends_on = [str(uuid.uuid4())]
    await db_session.flush()

    await service.recover_run(db_session, run.id, baseline_ready=True)

    assert await _task_waits(db_session, run.id) == []
    assert len(await _actions(db_session, run.id, "ask_human")) == 1


@pytest.mark.asyncio
async def test_owner_only_and_dependency_blocked_tasks_yield_one_ask_and_one_wait(db_session, test_project):
    service, _goal, run, owner_task, agent = await _blocked_setup(db_session, test_project, reason="need a decision")
    waiting_task = await _make_orchestrated_task(
        db_session, test_project.id, run.id, (await _make_gate(db_session, run.id)).id, agent.id, status="blocked",
    )
    waiting_task.metadata_ = {**waiting_task.metadata_, "blocked": {"reason": "upstream", "at": datetime.now(timezone.utc).isoformat()}}
    dep = await _with_dependency(db_session, test_project, run, waiting_task, agent)

    await service.recover_run(db_session, run.id, baseline_ready=True)
    await service.recover_run(db_session, run.id, baseline_ready=True)

    waits = await _task_waits(db_session, run.id)
    asks = await _actions(db_session, run.id, "ask_human")
    assert len(asks) == 1 and "need a decision" in asks[0].request["question"]
    assert len(waits) == 1 and waits[0].owner["task_id"] == str(waiting_task.id)
    assert waits[0].awaited_event["matcher"] == {"task_id": str(dep.id)}
    assert run.status == "blocked"


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
async def test_recover_run_asks_the_owner_when_a_rejected_review_has_no_producer(db_session, test_project):
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
    # only the gate's revision subject is blocked; the run keeps dispatching other work
    assert run.status == "running" and goal.status == "active"
    assert [(b["kind"], b["category"], b["scope"]) for b in run.active_blockers] == [
        ("owner_escalation", "capability_unavailable", f"gate:{gate.id}:revision"),
    ]
    assert len(ask_actions) == 1
    assert "No producer task or assignee can revise" in ask_actions[0].request["question"]

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


def _fake_task_run(calls=None, *, fail_with=None, mark_in_progress=False):
    """Stand-in for TaskService.run: optionally flips status like the real run, then fails or creates a pending session."""

    async def _run(self, db, project_id, task_id, **kwargs):
        if calls is not None:
            calls.append(task_id)
        task = await db.get(Task, task_id)
        if mark_in_progress:
            task.status = "in_progress"
            await db.flush()
        if fail_with is not None:
            raise fail_with
        session = Session(
            agent_id=task.assigned_to,
            task_id=task.id,
            project_id=project_id,
            adapter_type="api",
            status="pending",
            input_context={},
            metadata_={},
            origin="manual",
        )
        db.add(session)
        await db.flush()
        return task, session.id

    return _run


async def _add_session_with_status(db_session, task: Task, agent_id, status: str) -> Session:
    session = Session(
        agent_id=agent_id,
        task_id=task.id,
        project_id=task.project_id,
        adapter_type="api",
        status=status,
        input_context={},
        metadata_={},
        origin="manual",
    )
    db_session.add(session)
    await db_session.flush()
    return session


@pytest.mark.asyncio
async def test_execute_reassign_blocked_task_preserves_description_metadata_and_sessions(
    db_session,
    test_project,
    monkeypatch,
):
    calls: list = []
    monkeypatch.setattr("huddleroom.services.orchestration_service.TaskService.run", _fake_task_run(calls))

    original = _agent("developer", "developer", ["implementation"])
    alternate = _agent("alternate", "developer", ["implementation"])
    db_session.add_all([original, alternate])
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(
        db_session, test_project.id, run.id, gate.id, original.id, status="blocked",
    )
    task.metadata_ = {**task.metadata_, "orchestration_contract": {"owner": "test"}}
    await db_session.flush()
    prior = await _add_failed_session(db_session, task, original.id)
    description_before = task.description
    plan_item_before = task.metadata_["orchestration_plan_item"]

    action = await service.execute_reassign_task_action(
        db_session,
        run_id=run.id,
        request={"action_type": "reassign_task", "task_id": str(task.id), "agent_id": str(alternate.id)},
        idempotency_key=f"run:{run.id}:kind:reassign_task:task:{task.id}:agent:{alternate.id}",
    )

    await db_session.refresh(task)
    await db_session.refresh(prior)
    assert action.status == "completed"
    assert calls == [task.id]
    assert task.assigned_to == alternate.id
    assert task.description == description_before
    assert task.metadata_["orchestration_plan_item"] == plan_item_before
    assert task.metadata_["orchestration_contract"] == {"owner": "test"}
    assert task.metadata_["orchestration"]["plan_item_gate_id"] == str(gate.id)
    assert prior.status == "failed"
    assert prior.task_id == task.id
    assert await db_session.scalar(select(count(Session.id)).where(Session.task_id == task.id)) == 2


@pytest.mark.asyncio
async def test_execute_retry_blocked_task_succeeds(db_session, test_project, monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        "huddleroom.services.orchestration_service.TaskService.run",
        _fake_task_run(calls, mark_in_progress=True),
    )

    agent = _agent("developer", "developer", ["implementation"])
    db_session.add(agent)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(
        db_session, test_project.id, run.id, gate.id, agent.id, status="blocked",
    )

    action = await service.execute_retry_task_action(
        db_session,
        run_id=run.id,
        request={"action_type": "retry_task", "task_id": str(task.id)},
        idempotency_key=f"run:{run.id}:kind:retry_task:task:{task.id}",
    )

    await db_session.refresh(task)
    assert action.status == "completed"
    assert calls == [task.id]
    assert task.assigned_to == agent.id
    assert task.status == "in_progress"


async def _add_latest_failed_session(
    db_session, task: Task, agent_id, *, error: str, metadata: dict, started_at=None,
) -> Session:
    session = Session(
        agent_id=agent_id,
        task_id=task.id,
        project_id=task.project_id,
        adapter_type="cli",
        status="failed",
        input_context={},
        error=error,
        metadata_=metadata,
        origin="auto",
        started_at=started_at,
    )
    db_session.add(session)
    await db_session.flush()
    return session


@pytest.mark.asyncio
async def test_execute_retry_refuses_when_agent_never_launched(db_session, test_project, monkeypatch):
    calls: list = []
    monkeypatch.setattr("huddleroom.services.orchestration_service.TaskService.run", _fake_task_run(calls))

    agent = _agent("developer", "developer", ["implementation"])
    db_session.add(agent)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, agent.id, status="failed")
    await _add_latest_failed_session(
        db_session, task, agent.id,
        error="orchestration_lineage_invalid",
        metadata={"attempt": {"effect_state": "not_started"}},
    )

    with pytest.raises(HTTPException) as exc:
        await service.execute_retry_task_action(
            db_session,
            run_id=run.id,
            request={"action_type": "retry_task", "task_id": str(task.id)},
            idempotency_key=f"run:{run.id}:kind:retry_task:task:{task.id}",
        )
    assert exc.value.status_code == 409
    assert "Not retryable" in exc.value.detail
    assert calls == []
    assert [a.status for a in await _actions(db_session, run.id, "retry_task")] == ["failed"]


@pytest.mark.asyncio
async def test_execute_retry_proceeds_after_agent_ran_and_failed(db_session, test_project, monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        "huddleroom.services.orchestration_service.TaskService.run",
        _fake_task_run(calls, mark_in_progress=True),
    )

    agent = _agent("developer", "developer", ["implementation"])
    db_session.add(agent)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, agent.id, status="failed")
    await _add_latest_failed_session(
        db_session, task, agent.id,
        error="exit 1",
        metadata={"attempt": {"effect_state": "started"}},
    )

    action = await service.execute_retry_task_action(
        db_session,
        run_id=run.id,
        request={"action_type": "retry_task", "task_id": str(task.id)},
        idempotency_key=f"run:{run.id}:kind:retry_task:task:{task.id}",
    )

    assert action.status == "completed"
    assert calls == [task.id]


@pytest.mark.asyncio
async def test_execute_retry_proceeds_after_manual_session_ran_and_failed(db_session, test_project, monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        "huddleroom.services.orchestration_service.TaskService.run",
        _fake_task_run(calls, mark_in_progress=True),
    )

    agent = _agent("developer", "developer", ["implementation"])
    db_session.add(agent)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, agent.id, status="failed")
    await _add_latest_failed_session(
        db_session, task, agent.id,
        error="exit 1", metadata={}, started_at=datetime.now(timezone.utc),
    )

    action = await service.execute_retry_task_action(
        db_session,
        run_id=run.id,
        request={"action_type": "retry_task", "task_id": str(task.id)},
        idempotency_key=f"run:{run.id}:kind:retry_task:task:{task.id}",
    )

    assert action.status == "completed"
    assert calls == [task.id]


@pytest.mark.asyncio
async def test_execute_retry_refuses_stale_pending_session_without_attempt_metadata(
    db_session, test_project, monkeypatch,
):
    calls: list = []
    monkeypatch.setattr("huddleroom.services.orchestration_service.TaskService.run", _fake_task_run(calls))

    agent = _agent("developer", "developer", ["implementation"])
    db_session.add(agent)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, agent.id, status="failed")
    await _add_latest_failed_session(db_session, task, agent.id, error="stale_pending_session", metadata={})

    with pytest.raises(HTTPException) as exc:
        await service.execute_retry_task_action(
            db_session,
            run_id=run.id,
            request={"action_type": "retry_task", "task_id": str(task.id)},
            idempotency_key=f"run:{run.id}:kind:retry_task:task:{task.id}",
        )
    assert exc.value.status_code == 409
    assert calls == []
    assert [a.status for a in await _actions(db_session, run.id, "retry_task")] == ["failed"]


@pytest.mark.asyncio
async def test_execute_retry_and_reassign_refuse_task_with_active_session(db_session, test_project, monkeypatch):
    calls: list = []
    monkeypatch.setattr("huddleroom.services.orchestration_service.TaskService.run", _fake_task_run(calls))

    original = _agent("developer", "developer", ["implementation"])
    alternate = _agent("alternate", "developer", ["implementation"])
    db_session.add_all([original, alternate])
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(
        db_session, test_project.id, run.id, gate.id, original.id, status="blocked",
    )
    await _add_session_with_status(db_session, task, original.id, "running")

    with pytest.raises(HTTPException) as retry_exc:
        await service.execute_retry_task_action(
            db_session,
            run_id=run.id,
            request={"action_type": "retry_task", "task_id": str(task.id)},
            idempotency_key=f"run:{run.id}:kind:retry_task:task:{task.id}",
        )
    assert retry_exc.value.status_code == 409

    with pytest.raises(HTTPException) as reassign_exc:
        await service.execute_reassign_task_action(
            db_session,
            run_id=run.id,
            request={"action_type": "reassign_task", "task_id": str(task.id), "agent_id": str(alternate.id)},
            idempotency_key=f"run:{run.id}:kind:reassign_task:task:{task.id}:agent:{alternate.id}",
        )
    assert reassign_exc.value.status_code == 409

    await db_session.refresh(task)
    assert calls == []
    assert task.status == "blocked"
    assert task.assigned_to == original.id
    assert [a.status for a in await _actions(db_session, run.id, "retry_task")] == ["failed"]
    assert [a.status for a in await _actions(db_session, run.id, "reassign_task")] == ["failed"]


@pytest.mark.asyncio
async def test_execute_reassign_run_failure_restores_blocked_status_and_agent(
    db_session,
    test_project,
    monkeypatch,
):
    monkeypatch.setattr(
        "huddleroom.services.orchestration_service.TaskService.run",
        _fake_task_run(fail_with=HTTPException(status_code=409, detail="reassign boom"), mark_in_progress=True),
    )

    original = _agent("developer", "developer", ["implementation"])
    alternate = _agent("alternate", "developer", ["implementation"])
    db_session.add_all([original, alternate])
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(
        db_session, test_project.id, run.id, gate.id, original.id, status="blocked",
    )

    with pytest.raises(HTTPException) as exc:
        await service.execute_reassign_task_action(
            db_session,
            run_id=run.id,
            request={"action_type": "reassign_task", "task_id": str(task.id), "agent_id": str(alternate.id)},
            idempotency_key=f"run:{run.id}:kind:reassign_task:task:{task.id}:agent:{alternate.id}",
        )
    assert exc.value.status_code == 409

    await db_session.refresh(task)
    reassign_actions = await _actions(db_session, run.id, "reassign_task")
    assert [a.status for a in reassign_actions] == ["failed"]
    assert task.status == "blocked"
    assert task.assigned_to == original.id


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["in_progress", "ready", "done"])
async def test_execute_retry_and_reassign_still_refuse_other_statuses(db_session, test_project, monkeypatch, status):
    calls: list = []
    monkeypatch.setattr("huddleroom.services.orchestration_service.TaskService.run", _fake_task_run(calls))

    original = _agent("developer", "developer", ["implementation"])
    alternate = _agent("alternate", "developer", ["implementation"])
    db_session.add_all([original, alternate])
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(
        db_session, test_project.id, run.id, gate.id, original.id, status=status,
    )

    with pytest.raises(HTTPException) as retry_exc:
        await service.execute_retry_task_action(
            db_session,
            run_id=run.id,
            request={"action_type": "retry_task", "task_id": str(task.id)},
            idempotency_key=f"run:{run.id}:kind:retry_task:task:{task.id}",
        )
    assert retry_exc.value.status_code == 409

    with pytest.raises(HTTPException) as reassign_exc:
        await service.execute_reassign_task_action(
            db_session,
            run_id=run.id,
            request={"action_type": "reassign_task", "task_id": str(task.id), "agent_id": str(alternate.id)},
            idempotency_key=f"run:{run.id}:kind:reassign_task:task:{task.id}:agent:{alternate.id}",
        )
    assert reassign_exc.value.status_code == 409
    assert calls == []


@pytest.mark.asyncio
async def test_retry_after_failed_first_attempt_creates_trigger_and_replay_is_idempotent(
    db_session,
    test_project,
    monkeypatch,
):
    calls: list = []
    monkeypatch.setattr(
        "huddleroom.services.orchestration_service.TaskService.run",
        _fake_task_run(calls, fail_with=HTTPException(status_code=409, detail="retry boom")),
    )

    agent = _agent("developer", "developer", ["implementation"])
    db_session.add(agent)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, agent.id, status="failed")
    first_key = f"run:{run.id}:kind:retry_task:task:{task.id}"

    with pytest.raises(HTTPException):
        await service.execute_retry_task_action(
            db_session,
            run_id=run.id,
            request={"action_type": "retry_task", "task_id": str(task.id)},
            idempotency_key=first_key,
        )

    # a second attempt after the failed first one is a corrective fix -> one recovery trigger, no retry.
    await service._retry_failed_task_once(db_session, goal, run, task)
    await service._retry_failed_task_once(db_session, goal, run, task)
    actions = await _actions(db_session, run.id, "retry_task")
    assert [a.idempotency_key for a in actions] == [first_key]
    assert actions[0].status == "failed"
    assert len(await _actions(db_session, run.id, "recovery_trigger")) == 1

    # Replaying the same attempt returns the same action without creating another.
    replay = await service.execute_retry_task_action(
        db_session,
        run_id=run.id,
        request=actions[0].request,
        idempotency_key=first_key,
    )
    assert replay.id == actions[0].id
    assert len(await _actions(db_session, run.id, "retry_task")) == 1


@pytest.mark.asyncio
async def test_reassign_after_failed_first_attempt_creates_one_trigger(
    db_session,
    test_project,
    monkeypatch,
):
    monkeypatch.setattr(
        "huddleroom.services.orchestration_service.TaskService.run",
        _fake_task_run(fail_with=HTTPException(status_code=409, detail="reassign boom")),
    )

    original = _agent("developer", "developer", ["implementation"])
    alternate = _agent("alternate", "developer", ["implementation"])
    db_session.add_all([original, alternate])
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(
        db_session, test_project.id, run.id, gate.id, original.id, status="failed",
    )

    await service._reassign_or_ask_human(db_session, goal, run, task)
    await service._reassign_or_ask_human(db_session, goal, run, task)

    # reassignment needs approval -> one idempotent recovery trigger, never a reassign row.
    assert await _actions(db_session, run.id, "reassign_task") == []
    assert len(await _actions(db_session, run.id, "recovery_trigger")) == 1
    assert task.assigned_to == original.id


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

    # with an alternate agent the reassignment is a recovery trigger (no ask); the owner question --
    # and this no-partial-persist guard -- remain for the no-alternate case.
    original = _agent("developer", "developer", ["implementation"])
    db_session.add(original)
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


# ---------------------------------------------------------------------------
# SPR 180/184: every corrective fix needs approval. Legacy recover_run
# paths create one idempotent recovery trigger instead of mutating.
# ---------------------------------------------------------------------------

_FINGERPRINTED_RUN_FIELDS = ("status", "phase", "plan_state", "retry_state")


def _run_fingerprint_fields(goal, run) -> dict:
    blockers = [b for b in (run.active_blockers or []) if b.get("kind") != "recovery_blocked"]
    return {"goal_status": goal.status, "blockers": blockers,
            **{field: getattr(run, field) for field in _FINGERPRINTED_RUN_FIELDS}}


async def _session_count(db_session, task) -> int:
    return await db_session.scalar(select(count(Session.id)).where(Session.task_id == task.id))


@pytest.mark.asyncio
async def test_legacy_recover_run_cannot_retry_failed_task_without_approval(db_session, test_project):
    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, developer.id)
    await _add_failed_session(db_session, task, developer.id)
    before = _run_fingerprint_fields(goal, run)

    await service.recover_run(db_session, run.id, baseline_ready=True)
    await service.recover_run(db_session, run.id, baseline_ready=True)  # replay is idempotent

    retry_actions = await _actions(db_session, run.id, "retry_task")
    recovery_triggers = await _actions(db_session, run.id, "recovery_trigger")
    await db_session.refresh(task)
    assert retry_actions == []
    assert len(recovery_triggers) == 1
    trigger = recovery_triggers[0].request
    assert (trigger["goal_id"], trigger["run_id"]) == (str(goal.id), str(run.id))
    assert trigger["retry_target"]["kind"] in {"action", "decision_loop"}
    assert task.status == "failed" and task.assigned_to == developer.id
    assert await _session_count(db_session, task) == 1
    assert _run_fingerprint_fields(goal, run) == before


@pytest.mark.asyncio
async def test_legacy_recover_run_cannot_reassign_failed_task_without_approval(db_session, test_project):
    original = _agent("developer", "developer", ["implementation"])
    alternate = _agent("alternate", "developer", ["implementation"])
    db_session.add_all([original, alternate])
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, original.id)
    await _add_failed_session(db_session, task, original.id)
    await _add_failed_session(db_session, task, original.id)
    before = _run_fingerprint_fields(goal, run)

    await service.recover_run(db_session, run.id, baseline_ready=True)
    await service.recover_run(db_session, run.id, baseline_ready=True)

    await db_session.refresh(task)
    assert await _actions(db_session, run.id, "reassign_task") == []
    assert await _actions(db_session, run.id, "ask_human") == []
    assert len(await _actions(db_session, run.id, "recovery_trigger")) == 1
    assert task.assigned_to == original.id and task.status == "failed"
    assert await _session_count(db_session, task) == 2
    assert _run_fingerprint_fields(goal, run) == before


@pytest.mark.asyncio
async def test_rejected_review_revises_the_producer_without_a_recovery_trigger(db_session, test_project):
    developer = _agent("developer", "developer", ["implementation"])
    reviewer = _agent("reviewer", "reviewer", ["review"])
    db_session.add_all([developer, reviewer])
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id, status="failed", failure_reason="Rejected evidence from review")
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, developer.id, status="done")
    review_session = Session(
        agent_id=reviewer.id, task_id=task.id, project_id=test_project.id, adapter_type="api", status="completed",
        input_context={}, output="changes requested", metadata_={}, origin="auto",
        started_at=datetime.now(timezone.utc), ended_at=datetime.now(timezone.utc),
    )
    db_session.add(review_session)
    await db_session.flush()
    await _add_gate_evidence(db_session, run.id, gate.id, source_type="review", source_id=review_session.id,
                             producer_agent_id=reviewer.id, verdict="rejected", event_seq=30)
    await _refresh_baseline_processes(db_session, goal, run)
    tasks_before = await db_session.scalar(select(count(Task.id)))

    await service.recover_run(db_session, run.id, baseline_ready=True)
    await service.recover_run(db_session, run.id, baseline_ready=True)

    [revision_action] = await _actions(db_session, run.id, "create_delegation_task")
    revision = await db_session.get(Task, revision_action.target_id)
    assert await db_session.scalar(select(count(Task.id))) == tasks_before + 1
    assert revision.assigned_to == developer.id and revision.parent_id == task.id
    assert await _actions(db_session, run.id, "recovery_trigger") == []
    assert gate.status == "failed"


@pytest.mark.asyncio
async def test_legacy_blocked_task_with_missing_agent_creates_trigger_not_reassignment(
    db_session, test_project, monkeypatch,
):
    calls: list = []
    monkeypatch.setattr("huddleroom.services.orchestration_service.TaskService.run", _fake_task_run(calls))
    original = _agent("original", "developer", ["implementation"], is_active=False)
    alternate = _agent("alternate", "developer", ["implementation"])
    db_session.add_all([original, alternate])
    await db_session.flush()
    service, _goal, run, task, _ = await _blocked_setup(db_session, test_project, agent=original, reason="no owner")

    await service.recover_run(db_session, run.id, baseline_ready=True)
    await service.recover_run(db_session, run.id, baseline_ready=True)

    await db_session.refresh(task)
    assert calls == []
    assert await _actions(db_session, run.id, "reassign_task") == []
    assert task.assigned_to == original.id and task.status == "blocked"
    assert len(await _actions(db_session, run.id, "recovery_trigger")) == 1


@pytest.mark.asyncio
async def test_legacy_blocked_task_with_failed_session_creates_trigger_not_retry(db_session, test_project, monkeypatch):
    calls: list = []
    monkeypatch.setattr("huddleroom.services.orchestration_service.TaskService.run", _fake_task_run(calls))
    service, _goal, run, task, agent = await _blocked_setup(db_session, test_project, reason="crashed")
    await _add_failed_session(db_session, task, agent.id)

    await service.recover_run(db_session, run.id, baseline_ready=True)
    await service.recover_run(db_session, run.id, baseline_ready=True)

    assert calls == []
    assert await _actions(db_session, run.id, "retry_task") == []
    assert await _session_count(db_session, task) == 1
    assert len(await _actions(db_session, run.id, "recovery_trigger")) == 1


@pytest.mark.asyncio
async def test_failed_recovery_remediation_task_never_starts_a_recursive_lineage(db_session, test_project):
    """A task produced by an approved recovery effect fails inside its own lineage: no new trigger, no retry."""
    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, developer.id)
    effect = OrchestrationAction(
        run_id=run.id, idempotency_key=f"recovery:{uuid.uuid4()}:apply:call-1:effect",
        action_type="create_delegation_task", request={}, status="completed", target_type="task", target_id=task.id,
    )
    db_session.add(effect)
    await db_session.flush()
    task.metadata_ = {**task.metadata_, "orchestration": {**task.metadata_["orchestration"], "action_id": str(effect.id)}}
    await _add_failed_session(db_session, task, developer.id)
    await db_session.flush()

    await service.recover_run(db_session, run.id, baseline_ready=True)

    assert await _actions(db_session, run.id, "retry_task") == []
    assert await _actions(db_session, run.id, "reassign_task") == []
    assert await _actions(db_session, run.id, "recovery_trigger") == []
