import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun, OrchestrationWait
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_supervision import OrchestrationSupervisionService, SupervisionAssessment

pytestmark = pytest.mark.asyncio

TASK_A, TASK_B = str(uuid.uuid4()), str(uuid.uuid4())


async def _run(db, project):
    goal = OrchestrationGoal(project_id=project.id, objective="Keep moving", status="active", goal_type="outcome")
    db.add(goal)
    await db.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized")
    db.add(run)
    await db.flush()
    return goal, run


def _assessment(**disposition):
    return SupervisionAssessment(disposition={
        "action_type": "continue", "origin": "test", "reason": "Why", "expected_result": "Expected", "contract_version": "start",
        **disposition,
    })


def _wake_when():
    return {
        "events": [
            {"event_type": "task.status_changed", "matcher": {"task_id": TASK_A}},
            {"event_type": "task.status_changed", "matcher": {"task_id": TASK_B}},
        ],
        "expected_result": "Task moved on.",
    }


async def _waits(db, run):
    await db.flush()
    return list((await db.scalars(select(OrchestrationWait).where(OrchestrationWait.run_id == run.id))).all())


async def test_continue_with_wake_when_creates_grouped_waits_owned_by_action_id(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    service = OrchestrationService()
    action = await service.supervision.apply_disposition(db_session, goal, run, _assessment(request={"wake_when": _wake_when()}))
    waits = await _waits(db_session, run)
    assert len(waits) == 2
    assert all(w.owner["id"] == str(action.id) for w in waits)
    assert sorted(w.wait_key for w in waits) == sorted(
        f"run:{run.id}:wait:decision:{action.id}:event:{n}" for n in range(2)
    )


async def test_continue_without_wake_when_creates_backstop_wait(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    action = await OrchestrationService().supervision.apply_disposition(db_session, goal, run, _assessment())
    waits = await _waits(db_session, run)
    assert len(waits) == 1
    assert waits[0].wait_key == f"run:{run.id}:wait:decision:{action.id}:recheck"
    assert waits[0].owner["id"] == str(action.id)


async def test_continue_replay_does_not_duplicate_waits(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    service = OrchestrationService()
    first = await service.supervision.apply_disposition(db_session, goal, run, _assessment(request={"wake_when": _wake_when()}))
    second = await service.supervision.apply_disposition(db_session, goal, run, _assessment(request={"wake_when": _wake_when()}))
    assert first.id == second.id
    assert len(await _waits(db_session, run)) == 2


async def test_continue_wait_conflict_is_swallowed(db_session, test_project, monkeypatch):
    goal, run = await _run(db_session, test_project)
    service = OrchestrationService()

    async def conflict(*_args, **_kwargs):
        raise HTTPException(status_code=409, detail="wait replay conflicts with existing contract")
    monkeypatch.setattr(OrchestrationSupervisionService, "create_wait", conflict)
    action = await service.supervision.apply_disposition(db_session, goal, run, _assessment(request={"wake_when": _wake_when()}))
    assert action.action_type == "noop" and action.status == "completed"
    assert await _waits(db_session, run) == []


async def test_non_continue_dispositions_create_no_waits(db_session, test_project, monkeypatch):
    goal, run = await _run(db_session, test_project)
    service = OrchestrationService()
    calls = []

    async def spy(*_args, **_kwargs):
        calls.append(1)
    monkeypatch.setattr(OrchestrationSupervisionService, "create_orchestrator_waits", spy)
    await service.supervision.apply_disposition(db_session, goal, run, _assessment(
        action_type="attention", request={"wake_when": _wake_when()},
    ))
    assert calls == []
    assert await _waits(db_session, run) == []


async def test_continue_on_blocked_run_records_action_but_no_waits(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    goal.status, run.status = "blocked", "blocked"
    await db_session.flush()
    action = await OrchestrationService().supervision.apply_disposition(db_session, goal, run, _assessment(request={"wake_when": _wake_when()}))
    assert action.action_type == "noop" and action.status == "completed"
    assert await _waits(db_session, run) == []


async def test_repeated_continue_after_clear_leaves_open_wait_and_reconcile_waits(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    service = OrchestrationService()
    first = await service.supervision.apply_disposition(db_session, goal, run, _assessment(request={"wake_when": _wake_when()}))
    assert await service.supervision.clear_matching_waits(
        db_session, run, event_type="task.status_changed", matcher={"task_id": TASK_A}) == 2
    second = await service.supervision.apply_disposition(db_session, goal, run, _assessment(request={"wake_when": _wake_when()}))
    assert first.id == second.id
    waits = await _waits(db_session, run)
    assert sorted(w.status for w in waits) == ["cleared", "cleared", "open", "open"]
    result = await service.supervision.reconcile_local(db_session, goal, run)
    assert result["outcome"] == "waiting"
