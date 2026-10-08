from datetime import datetime, timedelta, timezone
import uuid

import pytest
from sqlalchemy import select

from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun, OrchestrationWait
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_wake_when import (
    BACKSTOP_RECHECK_SECONDS,
    ORCHESTRATOR_WAIT_OWNER_TYPE,
    WAKE_RECHECK_EVENT_TYPE,
    clamp_recheck_seconds,
)


pytestmark = pytest.mark.asyncio


async def _run(db, project, *, goal_type="outcome"):
    goal = OrchestrationGoal(project_id=project.id, objective="Keep moving", status="active", goal_type=goal_type)
    db.add(goal)
    await db.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized")
    db.add(run)
    await db.flush()
    return run


async def _waits(db, run):
    return list((await db.scalars(select(OrchestrationWait).where(OrchestrationWait.run_id == run.id))).all())


def _recheck_seconds(wait):
    return (wait.due_recheck_at - wait.created_at).total_seconds()


async def _decide(db, run, parsed, *, llm_output=None):
    service = OrchestrationService()
    decision = await service.record_validated_decision(
        db,
        run_id=run.id,
        input_snapshot={"test": str(uuid.uuid4())},
        llm_output=llm_output or {},
        parsed_decision=parsed,
    )
    return service, decision


async def _noop_with_wake_when(db, run, wake_when):
    return await _decide(db, run, {"action_type": "noop", "reason": "Waiting.", "wake_when": wake_when})


def _wake_when(**extra):
    return {
        "events": [{"event_type": "task.status_changed", "matcher": {"task_id": str(uuid.uuid4())}}],
        "expected_result": "Task moved on.",
        **extra,
    }


async def test_accepted_noop_with_wake_when_creates_waits_owned_by_decision(db_session, test_project):
    run = await _run(db_session, test_project)
    service, decision = await _noop_with_wake_when(db_session, run, _wake_when())
    assert decision.validator_status == "accepted"

    action = await service._dispatch_execution_decision(db_session, run, decision)

    assert action.action_type == "noop"
    waits = await _waits(db_session, run)
    assert len(waits) == 1
    assert waits[0].wait_key == f"run:{run.id}:wait:decision:{decision.id}:event:0"
    assert waits[0].owner == {"type": ORCHESTRATOR_WAIT_OWNER_TYPE, "id": str(decision.id)}


async def test_noop_without_wake_when_creates_backstop_wait(db_session, test_project):
    run = await _run(db_session, test_project)
    service, decision = await _decide(db_session, run, {"action_type": "noop", "reason": "Waiting."})

    await service._dispatch_execution_decision(db_session, run, decision)

    waits = await _waits(db_session, run)
    assert len(waits) == 1
    assert waits[0].awaited_event["event_type"] == WAKE_RECHECK_EVENT_TYPE
    assert _recheck_seconds(waits[0]) == pytest.approx(clamp_recheck_seconds(BACKSTOP_RECHECK_SECONDS), abs=5)


async def test_noop_with_invalid_wake_when_shape_creates_backstop(db_session, test_project):
    run = await _run(db_session, test_project)
    service, decision = await _noop_with_wake_when(db_session, run, {"events": "not-a-list"})

    await service._dispatch_execution_decision(db_session, run, decision)

    waits = await _waits(db_session, run)
    assert len(waits) == 1
    assert waits[0].awaited_event["event_type"] == WAKE_RECHECK_EVENT_TYPE
    assert _recheck_seconds(waits[0]) == pytest.approx(clamp_recheck_seconds(BACKSTOP_RECHECK_SECONDS), abs=5)


async def test_rejected_decision_fallback_noop_creates_backstop_wait(db_session, test_project):
    run = await _run(db_session, test_project)
    service, decision = await _decide(db_session, run, {"action_type": "not_a_real_action", "reason": "x"})
    assert decision.validator_status == "rejected"

    action = await service._dispatch_execution_decision(db_session, run, decision)

    assert action.action_type == "noop"
    waits = await _waits(db_session, run)
    assert len(waits) == 1
    assert waits[0].owner["id"] == str(decision.id)
    assert _recheck_seconds(waits[0]) == pytest.approx(clamp_recheck_seconds(BACKSTOP_RECHECK_SECONDS), abs=5)


async def test_invalid_llm_output_creates_backstop_wait(db_session, test_project):
    run = await _run(db_session, test_project)
    service, decision = await _decide(db_session, run, {}, llm_output="not json")
    assert decision.validator_status == "rejected"

    action = await service._dispatch_execution_decision(db_session, run, decision)

    assert action.action_type == "noop"
    waits = await _waits(db_session, run)
    assert len(waits) == 1
    assert waits[0].owner["id"] == str(decision.id)


# Dispatch itself still creates no waits; the post-action wait lives in _act_until_wait.
async def test_non_noop_dispatch_creates_no_waits(db_session, test_project):
    run = await _run(db_session, test_project)
    service, decision = await _decide(db_session, run, {
        "action_type": "ask_human", "question": "Choose scope.", "reason": "Need input.",
    })
    assert decision.validator_status == "accepted"

    action = await service._dispatch_execution_decision(db_session, run, decision)

    assert action.action_type == "ask_human"
    assert await _waits(db_session, run) == []


async def test_stub_decision_wake_when_is_renormalized_and_clamped(db_session, test_project):
    run = await _run(db_session, test_project)
    service, decision = await _noop_with_wake_when(
        db_session, run, _wake_when(recheck_after_seconds=10 ** 9),
    )

    await service._dispatch_execution_decision(db_session, run, decision)

    waits = await _waits(db_session, run)
    assert waits
    assert all(_recheck_seconds(wait) == pytest.approx(clamp_recheck_seconds(10 ** 9), abs=5) for wait in waits)


# Plan R7: roadmap accepted plans rely on decision/context reuse, not waits.
async def test_roadmap_accepted_plan_noop_creates_no_waits(db_session, test_project):
    run = await _run(db_session, test_project, goal_type="roadmap")
    run.plan_state = {"status": "accepted"}
    await db_session.flush()
    service, decision = await _noop_with_wake_when(db_session, run, _wake_when())

    await service._dispatch_execution_decision(db_session, run, decision)

    assert await _waits(db_session, run) == []


async def test_replayed_dispatch_does_not_duplicate_waits(db_session, test_project):
    run = await _run(db_session, test_project)
    service, decision = await _noop_with_wake_when(db_session, run, _wake_when())

    await service._dispatch_execution_decision(db_session, run, decision)
    first = await _waits(db_session, run)
    await service._dispatch_execution_decision(db_session, run, decision)

    assert len(await _waits(db_session, run)) == len(first) == 1


# --- post-action wait after act-until-wait ends on a non-noop ---
async def _orch_waits(db, run):
    return [w for w in await _waits(db, run) if w.owner["type"] == ORCHESTRATOR_WAIT_OWNER_TYPE]


async def _act(db, project, action_type, *, step="next_action", goal_type="outcome", accepted=True, pre_wait=False, system_wait=True):
    from huddleroom.models.orchestration import OrchestrationAction

    run = await _run(db, project, goal_type=goal_type)
    if accepted:
        run.plan_state = {"status": "accepted"}
    action = OrchestrationAction(
        run_id=run.id, idempotency_key=f"k:{uuid.uuid4()}", action_type=action_type, status="completed",
    )
    db.add(action)
    await db.flush()
    service = OrchestrationService()
    if system_wait:
        db.add(OrchestrationWait(
            run_id=run.id, wait_key=f"run:{run.id}:wait:task:{uuid.uuid4()}",
            owner={"type": "task", "id": str(uuid.uuid4())},
            awaited_event={"event_type": "task.status_changed", "matcher": {}},
            due_recheck_at=datetime.now(timezone.utc) + timedelta(hours=1),
            fallback={"action_type": "continue"}, status="open",
        ))
        await db.flush()
    if pre_wait:
        await service.supervision.create_orchestrator_waits(db, run, owner_id=uuid.uuid4(), wake_when=None)

    async def once(_db, _goal, _run):
        return {"step": step, "action_id": str(action.id)}

    goal = await db.get(OrchestrationGoal, run.goal_id)
    await service._act_until_wait(db, goal, run, once, loop_steps=frozenset({"plan_decision", "next_action"}))
    return run, action


@pytest.mark.parametrize("step", ["next_action", "plan_decision"])
async def test_act_until_wait_non_noop_end_creates_post_action_wait(db_session, test_project, step):
    run, action = await _act(db_session, test_project, "create_delegation_task", step=step)
    waits = await _orch_waits(db_session, run)
    assert len(waits) == 1
    assert waits[0].status == "open"
    assert waits[0].owner == {"type": ORCHESTRATOR_WAIT_OWNER_TYPE, "id": str(action.id)}
    assert "follow_up_ids" in waits[0].fallback


async def test_act_until_wait_noop_end_creates_no_extra_wait(db_session, test_project):
    run, _ = await _act(db_session, test_project, "noop")
    assert await _orch_waits(db_session, run) == []


async def test_act_until_wait_stop_actions_create_no_post_wait(db_session, test_project):
    run, _ = await _act(db_session, test_project, "ask_human")
    assert await _orch_waits(db_session, run) == []


async def test_act_until_wait_roadmap_accepted_creates_no_post_wait(db_session, test_project):
    run, _ = await _act(db_session, test_project, "accept_plan", goal_type="roadmap")
    assert await _orch_waits(db_session, run) == []


async def test_act_until_wait_skips_when_orchestrator_wait_exists(db_session, test_project):
    run, action = await _act(db_session, test_project, "create_delegation_task", pre_wait=True)
    waits = await _orch_waits(db_session, run)
    assert waits
    assert all(w.owner["id"] != str(action.id) for w in waits)


async def test_act_until_wait_unaccepted_plan_creates_no_post_wait(db_session, test_project):
    run, _ = await _act(db_session, test_project, "create_delegation_task", step="plan_decision", accepted=False)
    assert await _orch_waits(db_session, run) == []


async def test_act_until_wait_no_system_waits_creates_no_post_wait(db_session, test_project):
    run, _ = await _act(db_session, test_project, "create_delegation_task", system_wait=False)
    assert await _orch_waits(db_session, run) == []


async def test_act_until_wait_with_open_system_wait_creates_post_action_wait(db_session, test_project):
    run, action = await _act(db_session, test_project, "create_delegation_task", system_wait=True)
    waits = await _orch_waits(db_session, run)
    assert len(waits) == 1 and waits[0].owner["id"] == str(action.id)


async def test_act_until_wait_cap_reached_creates_no_post_wait(db_session, test_project, monkeypatch):
    from huddleroom.config import settings
    from huddleroom.models.orchestration import OrchestrationAction

    monkeypatch.setattr(settings, "orchestration_max_actions_per_tick", 3)
    run = await _run(db_session, test_project)
    run.plan_state = {"status": "accepted"}
    actions = [
        OrchestrationAction(run_id=run.id, idempotency_key=f"k:{uuid.uuid4()}", action_type="create_delegation_task", status="completed")
        for _ in range(3)
    ]
    db_session.add_all(actions)
    db_session.add(OrchestrationWait(
        run_id=run.id, wait_key=f"run:{run.id}:wait:task:{uuid.uuid4()}",
        owner={"type": "task", "id": str(uuid.uuid4())},
        awaited_event={"event_type": "task.status_changed", "matcher": {}},
        due_recheck_at=datetime.now(timezone.utc) + timedelta(hours=1),
        fallback={"action_type": "continue"}, status="open",
    ))
    await db_session.flush()
    queue = iter(actions)

    async def once(_db, _goal, _run):
        return {"step": "next_action", "action_id": str(next(queue).id)}

    goal = await db_session.get(OrchestrationGoal, run.goal_id)
    service = OrchestrationService()
    await service._act_until_wait(db_session, goal, run, once, loop_steps=frozenset({"next_action"}))
    assert await _orch_waits(db_session, run) == []
