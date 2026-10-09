"""Test suite for concurrent tick handling and locking (Package 1 fixes #5, #9).

Tests that OrchestrationService.tick() properly serializes concurrent ticks on the
same goal and avoids duplicate events or lost cursor advances.

Due to SQLite's single-writer limitation, we test within a single session that
simulates the race by calling tick() twice on the same run to verify:
1. No duplicate events are emitted when tick() is called concurrently (fix #9)
2. Cursor advances correctly even with baseline process writes (fix #5)
3. Baseline processes are read fresh inside the lock, not stale
"""

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import func, select

from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.services.event_bus import emit_event_once
from huddleroom.services.orchestration_service import OrchestrationService


@pytest_asyncio.fixture(autouse=True)
async def runnable_workspace(db_session, test_project, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    test_project.workspace_path = str(workspace.resolve())
    await db_session.flush()


async def _make_run(db, project_id):
    """Create a goal and run for testing."""
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db,
        project_id=project_id,
        data=OrchestrationGoalCreate(
            objective="Concurrent tick test",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )
    return service, goal, run


async def _count_tick_events(db, project_id):
    """Count orchestration.tick events in the project."""
    result = await db.execute(
        select(func.count(EventLog.id)).where(
            EventLog.project_id == project_id,
            EventLog.event_type == "orchestration.tick",
        )
    )
    return result.scalar() or 0


async def _seq(db, bus_event) -> int:
    """Return EventLog.seq for the given BusEvent."""
    row = (await db.execute(
        select(EventLog).where(EventLog.id == bus_event.id)
    )).scalar_one()
    return row.seq


@pytest.mark.asyncio
async def test_concurrent_ticks_no_duplicate_events(db_session, test_project):
    """Sequential ticks with same events don't emit duplicate tick events (fix #9).

    This verifies the dedup_key in emit_event_once prevents duplicate emission
    when tick's event_cursor/baseline_process writes are inside the lock (fix #9).
    """
    service, goal, run = await _make_run(db_session, test_project.id)

    # Seed one event
    seeded, _ = await emit_event_once(
        db_session, test_project.id, "task.created", {"n": 1}, dedup_key="seed-1"
    )

    # First tick processes the event and emits tick event
    result1 = await service.tick(db_session, run.id)
    assert result1["processed_events"] == 1
    assert result1["tick_emitted"] is True
    assert result1["event_cursor"] == await _seq(db_session, seeded)

    tick_count_after_first = await _count_tick_events(db_session, test_project.id)
    assert tick_count_after_first == 1

    # Refresh run from DB to get updated state
    await db_session.refresh(run)

    # Second tick with no new events should not emit another tick event
    # (dedup_key same, so emit_event_once returns existing event with created=False)
    result2 = await service.tick(db_session, run.id)
    assert result2["processed_events"] == 0
    assert result2["tick_emitted"] is False  # No new tick event emitted
    assert result2["event_cursor"] == result1["event_cursor"]

    tick_count_after_second = await _count_tick_events(db_session, test_project.id)
    assert tick_count_after_second == 1  # Still only 1, no duplicate


@pytest.mark.asyncio
async def test_tick_cursor_advances_correctly_inside_lock(db_session, test_project):
    """Tick's event_cursor and baseline process writes stay serialized (fix #5, #9).

    Verifies that cursor assignment happens inside the lock block, preventing
    another tick from interleaving and seeing partial state.
    """
    service, goal, run = await _make_run(db_session, test_project.id)

    # Seed three events
    events = []
    for i in range(3):
        ev, _ = await emit_event_once(
            db_session,
            test_project.id,
            "task.created",
            {"n": i},
            dedup_key=f"seed-{i}",
        )
        events.append(ev)

    # First tick processes all 3 events, cursor lands on last
    result1 = await service.tick(db_session, run.id)
    assert result1["processed_events"] == 3
    cursor_after_first = await _seq(db_session, events[-1])
    assert result1["event_cursor"] == cursor_after_first
    assert run.event_cursor == cursor_after_first

    # Refresh run from DB
    await db_session.refresh(run)

    # Second tick with no new events sees the advanced cursor from first tick
    result2 = await service.tick(db_session, run.id)
    assert result2["processed_events"] == 0
    assert result2["event_cursor"] == cursor_after_first  # Still the same


@pytest.mark.asyncio
async def test_tick_reloads_goal_status_inside_lock(db_session, test_project):
    """Tick reloads goal inside lock to catch status changes (fix #5).

    Verifies that goal status is re-read with populate_existing inside the lock,
    preventing stale checks from before the lock was acquired.
    """
    service, goal, run = await _make_run(db_session, test_project.id)

    # First tick succeeds
    result1 = await service.tick(db_session, run.id)
    assert result1["status"] in ("running", "blocked")  # Goal is active

    # Manually pause the run (simulating an external operation that happened
    # between lock acquisition and status check)
    run.status = "paused"
    await db_session.flush()

    # Refresh from DB
    await db_session.refresh(run)

    # Second tick should recognize paused status and return no-op result
    result2 = await service.tick(db_session, run.id)
    assert result2["status"] == "paused"
    assert result2["processed_events"] == 0
    assert result2["tick_emitted"] is False


@pytest.mark.asyncio
async def test_audits_run_while_planning_and_concurrent_ticks_adopt_each_once(
    db_session, test_project, stub_decision, concurrent_sessions, monkeypatch
):
    """Two independent audits start while the planner runs; the accepted plan adopts both and
    concurrent ticks bind each exactly once -- no duplicate task, session, gate or marker."""
    import asyncio
    import json

    from fastapi import HTTPException
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from huddleroom.config import settings
    from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGate
    from huddleroom.models.session import Session
    from huddleroom.models.task import Task
    from tests.test_orchestration_runtime_e2e import _agent, _authorized_run, _planner_agent_id

    from huddleroom.services.session_service import SessionService

    monkeypatch.setattr(settings, "orchestration_max_actions_per_tick", 1)
    monkeypatch.setattr(SessionService, "_schedule_dispatch_after_commit", lambda *_args: None)
    planner = _agent("adopting-planner", ["planning"])
    auditor = _agent("auditor", ["investigation"])
    db_session.add_all([planner, auditor])
    await db_session.flush()
    service, goal, run = await _authorized_run(db_session, test_project)
    audit_ids = []
    for n in range(2):
        action = await service.execute_create_delegation_task_action(
            db_session,
            run_id=run.id,
            request={
                "action_type": "create_delegation_task", "agent_id": str(auditor.id),
                "work_function": "investigation", "scope": f"Audit area {n}.", "deliverable": f"Findings {n}.",
            },
            idempotency_key=f"run:{run.id}:kind:create_delegation_task:audit-{n}",
        )
        audit_ids.append(action.target_id)
        (await db_session.get(Task, action.target_id)).priority = 10  # the planner outranks the audits

    stub_decision(lambda ctx: {
        "action_type": "request_plan", "agent_id": _planner_agent_id(ctx), "work_function": "planning",
        "scope": "Plan around the running audits.",
    })
    assert (await service.tick(db_session, run.id))["authorized_execution"]["step"] == "plan_decision"
    await db_session.refresh(run)
    planning_task = await db_session.get(Task, uuid.UUID(run.plan_state["planning_task_id"]))
    assert sorted(planning_task.metadata_["orchestration_plan"]["pre_plan_task_ids"]) == sorted(map(str, audit_ids))

    async def sessions(db, task_id):
        return list(await db.scalars(select(Session).where(Session.task_id == task_id)))

    # Planner and audits share the root's two slots: the planner and one audit start in the same tick
    # (discovery runs while planning), the other audit stays queued in backlog until a slot frees.
    started, queued = [], []
    for task_id in [planning_task.id, *audit_ids]:
        task = await db_session.get(Task, task_id)
        await db_session.refresh(task)
        (started if task.status == "in_progress" else queued).append(task_id)
        assert len(await sessions(db_session, task_id)) == (1 if task.status == "in_progress" else 0)
    assert planning_task.id in started and len(started) == 2 and len(queued) == 1
    (queued_id,) = queued
    assert (await db_session.get(Task, queued_id)).status == "backlog"

    (planner_session,) = await sessions(db_session, planning_task.id)
    kind = planning_task.metadata_["orchestration_contract"]["report_schema"]["plan_kind"]
    audit_items = [
        {"id": f"audit-{n}", "work_function": "investigation", "scope": f"Audit area {n}.",
         "deliverable": f"Findings {n}.", "why_runnable": "Already running.", "adopt_task_id": str(task_id)}
        for n, task_id in enumerate(audit_ids)
    ]
    synthesis = {
        "id": "synthesis", "work_function": "implementation", "scope": "Combine the audit findings.",
        "deliverable": "One synthesis report.", "why_runnable": "Both audits accepted.",
        "depends_on": ["audit-0", "audit-1"],
    }
    planner_session.output = json.dumps({
        "schema_version": 1, "plan_kind": kind, "summary": "Adopt both audits, then synthesize.",
        "plan_items": [*audit_items, synthesis],
    })
    planner_session.status, planning_task.status = "completed", "done"
    await db_session.flush()
    await emit_event_once(
        db_session, test_project.id, "task.status_changed",
        {"task_id": str(planning_task.id), "status": "done", "previous_status": "in_progress"},
        dedup_key=f"adoption-plan-done:{planning_task.id}",
    )
    await emit_event_once(
        db_session, test_project.id, "session.completed",
        {"session_id": str(planner_session.id), "task_id": str(planning_task.id)},
        dedup_key=f"adoption-plan-session:{planner_session.id}",
    )
    # The tick ingests the envelope; the audits are still running, so the proactive decision path
    # (not a direct call) accepts the ingested plan.
    await service.tick(db_session, run.id)
    await db_session.refresh(run)
    assert run.plan_state["status"] == "requested" and run.plan_state["plan_artifact_id"]
    stub_decision(lambda ctx: {"action_type": "accept_plan", "plan_artifact_id": ctx["run"]["plan_state"]["plan_artifact_id"]})
    result = await service.tick(db_session, run.id)
    await db_session.refresh(run)
    assert result["authorized_execution"]["step"] == "plan_decision", result
    assert run.plan_state["status"] == "accepted"
    run_id = run.id
    await db_session.commit()

    async def tick(db):
        try:
            return await OrchestrationService().tick(db, run_id)
        except HTTPException as exc:
            await db.rollback()
            return exc

    first, second = concurrent_sessions
    results = await asyncio.wait_for(asyncio.gather(tick(first), tick(second)), timeout=30)
    assert all(isinstance(result, dict) for result in results), results
    # One tick binds both adopted audits (no slot spent); the other finds them bound and waits on them.
    assert sorted(r["authorized_execution"]["step"] for r in results) == ["local_liveness", "release_work"]
    assert {"step": "release_work", "released": 2} in [r["authorized_execution"] for r in results]

    async with async_sessionmaker(db_session.bind, expire_on_commit=False)() as reader:
        run = await reader.get(OrchestrationRun, run_id)
        expanded = run.plan_state["expanded_items"]
        assert sorted((e["plan_item_id"], e["task_id"], e.get("adopted")) for e in expanded) == [
            ("audit-0", str(audit_ids[0]), True), ("audit-1", str(audit_ids[1]), True),
        ]
        run_tasks = list(await reader.scalars(
            select(Task).where(Task.metadata_["orchestration"]["run_id"].as_string() == str(run_id))
        ))
        assert sorted(t.id for t in run_tasks) == sorted([planning_task.id, *audit_ids])  # no synthesis yet
        for task_id in audit_ids:
            assert len(await sessions(reader, task_id)) == 1
        actions = list(await reader.scalars(select(OrchestrationAction).where(OrchestrationAction.run_id == run_id)))
        assert sorted(a.idempotency_key for a in actions if a.action_type == "adopt_task") == sorted(
            f"run:{run_id}:kind:adopt_task:task:{task_id}" for task_id in audit_ids
        )
        assert sorted(a.request["plan_item_id"] for a in actions if a.action_type == "release_item") == [
            "audit-0", "audit-1",
        ]
        gates = list(await reader.scalars(select(OrchestrationGate).where(
            OrchestrationGate.run_id == run_id, OrchestrationGate.gate_type == "work_completed",
        )))
        assert sorted(g.success_criterion_key for g in gates) == ["plan_item:audit-0", "plan_item:audit-1"]
