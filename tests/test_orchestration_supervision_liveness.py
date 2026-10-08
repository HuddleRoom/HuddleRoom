import uuid
from types import SimpleNamespace
from copy import deepcopy
from datetime import timedelta
import pytest
from fastapi import HTTPException
from sqlalchemy import func, select

from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationEvidence, OrchestrationGoal, OrchestrationGate, OrchestrationRoadmapVersion, OrchestrationRun, OrchestrationWait
from huddleroom.models.session import Session
from huddleroom.models.meeting import Meeting
from huddleroom.models.task import Task
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision, OrchestrationProcessRun
from huddleroom.services.orchestration_authority_service import runtime_decision_identity
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_supervision import SupervisionAssessment
from huddleroom.routers.orchestration_goals import _detail


pytestmark = pytest.mark.asyncio


async def _run(db, project, *, goal_type="outcome"):
    goal = OrchestrationGoal(project_id=project.id, objective="Keep moving", status="active", goal_type=goal_type)
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


async def test_stale_control_or_contract_creates_no_disposition_action(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    service = OrchestrationService()
    run.status = "paused"
    with pytest.raises(HTTPException, match="not authorized"):
        await service.supervision.apply_disposition(db_session, goal, run, _assessment())
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction)) == 0


async def test_replan_disposition_uses_standard_roadmap_pipeline(db_session, test_project, monkeypatch):
    goal, run = await _run(db_session, test_project, goal_type="roadmap")
    service, supervision = OrchestrationService(), OrchestrationService().supervision
    supervision.orchestration = service

    async def contract(*_args): return "plan:1"
    async def key(*_args): return f"run:{run.id}:kind:request_roadmap_replan:version:1"
    async def executor(db, run_id, request, action_key):
        action = await service._existing_action_for_key(db, run_id, action_key)
        action.status = "completed"
        return action

    monkeypatch.setattr(supervision, "_contract_version", contract)
    monkeypatch.setattr(service, "roadmap_replan_action_key", key)
    monkeypatch.setattr(service, "execute_request_roadmap_replan_action", executor)
    action = await supervision.apply_disposition(db_session, goal, run, _assessment(
        action_type="replan", contract_version="plan:1", request={"agent_id": "00000000-0000-0000-0000-000000000001", "scope": "Adjust remaining work."},
    ))
    assert action.action_type == "request_roadmap_replan"
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationRoadmapVersion)) == 0


async def test_replan_invalidation_is_rejected_without_mutation(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    gate = OrchestrationGate(run_id=run.id, success_criterion_key="done", gate_type="work_completed")
    db_session.add(gate)
    await db_session.flush()
    evidence = OrchestrationEvidence(run_id=run.id, gate_id=gate.id, source_type="test", source_id=uuid.uuid4(), verdict="accepted", evidence_metadata={})
    db_session.add(evidence)
    await db_session.flush()
    with pytest.raises(HTTPException, match="invalidation"):
        await OrchestrationService().supervision.apply_disposition(db_session, goal, run, _assessment(
            action_type="replan", request={"invalidate_assumptions": ["api-stable"]},
        ))
    assert evidence.verdict == "accepted" and gate.status == "open"
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction)) == 0


async def test_dispatch_contract_persists_explicit_expected_result(db_session, test_project, monkeypatch):
    goal, run = await _run(db_session, test_project)
    service = OrchestrationService()

    async def executor(db, run_id, request, key):
        action = await service._existing_action_for_key(db, run_id, key)
        action.status = "completed"
        return action

    monkeypatch.setattr(service, "execute_noop_action", executor)
    action = await service.supervision.apply_disposition(db_session, goal, run, _assessment(reason="different reason", expected_result="different result"))
    assert action.dispatch_contract["expected_result"] == "different result"


async def test_missing_expected_result_creates_no_action(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    with pytest.raises(HTTPException, match="expected_result"):
        await OrchestrationService().supervision.apply_disposition(db_session, goal, run, _assessment(expected_result=""))
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction)) == 0


async def test_continue_is_durable_and_replays_once(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    service = OrchestrationService()
    first = await service.supervision.apply_disposition(db_session, goal, run, _assessment())
    second = await service.supervision.apply_disposition(db_session, goal, run, _assessment())
    assert first.id == second.id and first.action_type == "noop" and first.status == "completed"


async def test_owned_wait_replays_and_only_exact_event_clears(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    supervision = OrchestrationService().supervision
    first = await supervision.create_wait(
        db_session, run, "dependency:a", {"type": "task", "id": "a"},
        {"event_type": "task.status_changed", "subject_id": "a"}, 30,
        {"action_type": "attention", "reason": "dependency unresolved", "expected_result": "attention recorded"},
    )
    replay = await supervision.create_wait(
        db_session, run, "dependency:a", {"type": "task", "id": "a"},
        {"event_type": "task.status_changed", "subject_id": "a"}, 30,
        {"action_type": "attention", "reason": "dependency unresolved", "expected_result": "attention recorded"},
    )
    assert replay.id == first.id
    assert await supervision.clear_matching_waits(
        db_session, run, event_type="task.status_changed", subject_id="other"
    ) == 0
    assert first.status == "open"
    assert await supervision.clear_matching_waits(
        db_session, run, event_type="task.status_changed", subject_id="a"
    ) == 1
    assert first.status == "cleared"


async def test_active_task_wait_is_bound_to_its_session_and_task_occurrence(db_session, test_project, test_agent):
    goal, run = await _run(db_session, test_project)
    run.plan_state = {"status": "accepted"}
    task = Task(
        project_id=test_project.id, title="Work", status="in_progress", assigned_to=test_agent.id,
        metadata_={"orchestration": {"run_id": str(run.id)}},
    )
    db_session.add(task)
    await db_session.flush()
    session = Session(
        project_id=test_project.id, task_id=task.id, agent_id=test_agent.id,
        adapter_type="api", status="running",
    )
    db_session.add(session)
    await db_session.flush()

    result = await OrchestrationService().supervision.reconcile_local(db_session, goal, run)

    assert result["outcome"] == "waiting"
    wait = await db_session.get(OrchestrationWait, uuid.UUID(result["wait_id"]))
    assert wait.wait_key == f"run:{run.id}:wait:session:{session.id}:task:{task.id}:status"
    assert wait.awaited_event == {"event_type": "task.status_changed", "matcher": {"task_id": str(task.id)}}


async def test_active_session_prevents_orphaned_ownership(db_session, test_project, test_agent):
    goal, run = await _run(db_session, test_project)
    task = Task(
        project_id=test_project.id, title="Work", status="in_progress",
        metadata_={"orchestration": {"run_id": str(run.id)}},
    )
    db_session.add(task)
    await db_session.flush()
    db_session.add(Session(
        project_id=test_project.id, task_id=task.id, agent_id=test_agent.id,
        adapter_type="api", status="running",
    ))
    await db_session.flush()

    result = await OrchestrationService().supervision.reconcile_local(db_session, goal, run)

    assert result["outcome"] == "waiting"
    assert not any(blocker["kind"] == "orphaned_ownership" for blocker in run.active_blockers)


async def test_all_active_provider_sources_own_their_tasks(db_session, test_project, test_agent):
    goal, run = await _run(db_session, test_project)
    tasks = [
        Task(project_id=test_project.id, title=title, status="in_progress", assigned_to=test_agent.id,
             metadata_={"orchestration": {"run_id": str(run.id)}})
        for title in ("first", "second")
    ]
    db_session.add_all(tasks)
    await db_session.flush()
    db_session.add_all([
        Meeting(project_id=test_project.id, title=f"Meeting {task.title}", meeting_type="decision",
                status="active", source_task_id=task.id)
        for task in tasks
    ])
    session = Session(project_id=test_project.id, task_id=tasks[1].id, agent_id=test_agent.id,
                      adapter_type="api", status="running")
    db_session.add(session)
    await db_session.flush()

    result = await OrchestrationService().supervision.reconcile_local(db_session, goal, run)

    assert result["outcome"] == "durable_source_active"
    assert await db_session.scalar(select(OrchestrationWait).where(
        OrchestrationWait.wait_key == f"run:{run.id}:wait:session:{session.id}:task:{tasks[1].id}:status"
    )) is None


async def test_session_created_wait_clears_when_its_exact_predicate_stops_holding(db_session, test_project, test_agent):
    goal, run = await _run(db_session, test_project)
    task = Task(project_id=test_project.id, title="Work", status="in_progress", assigned_to=test_agent.id,
                metadata_={"orchestration": {"run_id": str(run.id)}})
    db_session.add(task)
    await db_session.flush()
    supervision = OrchestrationService().supervision
    await supervision.reconcile_local(db_session, goal, run)
    wait = await db_session.scalar(select(OrchestrationWait).where(
        OrchestrationWait.wait_key == f"run:{run.id}:wait:task:{task.id}:session_created"
    ))
    task.status = "ready"

    await supervision.reconcile_local(db_session, goal, run)

    assert wait.status == "cleared"


async def test_everyone_idle_is_fingerprinted_closeout_readiness(db_session, test_project, monkeypatch):
    goal, run = await _run(db_session, test_project)
    run.plan_state = {"status": "accepted", "version": 1}
    service = OrchestrationService()

    async def no_release(*_args):
        return 0

    monkeypatch.setattr(service, "_release_ready_work", no_release)

    first = await service.supervision.reconcile_local(db_session, goal, run)
    assert (goal.status, run.status, run.phase) == ("active", "running", "authorized")
    second = await service.supervision.reconcile_local(db_session, goal, run)

    assert first["outcome"] == second["outcome"] == "closeout_ready"
    assert first["fingerprint"] == second["fingerprint"]
    assert sum(blocker.get("kind") == "everyone_idle" for blocker in run.active_blockers) == 1


async def test_release_precedes_closeout_readiness(db_session, test_project, monkeypatch):
    goal, run = await _run(db_session, test_project)
    run.plan_state = {"status": "accepted"}
    service = OrchestrationService()

    async def release(*_args):
        return 1

    monkeypatch.setattr(service, "_release_ready_work", release)
    result = await service.supervision.reconcile_local(db_session, goal, run)
    assert result == {"outcome": "released", "count": 1}


async def test_open_gate_prevents_idle_closeout_before_verification(db_session, test_project, monkeypatch):
    goal, run = await _run(db_session, test_project)
    run.plan_state = {"status": "accepted"}
    db_session.add(OrchestrationGate(run_id=run.id, success_criterion_key="done", gate_type="work_completed"))
    service = OrchestrationService()

    async def no_release(*_args):
        return 0

    monkeypatch.setattr(service, "_release_ready_work", no_release)
    # An open gate without a dispatched verifier is work the executor may
    # still start; it is not proof of a missing owner.
    assert (await service.supervision.reconcile_local(db_session, goal, run))["outcome"] == "continue"


async def test_due_runtime_decision_wait_falls_back_without_waiting_on_itself(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    decision = OrchestrationAuthorityDecision(
        goal_id=goal.id, run_id=run.id, decision_key="runtime:test", runtime_identity=runtime_decision_identity(run.id, "test", "human", "start"),
        title="Test", authority="human", question="Continue?", options=[{"key": "yes"}], contract_version="start",
    )
    db_session.add(decision)
    await db_session.flush()
    service = OrchestrationService().supervision
    first = await service.reconcile_local(db_session, goal, run)
    wait = await db_session.get(OrchestrationWait, uuid.UUID(first["wait_id"]))
    wait.due_recheck_at = _utcnow() - timedelta(seconds=1)
    result = await service.reconcile_local(db_session, goal, run)
    assert result["outcome"] == "due_fallback"
    assert (await db_session.get(OrchestrationAction, uuid.UUID(result["action_id"]))).action_type == "record_warning"


async def test_due_wait_precedes_future_wait(db_session, test_project, test_agent):
    goal, run = await _run(db_session, test_project)
    supervision = OrchestrationService().supervision
    tasks = {}
    for title in ("future", "due"):
        tasks[title] = Task(
            project_id=test_project.id, title=title, status="in_progress", assigned_to=test_agent.id,
            metadata_={"orchestration": {"run_id": str(run.id)}},
        )
        db_session.add(tasks[title])
    await db_session.flush()
    future = await supervision.create_wait(
        db_session, run, "future", {"type": "task", "id": str(tasks["future"].id)},
        {"event_type": "task.status_changed", "matcher": {"task_id": str(tasks["future"].id)}}, 30,
        {"action_type": "attention", "reason": "future", "expected_result": "future completes"},
    )
    due = await supervision.create_wait(
        db_session, run, "due", {"type": "task", "id": str(tasks["due"].id)},
        {"event_type": "task.status_changed", "matcher": {"task_id": str(tasks["due"].id)}}, 30,
        {"action_type": "continue", "reason": "due", "expected_result": "due completes"},
    )
    due.due_recheck_at = _utcnow() - timedelta(seconds=1)

    result = await supervision.reconcile_local(db_session, goal, run)

    assert result["outcome"] == "due_fallback"
    assert due.status == "cleared" and future.status == "open"
    assert (await db_session.get(OrchestrationAction, uuid.UUID(result["action_id"]))).action_type == "noop"


async def test_non_runtime_pending_authority_decision_gets_a_durable_wait(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    decision = OrchestrationAuthorityDecision(
        goal_id=goal.id, run_id=run.id, decision_key="plan:test", title="Test", authority="human",
        question="Continue?", options=[{"key": "yes"}], contract_version="start",
    )
    db_session.add(decision)
    await db_session.flush()

    result = await OrchestrationService().supervision.reconcile_local(db_session, goal, run)

    assert result["outcome"] == "waiting"
    wait = await db_session.get(OrchestrationWait, uuid.UUID(result["wait_id"]))
    assert wait.awaited_event["matcher"] == {"decision_id": str(decision.id)}


async def test_superseded_process_does_not_hold_preplan_liveness(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    current = OrchestrationProcessRun(goal_id=goal.id, process_type="current", trigger_reason="test", status="completed")
    db_session.add(current)
    await db_session.flush()
    db_session.add(OrchestrationProcessRun(
        goal_id=goal.id, process_type="stale", trigger_reason="test", status="running", superseded_by_id=current.id,
    ))
    await db_session.flush()
    assert (await OrchestrationService().supervision.reconcile_local(db_session, goal, run))["outcome"] == "continue"


@pytest.mark.parametrize("owner,event,due,fallback", [
    ({}, {"event_type": "x"}, 1, {"action_type": "attention", "expected_result": "x"}),
    ({"type": "task", "id": "a"}, {}, 1, {"action_type": "attention", "expected_result": "x"}),
    ({"type": "task", "id": "a"}, {"event_type": "x"}, 0, {"action_type": "attention", "expected_result": "x"}),
    ({"type": "task", "id": "a"}, {"event_type": "x"}, 1, {}),
])
async def test_owned_wait_rejects_incomplete_contract(db_session, test_project, owner, event, due, fallback):
    _, run = await _run(db_session, test_project)
    with pytest.raises(HTTPException, match="wait (requires|fallback)"):
        await OrchestrationService().supervision.create_wait(
            db_session, run, "bad", owner, event, due, fallback
        )


async def test_goal_detail_projects_read_only_supervision_from_durable_sources(
    db_session, test_project, test_agent,
):
    """The detail projection is a stable read of current durable supervision facts."""
    goal, run = await _run(db_session, test_project)
    now = _utcnow()
    goal.success_criteria = [{"key": "ship", "description": "Ship the verified change."}]
    goal.budget = {"caps": {"max_tokens": "10"}}
    task = Task(
        project_id=test_project.id, title="Ship change", status="in_progress", assigned_to=test_agent.id,
        metadata_={"orchestration": {"run_id": str(run.id)}},
    )
    db_session.add(task)
    await db_session.flush()
    action = OrchestrationAction(
        run_id=run.id, idempotency_key=f"detail-worker:{task.id}", action_type="create_delegation_task",
        request={}, target_type="task", target_id=task.id, status="completed",
        budget_ledger={"allocation": {"max_tokens": "3"}, "reserved": {}, "committed": {},
                       "consumed": {"max_tokens": "3"}, "usage_state": "known"},
    )
    db_session.add(action)
    await db_session.flush()
    task.metadata_ = {"orchestration": {"run_id": str(run.id), "action_id": str(action.id)}}
    session = Session(
        project_id=test_project.id, task_id=task.id, agent_id=test_agent.id, adapter_type="api",
        status="running", runner_task_id="runner-live",
        metadata_={"orchestration": {"action_id": str(action.id)}},
    )
    gate = OrchestrationGate(
        run_id=run.id, success_criterion_key="ship", gate_type="work_completed", status="accepted",
        accepted_at=now,
    )
    db_session.add_all([action, session, gate])
    await db_session.flush()
    accepted = OrchestrationEvidence(
        run_id=run.id, gate_id=gate.id, source_type="verification", source_id=task.id,
        verdict="accepted", evidence_metadata={"claim": "verified"},
    )
    rejected = OrchestrationEvidence(
        run_id=run.id, gate_id=gate.id, source_type="task", source_id=task.id,
        verdict="rejected", evidence_metadata={"claim": "not evidence"},
    )
    oldest_wait = OrchestrationWait(
        run_id=run.id, wait_key="detail:oldest", owner={"type": "session", "id": str(session.id)},
        awaited_event={"event_type": "session.completed", "matcher": {"session_id": str(session.id)}},
        due_recheck_at=now + timedelta(minutes=1),
        fallback={"action_type": "attention", "reason": "worker", "expected_result": "worker result"},
        created_at=now - timedelta(minutes=2),
    )
    newest_wait = OrchestrationWait(
        run_id=run.id, wait_key="detail:newest", owner={"type": "task", "id": str(task.id)},
        awaited_event={"event_type": "task.status_changed", "matcher": {"task_id": str(task.id)}},
        due_recheck_at=now + timedelta(minutes=2),
        fallback={"action_type": "attention", "reason": "task", "expected_result": "task result"},
        created_at=now - timedelta(minutes=1),
    )
    oldest_direction = OrchestrationAuthorityDecision(
        goal_id=goal.id, run_id=run.id, decision_key="detail:oldest", title="Oldest direction",
        authority="human", question="Choose the durable direction.", options=[{"key": "continue"}],
        asked_at=now - timedelta(minutes=2),
    )
    newest_direction = OrchestrationAuthorityDecision(
        goal_id=goal.id, run_id=run.id, decision_key="detail:newest", title="Newer direction",
        authority="human", question="This must not win.", options=[{"key": "continue"}],
        asked_at=now - timedelta(minutes=1),
    )
    db_session.add_all([accepted, rejected, oldest_wait, newest_wait, oldest_direction, newest_direction])
    await db_session.flush()
    run.supervision_state = {
        "verified_progress": [{"criterion_key": "ship", "status": "verified"}],
        "last_assessment": {"useful_learning": [{"untrusted": "provider output"}], "criterion_progress": [{"key": "wrong"}]},
        "recovery": {"sessions": {str(session.id): {
            "session_id": str(session.id), "source_runner_id": "runner-live", "current_runner_id": "runner-live",
            "backend_observation": "active", "classification": "live", "disposition": "adopted",
            "assessed_at": now.isoformat(), "action_id": str(action.id), "wait_id": str(oldest_wait.id),
        }}},
    }
    await db_session.flush()
    before = {
        "budget": deepcopy(run.budget_state), "state": deepcopy(run.supervision_state),
        "blockers": deepcopy(run.active_blockers),
    }

    detail = await _detail(db_session, goal, run)
    supervision = detail.supervision.model_dump(mode="json")

    assert set(supervision) == {
        "condition", "operation", "next_action", "rationale", "criterion", "verified_progress",
        "useful_learning", "accepted_evidence", "workers", "waits", "recovery_history",
        "pending_direction", "budget", "transition",
    }
    assert supervision["condition"] == "needs_you"
    assert supervision["verified_progress"] == [{"criterion_key": "ship", "status": "verified"}]
    assert supervision["useful_learning"] == []
    assert supervision["criterion"]["key"] == "ship"
    assert supervision["criterion"]["status"] == "accepted"
    assert [item["id"] for item in supervision["accepted_evidence"]] == [str(accepted.id)]
    assert [item["id"] for item in supervision["waits"]] == [str(oldest_wait.id), str(newest_wait.id)]
    assert supervision["waits"][0]["event"] == "session.completed"
    assert supervision["waits"][0]["matcher"] == {"session_id": str(session.id)}
    assert supervision["pending_direction"]["id"] == str(oldest_direction.id)
    assert supervision["workers"] == [{
        "session_id": str(session.id), "task_id": str(task.id), "agent_id": str(test_agent.id),
        "task_title": task.title, "session_status": "running", "observed_liveness": "live",
        "runner_id": "runner-live", "observed_at": now.isoformat(),
    }]
    assert supervision["recovery_history"] == [{
        "session_id": str(session.id), "classification": "live", "disposition": "adopted",
        "backend_observation": "active", "assessed_at": now.isoformat(), "action_id": str(action.id),
        "wait_id": str(oldest_wait.id),
    }]
    assert supervision["budget"] == {
        "consumed": {"max_tokens": "3"}, "committed": {}, "reserved": {}, "remaining": {"max_tokens": "7"},
    }
    assert supervision["transition"] == {
        "id": str(oldest_direction.id), "key": f"pending_direction:{oldest_direction.id}",
        "kind": "pending_direction", "message": oldest_direction.title,
        "occurred_at": (now - timedelta(minutes=2)).isoformat().replace("+00:00", "Z"),
    }
    assert run.budget_state == before["budget"]
    assert run.supervision_state == before["state"]
    assert run.active_blockers == before["blockers"]


@pytest.mark.parametrize("state,expected", [
    ("working", "working"), ("waiting", "waiting"), ("needs_you", "needs_you"),
    ("needs_attention", "needs_attention"), ("paused", "paused"), ("stopped", "stopped"),
    ("cancelled", "cancelled"), ("completed", "completed"),
])
async def test_goal_detail_http_projects_each_durable_supervision_condition(
    client, db_session, test_project, state, expected,
):
    created = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals",
        json={"objective": f"Condition {state}", "success_criteria": [], "constraints": {}, "budget": {}},
    )
    assert created.status_code == 201, created.text
    goal_id = uuid.UUID(created.json()["goal"]["id"])
    run_id = uuid.UUID(created.json()["run"]["id"])
    goal = await db_session.get(OrchestrationGoal, goal_id)
    run = await db_session.get(OrchestrationRun, run_id)
    assert goal is not None and run is not None
    if state == "waiting":
        db_session.add(OrchestrationWait(
            run_id=run.id, wait_key=f"condition:{state}", owner={"type": "run", "id": str(run.id)},
            awaited_event={"event_type": "condition.done", "matcher": {}}, due_recheck_at=_utcnow() + timedelta(minutes=1),
            fallback={"action_type": "attention", "reason": "condition wait", "expected_result": "done"},
        ))
    elif state == "needs_you":
        db_session.add(OrchestrationAuthorityDecision(
            goal_id=goal.id, run_id=run.id, decision_key=f"condition:{state}", title="Direction required",
            authority="human", question="Which direction?", options=[{"key": "continue"}],
        ))
    elif state == "needs_attention":
        run.active_blockers = [{"kind": "condition_blocker", "reason": "condition attention"}]
    elif state == "paused":
        goal.status = run.status = "paused"
    elif state == "stopped":
        goal.goal_type = "continuous"
        goal.continuous_state = {"stopped_at": _utcnow().isoformat()}
    elif state in {"cancelled", "completed"}:
        goal.status = run.status = state
        run.completed_at = _utcnow()
    await db_session.flush()

    detail = await client.get(f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal_id}")
    assert detail.status_code == 200, detail.text
    supervision = detail.json()["supervision"]
    assert supervision["condition"] == expected
    if state == "waiting":
        assert supervision["transition"] is None


async def test_goal_detail_http_prioritizes_pending_direction_and_keeps_transition_stable(
    client, db_session, test_project,
):
    created = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals",
        json={"objective": "Priority", "success_criteria": [], "constraints": {}, "budget": {}},
    )
    goal_id, run_id = uuid.UUID(created.json()["goal"]["id"]), uuid.UUID(created.json()["run"]["id"])
    goal, run = await db_session.get(OrchestrationGoal, goal_id), await db_session.get(OrchestrationRun, run_id)
    direction = OrchestrationAuthorityDecision(
        goal_id=goal.id, run_id=run.id, decision_key="priority", title="Choose", authority="human",
        question="Choose a direction.", options=[{"key": "continue"}],
    )
    db_session.add_all([direction, OrchestrationWait(
        run_id=run.id, wait_key="priority", owner={"type": "run", "id": str(run.id)},
        awaited_event={"event_type": "ignored", "matcher": {}}, due_recheck_at=_utcnow() + timedelta(minutes=1),
        fallback={"action_type": "attention", "reason": "wait", "expected_result": "ignored"},
    )])
    await db_session.flush()
    first = await client.get(f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal_id}")
    second = await client.get(f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal_id}")
    assert first.status_code == second.status_code == 200
    first_supervision, second_supervision = first.json()["supervision"], second.json()["supervision"]
    assert first_supervision["condition"] == "needs_you"
    assert first_supervision["transition"] == second_supervision["transition"]


@pytest.mark.parametrize("state,operation,next_action", [
    ("paused", "Paused", "Resume goal"), ("stopped", "Stopped", "No action"),
    ("cancelled", "Cancelled", "No action"), ("completed", "Completed", "No action"),
])
async def test_goal_detail_http_control_state_outranks_stale_pending_direction(
    client, db_session, test_project, state, operation, next_action,
):
    created = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals",
        json={"objective": f"Control {state}", "success_criteria": [], "constraints": {}, "budget": {}},
    )
    goal_id, run_id = uuid.UUID(created.json()["goal"]["id"]), uuid.UUID(created.json()["run"]["id"])
    goal, run = await db_session.get(OrchestrationGoal, goal_id), await db_session.get(OrchestrationRun, run_id)
    db_session.add(OrchestrationAuthorityDecision(
        goal_id=goal.id, run_id=run.id, decision_key=f"stale:{state}", title="Stale direction",
        authority="human", question="This must not become actionable.", options=[{"key": "continue"}],
    ))
    if state == "paused":
        goal.status = run.status = "paused"
    elif state == "stopped":
        goal.goal_type, goal.continuous_state = "continuous", {"stopped_at": _utcnow().isoformat()}
    else:
        goal.status = run.status = state
        run.completed_at = _utcnow()
    await db_session.flush()

    first = await client.get(f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal_id}")
    second = await client.get(f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal_id}")
    assert first.status_code == second.status_code == 200
    supervision = first.json()["supervision"]
    assert (supervision["condition"], supervision["operation"], supervision["next_action"]) == (state, operation, next_action)
    assert supervision["pending_direction"] is None
    assert supervision["transition"]["kind"] == state
    assert supervision["transition"] == second.json()["supervision"]["transition"]


# --- Ordering, independent work, and verification exhaustion -------------------------------

async def _task(db, project, agent, run, title, **kwargs):
    task = Task(project_id=project.id, title=title, status=kwargs.pop("status", "in_progress"), assigned_to=agent.id,
                metadata_={"orchestration": {"run_id": str(run.id)}}, **kwargs)
    db.add(task)
    await db.flush()
    return task


async def _no_release(service, monkeypatch):
    async def release(*_args):
        return 0
    monkeypatch.setattr(service, "_release_ready_work", release)


async def _meeting_on(db, project, task):
    db.add(Meeting(project_id=project.id, title=f"M {task.title}", meeting_type="decision", status="active", source_task_id=task.id))
    await db.flush()


async def _graph_on(db, project, task):
    from huddleroom.models.graph import Graph, GraphRun
    graph = Graph(project_id=project.id, name="G", version="1", definition={}, triggers=[])
    db.add(graph)
    await db.flush()
    db.add(GraphRun(graph_id=graph.id, project_id=project.id, linked_task_id=task.id, current_node="n", status="active"))
    await db.flush()


async def test_meeting_and_graph_sources_together_hold_liveness(db_session, test_project, test_agent):
    goal, run = await _run(db_session, test_project)
    first, second = await _task(db_session, test_project, test_agent, run, "a"), await _task(db_session, test_project, test_agent, run, "b")
    await _meeting_on(db_session, test_project, first)
    await _graph_on(db_session, test_project, second)
    result = await OrchestrationService().supervision.reconcile_local(db_session, goal, run)
    assert result["outcome"] == "durable_source_active"
    assert not (await db_session.scalars(select(OrchestrationWait).where(OrchestrationWait.run_id == run.id))).all()


async def test_overdue_wait_is_processed_during_active_meeting(db_session, test_project, test_agent):
    goal, run = await _run(db_session, test_project)
    await _meeting_on(db_session, test_project, await _task(db_session, test_project, test_agent, run, "a"))
    db_session.add(OrchestrationAuthorityDecision(
        goal_id=goal.id, run_id=run.id, decision_key="plan:test", title="T", authority="human",
        question="Continue?", options=[{"key": "yes"}], contract_version="start",
    ))
    await db_session.flush()
    supervision = OrchestrationService().supervision
    first = await supervision.reconcile_local(db_session, goal, run)
    assert first["outcome"] == "durable_source_active"
    wait = await db_session.scalar(select(OrchestrationWait).where(OrchestrationWait.run_id == run.id))
    wait.due_recheck_at = _utcnow() - timedelta(seconds=1)
    assert (await supervision.reconcile_local(db_session, goal, run))["outcome"] == "due_fallback"


async def test_independent_criterion_proceeds_during_active_graph(db_session, test_project, test_agent, monkeypatch):
    goal, run = await _run(db_session, test_project)
    goal.success_criteria = [{"key": "indep", "description": "Independent"}]
    run.plan_state = {"status": "accepted"}
    service = OrchestrationService()
    await _no_release(service, monkeypatch)
    await _graph_on(db_session, test_project, await _task(db_session, test_project, test_agent, run, "a"))
    assert (await service.supervision.reconcile_local(db_session, goal, run)) == {"outcome": "continue"}


async def _dependent_setup(db, project, agent, monkeypatch, *, dependent):
    goal, run = await _run(db, project)
    run.plan_state = {"status": "accepted"}
    service = OrchestrationService()
    await _no_release(service, monkeypatch)
    source = await _task(db, project, agent, run, "source")
    await _task(db, project, agent, run, "stalled", status="blocked", depends_on=[str(source.id)] if dependent else [])
    await _meeting_on(db, project, source)
    return await service.supervision.reconcile_local(db, goal, run)


async def test_dependent_only_work_waits_during_active_source(db_session, test_project, test_agent, monkeypatch):
    result = await _dependent_setup(db_session, test_project, test_agent, monkeypatch, dependent=True)
    assert result["outcome"] == "durable_source_active"


async def test_independent_stalled_task_proceeds_during_active_source(db_session, test_project, test_agent, monkeypatch):
    result = await _dependent_setup(db_session, test_project, test_agent, monkeypatch, dependent=False)
    assert result == {"outcome": "continue"}


async def _gate_with_verifications(db, run, statuses):
    gate = OrchestrationGate(run_id=run.id, success_criterion_key="c", gate_type="work_completed")
    db.add(gate)
    await db.flush()
    for n, status in enumerate(statuses):
        db.add(OrchestrationAction(
            run_id=run.id, idempotency_key=f"v{n}", action_type="request_verification", status=status,
            request={"gate_id": str(gate.id)}, error="boom" if status == "failed" else None,
            created_at=_utcnow() + timedelta(seconds=n),
        ))
    await db.flush()
    return gate


async def _ask_actions(db, run):
    return list((await db.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "ask_human"))).all())


async def test_stale_failed_verification_followed_by_accepted_gate_raises_no_attention(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    gate = await _gate_with_verifications(db_session, run, ["failed", "failed", "failed"])
    gate.status = "accepted"
    result = await OrchestrationService().supervision.reconcile_local(db_session, goal, run)
    assert result == {"outcome": "continue"} and not await _ask_actions(db_session, run)
    assert not any(a.action_type == "record_warning" for a in (await db_session.scalars(select(OrchestrationAction))).all())


async def test_three_failed_verifications_ask_human_with_exact_question(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    gate = await _gate_with_verifications(db_session, run, ["failed"] * 3)
    supervision = OrchestrationService().supervision
    result = await supervision.reconcile_local(db_session, goal, run)
    assert result["outcome"] == "needs_attention"
    (ask,) = await _ask_actions(db_session, run)
    assert ask.idempotency_key == f"run:{run.id}:ask_human:verification_exhausted:{gate.id}:1"
    assert str(gate.id) in ask.request["question"] and "boom" in ask.request["question"]
    again = await supervision.reconcile_local(db_session, goal, run)
    assert again["outcome"] != "needs_attention" and len(await _ask_actions(db_session, run)) == 1


async def test_two_failed_verifications_stay_a_follow_up(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    await _gate_with_verifications(db_session, run, ["failed"] * 2)
    result = await OrchestrationService().supervision.reconcile_local(db_session, goal, run)
    assert result["outcome"] != "needs_attention" and not await _ask_actions(db_session, run)


async def _verification_asks_after(db, goal, run, statuses, preexisting=False):
    gate = await _gate_with_verifications(db, run, statuses)
    if preexisting:
        await OrchestrationService().execute_ask_human_action(db, run.id, {
            "action_type": "ask_human", "question": "old", "gate_id": str(gate.id),
        }, f"run:{run.id}:ask_human:verification_exhausted:{gate.id}:1")
    await OrchestrationService().supervision.reconcile_local(db, goal, run)
    return gate, [a.idempotency_key.rsplit(":", 1)[1] for a in await _ask_actions(db, run)
                  if "verification_exhausted" in a.idempotency_key]


async def test_verification_exhaustion_sequences(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    _, keys = await _verification_asks_after(db_session, goal, run, ["failed"] * 3)
    assert keys == ["1"]

    goal, run = await _run(db_session, test_project)
    _, keys = await _verification_asks_after(db_session, goal, run, ["failed", "completed", "failed", "failed"])
    assert keys == []

    goal, run = await _run(db_session, test_project)
    _, keys = await _verification_asks_after(db_session, goal, run, ["failed"] * 4)
    assert keys == []

    goal, run = await _run(db_session, test_project)
    _, keys = await _verification_asks_after(db_session, goal, run, ["failed"] * 6)
    assert keys == ["2"]

    goal, run = await _run(db_session, test_project)
    _, keys = await _verification_asks_after(db_session, goal, run, ["failed"] * 3, preexisting=True)
    assert keys == ["1"]


async def test_open_meeting_commitment_blocks_closeout_ready(db_session, test_project):
    from tests.test_orchestration_progress_view import _action_item, _meeting, _task as _pv_task
    goal, run = await _run(db_session, test_project)
    run.plan_state = {"status": "accepted"}
    meeting = await _meeting(db_session, test_project, await _pv_task(db_session, test_project, run))
    await _action_item(db_session, meeting, description="Open item")
    result = await OrchestrationService().supervision.reconcile_local(db_session, goal, run)
    assert result["outcome"] != "closeout_ready"


async def test_pending_decision_makes_gate_not_actionable_and_uncounted(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    run.plan_state = {"status": "accepted"}
    gate = await _gate_with_verifications(db_session, run, ["failed"] * 3)
    supervision = OrchestrationService().supervision
    assert (await supervision.reconcile_local(db_session, goal, run))["outcome"] == "needs_attention"
    for _ in range(6):
        await supervision.reconcile_local(db_session, goal, run)
    assert not (run.supervision_state or {}).get("unchanged")
    assert len(await _ask_actions(db_session, run)) == 1
    assert gate.status == "open"


async def test_identical_supervision_question_after_answer_is_new_ask(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    supervision = OrchestrationService().supervision
    ask = _assessment(action_type="ask_human", request={"question": "Which way?"})
    first = await supervision.apply_disposition(db_session, goal, run, ask)
    again = await supervision.apply_disposition(db_session, goal, run, ask)
    assert again.id == first.id
    decision = await db_session.get(OrchestrationAuthorityDecision, first.target_id)
    decision.status = "answered"
    await db_session.flush()
    second = await supervision.apply_disposition(db_session, goal, run, ask)
    assert second.id != first.id and second.target_id != first.target_id
    assert (await supervision.apply_disposition(db_session, goal, run, ask)).id == second.id


async def test_commitment_only_state_counts_then_asks_owner_naming_commitment(db_session, test_project, monkeypatch):
    from huddleroom.config import settings
    from tests.test_orchestration_progress_view import _action_item, _meeting, _task as _pv_task
    goal, run = await _run(db_session, test_project)
    run.plan_state = {"status": "accepted"}
    service = OrchestrationService()
    await _no_release(service, monkeypatch)
    source = await _pv_task(db_session, test_project, run)
    source.status = "done"  # no active work: only the commitment remains
    meeting = await _meeting(db_session, test_project, source)
    meeting.status = "completed"
    item = await _action_item(db_session, meeting, description="Ship the thing")
    await db_session.flush()
    now = _utcnow()
    outcomes = []
    for _ in range(4):
        now += timedelta(seconds=settings.orchestration_wake_max_seconds * 2 + 1)
        outcomes.append((await service.supervision.reconcile_local(db_session, goal, run, now=now))["outcome"])
    assert outcomes == ["continue"] * 3 + ["needs_attention"]
    (ask,) = await _ask_actions(db_session, run)
    assert str(item.id) in ask.request["question"]
    # Pending question: counting restarts after it is answered.
    state = dict(run.supervision_state)
    assert state["unchanged"]
    assert await service.supervision._proactive_outcome(
        db_session, run, SimpleNamespace(untracked_follow_ups=[], progress_view=[]), set(), state, now, []) is None
    assert run.supervision_state["unchanged"] == {}
