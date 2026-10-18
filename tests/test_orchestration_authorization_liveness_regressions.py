import uuid
from datetime import timedelta

import pytest
from fastapi import HTTPException

from huddleroom.models.task import Task
from huddleroom.services.orchestration_progress_view import OrchestrationProgressView
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.session_service import SessionService
from tests.conftest import owned_session
from tests.test_orchestration_progress_view import _run_with_criteria, _task
from tests.test_orchestration_roster_mutation import sessions  # noqa: F401 -- fixture

pytestmark = pytest.mark.asyncio


async def test_unreleased_plan_digest_survives_system_wait_branch_transition(db_session, test_project, test_agent, monkeypatch):
    from huddleroom.config import settings
    from huddleroom.models.base import _utcnow
    from huddleroom.models.orchestration import OrchestrationWait
    from huddleroom.models.session import Session

    goal, run = await _run_with_criteria(db_session, test_project, [])
    run.plan_state = {"status": "accepted", "accepted_plan_snapshot": {"items": [{"id": "dependent"}]}}
    task = await _task(db_session, test_project, run, status="backlog")
    task.assigned_to = test_agent.id
    task.metadata_ = {"orchestration": {"run_id": str(run.id), "authorization_wait": {
        "reason_code": "agent_outside_accepted_team", "required_decision": "reassign_task",
    }}}
    await db_session.flush()
    service = OrchestrationService()

    async def no_release(*_args):
        return 0

    monkeypatch.setattr(service, "_release_ready_work", no_release)
    supervision = service.supervision
    now = _utcnow()
    assert (await supervision.reconcile_local(db_session, goal, run, now=now))["outcome"] == "continue"
    initial = dict(run.supervision_state["unchanged"])
    later = now + timedelta(seconds=settings.orchestration_wake_max_seconds * 2 + 1)
    live_task = await _task(db_session, test_project, run, status="in_progress")
    live_session = Session(project_id=test_project.id, task_id=live_task.id, agent_id=test_agent.id,
                           adapter_type="api", status="running")
    db_session.add(live_session)
    await db_session.flush()
    wait = OrchestrationWait(
        run_id=run.id, wait_key="session:other", owner={"type": "session", "id": str(live_session.id)},
        awaited_event={"event_type": "session.completed", "subject_id": str(live_session.id)},
        due_recheck_at=later + timedelta(seconds=settings.orchestration_wake_max_seconds),
        fallback={"action_type": "attention", "reason": "Still running", "expected_result": "Session completes"},
    )
    db_session.add(wait)
    await db_session.flush()
    monkeypatch.setattr("huddleroom.services.orchestration_supervision._utcnow", lambda: later)
    assert (await supervision.reconcile_local(db_session, goal, run, now=later))["outcome"] == "continue"
    assert wait.status == "open"
    subsequent = run.supervision_state["unchanged"]
    assert subsequent["digest"] == initial["digest"]
    assert subsequent["n"] == initial["n"] + 1


async def test_task_scoped_authorization_wait_is_actionable_follow_up(db_session, test_project, test_agent):
    goal, run = await _run_with_criteria(db_session, test_project, [])
    run.plan_state = {"status": "accepted"}
    task = await _task(db_session, test_project, run, status="backlog")
    task.assigned_to = test_agent.id
    task.metadata_ = {"orchestration": {"run_id": str(run.id), "authorization_wait": {
        "reason_code": "agent_outside_accepted_team", "required_decision": "reassign_task",
    }}}
    await db_session.flush()
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    row = next(row for row in situation.untracked_follow_ups if row["kind"] == "authorization_wait")
    assert str(task.id) in row["id"]
    assert row["suggested_actions"] == ["reassign_task", "ask_human"]
    actionable, _ = OrchestrationService().supervision._actionable(situation, set())
    assert row in actionable
    assert row in situation.as_context()["untracked_follow_ups"]
    assert (await OrchestrationService().supervision.reconcile_local(db_session, goal, run))["outcome"] == "continue"


async def test_unreleased_plan_item_blocks_supervision_and_closeout(db_session, test_project, monkeypatch):
    from huddleroom.models.orchestration import OrchestrationGate

    goal, run = await _run_with_criteria(db_session, test_project, [])
    gate = OrchestrationGate(run_id=run.id, success_criterion_key="first", gate_type="work_completed", status="failed")
    db_session.add(gate)
    await db_session.flush()
    producer = await _task(db_session, test_project, run, status="done")
    run.plan_state = {"status": "accepted", "accepted_plan_snapshot": {"items": [
        {"id": "first"}, {"id": "dependent", "depends_on": ["first"]},
    ]}, "expanded_items": [{"plan_item_id": "first", "gate_id": str(gate.id), "task_id": str(producer.id)}]}
    service = OrchestrationService()

    async def no_release(*_args):
        return 0

    monkeypatch.setattr(service, "_release_ready_work", no_release)
    outcome = await service.supervision.reconcile_local(db_session, goal, run)
    assert outcome["outcome"] != "closeout_ready"
    assert any(b.get("kind") == "plan_items_unreleased" for b in run.active_blockers)
    assert next(b for b in run.active_blockers if b.get("kind") == "plan_items_unreleased")["item_ids"] == ["dependent"]
    with pytest.raises(HTTPException) as exc:
        await service._closeout_preconditions_manifest(db_session, goal, run)
    assert exc.value.status_code == 409
    assert "dependent" in str(exc.value.detail)


@pytest.mark.parametrize("incoming", [{"user_key": "new"}, {
    "orchestration": {"run_id": "tampered"}, "orchestration_contract": {"authorization_scope": "tampered"},
}])
async def test_task_put_preserves_server_owned_authorization_metadata(client, db_session, test_project, incoming):
    owned = {"orchestration": {"run_id": str(uuid.uuid4())},
             "orchestration_contract": {"authorization_scope": {"workspace_mutation": "none"}}}
    task = Task(project_id=test_project.id, title="Protected contract", metadata_={**owned, "old": True})
    db_session.add(task)
    await db_session.flush()
    response = await client.put(f"/api/v1/projects/{test_project.id}/tasks/{task.id}", json={"metadata": incoming})
    assert response.status_code == 200, response.text
    await db_session.refresh(task)
    for key, value in owned.items():
        assert task.metadata_[key] == value


async def test_claim_denies_missing_orchestration_run(db_session, test_project, test_agent):
    task = Task(project_id=test_project.id, title="Orphaned orchestration task", assigned_to=test_agent.id,
                metadata_={"orchestration": {"run_id": str(uuid.uuid4())}})
    db_session.add(task)
    await db_session.flush()
    with pytest.raises(HTTPException) as exc:
        await SessionService().preflight_claim(db_session, task, test_agent)
    assert exc.value.status_code == 409
    assert "authorization_denied" in str(exc.value.detail)


async def test_paused_run_preflight_claim_refuses_without_authorization_blocker(db_session, test_project, test_agent):
    goal, run = await _run_with_criteria(db_session, test_project, [])
    task = await _task(db_session, test_project, run, status="backlog")
    task.assigned_to = test_agent.id
    goal.status = run.status = "paused"
    await db_session.flush()
    with pytest.raises(HTTPException) as exc:
        await SessionService().preflight_claim(db_session, task, test_agent)
    assert exc.value.status_code == 409
    assert exc.value.detail == "authorization_denied: run_not_active"
    assert await SessionService.handle_claim_refusal(db_session, exc.value)
    assert not run.active_blockers


@pytest.mark.parametrize("status,detail", [(409, "unrelated conflict"), (400, "authorization_denied: run_not_active")])
async def test_claim_refusal_helper_does_not_handle_unrelated_errors(db_session, status, detail):
    assert not await SessionService.handle_claim_refusal(db_session, HTTPException(status_code=status, detail=detail))


@pytest.mark.parametrize("action_type", ["retry_task", "reassign_task"])
async def test_paused_claim_recovery_fails_action_without_raising(db_session, test_project, monkeypatch, action_type):
    from tests.test_orchestration_basic_recovery import (
        _agent, _make_run, _make_gate, _make_orchestrated_task, _add_failed_session,
    )

    original = _agent("original", "developer", ["implementation"])
    alternate = _agent("alternate", "developer", ["implementation"])
    db_session.add_all([original, alternate])
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, original.id)
    await _add_failed_session(db_session, task, original.id)
    run.status = goal.status = "paused"

    async def stopped_claim(*_args, **_kwargs):
        raise HTTPException(status_code=409, detail="authorization_denied: run_not_active")

    monkeypatch.setattr("huddleroom.services.orchestration_service.TaskService.run", stopped_claim)
    request = {"action_type": action_type, "task_id": str(task.id)}
    if action_type == "reassign_task":
        request["agent_id"] = str(alternate.id)
    action = await getattr(service, f"execute_{action_type}_action")(
        db_session, run.id, request, f"paused:{action_type}:{task.id}")
    assert action.status == "failed"
    assert task.status == "failed" and task.assigned_to == original.id
    assert "run_not_active" in str(action.error)


async def test_tick_sweeps_cancelled_task_authorization_blocker(sessions, tmp_path, owned_session_factory):
    from tests.test_orchestration_authorization import _started_with_backlog_task
    from tests.test_orchestration_roster_mutation import seed_project_user

    async with owned_session(sessions) as db:
        project, _ = await seed_project_user(db, tmp_path)
        service, goal, run, task = await _started_with_backlog_task(db, project, owns_setup=True)
        task.status = "cancelled"
        scope = f"authorization:{task.id}"
        run.active_blockers = [{"kind": "authorization", "scope": scope, "reason_code": "start_required"}]
        await db.commit()  # Owned setup, then the real final-admission tick.
        await service.tick(db, run.id)
        await db.refresh(run)
        assert not any(b.get("scope") == scope for b in run.active_blockers)


async def test_discovery_failed_source_never_reaches_the_retry_claim(db_session, test_project, test_agent, monkeypatch):
    """a failed discovery source starts recovery; the retry executor (and its claim) never runs."""
    from tests.test_orchestration_continuous_service import dispatched_discovery_source

    goal, run, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    task.status = session.status = "failed"

    async def stopped_claim(*_args, **_kwargs):
        raise AssertionError("the discovery retry must wait for an approved fix")

    monkeypatch.setattr(continuous.orchestration, "execute_retry_task_action", stopped_claim)
    result = await continuous.advance(db_session, goal, run)
    assert result["step"] == "waiting_discovery_recovery"
    assert not result.get("action_id")
