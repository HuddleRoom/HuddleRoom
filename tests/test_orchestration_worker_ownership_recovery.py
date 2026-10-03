"""Task 9 acceptance tests for durable orchestration-worker ownership."""
from datetime import datetime, timedelta, timezone
import asyncio
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from huddleroom.adapters.api_adapter import ApiAdapter
from huddleroom.adapters.cli_adapter import CliAdapter
from huddleroom.config import settings
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationDecision, OrchestrationGoal, OrchestrationRun, OrchestrationWait
from huddleroom.models.orchestration_memory import OrchestrationMemorySection
from huddleroom.models.agent import Agent
from huddleroom.models.project import Project
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.orchestration_recovery_service import OrchestrationRecoveryService, RunnerObservation
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.session_service import SessionService
from huddleroom.workers.session_tasks import (
    _mark_attempt_result, _require_runnable_session, mark_attempt_effect_started, mark_attempt_project_not_runnable,
)
from huddleroom.workers import orchestration_recovery_tasks as recovery_tasks


async def _owned(db, project, agent, *, session_status="running", task_status="in_progress", runner="runner-1",
                 attempt=None, resumable=False, provider=None, goal_status="active"):
    goal = OrchestrationGoal(project_id=project.id, objective="recover", original_request="recover",
                             success_criteria=[], constraints={}, budget={}, status=goal_status)
    db.add(goal); await db.flush()
    run = OrchestrationRun(goal_id=goal.id, status="paused" if goal_status == "paused" else "running", phase="baseline")
    db.add(run); await db.flush()
    action = OrchestrationAction(run_id=run.id, idempotency_key=f"seed:{uuid.uuid4()}", action_type="retry_task", request={}, status="completed",
        dispatch_contract={"owner": "orchestration_recovery"})
    db.add(action); await db.flush()
    task = Task(project_id=project.id, title="owned", description="owned", status=task_status, assigned_to=agent.id,
                metadata_={"orchestration": {"run_id": str(run.id), "action_id": str(action.id)}})
    db.add(task); await db.flush()
    metadata = {"orchestration": {"action_id": str(action.id)}}
    if attempt is not None:
        metadata["attempt"] = attempt
    session = Session(project_id=project.id, agent_id=agent.id, task_id=task.id, adapter_type="cli", status=session_status,
                      runner_task_id=runner, resumable=resumable, provider_session_id=provider, metadata_=metadata)
    db.add(session); await db.flush()
    action.request = {"task_id": str(task.id)}
    action.target_type, action.target_id = "session", session.id
    return goal, run, action, task, session


def _launch_fingerprint(session, task, agent, project):
    return CliAdapter.launch_fingerprint(session, task, agent, project, Path(project.workspace_path))


async def _applied(db, project, agent, *, status="running", backend="active", attempt=None, resumable=False,
                   provider=None, goal_status="active", task_status="in_progress", runner="runner-1"):
    goal, run, action, task, session = await _owned(db, project, agent, session_status=status, task_status=task_status,
        attempt=attempt, resumable=resumable, provider=provider, goal_status=goal_status, runner=runner)
    service = OrchestrationRecoveryService(); ready = datetime.now(timezone.utc) - timedelta(seconds=1)
    snapshot = await service.build_goal_snapshot(db, goal.id, ready)
    assert snapshot is not None and snapshot.sessions[0].session_id == session.id
    result = await service.apply_goal_recovery(db, snapshot, {
        session.id: RunnerObservation(session.runner_task_id, backend, datetime.now(timezone.utc))})
    return result, goal, run, action, task, session


@pytest.mark.asyncio
async def test_verified_ownership_requires_complete_same_project_chain(db_session, test_project, test_agent):
    goal, _, _, task, session = await _owned(db_session, test_project, test_agent)
    snapshot = await OrchestrationRecoveryService().build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))
    assert snapshot and [item.session_id for item in snapshot.sessions] == [session.id]
    task.metadata_ = {"orchestration": {"run_id": str(uuid.uuid4())}}
    assert (await OrchestrationRecoveryService().build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))).sessions == ()


@pytest.mark.asyncio
async def test_retry_ownership_accepts_missing_session_metadata_when_action_targets_session(
    db_session, test_project, test_agent,
):
    goal, _, action, task, session = await _owned(db_session, test_project, test_agent)
    action.action_type = "retry_task"
    action.request = {"task_id": str(task.id)}
    action.target_type, action.target_id = "session", session.id
    action.dispatch_contract = {"owner": "orchestration_recovery"}
    session.metadata_ = {"attempt": {}}

    snapshot = await OrchestrationRecoveryService().build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))

    assert snapshot and [item.session_id for item in snapshot.sessions] == [session.id]


@pytest.mark.asyncio
async def test_retry_ownership_rejects_wrong_session_target(db_session, test_project, test_agent):
    goal, _, action, task, session = await _owned(db_session, test_project, test_agent)
    action.action_type = "retry_task"
    action.request = {"task_id": str(task.id)}
    action.target_type, action.target_id = "task", task.id
    session.metadata_ = {"orchestration": {"action_id": str(action.id)}, "attempt": {}}

    snapshot = await OrchestrationRecoveryService().build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))

    assert snapshot and snapshot.sessions == ()


@pytest.mark.asyncio
async def test_current_verified_retry_ignores_older_unlinked_failure(db_session, test_project, test_agent, monkeypatch):
    goal, run, action, task, current = await _owned(
        db_session, test_project, test_agent, session_status="pending", task_status="in_progress",
    )
    action.action_type = "retry_task"
    action.request = {"task_id": str(task.id)}
    action.target_type, action.target_id = "session", current.id
    action.dispatch_contract = {"owner": "orchestration_recovery"}
    async def no_commit(): return None
    monkeypatch.setattr(db_session, "commit", no_commit)
    assert await _require_runnable_session(db_session, current.id, current.runner_task_id)
    current.status = task.status = "failed"
    older = Session(project_id=test_project.id, task_id=task.id, agent_id=test_agent.id, adapter_type="cli",
        status="failed", created_at=datetime.now(timezone.utc) - timedelta(seconds=1), metadata_={})
    db_session.add(older)
    await db_session.flush()

    created = await OrchestrationService().recover_run(db_session, run.id, baseline_ready=True)

    assert created == 0 and not run.active_blockers


@pytest.mark.asyncio
async def test_direct_retry_target_is_not_verified_ownership(db_session, test_project, test_agent):
    goal, run, recovery_action, task, older = await _owned(
        db_session, test_project, test_agent, session_status="failed", task_status="failed",
    )
    older.created_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    legacy_action = OrchestrationAction(run_id=run.id, idempotency_key=f"legacy:{uuid.uuid4()}",
        action_type="retry_task", request={"task_id": str(task.id)}, status="completed", target_type="task", target_id=task.id)
    db_session.add(legacy_action)
    await db_session.flush()
    task.metadata_ = {**task.metadata_, "orchestration": {**task.metadata_["orchestration"], "action_id": str(legacy_action.id)}}
    current = Session(project_id=test_project.id, task_id=task.id, agent_id=test_agent.id, adapter_type="cli",
        status="failed", metadata_={"orchestration": {"action_id": str(legacy_action.id)}})
    db_session.add(current)
    await db_session.flush()

    assert await OrchestrationRecoveryService().resolve_owned_session(db_session, current.id) is None
    created = await OrchestrationService().recover_run(db_session, run.id, baseline_ready=True)
    assert created == 1


@pytest.mark.asyncio
async def test_current_fenced_worker_ownership_failure_stays_out_of_legacy_recovery(db_session, test_project, test_agent, monkeypatch):
    goal, run, _, task, session = await _owned(
        db_session, test_project, test_agent, session_status="pending", task_status="in_progress",
    )
    async def no_commit(): return None
    monkeypatch.setattr(db_session, "commit", no_commit)
    assert await _require_runnable_session(db_session, session.id, session.runner_task_id)
    task.status = session.status = "failed"
    created = await OrchestrationService().recover_run(db_session, run.id, baseline_ready=True)
    assert created == 0 and task.status == "failed" and session.status == "failed"


@pytest.mark.asyncio
async def test_current_legacy_failure_is_not_hidden_by_older_fenced_worker_ownership_attempt(db_session, test_project, test_agent, monkeypatch):
    goal, run, _, task, older = await _owned(
        db_session, test_project, test_agent, session_status="pending", task_status="in_progress",
    )
    async def no_commit(): return None
    monkeypatch.setattr(db_session, "commit", no_commit)
    assert await _require_runnable_session(db_session, older.id, older.runner_task_id)
    task.status = older.status = "failed"
    older.created_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    current = Session(project_id=test_project.id, task_id=task.id, agent_id=test_agent.id, adapter_type="api",
        status="failed", metadata_={})
    db_session.add(current); await db_session.flush()
    created = await OrchestrationService().recover_run(db_session, run.id, baseline_ready=True)
    assert created == 1 and current.status == "failed" and older.status == "failed"


@pytest.mark.asyncio
async def test_snapshot_sessions_are_deterministically_ordered(db_session, test_project, test_agent):
    goal, _, action, task, first = await _owned(db_session, test_project, test_agent)
    other_task = Task(project_id=test_project.id, title="other", description="other", status="in_progress", assigned_to=test_agent.id,
        metadata_={"orchestration": {"run_id": str(goal.id)}})
    db_session.add(other_task); await db_session.flush()
    other_action = OrchestrationAction(run_id=action.run_id, idempotency_key=f"seed:{uuid.uuid4()}", action_type="retry_task", request={"task_id": str(other_task.id)}, status="completed")
    db_session.add(other_action); await db_session.flush()
    other_task.metadata_ = {"orchestration": {"run_id": str(action.run_id), "action_id": str(other_action.id)}}
    second = Session(project_id=test_project.id, agent_id=test_agent.id, task_id=other_task.id, adapter_type="cli",
        status="running", runner_task_id="runner-2", metadata_={"orchestration": {"action_id": str(other_action.id)}, "attempt": {}})
    db_session.add(second); await db_session.flush()
    other_action.target_type, other_action.target_id = "session", second.id
    other_action.dispatch_contract = {"owner": "orchestration_recovery"}
    snapshot = await OrchestrationRecoveryService().build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))
    assert [item.session_id for item in snapshot.sessions] == sorted(
        (first.id, second.id), key=lambda item: str(item)
    )


@pytest.mark.asyncio
async def test_unresolved_memory_remains_visible_while_owned_session_is_assessed(db_session, test_project, test_agent):
    goal, run, _, _, session = await _owned(db_session, test_project, test_agent)
    section = OrchestrationMemorySection(project_id=goal.project_id, goal_id=goal.id, run_id=run.id,
        section_key="ambiguous-memory", title="memory", body="fact", created_by="test", provenance={})
    db_session.add(section)
    await db_session.flush()
    snapshot = await OrchestrationRecoveryService().build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))
    result = await OrchestrationRecoveryService().apply_goal_recovery(db_session, snapshot, {
        session.id: RunnerObservation(session.runner_task_id, "active", datetime.now(timezone.utc))})
    waits = list(await db_session.scalars(select(OrchestrationWait).where(OrchestrationWait.run_id == run.id)))
    assert result.disposition == "live" and result.action_id is None and len(waits) == 1
    assert section.fact_status == "unverified" and "memory_upgrade_reconciled" not in (run.supervision_state or {})
    snapshot = await OrchestrationRecoveryService().build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))
    replay = await OrchestrationRecoveryService().apply_goal_recovery(db_session, snapshot, {
        session.id: RunnerObservation(session.runner_task_id, "active", datetime.now(timezone.utc))})
    waits = list(await db_session.scalars(select(OrchestrationWait).where(OrchestrationWait.run_id == run.id)))
    marker = run.supervision_state["memory_upgrade_reconciled"]
    assert replay.disposition == "live" and len(waits) == 2
    assert marker["outcome"] == "unresolved" and marker["wait_id"] in {str(wait.id) for wait in waits}


@pytest.mark.asyncio
async def test_owned_session_promotion_marks_once_without_starving_worker(db_session, test_project, test_agent):
    goal, run, _, _, session = await _owned(db_session, test_project, test_agent)
    decision = OrchestrationDecision(run_id=run.id, decision_type="test", input_snapshot={}, parsed_decision={}, validator_status="accepted")
    db_session.add(decision); await db_session.flush()
    section = OrchestrationMemorySection(project_id=goal.project_id, goal_id=goal.id, run_id=run.id,
        section_key="accepted-memory", title="memory", body="fact", created_by="test", provenance={"decision_id": str(decision.id)})
    db_session.add(section); await db_session.flush()
    service = OrchestrationRecoveryService()
    for _ in range(2):
        snapshot = await service.build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))
        result = await service.apply_goal_recovery(db_session, snapshot, {
            session.id: RunnerObservation(session.runner_task_id, "active", datetime.now(timezone.utc))})
        assert result.disposition == "live"
    assert section.fact_status == "accepted"
    assert run.supervision_state["memory_upgrade_reconciled"] == {
        "run_id": str(run.id), "outcome": "promoted", "unresolved": 0,
    }


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_old_sqlite_and_celery_live_workers_are_adopted(db_session, test_project, test_agent, monkeypatch):
    from huddleroom.workers import celery_app, task_runner
    monkeypatch.setattr(task_runner, "is_task_live", lambda _id: True)
    class Inspector:
        def active(self): return {"worker": [{"id": "celery-live"}]}
        def reserved(self): return {}
        def scheduled(self): return {}
    monkeypatch.setattr(celery_app, "app", SimpleNamespace(control=SimpleNamespace(inspect=lambda: Inspector())))
    sqlite, *_ = await _applied(db_session, test_project, test_agent,
        runner="sqlite-live", backend=recovery_tasks.observe_runner("sqlite-live", is_sqlite=True))
    celery, *_ = await _applied(db_session, test_project, test_agent,
        runner="celery-live", backend=recovery_tasks.observe_runner("celery-live", is_sqlite=False))
    assert sqlite.classification == celery.classification == "live"
    assert sqlite.disposition == celery.disposition == "live" and sqlite.wait_id and celery.wait_id


@pytest.mark.asyncio
async def test_old_sqlite_live_worker_is_adopted(db_session, test_project, test_agent, monkeypatch):
    from huddleroom.workers import task_runner
    monkeypatch.setattr(task_runner, "is_task_live", lambda _id: True)
    sqlite, *_ = await _applied(
        db_session, test_project, test_agent,
        runner="sqlite-live", backend=recovery_tasks.observe_runner("sqlite-live", is_sqlite=True),
    )
    assert sqlite.classification == sqlite.disposition == "live" and sqlite.wait_id


@pytest.mark.asyncio
async def test_paused_live_worker_is_adopted_without_dispatch(db_session, test_project, test_agent):
    result, *_ = await _applied(db_session, test_project, test_agent, goal_status="paused", backend="started")
    assert result.classification == "live" and result.disposition == "control_retained" and result.action_id is None


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_sqlite_absence_and_celery_pending_create_unknown_effect_wait(db_session, test_project, test_agent, monkeypatch):
    from huddleroom.workers import celery_app, task_runner
    monkeypatch.setattr(task_runner, "is_task_live", lambda _id: False)
    class Inspector:
        def active(self): return {}
        def reserved(self): return {}
        def scheduled(self): return {}
    app = SimpleNamespace(control=SimpleNamespace(inspect=lambda: Inspector()), AsyncResult=lambda _id: SimpleNamespace(state="PENDING"))
    monkeypatch.setattr(celery_app, "app", app)
    sqlite, *_ = await _applied(db_session, test_project, test_agent, status="pending",
        runner="sqlite-missing", backend=recovery_tasks.observe_runner("sqlite-missing", is_sqlite=True))
    celery, *_ = await _applied(db_session, test_project, test_agent, status="pending",
        runner="celery-pending", backend=recovery_tasks.observe_runner("celery-pending", is_sqlite=False))
    assert sqlite.classification == celery.classification == "unknown_external_effect"
    assert sqlite.wait_id and celery.wait_id


@pytest.mark.asyncio
async def test_sqlite_absence_creates_unknown_effect_wait(db_session, test_project, test_agent, monkeypatch):
    from huddleroom.workers import task_runner
    monkeypatch.setattr(task_runner, "is_task_live", lambda _id: False)
    sqlite, *_ = await _applied(
        db_session, test_project, test_agent, status="pending",
        runner="sqlite-missing", backend=recovery_tasks.observe_runner("sqlite-missing", is_sqlite=True),
    )
    assert sqlite.classification == "unknown_external_effect"
    assert sqlite.wait_id


@pytest.mark.asyncio
async def test_registry_loss_retries_only_durably_pre_effect_owned_attempt(
    db_session, test_project, test_agent,
):
    result, _, _, _, _, _ = await _applied(
        db_session, test_project, test_agent, status="pending", backend="unknown",
        attempt={
            "claimed_runner_task_id": "runner-1", "attempt_version": 1,
            "effect_state": "not_started", "usage_complete": False,
        },
    )
    assert result.classification == result.disposition == "interrupted_safe"
    assert result.action_id is not None and result.wait_id is None


@pytest.mark.asyncio
async def test_interrupted_safe_settles_source_budget_before_replacement_reservation(
    db_session, test_project, test_agent,
):
    goal, run, source_action, task, session = await _owned(
        db_session, test_project, test_agent, session_status="pending",
        attempt={
            "claimed_runner_task_id": "runner-1", "attempt_version": 1,
            "effect_state": "not_started", "usage_complete": False,
        },
    )
    goal.budget = {"caps": {"max_tokens": 20}}
    run.phase = "authorized"
    source_action.budget_ledger = {
        "allocation": {"max_tokens": "10"}, "reserved": {}, "committed": {"max_tokens": "10"},
        "consumed": {}, "usage_state": "known", "enforceability": "enforceable",
    }
    task.metadata_ = {
        **task.metadata_, "orchestration_contract": {"budget": {"caps": {"max_tokens": 10}}},
    }

    service = OrchestrationRecoveryService()
    snapshot = await service.build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))
    result = await service.apply_goal_recovery(db_session, snapshot, {
        session.id: RunnerObservation(session.runner_task_id, "unknown", datetime.now(timezone.utc)),
    })

    assert result.classification == result.disposition == "interrupted_safe"
    assert source_action.budget_ledger["consumed"] == {}
    assert source_action.budget_ledger["final_observation"] == (
        f"action:{source_action.id}:session:{session.id}:interrupted_safe:final"
    )
    replacement = await db_session.get(OrchestrationAction, result.action_id)
    assert replacement is not None and replacement.budget_ledger["committed"] == {"max_tokens": "10"}


@pytest.mark.asyncio
async def test_registry_loss_with_persisted_session_output_needs_attention(db_session, test_project, test_agent):
    goal, _, _, _, session = await _owned(
        db_session, test_project, test_agent, session_status="pending",
        attempt={
            "claimed_runner_task_id": "runner-1", "attempt_version": 1,
            "effect_state": "not_started", "usage_complete": False,
        },
    )
    session.output = "persisted provider output"
    snapshot = await OrchestrationRecoveryService().build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))
    result = await OrchestrationRecoveryService().apply_goal_recovery(db_session, snapshot, {
        session.id: RunnerObservation(session.runner_task_id, "unknown", datetime.now(timezone.utc)),
    })
    assert result.classification == result.disposition == "unknown_external_effect"
    assert result.action_id is None and result.wait_id is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("attempt,provider", [
    ({"attempt_version": 1, "effect_state": "not_started", "usage_complete": False}, None),
    ({"claimed_runner_task_id": "stale", "attempt_version": 1, "effect_state": "not_started", "usage_complete": False}, None),
    ({"claimed_runner_task_id": "runner-1", "attempt_version": True, "effect_state": "not_started", "usage_complete": False}, None),
    ({"claimed_runner_task_id": "runner-1", "attempt_version": 0, "effect_state": "not_started", "usage_complete": False}, None),
    ({"claimed_runner_task_id": "runner-1", "attempt_version": 1, "effect_state": "not_started"}, None),
    ({"effect_state": "started"}, None),
    ({"claimed_runner_task_id": "runner-1", "attempt_version": 1, "effect_state": "not_started", "usage_complete": False, "provider_session_id": "provider"}, "provider"),
    ({"claimed_runner_task_id": "runner-1", "attempt_version": 1, "effect_state": "not_started", "usage_complete": False, "result_status": "failed"}, None),
    ({"claimed_runner_task_id": "runner-1", "attempt_version": 1, "effect_state": "not_started", "usage_complete": False, "token_count_in": 1}, None),
])
async def test_registry_loss_with_effect_or_non_idempotent_evidence_needs_attention(
    db_session, test_project, test_agent, attempt, provider,
):
    result, *_ = await _applied(
        db_session, test_project, test_agent, status="pending", backend="unknown", attempt=attempt,
        provider=provider,
    )
    assert result.classification == result.disposition == "unknown_external_effect"
    assert result.action_id is None and result.wait_id is not None


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_celery_active_reserved_started_and_retrying_are_live(db_session, test_project, test_agent, monkeypatch):
    from huddleroom.workers import celery_app
    for state, task_state in (("active", None), ("reserved", None), ("started", "STARTED"), ("retrying", "RETRY")):
        class Inspector:
            def active(self): return {"worker": [{"id": state}]} if state == "active" else {}
            def reserved(self): return {"worker": [{"id": state}]} if state == "reserved" else {}
            def scheduled(self): return {}
        monkeypatch.setattr(celery_app, "app", SimpleNamespace(control=SimpleNamespace(inspect=lambda: Inspector()),
            AsyncResult=lambda _id: SimpleNamespace(state=task_state or "PENDING")))
        result, *_ = await _applied(db_session, test_project, test_agent, runner=state,
            backend=recovery_tasks.observe_runner(state, is_sqlite=False))
        assert result.classification == "live"


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_celery_terminal_states_reconcile_without_replay(db_session, test_project, test_agent, monkeypatch):
    from huddleroom.workers import celery_app
    for celery_state, expected in (("SUCCESS", "terminal_success"), ("FAILURE", "terminal_failure")):
        class Inspector:
            def active(self): return {}
            def reserved(self): return {}
            def scheduled(self): return {}
        monkeypatch.setattr(celery_app, "app", SimpleNamespace(control=SimpleNamespace(inspect=lambda: Inspector()),
            AsyncResult=lambda _id: SimpleNamespace(state=celery_state)))
        observed = recovery_tasks.observe_runner(celery_state, is_sqlite=False)
        result, _, run, _, _, _ = await _applied(db_session, test_project, test_agent, status="completed",
            task_status="completed", runner=celery_state, backend=observed)
        assert observed == expected and result.classification == "terminal" and result.action_id and result.wait_id is None
        assert await db_session.scalar(select(OrchestrationWait).where(OrchestrationWait.run_id == run.id)) is None


@pytest.mark.asyncio
async def test_backend_success_without_database_result_is_unknown(db_session, test_project, test_agent):
    result, *_ = await _applied(db_session, test_project, test_agent, backend="terminal_success")
    assert result.classification == "unknown_external_effect" and result.action_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("unknown", "terminal_success", "terminal_failure"))
async def test_failed_database_attempt_reconciles_without_backend_terminal_proof(
    db_session, test_project, test_agent, backend,
):
    result, *_ = await _applied(
        db_session, test_project, test_agent, status="failed", task_status="failed", backend=backend,
        resumable=False, attempt={"effect_state": "started", "usage_complete": False},
    )
    action = await db_session.get(OrchestrationAction, result.action_id)
    assert result.classification == result.disposition == "terminal" and result.wait_id is None
    assert action.action_type == "noop"


@pytest.mark.asyncio
async def test_database_terminal_state_precedes_stale_live_backend_state(db_session, test_project, test_agent):
    result, *_ = await _applied(db_session, test_project, test_agent, status="completed", backend="active")
    assert result.classification == "terminal" and result.action_id is not None and result.wait_id is None


@pytest.mark.asyncio
async def test_two_competing_workers_only_one_claim_enters_adapter(
    concurrent_sessions, tmp_path, monkeypatch,
):
    first, second = concurrent_sessions
    project = Project(name=f"claim-race-{uuid.uuid4()}", description="", workspace_path=str(tmp_path), config={})
    agent = Agent(name=f"claim-agent-{uuid.uuid4()}", role="developer", provider="openai",
                  model="gpt-4o-mini", adapter_type="api", capabilities=[], config={})
    first.add_all((project, agent))
    await first.flush()
    _, _, _, _, session = await _owned(first, project, agent, session_status="pending")
    session.adapter_type = "api"
    await first.commit()
    session_id, runner_task_id = session.id, session.runner_task_id
    barrier = asyncio.Barrier(2)

    async def claim(db, runner):
        await asyncio.wait_for(barrier.wait(), timeout=1)
        return await _require_runnable_session(db, session_id, runner)

    first_claim, second_claim = await asyncio.wait_for(
        asyncio.gather(claim(first, runner_task_id), claim(second, runner_task_id)), timeout=5,
    )
    assert sorted((first_claim, second_claim)) == [False, True]
    winner, loser = (first, second) if first_claim else (second, first)
    winner_runner = runner_task_id
    await loser.rollback()
    provider_calls = []
    async def provider_spy(*_args, **_kwargs): provider_calls.append("provider")
    monkeypatch.setattr(ApiAdapter, "_run_with_retry", provider_spy)
    await ApiAdapter().run(session_id, winner, runner_task_id=winner_runner)
    assert provider_calls == ["provider"]


@pytest.mark.asyncio
async def test_sqlite_claim_retries_after_rolling_back_locked_transaction(
    concurrent_sessions, tmp_path, monkeypatch,
):
    db_session, _ = concurrent_sessions
    project = Project(name=f"claim-retry-{uuid.uuid4()}", description="", workspace_path=str(tmp_path), config={})
    agent = Agent(name=f"claim-retry-agent-{uuid.uuid4()}", role="developer", provider="openai",
                  model="gpt-4o-mini", adapter_type="api", capabilities=[], config={})
    db_session.add_all((project, agent)); await db_session.flush()
    _, _, _, _, session = await _owned(db_session, project, agent, session_status="pending")
    session_id = session.id
    await db_session.commit()
    calls = 0

    async def claim_once(db, *_args):
        nonlocal calls
        calls += 1
        if calls == 1:
            await db.execute(select(Session.id).where(Session.id == session_id))
            raise OperationalError("update", {}, Exception("database is locked"))
        assert not db.in_transaction()
        return True

    monkeypatch.setattr("huddleroom.workers.session_tasks._require_runnable_session_once", claim_once)
    assert await _require_runnable_session(db_session, session_id, session.runner_task_id) is True
    assert calls == 2


@pytest.mark.asyncio
async def test_sqlite_claim_reraises_final_lock_after_rolling_back_transaction(
    concurrent_sessions, tmp_path, monkeypatch,
):
    db_session, _ = concurrent_sessions
    project = Project(name=f"claim-final-lock-{uuid.uuid4()}", description="", workspace_path=str(tmp_path), config={})
    agent = Agent(name=f"claim-final-lock-agent-{uuid.uuid4()}", role="developer", provider="openai",
                  model="gpt-4o-mini", adapter_type="api", capabilities=[], config={})
    db_session.add_all((project, agent)); await db_session.flush()
    _, _, _, _, session = await _owned(db_session, project, agent, session_status="pending")
    session_id = session.id
    await db_session.commit()
    calls = 0

    async def claim_once(db, *_args):
        nonlocal calls
        calls += 1
        await db.execute(select(Session.id).where(Session.id == session_id))
        raise OperationalError("update", {}, Exception("database is locked"))

    monkeypatch.setattr("huddleroom.workers.session_tasks._require_runnable_session_once", claim_once)
    with pytest.raises(OperationalError, match="database is locked"):
        await _require_runnable_session(db_session, session_id, session.runner_task_id)
    assert calls == 3
    assert not db_session.in_transaction()


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type", ("api", "cli"))
async def test_orchestration_pending_adapter_cannot_bypass_claim(
    db_session, test_project, test_agent, monkeypatch, adapter_type,
):
    _, _, _, _, session = await _owned(db_session, test_project, test_agent, session_status="pending")
    session.adapter_type = adapter_type
    provider_calls = []

    async def provider_spy(*_args, **_kwargs):
        provider_calls.append("provider")

    monkeypatch.setattr(ApiAdapter, "_run_with_retry", provider_spy)
    async def subprocess_spy(*_args, **_kwargs):
        provider_calls.append("subprocess")
        raise AssertionError("subprocess must not start")
    monkeypatch.setattr("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", subprocess_spy)
    adapter = ApiAdapter() if adapter_type == "api" else CliAdapter()
    await adapter.run(session.id, db_session, runner_task_id=session.runner_task_id)
    assert session.status == "pending" and session.input_context == {}
    assert provider_calls == []


@pytest.mark.asyncio
async def test_orchestration_running_adapter_requires_matching_runner_claim(db_session, test_project, test_agent):
    _, _, _, _, session = await _owned(db_session, test_project, test_agent, attempt={"claimed_runner_task_id": "old"})
    await ApiAdapter().run(session.id, db_session)
    assert session.status == "running" and session.input_context == {}


@pytest.mark.asyncio
async def test_invalid_orchestration_lineage_cannot_reach_api_provider(db_session, test_project, test_agent, monkeypatch):
    _, _, action, _, session = await _owned(db_session, test_project, test_agent, session_status="pending")
    action.target_type, action.target_id = "session", session.id
    action.dispatch_contract = {"owner": "not_orchestration_recovery"}
    provider_calls = []

    async def provider_spy(*_args, **_kwargs):
        provider_calls.append("provider")

    monkeypatch.setattr(ApiAdapter, "_run_with_retry", provider_spy)
    await ApiAdapter().run(session.id, db_session, runner_task_id="runner-1")
    assert provider_calls == [] and session.status == "pending"


@pytest.mark.asyncio
async def test_invalid_orchestration_lineage_cannot_reach_cli_subprocess(db_session, test_project, test_agent, monkeypatch):
    _, _, action, _, session = await _owned(db_session, test_project, test_agent, session_status="pending")
    action.target_type, action.target_id = "session", session.id
    action.dispatch_contract = {"owner": "not_orchestration_recovery"}

    async def subprocess_spy(*_args, **_kwargs):
        raise AssertionError("subprocess must not start")

    monkeypatch.setattr("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", subprocess_spy)
    await CliAdapter().run(session.id, db_session, runner_task_id="runner-1")
    assert session.status == "pending"


@pytest.mark.asyncio
async def test_manual_pending_adapter_path_remains_supported(db_session, test_project, test_agent, monkeypatch):
    session = Session(project_id=test_project.id, agent_id=test_agent.id, adapter_type="api", status="pending", metadata_={})
    db_session.add(session); await db_session.flush()
    async def no_call(*_args): return None
    monkeypatch.setattr(ApiAdapter, "_run_with_retry", no_call)
    await ApiAdapter().run(session.id, db_session)
    assert session.status == "running" and session.started_at is not None


@pytest.mark.asyncio
async def test_attempt_marker_tracks_claim_effect_provider_usage_and_result(db_session, test_project, test_agent, monkeypatch):
    _, _, _, _, session = await _owned(db_session, test_project, test_agent, session_status="pending")
    async def no_commit(): return None
    monkeypatch.setattr(db_session, "commit", no_commit)
    assert await _require_runnable_session(db_session, session.id, "runner-1")
    await mark_attempt_effect_started(db_session, session.id, "runner-1")
    session.status, session.provider_session_id = "failed", "provider-1"; session.metadata_["token_usage_complete"] = True
    await _mark_attempt_result(db_session, session.id, "runner-1")
    attempt = session.metadata_["attempt"]
    assert attempt["effect_state"] == "started" and attempt["provider_session_id"] == "provider-1" and attempt["usage_complete"] is True and attempt["result_status"] == "failed"


@pytest.mark.asyncio
async def test_stale_worker_cannot_update_new_attempt_marker(db_session, test_project, test_agent):
    _, _, _, _, session = await _owned(db_session, test_project, test_agent, attempt={"claimed_runner_task_id": "new", "effect_state": "not_started"})
    await mark_attempt_effect_started(db_session, session.id, "old")
    assert session.metadata_["attempt"]["effect_state"] == "not_started"


@pytest.mark.asyncio
async def test_stale_worker_cas_write_leaves_new_attempt_byte_for_byte_unchanged(
    db_session, concurrent_sessions, test_project, test_agent,
):
    _, _, _, _, session = await _owned(
        db_session, test_project, test_agent,
        attempt={"claimed_runner_task_id": "old", "attempt_version": 1, "effect_state": "not_started"},
    )
    await db_session.commit()
    stale, replacement = concurrent_sessions
    assert await stale.get(Session, session.id) is not None  # retain the old marker in this worker
    replacement_session = await replacement.get(Session, session.id)
    replacement_session.runner_task_id = "new"
    replacement_session.metadata_ = {
        **replacement_session.metadata_,
        "attempt": {"claimed_runner_task_id": "new", "attempt_version": 2, "effect_state": "not_started"},
    }
    await replacement.commit()

    assert await mark_attempt_effect_started(stale, session.id, "old") is False
    await replacement.refresh(replacement_session)
    verifier = replacement_session
    assert verifier.metadata_["attempt"] == {
        "claimed_runner_task_id": "new", "attempt_version": 2, "effect_state": "not_started",
    }


@pytest.mark.asyncio
async def test_two_database_workers_share_one_runner_claim(
    db_session, concurrent_sessions, test_project, test_agent, monkeypatch,
):
    _, _, _, _, session = await _owned(db_session, test_project, test_agent, session_status="pending")
    session.adapter_type = "api"
    await db_session.commit()
    first, second = concurrent_sessions
    assert await _require_runnable_session(first, session.id, "runner-1")
    assert not await _require_runnable_session(second, session.id, "runner-2")
    provider_calls = []
    async def provider_spy(*_args, **_kwargs): provider_calls.append("provider")
    monkeypatch.setattr(ApiAdapter, "_run_with_retry", provider_spy)
    await ApiAdapter().run(session.id, first, runner_task_id="runner-1")
    await ApiAdapter().run(session.id, second, runner_task_id="runner-2")
    assert provider_calls == ["provider"]


@pytest.mark.asyncio
async def test_stale_terminal_writer_cannot_publish_result_or_usage(
    db_session, concurrent_sessions, test_project, test_agent,
):
    _, _, _, _, session = await _owned(
        db_session, test_project, test_agent,
        attempt={"claimed_runner_task_id": "old", "attempt_version": 2, "effect_state": "started"},
    )
    await db_session.commit()
    stale, replacement = concurrent_sessions
    stale_session = await stale.get(Session, session.id)
    stale_session.status = "failed"
    stale_session.provider_session_id = "stale-provider"
    stale_session.metadata_ = {**stale_session.metadata_, "token_usage_complete": True}
    replacement_session = await replacement.get(Session, session.id)
    replacement_session.runner_task_id = "new"
    replacement_session.metadata_ = {
        **replacement_session.metadata_,
        "attempt": {"claimed_runner_task_id": "new", "attempt_version": 3, "effect_state": "started"},
    }
    await replacement.commit()

    assert await _mark_attempt_result(stale, session.id, "old") is False
    await stale.commit()  # A loser must also be safe if its caller commits afterward.
    await replacement.refresh(replacement_session)
    assert replacement_session.status == "running"
    assert replacement_session.provider_session_id is None
    assert replacement_session.metadata_["attempt"] == {
        "claimed_runner_task_id": "new", "attempt_version": 3, "effect_state": "started",
    }


@pytest.mark.asyncio
async def test_stale_project_rejection_cannot_replace_new_attempt_marker(
    db_session, concurrent_sessions, test_project, test_agent,
):
    _, _, _, _, session = await _owned(
        db_session, test_project, test_agent,
        attempt={"claimed_runner_task_id": "old", "attempt_version": 2, "effect_state": "started"},
    )
    await db_session.commit()
    stale, replacement = concurrent_sessions
    assert await stale.get(Session, session.id) is not None
    replacement_session = await replacement.get(Session, session.id)
    replacement_session.runner_task_id = "new"
    replacement_session.metadata_ = {
        **replacement_session.metadata_,
        "attempt": {"claimed_runner_task_id": "new", "attempt_version": 3, "effect_state": "started"},
    }
    await replacement.commit()

    assert await mark_attempt_project_not_runnable(stale, session.id, "old", "project_not_runnable: reset") is False
    await replacement.refresh(replacement_session)
    assert replacement_session.status == "running"
    assert replacement_session.metadata_["attempt"] == {
        "claimed_runner_task_id": "new", "attempt_version": 3, "effect_state": "started",
    }


@pytest.mark.asyncio
async def test_manual_pending_session_is_claimed_once(db_session, test_project, test_agent, monkeypatch):
    session = Session(project_id=test_project.id, agent_id=test_agent.id, adapter_type="api", status="pending", metadata_={})
    db_session.add(session)
    await db_session.flush()
    async def no_commit(): return None
    monkeypatch.setattr(db_session, "commit", no_commit)
    assert await _require_runnable_session(db_session, session.id)
    assert not await _require_runnable_session(db_session, session.id)


@pytest.mark.asyncio
async def test_api_revalidates_project_immediately_before_orchestration_provider_call(
    db_session, test_project, test_agent, monkeypatch,
):
    _, _, _, _, session = await _owned(
        db_session, test_project, test_agent,
        attempt={"claimed_runner_task_id": "runner-1", "attempt_version": 1, "effect_state": "not_started"},
    )
    order = []

    async def runnable(*_args):
        order.append("project")

    async def completion(**_kwargs):
        order.append("provider")
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    async def tool_loop(*, completion_fn, messages, **_kwargs):
        await completion_fn(messages=messages)
        return "done"

    async def no_commit():
        return None

    monkeypatch.setattr("huddleroom.adapters.api_adapter.ProjectService.require_runnable_project", runnable)
    monkeypatch.setattr("huddleroom.adapters.api_adapter.run_tool_loop", tool_loop)
    monkeypatch.setattr("litellm.acompletion", completion)
    monkeypatch.setattr(db_session, "commit", no_commit)
    await ApiAdapter()._run_with_retry(
        test_agent, session, [{"role": "user", "content": "run"}], "run", db_session, runner_task_id="runner-1",
    )
    assert order == ["project", "provider"]


@pytest.mark.asyncio
async def test_definitive_pre_effect_failure_uses_existing_retry_key_once(db_session, test_project, test_agent):
    result, _, _, _, task, _ = await _applied(
        db_session, test_project, test_agent, status="pending", backend="terminal_failure",
        attempt={
            "claimed_runner_task_id": "runner-1", "attempt_version": 1,
            "effect_state": "not_started", "usage_complete": False,
        },
    )
    assert result.classification == "interrupted_safe" and result.action_id is not None
    action = await db_session.get(OrchestrationAction, result.action_id)
    assert action.idempotency_key.endswith(f"task:{task.id}")
    assert action.dispatch_contract["owner"] == "orchestration_recovery"
    assert action.dispatch_contract["recovery_disposition"] == "interrupted_safe"
    assert action.dispatch_contract["task_id"] == str(task.id)


@pytest.mark.asyncio
async def test_exact_resume_reuses_same_session_provider_and_retry_key_once(db_session, test_project, test_agent, monkeypatch):
    test_agent.cli_runtime = "claude_code"
    goal, _, _, task, session = await _owned(db_session, test_project, test_agent, session_status="failed", task_status="failed",
        resumable=True, provider="provider", attempt={"effect_state": "started", "usage_complete": True,
            "result_status": "failed", "provider_session_id": "provider", "token_count_in": 1, "token_count_out": 1})
    session.metadata_ = {**session.metadata_, "token_usage_complete": True,
                         "_launch_fingerprint": _launch_fingerprint(session, task, test_agent, test_project)}
    resumed = []
    original_resume = SessionService.resume

    async def resume_spy(resume_service, db, session_id):
        resumed.append(session_id)
        return await original_resume(resume_service, db, session_id)

    monkeypatch.setattr(SessionService, "resume", resume_spy)
    service = OrchestrationRecoveryService()
    snapshot = await service.build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))
    result = await service.apply_goal_recovery(db_session, snapshot, {
        session.id: RunnerObservation(session.runner_task_id, "terminal_failure", datetime.now(timezone.utc))})
    action = await db_session.get(OrchestrationAction, result.action_id)
    assert result.classification == "resume_exact" and resumed == [session.id]
    assert action.idempotency_key.endswith(f"task:{task.id}")
    assert session.provider_session_id == "provider" and action.target_id == session.id


@pytest.mark.asyncio
async def test_fully_proven_failed_cli_resumes_when_backend_is_unavailable(db_session, test_project, test_agent, monkeypatch):
    test_agent.cli_runtime = "claude_code"
    goal, _, _, task, session = await _owned(db_session, test_project, test_agent, session_status="failed", task_status="failed",
        resumable=True, provider="provider", attempt={"effect_state": "started", "usage_complete": True,
            "result_status": "failed", "provider_session_id": "provider", "token_count_in": 1, "token_count_out": 1})
    session.metadata_ = {**session.metadata_, "token_usage_complete": True,
                         "_launch_fingerprint": _launch_fingerprint(session, task, test_agent, test_project)}
    resumed = []
    original_resume = SessionService.resume
    async def resume_spy(service, db, session_id):
        resumed.append(session_id); return await original_resume(service, db, session_id)
    monkeypatch.setattr(SessionService, "resume", resume_spy)
    snapshot = await OrchestrationRecoveryService().build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))
    result = await OrchestrationRecoveryService().apply_goal_recovery(db_session, snapshot, {
        session.id: RunnerObservation(session.runner_task_id, "unknown", datetime.now(timezone.utc))})
    assert result.classification == "resume_exact" and resumed == [session.id]


@pytest.mark.asyncio
async def test_exact_resume_uses_existing_budget_accounting_and_complete_contract(db_session, test_project, test_agent):
    test_agent.cli_runtime = "claude_code"
    goal, run, source_action, task, session = await _owned(db_session, test_project, test_agent,
        session_status="failed", task_status="failed", resumable=True, provider="provider",
        attempt={"effect_state": "started", "usage_complete": True, "result_status": "failed",
                 "provider_session_id": "provider", "token_count_in": 1, "token_count_out": 1})
    goal.budget = {"caps": {"max_tokens": 100}}
    run.phase = "authorized"
    task.metadata_ = {**task.metadata_, "orchestration_contract": {"budget": {"caps": {"max_tokens": 10}}}}
    session.metadata_ = {**session.metadata_, "token_usage_complete": True,
                         "_launch_fingerprint": _launch_fingerprint(session, task, test_agent, test_project)}
    snapshot = await OrchestrationRecoveryService().build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))
    result = await OrchestrationRecoveryService().apply_goal_recovery(db_session, snapshot, {
        session.id: RunnerObservation(session.runner_task_id, "terminal_failure", datetime.now(timezone.utc))})
    action = await db_session.get(OrchestrationAction, result.action_id)
    assert action.dispatch_contract["origin"] == "recovery_resume"
    assert action.dispatch_contract["provider_session_id"] == "provider"
    assert action.dispatch_contract["source_runner_task_id"] == "runner-1"
    assert action.dispatch_contract["goal_id"] == str(goal.id)
    assert action.dispatch_contract["run_id"] == str(run.id)
    assert action.dispatch_contract["source_action_id"] == str(source_action.id)
    assert action.dispatch_contract["task_id"] == str(task.id)
    assert action.budget_ledger["committed"] == {"max_tokens": "10"}
    assert session.metadata_["_run_config"]["max_tokens"] == 10


@pytest.mark.asyncio
async def test_unproven_failed_attempt_reconciles_terminal_noop(db_session, test_project, test_agent):
    result, _, run, _, _, _ = await _applied(db_session, test_project, test_agent, status="failed", task_status="failed", backend="terminal_failure", resumable=True, provider="provider", attempt={"effect_state": "started"})
    action = await db_session.get(OrchestrationAction, result.action_id)
    assert result.classification == result.disposition == "terminal" and result.wait_id is None
    assert action.action_type == "noop"


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ("workspace", "path", "api_base", "openai_key"))
async def test_exact_resume_refuses_launch_fingerprint_drift_without_dispatch_or_budget_mutation(
    db_session, test_project, test_agent, monkeypatch, tmp_path, drift,
):
    test_agent.cli_runtime = "claude_code"
    goal, run, _, task, session = await _owned(
        db_session, test_project, test_agent, session_status="failed", task_status="failed", resumable=True,
        provider="provider", attempt={"effect_state": "started", "usage_complete": True,
            "result_status": "failed", "provider_session_id": "provider", "token_count_in": 1, "token_count_out": 1},
    )
    session.metadata_ = {**session.metadata_, "token_usage_complete": True,
                         "_launch_fingerprint": _launch_fingerprint(session, task, test_agent, test_project)}
    if drift == "workspace":
        moved = tmp_path / "moved-workspace"; moved.mkdir()
        test_project.workspace_path = str(moved)
    elif drift == "path":
        monkeypatch.setenv("PATH", "/drifted-path")
    elif drift == "api_base":
        monkeypatch.setattr(settings, "api_base_url", "http://drifted-api")
    else:
        monkeypatch.setenv("OPENAI_API_KEY", "drifted-key")
    retry_state_before = run.retry_state
    called = False

    async def never_resume(*_args):
        nonlocal called
        called = True
        raise AssertionError("exact continuation dispatched")

    monkeypatch.setattr(SessionService, "resume", never_resume)
    snapshot = await OrchestrationRecoveryService().build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))
    result = await OrchestrationRecoveryService().apply_goal_recovery(db_session, snapshot, {
        session.id: RunnerObservation(session.runner_task_id, "terminal_failure", datetime.now(timezone.utc))})
    assert result.action_id is None and not called
    assert session.status == "failed" and session.resumable is True
    assert run.retry_state == retry_state_before
    assert await db_session.scalar(select(OrchestrationWait).where(OrchestrationWait.run_id == run.id)) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("resumable,provider,usage_complete", (
    (False, "provider", True), (True, None, True), (True, "provider", False),
))
async def test_started_nonresumable_nonidempotent_and_incomplete_usage_never_replay(
    db_session, test_project, test_agent, monkeypatch, resumable, provider, usage_complete,
):
    called = []
    async def never_resume(*args): called.append(args)
    monkeypatch.setattr(SessionService, "resume", never_resume)
    result, _, run, _, _, _ = await _applied(
        db_session, test_project, test_agent, status="failed", task_status="failed", backend="terminal_failure",
        resumable=resumable, provider=provider,
        attempt={"effect_state": "started", "usage_complete": usage_complete},
    )
    action = await db_session.get(OrchestrationAction, result.action_id)
    assert result.classification == result.disposition == "terminal" and not called and result.wait_id is None
    assert action.action_type == "noop"


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ("paused", "cancelled"))
async def test_pause_and_cancel_win_final_revalidation_race(
    db_session, concurrent_sessions, test_project, test_agent, control,
):
    goal, run, _, _, session = await _owned(db_session, test_project, test_agent)
    await db_session.commit()
    observer, controller = concurrent_sessions
    service = OrchestrationRecoveryService()
    snapshot = await service.build_goal_snapshot(observer, goal.id, datetime.now(timezone.utc))
    changed_goal = await controller.get(OrchestrationGoal, goal.id)
    changed_run = await controller.get(OrchestrationRun, run.id)
    changed_goal.status = changed_run.status = control
    await controller.commit()
    assert await service.apply_goal_recovery(observer, snapshot, {
        session.id: RunnerObservation("runner-1", "active", datetime.now(timezone.utc)),
    }) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", (
    "goal_control", "goal_authority", "run_control", "run_retry", "run_budget",
    "task_status", "task_identity", "action_status", "action_request", "action_target",
    "action_contract", "action_budget", "session_runner", "session_adapter", "session_provider",
    "session_started", "session_effect", "session_usage", "session_retry", "session_budget",
))
async def test_apply_rejects_each_second_session_snapshot_mutation(
    db_session, concurrent_sessions, test_project, test_agent, mutation,
):
    """The post-observation apply must see committed changes made elsewhere."""
    goal, run, action, task, session = await _owned(db_session, test_project, test_agent)
    await db_session.commit()
    reader, writer = concurrent_sessions
    service = OrchestrationRecoveryService()
    snapshot = await service.build_goal_snapshot(reader, goal.id, datetime.now(timezone.utc))
    await reader.rollback()  # release the observation read snapshot before the writer commits

    changed_goal = await writer.get(OrchestrationGoal, goal.id)
    changed_run = await writer.get(OrchestrationRun, run.id)
    changed_action = await writer.get(OrchestrationAction, action.id)
    changed_task = await writer.get(Task, task.id)
    changed_session = await writer.get(Session, session.id)
    if mutation == "goal_control":
        changed_goal.status = "blocked"
    elif mutation == "goal_authority":
        changed_goal.authority_model, changed_goal.manager_agent_id = "agent_manager", test_agent.id
    elif mutation == "run_control":
        changed_run.phase = "ready"
    elif mutation == "run_retry":
        changed_run.retry_state = {"changed": True}
    elif mutation == "run_budget":
        changed_run.budget_state = {"changed": True}
    elif mutation == "task_status":
        changed_task.status = "blocked"
    elif mutation == "task_identity":
        changed_task.metadata_ = {"orchestration": {"run_id": str(run.id), "action_id": str(action.id), "changed": True}}
    elif mutation == "action_status":
        changed_action.status = "reserved"
    elif mutation == "action_request":
        changed_action.request = {"task_id": str(task.id), "changed": True}
    elif mutation == "action_target":
        changed_action.target_id = uuid.uuid4()
    elif mutation == "action_contract":
        changed_action.dispatch_contract = {"changed": True}
    elif mutation == "action_budget":
        changed_action.budget_ledger = {"changed": True}
    elif mutation == "session_runner":
        changed_session.runner_task_id = "different-runner"
    elif mutation == "session_adapter":
        changed_session.adapter_type = "api"
    elif mutation == "session_provider":
        changed_session.provider_session_id = "different-provider"
    elif mutation == "session_started":
        changed_session.started_at = datetime.now(timezone.utc)
    else:
        attempt = dict(changed_session.metadata_.get("attempt") or {})
        attempt[{"session_effect": "effect_state", "session_usage": "usage_complete",
                 "session_retry": "retry", "session_budget": "budget"}[mutation]] = "changed"
        changed_session.metadata_ = {**changed_session.metadata_, "attempt": attempt}
    await writer.commit()

    assert await service.apply_goal_recovery(reader, snapshot, {}) is None


@pytest.mark.asyncio
async def test_generic_orphan_watchdog_excludes_verified_orchestration_only(db_session, test_project, test_agent):
    _, _, _, _, session = await _owned(db_session, test_project, test_agent)
    session.started_at = datetime.now(timezone.utc) - timedelta(hours=2)
    assert await OrchestrationRecoveryService().resolve_owned_session(db_session, session.id)
    assert await SessionService().recover_orphaned_sessions(db_session, timeout_seconds=60) == 0
    await db_session.refresh(session)
    assert session.status == "running"


@pytest.mark.asyncio
async def test_generic_watchdog_uses_durable_ownership_after_goal_leaves_recovery_states(
    db_session, test_project, test_agent,
):
    goal, run, _, _, session = await _owned(db_session, test_project, test_agent)
    goal.status = run.status = "cancelled"
    session.started_at = datetime.now(timezone.utc) - timedelta(hours=2)
    assert await OrchestrationRecoveryService().resolve_owned_session(db_session, session.id)
    assert await SessionService().recover_orphaned_sessions(db_session, timeout_seconds=60) == 0


@pytest.mark.asyncio
async def test_direct_task_dispatch_allows_absent_session_metadata_but_rejects_arbitrary_action(
    db_session, test_project, test_agent,
):
    _, _, action, task, session = await _owned(db_session, test_project, test_agent)
    action.action_type, action.target_type, action.target_id, action.dispatch_contract = (
        "create_delegation_task", "task", task.id, {},
    )
    session.metadata_ = {"attempt": {}}
    recovery = OrchestrationRecoveryService()
    assert await recovery.resolve_owned_session(db_session, session.id)
    action.action_type = "run_command"
    assert await recovery.resolve_owned_session(db_session, session.id) is None


@pytest.mark.asyncio
async def test_direct_ownership_rejects_session_metadata_mismatch_and_goal_project_mismatch(
    db_session, test_project, test_agent,
):
    goal, _, action, task, session = await _owned(db_session, test_project, test_agent)
    action.action_type, action.target_type, action.target_id, action.dispatch_contract = (
        "create_delegation_task", "task", task.id, {},
    )
    session.metadata_ = {"orchestration": {"action_id": str(uuid.uuid4())}, "attempt": {}}
    recovery = OrchestrationRecoveryService()
    assert await recovery.resolve_owned_session(db_session, session.id) is None
    session.metadata_ = {"attempt": {}}
    goal.project_id = uuid.uuid4()
    with db_session.no_autoflush:
        assert await recovery.resolve_owned_session(db_session, session.id) is None


@pytest.mark.asyncio
async def test_recovery_actions_and_waits_deduplicate_across_restart_and_concurrency(
    db_session, concurrent_sessions, test_project, test_agent,
):
    goal, run, _, _, session = await _owned(db_session, test_project, test_agent)
    await db_session.commit()
    first, second = concurrent_sessions
    service = OrchestrationRecoveryService()
    first_snapshot = await service.build_goal_snapshot(first, goal.id, datetime.now(timezone.utc))
    second_snapshot = await service.build_goal_snapshot(second, goal.id, datetime.now(timezone.utc))
    observation = {session.id: RunnerObservation("runner-1", "active", datetime.now(timezone.utc))}
    barrier = asyncio.Barrier(2)
    async def apply_and_commit(db, snapshot):
        await barrier.wait()
        result = await OrchestrationRecoveryService().apply_goal_recovery(db, snapshot, observation)
        await db.commit()
        return result
    result, again = await asyncio.gather(
        apply_and_commit(first, first_snapshot), apply_and_commit(second, second_snapshot),
    )
    assert result.wait_id == again.wait_id
    assert len(list(await second.scalars(select(OrchestrationWait).where(OrchestrationWait.run_id == run.id)))) == 1
    stored_run = await second.get(OrchestrationRun, run.id)
    await second.refresh(stored_run)
    stored = next(iter(stored_run.supervision_state["recovery"]["sessions"].values()))
    assert stored["wait_id"] == str(result.wait_id) and stored["action_id"] is None


@pytest.mark.asyncio
async def test_current_live_attempt_beats_older_terminal_attempt_and_waits_only(db_session, test_project, test_agent):
    goal, run, action, task, completed = await _owned(
        db_session, test_project, test_agent, session_status="completed", task_status="completed",
    )
    live = Session(project_id=test_project.id, agent_id=test_agent.id, task_id=task.id, adapter_type="cli",
        status="running", runner_task_id="runner-live", metadata_={"orchestration": {"action_id": str(action.id)}, "attempt": {}})
    db_session.add(live); await db_session.flush()
    live_action = OrchestrationAction(run_id=run.id, idempotency_key=f"seed:{uuid.uuid4()}", action_type="retry_task",
        request={"task_id": str(task.id)}, status="completed", target_type="session", target_id=live.id,
        dispatch_contract={"owner": "orchestration_recovery"})
    db_session.add(live_action)
    await db_session.flush()
    task.metadata_ = {**task.metadata_, "orchestration": {**task.metadata_["orchestration"], "action_id": str(live_action.id)}}
    live.metadata_ = {"orchestration": {"action_id": str(live_action.id)}, "attempt": {}}
    await db_session.flush()
    service = OrchestrationRecoveryService()
    snapshot = await service.build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))
    result = await service.apply_goal_recovery(db_session, snapshot, {
        completed.id: RunnerObservation(completed.runner_task_id, "terminal_success", datetime.now(timezone.utc)),
        live.id: RunnerObservation(live.runner_task_id, "active", datetime.now(timezone.utc)),
    })
    assert result.classification == "live" and result.action_id is None and result.wait_id is not None


@pytest.mark.asyncio
async def test_database_terminal_attempt_records_completed_noop_action_only(db_session, test_project, test_agent):
    result, _, run, _, _, _ = await _applied(
        db_session, test_project, test_agent, status="completed", task_status="completed", backend="terminal_success",
    )
    action = await db_session.get(OrchestrationAction, result.action_id)
    assert result.classification == "terminal" and result.wait_id is None
    assert action.status == "completed" and action.action_type == "noop"
    assert not list(await db_session.scalars(select(OrchestrationWait).where(OrchestrationWait.run_id == run.id)))


@pytest.mark.asyncio
async def test_missing_exact_retry_proof_reconciles_terminal_without_dispatch(db_session, test_project, test_agent):
    result, _, run, _, _, _ = await _applied(
        db_session, test_project, test_agent, status="failed", task_status="failed", backend="terminal_failure",
        resumable=True, provider="provider", attempt={"effect_state": "started", "usage_complete": True},
    )
    action = await db_session.get(OrchestrationAction, result.action_id)
    assert result.disposition == "terminal" and result.wait_id is None
    assert action.action_type == "noop"


@pytest.mark.asyncio
async def test_resume_refusal_persists_actionable_attention(db_session, test_project, test_agent, monkeypatch):
    test_agent.cli_runtime = "claude_code"
    goal, run, source, task, session = await _owned(
        db_session, test_project, test_agent, session_status="failed", task_status="failed", resumable=True,
        provider="provider", attempt={"effect_state": "started", "usage_complete": True,
            "result_status": "failed", "provider_session_id": "provider"},
    )
    session.metadata_ = {**session.metadata_, "token_usage_complete": True,
        "_launch_fingerprint": _launch_fingerprint(session, task, test_agent, test_project)}
    refused = []
    async def post_reservation_refusal(_service, db, run_id, *_args, **_kwargs):
        action = OrchestrationAction(run_id=run_id, idempotency_key=f"refused:{uuid.uuid4()}", action_type="retry_task", request={}, status="failed", budget_ledger={"reserved": {"max_tokens": "10"}})
        db.add(action); await db.flush(); refused.append(action.id)
        return action
    monkeypatch.setattr("huddleroom.services.orchestration_service.OrchestrationService.execute_retry_task_action", post_reservation_refusal)
    snapshot = await OrchestrationRecoveryService().build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))
    result = await OrchestrationRecoveryService().apply_goal_recovery(db_session, snapshot, {
        session.id: RunnerObservation(session.runner_task_id, "terminal_failure", datetime.now(timezone.utc))})
    assert result.action_id == refused[0] and result.wait_id is not None
    failed = await db_session.get(OrchestrationAction, refused[0])
    waits = list(await db_session.scalars(select(OrchestrationWait).where(OrchestrationWait.run_id == run.id)))
    assert failed.status == "failed" and failed.budget_ledger == {"reserved": {"max_tokens": "10"}}
    assert len(waits) == 1 and waits[0].fallback["action_type"] == "attention"


@pytest.mark.asyncio
async def test_recovery_state_preserves_sibling_sessions_and_action_wait_links(db_session, test_project, test_agent):
    result, _, run, _, _, _ = await _applied(db_session, test_project, test_agent, backend="active")
    entry = run.supervision_state["recovery"]["sessions"]
    value = next(iter(entry.values()))
    assert entry and result.wait_id and set(value) >= {"classification", "disposition", "backend_observation", "action_id", "wait_id"}
    assert value["action_id"] is None and value["wait_id"] == str(result.wait_id)
