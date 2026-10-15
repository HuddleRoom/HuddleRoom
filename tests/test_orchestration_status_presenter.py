"""SPR188 status contract through durable detail projection."""
from copy import deepcopy
from datetime import timedelta
import uuid
import pytest
from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationEvidence, OrchestrationGate, OrchestrationWait
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.session import Session
from huddleroom.routers.orchestration_goals import _detail
from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService
from tests.test_orchestration_supervision_liveness import _run, _task
pytestmark = pytest.mark.asyncio

async def _status(db, goal, run):
    return (await _detail(db, goal, run)).supervision.model_dump(mode="json")

@pytest.mark.parametrize("authority,condition", [("agent", "unknown"), ("human", "needs_you")])
async def test_no_worker_evidence_does_not_manufacture_work(db_session, test_project, authority, condition):
    goal, run = await _run(db_session, test_project)
    db_session.add(OrchestrationAuthorityDecision(goal_id=goal.id, run_id=run.id, decision_key=authority,
        title="Choose", authority=authority, question="Proceed?", options=[{"key": "yes"}]))
    await db_session.flush()
    result = await _status(db_session, goal, run)
    assert result["condition"] == condition
    assert result["status_line"]["current_state"]["kind"] == "unknown"
    assert result["status_line"]["current_state"]["summary"] == "No active worker confirmed."
    assert "Work in progress" not in result["operation"]

async def test_action_accounting_update_is_not_terminal_occurrence(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    now = _utcnow()
    action = OrchestrationAction(run_id=run.id, idempotency_key="old-action", action_type="noop",
        request={}, status="completed", created_at=now - timedelta(days=2))
    gate = OrchestrationGate(run_id=run.id, success_criterion_key="checkout", gate_type="work_completed")
    db_session.add_all([action, gate]); await db_session.flush()
    evidence = OrchestrationEvidence(run_id=run.id, gate_id=gate.id, source_type="verification",
        source_id=uuid.uuid4(), verdict="accepted", evidence_metadata={}, created_at=now - timedelta(days=1))
    db_session.add(evidence); await db_session.flush()
    before = await _status(db_session, goal, run)
    budget = OrchestrationBudgetService()
    snapshot = budget.snapshot(goal)
    budget.reserve(snapshot, action, {})
    budget.commit(snapshot, action)
    budget.settle(snapshot, action, {}, observation_id="accounting-only")
    await db_session.flush()
    assert action.updated_at > evidence.created_at
    after = await _status(db_session, goal, run)
    event = after["status_line"]["last_material_event"]
    assert event == before["status_line"]["last_material_event"]
    assert event["kind"] == "accepted_evidence"
    assert after["transition"]["key"] == event["key"]
    assert after["transition"]["occurred_at"] == event["occurred_at"]

async def test_latest_cleared_wait_is_history_not_current_attention(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    _, other_run = await _run(db_session, test_project)
    now = _utcnow()
    def wait(key, target, status, cleared):
        return OrchestrationWait(run_id=target.id, wait_key=key, status=status,
            owner={"type": "run", "id": str(target.id)}, awaited_event={"event_type": "task.status_changed"},
            due_recheck_at=now + timedelta(minutes=1), fallback={"action_type": "attention"},
            created_at=now - timedelta(days=2), cleared_at=cleared)
    open_wait = wait("open", run, "open", None)
    cleared = wait("cleared", run, "cleared", now)
    db_session.add_all([open_wait, cleared, wait("foreign", other_run, "cleared", now + timedelta(days=1)),
        wait("undated", run, "cleared", None), wait("wrong-status", run, "open", now + timedelta(days=1))])
    await db_session.flush()
    result = await _status(db_session, goal, run)
    event = result["status_line"]["last_material_event"]
    assert str(cleared.id) in event["key"]
    assert event["occurred_at"] == now.isoformat().replace("+00:00", "Z")
    assert str(cleared.id) not in {row["id"] for row in result["waits"]}
    assert str(cleared.id) not in str(result["status_line"]["attention_summary"])

async def test_read_only_snapshot_matches_execution_calculation_without_mutation(db_session, test_project, monkeypatch):
    goal, run = await _run(db_session, test_project)
    goal.budget = {"caps": {"max_tokens": "100"}}
    await db_session.flush()
    before = deepcopy(run.budget_state)
    original_flush = db_session.flush
    async def forbidden(*args, **kwargs):
        raise AssertionError("read projection must not flush")
    monkeypatch.setattr(db_session, "flush", forbidden)
    service = OrchestrationBudgetService()
    snapshot, usage = await service.project_snapshot_for_run(db_session, goal, run)
    assert run.budget_state == before
    assert usage == "known"
    monkeypatch.setattr(db_session, "flush", original_flush)
    assert await service.snapshot_for_run(db_session, goal, run) == snapshot
    assert run.budget_state["usage_state"] == usage


@pytest.mark.parametrize("classification,observation,expected", [
    ("live", "2026-10-14T12:00:00Z", "reported_active"),
    ("live", None, "liveness_unknown"), ("unknown", "bad-time", "liveness_unknown"),
])
async def test_owned_worker_validated_once_and_historical_debug_preserved(
    db_session, test_project, test_agent, monkeypatch, classification, observation, expected,
):
    from huddleroom.services.orchestration_recovery_service import OrchestrationRecoveryService
    import huddleroom.routers.orchestration_goals as router
    goal, run = await _run(db_session, test_project)
    task = await _task(db_session, test_project, test_agent, run, "Checkout")
    action = OrchestrationAction(run_id=run.id, idempotency_key="owned-worker", action_type="create_delegation_task",
        request={}, target_type="task", target_id=task.id, status="completed")
    db_session.add(action); await db_session.flush()
    task.metadata_ = {"orchestration": {"run_id": str(run.id), "action_id": str(action.id)}}
    now = _utcnow()
    sessions = [Session(project_id=test_project.id, task_id=task.id, agent_id=test_agent.id,
        adapter_type="api", status=status, started_at=now - timedelta(minutes=2),
        ended_at=now + timedelta(minutes=2) if status == "completed" else None,
        metadata_={"orchestration": {"action_id": str(action.id)}}) for status in ("running", "completed")]
    db_session.add_all(sessions); await db_session.flush()
    run.supervision_state = {"recovery": {"sessions": {str(sessions[0].id): {
        "classification": classification, "assessed_at": observation,
    }}}}
    await db_session.flush()
    original = OrchestrationRecoveryService.resolve_owned_session
    calls = []
    async def owned(self, db, session_id):
        calls.append(session_id)
        return await original(self, db, session_id)
    monkeypatch.setattr(OrchestrationRecoveryService, "resolve_owned_session", owned)
    presenter = router.build_orchestration_status_line
    async def checked_presenter(db, **kwargs):
        # The status presenter receives validated facts; no Session/Task db.get or ownership check may repeat.
        original_get = db.get
        async def checked_get(model, *args, **kw):
            assert model not in (Session, __import__("huddleroom.models.task", fromlist=["Task"]).Task)
            return await original_get(model, *args, **kw)
        monkeypatch.setattr(db, "get", checked_get)
        try:
            return await presenter(db, **kwargs)
        finally:
            monkeypatch.setattr(db, "get", original_get)
    monkeypatch.setattr(router, "build_orchestration_status_line", checked_presenter)
    result = await _status(db_session, goal, run)
    assert sorted(map(str, calls)) == sorted(str(row.id) for row in sessions)
    assert len(result["workers"]) == 2
    activities = result["status_line"]["current_state"]["activities"]
    assert len(activities) == 1 and activities[0]["state"] == expected
    assert "Checkout" in activities[0]["summary"]
    assert str(sessions[1].id) in result["status_line"]["last_material_event"]["key"]


async def test_chronology_timezone_ties_and_scoped_names_are_deterministic(db_session, test_project, test_agent):
    from datetime import datetime, timezone
    from huddleroom.services.orchestration_status_presenter import build_orchestration_status_line
    from huddleroom.models.project import Project
    from huddleroom.models.task import Task
    goal, run = await _run(db_session, test_project)
    goal.success_criteria = [{"key": "checkout", "description": "Checkout verified"}]
    other = Project(name="Foreign")
    db_session.add(other); await db_session.flush()
    foreign_task = Task(project_id=other.id, title="FOREIGN SECRET TITLE", status="ready")
    db_session.add(foreign_task); await db_session.flush()
    now = datetime(2026, 10, 14, 12, tzinfo=timezone.utc)
    old_gate = OrchestrationGate(id=uuid.uuid4(), run_id=run.id, success_criterion_key="checkout",
        gate_type="work_completed", status="failed", failed_at=now - timedelta(days=1))
    db_session.add(old_gate); await db_session.flush()
    items = [OrchestrationEvidence(id=uuid.UUID(int=i), run_id=run.id, gate_id=old_gate.id,
        source_type="verification", source_id=uuid.uuid4(), verdict="accepted", evidence_metadata={"secret": "PAYLOAD"},
        created_at=when) for i, when in [(2, now), (1, now.replace(tzinfo=None))]]
    db_session.add_all(items); await db_session.flush()
    run.active_blockers = [{"kind": "unknown", "task_id": str(foreign_task.id), "reason": "SECRET RAW REASON"}]
    run.updated_at = now + timedelta(days=5)
    facts = [{"session_id": str(uuid.uuid4()), "task_id": str(foreign_task.id), "agent_id": str(test_agent.id),
        "session_status": "running", "observed_liveness": "live", "observed_at": "2026-10-14T14:00:00+02:00"}]
    async def project(evidence):
        return await build_orchestration_status_line(db_session, project_id=test_project.id, goal=goal, run=run,
            actions=[], gates=[old_gate], evidence=evidence, open_waits=[], worker_facts=facts)
    first, condition = await project(items)
    second, _ = await project(list(reversed(items)))
    assert first == second and condition == "needs_attention"
    assert first.current_state.kind == "working"
    assert first.current_state.activities[0].observed_at == now
    assert first.attention_summary.occurred_at is None
    assert first.attention_summary.subject is None or first.attention_summary.subject.label is None
    assert first.last_material_event.key == f"evidence:{items[1].id}:accepted"
    assert first.last_material_event.occurred_at == now
    assert "Checkout verified" in first.last_material_event.summary
    rendered = first.model_dump_json()
    for secret in ("FOREIGN SECRET TITLE", "SECRET RAW REASON", "PAYLOAD"):
        assert secret not in rendered


async def test_cleared_wait_query_is_bounded_and_ties_use_stable_id(db_session, test_project):
    from sqlalchemy import event
    from huddleroom.services.orchestration_status_presenter import build_orchestration_status_line
    goal, run = await _run(db_session, test_project)
    now = _utcnow()
    for i in (2, 1):
        db_session.add(OrchestrationWait(id=uuid.UUID(int=i), run_id=run.id, wait_key=f"cleared-{i}",
            status="cleared", owner={"type": "run", "id": str(run.id)}, awaited_event={}, fallback={},
            due_recheck_at=now, created_at=now - timedelta(days=1), cleared_at=now))
    await db_session.flush()
    queries = []
    engine = db_session.bind.sync_engine
    def capture(conn, cursor, statement, parameters, context, many):
        if "orchestration_waits" in statement:
            queries.append((statement.lower(), parameters))
    event.listen(engine, "before_cursor_execute", capture)
    try:
        status, _ = await build_orchestration_status_line(db_session, project_id=test_project.id, goal=goal, run=run,
            actions=[], gates=[], evidence=[], open_waits=[], worker_facts=[])
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert status.last_material_event.key == f"wait:{uuid.UUID(int=1)}:cleared"
    assert status.attention_summary is None
    assert len(queries) == 1
    statement, params = queries[0]
    assert "limit" in statement and "order by" in statement and "cleared_at is not null" in statement
    assert "cleared" in params and 1 in params


@pytest.mark.parametrize("kind,forbidden", [("budget_integrity", "Budget measurement unavailable")])
async def test_structured_attention_does_not_mislabel_integrity_as_measurement(db_session, test_project, kind, forbidden):
    goal, run = await _run(db_session, test_project)
    run.active_blockers = [{"kind": kind, "scope": f"parent:{goal.id}"}]
    await db_session.flush()
    result = await _status(db_session, goal, run)
    assert result["condition"] == "needs_attention"
    assert forbidden not in result["status_line"]["attention_summary"]["summary"]
    assert "Budget integrity needs review." in result["status_line"]["attention_summary"]["summary"]


async def test_unknown_wait_check_does_not_promise_scheduled_check(db_session, test_project):
    from huddleroom.services.orchestration_status_presenter import build_orchestration_status_line
    from types import SimpleNamespace
    goal, run = await _run(db_session, test_project)
    wait = SimpleNamespace(id=uuid.uuid4(), owner={}, created_at=_utcnow(), due_recheck_at=None)
    status, _ = await build_orchestration_status_line(db_session, project_id=test_project.id, goal=goal, run=run,
        actions=[], gates=[], evidence=[], open_waits=[wait], worker_facts=[])
    assert status.attention_summary.next_check_at is None


@pytest.mark.parametrize("foreign", [False, True])
async def test_current_evidence_resolves_older_run_gate_only_inside_project(db_session, test_project, foreign):
    from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
    from huddleroom.models.project import Project
    goal, run = await _run(db_session, test_project)
    goal.success_criteria = [{"key": "checkout", "description": "Checkout verified"}]
    if foreign:
        project = Project(name="Foreign scope")
        db_session.add(project); await db_session.flush()
        source_goal = OrchestrationGoal(project_id=project.id, objective="PRIVATE FOREIGN GOAL", success_criteria=[{"key": "checkout", "description": "PRIVATE FOREIGN CRITERION"}])
        db_session.add(source_goal); await db_session.flush()
    else:
        source_goal = goal
    old_run = OrchestrationRun(goal_id=source_goal.id, status="completed", phase="authorized", started_at=_utcnow() - timedelta(days=2))
    db_session.add(old_run); await db_session.flush()
    gate = OrchestrationGate(run_id=old_run.id, success_criterion_key="checkout", gate_type="work_completed")
    db_session.add(gate); await db_session.flush()
    db_session.add(OrchestrationEvidence(run_id=run.id, gate_id=gate.id, source_type="verification", source_id=uuid.uuid4(), verdict="accepted", evidence_metadata={}, created_at=_utcnow() + timedelta(minutes=1)))
    await db_session.flush()
    result = await _status(db_session, goal, run)
    event = result["status_line"]["last_material_event"]
    assert event["kind"] == "accepted_evidence"
    if foreign:
        assert event["subject"] is None or event["subject"]["label"] is None
        assert "PRIVATE FOREIGN" not in str(result["status_line"])
    else:
        assert event["subject"]["id"] == str(gate.id)
        assert event["subject"]["label"] == "Checkout verified"
        assert event["subject"]["goal_id"] is None
        assert "Checkout verified" in event["summary"]
