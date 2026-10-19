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


# --- SPR188/SPR232 (G5): workflow facts ------------------------------------------------------

def _g5_decision(goal, run, key, *, minutes=0, **kwargs):
    return OrchestrationAuthorityDecision(goal_id=goal.id, run_id=run.id, decision_key=key, title=key,
        authority="human", question="Proceed?", options=[{"key": "yes"}],
        asked_at=_utcnow() + timedelta(minutes=minutes), **kwargs)


async def _g5_process(db, goal, run, process_type="agent_definition_review", **kwargs):
    from huddleroom.models.orchestration_process import OrchestrationProcessRun
    row = OrchestrationProcessRun(goal_id=goal.id, run_id=run.id, process_type=process_type,
                                  trigger_reason="test", **kwargs)
    db.add(row); await db.flush()
    return row


def _g5_answer(run, decisions, **contract):
    return OrchestrationAction(run_id=run.id, idempotency_key=f"answer-{uuid.uuid4()}", action_type="apply_authority_answer",
        request={"submission": {"decision_ids": [str(d.id) for d in decisions]}},
        status=contract.pop("status", "reserved"), dispatch_contract=contract)


async def test_g5_baseline_process_without_session_is_qualified_not_live(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    run.phase, run.baseline_authorized = "baseline", True
    process = await _g5_process(db_session, goal, run, started_at=_utcnow() - timedelta(minutes=3))
    await db_session.flush()
    result = await _status(db_session, goal, run)
    workflow = result["status_line"]["workflow"]
    assert workflow["phase"] == "baseline" and workflow["phase_label"] == "Baseline setup"
    assert workflow["process"]["id"] == str(process.id) and workflow["process_status"] == "running"
    assert workflow["operation_summary"] == (
        "Last recorded: reviewing agent definitions. Result not yet recorded; live activity unavailable.")
    assert workflow["occurred_at"] is not None
    state = result["status_line"]["current_state"]
    assert state["kind"] == "unknown" and state["summary"] == workflow["operation_summary"]
    assert "No active worker" not in state["summary"] and state["activities"] == []


async def test_g5_superseded_and_foreign_run_processes_and_decisions_are_excluded(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    _, other_run = await _run(db_session, test_project)
    old = await _g5_process(db_session, goal, run, "goal_definition")
    new = await _g5_process(db_session, goal, run, "team_hierarchy", status="completed")
    old.superseded_by_id = new.id
    await _g5_process(db_session, goal, other_run, "manager_selection")
    await db_session.flush()
    stale = _g5_decision(goal, run, "stale-source", minutes=5, source_process_run_id=old.id)
    foreign = _g5_decision(goal, other_run, "foreign", minutes=6)
    keep = _g5_decision(goal, run, "keep", minutes=1)
    db_session.add_all([stale, foreign, keep]); await db_session.flush()
    result = await _status(db_session, goal, run)
    workflow = result["status_line"]["workflow"]
    assert workflow["process"] is None and workflow["operation_summary"].startswith("No recorded operation")
    assert workflow["pending_decision_id"] == str(keep.id) and workflow["additional_pending_count"] == 0
    assert result["pending_direction"]["id"] == str(keep.id)
    assert result["status_line"]["attention_summary"]["key"] == f"decision:{keep.id}:pending"
    assert result["status_line"]["attention_summary"]["additional_count"] == 0


async def test_g5_multiple_pending_decisions_agree_across_attention_workflow_and_debug(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    older, newest = _g5_decision(goal, run, "older", minutes=-5), _g5_decision(goal, run, "newest", minutes=5)
    db_session.add_all([older, newest]); await db_session.flush()
    result = await _status(db_session, goal, run)
    workflow = result["status_line"]["workflow"]
    assert workflow["pending_decision_id"] == str(newest.id) and workflow["additional_pending_count"] == 1
    assert result["pending_direction"]["id"] == str(newest.id)
    assert result["status_line"]["attention_summary"]["key"] == f"decision:{newest.id}:pending"
    assert workflow["guidance"] == {"kind": "review_decision", "label": "Answer the owner decision",
                                    "decision_id": str(newest.id), "reason": None}
    assert result["next_action"] == "Answer the owner decision"


async def test_g5_paused_goal_keeps_owner_question_and_saved_answer(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    goal.status = run.status = "paused"
    pending = _g5_decision(goal, run, "paused-question")
    answered = _g5_decision(goal, run, "saved", status="answered", selected_option="yes")
    db_session.add_all([pending, answered]); await db_session.flush()
    action = _g5_answer(run, [answered], application_state="pending")
    db_session.add(action); await db_session.flush()
    result = await _status(db_session, goal, run)
    workflow = result["status_line"]["workflow"]
    assert result["condition"] == "paused" and result["status_line"]["current_state"]["summary"] == "Paused."
    assert result["pending_direction"]["id"] == str(pending.id)
    assert workflow["guidance"]["kind"] == "resume" and workflow["guidance"]["decision_id"] == str(pending.id)
    assert workflow["application"] == {"action_id": str(action.id), "decision_ids": [str(answered.id)],
                                       "state": "pending", "reason": None}
    assert result["next_action"] == "Resume goal"


@pytest.mark.parametrize("contract,status,state,kind,reason", [
    ({"application_state": "pending"}, "reserved", "pending", "wait_application", None),
    ({"application_state": "pending", "reason": "database is locked"}, "reserved", "pending", "wait_application",
     "The database was busy; it will be retried."),
    ({"application_state": "blocked", "reason": "source_process_superseded"}, "failed", "blocked", "review_blocker",
     "The step that asked was replaced."),
    ({"application_state": "failed", "reason": "x" * 400}, "failed", "failed", "review_blocker", "Application failed."),
])
async def test_g5_saved_answer_application_state_and_guidance(db_session, test_project, contract, status, state, kind, reason):
    goal, run = await _run(db_session, test_project)
    answered = _g5_decision(goal, run, "saved", status="answered", selected_option="yes")
    db_session.add(answered); await db_session.flush()
    db_session.add(_g5_answer(run, [answered], status=status, **contract)); await db_session.flush()
    workflow = (await _status(db_session, goal, run))["status_line"]["workflow"]
    assert workflow["application"]["state"] == state and workflow["application"]["reason"] == reason
    assert workflow["guidance"]["kind"] == kind


async def test_g5_applied_answer_is_not_pending_and_extra_applications_are_counted(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    answered = [_g5_decision(goal, run, f"saved-{i}", status="answered", selected_option="yes") for i in range(3)]
    db_session.add_all(answered); await db_session.flush()
    done = _g5_answer(run, answered[:1], status="completed", application_state="applied")
    first, second = (_g5_answer(run, [d], application_state="pending") for d in answered[1:])
    first.created_at, second.created_at = _utcnow() - timedelta(minutes=2), _utcnow() - timedelta(minutes=1)
    db_session.add_all([done, second, first]); await db_session.flush()
    workflow = (await _status(db_session, goal, run))["status_line"]["workflow"]
    assert workflow["application"]["action_id"] == str(first.id) and workflow["additional_application_count"] == 1
    done.created_at = _utcnow() - timedelta(minutes=9)
    for action in (first, second):
        action.status, action.dispatch_contract = "completed", {"application_state": "applied"}
    await db_session.flush()
    workflow = (await _status(db_session, goal, run))["status_line"]["workflow"]
    assert workflow["application"] is None and workflow["additional_application_count"] == 0


async def test_g5_unapplied_answer_needing_baseline_authorization_points_to_authorization(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    run.phase, run.baseline_authorized = "baseline", False
    answered = _g5_decision(goal, run, "saved", status="answered", selected_option="yes")
    db_session.add(answered); await db_session.flush()
    db_session.add(_g5_answer(run, [answered], application_state="pending", reason="Orchestration admission paused"))
    await db_session.flush()
    guidance = (await _status(db_session, goal, run))["status_line"]["workflow"]["guidance"]
    assert (guidance["kind"], guidance["label"]) == ("authorize_baseline", "Authorize remaining baseline")


async def test_g5_stale_plan_review_decision_is_not_selected(db_session, test_project):
    import json
    goal, run = await _run(db_session, test_project)
    artifact_id = uuid.uuid4()
    digest = "a" * 64
    identity = {"project_id": str(test_project.id), "goal_id": str(goal.id), "run_id": str(run.id),
                "artifact_id": str(artifact_id), "planning_task_id": str(uuid.uuid4()), "planning_version": 1,
                "winning_session_id": None, "provenance_hash": digest, "canonical_content_hash": digest,
                "renderer_version": 1, "rendered_content_hash": digest, "review_cycle": 1}
    record = {"identity": identity, "state": "ready", "markdown": "# Plan"}
    run.plan_state = {"plan_reviews": {str(artifact_id): record}}
    current = _g5_decision(goal, run, "roadmap_plan:current", minutes=1,
                           context=json.dumps({"plan_review_identity": identity}))
    cycle2 = {**identity, "review_cycle": 2}
    stale = _g5_decision(goal, run, "roadmap_plan:stale", minutes=9,
                         context=json.dumps({"plan_review_identity": cycle2}))
    malformed = _g5_decision(goal, run, "roadmap_plan:bad", minutes=8, context="not json")
    db_session.add_all([current, stale, malformed]); await db_session.flush()
    workflow = (await _status(db_session, goal, run))["status_line"]["workflow"]
    assert workflow["pending_decision_id"] == str(current.id) and workflow["additional_pending_count"] == 0
    run.plan_state = {"plan_reviews": {str(artifact_id): {**record, "state": "stale"}}}
    await db_session.flush()
    workflow = (await _status(db_session, goal, run))["status_line"]["workflow"]
    assert workflow["pending_decision_id"] is None


async def test_g5_planning_requires_exact_planner_binding(db_session, test_project, test_agent):
    goal, run = await _run(db_session, test_project)
    request = OrchestrationAction(run_id=run.id, idempotency_key="plan-request", action_type="request_plan",
                                  request={}, status="completed")
    db_session.add(request); await db_session.flush()
    task = await _task(db_session, test_project, test_agent, run, "Draft the plan")
    task.metadata_ = {"orchestration": {"run_id": str(run.id), "action_id": str(request.id), "work_function": "planning"}}
    run.plan_state = {"status": "requested", "planning_task_id": str(task.id), "request_action_id": str(request.id)}
    await db_session.flush()
    workflow = (await _status(db_session, goal, run))["status_line"]["workflow"]
    assert workflow["phase"] == "planning" and workflow["task"]["id"] == str(task.id)
    assert "Draft the plan" in workflow["operation_summary"] and "live activity unavailable" in workflow["operation_summary"]
    task.metadata_ = {"orchestration": {"run_id": str(uuid.uuid4()), "action_id": str(request.id), "work_function": "planning"}}
    await db_session.flush()
    workflow = (await _status(db_session, goal, run))["status_line"]["workflow"]
    assert workflow["task"] is None and "planner task unavailable" in workflow["operation_summary"]


async def test_g5_detail_get_is_read_only(db_session, test_project):
    from sqlalchemy import event
    goal, run = await _run(db_session, test_project)
    run.phase, run.baseline_authorized = "baseline", True
    answered = _g5_decision(goal, run, "saved", status="answered", selected_option="yes")
    db_session.add_all([_g5_decision(goal, run, "q"), answered]); await db_session.flush()
    db_session.add(_g5_answer(run, [answered], application_state="pending"))
    await _g5_process(db_session, goal, run)
    await db_session.flush()
    writes = []
    def capture(conn, cursor, statement, parameters, context, many):
        if statement.lstrip().split(None, 1)[0].upper() in {"INSERT", "UPDATE", "DELETE"}:
            writes.append(statement)
    engine = db_session.bind.sync_engine
    event.listen(engine, "before_cursor_execute", capture)
    try:
        await _status(db_session, goal, run)
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert writes == []


async def test_g5_paused_with_recorded_running_process_stays_paused_without_liveness_claim(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    goal.status = run.status = "paused"
    process = await _g5_process(db_session, goal, run, started_at=_utcnow() - timedelta(minutes=3))
    await db_session.flush()
    result = await _status(db_session, goal, run)
    workflow = result["status_line"]["workflow"]
    assert result["condition"] == "paused" and result["status_line"]["current_state"]["summary"] == "Paused."
    assert workflow["process"]["id"] == str(process.id) and workflow["process_status"] == "running"
    assert "Last recorded" in workflow["operation_summary"] and "live activity unavailable" in workflow["operation_summary"]
    assert "Working" not in result["status_line"]["current_state"]["summary"]
    assert result["status_line"]["current_state"]["activities"] == []
    assert workflow["guidance"]["kind"] == "resume"


async def test_g5_applied_answer_guidance_is_not_waiting_for_application(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    answered = _g5_decision(goal, run, "saved", status="answered", selected_option="yes")
    db_session.add(answered); await db_session.flush()
    db_session.add(_g5_answer(run, [answered], status="completed", application_state="applied")); await db_session.flush()
    workflow = (await _status(db_session, goal, run))["status_line"]["workflow"]
    assert workflow["application"] is None and workflow["pending_decision_id"] is None
    assert workflow["guidance"]["kind"] not in {"wait_application", "review_blocker", "authorize_baseline", "review_decision"}


async def test_g5_saved_answer_with_paused_baseline_resumes_first_then_authorizes(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    run.phase, run.baseline_authorized = "baseline", False
    goal.status = run.status = "paused"
    answered = _g5_decision(goal, run, "saved", status="answered", selected_option="yes")
    db_session.add(answered); await db_session.flush()
    db_session.add(_g5_answer(run, [answered], application_state="pending", reason="Orchestration admission paused"))
    await db_session.flush()
    first = (await _status(db_session, goal, run))["status_line"]["workflow"]["guidance"]
    assert first["kind"] == "resume" and first["label"] == "Resume goal"
    goal.status, run.status = "active", "running"
    await db_session.flush()
    after = (await _status(db_session, goal, run))["status_line"]["workflow"]["guidance"]
    assert (after["kind"], after["label"]) == ("authorize_baseline", "Authorize remaining baseline")


async def test_g5_detail_get_over_http_in_fresh_session_issues_no_writes(client, test_engine):
    from sqlalchemy import event, select
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
    from huddleroom.dependencies import get_db
    from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
    from huddleroom.models.project import Project
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as seed:
        project = Project(name="g5-http", config={})
        seed.add(project); await seed.flush()
        goal = OrchestrationGoal(project_id=project.id, objective="Keep moving", status="active", goal_type="outcome")
        seed.add(goal); await seed.flush()
        run = OrchestrationRun(goal_id=goal.id, status="running", phase="baseline", baseline_authorized=True)
        seed.add(run); await seed.flush()
        answered = _g5_decision(goal, run, "saved", status="answered", selected_option="yes")
        seed.add_all([_g5_decision(goal, run, "q"), answered]); await seed.flush()
        seed.add(_g5_answer(run, [answered], application_state="pending"))
        await _g5_process(seed, goal, run)
        await seed.commit()
        ids = (project.id, goal.id, run.id)

    async def snapshot():
        async with factory() as s:
            from sqlalchemy import text
            return {t: (await s.execute(text(f"select * from {t} order by 1"))).fetchall() for t in (
                "orchestration_goals", "orchestration_runs", "orchestration_actions",
                "orchestration_authority_decisions", "orchestration_process_runs")}
    before = await snapshot()

    async def fresh_db():
        async with factory() as s:
            yield s
    client._transport.app.dependency_overrides[get_db] = fresh_db
    writes = []
    def capture(conn, cursor, statement, parameters, context, many):
        if statement.lstrip().split(None, 1)[0].upper() in {"INSERT", "UPDATE", "DELETE"}:
            writes.append(statement)
    sync_engine = test_engine.sync_engine
    event.listen(sync_engine, "before_cursor_execute", capture)
    try:
        resp = await client.get(f"/api/v1/projects/{ids[0]}/orchestration/goals/{ids[1]}")
    finally:
        event.remove(sync_engine, "before_cursor_execute", capture)
    assert resp.status_code == 200, resp.text
    workflow = resp.json()["supervision"]["status_line"]["workflow"]
    assert workflow["pending_decision_id"] is not None and workflow["application"]["state"] == "pending"
    assert writes == []
    assert await snapshot() == before


@pytest.mark.parametrize("state", ["completed", "cancelled"])
async def test_g5_terminal_goal_has_no_needs_you_attention_or_pending(db_session, test_project, state):
    goal, run = await _run(db_session, test_project)
    goal.status = run.status = state
    run.completed_at = _utcnow()
    db_session.add(_g5_decision(goal, run, "late-question")); await db_session.flush()
    result = await _status(db_session, goal, run)
    assert result["status_line"]["attention_summary"] is None
    assert result["status_line"]["workflow"]["pending_decision_id"] is None and result["pending_direction"] is None


async def test_g5_other_run_source_process_decision_is_excluded(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    from huddleroom.models.orchestration import OrchestrationRun
    other_run = OrchestrationRun(goal_id=goal.id, status="completed", phase="authorized")
    db_session.add(other_run); await db_session.flush()
    foreign = await _g5_process(db_session, goal, other_run, "manager_selection")
    own = await _g5_process(db_session, goal, run, "goal_definition")
    db_session.add_all([_g5_decision(goal, run, "other-run-source", minutes=5, source_process_run_id=foreign.id),
                        _g5_decision(goal, run, "own-source", source_process_run_id=own.id)])
    await db_session.flush()
    workflow = (await _status(db_session, goal, run))["status_line"]["workflow"]
    assert workflow["additional_pending_count"] == 0
    assert workflow["pending_decision_id"] is not None


async def test_g5_initial_authorize_baseline_positive_and_negatives(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    run.phase, run.baseline_authorized = "baseline", False
    await db_session.flush()
    async def guidance():
        return (await _status(db_session, goal, run))["status_line"]["workflow"]["guidance"]["kind"]
    assert await guidance() == "authorize_baseline"
    run.baseline_authorized = True; await db_session.flush()
    assert await guidance() != "authorize_baseline"
    run.baseline_authorized, run.pause_requested = False, True; await db_session.flush()
    assert await guidance() != "authorize_baseline"
    run.pause_requested, goal.status = False, "paused"; await db_session.flush()
    assert await guidance() == "resume"
    goal.status = "active"
    run.active_blockers = [{"kind": "budget_integrity", "scope": "x"}]; await db_session.flush()
    assert await guidance() == "review_blocker"


async def test_g5_saved_answer_blocker_outranks_authorize_baseline_and_failed_does_not_authorize(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    run.phase, run.baseline_authorized = "baseline", False
    answered = _g5_decision(goal, run, "saved", status="answered", selected_option="yes")
    db_session.add(answered); await db_session.flush()
    action = _g5_answer(run, [answered], application_state="pending", reason="Orchestration admission paused")
    db_session.add(action); await db_session.flush()
    run.active_blockers = [{"kind": "budget_integrity", "scope": "x"}]; await db_session.flush()
    assert (await _status(db_session, goal, run))["status_line"]["workflow"]["guidance"]["kind"] == "review_blocker"
    run.active_blockers = []
    action.status, action.dispatch_contract = "failed", {"application_state": "failed"}
    await db_session.flush()
    assert (await _status(db_session, goal, run))["status_line"]["workflow"]["guidance"]["kind"] == "review_blocker"


async def test_g5_occurred_at_is_a_real_occurrence_not_updated_at(db_session, test_project, test_agent):
    goal, run = await _run(db_session, test_project)
    started = _utcnow() - timedelta(hours=2)
    process = await _g5_process(db_session, goal, run, started_at=started)
    process.updated_at = _utcnow() + timedelta(days=1); await db_session.flush()
    workflow = (await _status(db_session, goal, run))["status_line"]["workflow"]
    assert workflow["occurred_at"] == started.isoformat().replace("+00:00", "Z")
    process.status = "completed"
    request = OrchestrationAction(run_id=run.id, idempotency_key="plan-req2", action_type="request_plan", request={},
                                  status="completed", created_at=started)
    db_session.add(request); await db_session.flush()
    task = await _task(db_session, test_project, test_agent, run, "Plan")
    task.metadata_ = {"orchestration": {"run_id": str(run.id), "action_id": str(request.id), "work_function": "planning"}}
    run.plan_state = {"status": "requested", "planning_task_id": str(task.id), "request_action_id": str(request.id)}
    request.updated_at = _utcnow() + timedelta(days=1); await db_session.flush()
    workflow = (await _status(db_session, goal, run))["status_line"]["workflow"]
    assert workflow["occurred_at"] == started.isoformat().replace("+00:00", "Z")


async def test_g5_terminal_goal_hides_leftover_running_process(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    await _g5_process(db_session, goal, run)
    goal.status = run.status = "cancelled"; run.completed_at = _utcnow()
    await db_session.flush()
    result = await _status(db_session, goal, run)
    workflow = result["status_line"]["workflow"]
    assert workflow["process"] is None and workflow["task"] is None and workflow["occurred_at"] is None
    assert workflow["operation_summary"] == "Goal cancelled."
    assert result["status_line"]["current_state"]["summary"] == "Cancelled."
