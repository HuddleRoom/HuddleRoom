"""Task 14d: applying a goal-type correction to finite roots, retained gates, and their validation."""
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select

from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationGate,
    OrchestrationRun,
    OrchestrationWait,
    compute_goal_contract_fingerprint,
)
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision, OrchestrationProcessRun
from huddleroom.models.session import Session
from huddleroom.schemas.orchestration import GoalTypeCorrectionApplyRequest
from huddleroom.services.orchestration_goal_shape import OrchestrationGoalShapeService
from huddleroom.services.orchestration_roadmap_service import retained_gate_id
from huddleroom.services.orchestration_service import OrchestrationService
from tests.test_orchestration_goal_shape import _accepted_gate, _counts, _goal_run, _preview, _run_task


async def _request(db, goal, run, actor="human:u1", **kw):
    preview = await _preview(db, goal, run, actor, **kw)
    return GoalTypeCorrectionApplyRequest(**{
        "requested_goal_type": "roadmap", "expected_run_id": run.id,
        "expected_contract_fingerprint": compute_goal_contract_fingerprint(goal), "rationale": "fits better",
        **kw, "preview_fingerprint": preview.preview_fingerprint,
    }), preview


async def _apply(db, goal, request, actor="human:u1"):
    return await OrchestrationGoalShapeService().apply_correction(db, goal, request, actor)


def _decision(goal, run, key, status="pending", **kw):
    return OrchestrationAuthorityDecision(
        goal_id=goal.id, run_id=run.id, decision_key=key, title="t", status=status, authority="human",
        question="q?", options=["yes", "no"], **kw,
    )


def _wait(run):
    from datetime import datetime, timezone
    return OrchestrationWait(run_id=run.id, wait_key="w", owner={}, awaited_event={}, fallback={},
                             due_recheck_at=datetime.now(timezone.utc))


async def test_pre_start_correction_updates_same_goal_and_reruns_baseline(db_session, test_project):
    goal, old = await _goal_run(db_session, test_project, started=False, goal_type="outcome")
    process = OrchestrationProcessRun(goal_id=goal.id, run_id=old.id, process_type="goal_definition",
                                      status="completed", trigger_reason="initial")
    answered = _decision(goal, old, "goal_definition:scope", status="answered", selected_option="yes")
    db_session.add_all([process, answered])
    await db_session.flush()
    request, _ = await _request(db_session, goal, old)

    same, successor = await _apply(db_session, goal, request)

    assert same.id == goal.id
    assert (same.goal_type, same.planning_mode, same.goal_type_source, same.goal_type_rationale) == (
        "roadmap", "roadmap", "user", "fits better")
    assert (old.status, old.terminal_reason, old.phase, old.goal_type_snapshot) == (
        "cancelled", "reclassified", "baseline", "outcome")
    assert successor.id == uuid.uuid5(uuid.NAMESPACE_URL, f"rally:type-correction-successor:{old.id}")
    assert (successor.supersedes_run_id, successor.phase, successor.goal_type_snapshot) == (old.id, "baseline", "roadmap")
    assert successor.goal_contract_fingerprint is None
    assert successor.plan_state == {"retained_evidence_manifest": []}
    active = await OrchestrationService().get_active_run_for_goal(db_session, test_project.id, goal.id)
    assert active.id == successor.id
    rerun = await db_session.scalar(select(OrchestrationProcessRun).where(
        OrchestrationProcessRun.goal_id == goal.id, OrchestrationProcessRun.process_type == "goal_definition",
        OrchestrationProcessRun.superseded_by_id.is_(None)))
    assert rerun.id != process.id and (rerun.status, rerun.run_id) == ("running", successor.id)
    await db_session.refresh(answered)
    assert answered.status == "answered"
    events = (await db_session.scalars(select(EventLog).where(
        EventLog.event_type == "orchestration.goal_type_corrected"))).all()
    assert [e.payload for e in events] == [{
        "goal_id": str(goal.id), "old_run_id": str(old.id), "successor_run_id": str(successor.id),
        "from_type": "outcome", "to_type": "roadmap",
    }]


async def test_post_start_correction_audits_backlog_and_requires_fresh_start(db_session, test_project, test_agent):
    goal, old = await _goal_run(db_session, test_project, goal_type="outcome")
    old.plan_state = {"status": "accepted", "planning_version": 1}
    db_session.add(OrchestrationAction(run_id=old.id, idempotency_key=f"run:{old.id}:kind:authorize_execution",
                                       action_type="authorize_execution", status="completed"))
    backlog = await _run_task(db_session, test_project, old, status="backlog")
    failed = await _run_task(db_session, test_project, old, status="failed")
    done = await _run_task(db_session, test_project, old, status="done")
    pending = _decision(goal, old, "roadmap_plan:x")
    baseline_question = _decision(goal, old, "goal_definition:q")
    process = OrchestrationProcessRun(goal_id=goal.id, run_id=old.id, process_type="goal_definition",
                                      status="waiting_decision", trigger_reason="initial")
    db_session.add_all([
        Session(project_id=test_project.id, task_id=failed.id, agent_id=test_agent.id, adapter_type="api",
                status="failed"),
        _wait(old), pending, process,
    ])
    await db_session.flush()
    baseline_question.source_process_run_id = process.id
    db_session.add(baseline_question)
    await db_session.flush()
    frozen = (old.goal_type_snapshot, old.planning_mode_snapshot, old.goal_contract_fingerprint, dict(old.plan_state))
    request, _ = await _request(db_session, goal, old)

    goal, successor = await _apply(db_session, goal, request)

    assert (backlog.status, failed.status, done.status) == ("cancelled", "cancelled", "done")
    audit = await db_session.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == old.id, OrchestrationAction.action_type == "correct_goal_type"))
    assert (audit.status, audit.target_type, audit.target_id) == ("completed", "run", successor.id)
    assert sorted(audit.request["cancelled_tasks"], key=lambda t: t["previously_activated"]) == [
        {"task_id": str(backlog.id), "previously_activated": False},
        {"task_id": str(failed.id), "previously_activated": True},
    ]
    assert audit.request["previous_goal_type"] == "outcome" and audit.request["actor"] == "human:u1"
    for row in (pending, baseline_question):
        await db_session.refresh(row)
    assert (pending.status, baseline_question.status) == ("cancelled", "pending")
    assert await db_session.scalar(select(OrchestrationWait.status).where(OrchestrationWait.run_id == old.id)) == "cleared"
    assert (old.status, old.terminal_reason, old.phase) == ("cancelled", "reclassified", "authorized")
    assert (old.goal_type_snapshot, old.planning_mode_snapshot, old.goal_contract_fingerprint, old.plan_state) == frozen
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationRun).where(
        OrchestrationRun.supersedes_run_id == old.id)) == 1

    # The old Start does not carry over: the successor needs its own baseline and Start.
    svc = OrchestrationService()
    with pytest.raises(HTTPException) as exc:
        await svc.start_run(db_session, test_project.id, goal.id, actor="human:u1")
    assert exc.value.detail["conflict"] == "baseline_not_ready"
    successor.phase = "ready"
    await db_session.flush()
    _, started = await svc.start_run(db_session, test_project.id, goal.id, actor="human:u1")
    assert started.id == successor.id and started.phase == "authorized"
    assert await svc._existing_action_for_key(db_session, successor.id, f"run:{successor.id}:kind:authorize_execution")


async def test_live_session_blocks_and_writes_nothing(db_session, test_project, test_agent):
    goal, run = await _goal_run(db_session, test_project)
    task = await _run_task(db_session, test_project, run, status="in_progress")
    db_session.add(Session(project_id=test_project.id, task_id=task.id, agent_id=test_agent.id,
                           adapter_type="api", status="running"))
    await db_session.flush()
    request, preview = await _request(db_session, goal, run)
    before = await _counts(db_session)

    with pytest.raises(HTTPException) as exc:
        await _apply(db_session, goal, request)

    assert exc.value.status_code == 409 and exc.value.detail["conflict"] == "correction_blocked"
    assert {b["code"] for b in exc.value.detail["blockers"]} == {"active_session", "active_task"}
    assert exc.value.detail["preview"]["preview_fingerprint"] == preview.preview_fingerprint
    assert await _counts(db_session) == before
    assert (goal.goal_type, run.status, run.terminal_reason, task.status) == ("outcome", "running", None, "in_progress")


async def test_stale_preview_returns_fresh_preview(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    request, preview = await _request(db_session, goal, run)
    gate = await _accepted_gate(db_session, run)
    await _run_task(db_session, test_project, run, gate_id=str(gate.id))
    before = await _counts(db_session)

    with pytest.raises(HTTPException) as exc:
        await _apply(db_session, goal, request)

    fresh = await _preview(db_session, goal, run)
    assert exc.value.status_code == 409 and exc.value.detail["conflict"] == "stale_preview"
    assert exc.value.detail["preview"]["preview_fingerprint"] == fresh.preview_fingerprint != preview.preview_fingerprint
    assert await _counts(db_session) == before and goal.goal_type == "outcome"


async def test_replay_returns_same_successor_without_mutation(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    request, _ = await _request(db_session, goal, run)
    _, successor = await _apply(db_session, goal, request)
    before = await _counts(db_session)

    goal, replayed = await _apply(db_session, goal, request)

    assert replayed.id == successor.id and goal.goal_type == "roadmap"
    assert await _counts(db_session) == before


async def test_non_human_actor_is_forbidden(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project, goal_type_source="auto")
    request, _ = await _request(db_session, goal, run, actor="orchestrator")
    with pytest.raises(HTTPException) as exc:
        await _apply(db_session, goal, request, actor="orchestrator")
    assert exc.value.status_code == 403 and exc.value.detail["conflict"] == "explicit_user_request_required"
    assert goal.goal_type == "outcome"


async def _corrected_with_retained_gate(db, project):
    goal, old = await _goal_run(db, project, success_criteria=[{"key": "c1", "description": "works"}])
    source = await _accepted_gate(db, old, required_evidence={"success_criterion_keys": ["c1"]})
    task = await _run_task(db, project, old, gate_id=str(source.id))
    request, _ = await _request(db, goal, old)
    goal, successor = await _apply(db, goal, request)
    gate = await db.get(OrchestrationGate, retained_gate_id(successor.id, source.id))
    return goal, old, successor, source, task, gate


async def test_retained_gate_accepted_after_start_and_never_satisfies_criteria(db_session, test_project):
    goal, old, successor, source, task, gate = await _corrected_with_retained_gate(db_session, test_project)
    svc = OrchestrationService()

    assert gate is not None and (gate.gate_type, gate.status) == ("retained_evidence", "open")
    assert gate.required_evidence["source_task_id"] == str(task.id) and gate.required_evidence["min_count"] == 0
    assert successor.plan_state["retained_evidence_manifest"][0]["source_gate_id"] == str(source.id)
    await svc.validate_open_gates(db_session, successor.id)
    assert gate.status == "open" and gate.failure_reason.startswith("Waiting for Start")

    successor.goal_contract_fingerprint = compute_goal_contract_fingerprint(goal)  # what Start freezes
    await svc.validate_open_gates(db_session, successor.id)

    assert gate.status == "accepted"
    gates = await svc._current_run_gates(db_session, successor.id)
    assert gate in gates and svc._criterion_evidence_manifest(gates, [], goal) == {}


async def test_retained_gate_fails_when_requirements_change(db_session, test_project):
    goal, _, successor, _, _, gate = await _corrected_with_retained_gate(db_session, test_project)
    goal.objective = "Ship something else"
    successor.goal_contract_fingerprint = compute_goal_contract_fingerprint(goal)
    await OrchestrationService().validate_open_gates(db_session, successor.id)
    assert gate.status == "failed" and gate.failure_reason.startswith("Goal requirements changed")


async def test_changed_requirements_create_no_retained_gate(db_session, test_project):
    goal, old = await _goal_run(db_session, test_project)
    source = await _accepted_gate(db_session, old)
    await _run_task(db_session, test_project, old, gate_id=str(source.id))
    goal.objective = "Ship something else"
    request, _ = await _request(db_session, goal, old)
    _, successor = await _apply(db_session, goal, request)
    [entry] = successor.plan_state["retained_evidence_manifest"]
    assert entry["revalidation_required"] is True
    assert await db_session.get(OrchestrationGate, retained_gate_id(successor.id, source.id)) is None
