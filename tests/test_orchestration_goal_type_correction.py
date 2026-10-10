"""Task 14d: applying a goal-type correction to finite roots, retained gates, and their validation."""
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select

from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationEvidence,
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
    assert gate not in gates and svc._criterion_evidence_manifest(gates, [], goal) == {}


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


# --- Task 14f: Continuous root → finite drain, child correction + parent renewal ---

async def test_continuous_root_correction_waits_for_cases_then_stops_intake(db_session, test_project):
    from datetime import datetime

    from huddleroom.models.orchestration import OrchestrationBudgetReservation, OrchestrationGoal
    from huddleroom.services.orchestration_continuous_service import OrchestrationContinuousService
    from tests.test_orchestration_continuous_service import started_continuous

    goal, first = await started_continuous(db_session, test_project)
    svc = OrchestrationService()
    continuous = OrchestrationContinuousService(svc)
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    await continuous.claim_due_cycle(db_session, goal.id, due)
    await continuous.advance(db_session, goal, first, due)
    case = await db_session.scalar(select(OrchestrationGoal).where(OrchestrationGoal.parent_goal_id == goal.id))
    pending = [{"origin_key": "case:queued", "objective": "Queued case"}]
    goal.continuous_state = {**goal.continuous_state, "pending_candidates": pending}
    old = await svc.get_run_for_goal(db_session, test_project.id, goal.id)
    assert old.phase == "waiting_activation"

    request, _ = await _request(db_session, goal, old, requested_goal_type="outcome")
    with pytest.raises(HTTPException) as exc:
        await _apply(db_session, goal, request)
    assert exc.value.detail["conflict"] == "correction_blocked"
    assert [(b["code"], b["entity_id"]) for b in exc.value.detail["blockers"]] == [("active_child_goal", str(case.id))]
    assert goal.goal_type == "continuous" and not goal.continuous_state.get("stopped_at")

    case.status = "completed"
    case_run = await svc.get_run_for_goal(db_session, test_project.id, case.id)
    case_run.status = case_run.phase = "completed"
    await db_session.flush()
    request, _ = await _request(db_session, goal, old, requested_goal_type="outcome")
    goal, successor = await _apply(db_session, goal, request)

    stop = await db_session.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == old.id, OrchestrationAction.action_type == "stop_continuous"))
    assert stop.status == "completed"
    assert (stop.request["reason"], stop.request["actor"], stop.request["pending_candidates"]) == (
        "goal type corrected", "human:u1", pending)
    assert stop.request["stopped_at"] and goal.continuous_state == {}
    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.child_goal_id == case.id))
    assert reservation.status == "settled"
    assert (old.status, old.terminal_reason) == ("completed", "reclassified")
    assert (goal.goal_type, goal.planning_mode, goal.status) == ("outcome", "delivery", "active")
    assert (successor.supersedes_run_id, successor.phase, successor.goal_type_snapshot) == (old.id, "baseline", "outcome")
    # Intake stays stopped: no Continuous successor cycle appears for the old run.
    assert await db_session.get(
        OrchestrationRun, uuid.uuid5(uuid.NAMESPACE_URL, f"rally:continuous-successor:{old.id}")) is None
    active = await svc.get_active_run_for_goal(db_session, test_project.id, goal.id)
    assert active.id == successor.id


async def _released_child(db, project):
    from huddleroom.models.orchestration import OrchestrationGoal
    from tests.test_orchestration_roadmap_children import release

    roadmap, parent, parent_run, _, row = await release(db, project)
    child = await db.get(OrchestrationGoal, row.child_goal_id)
    child_run = await OrchestrationService().get_run_for_goal(db, project.id, child.id)
    return roadmap, parent, parent_run, child, child_run


async def test_child_correction_rejects_continuous_and_parent_explicit(db_session, test_project):
    _, _, _, child, child_run = await _released_child(db_session, test_project)
    assert child.goal_type_source == "parent_explicit"
    for requested, code in (("roadmap", "parent_explicit_requires_parent_plan"),
                            ("continuous", "continuous_child_requires_continuous_root")):
        request, _ = await _request(db_session, child, child_run, requested_goal_type=requested)
        with pytest.raises(HTTPException) as exc:
            await _apply(db_session, child, request)
        assert code in {b["code"] for b in exc.value.detail["blockers"]}
    child.goal_type_source = "parent_auto"
    request, _ = await _request(db_session, child, child_run, requested_goal_type="continuous")
    with pytest.raises(HTTPException) as exc:
        await _apply(db_session, child, request)
    assert [b["code"] for b in exc.value.detail["blockers"]] == ["continuous_child_requires_continuous_root"]
    assert (child.goal_type, child_run.status) == ("outcome", "running")


async def test_parent_auto_child_waits_for_parent_renewal_then_runs_without_start(
    db_session, test_project, monkeypatch,
):
    from huddleroom.models.orchestration import OrchestrationBudgetReservation

    roadmap, parent, parent_run, child, old = await _released_child(db_session, test_project)
    child.goal_type_source = "parent_auto"
    await db_session.flush()
    request, preview = await _request(db_session, child, old, requested_goal_type="roadmap")
    assert preview.required_authority == ["parent_release"] and preview.requires_new_start is False

    child, successor = await _apply(db_session, child, request)

    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.child_goal_id == child.id))
    assert reservation.status == "active"
    assert (old.status, old.terminal_reason) == ("cancelled", "reclassified")
    assert successor.plan_state["child_delta_baseline"]["status"] == "pending_renewal"
    svc = roadmap.orchestration
    idle = await svc.tick(db_session, successor.id)
    assert "authorized_execution" not in idle and successor.phase == "baseline"

    await svc.tick(db_session, parent_run.id)
    await roadmap.advance(db_session, parent, parent_run)  # idempotent

    assert successor.phase == "authorized"
    assert successor.plan_state["child_delta_baseline"] == {
        **successor.plan_state["child_delta_baseline"], "status": "accepted",
        "approval_reference": {"kind": "parent_renewal", "parent_run_id": str(parent_run.id)},
    }
    assert successor.goal_contract_fingerprint == compute_goal_contract_fingerprint(child)
    [authorize] = (await db_session.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == successor.id, OrchestrationAction.action_type == "authorize_execution"))).all()
    assert authorize.id == uuid.uuid5(uuid.NAMESPACE_URL, f"rally:authorize-child:{successor.id}")
    assert (authorize.status, authorize.request["actor"]) == ("completed", f"roadmap_parent:{parent.id}")
    assert authorize.idempotency_key == f"run:{successor.id}:kind:authorize_execution"
    assert reservation.status == "active"

    async def probe(db, goal, run):
        return {"step": "probe", "run_id": str(run.id)}

    monkeypatch.setattr(svc, "_advance_authorized_execution", probe)
    ran = await svc.tick(db_session, successor.id)  # no human Start
    assert ran["authorized_execution"] == {"step": "probe", "run_id": str(successor.id)}


async def test_apply_stale_run_and_stale_contract_conflicts_write_nothing(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    request, _ = await _request(db_session, goal, run)
    before = await _counts(db_session)
    for bad, code in (
        (request.model_copy(update={"expected_run_id": uuid.uuid4()}), "stale_run"),
        (request.model_copy(update={"expected_contract_fingerprint": "0" * 64}), "stale_contract"),
    ):
        with pytest.raises(HTTPException) as exc:
            await _apply(db_session, goal, bad)
        # The request change alters the preview fingerprint, so stale_preview wins; the fresh preview names the blocker.
        assert exc.value.status_code == 409 and exc.value.detail["conflict"] in ("stale_preview", code)
        assert code in {b["code"] for b in exc.value.detail["preview"]["blockers"]}
    assert await _counts(db_session) == before
    assert (goal.goal_type, run.status, run.terminal_reason) == ("outcome", "running", None)


# --- Task 14 fix round 1 ---

async def test_accepted_retained_gate_does_not_block_final_summary_manifest(db_session, test_project):
    goal, _, successor, _, _, retained = await _corrected_with_retained_gate(db_session, test_project)
    retained.status = "accepted"  # accepted without evidence (min_count 0)
    work = await _accepted_gate(db_session, successor)
    db_session.add(OrchestrationEvidence(run_id=successor.id, gate_id=work.id, source_type="artifact",
                                         source_id=uuid.uuid4(), verdict="accepted"))
    await db_session.flush()
    svc = OrchestrationService()

    assert retained not in await svc._current_run_gates(db_session, successor.id)
    gates, evidence = await svc._accepted_non_summary_manifest(db_session, successor.id)
    assert gates == [work] and [item.gate_id for item in evidence] == [work.id]


async def test_roadmap_to_outcome_successor_snapshot_counts_settled_child_spend(db_session, test_project):
    from decimal import Decimal

    from huddleroom.models.orchestration import OrchestrationBudgetReservation, OrchestrationGoal
    from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService

    _, parent, parent_run, child, child_run = await _released_child(db_session, test_project)
    child.status, child_run.status = "cancelled", "cancelled"
    await db_session.flush()
    request, _ = await _request(db_session, parent, parent_run, requested_goal_type="outcome")
    parent, successor = await _apply(db_session, parent, request)
    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.child_goal_id == child.id))
    reservation.status, reservation.settled_spend = "settled", {"max_tokens": "300"}
    await db_session.flush()
    budget = OrchestrationBudgetService()

    snapshot = await budget.snapshot_for_run(db_session, parent, successor)
    remaining = await budget.remaining(db_session, parent)

    assert parent.goal_type == "outcome" and isinstance(parent, OrchestrationGoal)
    assert Decimal(snapshot["consumed"]["max_tokens"]) == 300
    assert {k: Decimal(v) for k, v in snapshot["remaining"].items()} == {
        k: Decimal(v) for k, v in remaining["remaining"].items()}


async def test_continuous_correction_clears_continuous_state_round_trip(db_session, test_project):
    from tests.test_orchestration_continuous_service import direct_policy, started_continuous

    goal, first = await started_continuous(db_session, test_project)
    request, _ = await _request(db_session, goal, first, requested_goal_type="outcome")
    goal, finite = await _apply(db_session, goal, request)
    assert goal.continuous_state == {}
    stop = await db_session.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == first.id, OrchestrationAction.action_type == "stop_continuous"))
    assert stop.request["stopped_at"]

    request, _ = await _request(db_session, goal, finite, requested_goal_type="continuous")
    goal, again = await _apply(db_session, goal, request)
    assert goal.goal_type == "continuous" and goal.continuous_state == {}
    # a fresh Start is not short-circuited by the old started_at / stopped_at
    from huddleroom.schemas.orchestration import OrchestrationContinuousPolicy
    from huddleroom.services.orchestration_continuous_service import OrchestrationContinuousService
    policy = OrchestrationContinuousPolicy.model_validate(direct_policy(adapter_filter={"adapter_type": "schedule"}))
    await OrchestrationContinuousService(OrchestrationService()).update_policy(
        db_session, test_project.id, goal.id, policy, actor="human:u1")
    again.phase = "ready"
    await db_session.flush()
    await OrchestrationService().start_run(db_session, test_project.id, goal.id, actor="human:u1")
    assert goal.continuous_state.get("started_at") and not goal.continuous_state.get("stopped_at")


async def test_replay_key_from_another_goal_does_not_leak_successor(db_session, test_project):
    goal_a, run_a = await _goal_run(db_session, test_project)
    request, _ = await _request(db_session, goal_a, run_a)
    _, successor_a = await _apply(db_session, goal_a, request)
    goal_b, run_b = await _goal_run(db_session, test_project)
    before = await _counts(db_session)

    with pytest.raises(HTTPException) as exc:
        await _apply(db_session, goal_b, request)  # same expected_run_id + fingerprint => same idempotency key

    assert exc.value.status_code == 409
    assert await _counts(db_session) == before
    assert (goal_b.goal_type, run_b.terminal_reason) == ("outcome", None) and successor_a.goal_id == goal_a.id


async def test_paused_goal_stays_paused_after_correction(db_session, test_project):
    goal, old = await _goal_run(db_session, test_project)
    goal.status = old.status = "paused"
    await db_session.flush()
    request, _ = await _request(db_session, goal, old)

    goal, successor = await _apply(db_session, goal, request)

    assert (goal.goal_type, goal.status, successor.status) == ("roadmap", "paused", "paused")
    resumed, run = await OrchestrationService().resume_goal(db_session, test_project.id, goal.id)
    assert (resumed.status, run.id, run.status) == ("active", successor.id, "running")


async def test_paused_child_stays_paused_after_correction(db_session, test_project):
    _, _, _, child, old = await _released_child(db_session, test_project)
    child.goal_type_source = "parent_auto"
    child.status = old.status = "paused"
    await db_session.flush()
    request, _ = await _request(db_session, child, old, requested_goal_type="roadmap")

    child, successor = await _apply(db_session, child, request)

    assert (child.status, successor.status) == ("paused", "paused")


async def test_integrity_error_without_replay_returns_stale_run(db_session, test_project, monkeypatch):
    from sqlalchemy.exc import IntegrityError

    goal, run = await _goal_run(db_session, test_project)
    request, _ = await _request(db_session, goal, run)

    async def boom(*_args, **_kwargs):
        raise IntegrityError("x", {}, Exception())

    monkeypatch.setattr(OrchestrationGoalShapeService, "_apply_locked", boom)
    before = await _counts(db_session)
    with pytest.raises(HTTPException) as exc:
        await _apply(db_session, goal, request)

    assert exc.value.status_code == 409 and exc.value.detail["conflict"] == "stale_run"
    assert await _counts(db_session) == before
    assert (goal.goal_type, run.status, run.terminal_reason) == ("outcome", "running", None)
