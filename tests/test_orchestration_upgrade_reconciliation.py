"""Acceptance tests for goal-scoped memory provenance recovery."""
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationDecision, OrchestrationEvidence, OrchestrationGate, OrchestrationGoal, OrchestrationRun, OrchestrationWait
from huddleroom.models.orchestration_memory import OrchestrationMemorySection
from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration_process import OrchestrationWarning
from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService
from huddleroom.services.orchestration_recovery_service import OrchestrationRecoveryService


async def _goal_run(db, project):
    goal = OrchestrationGoal(project_id=project.id, objective="memory", original_request="memory", success_criteria=[], constraints={}, budget={})
    db.add(goal); await db.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="baseline")
    db.add(run); await db.flush()
    return goal, run


async def _section(db, goal, run, provenance, key=None):
    section = OrchestrationMemorySection(project_id=goal.project_id, goal_id=goal.id, run_id=run.id,
        section_key=key or f"section-{uuid.uuid4().hex}", title="section", body="fact", created_by="orchestrator",
        fact_status="unverified", provenance=provenance)
    db.add(section); await db.flush()
    return section


@pytest.mark.asyncio
async def test_unique_accepted_decision_provenance_promotes_memory(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    decision = OrchestrationDecision(run_id=run.id, decision_type="test", input_snapshot={}, parsed_decision={}, validator_status="accepted")
    db_session.add(decision); await db_session.flush()
    section = await _section(db_session, goal, run, {"decision_id": str(decision.id)})
    assert await OrchestrationMemoryService().reconcile_accepted_sources(db_session, goal.id, run.id) == 0
    assert section.fact_status == "accepted"


@pytest.mark.asyncio
async def test_unique_accepted_evidence_provenance_promotes_memory(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    gate = OrchestrationGate(run_id=run.id, success_criterion_key="proof", gate_type="work_completed", required_evidence={})
    db_session.add(gate); await db_session.flush()
    evidence = OrchestrationEvidence(run_id=run.id, gate_id=gate.id, source_type="task", verdict="accepted", evidence_metadata={})
    db_session.add(evidence); await db_session.flush()
    section = await _section(db_session, goal, run, {"evidence_id": str(evidence.id)})
    assert await OrchestrationMemoryService().reconcile_accepted_sources(db_session, goal.id, run.id) == 0
    assert section.fact_status == "accepted"


@pytest.mark.asyncio
async def test_cross_goal_or_run_provenance_never_promotes_memory(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project); other_goal, other_run = await _goal_run(db_session, test_project)
    decision = OrchestrationDecision(run_id=other_run.id, decision_type="test", input_snapshot={}, parsed_decision={}, validator_status="accepted")
    db_session.add(decision); await db_session.flush()
    section = await _section(db_session, goal, run, {"decision_id": str(decision.id)})
    assert await OrchestrationMemoryService().reconcile_accepted_sources(db_session, goal.id, run.id) == 1
    assert section.fact_status == "unverified" and other_goal.id != goal.id


@pytest.mark.asyncio
async def test_section_from_another_run_never_promotes_memory(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    other_run = OrchestrationRun(goal_id=goal.id, status="completed", phase="completed")
    db_session.add(other_run); await db_session.flush()
    decision = OrchestrationDecision(run_id=run.id, decision_type="test", input_snapshot={}, parsed_decision={}, validator_status="accepted")
    db_session.add(decision); await db_session.flush()
    section = await _section(db_session, goal, other_run, {"decision_id": str(decision.id)})
    assert await OrchestrationMemoryService().reconcile_accepted_sources(db_session, goal.id, run.id) == 0
    assert section.fact_status == "unverified"


@pytest.mark.asyncio
async def test_legacy_section_without_run_link_uses_exact_source_run(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    decision = OrchestrationDecision(run_id=run.id, decision_type="test", input_snapshot={}, parsed_decision={}, validator_status="accepted")
    db_session.add(decision); await db_session.flush()
    section = await _section(db_session, goal, run, {"decision_id": str(decision.id)})
    section.run_id = None; await db_session.flush()
    assert await OrchestrationMemoryService().reconcile_accepted_sources(db_session, goal.id, run.id) == 0
    assert section.fact_status == "accepted"


@pytest.mark.asyncio
async def test_ambiguous_missing_candidate_and_rejected_provenance_stays_unverified(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    rejected = OrchestrationDecision(run_id=run.id, decision_type="test", input_snapshot={}, parsed_decision={}, validator_status="rejected")
    candidate = OrchestrationDecision(run_id=run.id, decision_type="test", input_snapshot={}, parsed_decision={}, validator_status="pending")
    db_session.add_all([rejected, candidate]); await db_session.flush()
    sections = [await _section(db_session, goal, run, {}), await _section(db_session, goal, run, {"decision_id": str(rejected.id)}),
                await _section(db_session, goal, run, {"decision_id": str(candidate.id)}),
                await _section(db_session, goal, run, {"decision_id": str(rejected.id), "evidence_id": str(candidate.id)})]
    assert await OrchestrationMemoryService().reconcile_accepted_sources(db_session, goal.id, run.id) == 3
    assert {section.fact_status for section in sections} == {"unverified"}


@pytest.mark.asyncio
async def test_upgrade_with_no_owned_session_runs_from_goal_recovery(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    decision = OrchestrationDecision(run_id=run.id, decision_type="test", input_snapshot={}, parsed_decision={}, validator_status="accepted")
    db_session.add(decision); await db_session.flush()
    section = await _section(db_session, goal, run, {"decision_id": str(decision.id)})
    service = OrchestrationRecoveryService(); snapshot = await service.build_goal_snapshot(db_session, goal.id, __import__("datetime").datetime.now(__import__("datetime").timezone.utc))
    result = await service.apply_goal_recovery(db_session, snapshot, {})
    assert result.classification == "memory_only" and section.fact_status == "accepted"


@pytest.mark.asyncio
async def test_upgrade_uncertainty_creates_one_valid_run_wait(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    await _section(db_session, goal, run, {"decision_id": str(uuid.uuid4())})
    service = OrchestrationRecoveryService()
    for _ in range(2):
        snapshot = await service.build_goal_snapshot(db_session, goal.id, __import__("datetime").datetime.now(__import__("datetime").timezone.utc))
        await service.apply_goal_recovery(db_session, snapshot, {})
    waits = list(await db_session.scalars(select(OrchestrationWait).where(OrchestrationWait.run_id == run.id)))
    assert len(waits) == 1 and waits[0].owner == {"type": "run", "id": str(run.id)} and waits[0].awaited_event["event_type"]
    marker = run.supervision_state["memory_upgrade_reconciled"]
    assert marker["outcome"] == "unresolved" and marker["wait_id"] == str(waits[0].id)


async def _orchestrator_sections(db, goal, run):
    return [await OrchestrationMemoryService().upsert_section(
        db, goal.project_id, goal.id, section_key=key, title=key, body="context",
        run_id=run.id, created_by=f"orchestrator:{key}",
    ) for key in ("goal_definition", "work_report")]


@pytest.mark.asyncio
async def test_source_free_orchestrator_sections_do_not_open_upgrade_wait(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    sections = await _orchestrator_sections(db_session, goal, run)
    assert await OrchestrationMemoryService().reconcile_accepted_sources(db_session, goal.id, run.id) == 0
    recovery = OrchestrationRecoveryService()
    snapshot = await recovery.build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))
    result = await recovery.apply_goal_recovery(db_session, snapshot, {})
    assert result.wait_id is None
    assert not list(await db_session.scalars(select(OrchestrationWait).where(OrchestrationWait.run_id == run.id)))
    assert {section.fact_status for section in sections} == {"unverified"}


@pytest.mark.asyncio
@pytest.mark.parametrize("wait_status", ["open", "cleared"])
async def test_persisted_unresolved_memory_self_heals_and_resolves_attention(db_session, test_project, wait_status):
    goal, run = await _goal_run(db_session, test_project)
    sections = await _orchestrator_sections(db_session, goal, run)
    recovery = OrchestrationRecoveryService()
    wait = await recovery._wait(db_session, run, None, "upgrade_reconciliation")
    wait.status = wait_status
    warning = OrchestrationWarning(goal_id=goal.id, run_id=run.id, warning_type="supervision_attention",
        severity="warning", message="upgrade_reconciliation")
    unrelated = OrchestrationWarning(goal_id=goal.id, run_id=run.id, warning_type="supervision_attention",
        severity="warning", message="unknown_external_effect")
    db_session.add_all([warning, unrelated])
    run.supervision_state = {"memory_upgrade_reconciled": {
        "run_id": str(run.id), "outcome": "unresolved", "unresolved": 2, "wait_id": str(wait.id),
    }}
    await db_session.flush()
    snapshot = await recovery.build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))
    result = await recovery.apply_goal_recovery(db_session, snapshot, {})
    assert result.wait_id is None and wait.status == "cleared"
    assert run.supervision_state["memory_upgrade_reconciled"]["outcome"] == "promoted"
    assert run.supervision_state["memory_upgrade_reconciled"]["unresolved"] == 0
    assert warning.active is False and warning.resolved_by == "system" and unrelated.active is True
    assert {section.fact_status for section in sections} == {"unverified"}


@pytest.mark.asyncio
async def test_upgrade_wait_clears_on_emitted_resolution_event(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    decision = OrchestrationDecision(run_id=run.id, decision_type="test", input_snapshot={}, parsed_decision={}, validator_status="pending")
    db_session.add(decision); await db_session.flush()
    section = await _section(db_session, goal, run, {"decision_id": str(decision.id)})
    recovery = OrchestrationRecoveryService()
    snapshot = await recovery.build_goal_snapshot(db_session, goal.id, __import__("datetime").datetime.now(__import__("datetime").timezone.utc))
    await recovery.apply_goal_recovery(db_session, snapshot, {})
    wait = await db_session.scalar(select(OrchestrationWait).where(OrchestrationWait.run_id == run.id))
    assert wait.awaited_event == {"event_type": "memory.upgrade_resolved", "matcher": {
        "goal_id": str(goal.id), "run_id": str(run.id), "source": "accepted_source_reconciliation",
    }}
    decision.validator_status = "accepted"
    assert await OrchestrationMemoryService().reconcile_accepted_sources(db_session, goal.id, run.id) == 0
    event = await db_session.scalar(select(EventLog).where(EventLog.event_type == "memory.upgrade_resolved"))
    assert event and event.payload == {"goal_id": str(goal.id), "run_id": str(run.id),
                                      "source": "accepted_source_reconciliation"}
    assert section.fact_status == "accepted" and wait.status == "cleared" and wait.cleared_by_event_id == event.id


@pytest.mark.asyncio
async def test_empty_and_valid_provenance_keys_remain_ambiguous(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    decision = OrchestrationDecision(run_id=run.id, decision_type="test", input_snapshot={}, parsed_decision={}, validator_status="accepted")
    db_session.add(decision); await db_session.flush()
    section = await _section(db_session, goal, run, {"decision_id": "", "evidence_id": str(decision.id)})
    assert await OrchestrationMemoryService().reconcile_accepted_sources(db_session, goal.id, run.id) == 1
    assert section.fact_status == "unverified"


@pytest.mark.asyncio
async def test_upgrade_replay_is_idempotent_and_never_invokes_start(db_session, test_project, monkeypatch):
    goal, run = await _goal_run(db_session, test_project)
    decision = OrchestrationDecision(run_id=run.id, decision_type="test", input_snapshot={}, parsed_decision={}, validator_status="accepted")
    db_session.add(decision); await db_session.flush(); section = await _section(db_session, goal, run, {"decision_id": str(decision.id)})
    started = []
    async def start_spy(*args, **kwargs):
        started.append((args, kwargs))
    monkeypatch.setattr("huddleroom.services.orchestration_service.OrchestrationService.start_run", start_spy)
    service = OrchestrationRecoveryService()
    for _ in range(2):
        snapshot = await service.build_goal_snapshot(db_session, goal.id, __import__("datetime").datetime.now(__import__("datetime").timezone.utc))
        await service.apply_goal_recovery(db_session, snapshot, {})
    actions = list(await db_session.scalars(select(OrchestrationAction).where(OrchestrationAction.run_id == run.id)))
    assert section.fact_status == "accepted" and len(actions) == 1 and started == []


# ---------------------------------------------------------------------------
# SPR 180/184: upgrade reconciliation stays a non-corrective reconciliation; when the
# same pass would start a replacement attempt, that attempt becomes one blocker-recovery trigger.
# ---------------------------------------------------------------------------


async def _recovery_triggers(db, run):
    return list(await db.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "recovery_trigger")))


@pytest.mark.asyncio
async def test_upgrade_uncertainty_wait_is_not_a_recovery_trigger(db_session, test_project):
    """Compatibility pin: memory provenance reconciliation needs no approval."""
    goal, run = await _goal_run(db_session, test_project)
    await _section(db_session, goal, run, {"decision_id": str(uuid.uuid4())})
    service = OrchestrationRecoveryService()
    for _ in range(2):
        snapshot = await service.build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))
        await service.apply_goal_recovery(db_session, snapshot, {})
    assert await _recovery_triggers(db_session, run) == []


@pytest.mark.asyncio
async def test_upgrade_pass_with_replacement_attempt_creates_one_trigger_and_no_new_session(
    db_session, test_project, test_agent,
):
    from huddleroom.models.session import Session
    from huddleroom.services.orchestration_recovery_service import RunnerObservation
    from tests.test_orchestration_worker_ownership_recovery import _owned

    goal, run, _action, task, session = await _owned(
        db_session, test_project, test_agent, session_status="pending",
        attempt={"claimed_runner_task_id": "runner-1", "attempt_version": 1,
                 "effect_state": "not_started", "usage_complete": False},
    )
    await _section(db_session, goal, run, {"decision_id": str(uuid.uuid4())})  # unresolved provenance
    service = OrchestrationRecoveryService()
    for _ in range(2):
        snapshot = await service.build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))
        await service.apply_goal_recovery(db_session, snapshot, {
            session.id: RunnerObservation(session.runner_task_id, "unknown", datetime.now(timezone.utc))})

    sessions = list(await db_session.scalars(select(Session).where(Session.task_id == task.id)))
    # Fix-1 (C2): the dead pre-effect attempt is retired (non-corrective); only the replacement awaits approval.
    await db_session.refresh(session)
    assert [row.id for row in sessions] == [session.id] and session.status == "failed"
    assert len(await _recovery_triggers(db_session, run)) == 1
