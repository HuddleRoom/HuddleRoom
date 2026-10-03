"""Task 9 acceptance tests for goal-scoped memory provenance recovery."""
import uuid

import pytest
from sqlalchemy import select

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationDecision, OrchestrationEvidence, OrchestrationGate, OrchestrationGoal, OrchestrationRun, OrchestrationWait
from huddleroom.models.orchestration_memory import OrchestrationMemorySection
from huddleroom.models.event_log import EventLog
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
    assert await OrchestrationMemoryService().reconcile_accepted_sources(db_session, goal.id, run.id) == 4
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
    await _section(db_session, goal, run, {})
    service = OrchestrationRecoveryService()
    for _ in range(2):
        snapshot = await service.build_goal_snapshot(db_session, goal.id, __import__("datetime").datetime.now(__import__("datetime").timezone.utc))
        await service.apply_goal_recovery(db_session, snapshot, {})
    waits = list(await db_session.scalars(select(OrchestrationWait).where(OrchestrationWait.run_id == run.id)))
    assert len(waits) == 1 and waits[0].owner == {"type": "run", "id": str(run.id)} and waits[0].awaited_event["event_type"]
    marker = run.supervision_state["memory_upgrade_reconciled"]
    assert marker["outcome"] == "unresolved" and marker["wait_id"] == str(waits[0].id)


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
