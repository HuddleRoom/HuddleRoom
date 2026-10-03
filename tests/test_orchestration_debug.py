"""Tests for OrchestrationDebugService.step(), .retry_failed(), and .rerun_last()
-- advance, retry, or rerun one baseline process on demand, outside of tick(). See
huddleroom/services/orchestration_debug_service.py.
"""
import asyncio
import copy
import json
import uuid
from datetime import timedelta

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from huddleroom.models.agent import Agent
from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationEvidence,
    OrchestrationGate,
    OrchestrationGoal,
    OrchestrationRun,
)
from huddleroom.models.orchestration_process import OrchestrationProcessRun
from huddleroom.models.project import Project
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_debug_service import OrchestrationDebugService
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.usefixtures("safe_goal_analysis", "safe_effectiveness_review"),
]


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


async def _make_goal(db, project, *, weight="trivial", status="active"):
    goal = OrchestrationGoal(
        project_id=project.id,
        objective="Fix typo in README",
        success_criteria=[
            {"key": "done", "description": "it is done", "evidence": "diff merged and reviewed"}
        ],
        weight=weight,
        status=status,
    )
    db.add(goal)
    await db.flush()
    return goal


async def _make_run(db, goal, *, status=None, started_at=None):
    run = OrchestrationRun(goal_id=goal.id)
    if started_at is not None:
        run.started_at = started_at
    db.add(run)
    await db.flush()
    if status is not None:
        run.status = status
        await db.flush()
    return run


async def _seed_terminal(db, goal_id, process_type, run_id, *, terminal="completed"):
    """Fabricate a terminal predecessor row directly via OrchestrationProcessService,
    bypassing the real process's business logic (mirrors test_baseline_processes_locking.py).
    """
    svc = OrchestrationProcessService()
    if terminal == "completed":
        proc = await svc.start_process(
            db, goal_id, process_type=process_type, trigger_reason="test seed", run_id=run_id,
        )
        return await svc.complete_process(db, proc)
    return await svc.skip_process(
        db, goal_id, process_type=process_type, skipped_by="human:test-seed", reason="test seed",
        run_id=run_id,
    )


async def _seed_closeout_preconditions(db, project, goal, run):
    """Real gates/evidence/final-summary artifacts satisfying
    _closeout_preconditions_manifest, mirroring the completion_ready_goal
    fixture in tests/test_orchestration_goal_closeout.py."""
    writer = Agent(
        name=f"closeout-writer-{uuid.uuid4()}", role="writer", provider="openai",
        model="gpt-4o-mini", adapter_type="api", capabilities=["summarization"],
        config={}, is_active=True,
    )
    db.add(writer)
    await db.flush()
    gate = OrchestrationGate(
        run_id=run.id, success_criterion_key="plan_item:done", gate_type="work_completed",
        required_evidence={"success_criterion_keys": ["done"]}, status="accepted",
    )
    db.add(gate)
    await db.flush()
    evidence = OrchestrationEvidence(
        run_id=run.id, gate_id=gate.id, source_type="verification", source_id=goal.id,
        verdict="accepted", evidence_metadata={},
    )
    db.add(evidence)
    await db.flush()
    final_gate = OrchestrationGate(
        run_id=run.id, success_criterion_key="final_summary",
        gate_type="final_summary_accepted", required_evidence={}, status="accepted",
    )
    db.add(final_gate)
    await db.flush()
    task = Task(
        project_id=project.id, title="Summarize closeout", status="done",
        assigned_to=writer.id,
        metadata_={
            "orchestration": {
                "run_id": str(run.id), "gate_id": str(final_gate.id),
                "work_function": "summarization", "final_summary": True,
            }
        },
    )
    db.add(task)
    await db.flush()
    session = Session(
        project_id=project.id, task_id=task.id, agent_id=writer.id, adapter_type="api",
        status="completed",
        output=json.dumps({
            "summary": "Work is complete.",
            "criteria": [{"criterion_key": "done", "evidence_ids": [str(evidence.id)]}],
            "unresolved_gaps": [],
        }),
        metadata_={}, origin="auto",
    )
    db.add(session)
    await db.flush()
    final_evidence = OrchestrationEvidence(
        run_id=run.id, gate_id=final_gate.id, source_type="session", source_id=session.id,
        producer_agent_id=writer.id, verdict="accepted", evidence_metadata={},
    )
    db.add(final_evidence)
    await db.flush()


async def _seed_process_row(
    db, goal_id, process_type, *, status, process_version=1, input_snapshot=None,
    outputs=None, completed_at=None, created_at=None, run_id=None, id=None,
):
    """Directly fabricate a terminal OrchestrationProcessRun row, bypassing
    start_process/skip_process, so ordering tests can force explicit
    completed_at/created_at/id values instead of relying on wall-clock timing.
    """
    row = OrchestrationProcessRun(
        id=id or uuid.uuid4(),
        goal_id=goal_id,
        run_id=run_id,
        process_type=process_type,
        process_version=process_version,
        status=status,
        trigger_reason="test seed",
        input_snapshot=input_snapshot or {},
        outputs=outputs or {},
        completed_at=completed_at or _utcnow(),
    )
    db.add(row)
    await db.flush()
    if created_at is not None:
        row.created_at = created_at
        await db.flush()
    return row


async def _seed_full_baseline(db, goal, run):
    # goal_definition/manager_selection have no post-hoc fingerprint check in
    # _baseline_processes_ready_for_goal, so a synthetic "completed" row is
    # fine. agent_definition_review/team_hierarchy DO re-verify a completed
    # row's stored fingerprint against a freshly computed one -- a synthetic
    # row with no fingerprint in outputs would fail that check, so those two
    # are seeded "skipped" instead (still terminal, no fingerprint re-check).
    await _seed_terminal(db, goal.id, "goal_definition", run.id)
    await _seed_terminal(db, goal.id, "manager_selection", run.id)
    await _seed_terminal(db, goal.id, "agent_definition_review", run.id, terminal="skipped")
    await _seed_terminal(db, goal.id, "team_hierarchy", run.id, terminal="skipped")
    # skip_process's generic Phase 2 warning is active by default; closeout
    # preconditions require every active warning to be acknowledged.
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    warning_service = OrchestrationWarningService()
    for warning in await warning_service.list_warnings(db, goal.id, active_only=True):
        await warning_service.acknowledge_warning(db, warning, acknowledged_by="human:test-seed")


# ---------------------------------------------------------------------------
# 1. Six-type happy path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("process_type,expected_status", [
    ("goal_definition", "completed"),
    ("manager_selection", "completed"),
    ("agent_definition_review", "completed"),
    ("team_hierarchy", "completed"),
    ("effectiveness_review", "waiting_decision"),
    ("goal_closeout", "completed"),
])
async def test_six_type_happy_path(db_session, test_project, process_type, expected_status):
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)

    if process_type == "manager_selection":
        await _seed_terminal(db_session, goal.id, "goal_definition", run.id)
    elif process_type == "agent_definition_review":
        await _seed_terminal(db_session, goal.id, "goal_definition", run.id)
        await _seed_terminal(db_session, goal.id, "manager_selection", run.id)
    elif process_type == "team_hierarchy":
        await _seed_terminal(db_session, goal.id, "goal_definition", run.id)
        await _seed_terminal(db_session, goal.id, "manager_selection", run.id)
        await _seed_terminal(db_session, goal.id, "agent_definition_review", run.id)
    elif process_type == "effectiveness_review":
        run.started_at = _utcnow() - timedelta(hours=25)
        await db_session.flush()
    elif process_type == "goal_closeout":
        await _seed_full_baseline(db_session, goal, run)
        await _seed_closeout_preconditions(db_session, test_project, goal, run)

    result = await OrchestrationDebugService().step(db_session, test_project.id, goal.id, process_type)

    assert result["process_type"] == process_type
    assert result["goal_id"] == goal.id
    assert result["run_id"] == run.id
    assert result["process"]["status"] == expected_status

    current = await OrchestrationProcessService().get_current(db_session, goal.id, process_type)
    assert current is not None
    assert current.status == expected_status


# ---------------------------------------------------------------------------
# 2-6. Input validation / gating
# ---------------------------------------------------------------------------


async def test_unsupported_process_type_returns_400(db_session, test_project):
    goal = await _make_goal(db_session, test_project)
    await _make_run(db_session, goal)

    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationDebugService().step(db_session, test_project.id, goal.id, "not_a_real_process")

    assert exc_info.value.status_code == 400
    assert "not_a_real_process" in exc_info.value.detail


async def test_missing_goal_returns_404(db_session, test_project):
    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationDebugService().step(
            db_session, test_project.id, uuid.uuid4(), "goal_definition"
        )
    assert exc_info.value.status_code == 404


@pytest.mark.parametrize("goal_status", ["completed", "cancelled", "paused"])
async def test_non_advanceable_goal_status_returns_409(db_session, test_project, goal_status):
    goal = await _make_goal(db_session, test_project, status=goal_status)
    await _make_run(db_session, goal)

    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationDebugService().step(db_session, test_project.id, goal.id, "goal_definition")

    assert exc_info.value.status_code == 409
    assert goal_status in exc_info.value.detail


async def test_no_active_run_returns_409(db_session, test_project):
    goal = await _make_goal(db_session, test_project)

    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationDebugService().step(db_session, test_project.id, goal.id, "goal_definition")

    assert exc_info.value.status_code == 409
    assert "no active run" in exc_info.value.detail


async def test_retry_rejects_checkpoint_from_a_previous_run(db_session, test_project):
    """A frozen request cannot attach its effects to a newer active run."""
    goal = await _make_goal(db_session, test_project)
    old_run = await _make_run(db_session, goal, status="completed")
    current = await OrchestrationProcessService().start_process(
        db_session, goal.id, process_type="goal_definition", trigger_reason="old failure", run_id=old_run.id,
    )
    current.outputs = {"_lm_retry": {"version": 1, "kind": "goal_analysis", "request": {}}}
    active_run = await _make_run(db_session, goal)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationDebugService().retry_failed(
            db_session, test_project.id, goal.id, "goal_definition"
        )

    assert exc_info.value.status_code == 409
    assert active_run.id != current.run_id


async def test_retry_dispatches_valid_manager_request(db_session, test_project, monkeypatch):
    from huddleroom.services.orchestration_manager_analyzer import ManagerSelectionAnalyzer
    from huddleroom.services.orchestration_manager_selection import ManagerSelectionProcess

    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    current = await OrchestrationProcessService().start_process(
        db_session, goal.id, process_type="manager_selection", trigger_reason="test retry", run_id=run.id,
    )
    request = ManagerSelectionAnalyzer.build_request({
        "schema_version": 1, "goal": {},
        "candidates": [{"key": "human_as_manager"}],
        "deterministic_recommendation": "human_as_manager",
    })
    current.outputs = {"_lm_retry": {"kind": "manager_selection", "version": 1, "request": request}}
    called = False

    async def retry_failed(self, db, locked_goal, active_run, process_run):
        nonlocal called
        called = True
        assert process_run is current
        return {"process_type": "manager_selection", "status": "running"}

    monkeypatch.setattr(ManagerSelectionProcess, "retry_failed", retry_failed)
    result = await OrchestrationDebugService().retry_failed(
        db_session, test_project.id, goal.id, "manager_selection"
    )

    assert called is True
    assert result["process_type"] == "manager_selection"


async def test_retry_dispatches_frozen_team_hierarchy_request_and_creates_proposal_decision(
    db_session, test_project, monkeypatch
):
    from huddleroom.services.orchestration_team_hierarchy import TeamHierarchyProcess
    from huddleroom.services.orchestration_team_hierarchy_analyzer import (
        TeamHierarchyAnalysis,
        TeamHierarchyAnalyzer,
    )

    goal = await _make_goal(db_session, test_project, weight="substantial")
    run = await _make_run(db_session, goal)
    process = TeamHierarchyProcess()
    payload = await process._analysis_input(db_session, goal)
    current = await OrchestrationProcessService().start_process(
        db_session,
        goal.id,
        process_type="team_hierarchy",
        trigger_reason="failed hierarchy",
        run_id=run.id,
        input_snapshot={"fingerprint": process._semantic_fingerprint(payload)},
    )
    request = TeamHierarchyAnalyzer.build_request(payload)
    current.outputs = {"_lm_retry": {"kind": "team_hierarchy", "version": 1, "request": request}}
    captured = None

    async def review_request(self, frozen_request, *, project_id=None):
        nonlocal captured
        captured = frozen_request
        return TeamHierarchyAnalysis(
            proposed_agents=({
                "proposal_id": "writer",
                "definition": {
                    "name": "writer", "role": "writer", "description": "Summarizes completed work.",
                    "provider": "openai", "model": "gpt-4o-mini",
                    "system_prompt": "Summarize supplied work accurately and flag missing evidence.",
                    "adapter_type": "api", "cli_runtime": None,
                    "capabilities": ["summarization"], "config": {},
                },
            },),
            assignments=(), reporting_lines=(),
            documented_gaps=tuple(payload["required_work_functions"]),
            rationale="Required work remains documented as gaps.",
            self_review="Checked the proposal and exact documented gaps.",
        )

    monkeypatch.setattr(TeamHierarchyAnalyzer, "review_request", review_request)

    result = await OrchestrationDebugService().retry_failed(
        db_session, test_project.id, goal.id, "team_hierarchy"
    )

    assert captured == request
    assert result["process"] == {
        "process_type": "team_hierarchy", "status": "waiting_decision", "questions_created": 1,
    }
    decisions = await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id)
    assert [decision.decision_key for decision in decisions] == ["team_hierarchy:agent:writer"]


@pytest.mark.parametrize("status", ["waiting_decision", "completed", "skipped"])
async def test_retry_rejects_non_running_checkpoint_process(db_session, test_project, status):
    """Only the active running process may replay a saved LM request."""
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    current = await OrchestrationProcessService().start_process(
        db_session, goal.id, process_type="goal_definition", trigger_reason="test retry", run_id=run.id,
    )
    current.status = status
    current.outputs = {"_lm_retry": {"version": 1, "kind": "goal_analysis", "request": {}}}
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationDebugService().retry_failed(
            db_session, test_project.id, goal.id, "goal_definition"
        )

    assert exc_info.value.status_code == 409


async def test_retry_rejects_checkpoint_awaiting_authority_interview(db_session, test_project):
    """Retry cannot bypass a pending authority decision attached to its process."""
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    current = await OrchestrationProcessService().start_process(
        db_session, goal.id, process_type="goal_definition", trigger_reason="test retry", run_id=run.id,
    )
    current.outputs = {"_lm_retry": {"version": 1, "kind": "goal_analysis", "request": {}}}
    await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key="goal_definition:retry", title="Retry question",
        question="Proceed?", authority="human", run_id=run.id, source_process_run_id=current.id,
    )
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationDebugService().retry_failed(
            db_session, test_project.id, goal.id, "goal_definition"
        )

    assert exc_info.value.status_code == 409


async def test_retry_rejects_authority_interview_process_type(db_session, test_project):
    """Authority-interview is a process type, but never an LM retry action target."""
    goal = await _make_goal(db_session, test_project)
    await _make_run(db_session, goal)

    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationDebugService().retry_failed(
            db_session, test_project.id, goal.id, "authority_interview"
        )

    assert exc_info.value.status_code == 400


async def test_retry_rejects_when_no_active_run_exists(db_session, test_project):
    """A retry checkpoint without an active run is not executable."""
    goal = await _make_goal(db_session, test_project)
    current = await OrchestrationProcessService().start_process(
        db_session, goal.id, process_type="goal_definition", trigger_reason="orphaned retry",
    )
    current.outputs = {"_lm_retry": {"version": 1, "kind": "goal_analysis", "request": {}}}
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationDebugService().retry_failed(
            db_session, test_project.id, goal.id, "goal_definition"
        )

    assert exc_info.value.status_code == 409


async def test_retry_uses_current_identity_not_a_superseded_checkpoint(db_session, test_project):
    """A checkpoint on a superseded row cannot be selected by process type alone."""
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    process_service = OrchestrationProcessService()
    old = await process_service.start_process(
        db_session, goal.id, process_type="goal_definition", trigger_reason="old retry", run_id=run.id,
    )
    old.status = "completed"
    old.outputs = {"_lm_retry": {"version": 1, "kind": "goal_analysis", "request": {}}}
    await db_session.flush()
    current = await process_service.start_process(
        db_session, goal.id, process_type="goal_definition", trigger_reason="new process", run_id=run.id,
    )
    await db_session.flush()
    assert old.superseded_by_id == current.id

    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationDebugService().retry_failed(
            db_session, test_project.id, goal.id, "goal_definition"
        )

    assert exc_info.value.status_code == 409


async def test_step_reruns_waiting_process_and_cancels_its_pending_decisions(
    db_session, test_project, test_user
):
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    process_service = OrchestrationProcessService()
    old_process = await process_service.start_process(
        db_session,
        goal.id,
        process_type="goal_definition",
        trigger_reason="test seed",
        run_id=run.id,
        input_snapshot={"marker": "original"},
        process_version=7,
    )
    await process_service.park_process(db_session, old_process)

    decision_service = OrchestrationAuthorityDecisionService()
    pending_decisions = [
        await decision_service.create_pending(
            db_session,
            goal.id,
            decision_key=f"goal_definition:test-{index}",
            title=f"Test question {index}",
            question="What should happen?",
            authority="human",
            run_id=run.id,
            source_process_run_id=old_process.id,
        )
        for index in range(2)
    ]
    answered_decision = await decision_service.create_pending(
        db_session,
        goal.id,
        decision_key="goal_definition:answered",
        title="Answered question",
        question="What already happened?",
        authority="human",
        run_id=run.id,
        source_process_run_id=old_process.id,
    )
    await decision_service.answer_decision(
        db_session,
        answered_decision,
        selected_option="already answered",
        decided_by_user_id=test_user.id,
    )
    other_process = await process_service.start_process(
        db_session,
        goal.id,
        process_type="effectiveness_review",
        trigger_reason="other process",
        run_id=run.id,
    )
    other_pending_decision = await decision_service.create_pending(
        db_session,
        goal.id,
        decision_key="effectiveness_review:pending",
        title="Other process question",
        question="What remains pending?",
        authority="human",
        run_id=run.id,
        source_process_run_id=other_process.id,
    )

    result = await OrchestrationDebugService().step(
        db_session, test_project.id, goal.id, "goal_definition"
    )

    current = await process_service.get_current(db_session, goal.id, "goal_definition")
    assert current is not None
    assert current.id != old_process.id
    assert current.process_type == "goal_definition"
    assert current.input_snapshot == {"marker": "original"}
    assert current.process_version == 7
    assert result["process"]["status"] == current.status

    await db_session.refresh(old_process)
    assert old_process.superseded_by_id == current.id
    for decision in pending_decisions:
        await db_session.refresh(decision)
        assert decision.status == "cancelled"
        assert decision.source_process_run_id == old_process.id
    await db_session.refresh(answered_decision)
    assert answered_decision.status == "answered"
    await db_session.refresh(other_pending_decision)
    assert other_pending_decision.status == "pending"


async def test_step_waiting_reset_rolls_back_when_successor_advance_fails(
    db_session, test_project, monkeypatch
):
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    process_service = OrchestrationProcessService()
    old_process = await process_service.start_process(
        db_session,
        goal.id,
        process_type="goal_definition",
        trigger_reason="test seed",
        run_id=run.id,
    )
    await process_service.park_process(db_session, old_process)

    decision_service = OrchestrationAuthorityDecisionService()
    pending_decisions = [
        await decision_service.create_pending(
            db_session,
            goal.id,
            decision_key=f"goal_definition:pending-{index}",
            title=f"Pending question {index}",
            question="What should happen?",
            authority="human",
            run_id=run.id,
            source_process_run_id=old_process.id,
        )
        for index in range(2)
    ]

    async def fail_successor_advance(*_args):
        raise RuntimeError("successor advance failed")

    monkeypatch.setattr(OrchestrationDebugService, "_advance", fail_successor_advance)

    with pytest.raises(RuntimeError, match="successor advance failed"):
        await OrchestrationDebugService().step(
            db_session, test_project.id, goal.id, "goal_definition"
        )

    assert db_session.in_transaction()
    current = await process_service.get_current(db_session, goal.id, "goal_definition")
    assert current is not None
    assert current.id == old_process.id
    assert current.status == "waiting_decision"
    for decision in pending_decisions:
        await db_session.refresh(decision)
        assert decision.status == "pending"

    rows = (
        await db_session.execute(
            select(OrchestrationProcessRun).where(
                OrchestrationProcessRun.goal_id == goal.id,
                OrchestrationProcessRun.process_type == "goal_definition",
            )
        )
    ).scalars().all()
    assert [row.id for row in rows] == [old_process.id]


async def test_paused_run_returns_409_not_tickable(db_session, test_project):
    goal = await _make_goal(db_session, test_project)
    await _make_run(db_session, goal, status="paused")

    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationDebugService().step(db_session, test_project.id, goal.id, "goal_definition")

    assert exc_info.value.status_code == 409
    assert "not tickable" in exc_info.value.detail
    assert "paused" in exc_info.value.detail


# ---------------------------------------------------------------------------
# 7. Predecessor gating (non-closeout)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("process_type,predecessor", [
    ("manager_selection", "goal_definition"),
    ("agent_definition_review", "manager_selection"),
    ("team_hierarchy", "agent_definition_review"),
])
async def test_predecessor_not_terminal_returns_409(db_session, test_project, process_type, predecessor):
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)

    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationDebugService().step(db_session, test_project.id, goal.id, process_type)

    assert exc_info.value.status_code == 409
    assert predecessor in exc_info.value.detail
    assert process_type in exc_info.value.detail

    current = await OrchestrationProcessService().get_current(db_session, goal.id, process_type)
    assert current is None


# ---------------------------------------------------------------------------
# 8. goal_closeout predecessor gating
# ---------------------------------------------------------------------------


async def test_goal_closeout_requires_full_baseline_terminal(db_session, test_project):
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    # Only two of the four baseline processes are terminal.
    await _seed_terminal(db_session, goal.id, "goal_definition", run.id)
    await _seed_terminal(db_session, goal.id, "manager_selection", run.id)

    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationDebugService().step(db_session, test_project.id, goal.id, "goal_closeout")

    assert exc_info.value.status_code == 409
    # The message now names the specific not-ready predecessor instead of a blanket
    # "must all be terminal" (which was a lie when the real cause was a stale process).
    assert "goal_closeout cannot advance" in exc_info.value.detail
    assert "'agent_definition_review' is not yet complete" in exc_info.value.detail

    current = await OrchestrationProcessService().get_current(db_session, goal.id, "goal_closeout")
    assert current is None


async def test_goal_closeout_no_executable_work_completes(db_session, test_project):
    """(#1) A goal that reaches closeout with the full baseline terminal but no gates
    to authorize has no executable work; closeout completes so the step is terminal
    (done) rather than stuck, instead of raising on the missing manifest."""
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    await _seed_full_baseline(db_session, goal, run)
    # No gates/evidence/final-summary artifacts at all.

    result = await OrchestrationDebugService().step(
        db_session, test_project.id, goal.id, "goal_closeout"
    )

    assert result["process"]["status"] == "completed"
    assert result["process"]["completion_authorized"] is True
    assert result["process"]["no_executable_work"] is True

    current = await OrchestrationProcessService().get_current(db_session, goal.id, "goal_closeout")
    assert current is not None
    assert current.status == "completed"
    assert current.outputs.get("no_executable_work") is True


# ---------------------------------------------------------------------------
# 9. Side-effect boundary
# ---------------------------------------------------------------------------


async def test_step_never_chains_into_next_process(db_session, test_project):
    """(a) stepping manager_selection to completion leaves agent_definition_review untouched."""
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    await _seed_terminal(db_session, goal.id, "goal_definition", run.id)

    result = await OrchestrationDebugService().step(
        db_session, test_project.id, goal.id, "manager_selection"
    )
    assert result["process"]["status"] == "completed"

    current = await OrchestrationProcessService().get_current(
        db_session, goal.id, "agent_definition_review"
    )
    assert current is None


async def test_goal_closeout_step_does_not_complete_goal_or_run_and_leaves_no_other_side_effects(
    db_session, test_project
):
    """(b)(c)(d): OrchestrationAction/Task counts, event cursor, and pre-existing
    gate rows are unchanged by step(); the goal/run stay active/running even
    though the closeout process itself authorizes completion."""
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    await _seed_full_baseline(db_session, goal, run)
    await _seed_closeout_preconditions(db_session, test_project, goal, run)

    tasks_before = (
        await db_session.execute(select(Task).where(Task.project_id == test_project.id))
    ).scalars().all()
    actions_before = (
        await db_session.execute(select(OrchestrationAction).where(OrchestrationAction.run_id == run.id))
    ).scalars().all()
    gates_before = (
        await db_session.execute(
            select(OrchestrationGate)
            .where(OrchestrationGate.run_id == run.id)
            .order_by(OrchestrationGate.id)
        )
    ).scalars().all()
    gates_before_snapshot = [(gate.id, gate.status) for gate in gates_before]
    event_cursor_before = run.event_cursor

    result = await OrchestrationDebugService().step(db_session, test_project.id, goal.id, "goal_closeout")
    assert result["process"]["status"] == "completed"
    assert result["process"]["completion_authorized"] is True

    tasks_after = (
        await db_session.execute(select(Task).where(Task.project_id == test_project.id))
    ).scalars().all()
    actions_after = (
        await db_session.execute(select(OrchestrationAction).where(OrchestrationAction.run_id == run.id))
    ).scalars().all()
    gates_after = (
        await db_session.execute(
            select(OrchestrationGate)
            .where(OrchestrationGate.run_id == run.id)
            .order_by(OrchestrationGate.id)
        )
    ).scalars().all()
    gates_after_snapshot = [(gate.id, gate.status) for gate in gates_after]

    assert len(tasks_after) == len(tasks_before)
    assert len(actions_after) == len(actions_before) == 0
    assert gates_after_snapshot == gates_before_snapshot

    await db_session.refresh(run)
    await db_session.refresh(goal)
    assert run.event_cursor == event_cursor_before
    assert goal.status == "active"
    assert run.status == "running"


# ---------------------------------------------------------------------------
# 10. Concurrent idempotency
# ---------------------------------------------------------------------------


async def test_concurrent_step_serializes_and_does_not_duplicate_process_row(
    concurrent_sessions, tmp_path
):
    session1, session2 = concurrent_sessions
    workspace = tmp_path / "concurrent-debug-workspace"
    workspace.mkdir()
    project = Project(
        name=f"debug-concurrent-{uuid.uuid4()}",
        workspace_path=str(workspace.resolve()),
        config={},
    )
    session1.add(project)
    await session1.commit()

    goal = await _make_goal(session1, project)
    run = await _make_run(session1, goal)
    await session1.commit()

    svc = OrchestrationDebugService()
    results = await asyncio.gather(
        svc.step(session1, project.id, goal.id, "goal_definition"),
        svc.step(session2, project.id, goal.id, "goal_definition"),
    )

    assert [result["process"]["status"] for result in results] == ["completed", "completed"]

    rows = (
        await session1.execute(
            select(OrchestrationProcessRun).where(
                OrchestrationProcessRun.goal_id == goal.id,
                OrchestrationProcessRun.process_type == "goal_definition",
            )
        )
    ).scalars().all()
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# 11. Transaction ownership
# ---------------------------------------------------------------------------


async def test_step_preserves_caller_explicit_transaction(db_session, test_project):
    """db_session already holds an explicit (non-AUTOBEGIN) transaction; step()
    must not commit it out from under the caller, and the session must stay
    usable afterward."""
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)

    result = await OrchestrationDebugService().step(db_session, test_project.id, goal.id, "goal_definition")
    assert result["process"]["status"] == "completed"

    # Session must remain usable -- no InvalidRequestError from a transaction
    # step() should not have touched.
    assert db_session.in_transaction()
    again = (await db_session.execute(select(OrchestrationGoal).where(OrchestrationGoal.id == goal.id))).scalar_one()
    assert again.id == goal.id


async def test_step_commits_when_it_owns_the_transaction(concurrent_sessions, tmp_path):
    """A bare session (no explicit begin -- step() owns the AUTOBEGIN
    transaction) must durably commit, visible to a separate verify session."""
    writer, verifier = concurrent_sessions
    workspace = tmp_path / "durable-debug-workspace"
    workspace.mkdir()
    project = Project(
        name=f"debug-durable-{uuid.uuid4()}",
        workspace_path=str(workspace.resolve()),
        config={},
    )
    writer.add(project)
    await writer.commit()

    goal = await _make_goal(writer, project)
    run = await _make_run(writer, goal)
    await writer.commit()

    result = await OrchestrationDebugService().step(writer, project.id, goal.id, "goal_definition")
    assert result["process"]["status"] == "completed"

    persisted = await verifier.scalar(
        select(OrchestrationProcessRun).where(
            OrchestrationProcessRun.goal_id == goal.id,
            OrchestrationProcessRun.process_type == "goal_definition",
        )
    )
    assert persisted is not None
    assert persisted.status == "completed"


# ---------------------------------------------------------------------------
# rerun_last() -- auto-select and rerun the single latest terminal baseline
# process across all six types.
# ---------------------------------------------------------------------------


# 1. Selection ordering


async def test_select_last_terminal_picks_latest_completed_at(db_session, test_project):
    goal = await _make_goal(db_session, test_project)
    await _seed_process_row(
        db_session, goal.id, "goal_definition", status="completed",
        completed_at=_utcnow() - timedelta(hours=3),
    )
    await _seed_process_row(
        db_session, goal.id, "manager_selection", status="completed",
        completed_at=_utcnow() - timedelta(hours=2),
    )
    latest = await _seed_process_row(
        db_session, goal.id, "agent_definition_review", status="completed",
        completed_at=_utcnow() - timedelta(hours=1),
    )

    selected = await OrchestrationDebugService()._select_last_terminal_process(db_session, goal.id)
    assert selected.id == latest.id
    assert selected.process_type == "agent_definition_review"


async def test_select_last_terminal_ties_break_on_created_at(db_session, test_project):
    goal = await _make_goal(db_session, test_project)
    shared_completed_at = _utcnow() - timedelta(hours=1)
    await _seed_process_row(
        db_session, goal.id, "goal_definition", status="completed",
        completed_at=shared_completed_at, created_at=_utcnow() - timedelta(hours=5),
    )
    later = await _seed_process_row(
        db_session, goal.id, "manager_selection", status="completed",
        completed_at=shared_completed_at, created_at=_utcnow() - timedelta(hours=4),
    )

    selected = await OrchestrationDebugService()._select_last_terminal_process(db_session, goal.id)
    assert selected.id == later.id


async def test_select_last_terminal_ties_break_on_id(db_session, test_project):
    goal = await _make_goal(db_session, test_project)
    shared_ts = _utcnow() - timedelta(hours=1)
    low_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    high_id = uuid.UUID("ffffffff-ffff-ffff-ffff-fffffffffffe")
    await _seed_process_row(
        db_session, goal.id, "goal_definition", status="completed", id=low_id,
        completed_at=shared_ts, created_at=shared_ts,
    )
    await _seed_process_row(
        db_session, goal.id, "manager_selection", status="completed", id=high_id,
        completed_at=shared_ts, created_at=shared_ts,
    )

    selected = await OrchestrationDebugService()._select_last_terminal_process(db_session, goal.id)
    assert selected.id == high_id


# 2. Snapshot + version reuse


async def test_rerun_last_reuses_snapshot_and_version(db_session, test_project):
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    original_triggers = [{"name": "manual", "token": "orig-token", "detail": "orig detail"}]
    original_snapshot = {"triggers": original_triggers}
    original = await OrchestrationProcessService().start_process(
        db_session, goal.id, process_type="effectiveness_review", trigger_reason="test seed",
        run_id=run.id, input_snapshot=original_snapshot, process_version=2,
    )
    await OrchestrationProcessService().complete_process(
        db_session, original, outputs={"triggers": original_triggers}
    )

    result = await OrchestrationDebugService().rerun_last(db_session, test_project.id, goal.id)
    assert result["process_type"] == "effectiveness_review"

    new_current = await OrchestrationProcessService().get_current(db_session, goal.id, "effectiveness_review")
    assert new_current.id != original.id
    assert new_current.process_version == 2
    assert new_current.input_snapshot == original_snapshot
    assert new_current.outputs["triggers"] == original_triggers


# 2b. Fingerprinted types (team_hierarchy, agent_definition_review): drift vs.
# no-drift end state. input_snapshot for these two is a drift-detection
# fingerprint, not replay data -- see rerun_last()'s docstring.


async def test_rerun_last_team_hierarchy_no_drift_keeps_single_supersession(db_session, test_project):
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    await _seed_terminal(db_session, goal.id, "goal_definition", run.id)
    await _seed_terminal(db_session, goal.id, "manager_selection", run.id)
    await _seed_terminal(db_session, goal.id, "agent_definition_review", run.id)

    step_result = await OrchestrationDebugService().step(db_session, test_project.id, goal.id, "team_hierarchy")
    assert step_result["process"]["status"] == "completed"
    original = await OrchestrationProcessService().get_current(db_session, goal.id, "team_hierarchy")
    original_id = original.id
    original_version = original.process_version

    # No drift: fingerprint inputs (goal fields) are unchanged from the original run.
    result = await OrchestrationDebugService().rerun_last(db_session, test_project.id, goal.id)
    assert result["process_type"] == "team_hierarchy"

    rows = (
        await db_session.execute(
            select(OrchestrationProcessRun).where(
                OrchestrationProcessRun.goal_id == goal.id,
                OrchestrationProcessRun.process_type == "team_hierarchy",
            )
        )
    ).scalars().all()
    assert len(rows) == 2  # original + single successor -- no internal re-supersession (ghost row)

    current_rows = [row for row in rows if row.superseded_by_id is None]
    assert len(current_rows) == 1
    final = current_rows[0]
    assert final.id != original_id
    assert final.process_version == original_version
    assert final.status == "completed"


async def test_rerun_last_team_hierarchy_drift_recomputes_and_supersedes_again(db_session, test_project):
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    await _seed_terminal(db_session, goal.id, "goal_definition", run.id)
    await _seed_terminal(db_session, goal.id, "manager_selection", run.id)
    await _seed_terminal(db_session, goal.id, "agent_definition_review", run.id)

    step_result = await OrchestrationDebugService().step(db_session, test_project.id, goal.id, "team_hierarchy")
    assert step_result["process"]["status"] == "completed"
    original = await OrchestrationProcessService().get_current(db_session, goal.id, "team_hierarchy")
    original_id = original.id
    original_outputs = dict(original.outputs)
    original_completed_at = original.completed_at

    # Drift the fingerprint inputs: authority_model is part of _fingerprint()'s
    # payload (huddleroom/services/orchestration_team_hierarchy.py); flipping it
    # from None to "no_manager" (no manager assigned, so the CHECK constraint
    # is satisfied) changes the recomputed fingerprint without touching any
    # agent/roster state.
    goal.authority_model = "no_manager"
    await db_session.flush()

    result = await OrchestrationDebugService().rerun_last(db_session, test_project.id, goal.id)
    assert result["process_type"] == "team_hierarchy"

    old_row = await db_session.get(OrchestrationProcessRun, original_id)
    assert old_row is not None
    assert old_row.superseded_by_id is not None  # superseded, not abandoned in place
    assert old_row.outputs == original_outputs
    assert old_row.completed_at == original_completed_at

    current_rows = (
        await db_session.execute(
            select(OrchestrationProcessRun).where(
                OrchestrationProcessRun.goal_id == goal.id,
                OrchestrationProcessRun.process_type == "team_hierarchy",
                OrchestrationProcessRun.superseded_by_id.is_(None),
            )
        )
    ).scalars().all()
    assert len(current_rows) == 1  # the intermediate (ghost) row, if any, must be superseded
    final = current_rows[0]
    assert final.id != original_id
    assert final.status == "completed"

    tasks = (await db_session.execute(select(Task).where(Task.project_id == test_project.id))).scalars().all()
    assert tasks == []
    actions = (
        await db_session.execute(select(OrchestrationAction).where(OrchestrationAction.run_id == run.id))
    ).scalars().all()
    assert actions == []

    await db_session.refresh(goal)
    await db_session.refresh(run)
    assert goal.status not in ("completed", "cancelled")
    assert run.status not in ("completed", "cancelled")


# 3. Superseded-history preservation


async def test_rerun_last_preserves_superseded_row(db_session, test_project):
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    original = await _seed_terminal(db_session, goal.id, "goal_definition", run.id)
    original_id = original.id
    original_status = original.status
    original_outputs = dict(original.outputs)
    original_completed_at = original.completed_at

    result = await OrchestrationDebugService().rerun_last(db_session, test_project.id, goal.id)
    assert result["process_type"] == "goal_definition"

    old_row = await db_session.get(OrchestrationProcessRun, original_id)
    assert old_row is not None
    assert old_row.status == original_status
    assert old_row.outputs == original_outputs
    assert old_row.completed_at == original_completed_at

    new_current = await OrchestrationProcessService().get_current(db_session, goal.id, "goal_definition")
    assert new_current.id != original_id
    assert old_row.superseded_by_id == new_current.id


# 4. Skipped selection


async def test_rerun_last_selects_skipped_process(db_session, test_project):
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    original = await _seed_terminal(db_session, goal.id, "goal_definition", run.id, terminal="skipped")

    result = await OrchestrationDebugService().rerun_last(db_session, test_project.id, goal.id)
    assert result["process_type"] == "goal_definition"

    new_current = await OrchestrationProcessService().get_current(db_session, goal.id, "goal_definition")
    assert new_current.id != original.id
    assert new_current.status == "completed"


async def test_rerun_last_resolves_every_active_warning_linked_to_skipped_process(
    db_session, test_project
):
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    skipped = await _seed_terminal(db_session, goal.id, "goal_definition", run.id, terminal="skipped")
    warnings = OrchestrationWarningService()
    linked = await warnings.create_warning(
        db_session, goal.id, warning_type="extra_skipped_risk", severity="warning",
        message="Extra risk", run_id=run.id, source_process_run_id=skipped.id,
    )
    other = await warnings.create_warning(
        db_session, goal.id, warning_type="other_risk", severity="warning",
        message="Other risk", run_id=run.id,
    )

    result = await OrchestrationDebugService().rerun_last(db_session, test_project.id, goal.id)

    assert result["process"]["status"] == "completed"
    linked_warnings = [
        warning for warning in await warnings.list_warnings(db_session, goal.id)
        if warning.source_process_run_id == skipped.id
    ]
    assert linked_warnings and all(not warning.active for warning in linked_warnings)
    assert {(warning.resolved_by, warning.resolved_reason) for warning in linked_warnings} == {
        ("orchestrator", "successful rerun of skipped process")
    }
    assert other.active is True


async def test_rerun_selected_terminal_process_not_last(db_session, test_project):
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    selected = await _seed_terminal(db_session, goal.id, "goal_definition", run.id)
    await _seed_terminal(db_session, goal.id, "effectiveness_review", run.id)

    result = await OrchestrationDebugService().rerun_last(
        db_session, test_project.id, goal.id, process_type="goal_definition"
    )

    current = await OrchestrationProcessService().get_current(db_session, goal.id, "goal_definition")
    assert result["process_type"] == "goal_definition"
    assert current is not None
    assert current.id != selected.id


async def test_rerun_selected_terminal_cancels_only_its_pending_decisions(
    db_session, test_project
):
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    selected = await _seed_terminal(db_session, goal.id, "effectiveness_review", run.id)
    other = await _seed_terminal(db_session, goal.id, "goal_definition", run.id)
    decision_service = OrchestrationAuthorityDecisionService()
    selected_decision = await decision_service.create_pending(
        db_session, goal.id, decision_key="effectiveness_review:pending", title="Selected",
        question="Cancel this?", authority="human", run_id=run.id, source_process_run_id=selected.id,
    )
    other_decision = await decision_service.create_pending(
        db_session, goal.id, decision_key="goal_definition:pending", title="Other",
        question="Keep this?", authority="human", run_id=run.id, source_process_run_id=other.id,
    )

    await OrchestrationDebugService().rerun_last(
        db_session, test_project.id, goal.id, process_type="effectiveness_review"
    )

    await db_session.refresh(selected_decision)
    await db_session.refresh(other_decision)
    assert selected_decision.status == "cancelled"
    assert other_decision.status == "pending"


@pytest.mark.parametrize("status", ["running", "missing"])
async def test_rerun_selected_non_rerunnable_process_returns_409(db_session, test_project, status):
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    if status == "running":
        await OrchestrationProcessService().start_process(
            db_session, goal.id, process_type="effectiveness_review", trigger_reason="test seed", run_id=run.id
        )

    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationDebugService().rerun_last(
            db_session, test_project.id, goal.id, process_type="effectiveness_review"
        )

    assert exc_info.value.status_code == 409


# 5. waiting_decision rerun


def valid_goal_retry_checkpoint():
    return {
        "version": 1,
        "kind": "goal_analysis",
        "request": {"model": "test", "messages": [], "response_format": {"type": "json_object"}},
        "continuation": {
            "clarification_round": 0,
            "context": {},
            "working_goal": {"objective": "test", "success_criteria": []},
        },
    }


@pytest.mark.parametrize("status", ["waiting_decision", "running"])
async def test_rerun_last_recovers_retryable_stuck_process(db_session, test_project, status, monkeypatch):
    goal = await _make_goal(db_session, test_project, status="blocked")
    run = await _make_run(db_session, goal, status="blocked")
    process_service = OrchestrationProcessService()
    source = await process_service.start_process(
        db_session, goal.id, process_type="goal_definition", trigger_reason="failed LM",
        run_id=run.id, input_snapshot={"saved": True}, process_version=7,
    )
    if status == "waiting_decision":
        await process_service.park_process(db_session, source)
    source.outputs = {"_lm_retry": valid_goal_retry_checkpoint()}
    warning_service = OrchestrationWarningService()
    linked = await warning_service.create_warning(
        db_session, goal.id, warning_type="goal_definition_analyzer_error", severity="warning",
        message="LM failed", run_id=run.id, source_process_run_id=source.id,
    )
    source.outputs["_lm_retry"]["warning_id"] = str(linked.id)
    run.active_blockers = [
        {"kind": "goal_definition_analyzer_error", "warning_id": str(linked.id)},
        {"kind": "unrelated", "warning_id": "other"},
    ]
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key="goal_definition:pending", title="Pending",
        question="Cancel this?", authority="human", run_id=run.id, source_process_run_id=source.id,
    )
    seen = []

    async def advance(self, db, locked_goal, active_run, process_type):
        successor = await process_service.get_current(db, goal.id, process_type)
        seen.append(successor)
        assert successor.input_snapshot == {"saved": True}
        assert successor.process_version == 7 and successor.outputs == {}
        return {"process_type": process_type, "status": "running"}

    monkeypatch.setattr(OrchestrationDebugService, "_advance", advance)

    await OrchestrationDebugService().rerun_last(
        db_session, test_project.id, goal.id, "goal_definition"
    )

    await db_session.refresh(source)
    await db_session.refresh(linked)
    await db_session.refresh(run)
    await db_session.refresh(goal)
    await db_session.refresh(decision)
    assert seen and source.superseded_by_id == seen[0].id
    assert decision.status == "cancelled"
    assert linked.active is False
    assert run.active_blockers == [{"kind": "unrelated", "warning_id": "other"}]
    assert goal.status == run.status == "blocked"


async def test_rerun_last_stuck_retry_rollback_restores_source(db_session, test_project, monkeypatch):
    goal = await _make_goal(db_session, test_project, status="blocked")
    run = await _make_run(db_session, goal, status="blocked")
    process_service = OrchestrationProcessService()
    source = await process_service.start_process(
        db_session, goal.id, process_type="goal_definition", trigger_reason="failed LM", run_id=run.id,
    )
    source.outputs = {"_lm_retry": valid_goal_retry_checkpoint()}
    warning_service = OrchestrationWarningService()
    linked = await warning_service.create_warning(
        db_session, goal.id, warning_type="goal_definition_analyzer_error", severity="warning",
        message="LM failed", run_id=run.id, source_process_run_id=source.id,
    )
    source.outputs["_lm_retry"]["warning_id"] = str(linked.id)
    run.active_blockers = [{"kind": "goal_definition_analyzer_error", "warning_id": str(linked.id)}]
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key="goal_definition:pending", title="Pending",
        question="Cancel this?", authority="human", run_id=run.id, source_process_run_id=source.id,
    )

    async def fail_advance(self, db, locked_goal, active_run, process_type):
        raise RuntimeError("advance failed")

    monkeypatch.setattr(OrchestrationDebugService, "_advance", fail_advance)
    with pytest.raises(RuntimeError, match="advance failed"):
        await OrchestrationDebugService().rerun_last(
            db_session, test_project.id, goal.id, "goal_definition"
        )

    current = await process_service.get_current(db_session, goal.id, "goal_definition")
    await db_session.refresh(decision)
    await db_session.refresh(linked)
    await db_session.refresh(run)
    await db_session.refresh(goal)
    assert current.id == source.id
    assert current.superseded_by_id is None
    assert decision.status == "pending"
    assert linked.active is True
    assert run.active_blockers == [{"kind": "goal_definition_analyzer_error", "warning_id": str(linked.id)}]
    assert goal.status == run.status == "blocked"


async def test_rerun_last_reruns_waiting_process_and_cancels_its_pending_decisions(
    db_session, test_project, test_user
):
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    process_service = OrchestrationProcessService()
    parked = await process_service.start_process(
        db_session,
        goal.id,
        process_type="effectiveness_review",
        trigger_reason="test seed",
        run_id=run.id,
        input_snapshot={"marker": "original"},
        process_version=7,
    )
    await process_service.park_process(db_session, parked)

    decision_service = OrchestrationAuthorityDecisionService()
    pending_decision = await decision_service.create_pending(
        db_session,
        goal.id,
        decision_key="effectiveness_review:pending",
        title="Pending question",
        question="What should be discarded?",
        authority="human",
        run_id=run.id,
        source_process_run_id=parked.id,
    )
    answered_decision = await decision_service.create_pending(
        db_session,
        goal.id,
        decision_key="effectiveness_review:answered",
        title="Answered question",
        question="What should remain?",
        authority="human",
        run_id=run.id,
        source_process_run_id=parked.id,
    )
    await decision_service.answer_decision(
        db_session,
        answered_decision,
        selected_option="already answered",
        decided_by_user_id=test_user.id,
    )
    terminal_process = await _seed_terminal(db_session, goal.id, "goal_definition", run.id)
    other_pending_decision = await decision_service.create_pending(
        db_session,
        goal.id,
        decision_key="goal_definition:pending",
        title="Other process question",
        question="What should remain pending?",
        authority="human",
        run_id=run.id,
        source_process_run_id=terminal_process.id,
    )

    result = await OrchestrationDebugService().rerun_last(db_session, test_project.id, goal.id)
    assert result["process_type"] == "effectiveness_review"

    current = await process_service.get_current(db_session, goal.id, "effectiveness_review")
    assert current is not None
    assert current.id != parked.id
    assert current.input_snapshot["marker"] == "original"
    assert current.process_version == 7
    assert result["process"]["status"] == current.status

    await db_session.refresh(parked)
    assert parked.superseded_by_id == current.id
    await db_session.refresh(terminal_process)
    assert terminal_process.superseded_by_id is None
    await db_session.refresh(pending_decision)
    assert pending_decision.status == "cancelled"
    await db_session.refresh(answered_decision)
    assert answered_decision.status == "answered"
    await db_session.refresh(other_pending_decision)
    assert other_pending_decision.status == "pending"


# 6. Lifecycle 409s / 404


@pytest.mark.parametrize("goal_status", ["completed", "cancelled", "paused"])
async def test_rerun_last_non_advanceable_goal_status_returns_409(db_session, test_project, goal_status):
    goal = await _make_goal(db_session, test_project, status=goal_status)
    await _make_run(db_session, goal)

    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationDebugService().rerun_last(db_session, test_project.id, goal.id)

    assert exc_info.value.status_code == 409
    assert goal_status in exc_info.value.detail


async def test_rerun_last_missing_goal_returns_404(db_session, test_project):
    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationDebugService().rerun_last(db_session, test_project.id, uuid.uuid4())
    assert exc_info.value.status_code == 404


async def test_rerun_last_no_active_run_returns_409(db_session, test_project):
    goal = await _make_goal(db_session, test_project)

    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationDebugService().rerun_last(db_session, test_project.id, goal.id)

    assert exc_info.value.status_code == 409
    assert "no active run" in exc_info.value.detail


async def test_rerun_last_paused_run_returns_409_not_tickable(db_session, test_project):
    goal = await _make_goal(db_session, test_project)
    await _make_run(db_session, goal, status="paused")

    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationDebugService().rerun_last(db_session, test_project.id, goal.id)

    assert exc_info.value.status_code == 409
    assert "not tickable" in exc_info.value.detail
    assert "paused" in exc_info.value.detail


# 7. No terminal process to rerun


async def test_rerun_last_no_terminal_process_returns_409(db_session, test_project):
    goal = await _make_goal(db_session, test_project)
    await _make_run(db_session, goal)

    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationDebugService().rerun_last(db_session, test_project.id, goal.id)

    assert exc_info.value.status_code == 409
    assert "no baseline process to rerun" in exc_info.value.detail


# 8. Transactional rollback proof


async def test_rerun_last_advance_failure_keeps_skipped_warnings_active(
    db_session, test_project, monkeypatch
):
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    skipped = await _seed_terminal(
        db_session, goal.id, "goal_definition", run.id, terminal="skipped"
    )
    warnings = OrchestrationWarningService()

    async def fail_advance(self, db, locked_goal, active_run, process_type):
        raise RuntimeError("advance failed")

    monkeypatch.setattr(OrchestrationDebugService, "_advance", fail_advance)
    with pytest.raises(RuntimeError, match="advance failed"):
        await OrchestrationDebugService().rerun_last(db_session, test_project.id, goal.id)
    assert (await OrchestrationProcessService().get_current(
        db_session, goal.id, "goal_definition"
    )).id == skipped.id
    assert all(warning.active for warning in await warnings.list_warnings(
        db_session, goal.id, active_only=True
    ) if warning.source_process_run_id == skipped.id)


async def test_rerun_last_rolls_back_on_predecessor_failure(db_session, test_project):
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    await _seed_terminal(db_session, goal.id, "goal_definition", run.id)
    await _seed_terminal(db_session, goal.id, "manager_selection", run.id)
    await _seed_terminal(db_session, goal.id, "agent_definition_review", run.id)
    original_th = await _seed_terminal(db_session, goal.id, "team_hierarchy", run.id, terminal="skipped")

    # Break the live prerequisite team_hierarchy's rerun depends on: its
    # predecessor (agent_definition_review) is no longer terminal.
    adr_current = await OrchestrationProcessService().get_current(db_session, goal.id, "agent_definition_review")
    adr_current.status = "running"
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationDebugService().rerun_last(db_session, test_project.id, goal.id)

    assert exc_info.value.status_code == 409
    assert "agent_definition_review" in exc_info.value.detail
    assert "team_hierarchy" in exc_info.value.detail

    current_th = await OrchestrationProcessService().get_current(db_session, goal.id, "team_hierarchy")
    assert current_th.id == original_th.id
    assert current_th.superseded_by_id is None
    assert current_th.status == "skipped"

    rows = (
        await db_session.execute(
            select(OrchestrationProcessRun).where(
                OrchestrationProcessRun.goal_id == goal.id,
                OrchestrationProcessRun.process_type == "team_hierarchy",
            )
        )
    ).scalars().all()
    assert len(rows) == 1


# 9. Concurrent uniqueness


async def test_concurrent_rerun_last_keeps_exactly_one_current_row(concurrent_sessions, tmp_path):
    session1, session2 = concurrent_sessions
    workspace = tmp_path / "concurrent-rerun-workspace"
    workspace.mkdir()
    project = Project(
        name=f"debug-rerun-concurrent-{uuid.uuid4()}",
        workspace_path=str(workspace.resolve()),
        config={},
    )
    session1.add(project)
    await session1.commit()

    goal = await _make_goal(session1, project)
    run = await _make_run(session1, goal)
    await _seed_terminal(session1, goal.id, "goal_definition", run.id)
    await session1.commit()

    svc = OrchestrationDebugService()
    results = await asyncio.gather(
        svc.rerun_last(session1, project.id, goal.id),
        svc.rerun_last(session2, project.id, goal.id),
    )

    assert [result["process_type"] for result in results] == ["goal_definition", "goal_definition"]

    current_rows = (
        await session1.execute(
            select(OrchestrationProcessRun).where(
                OrchestrationProcessRun.goal_id == goal.id,
                OrchestrationProcessRun.process_type == "goal_definition",
                OrchestrationProcessRun.superseded_by_id.is_(None),
            )
        )
    ).scalars().all()
    assert len(current_rows) == 1


# 10. Side-effect boundary


async def test_rerun_last_only_changes_selected_type_rows(db_session, test_project):
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    ms_row = await _seed_terminal(db_session, goal.id, "manager_selection", run.id)
    adr_row = await _seed_terminal(db_session, goal.id, "agent_definition_review", run.id, terminal="skipped")
    th_row = await _seed_terminal(db_session, goal.id, "team_hierarchy", run.id, terminal="skipped")
    gd_row = await _seed_terminal(db_session, goal.id, "goal_definition", run.id)  # seeded last -> latest terminal

    tasks_before = (
        await db_session.execute(select(Task).where(Task.project_id == test_project.id))
    ).scalars().all()
    actions_before = (
        await db_session.execute(select(OrchestrationAction).where(OrchestrationAction.run_id == run.id))
    ).scalars().all()

    result = await OrchestrationDebugService().rerun_last(db_session, test_project.id, goal.id)
    assert result["process_type"] == "goal_definition"

    tasks_after = (
        await db_session.execute(select(Task).where(Task.project_id == test_project.id))
    ).scalars().all()
    actions_after = (
        await db_session.execute(select(OrchestrationAction).where(OrchestrationAction.run_id == run.id))
    ).scalars().all()
    assert len(tasks_after) == len(tasks_before)
    assert len(actions_after) == len(actions_before) == 0

    for row, process_type in (
        (ms_row, "manager_selection"),
        (adr_row, "agent_definition_review"),
        (th_row, "team_hierarchy"),
    ):
        current = await OrchestrationProcessService().get_current(db_session, goal.id, process_type)
        assert current.id == row.id
        assert current.superseded_by_id is None

    new_gd_current = await OrchestrationProcessService().get_current(db_session, goal.id, "goal_definition")
    assert new_gd_current.id != gd_row.id

    await db_session.refresh(goal)
    await db_session.refresh(run)
    assert goal.status == "active"
    assert run.status == "running"


# 11. Transaction ownership


async def test_rerun_last_preserves_caller_explicit_transaction(db_session, test_project):
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    await _seed_terminal(db_session, goal.id, "goal_definition", run.id)

    result = await OrchestrationDebugService().rerun_last(db_session, test_project.id, goal.id)
    assert result["process_type"] == "goal_definition"

    # Session must remain usable -- no InvalidRequestError from a transaction
    # rerun_last() should not have touched.
    assert db_session.in_transaction()
    again = (await db_session.execute(select(OrchestrationGoal).where(OrchestrationGoal.id == goal.id))).scalar_one()
    assert again.id == goal.id


async def test_rerun_last_commits_when_it_owns_the_transaction(concurrent_sessions, tmp_path):
    writer, verifier = concurrent_sessions
    workspace = tmp_path / "durable-rerun-workspace"
    workspace.mkdir()
    project = Project(
        name=f"debug-rerun-durable-{uuid.uuid4()}",
        workspace_path=str(workspace.resolve()),
        config={},
    )
    writer.add(project)
    await writer.commit()

    goal = await _make_goal(writer, project)
    run = await _make_run(writer, goal)
    original = await _seed_terminal(writer, goal.id, "goal_definition", run.id)
    await writer.commit()

    result = await OrchestrationDebugService().rerun_last(writer, project.id, goal.id)
    assert result["process_type"] == "goal_definition"

    persisted = await verifier.scalar(
        select(OrchestrationProcessRun).where(
            OrchestrationProcessRun.goal_id == goal.id,
            OrchestrationProcessRun.process_type == "goal_definition",
            OrchestrationProcessRun.superseded_by_id.is_(None),
        )
    )
    assert persisted is not None
    assert persisted.id != original.id
    assert persisted.status == "completed"


# ---------------------------------------------------------------------------
# HTTP endpoint contracts: POST .../debug/baseline/step, .../debug/baseline/rerun-last
# ---------------------------------------------------------------------------


def _step_url(project_id, goal_id):
    return f"/api/v1/projects/{project_id}/orchestration/goals/{goal_id}/debug/baseline/step"


def _rerun_url(project_id, goal_id):
    return f"/api/v1/projects/{project_id}/orchestration/goals/{goal_id}/debug/baseline/rerun-last"


def _baseline_step_url(project_id, goal_id):
    return f"/api/v1/projects/{project_id}/orchestration/goals/{goal_id}/baseline/step"


def _baseline_rerun_url(project_id, goal_id):
    return f"/api/v1/projects/{project_id}/orchestration/goals/{goal_id}/baseline/rerun"


def _baseline_retry_url(project_id, goal_id):
    return f"/api/v1/projects/{project_id}/orchestration/goals/{goal_id}/baseline/retry"


async def test_baseline_retry_endpoint_consumes_checkpoint_and_hides_private_outputs(
    client, auth_headers, db_session, test_project, monkeypatch
):
    """A retryable process exposes only retry availability, never its saved LM request."""
    monkeypatch.setattr("huddleroom.routers.orchestration_goals.settings.debug", False)
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    current = await OrchestrationProcessService().start_process(
        db_session, goal.id, process_type="goal_definition", trigger_reason="test retry", run_id=run.id,
    )
    current.outputs = {
        "_lm_retry": {
            "version": 1,
            "kind": "goal_analysis",
            "request": {"model": "test", "messages": [], "response_format": {"type": "json_object"}, "temperature": 0},
            "continuation": {"clarification_round": 0, "context": {}, "working_goal": {"objective": "test", "success_criteria": []}},
            "targets": [{"private": True}],
            "completed": {"private": True},
            "error": "private provider detail",
            "messages": ["private prompt"],
        }
    }
    await db_session.flush()

    before = await client.get(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/processes",
        headers=auth_headers,
    )
    assert before.status_code == 200, before.text
    outputs = next(item["outputs"] for item in before.json() if item["id"] == str(current.id))
    from huddleroom.config import settings
    assert outputs["lm_retry"]["available"] is True
    assert outputs["lm_retry"]["kind"] == "goal_analysis"
    assert outputs["lm_retry"]["warning_id"] is None
    assert outputs["lm_retry"]["model"] == settings.orchestration_model
    assert outputs["lm_retry"]["hint"] is not None
    assert settings.orchestration_model in outputs["lm_retry"]["hint"]
    assert not {"_lm_retry", "request", "messages", "targets", "completed", "error"} & outputs.keys()

    response = await client.post(
        _baseline_retry_url(test_project.id, goal.id),
        json={"process_type": "goal_definition"},
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    assert response.json()["action"] == "retry"

    after = await client.get(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/processes",
        headers=auth_headers,
    )
    assert after.status_code == 200, after.text
    outputs = next(item["outputs"] for item in after.json() if item["id"] == str(current.id))
    assert outputs["lm_retry"]["available"] is False

    consumed = await client.post(
        _baseline_retry_url(test_project.id, goal.id),
        json={"process_type": "goal_definition"},
        headers=auth_headers,
    )
    assert consumed.status_code == 409


@pytest.mark.parametrize("process_type,outputs", [
    ("goal_definition", {}),
    ("manager_selection", {"_lm_retry": {"kind": "goal_analysis"}}),
])
async def test_baseline_retry_endpoint_rejects_nonretryable_processes(
    client, auth_headers, db_session, test_project, process_type, outputs
):
    """Missing checkpoints and deterministic phases cannot be retried through the LM action."""
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    current = await OrchestrationProcessService().start_process(
        db_session, goal.id, process_type=process_type, trigger_reason="test retry", run_id=run.id,
    )
    current.outputs = outputs
    await db_session.flush()

    response = await client.post(
        _baseline_retry_url(test_project.id, goal.id),
        json={"process_type": process_type},
        headers=auth_headers,
    )
    assert response.status_code == 409


async def test_public_retry_descriptor_hides_corrupt_checkpoint():
    """Only a complete supported checkpoint may advertise an operator Retry."""
    from huddleroom.schemas.orchestration import public_process_outputs, valid_lm_retry_checkpoint

    assert public_process_outputs({"_lm_retry": {"version": 1, "kind": "goal_analysis"}})["lm_retry"]["available"] is False

    request = {"model": "test", "messages": [], "response_format": {"type": "json_object"}}
    valid_agent_checkpoint = {
        "version": 1, "kind": "agent_definition_review", "fingerprint": "fingerprint",
        "coverage_fingerprint": "coverage", "model": "test", "request": request,
        "cursor": 0,
        "targets": [{
            "agent_id": str(uuid.uuid4()), "agent_snapshot": {
                "name": "developer", "config": {}, "capabilities": [], "role": "developer",
                "description": "Does implementation.", "system_prompt": "Implement safely.",
                "provider": "test", "model": "test", "adapter_type": "api", "cli_runtime": None,
                "is_active": True,
            }, "goal_snapshot": {},
            "candidate_work_functions": ["implementation"], "deterministic_assessment": {
                "fit_summary": "fit", "proposed_work_functions": ["implementation"],
                "strengths": [], "risks": [], "recommended_changes": [],
                "approved_for_work_functions": ["implementation"], "warnings": [],
            },
            "load_snapshot": {"active_sessions": 0, "active_tasks": 0, "outcome_hints": 0},
        }],
        "completed": {},
    }
    valid_goal_checkpoint = {
        "version": 1, "kind": "goal_analysis", "request": request,
        "continuation": {
            "clarification_round": 0, "context": {},
            "working_goal": {"objective": "Fix the README", "success_criteria": []},
        },
    }
    malformed_agent = copy.deepcopy(valid_agent_checkpoint)
    malformed_agent["targets"] = [{}]
    assert valid_lm_retry_checkpoint(malformed_agent, "agent_definition_review") is False
    assert public_process_outputs({"_lm_retry": malformed_agent}, "agent_definition_review")["lm_retry"]["available"] is False
    malformed_goal = copy.deepcopy(valid_goal_checkpoint)
    malformed_goal["continuation"].pop("context")
    assert valid_lm_retry_checkpoint(malformed_goal, "goal_definition") is False
    assert public_process_outputs({"_lm_retry": valid_agent_checkpoint}, "goal_definition")["lm_retry"]["available"] is False
    assert valid_lm_retry_checkpoint(valid_agent_checkpoint, "agent_definition_review") is True
    malformed_snapshot = copy.deepcopy(valid_agent_checkpoint)
    malformed_snapshot["targets"][0]["agent_snapshot"].pop("description")
    assert valid_lm_retry_checkpoint(malformed_snapshot, "agent_definition_review") is False
    malformed_improvement = copy.deepcopy(valid_agent_checkpoint)
    malformed_improvement["cursor"] = 1
    malformed_improvement["completed"] = {malformed_improvement["targets"][0]["agent_id"]: {
        "status": "improvement_proposed", "problems": ["scope is vague"], "reason": "clarify scope",
        "approved_work_functions": ["implementation"], "proposed_description": None, "proposed_persona": None,
    }}
    assert valid_lm_retry_checkpoint(malformed_improvement, "agent_definition_review") is False
    assert public_process_outputs({"_lm_retry": malformed_improvement}, "agent_definition_review")["lm_retry"]["available"] is False
    malformed_goal_shape = copy.deepcopy(valid_goal_checkpoint)
    malformed_goal_shape["continuation"]["working_goal"] = {}
    assert valid_lm_retry_checkpoint(malformed_goal_shape, "goal_definition") is False
    assert public_process_outputs({"_lm_retry": malformed_goal_shape}, "goal_definition")["lm_retry"]["available"] is False


async def test_public_process_outputs_includes_model_and_hint_for_valid_checkpoint():
    """When retry is available, model and hint are populated with current orchestration settings."""
    from huddleroom.config import settings
    from huddleroom.schemas.orchestration import public_process_outputs

    request = {"model": "test", "messages": [], "response_format": {"type": "json_object"}}
    valid_goal_checkpoint = {
        "version": 1, "kind": "goal_analysis", "request": request,
        "continuation": {
            "clarification_round": 0, "context": {},
            "working_goal": {"objective": "Fix the README", "success_criteria": []},
        },
    }

    result = public_process_outputs({"_lm_retry": valid_goal_checkpoint}, "goal_definition")
    lm_retry = result["lm_retry"]

    assert lm_retry["available"] is True
    assert lm_retry["model"] == settings.orchestration_model
    assert lm_retry["hint"] is not None
    assert settings.orchestration_model in lm_retry["hint"]
    assert "upgrading" in lm_retry["hint"].lower()
    assert "stronger model" in lm_retry["hint"].lower()


async def test_public_process_outputs_excludes_model_and_hint_for_unavailable_checkpoint():
    """When retry is not available, model and hint are None."""
    from huddleroom.schemas.orchestration import public_process_outputs

    result = public_process_outputs({}, "goal_definition")
    lm_retry = result["lm_retry"]

    assert lm_retry["available"] is False
    assert lm_retry["model"] is None
    assert lm_retry["hint"] is None

    # Also test with malformed checkpoint
    result = public_process_outputs({"_lm_retry": {"version": 1, "kind": "goal_analysis"}}, "goal_definition")
    lm_retry = result["lm_retry"]

    assert lm_retry["available"] is False
    assert lm_retry["model"] is None
    assert lm_retry["hint"] is None


@pytest.mark.parametrize("corruption", [
    lambda checkpoint: checkpoint.update(targets=[{}]),
    lambda checkpoint: checkpoint.update(completed={"not-a-target": {}}),
    lambda checkpoint: checkpoint["targets"][0].update(deterministic_assessment={}),
    lambda checkpoint: checkpoint["targets"][0]["agent_snapshot"].pop("config"),
    lambda checkpoint: checkpoint.update(cursor=1, completed={checkpoint["targets"][0]["agent_id"]: {
        "status": "improvement_proposed", "problems": [], "reason": "clarify",
        "approved_work_functions": [], "proposed_description": None, "proposed_persona": None,
    }}),
])
async def test_baseline_retry_endpoint_rejects_malformed_agent_checkpoint_before_dispatch(
    client, auth_headers, db_session, test_project, monkeypatch, corruption
):
    """Malformed private retry state must be rejected before either LM retry owner runs."""
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    current = await OrchestrationProcessService().start_process(
        db_session, goal.id, process_type="agent_definition_review", trigger_reason="test retry", run_id=run.id,
    )
    checkpoint = {
        "version": 1, "kind": "agent_definition_review", "fingerprint": "fingerprint",
        "coverage_fingerprint": "coverage", "model": "test",
        "request": {"model": "test", "messages": [], "response_format": {"type": "json_object"}},
        "cursor": 0, "targets": [{
            "agent_id": str(uuid.uuid4()), "agent_snapshot": {
                "name": "developer", "config": {}, "capabilities": [], "role": "developer",
                "description": "Does implementation.", "system_prompt": "Implement safely.",
                "provider": "test", "model": "test", "adapter_type": "api", "cli_runtime": None,
                "is_active": True,
            }, "goal_snapshot": {},
            "candidate_work_functions": ["implementation"], "deterministic_assessment": {
                "fit_summary": "fit", "proposed_work_functions": ["implementation"],
                "strengths": [], "risks": [], "recommended_changes": [],
                "approved_for_work_functions": ["implementation"], "warnings": [],
            },
            "load_snapshot": {"active_sessions": 0, "active_tasks": 0, "outcome_hints": 0},
        }], "completed": {},
    }
    corruption(checkpoint)
    current.outputs = {"_lm_retry": checkpoint}
    await db_session.flush()

    async def should_not_retry(*_args, **_kwargs):
        raise AssertionError("malformed checkpoint reached an LM retry owner")

    monkeypatch.setattr("huddleroom.services.orchestration_agent_definition_review.AgentDefinitionReviewProcess.retry_failed", should_not_retry)
    monkeypatch.setattr("huddleroom.services.orchestration_goal_definition.GoalDefinitionProcess.retry_failed", should_not_retry)
    response = await client.post(
        _baseline_retry_url(test_project.id, goal.id), json={"process_type": "agent_definition_review"}, headers=auth_headers,
    )
    assert response.status_code == 409


async def test_baseline_retry_endpoint_rejects_incomplete_goal_continuation_before_dispatch(
    client, auth_headers, db_session, test_project, monkeypatch
):
    """A goal checkpoint without its frozen working goal must not invoke the LM owner."""
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    current = await OrchestrationProcessService().start_process(
        db_session, goal.id, process_type="goal_definition", trigger_reason="test retry", run_id=run.id,
    )
    current.outputs = {"_lm_retry": {
        "version": 1, "kind": "goal_analysis",
        "request": {"model": "test", "messages": [], "response_format": {"type": "json_object"}},
        "continuation": {"clarification_round": 0, "context": {}, "working_goal": {}},
    }}
    await db_session.flush()

    async def should_not_retry(*_args, **_kwargs):
        raise AssertionError("incomplete continuation reached the goal retry owner")

    monkeypatch.setattr("huddleroom.services.orchestration_goal_definition.GoalDefinitionProcess.retry_failed", should_not_retry)
    response = await client.post(
        _baseline_retry_url(test_project.id, goal.id), json={"process_type": "goal_definition"}, headers=auth_headers,
    )
    assert response.status_code == 409


async def test_baseline_step_endpoint_succeeds_when_debug_disabled(
    client, auth_headers, db_session, test_project, monkeypatch
):
    monkeypatch.setattr("huddleroom.routers.orchestration_goals.settings.debug", False)
    goal = await _make_goal(db_session, test_project)
    await _make_run(db_session, goal)
    await db_session.flush()

    resp = await client.post(
        _baseline_step_url(test_project.id, goal.id),
        json={"process_type": "goal_definition"},
        headers=auth_headers,
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["action"] == "step"
    assert body["process_type"] == "goal_definition"
    assert body["process"]["status"] == "completed"


@pytest.mark.unsupported_mode
async def test_baseline_step_endpoint_rejects_unauthenticated_requests_when_auth_enabled(
    client, db_session, test_project, monkeypatch
):
    from huddleroom.config import Settings

    monkeypatch.setattr("huddleroom.dependencies.settings", Settings(auth_enabled=True))
    goal = await _make_goal(db_session, test_project)
    await _make_run(db_session, goal)
    await db_session.flush()

    resp = await client.post(
        _baseline_step_url(test_project.id, goal.id),
        json={"process_type": "goal_definition"},
    )

    assert resp.status_code == 401


async def test_baseline_step_endpoint_returns_409_before_predecessor_completes(
    client, auth_headers, db_session, test_project, monkeypatch
):
    monkeypatch.setattr("huddleroom.routers.orchestration_goals.settings.debug", False)
    goal = await _make_goal(db_session, test_project)
    await _make_run(db_session, goal)
    await db_session.flush()

    resp = await client.post(
        _baseline_step_url(test_project.id, goal.id),
        json={"process_type": "manager_selection"},
        headers=auth_headers,
    )

    assert resp.status_code == 409


async def test_baseline_rerun_endpoint_requires_process_type(
    client, auth_headers, db_session, test_project, monkeypatch
):
    monkeypatch.setattr("huddleroom.routers.orchestration_goals.settings.debug", False)
    goal = await _make_goal(db_session, test_project)
    await _make_run(db_session, goal)
    await db_session.flush()

    resp = await client.post(_baseline_rerun_url(test_project.id, goal.id), headers=auth_headers)

    assert resp.status_code == 422


async def test_baseline_rerun_endpoint_reruns_requested_process_when_debug_disabled(
    client, auth_headers, db_session, test_project, monkeypatch
):
    monkeypatch.setattr("huddleroom.routers.orchestration_goals.settings.debug", False)
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    await _seed_terminal(db_session, goal.id, "goal_definition", run.id)
    await _seed_terminal(db_session, goal.id, "effectiveness_review", run.id)
    await db_session.flush()

    resp = await client.post(
        _baseline_rerun_url(test_project.id, goal.id),
        json={"process_type": "goal_definition"},
        headers=auth_headers,
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["action"] == "rerun_last"
    assert body["process_type"] == "goal_definition"
    assert body["process"]["status"] == "completed"


async def test_debug_step_endpoint_disabled_returns_404(client, auth_headers, db_session, test_project, monkeypatch):
    monkeypatch.setattr("huddleroom.routers.orchestration_goals.settings.debug", False)
    goal = await _make_goal(db_session, test_project)
    await _make_run(db_session, goal)
    await db_session.flush()

    resp = await client.post(
        _step_url(test_project.id, goal.id),
        json={"process_type": "goal_definition"},
        headers=auth_headers,
    )
    assert resp.status_code == 404


async def test_debug_rerun_endpoint_disabled_returns_404(client, auth_headers, db_session, test_project, monkeypatch):
    monkeypatch.setattr("huddleroom.routers.orchestration_goals.settings.debug", False)
    goal = await _make_goal(db_session, test_project)
    await _make_run(db_session, goal)
    await db_session.flush()

    resp = await client.post(_rerun_url(test_project.id, goal.id), headers=auth_headers)
    assert resp.status_code == 404


async def test_debug_step_endpoint_invalid_process_type_422(client, auth_headers, db_session, test_project, monkeypatch):
    monkeypatch.setattr("huddleroom.routers.orchestration_goals.settings.debug", True)
    goal = await _make_goal(db_session, test_project)
    await _make_run(db_session, goal)
    await db_session.flush()

    resp = await client.post(
        _step_url(test_project.id, goal.id),
        json={"process_type": "bogus"},
        headers=auth_headers,
    )
    assert resp.status_code == 422


async def test_debug_step_endpoint_success(client, auth_headers, db_session, test_project, monkeypatch):
    monkeypatch.setattr("huddleroom.routers.orchestration_goals.settings.debug", True)
    goal = await _make_goal(db_session, test_project)
    await _make_run(db_session, goal)
    await db_session.flush()

    resp = await client.post(
        _step_url(test_project.id, goal.id),
        json={"process_type": "goal_definition"},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["action"] == "step"
    assert body["process_type"] == "goal_definition"
    assert body["goal_id"] == str(goal.id)
    assert "run_id" in body
    assert "process" in body


async def test_debug_step_endpoint_missing_goal_404(client, auth_headers, test_project, monkeypatch):
    monkeypatch.setattr("huddleroom.routers.orchestration_goals.settings.debug", True)

    resp = await client.post(
        _step_url(test_project.id, uuid.uuid4()),
        json={"process_type": "goal_definition"},
        headers=auth_headers,
    )
    assert resp.status_code == 404


async def test_debug_step_endpoint_conflict_propagates_409(client, auth_headers, db_session, test_project, monkeypatch):
    monkeypatch.setattr("huddleroom.routers.orchestration_goals.settings.debug", True)
    goal = await _make_goal(db_session, test_project, status="paused")
    await _make_run(db_session, goal)
    await db_session.flush()

    resp = await client.post(
        _step_url(test_project.id, goal.id),
        json={"process_type": "goal_definition"},
        headers=auth_headers,
    )
    assert resp.status_code == 409


async def test_debug_rerun_endpoint_no_baseline_409(client, auth_headers, db_session, test_project, monkeypatch):
    monkeypatch.setattr("huddleroom.routers.orchestration_goals.settings.debug", True)
    goal = await _make_goal(db_session, test_project)
    await _make_run(db_session, goal)
    await db_session.flush()

    resp = await client.post(_rerun_url(test_project.id, goal.id), headers=auth_headers)
    assert resp.status_code == 409


async def test_debug_rerun_endpoint_accepts_selected_process_type(
    client, auth_headers, db_session, test_project, monkeypatch
):
    monkeypatch.setattr("huddleroom.routers.orchestration_goals.settings.debug", True)
    goal = await _make_goal(db_session, test_project)
    run = await _make_run(db_session, goal)
    await _seed_terminal(db_session, goal.id, "goal_definition", run.id)
    await _seed_terminal(db_session, goal.id, "effectiveness_review", run.id)
    await db_session.flush()

    resp = await client.post(
        _rerun_url(test_project.id, goal.id), json={"process_type": "goal_definition"}, headers=auth_headers
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["process_type"] == "goal_definition"
